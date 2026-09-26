"""O(3) and aligned-frame convolutions, including training and higher derivatives."""

import subprocess
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from e3nn import o3

from eqx import conv as eqx_conv
from eqx import o2
from eqx.ace import TACE
from eqx.models.tace.tece_oam_rra import BilinearACE


@pytest.mark.parametrize("degree", range(6))
def test_rotation_generators_match_wigner(degree, double_precision):
    from eqx.conv.o2_o3.geometry import generators

    angle = torch.zeros((), dtype=torch.float64)
    half_pi = angle.new_tensor(torch.pi / 2)
    matrices = (
        lambda a: o3.wigner_D(degree, angle, a, angle),
        lambda a: o3.wigner_D(degree, a, angle, angle),
        lambda a: o3.wigner_D(degree, -half_pi, a, half_pi),
    )
    expected = torch.stack(
        [torch.autograd.functional.jacobian(matrix, angle) for matrix in matrices]
    )
    torch.testing.assert_close(generators(degree), expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("degree", [1, 5, 10, 16])
@pytest.mark.parametrize("normalization", ["component", "integral", "norm"])
def test_generator_polynomials(degree, normalization, double_precision):
    from eqx.conv.angular import generator_scale, generators

    torch.manual_seed(91)
    vectors = torch.randn(19, 3)
    vectors[:3] = torch.eye(3)
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    matrix = torch.einsum("ba,aij->bij", vectors, generators(degree))
    matrix /= (degree * (degree + 1)) ** 0.5
    identity = torch.eye(2 * degree + 1)
    for harmonic in (1, 2):
        scale = generator_scale(degree, harmonic, normalization)
        value = matrix if harmonic == 1 else matrix @ matrix + identity / 3
        expected = torch.einsum(
            "aqc,bq->bca",
            o3.wigner_3j(degree, harmonic, degree),
            o3.spherical_harmonics(
                harmonic, vectors, normalize=False, normalization=normalization
            ),
        )
        torch.testing.assert_close(value * scale, expected, atol=5e-14, rtol=5e-13)


@pytest.mark.parametrize("degree", [2, 8, 16])
def test_high_degree_angular_derivatives(degree, double_precision):
    from eqx.conv.angular import generators
    from eqx.conv.o2_o3.geometry import angular_coefficients

    tp = o2.O3TensorProduct(f"{degree}e", "2e", f"{degree}e", [(0, 0, 0, "uvu", True)])
    conv = eqx_conv.O2O3TensorProductConv(tp, backend="torch")
    previous = angular_coefficients(conv.direction_metadata, 0)[0]
    for rank in range(1, 5):
        expected = torch.zeros(*previous.shape, 3)
        for axis, ell in enumerate((degree, degree, *([1] * (rank - 1)))):
            term = torch.tensordot(previous, -generators(ell), dims=([axis], [1]))
            expected += term.movedim(-1, axis)
        expected[..., 1] = 0
        actual = angular_coefficients(conv.direction_metadata, rank)[0]
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
        previous = actual


@pytest.mark.parametrize("degree", [4, 12])
@pytest.mark.parametrize("implementation", ["o3", "o2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_generator_cuda_paths(degree, implementation, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        torch.manual_seed(89)
        args = (
            f"3x{degree}e",
            "1o+2e",
            f"3x{degree}o+3x{degree}e+3x{degree}e+3x{degree + 2}e",
            [
                (0, 0, 0, "uvu", True),
                (0, 1, 1, "uvu", True),
                (0, 1, 2, "uvu", False),
                (0, 1, 3, "uvu", True),
            ],
        )
        options = dict(internal_weights=False, shared_weights=False)
        tp = o3.TensorProduct(*args, **options)
        reference = eqx_conv.O3TensorProductConv(tp, backend="torch").cuda()
        module = (
            eqx_conv.O3TensorProductConv(tp)
            if implementation == "o3"
            else eqx_conv.O2O3TensorProductConv(o2.O3TensorProduct(*args, **options))
        ).cuda()
        edges = torch.randint(3, (2, 5), device="cuda")
        inputs = [
            torch.randn(shape, device="cuda", requires_grad=True)
            for shape in ((3, tp.irreps_in1.dim), (5, 3), (5, 2), (5, tp.weight_numel))
        ]
        x, vectors, amplitudes, weights = inputs
        projection = x.new_empty(0, tp.weight_numel)
        expected = reference(
            x, None, weights, projection, edges, vectors=vectors, amplitudes=amplitudes
        )
        if implementation == "o3":
            actual = module(
                x,
                None,
                weights,
                projection,
                edges,
                vectors=vectors,
                amplitudes=amplitudes,
            )
        else:
            frame = o2.WignerD(degree + 2, degree + 2, method="recursive").cuda()
            actual = module(
                x,
                weights,
                projection,
                frame.forward_packed(vectors.detach()),
                amplitudes,
                edges,
                3,
                vectors=vectors,
            )
        tolerance = 3e-5 if dtype == torch.float32 else 2e-10
        for _ in range(3):
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            cotangent = torch.randn_like(actual) / actual.numel() ** 0.5
            actual, expected = [
                torch.cat(
                    [
                        g.flatten()
                        for g in torch.autograd.grad(
                            (value.sin() * cotangent).sum(), inputs, create_graph=True
                        )
                    ]
                )
                for value in (actual, expected)
            ]
        torch.testing.assert_close(
            actual, expected, atol=5 * tolerance, rtol=5 * tolerance
        )
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize(
    "normalization,order", [("component", 4), ("integral", 2), ("norm", 2)]
)
def test_low_degree_couplings_fourth_derivatives(
    normalization, order, double_precision
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(103)
    outputs, instructions = [], []
    for harmonic in range(3):
        for degree in range(abs(2 - harmonic), 3 + harmonic):
            instructions.append((0, harmonic, len(outputs), "uvu", degree != 0))
            outputs.append((2, (degree, (-1) ** harmonic)))
    tp = o2.O3TensorProduct(
        "2x2e",
        "0e+1o+2e",
        outputs,
        instructions,
        normalization=normalization,
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    module = eqx_conv.O2O3TensorProductConv(tp)
    reference = eqx_conv.O3TensorProductConv(
        o3.TensorProduct(
            "2x2e",
            "0e+1o+2e",
            outputs,
            instructions,
            internal_weights=False,
            shared_weights=False,
        ),
        backend="torch",
        normalization=normalization,
    ).cuda()
    frame = o2.WignerD(4, 4, method="recursive").cuda()
    edges = torch.tensor([[0, 1, 0], [1, 0, 1]], device="cuda")
    x = torch.randn(2, tp.input_dim, device="cuda")
    vectors = torch.tensor(
        [[0.0, 2.0, 0.0], [0.0, -2.0, 1e-8], [0.3, -0.7, 1.2]],
        device="cuda",
        requires_grad=True,
    )
    weights = torch.randn(3, tp.weight_numel, device="cuda", requires_grad=True)
    amplitudes = torch.randn(1, 3, device="cuda")
    inputs = vectors, weights
    packed = frame.forward_packed(vectors.detach())
    projection = x.new_empty(0, tp.weight_numel)
    actual = module(
        x, weights, projection, packed, amplitudes, edges, 2, vectors=vectors
    )
    expected = reference(
        x, None, weights, projection, edges, vectors=vectors, amplitudes=amplitudes
    )
    for _ in range(order):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        cotangent = torch.randn_like(actual) / actual.numel() ** 0.5
        actual, expected = [
            torch.cat(
                [
                    grad.flatten()
                    for grad in torch.autograd.grad(
                        (value.sin() * cotangent).sum(), inputs, create_graph=True
                    )
                ]
            )
            for value in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("shared", [False, True])
def test_hybrid_direction_derivatives(monkeypatch, shared, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.conv.o2_o3 import cuda

    monkeypatch.setattr(cuda, "CHUNK_SIZE", 2)
    torch.manual_seed(107)
    args = (
        "3x2e",
        "1o+2e+3o",
        "3x1o+3x2e+3x3o",
        [(0, 0, 0, "uvu", True), (0, 1, 1, "uvu", False), (0, 2, 2, "uvu", True)],
    )
    options = dict(internal_weights=False, shared_weights=False)
    tp = o2.O3TensorProduct(*args, **options).cuda()
    module = eqx_conv.O2O3TensorProductConv(tp)
    reference = eqx_conv.O3TensorProductConv(
        o3.TensorProduct(*args, **options), backend="torch"
    ).cuda()
    frame = o2.WignerD(3, 3, method="recursive").cuda()
    rows = 1 if shared else 5
    inputs = [
        torch.randn(shape, device="cuda", requires_grad=True)
        for shape in (
            (3, tp.input_dim),
            (rows, 3),
            (rows, 4),
            (4, tp.weight_numel),
            (rows, 3),
        )
    ]
    x, vectors, radial, projection, amplitudes = inputs
    edges = torch.randint(3, (2, 5), device="cuda")
    actual = module(
        x,
        radial,
        projection,
        frame.forward_packed(vectors.detach()),
        amplitudes,
        edges,
        3,
        vectors=vectors,
    )
    expected = reference(
        x,
        None,
        radial,
        projection,
        edges,
        vectors=vectors.expand(edges.size(1), -1),
        amplitudes=amplitudes,
    )
    for _ in range(2):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        cotangent = torch.randn_like(actual) / actual.numel() ** 0.5
        actual, expected = [
            torch.cat(
                [
                    grad.flatten()
                    for grad in torch.autograd.grad(
                        (value.sin() * cotangent).sum(), inputs, create_graph=True
                    )
                ]
            )
            for value in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("magnetic", [False, True])
@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("mmax", [0, 2])
def test_uv_o2_cuda_convolution(monkeypatch, magnetic, attention, mmax):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.o2 import (
        O2ScatterMagneticTensorProduct,
        UvO2ScatterTensorProduct,
    )

    torch.manual_seed(401)
    torch.set_default_dtype(torch.float64)
    time = hasattr(o3.Irrep("0e"), "t")
    irreps = (
        "2x0ee+2x0oo+2x1oe+2x1eo+2x2ee+2x2oo"
        if time
        else "2x0e+2x0o+2x1o+2x1e+2x2e+2x2o"
    )
    cls = O2ScatterMagneticTensorProduct if magnetic else UvO2ScatterTensorProduct
    args = (irreps, irreps, irreps) if magnetic else (irreps, irreps)
    module = (
        cls(
            *args,
            num_channel=2,
            mmax=mmax,
            even_scalar_act=torch.nn.SiLU(),
            odd_scalar_act=torch.nn.Tanh(),
            tensor_act=torch.nn.Sigmoid(),
            num_head=2,
            num_radial_basis=3,
            use_radial_rotary_attention=attention,
        )
        .cuda()
        .double()
    )
    frame = o2.WignerD(2, 2).cuda().double()
    for edges in (9, 0):

        def rand(*shape):
            return (
                torch.randn(*shape, device="cuda", dtype=torch.float64) * 0.2
            ).requires_grad_()

        x, mag = rand(4, o3.Irreps(irreps).dim), rand(edges, o3.Irreps(irreps).dim)
        vectors = rand(edges, 3)
        weights, radial, cutoff = (
            rand(edges, module.weight_numel),
            rand(edges, 3),
            rand(edges, 1).sigmoid(),
        )
        index = torch.randint(4, (2, edges), device="cuda")
        inputs = (x, vectors, weights, radial, cutoff) + ((mag,) if magnetic else ())
        inputs += tuple(module.parameters())

        def run(enabled):
            monkeypatch.setenv("TACE_USE_EQX", str(int(enabled)))
            w, wi = frame(vectors)
            args = (
                (x, mag, weights, index, w, wi)
                if magnetic
                else (x, weights, index, w, wi)
            )
            value = module(*args, edge_radial_basis=radial, edge_cutoff=cutoff)
            result = [value]
            if edges:
                for _ in range(3 if magnetic and attention and mmax == 2 else 2):
                    grads = torch.autograd.grad(
                        value.sin().sum(), inputs, create_graph=True, allow_unused=True
                    )
                    result.extend(g for g in grads if g is not None)
                    value = (
                        torch.cat([g.flatten() for g in grads if g is not None]) / 10
                    )
            return result

        expected, actual = run(False), run(True)
        assert len(actual) == len(expected)
        for i, (a, b) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(
                a, b, atol=2e-8, rtol=2e-8, msg=lambda message: f"Result {i}: {message}"
            )


def test_uv_o2_compile_and_force_training(monkeypatch, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.o2 import O2ScatterMagneticTensorProduct
    from tace.models.compile.compile import trace_to_fx

    monkeypatch.setenv("TACE_USE_EQX", "1")
    irreps = "2x0ee+2x1eo+2x1oe" if hasattr(o3.Irrep("0e"), "t") else "2x0e+2x1e+2x1o"
    module = O2ScatterMagneticTensorProduct(
        irreps,
        irreps,
        irreps,
        num_channel=2,
        mmax=1,
        even_scalar_act=torch.nn.SiLU(),
        odd_scalar_act=torch.nn.Tanh(),
        tensor_act=torch.nn.Sigmoid(),
        num_head=2,
        num_radial_basis=3,
        use_radial_rotary_attention=True,
    ).cuda()
    frame = o2.WignerD(1, 1).cuda()

    def evaluate(x, mag, weights, vectors, radial, cutoff, index):
        w, wi = frame(vectors)
        return module(x, mag, weights, index, w, wi, radial, cutoff)

    compiled = torch.compile(
        evaluate, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for nodes, edges in ((3, 5), (3, 0), (0, 0)):
        index = torch.randint(max(nodes, 1), (2, edges), device="cuda")
        inputs = [
            torch.randn(shape, device="cuda", requires_grad=True) * 0.1
            for shape in (
                (nodes, module.irreps_in.dim),
                (edges, module.magnetic_edge_irreps.dim),
                (edges, module.weight_numel),
                (edges, 3),
                (edges, 3),
                (edges, 1),
            )
        ]
        inputs[-1] = inputs[-1].sigmoid()
        actual, expected = compiled(*inputs, index), evaluate(*inputs, index)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        if edges:

            def energy_forces(*args):
                energy = evaluate(*args).square().sum()
                forces = -torch.autograd.grad(energy, args[3], create_graph=True)[0]
                return energy, forces

            traced = trace_to_fx(energy_forces, (*inputs, index))
            compiled_forces = torch.compile(traced, backend="aot_eager", fullgraph=True)
            a, b = compiled_forces(*inputs, index), energy_forces(*inputs, index)
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, atol=1e-9, rtol=1e-9)
            parameters = tuple(module.parameters())
            ga, gb = [
                torch.autograd.grad(e + f.square().sum(), parameters) for e, f in (a, b)
            ]
            for x, y in zip(ga, gb):
                torch.testing.assert_close(x, y, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("implementation", ["o3", "o2", "o2_direction"])
@pytest.mark.parametrize("shared", [False, True])
def test_native_cuda_graph_convolution(monkeypatch, implementation, shared):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.kernels import cuda_graph

    monkeypatch.setattr(cuda_graph, "EDGE_BUCKET", 8)
    monkeypatch.setattr(cuda_graph, "MIN_EDGES", 1)
    cuda_graph._GRAPHS.clear()
    tp_args = (
        "2x0e+2x1o",
        "0e+1o",
        "2x0e+2x1o+2x0e",
        [(0, 0, 0, "uvu", True), (0, 1, 1, "uvu", True), (1, 1, 2, "uvu", False)],
    )
    tp_cls = o3.TensorProduct if implementation == "o3" else o2.O3TensorProduct
    tp = tp_cls(*tp_args, internal_weights=False, shared_weights=False)
    cls = (
        eqx_conv.O3TensorProductConv
        if implementation == "o3"
        else eqx_conv.O2O3TensorProductConv
    )
    module = cls(tp).cuda().double()
    frame = o2.WignerD(1, 1).cuda().double()
    try:
        for edges in (9, 13, 9, 5, 1, 0):
            index = torch.randint(4, (2, edges), device="cuda")

            def rand(*shape):
                return (
                    torch.randn(*shape, device="cuda", dtype=torch.float64) * 0.2
                ).requires_grad_()

            x, vectors = rand(4, 8), rand(edges, 3)
            rows = 1 if shared else edges
            radial, projection, amplitudes = (
                rand(rows, 3),
                rand(3, tp.weight_numel),
                rand(rows, 2),
            )
            inputs = x, vectors, radial, projection, amplitudes

            def evaluate(enabled):
                monkeypatch.setenv("EQX_USE_CUDA_GRAPH", str(int(enabled)))
                if implementation == "o3":
                    value = module(
                        x,
                        None,
                        radial,
                        projection,
                        index,
                        vectors=vectors,
                        amplitudes=amplitudes,
                    )
                else:
                    value = module(
                        x,
                        radial,
                        projection,
                        frame.forward_packed(vectors),
                        amplitudes,
                        index,
                        4,
                        vectors=vectors if implementation == "o2_direction" else None,
                    )
                result = [value]
                if edges:
                    for _ in range(3):
                        grads = torch.autograd.grad(
                            value.sin().sum(), inputs, create_graph=True
                        )
                        result.extend(grads)
                        value = torch.cat([g.flatten() for g in grads]) / 10
                return result

            reference = evaluate(False)
            actual = evaluate(True)
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, atol=5e-9, rtol=5e-9)
            if edges:
                assert cuda_graph._GRAPHS
    finally:
        cuda_graph._GRAPHS.clear()


def test_native_cuda_graph_ace(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.kernels import cuda_graph

    tp = o3.TensorProduct(
        "2x0e",
        "2x0e",
        "2x0e",
        [(0, 0, 0, "uuu", False)],
        internal_weights=False,
        shared_weights=False,
    )
    linears = [
        o3.Linear("2x0e", "3x0e", internal_weights=False, shared_weights=False)
        for _ in range(3)
    ]
    module = TACE([tp, tp], linears).cuda().double()
    cuda_graph._GRAPHS.clear()
    try:
        for seed in (2, 3):
            torch.manual_seed(seed)
            types = torch.randint(2, (7,), device="cuda")
            inputs = [
                torch.randn(
                    7, 2, device="cuda", dtype=torch.float64, requires_grad=True
                )
            ]
            inputs += [
                torch.randn(
                    2, 6, device="cuda", dtype=torch.float64, requires_grad=True
                )
                for _ in linears
            ]
            results = []
            for enabled in (False, True):
                monkeypatch.setenv("EQX_USE_CUDA_GRAPH", str(int(enabled)))
                value = module(inputs[0], inputs[1:], types)
                result = [value]
                for _ in range(3):
                    grads = torch.autograd.grad(
                        value.sin().sum(), inputs, create_graph=True
                    )
                    result.extend(grads)
                    value = torch.cat([g.flatten() for g in grads]) / 10
                results.append(result)
            for a, b in zip(*results):
                torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
        assert cuda_graph._GRAPHS
    finally:
        cuda_graph._GRAPHS.clear()


def test_native_cuda_graph_outer_capture(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.kernels import cuda_graph

    monkeypatch.setenv("EQX_USE_CUDA_GRAPH", "1")
    monkeypatch.setattr(cuda_graph, "MIN_EDGES", 1)
    monkeypatch.setattr(cuda_graph, "EDGE_BUCKET", 8)
    cuda_graph._GRAPHS.clear()
    tp = o3.TensorProduct(
        "2x0e",
        "0e",
        "2x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    module = eqx_conv.O3TensorProductConv(tp).cuda().double()
    x = torch.randn(4, 2, device="cuda", dtype=torch.float64, requires_grad=True)
    attrs = torch.randn(11, 1, device="cuda", dtype=torch.float64)
    weight = torch.randn(11, 2, device="cuda", dtype=torch.float64)
    projection = weight.new_empty(0, 2)
    index = torch.randint(4, (2, 11), device="cuda")

    def evaluate():
        value = module(x, attrs, weight, projection, index)
        return value, torch.autograd.grad(value.square().sum(), x)[0]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Warm the ordinary (unpadded) kernels that the outer graph will capture.
    monkeypatch.setenv("EQX_USE_CUDA_GRAPH", "0")
    with torch.cuda.stream(stream):
        evaluate()
        evaluate()
    torch.cuda.current_stream().wait_stream(stream)
    monkeypatch.setenv("EQX_USE_CUDA_GRAPH", "1")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = evaluate()
    assert not cuda_graph._GRAPHS
    with torch.no_grad():
        x.mul_(0.7)
        weight.add_(0.3)
        index.copy_(torch.randint_like(index, 4))
    graph.replay()
    monkeypatch.setenv("EQX_USE_CUDA_GRAPH", "0")
    for a, b in zip(actual, evaluate()):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mask", range(16))
@pytest.mark.parametrize("compile_enabled", [False, True])
def test_scatter_backend_priority(monkeypatch, mask, compile_enabled):
    from tace.models._e3nn.fused import O3ScatterTensorProduct
    from tace.utils.env import ACCELERATION_ENV

    enabled = [
        name for i, name in enumerate(("eqx", "oeq", "eqt", "cue")) if mask & (1 << i)
    ]
    for name in ("eqx", "oeq", "eqt", "cue"):
        monkeypatch.setenv(ACCELERATION_ENV[name], "1" if name in enabled else "0")
    monkeypatch.setenv("TACE_USE_COMPILE", "1" if compile_enabled else "0")
    expected = next((name for name in enabled if name != "eqt"), None)
    if expected == "cue" and compile_enabled:
        expected = None
    constructed = []

    def oeq(**kwargs):
        constructed.append("oeq")
        return torch.nn.Identity()

    def cue(**kwargs):
        constructed.append("cue")
        return torch.nn.Identity()

    monkeypatch.setitem(
        sys.modules,
        "tace.models.oeq",
        types.SimpleNamespace(e3nnOeqScatterTensorProduct=oeq),
    )
    monkeypatch.setitem(
        sys.modules,
        "tace.models.cue",
        types.SimpleNamespace(e3nnCueScatterTensorProduct=cue),
    )
    module = O3ScatterTensorProduct("2x0e", "0e", "2x0e")
    assert module.use_eqx == (expected == "eqx")
    assert module.use_oeq == (expected == "oeq")
    assert module.use_cue == (expected == "cue")
    assert constructed == ([expected] if expected in ("oeq", "cue") else [])


@pytest.fixture(scope="module")
def so2_v021():
    """Load the unmodified release reference without shipping legacy operators."""
    result = subprocess.run(
        ["git", "show", "v0.2.1:tace/models/_e3nn/legacy_so2.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        pytest.skip("The v0.2.1 tag is required for the release comparison.")
    module = types.ModuleType("tace.models._e3nn._reference_so2_v021")
    exec(compile(result.stdout, "v0.2.1/legacy_so2.py", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mmax", [0, 1, 2])
@pytest.mark.parametrize("ece", [False, True])
@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("gate_m0", [False, True])
def test_tece_v021_state_dict(so2_v021, device, mmax, ece, attention, gate_m0):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from tace.models._e3nn.tece_oam_rra import Convolution
    from tace.models.layout import LayoutTransform
    from tace.models.mlp import get_scaled_activation

    torch.manual_seed(18)
    irreps = o3.Irreps("4x0e+4x1o+4x2e")
    kwargs = dict(
        mmax=mmax,
        lmax=2,
        num_channel=4,
        num_head=2,
        edge_ace_hidden=3,
        num_radial_basis=5,
        gate_m0=gate_m0,
        use_asymmetric_contraction=ece,
        use_radial_rotary_attention=attention,
        reshape_in=LayoutTransform(irreps),
        reshape_out=LayoutTransform(irreps),
        scalar_act=get_scaled_activation("silu"),
        tensor_act=get_scaled_activation("silu" if ece else "sigmoid"),
    )
    reference = so2_v021.uvSO2Convolution(
        **deepcopy(kwargs),
        so2_linear_type="w1",
        use_temperature=True,
        use_radial_phase=True,
    ).to(device=device, dtype=torch.float64)
    if attention:
        with torch.no_grad():
            reference.radial_proj.weight.normal_(std=0.2)
            reference.temperature_logit.uniform_(-1, 1)
    model = Convolution(**kwargs).to(device=device, dtype=torch.float64)
    model.load_state_dict(reference.state_dict(), strict=True)
    assert isinstance(model.linear_up, o2.Linear)
    assert isinstance(model.linear_down, o2.Linear)
    assert isinstance(model.nonlinearity, o2.Gate)
    if ece:
        assert isinstance(model.ece, o2.TensorProduct)
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )
    restored = deepcopy(model)
    restored.load_state_dict(model.state_dict(), strict=True)
    nodes, edges = 4, 7
    degrees = torch.tensor(
        [
            ell
            for m in range(mmax + 1)
            for _ in range(1 if m == 0 else 2)
            for ell in range(m, 3)
        ],
        device=device,
    )
    columns = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2], device=device)
    mask = degrees[:, None] == columns[None, :]
    inputs = [
        torch.randn(*shape, device=device, dtype=torch.float64)
        .mul_(0.25)
        .requires_grad_()
        for shape in (
            (nodes, irreps.dim),
            (edges, model.weight_numel),
            (edges, len(degrees), 9),
            (edges, 9, len(degrees)),
            (edges, 5),
        )
    ]
    x, weight, rotation, inverse, radial = inputs
    cutoff = torch.rand(edges, 1, device=device, dtype=torch.float64).requires_grad_()
    with torch.no_grad():
        cutoff[0] = 0
    index = torch.tensor([[0, 1, 2, 3, 1, 2, 3], [1, 2, 3, 0, 0, 0, 0]], device=device)
    args = (x, weight, index, cutoff, rotation * mask, inverse * mask.T, radial)
    expected = reference(*args)
    actual = model(*args, fused=device == "cuda")
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=1e-10)
    torch.testing.assert_close(restored(*args), expected, atol=2e-12, rtol=1e-10)
    if mmax == 2 and ece and attention and not gate_m0 and device == "cpu":
        from tace.models.compile.compile import trace_to_fx

        def evaluate(*values):
            output = model(*values)
            derivative = torch.autograd.grad(
                output.sum(), values[0], create_graph=True
            )[0]
            return output, derivative

        traced = trace_to_fx(evaluate, args)
        assert all(
            "eqx." not in str(node.target)
            for node in traced.graph.nodes
            if node.op == "call_function"
        )
        exported = torch.export.export(traced, args, strict=False)
        compiled = torch.compile(exported.module(), backend="aot_eager", fullgraph=True)
        for a, b in zip(compiled(*args), evaluate(*args)):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    losses = [out.square().sum() for out in (actual, expected)]
    actual_grads = torch.autograd.grad(
        losses[0], tuple(model.parameters()), retain_graph=True
    )
    reference_grads = torch.autograd.grad(
        losses[1], tuple(reference.parameters()), retain_graph=True
    )
    gradient_state = dict(reference.state_dict())
    gradient_state.update(
        (name, grad)
        for (name, _), grad in zip(reference.named_parameters(), reference_grads)
    )
    restored.load_state_dict(gradient_state, strict=True)
    for grad, parameter in zip(actual_grads, restored.parameters()):
        torch.testing.assert_close(grad, parameter, atol=1e-9, rtol=1e-8)
    for _ in range(3):
        gradients = [
            torch.autograd.grad(
                loss, (*inputs, cutoff), create_graph=True, allow_unused=True
            )
            for loss in losses
        ]
        for a, b in zip(*gradients):
            if a is None or b is None:
                assert a is None and b is None
            else:
                torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)
        losses = [
            sum(g.square().sum() for g in gs if g is not None) for gs in gradients
        ]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("edges", [0, 13])
def test_graph_attention_derivatives(device, edges):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from eqx.conv.attention import graph_softmax
    from tace.models.softmax import GraphSoftmax

    target = torch.arange(edges, device=device) % 4
    scores = torch.randn(
        edges, 3, dtype=torch.float64, device=device, requires_grad=True
    )
    weight = torch.rand(edges, 1, dtype=torch.float64, device=device)
    weight[::3] = 0
    weight.requires_grad_()
    actual = graph_softmax(scores, target, 5, weight, eps=1e-3)
    expected = GraphSoftmax(eps=1e-3)(scores, target, num_nodes=5, exp_rescale=weight)
    torch.testing.assert_close(actual, expected)
    losses = [x.square().sum() for x in (actual, expected)]
    for _ in range(3):
        grads = [
            torch.autograd.grad(loss, (scores, weight), create_graph=True)
            for loss in losses
        ]
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-8)
        losses = [sum(g.square().sum() for g in values) for values in grads]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("nodes", [0, 5])
def test_streaming_graph_attention_weights(device, nodes):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from eqx.conv.attention import StreamingGraphAttention
    from tace.models.softmax import GraphSoftmax

    edges, heads = (9 if nodes else 0), 2
    target = torch.arange(edges, device=device) % 3
    inputs = [
        torch.zeros(nodes, 1, device=device, dtype=torch.float64),
        torch.randn(edges, heads, device=device, dtype=torch.float64).requires_grad_(),
        torch.randn(edges, 3, 4, device=device, dtype=torch.float64).requires_grad_(),
        torch.rand(edges, 1, device=device, dtype=torch.float64).requires_grad_(),
        torch.rand(edges, heads, device=device, dtype=torch.float64).requires_grad_(),
        target,
        torch.ones(edges, device=device, dtype=torch.bool),
    ]
    with torch.no_grad():
        inputs[3][::3] = 0

    def fake(v):
        return (
            v[0].new_empty(nodes, 3, 4),
            v[0].new_empty(nodes, heads),
            v[0].new_empty(nodes, heads),
        )

    attention = StreamingGraphAttention(
        lambda v: (v[1], v[2]),
        fake,
        {i: 0 for i in range(1, 7)},
        target_slot=5,
        normalizer_slot=3,
        value_weight_slot=4,
        valid_slot=6,
        tile_size=4,
        eps=1e-3,
    )
    actual = attention(*inputs)[0]
    weight = (
        GraphSoftmax(eps=1e-3)(
            inputs[1], target, num_nodes=nodes, exp_rescale=inputs[3]
        )
        * inputs[4]
    )
    value = (inputs[2].view(edges, 3, heads, 2) * weight[:, None, :, None]).reshape(
        edges, 3, 4
    )
    expected = value.new_zeros(nodes, 3, 4).index_add(0, target, value)
    torch.testing.assert_close(actual, expected)
    losses = [x.square().sum() for x in (actual, expected)]
    for _ in range(3):
        grads = [
            torch.autograd.grad(loss, inputs[1:5], create_graph=True) for loss in losses
        ]
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-8)
        losses = [sum(g.square().sum() for g in values) for values in grads]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("edges", [0, 7, 65, 257])
def test_tece_streaming_derivatives(monkeypatch, device, edges):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from eqx.models.tace.tece_oam_rra.interaction import stream
    from tace.models._e3nn.tece_oam_rra import Convolution
    from tace.models.layout import LayoutTransform

    monkeypatch.setenv("TACE_USE_EQX", "1")
    irreps = o3.Irreps("4x0e+4x1o+4x2e")
    module = Convolution(
        2,
        2,
        4,
        2,
        2,
        3,
        False,
        True,
        True,
        LayoutTransform(irreps),
        LayoutTransform(irreps),
        torch.nn.SiLU(),
        torch.nn.Sigmoid(),
    ).to(device=device, dtype=torch.float64)
    with torch.no_grad():
        module.radial_proj.weight.normal_(std=0.1)
    nodes = 256 if edges == 65 else 4
    values = [
        torch.randn(*shape, device=device, dtype=torch.float64)
        .mul_(0.15)
        .requires_grad_()
        for shape in [
            (nodes, irreps.dim),
            (edges, 3),
            (3, module.weight_numel),
            (module.weight_numel,),
            (edges, 9, 9),
            (edges, 9, 9),
            (edges, 3),
        ]
    ]
    x, radial, projection, bias, rotation, inverse, basis = values
    # Frames rotate each O(3) degree independently; rows are ordered by m.
    degrees = torch.tensor([0, 1, 2, 1, 2, 1, 2, 2, 2], device=device)
    columns = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2], device=device)
    mask = degrees[:, None] == columns[None, :]
    rotation, inverse = rotation * mask, inverse * mask.T
    cutoff = torch.rand(edges, 1, device=device, dtype=torch.float64).requires_grad_()
    index = torch.stack(
        (
            torch.arange(edges, device=device) % nodes,
            (torch.arange(edges, device=device) + 1) % nodes,
        )
    )
    if edges == 257:
        # Exercise split neighborhoods, empty receivers and zero edge weights.
        index[1] = (torch.arange(edges, device=device) % 5 == 0).long()
        with torch.no_grad():
            cutoff[::7] = 0
    actual = stream(
        module,
        x,
        radial,
        projection,
        bias,
        index,
        cutoff,
        rotation,
        inverse,
        basis,
    )
    expected = module(
        x, radial @ projection + bias, index, cutoff, rotation, inverse, basis
    )
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    inputs = (*values, cutoff, *module.parameters())
    losses = [v.square().sum() for v in (actual, expected)]
    for _ in range(3):
        grads = [
            torch.autograd.grad(loss, inputs, create_graph=True, allow_unused=True)
            for loss in losses
        ]
        for i, (a, b) in enumerate(zip(*grads)):
            a = torch.zeros_like(inputs[i]) if a is None else a
            b = torch.zeros_like(inputs[i]) if b is None else b
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)
        losses = [
            sum(g.square().sum() for g in values if g is not None) for values in grads
        ]
    if device == "cuda" and edges == 65:
        with torch.no_grad():
            module.temperature_logit.add_(0.2)
            projection.add_(0.01)
        actual = stream(
            module,
            x,
            radial,
            projection,
            bias,
            index.flip(0),
            cutoff,
            rotation,
            inverse,
            basis,
        )
        expected = module(
            x,
            radial @ projection + bias,
            index.flip(0),
            cutoff,
            rotation,
            inverse,
            basis,
        )
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_online_attention_merge(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from eqx.conv.attention import merge_attention

    torch.manual_seed(6)
    scores = torch.randn(3, 5, 7, 2, device="cuda", dtype=dtype) * 20 + 1000
    cutoff = torch.rand(3, 5, 7, 1, device="cuda", dtype=dtype)
    value = torch.randn(3, 5, 7, 4, 2, 3, device="cuda", dtype=dtype)
    scores[0] = -torch.inf
    scores[1, 0] = -torch.inf
    cutoff[2, 1:] = 0
    maximum = scores.amax(2)
    shift = torch.where(torch.isfinite(maximum), maximum, 0)
    exponential = (scores - shift[:, :, None]).exp()
    denominator = (exponential * cutoff).sum(2)
    partial = (
        exponential[:, :, :, None, :, None]
        * cutoff[:, :, :, None, :, None].square()
        * value
    ).sum(2)
    eps = 0.03
    outputs = [
        value.new_empty(3, 4, 6),
        value.new_empty(3, 2),
        value.new_empty(3, 2),
    ]
    merge_attention(partial.reshape(3, 5, 4, 6), denominator, maximum, 6, eps, outputs)
    global_maximum = scores.amax((1, 2))
    global_maximum = torch.where(torch.isfinite(global_maximum), global_maximum, 0)
    exponential = (scores - global_maximum[:, None, None]).exp()
    total = (exponential * cutoff).sum((1, 2)) + eps
    expected = (
        exponential[:, :, :, None, :, None]
        * cutoff[:, :, :, None, :, None].square()
        * value
    ).sum((1, 2))
    expected = (expected / total[:, None, :, None]).reshape(3, 4, 6)
    for actual, reference in zip(outputs, (expected, total, global_maximum)):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("ece", [False, True])
@pytest.mark.parametrize("gate_m0", [False, True])
@pytest.mark.parametrize("mmax", [0, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tece_native_variants(monkeypatch, ece, gate_m0, mmax, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from eqx.kernels import recompute
    from eqx.models.tace.tece_oam_rra.interaction import stream
    from tace.models._e3nn.tece_oam_rra import Convolution
    from tace.models.layout import LayoutTransform
    from tace.models.mlp import get_scaled_activation

    def no_replay(*args, **kwargs):
        raise AssertionError("The native interaction must not use Graph replay.")

    monkeypatch.setattr(recompute, "replay", no_replay)
    torch.manual_seed(20)
    channels = 16 if ece and mmax == 2 and dtype == torch.float64 else 2
    irreps = o3.Irreps([(channels, (l, (-1) ** l)) for l in range(3)])
    module = Convolution(
        mmax,
        2,
        channels,
        1,
        2,
        3,
        gate_m0,
        ece,
        True,
        LayoutTransform(irreps),
        LayoutTransform(irreps),
        get_scaled_activation("tanh"),
        get_scaled_activation("silu"),
    ).to(device="cuda", dtype=dtype)
    angular = sum((3 - m) * (1 if m == 0 else 2) for m in range(mmax + 1))
    inputs = [
        (torch.randn(*shape, device="cuda", dtype=dtype) * 0.1).requires_grad_()
        for shape in (
            (4, irreps.dim),
            (5, 3),
            (3, module.weight_numel),
            (5, angular, 9),
            (5, 9, angular),
            (5, 3),
        )
    ]
    x, radial, projection, rotation, inverse, basis = inputs
    cutoff = torch.tensor(
        [[0.0], [0.0], [0.7], [1.0], [0.2]],
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    index = torch.tensor([[0, 1, 2, 3, 0], [1, 1, 2, 2, 2]], device="cuda")
    with torch.no_grad():
        module.radial_proj.weight.normal_(std=0.2)
    actual = stream(
        module, x, radial, projection, None, index, cutoff, rotation, inverse, basis
    )
    expected = module(x, radial @ projection, index, cutoff, rotation, inverse, basis)
    atol, rtol = (2e-6, 1e-4) if dtype == torch.float32 else (2e-11, 1e-10)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    variables = (*inputs, cutoff, *module.parameters())
    grads = [
        torch.autograd.grad(value.square().sum(), variables, create_graph=True)
        for value in (actual, expected)
    ]
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=10 * atol, rtol=10 * rtol)


def test_shared_metadata_cache():
    from eqx import utils
    from eqx.conv.program import parse_metadata

    assert parse_metadata is utils.parse_metadata
    specification = (2, ((0, 1, "uvu", True, 0.5),))
    metadata = repr(specification)
    result = parse_metadata(metadata)
    assert result == specification
    assert parse_metadata(metadata) is result
    assert parse_metadata.cache_info().maxsize == 256
    assert parse_metadata("None") is None
    with pytest.raises(ValueError):
        parse_metadata("tuple()")


def test_convolution_package_layout():
    from eqx import ace, models
    from eqx.conv import uu_o2, uv_o2
    from eqx.kernels.channel_product import local_product
    from eqx.models.mace import convert_mace_to_eqx
    from eqx.models.tace import tece_oam_rra
    from eqx.models.tace.tece_oam_rra import LocalSplit
    from tace.models._e3nn.tece_oam_rra import Convolution

    assert ace.TACE is TACE
    assert models.__name__ == "eqx.models"
    assert TACE.__module__ == "eqx.ace.tace"
    assert not hasattr(eqx_conv, "TACE")
    assert BilinearACE.__module__ == "eqx.models.tace.tece_oam_rra.product"
    assert Convolution.__module__ == "tace.models._e3nn.tece_oam_rra"
    assert not hasattr(tece_oam_rra, "Convolution")
    assert uu_o2.UuO2TensorProductConv is eqx_conv.UuO2TensorProductConv
    assert uv_o2.UvO2TensorProductConv is eqx_conv.UvO2TensorProductConv
    assert issubclass(LocalSplit, torch.autograd.Function)
    assert callable(local_product)
    assert callable(convert_mace_to_eqx)
    assert convert_mace_to_eqx.__module__ == "eqx.models.mace.conversion"


@pytest.fixture
def mace_model(double_precision):
    pytest.importorskip("mace")
    import numpy as np
    from mace import modules

    def make(interaction="RealAgnosticResidualInteractionBlock"):
        return modules.ScaleShiftMACE(
            r_max=3.0,
            num_bessel=3,
            num_polynomial_cutoff=5,
            max_ell=2,
            interaction_cls=getattr(modules, interaction),
            interaction_cls_first=modules.RealAgnosticInteractionBlock,
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("2x0e+2x1o+2x2e"),
            MLP_irreps=o3.Irreps("2x0e"),
            atomic_energies=np.array([0.2, -0.3]),
            avg_num_neighbors=2.0,
            atomic_numbers=[1, 8],
            correlation=2,
            gate=torch.nn.functional.silu,
            radial_MLP=[4],
            atomic_inter_scale=1.2,
            atomic_inter_shift=-0.4,
        )

    return make


@pytest.fixture
def mace_data(mace_model):
    from ase import Atoms
    from mace import data
    from mace.tools import AtomicNumberTable, torch_geometric

    atoms = Atoms(
        "OH2",
        positions=[[0.2, 0.4, 0.1], [1.1, 0.2, 0.3], [-0.1, 1.3, 0.5]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )
    graph = data.AtomicData.from_config(
        data.config_from_atoms(atoms),
        z_table=AtomicNumberTable([1, 8]),
        cutoff=3.0,
    )
    other = deepcopy(graph)
    other.positions = other.positions * 1.1
    return atoms, torch_geometric.Batch.from_data_list([graph, other])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "interaction",
    [
        "RealAgnosticInteractionBlock",
        "RealAgnosticResidualInteractionBlock",
        "RealAgnosticDensityInteractionBlock",
        "RealAgnosticDensityResidualInteractionBlock",
        "RealAgnosticAttResidualInteractionBlock",
        "RealAgnosticResidualNonLinearInteractionBlock",
    ],
)
def test_mace_conversion_training(mace_model, mace_data, interaction, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.models.mace import convert_mace_to_eqx

    reference = mace_model(interaction).to(device)
    converted = deepcopy(reference)
    parameters = tuple(converted.parameters())
    assert convert_mace_to_eqx(converted, inplace=True) is converted
    assert set(converted.parameters()) == set(parameters)
    assert type(converted) is type(reference)
    assert all(not hasattr(layer, "conv_fusion") for layer in reference.interactions)
    batch = mace_data[1].to(device)
    expected = reference(batch.clone().to_dict(), training=True, compute_stress=True)
    actual = converted(batch.clone().to_dict(), training=True, compute_stress=True)
    for key in ("energy", "forces", "stress", "virials"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-10, rtol=2e-9)
    # Force and stress losses differentiate through the convolution twice.
    for output in (actual, expected):
        loss = sum(output[key].square().sum() for key in ("energy", "forces", "stress"))
        loss.backward()
    for actual_param, expected_param in zip(parameters, reference.parameters()):
        if expected_param.grad is None:
            assert actual_param.grad is None
        else:
            torch.testing.assert_close(
                actual_param.grad, expected_param.grad, atol=2e-9, rtol=2e-8
            )
    projection = converted.interactions[1].conv_tp.projection.weight
    before = projection.detach().clone()
    torch.optim.SGD(converted.parameters(), lr=0.01).step()
    assert not torch.equal(projection, before)


def test_mace_conversion_ase_and_checkpoint(mace_model, mace_data, tmp_path):
    from mace.calculators import MACECalculator

    from eqx.models.mace import convert_mace_to_eqx

    original = mace_model().eval()
    original.interactions[0].conv_tp_weights.requires_grad_(False)
    # Conversion follows the model precision without changing the caller's default.
    torch.set_default_dtype(torch.float32)
    converted = convert_mace_to_eqx(original)
    assert torch.get_default_dtype() == torch.float32
    assert not converted.training
    assert next(converted.parameters()).dtype == torch.float64
    assert not converted.interactions[0].conv_tp.projection.weight.requires_grad
    assert converted is not original
    assert not hasattr(original.interactions[0], "conv_fusion")
    assert convert_mace_to_eqx(converted, inplace=True) is converted
    assert convert_mace_to_eqx(converted) is not converted
    checkpoint = tmp_path / "mace-eqx.model"
    torch.save(converted, checkpoint)
    restored = torch.load(checkpoint, weights_only=False)
    reloaded = convert_mace_to_eqx(original)
    reloaded.load_state_dict(converted.state_dict(), strict=True)
    results = []
    for model in (original, restored, reloaded):
        atoms = mace_data[0].copy()
        atoms.calc = MACECalculator(models=model, device="cpu", default_dtype="float64")
        results.append(
            (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
        )
    for actual in results[1:]:
        for value, expected in zip(actual, results[0]):
            torch.testing.assert_close(
                torch.as_tensor(value), torch.as_tensor(expected), atol=2e-10, rtol=2e-9
            )


def test_mace_conversion_solid_harmonics(mace_model, mace_data):
    from eqx.models.mace import convert_mace_to_eqx

    original = mace_model()
    original.spherical_harmonics.normalize = False
    converted = convert_mace_to_eqx(original)
    batch = mace_data[1]
    expected = original(batch.clone().to_dict(), training=True)
    actual = converted(batch.clone().to_dict(), training=True)
    for key in ("energy", "forces"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_mace_conversion_cueq(mace_model, mace_data, layout):
    if not torch.cuda.is_available():
        pytest.skip("cuEquivariance requires CUDA")
    pytest.importorskip("cuequivariance_torch")
    from mace.cli.convert_e3nn_cueq import run

    from eqx.models.mace import convert_mace_to_eqx

    original = mace_model().cuda()
    if layout == "ir_mul":
        converted = convert_mace_to_eqx(original, enable_cueq=True)
    else:
        converted = convert_mace_to_eqx(run(original, device="cuda", layout=layout))
    assert converted.interactions[0].cueq_config.layout_str == layout
    batch = mace_data[1].cuda()
    expected = original(batch.clone().to_dict(), training=True, compute_stress=True)
    actual = converted(batch.clone().to_dict(), training=True, compute_stress=True)
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-10, rtol=2e-9)
    loss = sum(actual[key].square().sum() for key in ("energy", "forces", "stress"))
    loss.backward()
    gradient = converted.interactions[1].conv_tp.projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("direction", [False, True])
@pytest.mark.parametrize(
    "mmax,shared,projected",
    [(0, False, False), (1, True, False), (2, False, True), (1, True, True)],
)
def test_uu_o2_convolution_derivatives(
    double_precision, device, direction, mmax, shared, projected
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    frame_in = o2.LocalFrame("2x0e+2x1o+2x1e+2x2o", mmax=mmax).to(device)
    frame_out = o2.LocalFrame("2x0o+2x1e+2x1o+2x2e", mmax=mmax).to(device)
    linear = o2.UuLinear(frame_in.irreps_out, frame_out.irreps_out, 2)
    module = eqx_conv.UuO2TensorProductConv(frame_in, linear, frame_out).to(device)
    frame = o2.WignerD(2, 2).to(device)
    index = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 2, 0]], device=device)
    rows = 1 if shared else index.size(1)

    def rand(*shape):
        return (torch.randn(*shape, device=device) * 0.2).requires_grad_()

    x, vectors = rand(3, frame_in.input_dim), rand(rows, 3)
    radial = rand(rows, 3 if projected else linear.weight_numel)
    projection = (
        rand(3, linear.weight_numel)
        if projected
        else torch.empty(0, linear.weight_numel, device=device)
    )
    cutoff = rand(rows, 1)
    packed = frame.forward_packed(vectors)
    inputs = (x, vectors, radial, cutoff) + ((projection,) if projected else ())
    actual = module(
        x,
        radial,
        projection,
        packed,
        cutoff,
        index,
        3,
        vectors=vectors if direction else None,
    )
    wigner, inverse = frame(vectors.expand(index.size(1), -1))
    local = frame_in.to_local(x[index[0]], wigner)
    message = frame_out.to_global(
        linear(local, radial @ projection if projected else radial), inverse
    )
    expected = x.new_zeros(3, frame_out.input_dim).index_add(
        0, index[1], message * cutoff
    )
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
    for _ in range(3 if direction and mmax == 1 and projected else 2):
        gradients = [
            torch.autograd.grad(value.sin().sum(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-8)
        actual, expected = [
            torch.cat([g.flatten() for g in grads]) / 20 for grads in gradients
        ]


@pytest.mark.parametrize("channels,shared", [(3, False), (33, True)])
def test_uu_o2_chunked_reduction(monkeypatch, double_precision, channels, shared):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.conv.o2_o3 import cuda

    monkeypatch.setattr(cuda, "CHUNK_SIZE", 1025)
    frame_in = o2.LocalFrame(f"{channels}x1e+{channels}x1o").cuda()
    frame_out = o2.LocalFrame(f"{channels}x0o+{channels}x1o").cuda()
    linear = o2.UuLinear(frame_in.irreps_out, frame_out.irreps_out, channels)
    module = eqx_conv.UuO2TensorProductConv(frame_in, linear, frame_out).cuda()
    reference = eqx_conv.UuO2TensorProductConv(
        frame_in, linear, frame_out, backend="torch"
    ).cuda()
    frame = o2.WignerD(1, 1).cuda()
    edges, nodes = 2051, 129
    index = torch.randint(nodes, (2, edges), device="cuda")
    rows = 1 if shared else edges
    inputs = [
        (0.2 * torch.randn(shape, device="cuda")).requires_grad_()
        for shape in (
            (nodes, frame_in.input_dim),
            (rows, 3),
            (rows, 4),
            (4, linear.weight_numel),
            (rows, 1),
        )
    ]
    x, vectors, radial, projection, cutoff = inputs
    packed = frame.forward_packed(vectors)
    args = x, radial, projection, packed, cutoff, index, nodes
    actual, expected = module(*args, vectors=vectors), reference(*args)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-10)
    for _ in range(2):
        gradients = [
            torch.autograd.grad(value.sin().mean(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=3e-10, rtol=3e-9)
        actual, expected = [
            torch.cat([g.flatten() for g in values]) for values in gradients
        ]


def test_uu_o2_compile_and_empty_graph(double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    frame_in = o2.LocalFrame("3x0e+3x1e").cuda()
    frame_out = o2.LocalFrame("3x0o+3x1o").cuda()
    linear = o2.UuLinear(frame_in.irreps_out, frame_out.irreps_out, 3)
    module = eqx_conv.UuO2TensorProductConv(frame_in, linear, frame_out).cuda()
    frame = o2.WignerD(1, 1).cuda()

    def evaluate(x, vectors, radial, projection, cutoff, index):
        return module(
            x,
            radial,
            projection,
            frame.forward_packed(vectors.detach()),
            cutoff,
            index,
            x.size(0),
            vectors=vectors,
        )

    compiled = torch.compile(evaluate, fullgraph=True, dynamic=True)
    for nodes, edges in ((3, 7), (3, 1), (3, 0), (0, 0)):
        index = torch.randint(max(nodes, 1), (2, edges), device="cuda")
        inputs = [
            torch.randn(shape, device="cuda", requires_grad=True)
            for shape in (
                (nodes, frame_in.input_dim),
                (edges, 3),
                (edges, 2),
                (2, linear.weight_numel),
                (edges, 1),
            )
        ]
        actual, expected = compiled(*inputs, index), evaluate(*inputs, index)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        gradients = [
            torch.autograd.grad(value.square().sum(), inputs)
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-9)
        if edges == 7:
            from tace.models.compile.compile import trace_to_fx

            def energy_forces(*values):
                energy = evaluate(*values).square().sum()
                forces = -torch.autograd.grad(energy, values[1], create_graph=True)[0]
                return energy, forces

            traced = trace_to_fx(energy_forces, (*inputs, index))
            compiled_forces = torch.compile(traced, fullgraph=True)
            actual = compiled_forces(*inputs, index)
            expected = energy_forces(*inputs, index)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-9)
            gradients = [
                torch.autograd.grad(energy + forces.square().sum(), inputs)
                for energy, forces in (actual, expected)
            ]
            for a, b in zip(*gradients):
                torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-8)


def test_uu_o2_time_parity_and_duplicate_entries(double_precision):
    if not hasattr(o3.Irrep("0e"), "t"):
        pytest.skip("Time-reversal representations are unavailable")
    irreps_in = "2x0ee+2x1eo+2x1oe+2x1eo"
    irreps_out = "2x0oo+2x1oe+2x1oo+2x1eo"
    frame_in, frame_out = o2.LocalFrame(irreps_in), o2.LocalFrame(irreps_out)
    linear = o2.UuLinear(frame_in.irreps_out, frame_out.irreps_out, 2)
    module = eqx_conv.UuO2TensorProductConv(frame_in, linear, frame_out)
    index = torch.tensor([[0, 1, 0], [1, 0, 1]])
    features = torch.randn(2, frame_in.input_dim)
    vectors = torch.randn(3, 3)
    weights = torch.randn(3, linear.weight_numel)
    cutoff = torch.rand(3, 1)
    frame = o2.WignerD(1, 1)
    wigner, inverse = frame(vectors)
    expected = frame_out.to_global(
        linear(frame_in.to_local(features[index[0]], wigner), weights), inverse
    )
    expected = features.new_zeros(2, frame_out.input_dim).index_add(
        0, index[1], expected * cutoff
    )
    actual = module(
        features,
        weights,
        weights.new_empty(0, linear.weight_numel),
        frame.forward_packed(vectors),
        cutoff,
        index,
        2,
        vectors=vectors,
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_local_channel_scaling():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.kernels.channel_product import local_product

    indices = [0, 1, 2, 1, 2]
    metadata = repr(
        (4, (5, 3, 5), tuple(((i, j, i), 1.0) for i, j in enumerate(indices)))
    )
    x = torch.randn(3, 5, 8, device="cuda", dtype=torch.float64)[
        ..., ::2
    ].requires_grad_()
    w = torch.randn(3, 3, 4, device="cuda", dtype=torch.float64, requires_grad=True)
    actual = local_product(metadata, x, w).view_as(x)
    expected = x * w[:, indices]
    torch.testing.assert_close(actual, expected)
    losses = [z.square().sum() for z in (actual, expected)]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, (x, w), create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)
        losses = [sum(g.square().sum() for g in gs) for gs in gradients]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("experts", [1, 2, 4])
@pytest.mark.parametrize("nodes", [0, 3, 5])
@pytest.mark.parametrize("shared", [False, True])
def test_bilinear_ace(device, experts, nodes, shared):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = o3.TensorProduct(
        "4x0e+4x1o",
        "4x0e+4x1o+4x0e",
        "4x0e+4x0e+4x1o",
        [(0, 0, 0, "uuu", True), (1, 1, 1, "uuu", True), (1, 2, 2, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    linear = o3.Linear(
        [(mul // experts, ir) for mul, ir in tp.irreps_out.simplify()],
        f"{4 // experts}x0e+{4 // experts}x1o",
        internal_weights=False,
        shared_weights=False,
    )
    shared_linear = (
        o3.Linear(
            tp.irreps_out.simplify(),
            "4x0e+4x1o",
            internal_weights=False,
            shared_weights=False,
        )
        if shared
        else None
    )
    module = BilinearACE(tp, linear, experts, shared_linear=shared_linear).to(
        device=device, dtype=torch.float64
    )
    x, y, gates = [
        torch.randn(nodes, 2 * size, dtype=torch.float64, device=device)[
            :, ::2
        ].requires_grad_()
        for size in (16, 20, tp.weight_numel)
    ]
    weight = torch.randn(
        3,
        experts,
        linear.weight_numel,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    types = torch.arange(nodes, device=device) % 3
    shared_weight = (
        torch.randn(
            shared_linear.weight_numel,
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        if shared
        else None
    )
    actual = module(x, y, gates, weight, types, shared_weight)
    # Compare against the existing tensor product and independent coefficient map.
    module.coefficients.backend = "torch"
    expected = module.coefficients(tp(x, y, gates), weight, types)
    inputs = (x, y, gates, weight)
    if shared:
        expected = expected + shared_linear(
            tp(x, y, gates), shared_weight.expand(nodes, -1)
        )
        inputs = (*inputs, shared_weight)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    losses = [value.square().sum() for value in (actual, expected)]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(
                a, b, atol=1e-8, rtol=1e-9, msg=f"Derivative order {_ + 1}"
            )
        losses = [sum(value.square().sum() for value in grad) for grad in gradients]
    assert not module.state_dict()
    if device == "cuda" and nodes and experts == 2:
        compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(x, y, gates, weight, types, shared_weight),
            expected,
            atol=1e-11,
            rtol=1e-11,
        )


@pytest.mark.parametrize("matrix", ["0", "1"])
@pytest.mark.parametrize(
    "experts,shared,agnostic",
    [(1, False, False), (2, False, False), (2, True, False), (1, False, True)],
)
def test_bilinear_product_integration(monkeypatch, matrix, experts, shared, agnostic):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.prod import BilinearMoEACE
    from tace.models.linear import switch_e3nn_weight_layout

    monkeypatch.setenv("TACE_USE_EQX", "1")
    model = (
        BilinearMoEACE(
            layer=0,
            num_layers=1,
            num_elements=2,
            Lmax=1,
            lmax=1,
            num_channel=4,
            num_expert=experts,
            num_channel_per_expert=2,
            target_irreps=o3.Irreps("0e+1o"),
            irreps_in=o3.Irreps("4x0e+4x1o"),
            correlation=[2],
            l1l2=None,
            bias=True,
            nonlinear="silu_bilineargate",
            parity=True,
            use_shared_expert=shared,
            agnostic=agnostic,
        )
        .double()
        .cuda()
    )
    switch_e3nn_weight_layout(model, "matrix" if matrix == "1" else "flat")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "bias" in name:
                p.normal_()
    reference = deepcopy(model)
    del reference.eqx_ace
    x = torch.randn(3, 16, device="cuda", dtype=torch.float64, requires_grad=True)
    attrs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.2, 0.8]], device="cuda", dtype=torch.float64
    )
    batch = torch.zeros(3, device="cuda", dtype=torch.long)
    actual, expected = [m(x, attrs, None, batch) for m in (model, reference)]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    gradients = [
        torch.autograd.grad(y.square().sum(), (x, *m.parameters()), create_graph=True)
        for y, m in ((actual, model), (expected, reference))
    ]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
    # A loss on input derivatives exercises force-training parameter gradients.
    second = [
        torch.autograd.grad(
            g[0].square().sum(), tuple(m.parameters()), allow_unused=True
        )
        for g, m in zip(gradients, (model, reference))
    ]
    for a, b in zip(*second):
        if a is not None:
            torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)
    assert model.state_dict().keys() == reference.state_dict().keys()


def test_cuda_cache_does_not_log_lock_creation(tmp_path, monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    from eqx.kernels.cuda import compile_binary

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    caplog.set_level(logging.DEBUG, logger="filelock")
    calls = []
    compiler = SimpleNamespace(
        version=lambda: (1, 0),
        compile=lambda source, options: calls.append(source) or b"test binary",
    )
    for _ in range(2):
        assert compile_binary("test source", (), compiler) == b"test binary"
    assert calls == ["test source"]
    assert not [record for record in caplog.records if record.name == "filelock"]
    logger = logging.getLogger("filelock")
    logger.debug("Other lock: %s", str(tmp_path / "other.lock"))
    logger.warning("Cache warning: %s", str(tmp_path / "eqx/cuda/test.lock"))
    assert len([record for record in caplog.records if record.name == "filelock"]) == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_nodes", [0, 3])
def test_ace_external_coefficients(device, num_nodes):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    irreps = o3.Irreps("2x0e+2x1o")
    tp = o3.TensorProduct(
        irreps,
        irreps,
        "2x0e+2x0e+2x1o",
        [(0, 0, 0, "uuu", False), (1, 1, 1, "uuu", False), (0, 1, 2, "uuu", False)],
        internal_weights=False,
        shared_weights=False,
    )
    linears = [
        o3.Linear(inp, irreps, internal_weights=False, shared_weights=False)
        for inp in (irreps, tp.irreps_out.simplify())
    ]
    module = TACE([tp], linears).to(device=device, dtype=torch.float64)
    x = torch.randn(num_nodes, 16, device=device, dtype=torch.float64)[
        :, ::2
    ].requires_grad_()
    types = (torch.arange(num_nodes * 2, device=device) % 3)[::2]
    weights = [
        torch.randn(
            3,
            linear.weight_numel,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        for linear in linears
    ]
    actual = module(x, weights, types)
    expected = linears[0](x, weights[0][types]) + linears[1](
        tp(x, x), weights[1][types]
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    gradients = [
        torch.autograd.grad(y.square().sum(), (x, *weights), create_graph=True)
        for y in (actual, expected)
    ]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    assert not module.state_dict()
    if device == "cuda" and num_nodes:
        compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(x, weights, types), expected, atol=1e-12, rtol=1e-12
        )


@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
def test_cartesian_harmonic_tables(normalization):
    from eqx.conv.o3.harmonics import polynomial_coefficients

    vectors = torch.randn(7, 3, dtype=torch.float64)
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    for degree in range(9):
        actual = torch.stack(
            [
                sum(
                    coefficient * (vectors ** vectors.new_tensor(powers)).prod(-1)
                    for powers, coefficient in terms
                )
                for terms in polynomial_coefficients(degree, normalization)
            ],
            dim=-1,
        )
        expected = o3.spherical_harmonics(
            degree, vectors, normalize=False, normalization=normalization
        )
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
@pytest.mark.parametrize("shared,direct", [(False, False), (True, True)])
def test_o3_cartesian_derivatives(
    device, normalization, shared, direct, double_precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = o3.TensorProduct(
        "3x1o+3x2e",
        "2x1o+2e",
        "3x1o+3x2e+3x3o",
        [
            (0, 0, 1, "uvu", True),
            (1, 0, 0, "uvu", True),
            (1, 0, 2, "uvu", True),
            (1, 1, 1, "uvu", False),
        ],
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    conv = eqx_conv.O3TensorProductConv(tp, normalization=normalization).to(device)
    reference = eqx_conv.O3TensorProductConv(
        tp, normalization=normalization, backend="torch"
    ).to(device)
    edges = torch.randint(4, (2, 9), device=device)
    x = torch.randn(4, tp.irreps_in1.dim, device=device, requires_grad=True)
    vectors = torch.randn(9, 3, device=device, requires_grad=True)
    amplitudes = torch.randn(1 if shared else 9, 3, device=device, requires_grad=True)
    radial = torch.randn(
        1 if shared else 9,
        tp.weight_numel if direct else 4,
        device=device,
        requires_grad=True,
    )
    projection = torch.randn(
        0 if direct else 4, tp.weight_numel, device=device, requires_grad=True
    )
    inputs = x, vectors, amplitudes, radial, projection
    actual = conv(
        x, None, radial, projection, edges, vectors=vectors, amplitudes=amplitudes
    )
    expected = reference(
        x, None, radial, projection, edges, vectors=vectors, amplitudes=amplitudes
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for _ in range(3):
        seed = torch.randn_like(actual) / max(1, actual.numel()) ** 0.5
        a = torch.autograd.grad((actual.sin() * seed).sum(), inputs, create_graph=True)
        b = torch.autograd.grad(
            (expected.sin() * seed).sum(), inputs, create_graph=True
        )
        for value, target in zip(a, b):
            torch.testing.assert_close(value, target, atol=3e-9, rtol=3e-9)
        actual, expected = (
            torch.cat([value.flatten() for value in values]) for values in (a, b)
        )


def test_o3_cartesian_compile_and_empty_graph(double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = o3.TensorProduct(
        "2x1o",
        "1o",
        "2x0e+2x2e",
        [(0, 0, 0, "uvu", True), (0, 0, 1, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    conv = eqx_conv.O3TensorProductConv(tp, normalize=False).cuda()
    reference = eqx_conv.O3TensorProductConv(
        tp, normalize=False, backend="torch"
    ).cuda()
    compiled = torch.compile(conv, backend="aot_eager", fullgraph=True, dynamic=True)
    for edges_count in (1, 7, 0):
        edges = torch.randint(3, (2, edges_count), device="cuda")
        x = torch.randn(3, 6, device="cuda", requires_grad=True)
        vectors = torch.randn(edges_count, 3, device="cuda", requires_grad=True)
        radial = torch.randn(edges_count, 2, device="cuda", requires_grad=True)
        projection = torch.randn(2, tp.weight_numel, device="cuda", requires_grad=True)
        amplitudes = torch.randn(edges_count, 1, device="cuda", requires_grad=True)
        inputs = x, vectors, radial, projection, amplitudes
        actual = compiled(
            x, None, radial, projection, edges, vectors=vectors, amplitudes=amplitudes
        )
        expected = reference(
            x, None, radial, projection, edges, vectors=vectors, amplitudes=amplitudes
        )
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
        a = torch.autograd.grad(actual.square().sum(), inputs)
        b = torch.autograd.grad(expected.square().sum(), inputs)
        for value, target in zip(a, b):
            torch.testing.assert_close(value, target, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("source_degree", [0, 5])
def test_sparse_direction_rotations(source_degree, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    harmonic_degree = 5 if source_degree == 0 else 1
    tp = o2.O3TensorProduct(
        [(3, (source_degree, (-1) ** source_degree))],
        [(1, (harmonic_degree, (-1) ** harmonic_degree))],
        [(3, (5, (-1) ** (source_degree + harmonic_degree)))],
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    conv = eqx_conv.O2O3TensorProductConv(tp).cuda()
    reference = eqx_conv.O2O3TensorProductConv(tp, backend="torch").cuda()
    frame = o2.WignerD(5, 5).cuda()
    axes = torch.eye(3, device="cuda")
    vectors = torch.cat(
        (axes, -axes, axes + 1e-8, -axes + 1e-8, torch.randn(3, 3, device="cuda"))
    ).requires_grad_()
    edges = torch.randint(4, (2, vectors.size(0)), device="cuda")
    x = torch.randn(4, tp.input_dim, device="cuda", requires_grad=True)
    weights = torch.randn(
        vectors.size(0), tp.weight_numel, device="cuda", requires_grad=True
    )
    amplitude = torch.randn(vectors.size(0), 1, device="cuda", requires_grad=True)
    args = (
        x,
        weights,
        weights.new_empty(0, tp.weight_numel),
        frame.forward_packed(vectors),
        amplitude,
        edges,
        4,
    )
    actual, expected = conv(*args, vectors=vectors), reference(*args)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    for _ in range(2):
        seed = torch.randn_like(actual) / actual.numel() ** 0.5
        derivatives = [
            torch.autograd.grad(
                (value.sin() * seed).sum(),
                (x, vectors, weights, amplitude),
                create_graph=True,
                retain_graph=True,
            )
            for value in (actual, expected)
        ]
        actual, expected = (
            torch.cat([g.flatten() for g in gradients]) for gradients in derivatives
        )
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)


def test_convolution_backends_and_modes():
    from eqx.kernels import wigner_D

    # Exported graphs can resolve the operator before any frame is evaluated.
    assert torch.ops.eqx.quaternion_polynomial.default is not None
    tp = o2.O3TensorProduct("2x0e", "0e", "3x0e", [(0, 0, 0, "uvw", True)])
    assert tp.convolution.instructions[0].connection_mode == "uvw"
    with pytest.raises(NotImplementedError, match="only 'uvu'"):
        eqx_conv.O2O3TensorProductConv(tp, backend="cuda")
    with pytest.raises(ValueError, match="backend must be torch or cuda"):
        eqx_conv.O2O3TensorProductConv(tp, backend="triton")
    with pytest.raises(ValueError, match="backend must be torch or cuda"):
        wigner_D(o2.WignerD(0, 0), torch.randn(2, 3), backend="triton")
    with pytest.raises(ValueError, match="method must be"):
        o2.WignerD(0, 0, method="unknown")
    with pytest.raises(ValueError, match="CUDA float32 or float64"):
        o2.WignerD(1, 1, method="quaternion")(torch.randn(2, 3))


def test_quaternion_coefficients():
    from e3nn import o3

    from eqx.kernels.quaternion import polynomial_coefficients
    from eqx.o2.rotation_matrix import _quaternion_to_matrix

    q = torch.randn(
        6, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(13)
    )
    q = q / q.norm(dim=-1, keepdim=True)
    q = torch.cat((q, -q))
    pointers, exponents, coefficients = polynomial_coefficients(6)
    terms = (q[:, None] ** exponents).prod(-1) * coefficients
    indices = torch.repeat_interleave(
        torch.arange(pointers.numel() - 1), pointers.diff()
    )
    actual = q.new_zeros(q.size(0), pointers.numel() - 1).index_add(1, indices, terms)
    rotation = _quaternion_to_matrix(q)
    blocks = [q.new_ones(q.size(0), 1, 1), rotation]
    for l in range(2, 7):
        cg = o3.wigner_3j(1, l - 1, l, dtype=torch.float64)
        blocks.append(
            torch.einsum("abm,eac,ebd,cdn->emn", cg, rotation, blocks[-1], cg)
            * (2 * l + 1)
        )
    expected = torch.cat([block.flatten(1) for block in blocks], dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("lmax", [0, 1, 3, 5])
def test_quaternion_wigner_frames(lmax):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.kernels import wigner_D

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        axes = torch.eye(3, device="cuda")
        vectors = torch.cat((axes, -axes, axes + 1e-8, -axes + 1e-8))
        # Non-contiguous input, including directions on the chart boundaries.
        vectors = torch.stack((vectors, vectors), dim=-1)[..., 0].requires_grad_()
        frame = o2.WignerD(min(1, lmax), lmax).cuda()
        reference = o2.WignerD(min(1, lmax), lmax, method="recursive").cuda()
        for actual, expected in zip(frame(vectors), reference(vectors)):
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(
            frame.forward_packed(vectors), wigner_D(frame, vectors), atol=0, rtol=0
        )
        for actual, expected in zip(
            frame.matrix_blocks(vectors), reference.matrix_blocks(vectors)
        ):
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        for actual, expected in zip(frame(vectors[:0]), reference(vectors[:0])):
            torch.testing.assert_close(actual, expected)
        if lmax:
            for method in ("quaternion", "recursive"):
                values = frame.forward_packed(vectors, method=method)
                gradient = torch.autograd.grad(
                    values.sin().sum(), vectors, create_graph=True
                )[0]
                assert torch.isfinite(gradient).all()
                assert torch.isfinite(
                    torch.autograd.grad(gradient.square().sum(), vectors)[0]
                ).all()
    finally:
        torch.set_default_dtype(previous)


def test_quaternion_wigner_compile_and_capture():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    frame = o2.WignerD(3, 3).cuda()
    function = torch.compile(
        frame.forward_packed, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for count in (3, 7, 0):
        vectors = torch.randn(count, 3, device="cuda", requires_grad=True)
        actual, expected = function(vectors), frame.forward_packed(vectors)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual.square().sum(), vectors)[0]
        expected_grad = torch.autograd.grad(expected.square().sum(), vectors)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    vectors = torch.randn(11, 3, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            frame.forward_packed(vectors)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = frame.forward_packed(vectors)
    vectors.normal_()
    graph.replay()
    torch.testing.assert_close(output, frame.forward_packed(vectors))


@pytest.mark.parametrize("channels,edge_count", [(3, 5), (64, 5), (129, 5), (3, 1031)])
@pytest.mark.parametrize("merge_paths", [False, True])
def test_same_degree_output_rotations(channels, edge_count, merge_paths):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(37)
        tp = o2.O3TensorProduct(
            f"{channels}x0e+{channels}x1o+{channels}x2e",
            "0e+1o+2e",
            (
                f"{channels}x1o+{channels}x2e"
                if merge_paths
                else "+".join([f"{channels}x1o"] * 4 + [f"{channels}x2e"] * 2)
            ),
            [
                (a, b, int(i >= 4) if merge_paths else i, "uvu", True)
                for i, (a, b) in enumerate(
                    [(0, 1), (1, 0), (1, 2), (2, 1), (1, 1), (2, 0)]
                )
            ],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
        plan = eqx_conv.O2O3TensorProductConv(tp).cuda()
        reference = eqx_conv.O2O3TensorProductConv(tp, backend="torch").cuda()
        frame = o2.WignerD(2, 2).cuda()
        edges = torch.randint(4, (2, edge_count), device="cuda")
        x = torch.randn(4, tp.input_dim, device="cuda", requires_grad=True)
        vectors = torch.randn(edge_count, 3, device="cuda", requires_grad=True)
        radial = torch.randn(edge_count, 4, device="cuda", requires_grad=True)
        projection = torch.randn(4, tp.weight_numel, device="cuda", requires_grad=True)
        amplitude = torch.randn(edge_count, 3, device="cuda", requires_grad=True)
        inputs = x, vectors, radial, projection, amplitude
        arguments = (
            x,
            radial,
            projection,
            frame.forward_packed(vectors),
            amplitude,
            edges,
            4,
        )
        actual, expected = plan(*arguments), reference(*arguments)
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        # Force training requires double backward. Exercise one further
        # derivative with the default CUDA backend.
        for _ in range(3):
            seed = torch.randn_like(actual) / actual.numel() ** 0.5
            derivatives = []
            for value in (actual, expected):
                grads = torch.autograd.grad(
                    (value * seed).sum(), inputs, create_graph=True, retain_graph=True
                )
                derivatives.append(torch.cat([grad.flatten() for grad in grads]))
            actual, expected = derivatives
            torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-9)
    finally:
        torch.set_default_dtype(previous)


def test_shared_path_rotation_schedule():
    from eqx.conv.o2_o3.schedule import rotation_groups

    # Three instructions sum into one output, while the fourth writes an
    # independent entry of the same degree. Do not split or merge those roles.
    paths = [
        ((0, output, 2, 2, 3, 17, 1, 680, 2 * i, i), ())
        for i, output in enumerate((0, 34, 0, 0))
    ]
    _, groups = rotation_groups(paths)
    assert groups == ((0, 2, 3), (1,))


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("contributions,inactive", [(1, False), (2, False), (1, True)])
def test_projected_weight_only_workspace(monkeypatch, shared, contributions, inactive):
    from types import SimpleNamespace

    from eqx.conv.o2_o3.schedule import project

    torch.manual_seed(46)
    rows = 1 if shared else 7
    radial = torch.randn(rows, 3, dtype=torch.float64)
    projection = torch.randn(3, 11, dtype=torch.float64)
    expected = torch.randn(7, 11, dtype=torch.float64)
    if inactive:
        expected[:, -1] = 0
    plan = SimpleNamespace(
        weight_numel=11,
        path_data=(("uvu", (0, 0, 10 if inactive else 11, 11, 1, 1, 0, 0, 0, 0)),),
    )
    gr, gp = torch.zeros_like(radial), torch.zeros_like(projection)
    edges = torch.arange(7)
    dummy = torch.ones(rows, 1, dtype=torch.float64)
    values = (dummy, radial, projection, dummy, dummy, dummy, dummy)
    products = []
    clears = []
    mm = torch.mm
    zero = torch.Tensor.zero_

    def record(a, b, **kwargs):
        products.append(a.shape)
        return mm(a, b, **kwargs)

    def record_zero(value):
        clears.append(value.shape)
        return zero(value)

    monkeypatch.setattr(torch, "mm", record)
    monkeypatch.setattr(torch.Tensor, "zero_", record_zero)

    def contract(plan, source, target, calls, layout, initialize):
        assert bool(initialize) == (not shared and contributions == 1 and not inactive)
        for outputs, operands, results, weighted in calls:
            assert outputs == (1,)
            assert operands[1].untyped_storage().nbytes() == radial.element_size()
            result = expected[source] / contributions
            if initialize:
                assert initialize[0] is results[0]
                results[0].copy_(result)
            else:
                results[0].add_(result.sum(0, keepdim=True) if shared else result)

    project(
        plan,
        edges,
        edges,
        [((1, 2), values, (gr, gp), False)] * contributions,
        contract,
        chunk_size=3,
    )
    assert not products
    assert len(clears) == (1 if shared else 3 if contributions > 1 or inactive else 0)
    reduced = expected.sum(0, keepdim=True) if shared else expected
    torch.testing.assert_close(gr, reduced @ projection.T)
    torch.testing.assert_close(gp, radial.T @ reduced)

    # Shared projections are evaluated once per call, not once per chunk and
    # not cached across parameter updates.
    def forward(plan, source, target, calls, layout, initialize):
        assert not initialize
        weights = calls[0][1][1]
        inputs = radial if shared else radial[source]
        torch.testing.assert_close(weights, mm(inputs, projection))

    for _ in range(2):
        products.clear()
        project(
            plan,
            edges,
            edges,
            [((6,), values, (dummy,), False)],
            forward,
            chunk_size=3,
        )
        assert len(products) == (1 if shared else 3)
        projection.add_(0.25)


@pytest.mark.parametrize("shared", [False, True])
def test_projected_adjoint_reuse(shared):
    from eqx.conv.radial import project

    torch.manual_seed(47)
    edges = torch.arange(7)
    angular = torch.randn(7, 2, dtype=torch.float64)
    expected = torch.randn(7, 11, dtype=torch.float64)
    calls = []
    for _ in range(2):
        radial = torch.randn(1 if shared else 7, 3, dtype=torch.float64)
        projection = torch.randn(3, 11, dtype=torch.float64)
        values = (angular, radial, projection, angular, angular)
        gradients = (torch.zeros_like(radial), torch.zeros_like(projection))
        calls.append(((1, 2), values, gradients, False))

    def contract(source, target, terms, layout, initialize):
        # Different projection transposes share one angular weight adjoint.
        assert len(terms) == 1
        outputs, _, results, _ = terms[0]
        assert outputs == (1,)
        value = expected[source]
        if shared:
            results[0].add_(value.sum(0, keepdim=True))
        else:
            assert initialize == (results[0],)
            results[0].copy_(value)

    project(
        11, edges, edges, calls, contract, complete=True, output_role=4, chunk_size=3
    )
    reduced = expected.sum(0, keepdim=True) if shared else expected
    for _, (_, radial, projection, _, _), (gr, gp), _ in calls:
        torch.testing.assert_close(gr, reduced @ projection.T)
        torch.testing.assert_close(gp, radial.T @ reduced)


def test_derivative_partition_reuses_rotations():
    from eqx.conv.o2_o3.schedule import split_program

    x, w, projection, din, dout, s = (object() for _ in range(6))
    cotangents = (object(), object())
    calls = [
        ((0,), (x, w, projection, din, dout, s, y), (object(),), False)
        for y in cotangents * 2
    ]
    groups = split_program(calls)
    assert sorted(id(call[2][0]) for group in groups for call in group) == sorted(
        id(call[2][0]) for call in calls
    )
    assert [len(group) for group in groups] == [2, 2]
    for group in groups:
        assert group[0][1][6] is group[1][1][6]


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("normalization", ["norm", "component", "integral"])
def test_direction_coefficients_at_pole(degree, normalization):
    from eqx.conv.o2_o3.geometry import angular_coefficients

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        tp = o2.O3TensorProduct(
            "2x3e",
            o3.Irreps([(1, (degree, (-1) ** degree))]),
            o3.Irreps([(2, (3, (-1) ** degree))]),
            [(0, 0, 0, "uvu", True)],
            normalization=normalization,
        )
        conv = eqx_conv.O2O3TensorProductConv(tp)
        pole = torch.tensor([0.0, 1.0, 0.0])

        def harmonics(vectors):
            return o3.spherical_harmonics(
                degree, vectors, normalize=True, normalization=normalization
            )

        def derivative(vectors):
            jacobian = torch.autograd.functional.jacobian(
                harmonics, vectors, create_graph=True
            )
            return torch.linalg.cross(vectors.expand_as(jacobian), jacobian)

        jacobian = torch.autograd.functional.jacobian(derivative, pole)
        values = (
            harmonics(pole),
            derivative(pole),
            torch.linalg.cross(pole.expand_as(jacobian), jacobian),
        )
        cg = o3.wigner_3j(3, degree, 3) * tp.instructions[0].path_weight
        actual = []
        for rank, value in enumerate(values):
            actual.append(angular_coefficients(conv.direction_metadata, rank)[0])
            expected = torch.einsum("aqb,q...->ab...", cg, value)
            torch.testing.assert_close(actual[-1], expected, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(
            actual[2][..., 0, 0] + actual[2][..., 2, 2],
            -degree * (degree + 1) * actual[0],
            atol=2e-12,
            rtol=2e-12,
        )
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize(
    "device,backend,mode,channels",
    [
        ("cpu", "torch", "uvu", 2),
        ("cpu", "torch", "uvw", 2),
        ("cuda", "torch", "uvu", 2),
        ("cuda", "cuda", "uvu", 2),
        ("cuda", "cuda", "uvu", 65),
        ("cuda", "cuda", "uvu", 257),
    ],
)
@pytest.mark.parametrize(
    "projected,shared", [(False, False), (True, False), (True, True)]
)
def test_direction_derivatives(
    monkeypatch, device, backend, mode, channels, projected, shared
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.conv.o2_o3 import cuda

    monkeypatch.setattr(cuda, "CHUNK_SIZE", 3)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(76)
        tp = o2.O3TensorProduct(
            f"{channels}x1o+{channels}x2e",
            "0e+1o+2e",
            "+".join(f"{channels}x{ir}" for ir in ("1o", "2e", "1e", "2o")),
            [
                (0, 0, 0, mode, True),
                (1, 0, 1, mode, True),
                (0, 1, 2, mode, True),
                (1, 1, 3, mode, True),
                (1, 2, 1, "uvu", False),
            ],
            internal_weights=False,
            shared_weights=False,
        ).to(device)
        plan = eqx_conv.O2O3TensorProductConv(tp, backend=backend).to(device)
        reference = eqx_conv.O2O3TensorProductConv(tp, backend="torch").to(device)
        frame = o2.WignerD(2, 2).to(device)
        edges = torch.tensor([[0, 1, 2, 0], [2, 2, 1, 1]], device=device)

        def rand(*shape):
            return torch.randn(*shape, device=device, requires_grad=True)

        rows = 1 if shared else 4
        x, vectors = rand(3, tp.input_dim), rand(rows, 3)
        radial = rand(rows, 3 if projected else tp.weight_numel)
        projection = (
            rand(3, tp.weight_numel)
            if projected
            else radial.new_empty(0, tp.weight_numel)
        )
        amplitude = rand(rows, 3)
        inputs = (
            (x, vectors, radial, amplitude, projection)
            if projected
            else (x, vectors, radial, amplitude)
        )
        packed = frame.forward_packed(vectors)
        args = x, radial, projection, packed, amplitude, edges, 3
        actual = plan(*args, vectors=vectors).sin()
        expected = reference(*args).sin()
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
        for _ in range(3):
            cotangent = torch.randn_like(actual) / actual.numel() ** 0.5
            derivatives = [
                torch.autograd.grad(
                    (value * cotangent).sum(),
                    inputs,
                    create_graph=True,
                    retain_graph=True,
                )
                for value in (actual, expected)
            ]
            actual, expected = [
                torch.cat([g.flatten() for g in grads]) for grads in derivatives
            ]
            torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("projected", [False, True])
def test_direction_zero_order_and_empty_edges(device, projected):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = (
        o2.O3TensorProduct("2x2e", "0e", "2x2e", [(0, 0, 0, "uvu", True)])
        .to(device)
        .double()
    )
    plan = eqx_conv.O2O3TensorProductConv(tp).to(device)
    frame = o2.WignerD(2, 2).to(device).double()
    for size in (0, 4):
        x = torch.randn(3, 10, device=device, dtype=torch.float64, requires_grad=True)
        vectors = torch.randn(
            size, 3, device=device, dtype=torch.float64, requires_grad=True
        )
        weights = torch.randn(
            size,
            3 if projected else 2,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        projection = torch.randn(
            3 if projected else 0,
            2,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        edges = torch.randint(3, (2, size), device=device)
        output = plan(
            x,
            weights,
            projection,
            frame.forward_packed(vectors),
            weights.new_ones(size, 1),
            edges,
            3,
            vectors=vectors,
        )
        gradient = torch.autograd.grad(
            output.square().sum(), vectors, create_graph=True
        )[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
        # A vanishing angular derivative must write zero even when the
        # projected weight workspace has not been initialized.
        second = torch.autograd.grad(gradient.sum(), (vectors, weights, projection))
        for value in second:
            torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)


@pytest.mark.parametrize("channels", [3, 65])
def test_direction_node_reductions(monkeypatch, channels):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.conv.o2_o3 import cuda

    monkeypatch.setattr(cuda, "ROW_SIZE", 8)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(82)
        tp = o2.O3TensorProduct(
            f"{channels}x1o+{channels}x2e",
            "0e+1o+2e",
            f"{channels}x1o+{channels}x2e+{channels}x1e",
            [(0, 0, 0, "uvu", True), (1, 2, 1, "uvu", True), (0, 1, 2, "uvu", True)],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
        module, reference = (
            eqx_conv.O2O3TensorProductConv(tp),
            eqx_conv.O2O3TensorProductConv(tp, backend="torch"),
        )
        frame = o2.WignerD(2, 2).cuda()
        edges = torch.randint(37, (2, 1031), device="cuda")
        # Include isolated nodes, complete rows, and split high-degree rows.
        edges[:, :257] = 0
        edges[:, 257:] = edges[:, 257:] % 29 + 1
        edges[:, -6:] = torch.tensor(
            [[31, 31, 32, 33, 33, 34], [33, 34, 31, 31, 32, 33]], device="cuda"
        )
        x = torch.randn(37, tp.input_dim, device="cuda", requires_grad=True)
        vectors = torch.randn(1031, 3, device="cuda", requires_grad=True)
        radial = torch.randn(1031, 3, device="cuda", requires_grad=True)
        projection = torch.randn(3, tp.weight_numel, device="cuda", requires_grad=True)
        amplitude = torch.randn(1031, 3, device="cuda", requires_grad=True)
        inputs = x, vectors, radial, projection, amplitude
        args = (
            x,
            radial,
            projection,
            frame.forward_packed(vectors),
            amplitude,
            edges,
            37,
        )
        actual = module(*args, vectors=vectors).sin()
        expected = reference(*args).sin()
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        for _ in range(3):
            seed = torch.randn_like(actual) / actual.numel() ** 0.5
            actual, expected = [
                torch.cat(
                    [
                        grad.flatten()
                        for grad in torch.autograd.grad(
                            (value * seed).sum(),
                            inputs,
                            create_graph=True,
                            retain_graph=True,
                        )
                    ]
                )
                for value in (actual, expected)
            ]
            torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-9)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("edge_count", [35, 1031])
def test_direction_compile_and_capture(edge_count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.kernels import wigner_D

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        tp = o2.O3TensorProduct(
            "2x1o", "1o", "2x0e+2x1e", [(0, 0, i, "uvu", True) for i in range(2)]
        ).cuda()
        plan = eqx_conv.O2O3TensorProductConv(tp).cuda()
        reference = eqx_conv.O2O3TensorProductConv(tp, backend="torch").cuda()
        frame = o2.WignerD(1, 1).cuda()
        x = torch.randn(3, 6, device="cuda", requires_grad=True)
        vectors = torch.randn(edge_count, 3, device="cuda", requires_grad=True)
        radial = torch.randn(edge_count, 3, device="cuda", requires_grad=True)
        projection = torch.randn(3, tp.weight_numel, device="cuda", requires_grad=True)
        edges = torch.randint(3, (2, edge_count), device="cuda")

        def evaluate(x, vectors, radial, projection, edges):
            packed = wigner_D(frame, vectors.detach())
            return plan(
                x,
                radial,
                projection,
                packed,
                radial.new_ones(radial.size(0), 1),
                edges,
                3,
                vectors=vectors,
            )

        for _ in range(2):
            torch.autograd.grad(
                evaluate(x, vectors, radial, projection, edges).square().sum(),
                (x, vectors, radial, projection),
            )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = evaluate(x, vectors, radial, projection, edges)
            actual_grads = torch.autograd.grad(
                actual.square().sum(), (x, vectors, radial, projection)
            )
        edges[0].add_(1).remainder_(3)
        edges[1].add_(2).remainder_(3)
        graph.replay()
        expected = evaluate(x, vectors, radial, projection, edges)
        torch.testing.assert_close(actual, expected)
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (x, vectors, radial, projection)
        )
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
        compiled = torch.compile(evaluate, fullgraph=True, dynamic=True)
        for size in (edge_count, edge_count // 2):
            actual = compiled(
                x, vectors[:size], radial[:size], projection, edges[:, :size]
            )
            expected = reference(
                x,
                radial[:size],
                projection,
                frame.forward_packed(vectors[:size]),
                radial.new_ones(size, 1),
                edges[:, :size],
                3,
            )
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
            actual_grads = torch.autograd.grad(
                actual.square().sum(), (x, vectors, radial, projection)
            )
            expected_grads = torch.autograd.grad(
                expected.square().sum(), (x, vectors, radial, projection)
            )
            for a, b in zip(actual_grads, expected_grads):
                torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_direction_derivatives_on_axes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from e3nn import o3

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(42)
        instructions = [(0, 0, i, "uvu", True) for i in range(3)]
        tp = o2.O3TensorProduct(
            "1o",
            "1o",
            "0e+1e+2e",
            instructions,
            internal_weights=False,
            shared_weights=False,
        ).to(device)
        reference = o3.TensorProduct(
            "1o",
            "1o",
            "0e+1e+2e",
            instructions,
            internal_weights=False,
            shared_weights=False,
        ).to(device)
        plan, frame = (
            eqx_conv.O2O3TensorProductConv(tp).to(device),
            o2.WignerD(2, 2).to(device),
        )
        vectors = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1e-8, 1.0, -1e-8],
                [-1e-8, -1.0, 1e-8],
            ],
            device=device,
            requires_grad=True,
        )
        edges = torch.stack(
            (torch.arange(8, device=device), torch.arange(8, device=device))
        )
        x = torch.randn(8, 3, device=device, requires_grad=True)
        weights = torch.randn(8, tp.weight_numel, device=device, requires_grad=True)
        # The unused empty projection remains a valid differentiable operand.
        projection = torch.empty(0, tp.weight_numel, device=device, requires_grad=True)
        actual = plan(
            x,
            weights,
            projection,
            frame.forward_packed(vectors),
            weights.new_ones(8, 1),
            edges,
            8,
            vectors=vectors,
        )
        expected = reference(
            x, o3.spherical_harmonics([1], vectors, True, "component"), weights
        )
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        for _ in range(3):
            cotangent = torch.randn_like(actual) / actual.numel() ** 0.5
            actual = torch.autograd.grad(
                (actual.sin() * cotangent).sum(),
                vectors,
                create_graph=True,
                retain_graph=True,
            )[0]
            expected = torch.autograd.grad(
                (expected.sin() * cotangent).sum(),
                vectors,
                create_graph=True,
                retain_graph=True,
            )[0]
            torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        (gradient,) = torch.autograd.grad(actual.sum(), projection)
        assert gradient.shape == projection.shape
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("owner", [0, 1])
def test_convolution_graph_order(owner):
    from eqx.conv.graph import prepare_graph

    edges = torch.tensor([[0, 0, 1, 0, 3, 0, 2, 0, 0], [2, 1, 2, 0, 2, 2, 1, 2, 2]])
    order = prepare_graph(edges[0], edges[1], owner)
    assert prepare_graph(edges[0], edges[1], owner) is order
    assert prepare_graph(edges[0].detach(), edges[1].detach(), owner) is order
    torch.testing.assert_close(order, edges[owner].argsort(stable=True))
    edges[owner, 0] = 4
    updated = prepare_graph(edges[0].detach(), edges[1].detach(), owner)
    assert updated is not order
    torch.testing.assert_close(updated, edges[owner].argsort(stable=True))
    assert prepare_graph(edges[0, :0], edges[1, :0], owner).shape == (0,)


@pytest.mark.parametrize("degree", [0, 1, 3, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["recursive", "quaternion"])
def test_fused_wigner_derivatives(degree, dtype, method):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.kernels import wigner_D

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        torch.manual_seed(27)
        frame = o2.WignerD(degree, degree).cuda()
        vectors = torch.randn(5, 3, device="cuda", requires_grad=True)
        actual = wigner_D(frame, vectors, method=method)
        expected = frame.forward_packed(vectors, method="recursive")
        tolerance = 2e-5 if dtype == torch.float32 else 2e-12
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(
            wigner_D(frame, vectors.detach(), method=method),
            expected.detach(),
            atol=tolerance,
            rtol=tolerance,
        )
        if degree:
            actual, expected = actual.sin(), expected.sin()
            for _ in range(4 if degree == 1 and dtype == torch.float64 else 3):
                seed = torch.randn_like(actual) / actual.numel() ** 0.5
                actual = torch.autograd.grad(
                    (actual * seed).sum(), vectors, create_graph=True, retain_graph=True
                )[0]
                expected = torch.autograd.grad(
                    (expected * seed).sum(),
                    vectors,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                torch.testing.assert_close(
                    actual, expected, atol=10 * tolerance, rtol=10 * tolerance
                )
        empty = vectors[:0]
        torch.testing.assert_close(
            wigner_D(frame, empty, method=method),
            frame.forward_packed(empty, method="recursive"),
        )
        torch.testing.assert_close(
            wigner_D(frame, empty.detach(), method=method),
            frame.forward_packed(empty, method="recursive").detach(),
        )
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("row_size", [3, 128])
def test_cuda_streams_capture_compile(monkeypatch, row_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.conv.o2_o3 import cuda

    monkeypatch.setattr(cuda, "ROW_SIZE", row_size)
    tp = (
        o2.O3TensorProduct("2x0e", "0e", "2x0e", [(0, 0, 0, "uvu", True)])
        .cuda()
        .double()
    )
    plan = eqx_conv.O2O3TensorProductConv(tp).cuda().double()
    reference = eqx_conv.O2O3TensorProductConv(tp, backend="torch").cuda().double()
    x = torch.randn(4, 2, device="cuda", dtype=torch.float64)
    weights = torch.randn(1031, 2, device="cuda", dtype=torch.float64)
    projection = weights.new_empty(0, 2)
    d = weights.new_ones(1031, 1)
    edges = torch.randint(4, (2, 1031), device="cuda")
    args = (x, weights, projection, d, d, edges, 4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan(*args)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(plan(*args), reference(*args), atol=2e-10, rtol=2e-10)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = plan(*args)
    for shift in (0, 1, 2):
        edges[1].add_(shift).remainder_(4)
        graph.replay()
        torch.testing.assert_close(actual, reference(*args), atol=2e-10, rtol=2e-10)
    x.requires_grad_()
    compiled = torch.compile(plan, fullgraph=True, dynamic=True)
    for size in (1031, 777):
        args = (x, weights[:size], projection, d[:size], d[:size], edges[:, :size], 4)
        actual, expected = compiled(*args), reference(*args)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        (a,) = torch.autograd.grad(actual.square().sum(), x)
        (b,) = torch.autograd.grad(expected.square().sum(), x)
        torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-10)


@pytest.mark.parametrize(
    "device,backend,mode",
    [
        ("cpu", "torch", "uvu"),
        ("cpu", "torch", "uvw"),
        ("cuda", "torch", "uvu"),
        ("cuda", "torch", "uvw"),
        ("cuda", "cuda", "uvu"),
    ],
)
@pytest.mark.parametrize(
    "radial_channels,dtype",
    [
        (0, torch.float64),
        (3, torch.float64),
        (128, torch.float64),
        (129, torch.float32),
        (33, torch.float32),
    ],
)
def test_streaming_derivatives(
    monkeypatch, device, backend, mode, radial_channels, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if backend == "cuda":
        from eqx.conv.o2_o3 import cuda

        monkeypatch.setattr(cuda, "CHUNK_SIZE", 3)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        channels = 17 if radial_channels == 33 else 2
        tp = o2.O3TensorProduct(
            f"{channels}x1o",
            "0e+1o+2e",
            "+".join(f"{channels}x{ir}" for ir in ("0e", "1e", "2e")),
            [(0, 1, i, mode, True) for i in range(3)],
            shared_weights=False,
            internal_weights=False,
        ).to(device)
        tp.convolution = eqx_conv.O2O3TensorProductConv(tp, backend=backend).to(device)
        frame = o2.WignerD(2, 2).to(device)
        generator = torch.Generator(device=device).manual_seed(42)

        def rand(*shape):
            return torch.randn(
                *shape, generator=generator, device=device, requires_grad=True
            )

        edges = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 1]], device=device)
        shared = radial_channels == 129
        size = 1 if shared else 4
        x, r = rand(3, tp.input_dim), rand(size, 3)
        amplitude = rand(size, 3)
        if radial_channels:
            radial = rand(size, radial_channels)
            projection = rand(radial_channels, tp.weight_numel) / radial_channels**0.5
            weights = (radial @ projection).expand(4, -1)
            arguments = dict(radial_features=radial, radial_weight=projection)
            inputs = x, r, radial, projection, amplitude
        else:
            weights = rand(4, tp.weight_numel)
            arguments = dict(weight=weights)
            inputs = x, r, weights, amplitude
        d, di = frame(r.expand(4, -1))
        expected = x.new_zeros(3, tp.irreps_out.dim).index_add(
            0, edges[1], tp(x[edges[0]], d, di, weights, amplitude.expand(4, -1))
        )
        actual = tp.forward_scatter(
            x, edges, frame.forward_packed(r), harmonic_scale=amplitude, **arguments
        )
        tolerance = 3e-5 if dtype == torch.float32 else 1e-11
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        if radial_channels == 33:
            # A nonlinear loss also differentiates the output cotangent.
            actual, expected = actual.sin(), expected.sin()
        for _ in range(3):
            cotangent = torch.randn(actual.shape, generator=generator, device=device)
            if radial_channels == 33:
                cotangent = cotangent / actual.numel() ** 0.5
            actual_grads = torch.autograd.grad(
                (actual * cotangent).sum(), inputs, create_graph=True, retain_graph=True
            )
            expected_grads = torch.autograd.grad(
                (expected * cotangent).sum(),
                inputs,
                create_graph=True,
                retain_graph=True,
            )
            for grad, reference in zip(actual_grads, expected_grads):
                torch.testing.assert_close(
                    grad,
                    reference,
                    atol=5e-4 if dtype == torch.float32 else 3e-9,
                    rtol=5e-5 if dtype == torch.float32 else 3e-9,
                )
            actual = torch.cat([grad.flatten() for grad in actual_grads])
            expected = torch.cat([grad.flatten() for grad in expected_grads])
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize(
    "device,backend",
    [("cpu", "torch"), ("cuda", "torch"), ("cuda", "cuda")],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_streaming_mixed_paths_and_empty_edges(device, backend, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        module = o2.O3TensorProduct(
            "3x0e+35x1o+2x1e",
            "0e+1o+2e",
            f"3x0e+35x0e+{2 if backend == 'cuda' else 19}x1e+35x2e+35x2e",
            [
                (0, 0, 0, "uvu", False),
                (1, 1, 1, "uvu", True),
                (2, 0, 2, "uvu" if backend == "cuda" else "uvw", True),
                (1, 1, 3, "uvu", True),
                (1, 1, 4, "uvu", True),
            ],
            internal_weights=True,
            shared_weights=True,
        ).to(device)
        module.convolution = eqx_conv.O2O3TensorProductConv(module, backend=backend).to(
            device
        )
        frame = o2.WignerD(3, 3).to(device)
        features = torch.randn(3, module.input_dim, device=device, requires_grad=True)
        vectors = torch.randn(1, 3, device=device, requires_grad=True)
        edges = torch.tensor([[0, 1, 2, 1], [1, 2, 0, 0]], device=device)
        d, di = frame(vectors.expand(4, 3))
        reference = features.new_zeros(3, module.irreps_out.dim).index_add(
            0, edges[1], module(features[edges[0]], d, di)
        )
        actual = module.forward_scatter(features, edges, frame.forward_packed(vectors))
        tolerance = 3e-5 if dtype == torch.float32 else 1e-10
        torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
        inputs = features, vectors, module.weight
        a = torch.autograd.grad(actual.square().sum(), inputs, create_graph=True)
        b = torch.autograd.grad(reference.square().sum(), inputs, create_graph=True)
        for value, expected in zip(a, b):
            torch.testing.assert_close(
                value, expected, atol=tolerance * 100, rtol=tolerance * 10
            )

        empty = module.forward_scatter(
            features, edges[:, :0], frame.forward_packed(vectors[:0])
        )
        assert empty.shape == (3, module.irreps_out.dim)
        assert torch.count_nonzero(empty) == 0
        grads = torch.autograd.grad(empty.sum(), (features, module.weight))
        assert all(torch.count_nonzero(grad) == 0 for grad in grads)
        no_nodes = module.forward_scatter(
            features[:0], edges[:, :0], frame.forward_packed(vectors[:0])
        )
        assert no_nodes.shape == (0, module.irreps_out.dim)
        grads = torch.autograd.grad(no_nodes.sum(), (features, module.weight))
        assert all(torch.count_nonzero(grad) == 0 for grad in grads)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize(
    "device,dtype,channels,mode",
    [
        ("cpu", torch.float64, 2, "uvu"),
        ("cpu", torch.float64, 2, "uvw"),
        ("cuda", torch.float64, 2, "uvu"),
        ("cuda", torch.float32, 17, "uvu"),
    ],
)
@pytest.mark.parametrize(
    "projected,shared", [(False, False), (False, True), (True, False), (True, True)]
)
def test_mixed_path_higher_derivatives(
    device, dtype, channels, mode, projected, shared
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with torch.random.fork_rng():
        torch.manual_seed(4)
        tp = o2.O3TensorProduct(
            f"{channels}x0e",
            "0e",
            f"{channels}x0e+{channels}x0e",
            [(0, 0, 0, "uvu", False), (0, 0, 1, mode, True)],
            internal_weights=False,
            shared_weights=False,
        ).to(device=device, dtype=dtype)
        plan = eqx_conv.O2O3TensorProductConv(
            tp, backend="cuda" if device == "cuda" else "torch"
        ).to(device=device, dtype=dtype)
        edges = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], device=device)

        def rand(*shape):
            return torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)

        x = rand(3, channels)
        r = rand(1 if shared else 4, 3 if projected else tp.weight_numel)
        w = rand(3 if projected else 0, tp.weight_numel)
        d, s = rand(4, 1), rand(4, 1)
        weight = r @ w if projected else r
        expected = x.new_zeros(3, 2 * channels).index_add(
            0,
            edges[1],
            tp(x[edges[0]], d[:, :, None], d[:, :, None], weight.expand(4, -1), s),
        )
        actual = plan(x, r, w, d, s, edges, 3)
        inputs = (x, r, w, d, s) if projected else (x, r, d, s)
        torch.testing.assert_close(actual, expected)
        if not projected:
            # An absent projection has no dependence on the remaining factors.
            empty = torch.autograd.grad(
                actual.sum(), w, create_graph=True, retain_graph=True
            )[0]
            assert empty.numel() == 0
            gradients = torch.autograd.grad(
                empty.sum(), inputs, allow_unused=True, retain_graph=True
            )
            assert all(g is None or torch.count_nonzero(g) == 0 for g in gradients)
        # Starting with a weight derivative catches reintroduced unweighted paths.
        actual = torch.autograd.grad(actual.sum(), r, create_graph=True)[0]
        expected = torch.autograd.grad(expected.sum(), r, create_graph=True)[0]
        for _ in range(2):
            a = torch.autograd.grad(
                actual.sum(), inputs, create_graph=True, allow_unused=True
            )
            b = torch.autograd.grad(
                expected.sum(), inputs, create_graph=True, allow_unused=True
            )
            for value, reference, input in zip(a, b, inputs):
                value = torch.zeros_like(input) if value is None else value
                reference = torch.zeros_like(input) if reference is None else reference
                tolerance = 1e-10 if dtype == torch.float64 else 2e-5
                torch.testing.assert_close(
                    value, reference, atol=tolerance, rtol=tolerance
                )
            actual = sum(value.square().sum() for value in a if value is not None)
            expected = sum(value.square().sum() for value in b if value is not None)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "interaction,bias",
    [("cgtp", False), ("cgtp", True), ("o2_cgtp", True), (["cgtp", "o2"], False)],
)
def test_streaming_force_training(monkeypatch, device, interaction, bias):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.lightning import convert_cgtp, load_tace
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    for name in ("TACE_USE_EQX", "TACE_USE_EQT", "TACE_USE_OEQ", "TACE_USE_CUE"):
        monkeypatch.setenv(name, "0")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        config = deepcopy(DEFAULT_MODEL_CONFIG)
        config.update(
            cutoff=4.0,
            max_neighbors=None,
            num_layers=2,
            num_channel=3,
            Lmax=2,
            lmax=2,
            mmax=2,
            parity=True,
            statistics=[
                dict(atomic_numbers=[1], avg_num_neighbors=2.0, atomic_energy={1: 0.0})
            ],
            target_property=["energy", "forces", "stress", "virials"],
        )
        config["node_embedding"]["type"] = "linear"
        config["atomic_basis"]["type"] = interaction
        config["readout_emlp"]["use_one_body_magmoms"] = False
        config["radial_basis"]["hidden"] = [128 if bias else 4]
        config["radial_basis"]["bias"] = bias
        if isinstance(interaction, list):
            config["radial_basis"]["apply_cutoff"] = False
        config["readout_emlp"]["hidden"] = [2]
        config["scale_shift"]["enable"] = False
        reference = TensorModel(e3nnTACE(**deepcopy(config))).train().to(device)
        monkeypatch.setenv("TACE_USE_EQX", "1")
        model = convert_cgtp(reference, "o2")
        representation = model.readout_fn.representation
        mixed = isinstance(interaction, list)
        assert representation.use_packed_wigner == (not mixed)
        assert representation.use_o3_angular_basis == mixed

        def no_edge_message(*args, **kwargs):
            raise AssertionError("The streamed convolution must not call an edge TP")

        for layer in representation.interactions:
            if getattr(layer, "use_eqx", False) and hasattr(layer.rejector, "tp"):
                monkeypatch.setattr(layer.rejector.tp, "forward", no_edge_message)
                monkeypatch.setattr(layer.edge_info, "forward", no_edge_message)

        data = dict(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]], device=device
            ),
            node_attrs=torch.ones(3, 1, device=device),
            edge_index=torch.tensor(
                [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], device=device
            ),
            edge_shifts=torch.zeros(6, 3, device=device),
            lattice=torch.eye(3, device=device).unsqueeze(0) * 8,
            batch=torch.zeros(3, dtype=torch.long, device=device),
            ptr=torch.tensor([0, 3], device=device),
            fidelity_idx=torch.zeros(1, dtype=torch.long, device=device),
        )
        outputs = []
        for network, enabled in ((reference, "0"), (model, "1")):
            monkeypatch.setenv("TACE_USE_EQX", enabled)
            output = network({key: value.clone() for key, value in data.items()})
            outputs.append(output)
            sum(
                output[key].square().sum() for key in ("energy", "forces", "stress")
            ).backward()
        for key in ("energy", "forces", "stress", "virials"):
            torch.testing.assert_close(
                outputs[0][key], outputs[1][key], atol=2e-9, rtol=2e-8
            )
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            expected = reference_parameters[name].grad
            if expected is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(
                    parameter.grad, expected, atol=2e-8, rtol=2e-7
                )
        if device == "cpu" and interaction == "cgtp" and not bias:
            next(reference.parameters()).requires_grad_(False)
            reference.retain_graph = True
            loaded = load_tace(reference, device="cpu")
            assert loaded is reference
            assert not loaded.readout_fn.representation.use_o2
            converted = convert_cgtp(reference)
            assert converted.readout_fn.representation.use_packed_wigner
            for enabled in ("0", "1", "0"):
                monkeypatch.setenv("TACE_USE_EQX", enabled)
                assert converted.readout_fn.representation.use_packed_wigner == (
                    enabled == "1"
                )
                output = converted({key: value.clone() for key, value in data.items()})
                for key in ("energy", "forces", "stress", "virials"):
                    torch.testing.assert_close(
                        output[key], outputs[0][key], atol=2e-9, rtol=2e-8
                    )
            monkeypatch.setenv("TACE_USE_EQX", "0")
            restored = convert_cgtp(converted)
            assert not next(restored.parameters()).requires_grad
            assert restored.retain_graph
            assert not restored.readout_fn.representation.use_packed_wigner
            assert not any(
                getattr(layer, "use_eqx", False)
                for layer in restored.readout_fn.representation.interactions
            )
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(
                    restored.state_dict()[name], value, atol=0, rtol=0
                )
    finally:
        torch.set_default_dtype(previous_dtype)


def test_streaming_finite_differences():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        tp = o2.O3TensorProduct(
            "1o",
            "1o",
            "0e+1e",
            [(0, 0, i, "uvu", True) for i in range(2)],
            internal_weights=False,
            shared_weights=False,
        )
        edges = torch.tensor([[0, 1], [1, 0]])
        frame = o2.WignerD(1, 1)
        inputs = (
            torch.randn(2, 3, requires_grad=True),
            torch.randn(2, 3, requires_grad=True),
            torch.randn(2, 2, requires_grad=True),
            torch.randn(2, tp.weight_numel, requires_grad=True),
        )

        def function(x, r, radial, projection):
            return tp.forward_scatter(
                x,
                edges,
                frame.forward_packed(r),
                radial_features=radial,
                radial_weight=projection,
            )

        assert torch.autograd.gradcheck(function, inputs, fast_mode=True)
        assert torch.autograd.gradgradcheck(function, inputs, fast_mode=True)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("degree", [4, 8])
def test_streaming_high_degree(degree):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        tp = o2.O3TensorProduct(
            f"3x{degree}e",
            "1o",
            f"3x{degree}o",
            [(0, 0, 0, "uvu", True)],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
        tp.convolution = eqx_conv.O2O3TensorProductConv(tp, backend="cuda").cuda()
        frame = o2.WignerD(degree, degree).cuda()
        torch.manual_seed(71)
        x = torch.randn(3, tp.input_dim, device="cuda", requires_grad=True)
        vectors = torch.randn(5, 3, device="cuda", requires_grad=True)
        weight = torch.randn(5, tp.weight_numel, device="cuda", requires_grad=True)
        edges = torch.tensor([[0, 1, 2, 0, 1], [2, 0, 1, 1, 2]], device="cuda")
        d, di = frame(vectors)
        expected = x.new_zeros(3, tp.irreps_out.dim).index_add(
            0, edges[1], tp(x[edges[0]], d, di, weight)
        )
        actual = tp.forward_scatter(x, edges, frame.forward_packed(vectors), weight)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        a = torch.autograd.grad(actual.square().sum(), (x, vectors, weight))
        b = torch.autograd.grad(expected.square().sum(), (x, vectors, weight))
        for value, reference in zip(a, b):
            torch.testing.assert_close(value, reference, atol=2e-9, rtol=2e-9)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("degree,edges_count", [(1, 35), (1, 1025), (5, 1025), (8, 35)])
def test_streaming_projected_tiles(monkeypatch, degree, edges_count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.conv.o2_o3 import cuda as convolution_kernel

    monkeypatch.setattr(convolution_kernel, "CHUNK_SIZE", 1024)
    previous_dtype = torch.get_default_dtype()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.manual_seed(23)
        tp = o2.O3TensorProduct(
            f"17x{degree}e",
            "1o",
            "+".join(f"17x{l}o" for l in (degree - 1, degree, degree + 1)),
            [(0, 0, i, "uvu", True) for i in range(3)],
            internal_weights=False,
            shared_weights=False,
        ).cuda()
        tp.convolution = eqx_conv.O2O3TensorProductConv(tp, backend="cuda").cuda()
        frame = o2.WignerD(degree + 1, degree + 1).cuda()
        x = torch.randn(7, tp.input_dim, device="cuda", requires_grad=True)
        vectors = torch.randn(edges_count, 3, device="cuda", requires_grad=True)
        radial = torch.randn(edges_count, 129, device="cuda", requires_grad=True)
        projection = (
            torch.randn(129, tp.weight_numel, device="cuda") / 129**0.5
        ).requires_grad_()
        # Noncontiguous indices and incomplete tiles exercise both masks.
        edges = torch.randint(7, (2, 2 * edges_count), device="cuda")[:, ::2]
        d, di = frame(vectors)
        packed = frame.forward_packed(vectors)
        expected = x.new_zeros(7, tp.irreps_out.dim).index_add(
            0, edges[1], tp(x[edges[0]], d, di, radial @ projection)
        )
        expected_output = expected
        inputs = x, vectors, radial, projection
        output_cotangent = torch.randn_like(expected) / edges_count**0.5
        cotangent = output_cotangent
        references = []
        for _ in range(3):
            grads = torch.autograd.grad(
                (expected * cotangent).sum(),
                inputs,
                create_graph=True,
                retain_graph=True,
            )
            references.append(grads)
            expected = torch.cat([grad.flatten() for grad in grads])
            cotangent = torch.ones_like(expected) / expected.numel() ** 0.5

        actual = tp.forward_scatter(
            x, edges, packed, radial_features=radial, radial_weight=projection
        )
        torch.testing.assert_close(actual, expected_output, atol=2e-4, rtol=2e-4)
        cotangent = output_cotangent
        for reference in references:
            grads = torch.autograd.grad(
                (actual * cotangent).sum(),
                inputs,
                create_graph=True,
                retain_graph=True,
            )
            for value, expected in zip(grads, reference):
                torch.testing.assert_close(value, expected, atol=2e-4, rtol=2e-4)
            actual = torch.cat([grad.flatten() for grad in grads])
            cotangent = torch.ones_like(actual) / actual.numel() ** 0.5
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_default_dtype(previous_dtype)


def test_replay_tf32_cache():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.kernels.recompute import Replay

    previous = torch.backends.cuda.matmul.allow_tf32
    program = Replay(lambda values, create_graph: (values[0] @ values[1],))
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    try:
        for enabled in (False, True, False):
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.testing.assert_close(program(x, y)[0], x @ y)
        assert len(program.graphs) == 2
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def layout(features, irreps, inverse=False):
    values = []
    for (mul, ir), section in zip(irreps, irreps.slices()):
        shape = (ir.dim, mul) if inverse else (mul, ir.dim)
        values.append(
            features[..., section]
            .reshape(*features.shape[:-1], *shape)
            .transpose(-1, -2)
            .flatten(-2)
        )
    return torch.cat(values, dim=-1)


def tensor_product(channels=3, merge=False, unweighted=False):
    return o3.TensorProduct(
        f"{channels}x0e+{channels}x1o+2x2e",
        "0e+2x1o+1x2e",
        f"{channels}x1o" + ("" if merge else f"+{channels}x1o") + "+2x2e",
        [
            (0, 1, 0, "uvu", True),
            (1, 0, 0 if merge else 1, "uvu", not unweighted),
            (2, 0, 1 if merge else 2, "uvu", True),
        ],
        internal_weights=False,
        shared_weights=False,
    )


def reference(tp, x, attrs, radial, projection, edges):
    weights = radial @ projection if projection.numel() else radial
    message = tp(
        layout(x, tp.irreps_in1, True)[edges[0]],
        layout(attrs, tp.irreps_in2, True),
        weights,
    )
    result = x.new_zeros(x.size(0), tp.irreps_out.dim).index_add(0, edges[1], message)
    return layout(result, tp.irreps_out)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "shared,direct,merge,unweighted",
    [
        (False, False, False, False),
        (True, False, True, False),
        (False, True, True, True),
    ],
)
def test_values_and_recursive_derivatives(
    device, shared, direct, merge, unweighted, double_precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = tensor_product(merge=merge, unweighted=unweighted).to(device)
    conv = eqx_conv.O3TensorProductConv(tp).to(device)
    edges = torch.randint(5, (2, 19), device=device)
    x = torch.randn(5, tp.irreps_in1.dim, device=device, requires_grad=True)
    attrs = torch.randn(
        1 if shared else 19, tp.irreps_in2.dim, device=device, requires_grad=True
    )
    radial = torch.randn(
        1 if shared else 19,
        tp.weight_numel if direct else 4,
        device=device,
        requires_grad=True,
    )
    projection = torch.randn(
        0 if direct else 4, tp.weight_numel, device=device, requires_grad=True
    )
    inputs = (x, attrs, radial) if direct else (x, attrs, radial, projection)
    actual = conv(x, attrs, radial, projection, edges)
    expected = reference(tp, x, attrs, radial, projection, edges)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for _ in range(3):
        seed = torch.randn_like(actual) / actual.numel() ** 0.5
        a = torch.autograd.grad((actual.sin() * seed).sum(), inputs, create_graph=True)
        b = torch.autograd.grad(
            (expected.sin() * seed).sum(), inputs, create_graph=True
        )
        for value, target in zip(a, b):
            torch.testing.assert_close(value, target, atol=2e-10, rtol=2e-10)
        actual, expected = (
            torch.cat([value.flatten() for value in values]) for values in (a, b)
        )


@pytest.mark.parametrize("channels", [1, 32, 65])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_degrees_and_channels(channels, dtype, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.set_default_dtype(dtype)
    tp = o3.TensorProduct(
        f"{channels}x3o",
        "2e",
        f"{channels}x3o+{channels}x5o",
        [(0, 0, 0, "uvu", True), (0, 0, 1, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    conv = eqx_conv.O3TensorProductConv(tp).cuda()
    edges = torch.randint(7, (2, 33), device="cuda")
    # Noncontiguous operands are accepted without changing their derivatives.
    x = torch.randn(tp.irreps_in1.dim, 7, device="cuda").T.requires_grad_()
    attrs = torch.randn(tp.irreps_in2.dim, 33, device="cuda").T.requires_grad_()
    radial = torch.randn(8, 33, device="cuda").T.requires_grad_()
    projection = torch.randn(tp.weight_numel, 8, device="cuda").T.requires_grad_()
    args = x, attrs, radial, projection
    actual, expected = conv(*args, edges), reference(tp, *args, edges)
    tolerance = 2e-4 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    a = torch.autograd.grad(actual.square().mean(), args)
    b = torch.autograd.grad(expected.square().mean(), args)
    for value, target in zip(a, b):
        torch.testing.assert_close(value, target, atol=tolerance, rtol=tolerance)


def test_empty_and_unsupported(double_precision):
    tp = tensor_product()
    conv = eqx_conv.O3TensorProductConv(tp)
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        inputs = [
            torch.randn(shape, device=device, requires_grad=True)
            for shape in (
                (0, tp.irreps_in1.dim),
                (0, tp.irreps_in2.dim),
                (0, 3),
                (3, tp.weight_numel),
            )
        ]
        result = conv.to(device)(
            *inputs, torch.empty(2, 0, dtype=torch.long, device=device)
        )
        assert result.shape == (0, tp.irreps_out.dim)
        for value in torch.autograd.grad(result.sum(), inputs):
            assert not value.count_nonzero()
    with pytest.raises(ValueError, match="backend"):
        eqx_conv.O3TensorProductConv(tp, backend="triton")
    with pytest.raises(NotImplementedError, match="uvu"):
        eqx_conv.O3TensorProductConv(o3.FullyConnectedTensorProduct("0e", "0e", "0e"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tace_radial_bias_cutoff_and_force_training(
    monkeypatch, device, double_precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.fused import O3ScatterTensorProduct
    from tace.models.mlp import MLP

    monkeypatch.setenv("TACE_USE_OEQ", "0")
    monkeypatch.setenv("TACE_USE_CUE", "0")
    monkeypatch.setenv("TACE_USE_EQX", "0")
    module = O3ScatterTensorProduct("3x0e+3x1o", "0e+1o", "3x0e+3x1o+3x2e").to(device)
    mlp = MLP([4, 5, module.weight_numel], bias=True).to(device)
    positions = torch.randn(5, 3, device=device, requires_grad=True)
    x = torch.randn(5, module.irreps_in1.dim, device=device, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 0]], device=device)
    vectors = positions[edges[1]] - positions[edges[0]]
    lengths = vectors.square().sum(-1, keepdim=True)
    radial = torch.cat([lengths**n for n in range(4)], -1)
    cutoff = torch.exp(-lengths)
    attrs = o3.spherical_harmonics(module.irreps_in2, vectors, normalize=True)
    expected = module(x, attrs, mlp(radial) * cutoff, edges)
    hidden = mlp.mlp[:-1](radial)
    last = mlp.mlp[-1]
    hidden = torch.cat((hidden, torch.ones_like(hidden[:, :1])), -1)
    weight = torch.cat((last.get_weight(), last.bias.unsqueeze(0)), 0)
    actual = module.forward_stream(x, attrs, hidden, weight, edges, cutoff)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    monkeypatch.setenv("TACE_USE_EQX", "1")
    torch.testing.assert_close(module(x, attrs, mlp(radial) * cutoff, edges), expected)
    forces = [
        torch.autograd.grad(value.square().sum(), positions, create_graph=True)[0]
        for value in (actual, expected)
    ]
    torch.testing.assert_close(*forces, atol=2e-9, rtol=2e-10)
    parameters = tuple(mlp.parameters()) + (x,)
    gradients = [
        torch.autograd.grad(force.square().sum(), parameters, retain_graph=True)
        for force in forces
    ]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-9)


def test_compile_and_cuda_graph(double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = tensor_product().cuda()
    conv = eqx_conv.O3TensorProductConv(tp).cuda()
    compiled = torch.compile(conv, backend="aot_eager", fullgraph=True, dynamic=True)
    for count in (17, 23, 0):
        args = [
            torch.randn(shape, device="cuda", requires_grad=True)
            for shape in (
                (5, tp.irreps_in1.dim),
                (count, tp.irreps_in2.dim),
                (count, 4),
                (4, tp.weight_numel),
            )
        ]
        edges = torch.randint(5, (2, count), device="cuda")
        actual, expected = compiled(*args, edges), conv(*args, edges)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), args),
            torch.autograd.grad(expected.square().sum(), args),
        ):
            torch.testing.assert_close(a, b)

    args = [
        torch.randn(shape, device="cuda", requires_grad=True)
        for shape in (
            (5, tp.irreps_in1.dim),
            (17, tp.irreps_in2.dim),
            (17, 4),
            (4, tp.weight_numel),
        )
    ]
    edges = torch.randint(5, (2, 17), device="cuda")

    def evaluate():
        output = conv(*args, edges)
        gradient = torch.autograd.grad(output.square().sum(), args, create_graph=True)
        second = torch.autograd.grad(
            sum(value.square().sum() for value in gradient), args
        )
        return output, *gradient, *second

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            evaluate()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = evaluate()
    with torch.no_grad():
        for value in args:
            value.normal_()
        edges.random_(5)
    graph.replay()
    for a, b in zip(actual, evaluate()):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-10)


@pytest.mark.parametrize("channels", [3, 65])
@pytest.mark.parametrize(
    "shared_attrs,shared_radial",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_o3_chunked_shared_gradients(
    monkeypatch, channels, shared_attrs, shared_radial, double_precision
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.conv.o3 import cuda

    # Two grouped chunks and a one-edge tail exercise private and broadcast
    # gradients without mistaking the tail for a shared operand.
    monkeypatch.setattr(cuda, "CHUNK_SIZE", 1025)
    tp = tensor_product(channels, merge=True, unweighted=True).cuda()
    conv = eqx_conv.O3TensorProductConv(tp).cuda()
    edges = torch.randint(257, (2, 2051), device="cuda")
    inputs = [
        torch.randn(shape, device="cuda", requires_grad=True)
        for shape in (
            (257, tp.irreps_in1.dim),
            (1 if shared_attrs else 2051, tp.irreps_in2.dim),
            (1 if shared_radial else 2051, 5),
            (5, tp.weight_numel),
        )
    ]
    actual = conv(*inputs, edges)
    expected = reference(tp, *inputs, edges)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-11)
    for _ in range(2):
        seed = torch.randn_like(actual) / actual.numel() ** 0.5
        gradients = [
            torch.autograd.grad((value.sin() * seed).sum(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-10)
        actual, expected = (
            torch.cat([g.flatten() for g in values]) for values in gradients
        )


def test_o3_weight_adjoint_reuses_angular_contractions(double_precision):
    from eqx.conv.o3.codegen import angular_source

    tp = tensor_product()
    conv = eqx_conv.O3TensorProductConv(tp)
    for path in conv.paths:
        values = angular_source(
            path,
            tuple(range(5)),
            (
                tp.irreps_in1.dim,
                tp.weight_numel,
                0,
                tp.irreps_in2.dim,
                tp.irreps_out.dim,
            ),
            (False,) * 5,
            {0, 1, 3},
            {},
            [],
        )
        assert all(role != 4 for _, role, _ in values)
        assert all((v, 1, 0) in values for v in range(path[4]))


def test_o3_mixed_node_adjoint_destination(double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.conv.o3.convolution import contraction

    tp = o3.TensorProduct(
        "3x0e",
        "0e",
        "3x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    conv = eqx_conv.O3TensorProductConv(tp)
    target = torch.arange(257, device="cuda").repeat_interleave(8)
    source = torch.randint(257, target.shape, device="cuda")
    edges = torch.stack((source, target))
    x = torch.randn(257, 3, device="cuda", requires_grad=True)
    attrs = torch.randn(target.numel(), 1, device="cuda")
    weights = torch.randn(target.numel(), 3, device="cuda")
    projection = x.new_empty(0, 3)
    cotangent = torch.randn_like(x)
    expected = reference(tp, x, attrs, weights, projection, edges)
    expected = expected + torch.autograd.grad((expected * cotangent).sum(), x)[0]
    program = repr((((0, 1, 2, 3, 4), False, ((0, 0), (4, 0))),))
    for _ in range(3):
        actual = contraction(
            conv.kernel_metadata,
            program,
            source,
            target,
            [x, weights, projection, attrs, cotangent],
        )[0]
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("interaction", ["cgtp", "uu_o2", ["uu_o2", "o2"]])
def test_tace_model_force_training(monkeypatch, double_precision, interaction):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
    from tace.models._e3nn.fused import O3ScatterTensorProduct, UuO2ScatterTensorProduct
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    for name in ("EQX", "OEQ", "CUE", "EQT"):
        monkeypatch.setenv(f"TACE_USE_{name}", "0")
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=4.0,
        max_neighbors=None,
        num_layers=2,
        num_channel=3,
        Lmax=2,
        lmax=2,
        mmax=1,
        statistics=[
            dict(atomic_numbers=[1], avg_num_neighbors=2.0, atomic_energy={1: 0.0})
        ],
        target_property=["energy", "forces", "stress", "virials"],
    )
    config["atomic_basis"]["type"] = interaction
    config["atomic_basis"]["use_radial_rotary_attention"] = False
    config["atomic_basis"]["num_head"] = 1
    config["node_embedding"]["type"] = "linear"
    config["radial_basis"]["hidden"] = [4]
    config["radial_basis"]["bias"] = True
    config["radial_basis"]["apply_cutoff"] = False
    config["readout_emlp"]["hidden"] = [3]
    config["readout_emlp"]["use_one_body_magmoms"] = False
    config["scale_shift"]["enable"] = False
    reference_model = TensorModel(e3nnTACE(**config)).cuda().train()
    model = deepcopy(reference_model)
    assert model.state_dict().keys() == reference_model.state_dict().keys()
    data = dict(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]], device="cuda"
        ),
        node_attrs=torch.ones(3, 1, device="cuda"),
        edge_index=torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], device="cuda"
        ),
        edge_shifts=torch.zeros(6, 3, device="cuda"),
        lattice=torch.eye(3, device="cuda").unsqueeze(0) * 8,
        batch=torch.zeros(3, dtype=torch.long, device="cuda"),
        ptr=torch.tensor([0, 3], device="cuda"),
        fidelity_idx=torch.zeros(1, dtype=torch.long, device="cuda"),
    )

    def no_edge_message(*args):
        raise AssertionError(
            "The fused interaction must not materialize edge messages or weights"
        )

    for layer in model.readout_fn.representation.interactions:
        if hasattr(layer.rejector, "tp"):
            monkeypatch.setattr(layer.rejector.tp, "forward", no_edge_message)
        elif isinstance(layer.rejector, UuO2ScatterTensorProduct):
            monkeypatch.setattr(layer.rejector.linear, "forward", no_edge_message)
            monkeypatch.setattr(
                layer.rejector.local_frame_in, "to_local", no_edge_message
            )
            monkeypatch.setattr(
                layer.rejector.local_frame_out, "to_global", no_edge_message
            )
        if isinstance(
            layer.rejector, (O3ScatterTensorProduct, UuO2ScatterTensorProduct)
        ):
            monkeypatch.setattr(layer.edge_info, "forward", no_edge_message)
    results = []
    for network, enabled in ((reference_model, "0"), (model, "1")):
        monkeypatch.setenv("TACE_USE_EQX", enabled)
        output = network({key: value.clone() for key, value in data.items()})
        results.append(output)
        sum(
            output[key].square().sum() for key in ("energy", "forces", "stress")
        ).backward()
    for key in ("energy", "forces", "stress", "virials"):
        torch.testing.assert_close(
            results[0][key], results[1][key], atol=2e-9, rtol=2e-8
        )
    reference_parameters = dict(reference_model.named_parameters())
    for name, parameter in model.named_parameters():
        expected = reference_parameters[name].grad
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, atol=2e-8, rtol=2e-7)
