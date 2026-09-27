"""Integration of EQX operators with TACE models."""

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
from tace.models.layout import LayoutTransform
from tace.models.linear import (
    e3nnElementLinear,
    e3nnLinear,
    e3nnMoEElementLinear,
    enable_lora,
)
from tace.utils.env import EQX_KERNELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


@pytest.mark.parametrize("method", ["baseline", "generator", "cg", "wigner", "auto"])
def test_o2_cgtp_stream_methods(method, monkeypatch, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.fused import O2ScatterTensorProduct, O3ScatterTensorProduct

    for backend in ("EQX", "OEQ", "EQT", "CUEQ"):
        monkeypatch.setenv(f"TACE_USE_{backend}", "0")
    irreps_in, irreps_sh, irreps_out = "2x1o", "0e+1o", "2x0e+2x1o+2x1e+2x2e"
    module = O2ScatterTensorProduct(irreps_in, irreps_sh, irreps_out).cuda()
    module.eqx_tp.method = method
    reference = O3ScatterTensorProduct(irreps_in, irreps_sh, irreps_out).cuda()
    x = torch.randn(3, 6, device="cuda", requires_grad=True)
    vectors = torch.randn(8, 3, device="cuda", requires_grad=True)
    radial = torch.randn(8, 4, device="cuda", requires_grad=True)
    projection = torch.randn(4, module.weight_numel, device="cuda", requires_grad=True)
    cutoff = torch.rand(8, 1, device="cuda", requires_grad=True)
    edges = torch.randint(3, (2, 8), device="cuda")
    graph = types.SimpleNamespace(
        edge_vector=vectors, edge_length=vectors.norm(dim=-1, keepdim=True) + 1e-9
    )
    actual = module.forward_stream(x, radial, projection, edges, None, cutoff, graph)
    harmonics = o3.spherical_harmonics(
        irreps_sh, vectors / graph.edge_length, False, "component"
    )
    expected = reference(x, harmonics, (radial @ projection) * cutoff, edges)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    inputs = x, vectors, radial, projection, cutoff
    gradients = [
        torch.autograd.grad(y.square().sum(), inputs, create_graph=True)
        for y in (actual, expected)
    ]
    for value, target in zip(*gradients):
        torch.testing.assert_close(value, target, atol=2e-9, rtol=2e-9)
    gradients = [
        torch.autograd.grad(g[1].square().sum(), inputs, retain_graph=True)
        for g in gradients
    ]
    for value, target in zip(*gradients):
        torch.testing.assert_close(value, target, atol=2e-7, rtol=2e-8)


@pytest.mark.parametrize("wigner_lmax", [2, 4])
@pytest.mark.parametrize("basis_change", [True, False])
def test_local_frame_roundtrip_flattened_ir_mul(wigner_lmax, basis_change):
    irreps = o3.Irreps("2x0e+1x0o+3x1e+2x1o+1x2e+3x2o")
    frame = o2.LocalFrame(irreps, basis_change=basis_change).to(DEVICE, DTYPE)
    layout = LayoutTransform(
        irreps,
        layout_in="flatten_mul_ir",
        layout_out="flatten_ir_mul",
    ).to(DEVICE)
    vectors = torch.randn(7, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(wigner_lmax, wigner_lmax).to(DEVICE, DTYPE)(vectors)
    features = torch.randn(7, irreps.dim, dtype=DTYPE, device=DEVICE)

    local = frame(layout(features), wigner)
    assert frame.irreps_out == o2.Irreps("5x0e+7x0o+9x1m+4x2m")
    options = "mmax=2" + ("" if basis_change else ", basis_change=False")
    assert repr(frame) == (
        f"LocalFrame({frame.global_irreps} -> {frame.local_irreps})({options})"
    )
    reverse_frame = o2.LocalFrame(irreps, reverse=True, basis_change=basis_change)
    assert repr(reverse_frame) == (
        f"LocalFrame({frame.local_irreps} -> {frame.global_irreps})({options})"
    )
    if basis_change:
        default = o2.LocalFrame(irreps).to(DEVICE, DTYPE)
        torch.testing.assert_close(
            default(layout(features), wigner), local, atol=0, rtol=0
        )
    assert local.shape == (7, frame.irreps_out.dim)
    torch.testing.assert_close(
        layout.inverse(frame.to_global(local, wigner_inv)),
        features,
    )


@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize("normalization", ["component", "integral", "norm"])
@pytest.mark.parametrize("batch_size", [0, 5])
def test_o3_tensor_product_matches_edge_cgtp(
    double_precision, mode, normalization, batch_size
):
    from tace.models._e3nn.paths import generate_paths

    irreps_in = o3.Irreps("2x0e+2x1o+3x1e+3x2o+2x3e")
    irreps_sh = o3.Irreps.spherical_harmonics(4)
    irreps_out = o3.Irreps("2x0e+2x0o+2x1e+2x1o+2x2e+2x2o+2x3e+2x3o")
    instructions, irreps_out = generate_paths(
        irreps_out,
        irreps_in,
        irreps_sh,
        e3nn_mode=mode,
        trainable=True,
    )
    kwargs = dict(internal_weights=False, shared_weights=False)
    reference = o3.TensorProduct(
        irreps_in, irreps_sh, irreps_out, instructions, **kwargs
    )
    module = o2.O3TensorProduct(
        irreps_in,
        irreps_sh,
        irreps_out,
        instructions,
        normalization=normalization,
        **kwargs,
    )
    assert not module.local_frame_in.basis_change
    assert not module.local_frame_out.basis_change
    x = torch.randn(batch_size, irreps_in.dim, requires_grad=True)
    r = torch.randn(batch_size, 3, requires_grad=True)
    w = torch.randn(batch_size, module.weight_numel, requires_grad=True)
    layout_in = LayoutTransform(
        irreps_in,
        layout_in="flatten_mul_ir",
        layout_out="flatten_ir_mul",
    )
    layout_out = LayoutTransform(
        irreps_out,
        layout_in="flatten_ir_mul",
        layout_out="flatten_mul_ir",
    )
    d, di = o2.WignerD(3, 3)(r)
    actual = layout_out(module(layout_in(x), d, di, w))
    expected = reference(
        x, o3.spherical_harmonics(irreps_sh, r, True, normalization), w
    )
    assert module.weight_numel == reference.weight_numel
    assert not any(isinstance(child, o3.TensorProduct) for child in module.modules())
    torch.testing.assert_close(module.output_mask, reference.output_mask)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    if batch_size:
        for actual_grad, expected_grad in zip(
            torch.autograd.grad(actual.square().sum(), (x, r, w)),
            torch.autograd.grad(expected.square().sum(), (x, r, w)),
        ):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-8, rtol=2e-9)


@pytest.mark.parametrize("irrep_normalization", ["component", "norm", "none"])
@pytest.mark.parametrize("path_normalization", ["element", "path", "none"])
def test_o3_tensor_product_normalization_and_second_derivatives(
    double_precision,
    irrep_normalization,
    path_normalization,
):
    irreps_in = o3.Irreps("2x1o")
    irreps_sh = o3.Irreps.spherical_harmonics(2)
    irreps_out = o3.Irreps("2x1o+2x1e+2x2e+2x0o")
    instructions = [
        (0, 0, 0, "uvu", True, 0.7),
        (0, 2, 0, "uvu", True, 1.3),
        (0, 1, 1, "uvu", False, 0.9),
        (0, 1, 2, "uvu", True, 1.1),
    ]
    kwargs = dict(
        irrep_normalization=irrep_normalization,
        path_normalization=path_normalization,
        in1_var=[0.8],
        in2_var=[1.2, 0.9, 1.5],
        out_var=[0.7, 1.1, 1.3, 1.0],
        internal_weights=False,
        shared_weights=False,
    )
    reference = o3.TensorProduct(
        irreps_in, irreps_sh, irreps_out, instructions, **kwargs
    )
    module = o2.O3TensorProduct(
        irreps_in, irreps_sh, irreps_out, instructions, **kwargs
    )
    x = torch.randn(4, irreps_in.dim, requires_grad=True)
    r = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(1, module.weight_numel, requires_grad=True)
    layout_in = LayoutTransform(
        irreps_in, layout_in="flatten_mul_ir", layout_out="flatten_ir_mul"
    )
    layout_out = LayoutTransform(
        irreps_out, layout_in="flatten_ir_mul", layout_out="flatten_mul_ir"
    )
    d, di = o2.WignerD(2, 2)(r)
    # A nonunit input vector tests the optional per-degree amplitude and its gradient.
    scale = r.square().sum(-1, keepdim=True).sqrt().pow(torch.arange(3))
    actual = layout_out(module(layout_in(x), d, di, weight, scale))
    expected = reference(
        x, o3.spherical_harmonics(irreps_sh, r, False, "component"), weight
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    grads = [
        torch.autograd.grad(value.square().sum(), (x, r, weight), create_graph=True)
        for value in (actual, expected)
    ]
    for actual_grad, expected_grad in zip(*grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-8, rtol=2e-9)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(grads[0][1].square().sum(), (x, r, weight)),
        torch.autograd.grad(grads[1][1].square().sum(), (x, r, weight)),
    ):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-8)


def test_o3_tensor_product_internal_weights_axes_and_truncated_frames(double_precision):
    module = o2.O3TensorProduct(
        "2x1o",
        "1x1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
    )
    reference = o3.TensorProduct(
        module.irreps_in1,
        module.irreps_in2,
        module.irreps_out,
        [(0, 0, i, "uvu", True) for i in range(3)],
    )
    r = torch.cat((torch.eye(3), -torch.eye(3)))
    x = torch.randn(6, 3, 6)
    layout_in = LayoutTransform(
        module.irreps_in1, layout_in="flatten_mul_ir", layout_out="flatten_ir_mul"
    )
    layout_out = LayoutTransform(
        module.irreps_out, layout_in="flatten_ir_mul", layout_out="flatten_mul_ir"
    )
    d, di = o2.WignerD(2, 2)(r)
    actual = layout_out(module(layout_in(x), d, di))
    expected = reference(
        x, o3.spherical_harmonics([1], r, True, "component")[:, None], module.weight
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    d, di = o2.WignerD(1, 2)(r)
    with pytest.raises(ValueError, match="orders"):
        module(layout_in(x), d, di)
    with pytest.raises(ValueError, match="spherical harmonics"):
        o2.O3TensorProduct("1o", "1e", "0o", [(0, 0, 0, "uvu", True)])


@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed O(3) irreps do not expose time-reversal parity.",
)
def test_o3_tensor_product_time_reversal(double_precision):
    from tace.models.time_reversal import with_time_reversal

    irreps_in = with_time_reversal(o3.Irreps("2x1e"), -1)
    irreps_out = with_time_reversal(o3.Irreps("2x0o+2x1o+2x2o"), -1)
    module = o2.O3TensorProduct(
        irreps_in,
        o3.Irreps.spherical_harmonics(1)[1:],
        irreps_out,
        [(0, 0, i, "uvu", True) for i in range(3)],
    )
    x = torch.randn(4, irreps_in.dim)
    d, di = o2.WignerD(2, 2)(torch.randn(4, 3))
    torch.testing.assert_close(module(-x, d, di), -module(x, d, di))


@pytest.mark.parametrize("matrix", ["0", "1"])
@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("num_nodes", [0, 7])
def test_shared_linear(matrix, internal, num_nodes, double_precision):
    import copy

    actual = e3nnLinear(
        "2x0e+0e+3x1o+2x5o",
        "3x0e+0e+2x1o+5o",
        bias=True,
        internal_weights=internal,
        use_matrix_weight=matrix,
    )
    reference = copy.deepcopy(actual)
    reference.linear = o3.Linear(
        actual.irreps_in,
        actual.irreps_out,
        internal_weights=False,
        shared_weights=False,
    )
    reference.load_state_dict(actual.state_dict(), strict=True)
    x = torch.randn(num_nodes, actual.irreps_in.dim, requires_grad=True)
    weight = (
        None
        if internal
        else torch.randn(num_nodes, actual.weight_numel, requires_grad=True)
    )
    values = [module(x, weight) for module in (actual, reference)]
    torch.testing.assert_close(*values, atol=1e-12, rtol=1e-12)
    inputs = [
        (x, *module.parameters()) if internal else (x, weight, *module.parameters())
        for module in (actual, reference)
    ]
    for _ in range(3):
        derivatives = [
            torch.autograd.grad(y.sin().sum(), args, create_graph=True)
            for y, args in zip(values, inputs)
        ]
        for a, b in zip(*derivatives):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        values = [torch.cat([g.flatten() for g in grads]) for grads in derivatives]


@pytest.mark.parametrize("experts", [1, 2])
@pytest.mark.parametrize("matrix", ["0", "1"])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tace_indexed_linear(monkeypatch, experts, matrix, lora, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.set_default_dtype(dtype)
    cls = e3nnElementLinear if experts == 1 else e3nnMoEElementLinear
    module = cls(
        "4x0e+2x0e+4x1o+2x1e",
        "6x0e+4x1o+2x1o+2x2o",
        num_elements=3,
        bias=True,
        use_matrix_weight=matrix,
        **({} if experts == 1 else {"num_experts": experts}),
    ).cuda()
    if lora:
        enable_lora(module, r=2, freeze_base=False)
        with torch.no_grad():
            for parameter in module.lora_B:
                parameter.normal_()
    with torch.no_grad():
        module.bias.normal_()
    attrs = torch.rand(5, 3, device="cuda", requires_grad=True)
    x = torch.randn(5, module.irreps_in.dim, device="cuda", requires_grad=True)
    # A supplied element index takes precedence over attrs.argmax for weights.
    types = torch.tensor([2, 0, 1, 2, 1], device="cuda")
    monkeypatch.setenv("TACE_USE_EQX", "1")
    outputs, derivatives = [], []
    inputs = (x, attrs, *module.parameters())
    for enabled in (False, True):
        monkeypatch.setitem(EQX_KERNELS, "linear", enabled)
        output = module(x, attrs, types)
        first = torch.autograd.grad(output.square().mean(), x, create_graph=True)[0]
        derivatives.append(
            torch.autograd.grad(
                output.square().mean() + first.square().mean(),
                inputs,
                allow_unused=True,
            )
        )
        outputs.append(output)
    tol = 3e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(*outputs, atol=tol, rtol=tol)
    for actual, expected in zip(*derivatives):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    if dtype == torch.float64 and matrix == "1" and lora:
        compiled = torch.compile(
            module, backend="aot_eager", fullgraph=True, dynamic=True
        )
        output = compiled(x, attrs, types)
        torch.testing.assert_close(output, outputs[1], atol=tol, rtol=tol)
        torch.testing.assert_close(
            torch.autograd.grad(output.square().mean(), x)[0],
            torch.autograd.grad(module(x, attrs, types).square().mean(), x)[0],
            atol=tol,
            rtol=tol,
        )


@pytest.mark.parametrize("magnetic", [False, True])
@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("mmax", [0, 2])
@pytest.mark.parametrize("packed", [False, True])
def test_uv_o2_cuda_convolution(monkeypatch, magnetic, attention, mmax, packed):
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
            w, wi = (
                (frame.forward_packed(vectors), None)
                if enabled and packed
                else frame(vectors)
            )
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


@pytest.mark.parametrize("packed", [False, True])
def test_uv_o2_compile_and_force_training(monkeypatch, double_precision, packed):
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
        w, wi = (frame.forward_packed(vectors), None) if packed else frame(vectors)
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
@pytest.mark.parametrize("edges", [0, 7, 65, 257, 1025])
def test_tece_streaming_derivatives(monkeypatch, device, edges):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from eqx.models.tace.tece_oam_rra.interaction import stream
    from tace.models._e3nn.tece_oam_rra import Convolution
    from tace.models.layout import LayoutTransform

    if edges == 1025:
        from functools import partial

        from eqx.models.tace.tece_oam_rra import execution

        monkeypatch.setattr(
            execution, "forward", partial(execution.forward, tile_size=256)
        )
        monkeypatch.setattr(
            execution, "launch", partial(execution.launch, tile_size=256)
        )

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
    if edges >= 257:
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
    if device == "cuda" and edges == 7:
        compiled = torch.compile(
            lambda *args: stream(module, *args), backend="aot_eager", fullgraph=True
        )
        output = compiled(
            x, radial, projection, bias, index, cutoff, rotation, inverse, basis
        )
        torch.testing.assert_close(output, expected, atol=1e-11, rtol=1e-10)
        gradients = [
            torch.autograd.grad(
                value.sum(), inputs, allow_unused=True, retain_graph=True
            )
            for value in (output, expected)
        ]
        for a, b in zip(*gradients):
            if a is not None and b is not None:
                torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)
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
        assert not representation.use_packed_wigner
        assert representation.use_local_frame == mixed
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
            assert not converted.readout_fn.representation.use_packed_wigner
            for enabled in ("0", "1", "0"):
                monkeypatch.setenv("TACE_USE_EQX", enabled)
                assert not converted.readout_fn.representation.use_packed_wigner
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


@pytest.mark.parametrize("interaction", ["cgtp", "uu_o2", "o2", ["uu_o2", "o2"]])
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
    monkeypatch.setenv("TACE_USE_EQX", "1")
    model = TensorModel(e3nnTACE(**config)).cuda().train()
    model.load_state_dict(reference_model.state_dict(), strict=True)
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

    if interaction == "uu_o2":
        monkeypatch.setattr(
            model.readout_fn.representation.o2_angular_basis,
            "forward",
            no_edge_message,
        )
        monkeypatch.setattr(
            model.readout_fn.representation.o2_angular_basis,
            "forward_packed",
            no_edge_message,
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
