"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import conv as eqx_conv
from equivariantx import o2
from equivariantx.tests.conv.testing import assert_native_autograd
from equivariantx.tests.conv.testing import layout


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("normalization", ["component", "integral", "norm"])
@pytest.mark.parametrize(
    "method", ["baseline", "generator", "recurrence", "cg", "wigner"]
)
def test_transverse_convolution(
    backend, normalization, method, double_precision, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    a, b, c = "2x3o+2x2e", "1o+2e+3o", "2x3e+2x2e+2x3e+2x4o"
    instructions = [
        (0, 0, 0, "uvu", True),
        (1, 1, 1, "uvu", True),
        (0, 2, 2, "uvu", True),
        (1, 2, 3, "uvu", False),
    ]
    options = dict(internal_weights=False, shared_weights=False)
    tp = o2.O3TensorProduct(
        a, b, c, instructions, normalization=normalization, **options
    ).cuda()
    module = eqx_conv.O2O3TensorProductConv(tp, backend=backend, method=method).cuda()
    calls = []
    if backend == "torch":
        for coupling in module.transverse_couplings.values():
            coupling.register_forward_hook(lambda *_: calls.append("generator"))
        for harmonic in module.harmonics.values():
            harmonic.register_forward_hook(lambda *_: calls.append("cg"))
        forward_packed = module.frame.forward_packed

        def wigner(*args, **kwargs):
            assert kwargs["method"] == "recursive"
            calls.append("wigner")
            return forward_packed(*args, **kwargs)

        monkeypatch.setattr(module.frame, "forward_packed", wigner)
    reference = o3.TensorProduct(a, b, c, instructions, **options).cuda()
    x = torch.randn(4, tp.input_dim, device="cuda", requires_grad=True)
    vectors = (
        torch.cat((torch.eye(3), -torch.eye(3), torch.randn(2, 3)))
        .cuda()
        .requires_grad_()
    )
    edges = torch.randint(4, (2, 8), device="cuda")
    radial = torch.randn(8, 3, device="cuda", requires_grad=True)
    projection = torch.randn(3, tp.weight_numel, device="cuda", requires_grad=True)
    amplitudes = torch.randn(1, 3, device="cuda", requires_grad=True)
    inputs = x, vectors, radial, projection, amplitudes
    actual = module(x, radial, projection, None, amplitudes, edges, 4, vectors=vectors)
    if backend == "torch":
        assert_native_autograd(actual)
        assert set(calls) == {
            "generator" if method in ("baseline", "recurrence") else method
        }
    harmonics = torch.cat(
        [
            o3.spherical_harmonics(
                ir.l, vectors, normalize=True, normalization=normalization
            )
            * amplitudes[:, i : i + 1]
            for i, (_, ir) in enumerate(tp.irreps_in2)
        ],
        -1,
    )
    message = layout(
        reference(
            layout(x, tp.irreps_in1, inverse=True)[edges[0]],
            harmonics,
            radial @ projection,
        ),
        tp.irreps_out,
    )
    expected = torch.zeros_like(actual).index_add(0, edges[1], message)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    actual, expected = actual.sin(), expected.sin()
    for _ in range(3):
        probe = torch.randn_like(actual) / actual.numel() ** 0.5
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        value, inputs, probe, create_graph=True, retain_graph=True
                    )
                ]
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        if backend == "torch":
            assert_native_autograd(actual)


@pytest.mark.parametrize("method", ["generator", "recurrence", "cg", "wigner"])
@pytest.mark.parametrize("shared", [False, True])
def test_o2_torch_channel_mixing(method, shared, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    args = (
        "2x2e",
        "1o",
        "3x1o+4x2o+3x1o",
        [(0, 0, 0, "uvw", True), (0, 0, 1, "uvw", True), (0, 0, 2, "uvw", True)],
    )
    options = dict(internal_weights=False, shared_weights=False)
    tp = o2.O3TensorProduct(*args, **options).cuda()
    module = eqx_conv.O2O3TensorProductConv(tp, backend="torch", method=method).cuda()
    expected_tp = o3.TensorProduct(*args, **options).cuda()
    index = torch.tensor([[0, 1, 0, 2], [2, 0, 1, 1]], device="cuda")
    rows = 1 if shared else index.size(1)
    inputs = [
        torch.randn(shape, device="cuda", requires_grad=True)
        for shape in ((3, tp.input_dim), (rows, 3), (rows, tp.weight_numel), (rows, 1))
    ]
    x, vectors, weights, amplitudes = inputs
    projection = x.new_empty(0, tp.weight_numel)
    actual = module(x, weights, projection, None, amplitudes, index, 3, vectors=vectors)
    attrs = o3.spherical_harmonics("1o", vectors, True, "component") * amplitudes
    message = layout(
        expected_tp(layout(x, tp.irreps_in1, inverse=True)[index[0]], attrs, weights),
        tp.irreps_out,
    )
    expected = torch.zeros_like(actual).index_add(0, index[1], message)
    for order in range(4):
        assert_native_autograd(actual)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        if order < 3:
            actual, expected = [
                torch.cat(
                    [
                        g.flatten()
                        for g in torch.autograd.grad(
                            value.sin().sum(), inputs, create_graph=True
                        )
                    ]
                )
                / 20
                for value in (actual, expected)
            ]


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("degrees", [(5, 4, 5), (7, 6, 6)])
@pytest.mark.parametrize(
    "method", ["baseline", "generator", "recurrence", "cg", "wigner"]
)
def test_o2_cgtp_high_degree(backend, degrees, method, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    l1, l2, l3 = degrees
    a = o3.Irreps([(2, (l1, (-1) ** l1))])
    b = o3.Irreps([(1, (l2, (-1) ** l2))])
    c = o3.Irreps([(2, (l3, (-1) ** (l1 + l2)))])
    instructions = [(0, 0, 0, "uvu", True)]
    options = dict(internal_weights=False, shared_weights=False)
    tp = o2.O3TensorProduct(a, b, c, instructions, **options).cuda()
    module = eqx_conv.O2O3TensorProductConv(tp, backend=backend, method=method).cuda()
    reference = o3.TensorProduct(a, b, c, instructions, **options).cuda()
    x = torch.randn(4, a.dim, device="cuda", requires_grad=True)
    vectors = torch.randn(8, 3, device="cuda", requires_grad=True)
    weights = torch.randn(8, tp.weight_numel, device="cuda", requires_grad=True)
    index = torch.randint(4, (2, 8), device="cuda")
    projection = torch.empty(0, tp.weight_numel, device="cuda")
    amplitudes = torch.ones(8, 1, device="cuda")
    actual = module(x, weights, projection, None, amplitudes, index, 4, vectors=vectors)
    harmonics = o3.spherical_harmonics(b, vectors, True, "component")
    message = layout(
        reference(layout(x, a, inverse=True)[index[0]], harmonics, weights), c
    )
    expected = torch.zeros_like(actual).index_add(0, index[1], message)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    gradients = [
        torch.autograd.grad(y.square().sum(), (x, vectors, weights))
        for y in (actual, expected)
    ]
    for value, target in zip(*gradients):
        torch.testing.assert_close(value, target, atol=2e-8, rtol=2e-9)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize(
    "method", ["baseline", "generator", "recurrence", "cg", "wigner", "auto"]
)
def test_o2_cgtp_empty_methods(backend, method, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = o2.O3TensorProduct(
        "2x1o",
        "1o",
        "2x2e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    module = eqx_conv.O2O3TensorProductConv(tp, backend=backend, method=method).cuda()
    x = torch.randn(2, tp.input_dim, device="cuda", requires_grad=True)
    weights = torch.empty(0, tp.weight_numel, device="cuda", requires_grad=True)
    projection = torch.empty(0, tp.weight_numel, device="cuda")
    vectors = torch.empty(0, 3, device="cuda", requires_grad=True)
    amplitudes = torch.empty(0, 1, device="cuda", requires_grad=True)
    edges = torch.empty(2, 0, device="cuda", dtype=torch.long)
    actual = module(x, weights, projection, None, amplitudes, edges, 2, vectors=vectors)
    torch.testing.assert_close(actual, torch.zeros_like(actual))
    if backend == "torch":
        assert_native_autograd(actual)
    for value in torch.autograd.grad(actual.sum(), (x, weights, vectors, amplitudes)):
        torch.testing.assert_close(value, torch.zeros_like(value))
    assert not module.tuning_results


@pytest.mark.parametrize(
    "degrees",
    [(0, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2), (5, 1, 6), (12, 2, 14), (12, 5, 13)],
)
def test_cg_generator_constraints(degrees, double_precision):
    from equivariantx.o3.rotation import so3_generators

    ell1, ell2, ell3 = degrees
    cg = o3.wigner_3j(ell1, ell2, ell3)
    for g1, g2, g3 in zip(*(so3_generators(ell) for ell in degrees)):
        left = torch.einsum("dk,abk->abd", g3, cg)
        right = torch.einsum("qbd,qa->abd", cg, g1)
        right += torch.einsum("aqd,qb->abd", cg, g2)
        torch.testing.assert_close(left, right, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("degree", [1, 5, 10, 16])
@pytest.mark.parametrize("normalization", ["component", "integral", "norm"])
def test_generator_polynomials(degree, normalization, double_precision):
    from equivariantx.ops.kernels.cuda.angular import generator_scale
    from equivariantx.o3.rotation import so3_generators

    torch.manual_seed(91)
    vectors = torch.randn(19, 3)
    vectors[:3] = torch.eye(3)
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    matrix = torch.einsum("ba,aij->bij", vectors, so3_generators(degree))
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
    from equivariantx.o3.rotation import so3_generators
    from equivariantx.utils.geometry import angular_coefficients

    tp = o2.O3TensorProduct(f"{degree}e", "2e", f"{degree}e", [(0, 0, 0, "uvu", True)])
    conv = eqx_conv.O2O3TensorProductConv(tp, backend="torch")
    previous = angular_coefficients(conv.direction_metadata, 0)[0]
    for rank in range(1, 5):
        expected = torch.zeros(*previous.shape, 3)
        for axis, ell in enumerate((degree, degree, *([1] * (rank - 1)))):
            term = torch.tensordot(previous, -so3_generators(ell), dims=([axis], [1]))
            expected += term.movedim(-1, axis)
        expected[..., 1] = 0
        actual = angular_coefficients(conv.direction_metadata, rank)[0]
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
        previous = actual


@pytest.mark.parametrize("degree", [4, 12])
@pytest.mark.parametrize("implementation", ["o3", "o2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_high_degree_cuda_paths(degree, implementation, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        torch.manual_seed(89)
        args = (
            f"3x{degree}e",
            "1o+2e",
            f"3x{degree}o+3x{degree}e+3x{degree}e+3x{degree + 1}o+3x{degree + 2}e",
            [
                (0, 0, 0, "uvu", True),
                (0, 1, 1, "uvu", True),
                (0, 1, 2, "uvu", False),
                (0, 0, 3, "uvu", True),
                (0, 1, 4, "uvu", True),
            ],
        )
        options = dict(internal_weights=False, shared_weights=False)
        tp = o3.TensorProduct(*args, **options)
        reference = eqx_conv.O3TensorProductConv(tp, backend="torch").cuda()
        # Autotuning is tested separately; keep path selection deterministic here.
        module = (
            eqx_conv.O3TensorProductConv(tp, backend="cuda")
            if implementation == "o3"
            else eqx_conv.O2O3TensorProductConv(
                o2.O3TensorProduct(*args, **options), method="baseline", backend="cuda"
            )
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
            actual = module(
                x,
                weights,
                projection,
                None,
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
    module = eqx_conv.O2O3TensorProductConv(tp, backend="cuda")
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
    import equivariantx.ops.kernels.cuda.o2_o3.convolution as cuda

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
    module = eqx_conv.O2O3TensorProductConv(tp, backend="cuda")
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


def test_convolution_backends_and_modes():
    from equivariantx.ops.wigner import wigner_D

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
    vectors = torch.randn(2, 3)
    torch.testing.assert_close(
        o2.WignerD(1, 1, method="quaternion")(vectors),
        o2.WignerD(1, 1, method="recursive")(vectors),
    )


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
        plan = eqx_conv.O2O3TensorProductConv(tp, backend="cuda").cuda()
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
    from equivariantx.ops.kernels.cuda.o2_o3.schedule import rotation_groups

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

    from equivariantx.ops.kernels.cuda.o2_o3.schedule import project

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
    from equivariantx.ops.kernels.cuda.radial import project

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
    from equivariantx.ops.kernels.cuda.o2_o3.schedule import split_program

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


@pytest.mark.parametrize("row_size", [3, 128])
def test_cuda_streams_capture_compile(monkeypatch, row_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import equivariantx.ops.kernels.cuda.o2_o3.convolution as cuda

    monkeypatch.setattr(cuda, "ROW_SIZE", row_size)
    tp = (
        o2.O3TensorProduct("2x0e", "0e", "2x0e", [(0, 0, 0, "uvu", True)])
        .cuda()
        .double()
    )
    plan = eqx_conv.O2O3TensorProductConv(tp, backend="cuda").cuda().double()
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
        (1024, torch.float64),
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
        import equivariantx.ops.kernels.cuda.o2_o3.convolution as cuda

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
            # Native autograd can return a constant zero without a grad_fn.
            if empty.requires_grad:
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
    import equivariantx.ops.kernels.cuda.o2_o3.convolution as convolution_kernel

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


@pytest.mark.parametrize("directions", [False, True])
def test_o2_requested_gradients(monkeypatch, directions, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.kernels.cuda.runtime import runtime

    tp = o2.O3TensorProduct(
        "3x1o", "1o", "3x0e+3x1e+3x2e", [(0, 0, i, "uvu", True) for i in range(3)]
    ).cuda()
    module = eqx_conv.O2O3TensorProductConv(
        tp, method="baseline", backend="cuda"
    ).cuda()
    reference = eqx_conv.O2O3TensorProductConv(tp, backend="torch").cuda()
    frame = o2.WignerD(2, 2).cuda()
    edges = torch.randint(4, (2, 13), device="cuda")
    inputs = tuple(
        torch.randn(shape, device="cuda", requires_grad=True)
        for shape in ((4, 9), (13, 3), (13, 3), (3, tp.weight_numel))
    )
    x, vectors, radial, projection = inputs
    args = (
        x,
        radial,
        projection,
        frame.forward_packed(vectors),
        x.new_ones(13, 1),
        edges,
        4,
    )
    masks = []
    query = runtime().gradient_mask

    def record(offset, count):
        mask = query(offset, count)
        masks.append(mask)
        return mask

    monkeypatch.setattr(runtime(), "gradient_mask", record)
    actual = module(*args, vectors=vectors if directions else None)
    expected = reference(*args)
    actual = torch.autograd.grad(actual.sin().sum(), x, create_graph=True)[0]
    count = len(masks)
    expected = torch.autograd.grad(expected.sin().sum(), x, create_graph=True)[0]
    # The native PyTorch backend must not load or call the CUDA runtime.
    assert len(masks) == count
    assert masks and all(sum(mask) == 1 for mask in masks)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    actual = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected = torch.autograd.grad(expected.square().sum(), inputs)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, atol=2e-9, rtol=2e-9)
