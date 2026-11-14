"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx import co3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_preserves_cuda_provider(device, monkeypatch, double_precision):
    from unittest.mock import patch

    from equivariantx.co3.ictp.tensor_product import CartesianProduct
    from equivariantx.ops import config
    from equivariantx.ops.cuda.co3.ictp import tensor_product as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module, reference = (
        CartesianProduct(1, 1, 2, backend="cuda").to(device),
        CartesianProduct(1, 1, 2).to(device),
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (3, 0):
        values = (
            torch.randn(count, 3, device=device),
            torch.randn(count, 3, device=device),
        )
        with patch.object(kernel, "angular", wraps=kernel.angular) as selected:
            actual = module(*values)
        assert selected.call_count == (device == "cuda")
        torch.testing.assert_close(actual, reference(*values))
    assert module.backend == "cuda" and config.TORCH_REFERENCE is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("grouped", [False, True])
def test_reference_bypasses_children(device, grouped, monkeypatch, double_precision):
    import copy

    from equivariantx.co3.ictp.tensor_product import CartesianProduct
    from equivariantx.ops import config
    from equivariantx.ops.torch.co3.ictp import tensor_product as kernel
    from equivariantx.tests.conv.testing import assert_native_autograd

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = (
        co3.ictp.TensorProduct("2x1o", "2x1o", "2x2e", [(0, 0, 0, "uvu", True)])
        if grouped
        else CartesianProduct(3, 3, 6)
    ).to(device)
    reference = copy.deepcopy(module)

    def forbidden(*args, **kwargs):
        raise AssertionError("The ordinary product dispatched a child backend.")

    for child in module.modules():
        if child is not module and hasattr(child, "backend"):
            child.backend = "cuda"
            if child is getattr(module, "projection", None) and not child.groups:
                continue
            for method in ("forward", "pack", "unpack"):
                if hasattr(child, method):
                    monkeypatch.setattr(child, method, forbidden)
    state = [
        (name, child.backend)
        for name, child in module.named_modules()
        if hasattr(child, "backend")
    ]
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    matrix = co3.path_matrix(3).to(device)
    for count in (2, 0):
        values = tuple(
            (
                torch.randn(count, 6 if grouped else 7, device=device) * 0.1
            ).requires_grad_()
            for _ in range(2)
        )

        def run(layer, x, y):
            return layer(x, y) if grouped else layer(x @ matrix.T, y @ matrix.T)

        actual, expected = run(module, *values), run(reference, *values)
        for _ in range(4):
            torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
            assert_native_autograd(actual)
            actual, expected = [
                sum(
                    g.sin().sum()
                    for g in torch.autograd.grad(
                        value.sin().sum(), values, create_graph=True
                    )
                )
                for value in (actual, expected)
            ]
        tangents = tuple(torch.randn_like(value) for value in values)
        torch.testing.assert_close(
            torch.func.jvp(lambda *xs: run(module, *xs), values, tangents),
            torch.func.jvp(lambda *xs: run(reference, *xs), values, tangents),
        )
    assert state == [
        (name, child.backend)
        for name, child in module.named_modules()
        if hasattr(child, "backend")
    ]
    assert config.TORCH_REFERENCE is True
    if not grouped:
        module.backend = "cuda"
        monkeypatch.setattr(config, "TORCH_REFERENCE", False)
        values = tuple(
            torch.randn(2, 7, device=device, requires_grad=True) @ matrix.T
            for _ in range(2)
        )
        actual = kernel.reference(module, *values)
        torch.testing.assert_close(actual, kernel.reference(reference, *values))
        assert_native_autograd(actual)


@pytest.mark.parametrize(
    "mode,multiplicities,normalization,path_normalization",
    [
        ("uvv", (2, 3, 3), "component", "element"),
        ("uuu", (2, 2, 2), "component", "element"),
        ("uuw", (2, 2, 3), "component", "element"),
        ("uvw", (2, 3, 4), "component", "element"),
        ("uvuv", (2, 3, 6), "component", "element"),
    ]
    + [
        ("uvu", (2, 3, 2), normalization, path_normalization)
        for normalization in ("component", "norm", "none")
        for path_normalization in ("element", "path", "none")
    ],
)
@pytest.mark.parametrize(
    "tp_type", [co3.ictd.TensorProduct, co3.ictp.TensorProduct], ids=["ictd", "ictp"]
)
def test_tensor_product(
    double_precision, mode, multiplicities, normalization, path_normalization, tp_type
):
    assert co3.TensorProduct is co3.ictd.TensorProduct
    u, v, w = multiplicities
    a, b, c = f"{u}x1o+{u}x2e", f"{v}x1o", f"{w}x1e+{w}x2e+{w}x1o+{w}x3o"
    instructions = [
        (0, 0, 0, mode, True),
        (0, 0, 1, mode, True),
        (1, 0, 2, mode, True, 0.7),
        (1, 0, 3, mode, True),
        (1, 0, 2, mode, True, 1.3),
    ]
    kwargs = dict(
        internal_weights=False,
        shared_weights=False,
        irrep_normalization=normalization,
        path_normalization=path_normalization,
        in1_var=[0.8, 1.3],
        in2_var=[1.2],
        out_var=[0.5, 1.0, 0.7, 0.9],
    )
    ref = o3.TensorProduct(a, b, c, instructions, **kwargs).to(DEVICE)
    module = tp_type(a, b, c, instructions, **kwargs).to(DEVICE)
    x = torch.randn(2, ref.irreps_in1.dim, device=DEVICE, requires_grad=True)
    y = torch.randn(2, ref.irreps_in2.dim, device=DEVICE, requires_grad=True)
    weights = torch.randn(2, ref.weight_numel, device=DEVICE, requires_grad=True)
    transforms = [
        co3.ChangeOfBasis(ir, inverse=inverse).to(DEVICE)
        for ir, inverse in ((a, False), (b, False), (c, True))
    ]
    actual = transforms[2](module(transforms[0](x), transforms[1](y), weights))
    expected = ref(x, y, weights)
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    for _ in range(2):
        seed = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        (val * seed).sum(),
                        (x, y, weights),
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for val in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    assert module(transforms[0](x[:0]), transforms[1](y[:0]), weights[:0]).shape == (
        0,
        module.irreps_out.dim,
    )


@pytest.mark.parametrize(
    "tp_type", [co3.ictd.TensorProduct, co3.ictp.TensorProduct], ids=["ictd", "ictp"]
)
def test_high_degree_and_compile(double_precision, tp_type):
    assert co3.path_normalization(0, 2000, 2000) == pytest.approx((4001) ** 0.5)
    kwargs = dict(internal_weights=False, shared_weights=False)
    module = tp_type("1x4e", "1x3o", "1x4o", [(0, 0, 0, "uuu", True)], **kwargs).to(
        DEVICE
    )
    ref = o3.TensorProduct(
        "1x4e", "1x3o", "1x4o", [(0, 0, 0, "uuu", True)], **kwargs
    ).to(DEVICE)
    x = torch.randn(2, 9, device=DEVICE)
    y = torch.randn(2, 7, device=DEVICE)
    w = torch.randn(2, 1, device=DEVICE)
    cx = co3.ChangeOfBasis("1x4e").to(DEVICE)(x)
    cy = co3.ChangeOfBasis("1x3o").to(DEVICE)(y)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    actual = co3.ChangeOfBasis("1x4o", True).to(DEVICE)(compiled(cx, cy, w))
    torch.testing.assert_close(actual, ref(x, y, w), atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize(
    "tp_type", [co3.ictd.TensorProduct, co3.ictp.TensorProduct], ids=["ictd", "ictp"]
)
def test_all_angular_paths(double_precision, tp_type):
    for l1 in range(5):
        for l2 in range(5):
            a, b = f"1x{l1}e", f"1x{l2}o"
            x = torch.eye(2 * l1 + 1, device=DEVICE).unsqueeze(1)
            y = torch.eye(2 * l2 + 1, device=DEVICE).unsqueeze(0)
            cx = co3.ChangeOfBasis(a).to(DEVICE)(x)
            cy = co3.ChangeOfBasis(b).to(DEVICE)(y)
            for l3 in range(abs(l1 - l2), min(l1 + l2, 6) + 1):
                out = f"1x{l3}o"
                tp = tp_type(a, b, out, [(0, 0, 0, "uuu", False)]).to(DEVICE)
                actual = co3.ChangeOfBasis(out, True).to(DEVICE)(tp(cx, cy))
                cg = o3.wigner_3j(l1, l2, l3, dtype=torch.float64, device=DEVICE)
                expected = (
                    torch.einsum("...i,...j,ijk->...k", x, y, cg) * (2 * l3 + 1) ** 0.5
                )
                torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize(
    "tp_type", [co3.ictd.TensorProduct, co3.ictp.TensorProduct], ids=["ictd", "ictp"]
)
def test_broadcast_weights(double_precision, tp_type):
    kwargs = dict(internal_weights=False, shared_weights=False)
    a, b, c = "2x1e", "2x2o", "2x1o+2x2o"
    ins = [(0, 0, 0, "uuu", True), (0, 0, 1, "uuu", True)]
    tp = tp_type(a, b, c, ins, **kwargs).to(DEVICE)
    ref = o3.TensorProduct(a, b, c, ins, **kwargs).to(DEVICE)
    x = torch.randn(o3.Irreps(a).dim, 3, device=DEVICE).T
    y = torch.randn(1, o3.Irreps(b).dim, device=DEVICE)
    w = torch.randn(1, tp.weight_numel, device=DEVICE)
    result = tp(
        co3.ChangeOfBasis(a).to(DEVICE)(x), co3.ChangeOfBasis(b).to(DEVICE)(y), w
    )
    torch.testing.assert_close(
        co3.ChangeOfBasis(c, True).to(DEVICE)(result), ref(x, y, w)
    )


@pytest.mark.parametrize(
    "degrees", [(6, 3, 7), (7, 2, 8), (5, 4, 5), (6, 6, 0), (6, 5, 10)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ictp_ictd_equivalence(degrees, dtype):
    l1, l2, l3 = degrees
    args = f"2x{l1}o", f"2x{l2}e", f"2x{l3}o", [(0, 0, 0, "uuu", True)]
    reference = co3.ictd.TensorProduct(*args).to(device=DEVICE, dtype=dtype)
    module = co3.ictp.TensorProduct(*args).to(device=DEVICE, dtype=dtype)
    module.load_state_dict(reference.state_dict())
    x = torch.randn(3, 2 * (2 * l1 + 1), device=DEVICE, dtype=dtype, requires_grad=True)
    y = torch.randn(3, 2 * (2 * l2 + 1), device=DEVICE, dtype=dtype, requires_grad=True)
    cx, cy = [
        co3.ChangeOfBasis(ir).to(device=DEVICE, dtype=dtype)(value)
        for ir, value in ((args[0], x), (args[1], y))
    ]
    actual, expected = module(cx, cy), reference(cx, cy)
    tol = 3e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    losses = [value.square().sum() for value in (actual, expected)]
    for _ in range(2):
        gradients = [
            torch.autograd.grad(loss, (x, y), create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=10 * tol, rtol=10 * tol)
        tangents = [torch.randn_like(t) for t in (x, y)]
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]


def test_tensor_product_projection(double_precision):
    a, b = "2x0e+2x1o+2x2e+2x4e", "2x2e+2x1o"
    c = "2x2e+2x3o+2x2e+2x1o+2x4e+2x0e+2x2o"
    instructions = [
        (0, 0, 0, "uuu", True),
        (2, 0, 0, "uuu", True),
        (3, 1, 1, "uuu", True),
        (2, 0, 2, "uuu", True),
        (0, 1, 3, "uuu", True),
        (3, 0, 4, "uuu", True),
        (2, 0, 5, "uuu", True),
        (2, 1, 6, "uuu", True),
    ]
    module = co3.TensorProduct(a, b, c, instructions).to(DEVICE)
    assert module.projection_indices == (0, 2, 4, 6)
    assert module.projection.groups == ((2, ((0, 2), (2, 2), (6, 2))), (4, ((4, 2),)))
    reference = o3.TensorProduct(a, b, c, instructions).to(DEVICE)
    x = torch.randn(2, reference.irreps_in1.dim, device=DEVICE, requires_grad=True)
    y = torch.randn(2, reference.irreps_in2.dim, device=DEVICE, requires_grad=True)
    transforms = [
        co3.ChangeOfBasis(ir, inverse=inverse).to(DEVICE)
        for ir, inverse in ((a, False), (b, False), (c, True))
    ]
    expected = reference(x, y, module.weight)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    for evaluate in (module, compiled):
        cx, cy = transforms[0](x), transforms[1](y)
        actual = transforms[2](evaluate(cx, cy))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual_grad = torch.autograd.grad(actual.square().sum(), (x, y, module.weight))
        expected_grad = torch.autograd.grad(
            expected.square().sum(), (x, y, module.weight), retain_graph=True
        )
        for actual, expected_value in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected_value, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(
    "tp_type", [co3.ictd.TensorProduct, co3.ictp.TensorProduct], ids=["ictd", "ictp"]
)
def test_tensor_product_grouping(double_precision, weighted, tp_type):
    a = "2x1o+2x2e+2x1e+2x1o"
    b = "2x1o+2x1e"
    c = "2x2e+2x2o+2x2e+2x0e"
    instructions = [
        (0, 0, 0, "uuu", weighted),
        (1, 0, 1, "uuu", weighted),
        (2, 1, 0, "uuu", weighted, 0.4),
        (3, 0, 2, "uuu", weighted, 1.7),
        (0, 1, 1, "uuu", weighted),
    ]
    options = dict(internal_weights=False, shared_weights=False)
    module = tp_type(a, b, c, instructions, **options).to(DEVICE)
    reference = o3.TensorProduct(a, b, c, instructions, **options).to(DEVICE)
    assert len(module.groups) == 2
    assert module.weight_numel == reference.weight_numel
    x = torch.randn(2, 1, reference.irreps_in1.dim, device=DEVICE, requires_grad=True)
    y = torch.randn(1, 3, reference.irreps_in2.dim, device=DEVICE, requires_grad=True)
    weights = torch.randn(2, 3, module.weight_numel, device=DEVICE, requires_grad=True)
    forward = [co3.ChangeOfBasis(ir).to(DEVICE) for ir in (a, b)]
    inverse = co3.ChangeOfBasis(c, inverse=True).to(DEVICE)
    actual = inverse(module(forward[0](x), forward[1](y), weights))
    expected = reference(x, y, weights)
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    losses = [value.square().sum() for value in (actual, expected)]
    inputs = (x, y, weights) if weighted else (x, y)
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for left, right in zip(*gradients):
            torch.testing.assert_close(left, right, atol=2e-10, rtol=2e-10)
        tangents = [torch.randn_like(t) for t in inputs]
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]
    assert module(forward[0](x[:0]), forward[1](y), weights[:0]).shape == (
        0,
        3,
        module.irreps_out.dim,
    )


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("degrees", [(1, 1, 2), (2, 2, 2), (3, 2, 3), (3, 3, 4)])
def test_sparse_cartesian_product(backend, degrees, double_precision, monkeypatch):
    from equivariantx.co3.ictp.tensor_product import CartesianProduct
    from equivariantx.ops import config
    from equivariantx.ops.torch.co3.ictp.tensor_product import reference

    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    layer = CartesianProduct(*degrees, backend=backend).double().to(device)
    x = (torch.randn(2, 1, 3 ** degrees[0], device=device) * 0.1).requires_grad_()
    y = (torch.randn(1, 3, 3 ** degrees[1], device=device) * 0.1).requires_grad_()
    actual, expected = layer(x, y), reference(layer, x, y)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    for _ in range(4):
        probe = torch.randn_like(actual) / actual.numel() ** 0.5
        actual, expected = [
            torch.cat(
                [
                    v.flatten()
                    for v in torch.autograd.grad(
                        (value.sin() * probe).sum(), (x, y), create_graph=True
                    )
                ]
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(
        torch.func.jvp(layer, (x, y), (x, y)),
        torch.func.jvp(lambda a, b: reference(layer, a, b), (x, y), (x, y)),
    )
    layer.float().double()
    torch.testing.assert_close(
        layer(x, y), reference(layer, x, y), atol=3e-12, rtol=3e-12
    )
    assert layer(x[:0], y).shape == (0, 3, 3 ** degrees[2])
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    torch.testing.assert_close(
        layer(x, y), reference(layer, x, y), atol=3e-12, rtol=3e-12
    )
