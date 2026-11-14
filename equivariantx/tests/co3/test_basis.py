"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx import co2, co3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from e3nn import nn


def test_reference_preserves_cuda_basis_provider(monkeypatch, double_precision):
    from unittest.mock import patch

    from equivariantx.ops import config
    from equivariantx.ops.cuda.co3 import basis as kernel

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = co3.ChangeOfBasis("2x2e", backend="cuda").cuda()
    reference = co3.ChangeOfBasis("2x2e").cuda()
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (3, 0):
        value = torch.randn(count, 10, device="cuda")
        with patch.object(
            kernel, "change_basis", wraps=kernel.change_basis
        ) as selected:
            actual = module(value)
        assert selected.call_count == 1
        torch.testing.assert_close(actual, reference(value))
    assert module.backend == "cuda" and config.TORCH_REFERENCE is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("switch", [False, True])
def test_projector_reference_bypasses_children(
    device, switch, monkeypatch, double_precision
):
    import copy

    from equivariantx.ops import config
    from equivariantx.ops.torch.co3 import basis as kernel
    from equivariantx.tests.conv.testing import assert_native_autograd

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = co3.Projector("1x6e").to(device)
    reference = copy.deepcopy(module)
    for child in module.symmetric_bases.values():
        child.backend = "cuda"

        def forbidden(*args, **kwargs):
            raise AssertionError("An ordinary projector dispatched its child backend.")

        monkeypatch.setattr(child, "pack", forbidden)
        monkeypatch.setattr(child, "unpack", forbidden)
    monkeypatch.setattr(config, "TORCH_REFERENCE", switch)
    if not switch:
        module.backend = "cuda"

    def run(layer, value):
        return (
            layer(value)
            if switch
            else kernel.project_reference(layer, [value.unflatten(-1, (1, 729))])[
                0
            ].flatten(-2)
        )

    for count in (2, 0):
        value = (torch.randn(count, 729, device=device) * 0.1).requires_grad_()
        actual, expected = run(module, value), run(reference, value)
        for _ in range(4):
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
            assert_native_autograd(actual)
            actual, expected = [
                torch.autograd.grad(output.sin().sum(), value, create_graph=True)[0]
                for output in (actual, expected)
            ]
        tangent = torch.randn_like(value)
        torch.testing.assert_close(
            torch.func.jvp(lambda x: run(module, x), (value,), (tangent,)),
            torch.func.jvp(lambda x: run(reference, x), (value,), (tangent,)),
        )
    assert all(child.backend == "cuda" for child in module.symmetric_bases.values())
    assert config.TORCH_REFERENCE == switch


@pytest.mark.parametrize("cartesian", [False, True])
def test_conversion_and_deferred_projection(double_precision, cartesian):
    a, b, c = "2x2e", "1x2e", "2x2e+2x3e"
    ins = [
        (0, 0, 0, "uvu", True, 0.4),
        (0, 0, 0, "uvu", True, 1.7),
        (0, 0, 1, "uvu", True, 0.8),
    ]
    cls = co3.TensorProduct if cartesian else o3.TensorProduct
    ref = cls(
        a, b, c, ins, path_normalization="path", in1_var=[0.7], out_var=[1.3, 0.8]
    ).to(DEVICE)
    ref.eval()
    ref.weight.requires_grad_(False)
    # Conversion must not lose precision through the ambient default dtype.
    torch.set_default_dtype(torch.float32)
    try:
        module = co2.O3TensorProduct.from_tensor_product(
            ref, input_basis="spherical", project=False
        )
    finally:
        torch.set_default_dtype(torch.float64)
    assert not module.training and not module.weight.requires_grad
    torch.testing.assert_close(module.weight, ref.weight)
    x = torch.randn(4, o3.Irreps(a).dim, device=DEVICE)
    r = torch.randn(4, 3, device=DEVICE)
    if cartesian:
        cx = co3.ChangeOfBasis(a).to(DEVICE)(x)
        y = co3.CartesianHarmonics(co3.Irreps(b), True, "component").to(DEVICE)(r)
        expected = co3.ChangeOfBasis(c, True).to(DEVICE)(ref(cx, y))
    else:
        expected = ref(x, o3.spherical_harmonics(o3.Irreps(b), r, True, "component"))
    raw = module(x, r)
    actual = co3.ChangeOfBasis(c, True).to(DEVICE)(raw)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    down = co3.Linear(c, "1x2e+1x3e", output_basis="spherical").to(DEVICE)
    torch.testing.assert_close(
        down(raw), down(co3.Projector(c).to(DEVICE)(raw)), atol=2e-12, rtol=2e-12
    )
    for sign in (-1, 1):
        q = sign * o3.rand_matrix(device=DEVICE)
        rotated_x = x @ o3.Irreps(a).D_from_matrix(q.cpu()).to(DEVICE).T
        rotated = co3.ChangeOfBasis(c, True).to(DEVICE)(module(rotated_x, r @ q.T))
        transformed = actual @ o3.Irreps(c).D_from_matrix(q.cpu()).to(DEVICE).T
        torch.testing.assert_close(rotated, transformed, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("rank", [6, 10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_high_rank_projection(rank, dtype):
    module = co3.Projector(f"2x{rank}e").to(device=DEVICE, dtype=dtype)
    features = torch.randn(
        3, module.irreps_in.dim, device=DEVICE, dtype=dtype, requires_grad=True
    )
    matrix = co3.path_matrix(rank).to(DEVICE)
    expected = (features.double().unflatten(-1, (2, 3**rank)) @ matrix) @ matrix.T
    expected = expected.flatten(-2)
    actual = module(features)
    tol = 3e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual.double(), expected, atol=tol, rtol=tol)
    torch.testing.assert_close(module(actual), actual, atol=tol, rtol=tol)
    assert module(features[:0]).shape == (0, module.irreps_out.dim)
    losses = [value.square().sum() for value in (actual, expected)]
    for _ in range(2):
        gradients = [
            torch.autograd.grad(loss, features, create_graph=True)[0] for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=tol, rtol=tol)
        tangent = torch.randn_like(features)
        losses = [(gradient * tangent).sum() for gradient in gradients]


def test_time_reversal_labels(double_precision):
    try:
        o3.Irrep("1eo")
    except ValueError:
        pytest.skip("Time-reversal representations are not installed")
    a, b, c = "2x1eo", "1x1oe", "2x0oo+2x1oo+2x2oo"
    ins = [(0, 0, i, "uvu", True) for i in range(3)]
    for tp_type in (co3.ictd.TensorProduct, co3.ictp.TensorProduct):
        tp = tp_type(a, b, c, ins).to(DEVICE)
        x, y = (
            tp.irreps_in1.randn(3, -1, device=DEVICE),
            tp.irreps_in2.randn(3, -1, device=DEVICE),
        )
        torch.testing.assert_close(tp(-x, y), -tp(x, y))
    gate = co3.Gate("1x0eo", [torch.tanh], "1x0ee", [torch.sigmoid], "1x1eo").to(DEVICE)
    ref = nn.Gate("1x0eo", [torch.tanh], "1x0ee", [torch.sigmoid], "1x1eo").to(DEVICE)
    h = torch.randn(3, ref.irreps_in.dim, device=DEVICE)
    actual = co3.ChangeOfBasis(gate.irreps_out, True).to(DEVICE)(
        gate(co3.ChangeOfBasis(gate.irreps_in).to(DEVICE)(h))
    )
    torch.testing.assert_close(actual, ref(h))
    irreps = co3.Irreps("2x0eo+1x1ee+2x1eo+1x2oo")
    assert co3.Irrep(1, 1, -1) == co3.Irrep("1eo")
    x = irreps.randn(3, -1, device=DEVICE)
    change = co3.ChangeOfBasis(irreps, inverse=True).to(DEVICE)
    signs = x.new_tensor([ir.t for mul, ir in irreps for _ in range(mul * ir.dim)])
    for matrix in (torch.eye(3, device=DEVICE), -o3.rand_matrix(device=DEVICE)):
        spatial = irreps.D_from_matrix(matrix)
        combined = irreps.D_from_matrix(matrix, time_reversal=True)
        torch.testing.assert_close(x @ combined.T, (x * signs) @ spatial.T)
        spherical = irreps.spherical().D_from_matrix(matrix.cpu()).to(DEVICE)
        torch.testing.assert_close(
            change(x @ combined.T), change(x * signs) @ spherical.T
        )
    linear = co3.Linear(irreps, irreps, biases=True).to(DEVICE)
    torch.testing.assert_close(linear(x * signs), linear(x) * signs)
    assert linear.bias_numel == 0
    angles = o3.rand_angles(device=DEVICE)
    ir = co3.Irrep("2oo")
    kt = torch.tensor(1, device=DEVICE)
    torch.testing.assert_close(
        ir.D_from_angles(*angles, kt=kt), -ir.D_from_angles(*angles)
    )
    quaternion = o3.angles_to_quaternion(*angles)
    torch.testing.assert_close(
        ir.D_from_quaternion(quaternion, kt=kt), -ir.D_from_quaternion(quaternion)
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_method_precision(device, double_precision):
    from equivariantx.tests.utils.testing import check_method_precision

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layer = co3.Projector("2x2e").to(device)
    check_method_precision(
        layer,
        [
            ("forward_blocks", ([torch.randn(3, 2, 9, device=device)],), {}),
        ],
    )
