"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import co2, co3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from e3nn import nn


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


def test_irreps_and_basis(double_precision):
    irreps = co3.Irreps("2x0e+1x1o+2x2e+1x2o+1x1o")
    assert irreps.dim == 2 + 3 + 27 + 3
    assert irreps.regroup() == co3.Irreps("2x0e+2x1o+1x2o+2x2e")
    assert co3.Irreps([co3.Irrep("1o"), co3.Irrep("2e")]).dim == 12
    assert irreps.filter(lmax=1).dim == 8
    with pytest.raises(ValueError):
        irreps.index("0o")
    to_cart = co3.ChangeOfBasis(irreps).to(DEVICE)
    to_sph = co3.ChangeOfBasis(irreps, inverse=True).to(DEVICE)
    x = torch.randn(4, irreps.spherical().dim, device=DEVICE)
    torch.testing.assert_close(to_sph(to_cart(x)), x, atol=2e-14, rtol=2e-14)
    for sign in (1, -1):
        matrix = sign * o3.rand_matrix(device=DEVICE)
        expected = to_cart(
            x @ irreps.spherical().D_from_matrix(matrix.cpu()).to(DEVICE).T
        )
        actual = to_cart(x) @ irreps.D_from_matrix(matrix).T
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    raw = torch.randn(4, irreps.dim, device=DEVICE)
    project = co3.Projector(irreps).to(DEVICE)
    torch.testing.assert_close(project(raw), to_cart(to_sph(raw)))
    torch.testing.assert_close(project(project(raw)), project(raw))
    assert irreps.randn(0, -1, device=DEVICE).shape == (0, irreps.dim)


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
