"""Tests for the corresponding EQX source module."""

import math
import pytest
import torch
from e3nn import o3
from equivariantx import o3 as eqx_o3

IRREPS = "0e+0o+1e+1o+2e+2o"


@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"), reason="Requires time-reversal e3nn"
)
@pytest.mark.parametrize("kind", ["gaunt", "matrix"])
def test_tensor_product_time_reversal(kind, device, double_precision):
    irreps = o3.Irreps("2x0ee+2x0eo+2x1ee+2x1eo+2x1oe+2x1oo+2x2ee+2x2eo")
    cls = eqx_o3.GauntTensorProduct if kind == "gaunt" else eqx_o3.MatrixTensorProduct
    tp = cls(irreps, irreps, irreps, internal_weights=False).to(device)
    signs = torch.tensor(
        [ir.t for mul, ir in irreps for _ in range(mul * ir.dim)], device=device
    )
    x, y = [
        torch.randn(3, irreps.dim, device=device, requires_grad=True) for _ in range(2)
    ]
    w = torch.randn(tp.weight_numel, device=device, requires_grad=True)
    actual, expected = tp(x * signs, y * signs, w), tp(x, y, w) * signs
    if kind == "gaunt":
        reference = cls(
            irreps, irreps, irreps, method="coefficients", internal_weights=False
        ).to(device)
        torch.testing.assert_close(
            tp(x, y, w), reference(x, y, w), atol=2e-12, rtol=2e-12
        )
    rotation = irreps.D_from_matrix(-o3.rand_matrix(dtype=x.dtype)).to(device)
    torch.testing.assert_close(
        tp(x @ rotation.T, y @ rotation.T, w),
        tp(x, y, w) @ rotation.T,
        atol=2e-11,
        rtol=2e-11,
    )
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        probe = torch.randn_like(actual)
        gradients = [
            torch.autograd.grad((z.sin() * probe).sum(), (x, y, w), create_graph=True)
            for z in (actual, expected)
        ]
        actual, expected = [torch.cat([g.flatten() for g in gs]) for gs in gradients]


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_product_equivariance(cls, device, dtype, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = cls(IRREPS, IRREPS, IRREPS, shared_weights=False).to(
        device=device, dtype=dtype
    )
    x = torch.randn(3, module.irreps_in1.dim, device=device, dtype=dtype)
    y = torch.randn_like(x)
    weight = torch.randn(3, module.weight_numel, device=device, dtype=dtype)
    for matrix in (o3.rand_matrix(), -torch.eye(3), -o3.rand_matrix()):
        rotation = module.irreps_in1.D_from_matrix(matrix).to(
            device=device, dtype=dtype
        )
        actual = module(x @ rotation.T, y @ rotation.T, weight)
        expected = module(x, y, weight) @ rotation.T
        tolerance = 2e-5 if dtype == torch.float32 else 2e-12
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_antisymmetric_coupling(double_precision):
    x, y = torch.randn(2, 7, 3)
    matrix = eqx_o3.MatrixTensorProduct("1o", "1o", "1e")
    weight = torch.ones(matrix.weight_numel)
    result = matrix(x, y, weight)
    assert result.norm() > 0.1
    torch.testing.assert_close(result, -matrix(y, x, weight), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(
        matrix(x, x, weight), torch.zeros_like(result), atol=1e-13, rtol=0
    )
    for method in ("quadrature", "coefficients"):
        gaunt = eqx_o3.GauntTensorProduct("1o", "1o", "1e", method=method)
        assert not gaunt(x, y).count_nonzero()
        assert not gaunt.output_mask.count_nonzero()


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
@pytest.mark.parametrize("normalization", ["component", "norm"])
def test_product_variance_normalization(cls, normalization, double_precision):
    module = cls(IRREPS, IRREPS, IRREPS, irrep_normalization=normalization)
    features = torch.eye(module.irreps_in1.dim)
    if normalization == "norm":
        scale = torch.cat(
            [torch.full((ir.dim,), math.sqrt(ir.dim)) for _, ir in module.irreps_in1]
        )
        features = features / scale
    coefficients = module(
        features[:, None], features[None], torch.ones(module.weight_numel)
    )
    variances = coefficients.square().sum((0, 1))
    expected = torch.ones_like(variances)
    if normalization == "norm":
        expected /= scale.square()
    torch.testing.assert_close(variances, expected, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
def test_regroup_and_broadcast(cls, double_precision):
    module = cls("1o+2x0e+1o", "0e+1o", "0e+2x1o+0e", shared_weights=False)
    reference = cls("2x0e+2x1o", "0e+1o", "2x0e+2x1o", shared_weights=False)
    x = torch.randn(3, 1, 8)
    y = torch.randn(1, 2, 4)
    weight = torch.randn(3, 2, module.weight_numel)
    packed = torch.cat((x[..., 3:5], x[..., :3], x[..., 5:]), -1)
    expected = reference(packed, y, weight)
    expected = torch.cat((expected[..., :1], expected[..., 2:], expected[..., 1:2]), -1)
    torch.testing.assert_close(module(x, y, weight), expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradcheck_and_gradgradcheck(cls, device, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = cls("0e+1o", "0o+1e", "0o+1e+2o", shared_weights=False).to(device)
    inputs = (
        torch.randn(2, 4, device=device, requires_grad=True),
        torch.randn(2, 4, device=device, requires_grad=True),
        torch.randn(2, module.weight_numel, device=device, requires_grad=True),
    )
    assert torch.autograd.gradcheck(module, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(module, inputs, fast_mode=True)


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
def test_precision_state_and_empty_batch(cls, double_precision):
    args = ("0e+1o+3o", "0e+1o", "0e+1o+2e+3o")
    module = cls(*args)
    torch.set_default_dtype(torch.float32)
    restored = cls(*args).double()
    torch.set_default_dtype(torch.float64)
    restored.load_state_dict(module.state_dict())
    for name, value in module.named_buffers():
        torch.testing.assert_close(
            value, dict(restored.named_buffers())[name], atol=0, rtol=0
        )
    x = torch.randn(2, module.irreps_in1.dim)
    y = torch.randn(2, module.irreps_in2.dim)
    torch.testing.assert_close(restored(x, y), module(x, y), atol=0, rtol=0)
    empty = module(x[:0], y[:0])
    assert empty.shape == (0, module.irreps_out.dim)
    assert torch.autograd.grad(empty.sum(), module.weight)[0].count_nonzero() == 0


@pytest.mark.parametrize("degree", [5, 6])
def test_high_degree_products(degree, double_precision):
    irreps = f"0e+{degree}e+{degree}o"
    gaunt = eqx_o3.GauntTensorProduct(irreps, irreps, irreps)
    reference = eqx_o3.GauntTensorProduct(irreps, irreps, irreps, method="coefficients")
    reference.load_state_dict(gaunt.state_dict())
    x, y = torch.randn(2, 3, gaunt.irreps_in1.dim)
    torch.testing.assert_close(gaunt(x, y), reference(x, y), atol=2e-12, rtol=2e-12)
    matrix = eqx_o3.MatrixTensorProduct(irreps, irreps, irreps)
    rotation = matrix.irreps_in1.D_from_matrix(-o3.rand_matrix())
    torch.testing.assert_close(
        matrix(x @ rotation.T, y @ rotation.T),
        matrix(x, y) @ rotation.T,
        atol=2e-11,
        rtol=2e-11,
    )


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
def test_invalid_layout_and_weights(cls):
    with pytest.raises(ValueError, match="uniform"):
        cls("2x0e+1o", "0e", "2x0e")
    with pytest.raises(ValueError, match="multiplicities"):
        cls("2x0e", "3x0e", "3x0e")
    with pytest.raises(ValueError, match="shared"):
        cls("0e", "0e", "0e", internal_weights=True, shared_weights=False)
    module = cls("0e", "0e", "0e", internal_weights=False)
    with pytest.raises(ValueError, match="External weights"):
        module(torch.ones(1), torch.ones(1))
