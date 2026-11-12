"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import conv as eqx_conv
from equivariantx import o2
from equivariantx.ops.precision import (
    validate_module_precision,
    validate_precision,
    validate_uniform,
)


def test_public_harmonics_reject_autocast():
    from equivariantx.co3 import CartesianHarmonics

    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(TypeError, match="autocast"):
            CartesianHarmonics(2, normalize=False)(torch.ones(3, 3))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_public_harmonics_reject_reduced_precision(dtype):
    from equivariantx.co3 import CartesianHarmonics

    module = CartesianHarmonics(2, normalize=False)
    with pytest.raises(TypeError, match="float32 or float64"):
        module(torch.ones(3, 3, dtype=dtype))


def test_module_state_checked():
    module = torch.nn.Linear(3, 2).to(dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="float32 or float64"):
        validate_module_precision(module, [torch.ones(2, 3)])


def test_validation_is_traceable():
    def function(value):
        validate_precision([value])
        return value.square()

    compiled = torch.compile(function, backend="eager", fullgraph=True)
    value = torch.randn(3, requires_grad=True)
    torch.testing.assert_close(compiled(value), function(value))


def test_autocast_rejected():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(TypeError, match="autocast"):
            validate_precision([torch.ones(2)])


def test_mixed_geometry_precision():
    values = [torch.ones(2, dtype=torch.float32), torch.ones(3, dtype=torch.float64)]
    validate_precision(values)
    with pytest.raises(ValueError, match="share a dtype"):
        validate_uniform(values)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.complex64, torch.complex128]
)
def test_reduced_storage_rejected(dtype):
    with pytest.raises(TypeError, match="float32 or float64"):
        validate_precision([{"features": [torch.ones(3, dtype=dtype)]}])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_supported_storage(dtype):
    validate_precision([torch.ones(3, dtype=dtype), {"index": torch.arange(3)}])


@pytest.mark.parametrize("kind", ["o3", "o2", "co3", "uu"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_convolution_geometry_precision(kind, backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    args = (
        "2x0e+2x1o",
        "0e+1o",
        "2x0e+2x1o",
        [(0, 0, 0, "uvu", True), (1, 1, 0, "uvu", True), (0, 1, 1, "uvu", True)],
    )
    from equivariantx import co3

    tp_cls = {
        "o3": o3.TensorProduct,
        "o2": o2.O3TensorProduct,
        "co3": co3.TensorProduct,
        "uu": o2.O3TensorProduct,
    }[kind]
    tp = (
        tp_cls(
            *args,
            internal_weights=False,
            shared_weights=False,
        )
        .float()
        .to(device)
    )
    if kind == "uu":
        local_frame = o2.LocalFrame(args[0]).to(device)
        tp = o2.UuLinear(local_frame.irreps_out, local_frame.irreps_out, num_channel=2)
        module = (
            eqx_conv.UuO2TensorProductConv(
                local_frame,
                tp,
                local_frame,
                backend=backend,
            )
            .float()
            .to(device)
        )
    else:
        module = {
            "o3": eqx_conv.O3TensorProductConv,
            "o2": eqx_conv.O2O3TensorProductConv,
            "co3": eqx_conv.CartesianTensorProductConv,
        }[kind](
            tp,
            backend=backend,
        ).to(device)
    frame = o2.WignerD(1, 1).to(device)
    x = torch.randn(3, 8, dtype=torch.float32, device=device, requires_grad=True)
    vectors = torch.randn(5, 3, dtype=torch.float64, device=device, requires_grad=True)
    radial = torch.randn(5, 3, dtype=torch.float64, device=device, requires_grad=True)
    projection = torch.randn(
        3,
        tp.weight_numel,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    index = torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 0]], device=device)
    amplitudes = torch.ones(
        1, 1 if kind == "uu" else 2, dtype=torch.float64, device=device
    )
    angular = o3.spherical_harmonics(
        [0, 1],
        vectors,
        normalize=True,
        normalization="component",
    )
    if kind in ("o3", "co3"):
        actual = module(x, angular, radial, projection, index, vectors=vectors)
        expected = module(x, angular.float(), radial.float(), projection, index)
    else:
        actual = module(
            x,
            radial,
            projection,
            None,
            amplitudes,
            index,
            3,
            vectors=vectors,
        )
        expected = module(
            x,
            radial.float(),
            projection,
            frame.forward_packed(vectors).float(),
            amplitudes.float(),
            index,
            3,
        )
    assert actual.dtype == torch.float32
    variables = (x, vectors, radial, projection)
    for _ in range(2):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        results = [
            torch.autograd.grad(value.square().sum(), variables, create_graph=True)
            for value in (actual, expected)
        ]
        for gradients in results:
            assert tuple(g.dtype for g in gradients) == tuple(
                v.dtype for v in variables
            )
            assert all(torch.isfinite(g).all() for g in gradients)
        actual, expected = (
            torch.cat([g.flatten() for g in grads]) for grads in results
        )


def test_cuda_operand_precision_validation():
    from equivariantx.ops.registry import validate_operands

    validate_operands([torch.zeros(2), torch.ones(3), torch.zeros(2, dtype=torch.long)])
    with pytest.raises(ValueError, match="share a dtype"):
        validate_operands([torch.zeros(2), [torch.ones(2, dtype=torch.float64)]])
    with pytest.raises(TypeError, match="float32 or float64"):
        validate_operands([torch.zeros(2, dtype=torch.bfloat16)])


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_uv_convolution_geometry_precision(backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    frame = o2.LocalFrame("2x0e+2x1o").to(device)
    scalars = frame.irreps_out.filter(keep=lambda entry: entry.ir.m == 0)
    tensors = frame.irreps_out.filter(keep=lambda entry: entry.ir.m > 0)
    gate = (
        o2.Gate(
            scalars,
            [torch.nn.SiLU() for _ in scalars],
            o2.Irreps([(o2.Irrep("0e"), tensors.num_irreps)]),
            [torch.nn.Sigmoid()],
            tensors,
        )
        .float()
        .to(device)
    )
    up = o2.Linear((frame.irreps_out * 2).regroup(), gate.irreps_in).float().to(device)
    down = o2.Linear(gate.irreps_out, frame.irreps_out).float().to(device)
    layer = (
        eqx_conv.UvO2TensorProductConv(
            frame,
            up,
            gate,
            down,
            frame,
            backend=backend,
        )
        .float()
        .to(device)
    )
    features = torch.randn(3, frame.input_dim, device=device, requires_grad=True)
    vectors = torch.randn(4, 3, dtype=torch.float64, device=device, requires_grad=True)
    weights = torch.randn(
        4,
        layer.weight_numel,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    cutoff = torch.rand(4, 1, dtype=torch.float64, device=device, requires_grad=True)
    index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device)
    parameters = up.weight, up.bias, down.weight, down.bias
    actual = layer(
        features,
        None,
        weights,
        index,
        None,
        None,
        cutoff,
        parameters,
        vectors=vectors,
    )
    wigner = layer.wigner.forward_packed(vectors).float()
    expected = layer(
        features,
        None,
        weights.float(),
        index,
        wigner,
        None,
        cutoff.float(),
        parameters,
    )
    variables = features, vectors, weights, cutoff, up.weight, down.weight
    for _ in range(2):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        gradients = [
            torch.autograd.grad(value.square().sum(), variables, create_graph=True)
            for value in (actual, expected)
        ]
        actual, expected = [
            torch.cat([g.flatten() for g in result]) for result in gradients
        ]
