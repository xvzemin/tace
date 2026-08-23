import pytest
import torch

import tace.models.eqt.equitorch.nn.sparse_product as sparse_product
from tace.models.eqt.equitorch.nn import TensorProduct


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def _reference_segment(
    source: torch.Tensor,
    indptr: torch.Tensor,
    *,
    reduce: str,
) -> torch.Tensor:
    assert reduce == "sum"
    indptr = indptr.squeeze(0)
    return torch.stack(
        [
            source[:, start:end].sum(dim=1)
            for start, end in zip(indptr[:-1].tolist(), indptr[1:].tolist())
        ],
        dim=1,
    )


def _value_and_gradients(
    tensor_product: TensorProduct,
    input1: torch.Tensor,
    input2: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    output = tensor_product(input1, input2)
    gradients = torch.autograd.grad(
        output.square().sum(),
        (input1, input2, tensor_product.weight),
    )
    return (output.detach(), *(gradient.detach() for gradient in gradients))


def test_eqt_native_scatter_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    tensor_product = TensorProduct(
        "0e+1o+2e",
        "0e+1o+2e",
        "0e+1o+2e",
        channels_in1=64,
        channels_in2=64,
        channels_out=64,
        internal_weights=True,
    ).to(device=DEVICE, dtype=DTYPE)
    input1 = torch.randn(
        8,
        tensor_product.irreps_in1_dim,
        64,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )
    input2 = torch.randn_like(input1, requires_grad=True)

    installed_segment = sparse_product._segment_csr
    monkeypatch.setattr(sparse_product, "_segment_csr", _reference_segment)
    expected = _value_and_gradients(tensor_product, input1, input2)
    if installed_segment is not None:
        monkeypatch.setattr(sparse_product, "_segment_csr", installed_segment)
        accelerated = _value_and_gradients(tensor_product, input1, input2)
        for accelerated_tensor, expected_tensor in zip(accelerated, expected):
            torch.testing.assert_close(accelerated_tensor, expected_tensor)
    monkeypatch.setattr(sparse_product, "_segment_csr", None)
    actual = _value_and_gradients(tensor_product, input1, input2)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_eqt_native_scatter_fallback_third_derivative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1)
    tensor_product = TensorProduct(
        "0e+1o+2e",
        "0e+1o+2e",
        "0e+1o+2e",
        channels_in1=4,
        channels_in2=4,
        channels_out=4,
        internal_weights=True,
    ).to(device=DEVICE, dtype=DTYPE)
    input1 = torch.randn(
        3,
        tensor_product.irreps_in1_dim,
        4,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )
    input2 = torch.randn_like(input1, requires_grad=True)

    monkeypatch.setattr(sparse_product, "_segment_csr", None)
    output = tensor_product(input1, input2)
    grad1 = torch.autograd.grad(output.sin().sum(), input1, create_graph=True)[0]
    grad2 = torch.autograd.grad(grad1.square().sum(), input1, create_graph=True)[0]
    grad3 = torch.autograd.grad(grad2.square().sum(), input1)[0]
    assert torch.isfinite(grad3).all()
