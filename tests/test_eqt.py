from typing import Optional

import pytest
import torch
from e3nn import o3

import tace.models.eqt.equitorch.nn.sparse_product as sparse_product
from tace.models._e3nn.prod import CgtpACE
from tace.models.eqt.equitorch.nn import TensorProduct
from tace.utils.env import acceleration_enabled


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def _product(correlation: int) -> CgtpACE:
    return CgtpACE(
        layer=0,
        num_layers=1,
        num_elements=1,
        Lmax=0,
        lmax=1,
        num_channel=2,
        num_expert=None,
        num_channel_per_expert=None,
        target_irreps=o3.Irreps("0e"),
        irreps_in=o3.Irreps("2x0e+2x1o"),
        correlation=[correlation],
        l1l2=None,
        bias=False,
        nonlinear=None,
    )


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


@pytest.mark.parametrize(
    ("setting", "correlation", "expected"),
    [
        (None, 2, False),
        (None, 3, True),
        ("0", 3, False),
        ("1", 2, True),
    ],
)
def test_product_eqt_selection(
    monkeypatch: pytest.MonkeyPatch,
    setting: Optional[str],
    correlation: int,
    expected: bool,
) -> None:
    if setting is None:
        monkeypatch.delenv("TACE_USE_EQT", raising=False)
    else:
        monkeypatch.setenv("TACE_USE_EQT", setting)

    product = _product(correlation)
    assert all(ace.use_eqt is expected for ace in product.aces)
    assert all(hasattr(ace, "fused_tp") is expected for ace in product.aces)


@pytest.mark.parametrize(
    ("setting", "expected"),
    [(None, None), ("0", False), ("1", True)],
)
def test_acceleration_setting(
    monkeypatch: pytest.MonkeyPatch,
    setting: Optional[str],
    expected: Optional[bool],
) -> None:
    if setting is None:
        monkeypatch.delenv("TACE_USE_EQT", raising=False)
    else:
        monkeypatch.setenv("TACE_USE_EQT", setting)

    assert acceleration_enabled("eqt") is expected


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
