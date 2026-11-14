"""Direction-contraction providers and their multilinear transposes."""

import pytest
import torch

from equivariantx import conv, o2
from equivariantx.ops import get_kernel
from equivariantx.ops.common.geometry import direction_contraction


def direction_case(rank, mode, projected, shared, *, edges=4, device="cpu"):
    tensor_product = o2.O3TensorProduct(
        "2x1o + 2x2e",
        "0e + 1o + 2e",
        "2x1o + 2x2e + 2x1e + 2x2o",
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
    layer = conv.O2O3TensorProductConv(tensor_product).to(device)
    rows = 1 if shared else edges
    vectors = torch.randn(rows, 3, device=device) + 0.5
    frame = o2.WignerD(2, 2, method="recursive").to(device)
    rotation = frame.forward_packed(vectors)
    width = tensor_product.weight_numel
    operands = [
        torch.randn(3, tensor_product.input_dim, device=device),
        torch.randn(rows, 3 if projected else width, device=device),
        torch.randn(3, width, device=device)
        if projected
        else vectors.new_empty(0, width),
        rotation.detach(),
        rotation.detach(),
        torch.randn(rows, 3, device=device),
        torch.randn(4, tensor_product.irreps_out.dim, device=device),
        *[torch.randn(rows, 3, device=device) for _ in range(rank)],
    ]
    source = torch.tensor([0, 1, 2, 0], device=device)[:edges]
    target = torch.tensor([1, 2, 1, 3], device=device)[:edges]
    return layer.direction_metadata, vectors, source, target, operands


def program(rank, count, roles):
    return repr(
        (
            (
                rank,
                tuple(range(count)),
                False,
                tuple((role, i) for i, role in enumerate(roles)),
            ),
        )
    )


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize(
    "projected,shared", [(False, False), (True, False), (True, True)]
)
def test_provider_and_transposes(double_precision, rank, mode, projected, shared):
    metadata, vectors, source, target, operands = direction_case(
        rank, mode, projected, shared
    )
    roles = [0, 1, *([2] if projected else []), 5, 6, *range(7, len(operands))]
    active = [role for role in roles if role != 6]
    for role in active:
        operands[role].requires_grad_()
    description = program(rank, len(operands), roles)
    actual = get_kernel("direction_contraction", "torch")(
        metadata, description, vectors, source, target, operands, False
    )
    loss = (actual[roles.index(6)] * operands[6]).sum()
    gradients = torch.autograd.grad(loss, [operands[role] for role in active])
    for role, gradient in zip(active, gradients):
        torch.testing.assert_close(
            actual[roles.index(role)], gradient, atol=2e-11, rtol=2e-11
        )
    if torch.cuda.is_available():
        expected = get_kernel("direction_contraction", "cuda")(
            metadata,
            description,
            vectors.cuda(),
            source.cuda(),
            target.cuda(),
            [value.detach().cuda() for value in operands],
            True,
        )
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference.cpu(), atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_empty_edges(double_precision, projected, shared):
    metadata, vectors, source, target, operands = direction_case(
        2, "uvu", projected, shared, edges=0
    )
    roles = [0, 1, *([2] if projected else []), 5, 6, 7, 8]
    results = get_kernel("direction_contraction", "torch")(
        metadata,
        program(2, len(operands), roles),
        vectors,
        source,
        target,
        operands,
        False,
    )
    for role, result in zip(roles, results):
        torch.testing.assert_close(result, torch.zeros_like(operands[role]))


@pytest.mark.parametrize("allow_tf32", [False, True])
def test_mixed_channels_fp32(request, double_precision, allow_tf32):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous = torch.backends.cuda.matmul.allow_tf32
    request.addfinalizer(
        lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", previous)
    )
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    metadata, vectors, source, target, operands = direction_case(2, "uvw", True, True)
    values = [value.to(device="cuda", dtype=torch.float32) for value in operands]
    roles = [0, 1, 2, 5, 6, 7, 8]
    description = program(2, len(values), roles)
    actual = get_kernel("direction_contraction", "cuda")(
        metadata,
        description,
        vectors.float().cuda(),
        source.cuda(),
        target.cuda(),
        values,
        True,
    )
    expected = get_kernel("direction_contraction", "torch")(
        metadata,
        description,
        vectors,
        source,
        target,
        [value.cpu().double() for value in values],
        False,
    )
    for value, reference in zip(actual, expected):
        assert value.dtype == torch.float32
        value = value.cpu().double()
        if allow_tf32:
            # Relative entry errors are ill-conditioned near cancellations.
            residual = value - reference
            assert residual.norm() <= 2e-3 * reference.norm().clamp_min(1e-12)
            assert residual.abs().max() <= 2e-3 * reference.abs().max().clamp_min(1e-12)
        else:
            torch.testing.assert_close(value, reference, atol=2e-5, rtol=5e-5)


@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["uvu", "uvw"])
def test_recursive_adjoints(double_precision, rank, device, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    metadata, vectors, source, target, operands = direction_case(
        rank, mode, True, True, device=device
    )
    active = [0, 1, 2, 5, *range(7, len(operands))]
    for role in active:
        operands[role].requires_grad_()
    description = program(rank, len(operands), [6])
    actual = direction_contraction(
        metadata, description, vectors, source, target, operands, True
    )[0]
    expected = get_kernel("direction_contraction", "torch")(
        metadata, description, vectors, source, target, operands, False
    )[0]
    parameters = tuple(operands[role] for role in active)
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)
        seed = torch.randn_like(actual) / max(1, actual.numel())
        derivatives = [
            torch.autograd.grad(
                (value.sin() * seed).sum(),
                parameters,
                create_graph=True,
                retain_graph=True,
            )
            for value in (actual, expected)
        ]
        actual, expected = [
            torch.cat([value.flatten() for value in gradient])
            for gradient in derivatives
        ]
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)
