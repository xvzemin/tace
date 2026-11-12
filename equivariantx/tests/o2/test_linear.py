"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
from equivariantx.tests.o2.testing import _transform
from equivariantx.tests.o2.testing import _time_reverse


@pytest.mark.parametrize("normalization", ["element", "path"])
@pytest.mark.parametrize("weight_batch", [(), (1,), (3,)])
def test_linear_grouped_instructions(double_precision, normalization, weight_batch):
    module = o2.Linear(
        "2x1m+3x1m+0e",
        "3x1m+2x1m+0e",
        internal_weights=False,
        shared_weights=False,
        path_normalization=normalization,
    )
    x = torch.randn(3, module.irreps_in.dim, requires_grad=True)
    weight = torch.randn(*weight_batch, module.weight_numel, requires_grad=True)
    inputs = [
        x[..., s].reshape(3, ir.dim, mul)
        for (ir, mul), s in zip(module.irreps_in, module.irreps_in.slices())
    ]
    outputs = []
    for i_out, (ir, mul) in enumerate(module.irreps_out):
        terms = [
            torch.matmul(
                inputs[ins.i_in], module.weight_view_for_instruction(i, weight)
            )
            * ins.path_weight
            for i, ins in enumerate(module._weight_instructions)
            if ins.i_out == i_out
        ]
        outputs.append(sum(terms).reshape(3, ir.dim * mul))
    actual, expected = module(x, weight), torch.cat(outputs, dim=-1)
    for _ in range(2):
        torch.testing.assert_close(actual, expected)
        gradients = [
            torch.autograd.grad(
                value.sin().sum(), (x, weight), create_graph=True, retain_graph=True
            )
            for value in (actual, expected)
        ]
        actual, expected = [
            torch.cat([g.flatten() for g in values]) for values in gradients
        ]


@pytest.mark.parametrize("reflected", [False, True])
def test_o2_linear_is_equivariant(reflected):
    module = o2.Linear(
        "2x0e+3x1m",
        "4x0e+2x1m+0o",
        biases=True,
    ).to(DEVICE, DTYPE)
    features = module.irreps_in.randn(6, -1, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.41, dtype=DTYPE, device=DEVICE)
    expected = _transform(module(features), module.irreps_out, angle, reflected)
    actual = module(_transform(features, module.irreps_in, angle, reflected))
    torch.testing.assert_close(actual, expected)


def test_o2_linear_external_weights_broadcast_and_zero_pad():
    module = o2.Linear(
        "2x0e+3x1m",
        "4x0e+2x1m+0o",
        internal_weights=False,
        shared_weights=False,
    ).to(DEVICE, DTYPE)
    features = module.irreps_in.randn(5, -1, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(5, module.weight_numel, dtype=DTYPE, device=DEVICE)
    output = module(features, weights)
    reference = torch.stack(
        [module(features[index], weights[index]) for index in range(5)]
    )
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(output[:, -1], torch.zeros_like(output[:, -1]))

    singleton = torch.randn(1, module.weight_numel, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        module(features, singleton), module(features, singleton[0])
    )


@pytest.mark.parametrize("batch_size", [0, 5])
@pytest.mark.parametrize("weight_batch", ["shared", "singleton", "edge"])
def test_o2_uu_linear_matches_channel_diagonal_linear(
    double_precision, batch_size, weight_batch
):
    module = o2.UuLinear(
        "4x0ee+2x0ee+4x0oe+4x1me+2x1mo",
        "4x0ee+2x0oe+2x1me+4x1mo+2x2me",
        num_channel=2,
    ).to(DEVICE, DTYPE)
    reference = o2.Linear(
        module.irreps_in,
        module.irreps_out,
        internal_weights=False,
        shared_weights=False,
    ).to(DEVICE, DTYPE)
    features = module.irreps_in.randn(
        batch_size,
        -1,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )
    leading = (
        ()
        if weight_batch == "shared"
        else (1 if weight_batch == "singleton" else batch_size,)
    )
    weight = torch.randn(
        *leading,
        module.weight_numel,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )
    matrices = []
    offset = 0
    for instruction in module.instructions:
        copies_in, copies_out, channels = instruction.path_shape
        size = copies_in * copies_out * channels
        matrix = weight[..., offset : offset + size].reshape(
            *leading, copies_in, copies_out, channels
        )
        matrix = torch.einsum(
            "...iju,uv->...iujv",
            matrix,
            torch.eye(channels, device=DEVICE, dtype=DTYPE),
        )
        matrices.append(
            matrix.reshape(*leading, copies_in * copies_out * channels**2)
            * channels**0.5
        )
        offset += size
    expected = reference(features, torch.cat(matrices, dim=-1))
    actual = module(features, weight)
    torch.testing.assert_close(actual, expected)
    assert list(module.parameters()) == []
    for _ in range(2):
        gradients = [
            torch.autograd.grad(
                output.square().sum(),
                (features, weight),
                create_graph=True,
                retain_graph=True,
            )
            for output in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)
        actual, expected = [
            torch.cat([g.flatten() for g in grads]) for grads in gradients
        ]

    angle = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)
    for reflected, time_reversal in ((False, False), (True, False), (False, True)):
        torch.testing.assert_close(
            module(
                _transform(features, module.irreps_in, angle, reflected, time_reversal),
                weight,
            ),
            _transform(
                module(features, weight),
                module.irreps_out,
                angle,
                reflected,
                time_reversal,
            ),
        )


def test_o2_uu_linear_compiles_with_dynamic_batches(double_precision):
    module = o2.UuLinear("4x0e+4x1m", "2x0e+6x1m+2x0o", 2)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for batch_size in (3, 7, 0):
        features = torch.randn(batch_size, module.irreps_in.dim, requires_grad=True)
        weight = torch.randn(batch_size, module.weight_numel, requires_grad=True)
        actual, expected = compiled(features, weight), module(features, weight)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), (features, weight)),
            torch.autograd.grad(expected.square().sum(), (features, weight)),
        ):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("path_normalization", ["element", "path"])
@pytest.mark.parametrize("shared_weights", [False, True])
def test_o2_scalar_linear_matches_o3_normalization(
    double_precision, path_normalization, shared_weights
):
    irreps_in, irreps_out = "2x0e+3x0e+2x0o", "4x0e+2x0o"
    kwargs = dict(
        internal_weights=False,
        shared_weights=shared_weights,
        biases=shared_weights,
        path_normalization=path_normalization,
    )
    module = o2.Linear(irreps_in, irreps_out, **kwargs)
    reference = o3.Linear(irreps_in, irreps_out, **kwargs)
    assert module.weight_numel == reference.weight_numel
    x = torch.randn(5, module.irreps_in.dim, requires_grad=True)
    weight_shape = (
        (module.weight_numel,) if shared_weights else (5, module.weight_numel)
    )
    weight = torch.randn(weight_shape, requires_grad=True)
    bias = (
        torch.randn(module.bias_numel, requires_grad=True) if shared_weights else None
    )
    actual, expected = module(x, weight, bias), reference(x, weight, bias)
    torch.testing.assert_close(actual, expected)
    inputs = (x, weight, bias) if shared_weights else (x, weight)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), inputs),
        torch.autograd.grad(expected.square().sum(), inputs),
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("batch_size", [0, 3])
def test_o2_linear_broadcasts_bias_and_unconnected_outputs(
    double_precision, batch_size
):
    module = o2.Linear(
        "2x0e+1m",
        "0e+1m+0o",
        internal_weights=False,
        shared_weights=False,
        biases=True,
    )
    x = torch.randn(1, 1, module.irreps_in.dim, requires_grad=True)
    weight = torch.randn(2, 1, module.weight_numel, requires_grad=True)
    bias = torch.randn(1, batch_size, module.bias_numel, requires_grad=True)
    output = module(x, weight, bias)
    expected = module(
        x.expand(2, batch_size, -1),
        weight.expand(2, batch_size, -1),
        bias.expand(2, batch_size, -1),
    )
    assert output.shape == (2, batch_size, module.irreps_out.dim)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(output[..., -1], torch.zeros_like(output[..., -1]))
    for grad in torch.autograd.grad(output.square().sum(), (x, weight, bias)):
        assert torch.isfinite(grad).all()


def test_o2_linear_preserves_time_parity():
    irreps = o2.Irreps("2x0ee+3x0eo+2x1me+3x1mo")
    linear = o2.Linear(irreps, irreps, biases=True).to(DEVICE, DTYPE)
    assert linear.bias_numel == 2
    assert all(
        linear.irreps_in[instruction.i_in].ir.t
        == linear.irreps_out[instruction.i_out].ir.t
        for instruction in linear._weight_instructions
    )
    features = irreps.randn(5, -1, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        linear(_time_reverse(features, irreps)),
        _time_reverse(linear(features), irreps),
    )


@pytest.mark.parametrize("batch", [0, 3])
def test_uu_linear_expanded_paths(batch, double_precision):
    expanded = o2.UuLinear("4x0e+4x1m", "6x0e+6x1m", 2, path_mode="expand")
    reduced = o2.UuLinear(expanded.irreps_in, expanded.requested_irreps_out, 2)
    x = torch.randn(batch, expanded.irreps_in.dim)
    weight = torch.randn(batch, expanded.weight_numel)
    y = expanded(x, weight)
    actual = torch.cat(
        [
            part.reshape(batch, ir.dim, 2, 3, 2).sum(-3).flatten(-3)
            for part, (ir, _) in zip(
                y.split([entry.dim for entry in expanded.irreps_out], -1),
                expanded.irreps_out,
            )
        ],
        -1,
    )
    torch.testing.assert_close(actual / 2**0.5, reduced(x, weight))
