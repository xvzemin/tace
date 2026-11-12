"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx import o2
from equivariantx.tests.o2.testing import _time_reverse, _transform
from equivariantx.tests.ops.testing import assert_functional_derivatives

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def test_gate_derivatives(double_precision):
    gate = o2.Gate(
        "0e+0o",
        [torch.nn.SiLU(), torch.nn.Tanh()],
        "2x0o+3x0e",
        [torch.nn.Tanh(), torch.nn.Sigmoid()],
        "1m+0o+2x2m+0e",
    )
    features = torch.randn(2, gate.irreps_in.dim, requires_grad=True)
    assert torch.autograd.gradcheck(gate, (features,), fast_mode=True)
    assert torch.autograd.gradgradcheck(gate, (features,), fast_mode=True)

    tangent = torch.randn_like(features)
    _, expected = torch.autograd.functional.jvp(gate, features, tangent)
    _, actual = torch.func.jvp(gate, (features,), (tangent,))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.vmap(gate)(features), gate(features))


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_functional_derivatives(backend, compiler, double_precision):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    gate = o2.Gate(
        "0e+0o",
        [torch.nn.SiLU(), torch.nn.Tanh()],
        "2x0o+3x0e",
        [torch.nn.Tanh(), torch.nn.Sigmoid()],
        "1m+0o+2x2m+0e",
        backend=backend,
    ).to(device)

    def function(features, index):
        return gate(features.sin())

    def reference(features, index):
        inputs = features.sin().split(gate._input_sizes, dim=-1)
        scalars = gate.act_scalars(gate._select_entries(inputs, gate._selections[0]))
        gates = gate.act_gates(gate._select_entries(inputs, gate._selections[1]))
        values = gate._select_entries(inputs, gate._selections[2])
        values = values.index_select(-1, gate._gated_indices) * gate._gated_signs
        output = values * gates.index_select(-1, gate._expanded_indices)
        return torch.cat((scalars, output), dim=-1)

    features = torch.randn(3, gate.irreps_in.dim, device=device) * 0.2
    assert_functional_derivatives(
        function, reference, (features, torch.arange(3, device=device)), compiler
    )


@pytest.mark.parametrize("reflected", [False, True])
def test_o2_gate_is_equivariant(reflected):
    module = o2.Gate(
        "2x0e+0o",
        [torch.nn.SiLU(), torch.nn.Tanh()],
        "3x0e",
        [torch.nn.Sigmoid()],
        "3x1m",
    ).to(DEVICE, DTYPE)
    features = module.irreps_in.randn(5, -1, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.29, dtype=DTYPE, device=DEVICE)
    expected = _transform(module(features), module.irreps_out, angle, reflected)
    actual = module(_transform(features, module.irreps_in, angle, reflected))
    torch.testing.assert_close(actual, expected)


def test_o2_odd_scalar_activation_is_odd():
    activation = o2.Activation("2x0o", [torch.nn.Tanh()])
    features = torch.randn(4, 2, dtype=DTYPE)
    torch.testing.assert_close(activation(-features), -activation(features))


def test_o2_gated_linear_compiles_with_dynamic_batches(double_precision):
    gate = o2.Gate(
        "2x0ee+0oo",
        [torch.nn.SiLU(), torch.tanh],
        "4x0ee",
        [torch.sigmoid],
        "2x1me+2x1mo",
    )
    linear_up = o2.Linear(
        "2x0ee+0oo+2x1me+2x1mo",
        gate.irreps_in,
        internal_weights=False,
        shared_weights=False,
    )
    linear_down = o2.Linear(gate.irreps_out, linear_up.irreps_in)

    def forward(features, weight):
        return linear_down(gate(linear_up(features, weight)))

    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True, dynamic=True)
    for batch_size in (3, 1, 0):
        features = linear_up.irreps_in.randn(batch_size, -1, requires_grad=True)
        weight = torch.randn(batch_size, linear_up.weight_numel, requires_grad=True)
        actual, expected = compiled(features, weight), forward(features, weight)
        torch.testing.assert_close(actual, expected)
        inputs = (features, weight, linear_down.weight)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), inputs),
            torch.autograd.grad(expected.square().sum(), inputs),
        ):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize(
    ("scalars", "gates", "gated"),
    [("", "", ""), ("0o", "", ""), ("", "0o", "1m")],
)
def test_o2_gate_supports_empty_sectors(double_precision, scalars, gates, gated):
    module = o2.Gate(
        scalars,
        [torch.tanh] if scalars else [],
        gates,
        [torch.tanh] if gates else [],
        gated,
    )
    for batch_size in (4, 0):
        features = module.irreps_in.randn(batch_size, -1, requires_grad=True)
        output = module(features)
        assert output.shape == (batch_size, module.irreps_out.dim)
        transformed = _transform(features, module.irreps_in, torch.tensor(0.3), True)
        torch.testing.assert_close(
            module(transformed),
            _transform(output, module.irreps_out, torch.tensor(0.3), True),
        )
        (grad,) = torch.autograd.grad(output.sum(), (features,))
        assert torch.isfinite(grad).all()


def test_o2_activation_and_gate_preserve_time_parity():
    odd_activation = o2.Activation("3x0eo", [torch.nn.Tanh()])
    features = torch.randn(4, 3, dtype=DTYPE)
    torch.testing.assert_close(odd_activation(-features), -odd_activation(features))

    gate = o2.Gate(
        "2x0ee",
        [torch.nn.SiLU()],
        "3x0oo",
        [torch.nn.Tanh()],
        "3x1mo",
    ).to(DEVICE, DTYPE)
    features = gate.irreps_in.randn(7, -1, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        gate(_time_reverse(features, gate.irreps_in)),
        _time_reverse(gate(features), gate.irreps_out),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_o2_gate(dtype):
    from equivariantx import o2

    args = (
        "2x0ee+1x0oe",
        [torch.nn.SiLU(), torch.tanh],
        "2x0ee+3x0oo",
        [torch.sigmoid, torch.tanh],
        "1x0oe+2x1mo+2x2me",
    )
    reference = o2.Gate(*args).to("cuda", dtype)
    fused = o2.Gate(*args, backend="cuda").to("cuda", dtype)
    for batch in ((), (2, 3), (0,)):
        features = torch.randn(
            *batch, fused.irreps_in.dim, device="cuda", dtype=dtype, requires_grad=True
        )
        outputs = [module(features) for module in (fused, reference)]
        for _ in range(3):
            torch.testing.assert_close(*outputs)
            outputs = [
                torch.autograd.grad(value.sin().sum(), features, create_graph=True)[0]
                for value in outputs
            ]
        for reflected, time_reversal in ((True, False), (False, True), (True, True)):
            angle = features.new_tensor(0.37)

            def transform(irreps):
                return irreps.D_from_angle(
                    angle, reflected=reflected, time_reversal=time_reversal
                )

            torch.testing.assert_close(
                fused(features @ transform(fused.irreps_in).T),
                fused(features) @ transform(fused.irreps_out).T,
            )
