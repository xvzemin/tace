"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx import conv as eqx_conv
from equivariantx import o2


@pytest.mark.parametrize("training", [False, True])
def test_o2_cgtp_autotune(training, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = o2.O3TensorProduct(
        "8x2e",
        "2e",
        "8x2e+8x2e",
        [(0, 0, 0, "uvu", True), (0, 0, 1, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).cuda()
    module = eqx_conv.O2O3TensorProductConv(tp, backend="cuda").cuda().train(training)
    x = torch.randn(16, tp.input_dim, device="cuda", requires_grad=True)
    radial = torch.randn(128, 4, device="cuda", requires_grad=True)
    projection = torch.randn(4, tp.weight_numel, device="cuda", requires_grad=True)
    amplitudes = torch.randn(128, 1, device="cuda", requires_grad=True)
    vectors = torch.randn(128, 3, device="cuda", requires_grad=True)
    edges = torch.randint(16, (2, 128), device="cuda")
    inputs = x, radial, projection, amplitudes, vectors
    saved = [v.detach().clone() for v in inputs]
    rng = torch.cuda.get_rng_state()

    def evaluate(method=None):
        return module(
            x,
            radial,
            projection,
            None,
            amplitudes,
            edges,
            16,
            vectors=vectors,
            method=method,
        )

    actual = evaluate()
    expected = evaluate("baseline")
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    assert len(module.tuning_results) == 1
    result = next(iter(module.tuning_results.values()))
    assert result["derivatives"] == (2 if training else 1)
    assert module.selected_method == min(
        result["milliseconds"], key=result["milliseconds"].get
    )
    assert all(torch.isfinite(torch.tensor(v)) for v in result["milliseconds"].values())
    print(f"\nO2 CGTP training={training}: {result}")
    torch.testing.assert_close(evaluate(), actual)
    assert len(module.tuning_results) == 1
    module(
        x,
        radial[:-1],
        projection,
        None,
        amplitudes[:-1],
        edges[:, :-1],
        16,
        vectors=vectors[:-1],
    )
    assert len(module.tuning_results) == 1
    assert torch.equal(rng, torch.cuda.get_rng_state())
    for value, copy in zip(inputs, saved):
        assert value.grad is None
        torch.testing.assert_close(value, copy, rtol=0, atol=0)
    gradients = [
        torch.autograd.grad(y.square().sum(), inputs) for y in (actual, expected)
    ]
    for value, reference in zip(*gradients):
        torch.testing.assert_close(value, reference, atol=2e-9, rtol=2e-9)

    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    compiled_value = compiled(
        x, radial, projection, None, amplitudes, edges, 16, vectors=vectors
    )
    torch.testing.assert_close(
        compiled_value,
        expected,
        atol=2e-10,
        rtol=2e-10,
    )
    gradients = [
        torch.autograd.grad(y.square().sum(), inputs)
        for y in (compiled_value, evaluate("baseline"))
    ]
    for value, reference in zip(*gradients):
        torch.testing.assert_close(value, reference, atol=2e-9, rtol=2e-9)
    module.tuning_results.clear()
    module.selected_method = "baseline"
    torch.testing.assert_close(
        compiled(x, radial, projection, None, amplitudes, edges, 16, vectors=vectors),
        expected,
        atol=2e-10,
        rtol=2e-10,
    )
    assert not module.tuning_results
