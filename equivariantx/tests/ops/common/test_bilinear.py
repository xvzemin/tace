"""Functional derivatives of fused product, gates and expert maps."""

import pytest
import torch
from e3nn import o3

from equivariantx.mbe.ace import BilinearContraction
from equivariantx.tests.ops.testing import assert_functional_derivatives


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA products")
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_functional_derivatives(compiler, double_precision):
    irreps = "4x0e+4x1o"
    product = o3.TensorProduct(
        irreps,
        irreps,
        irreps,
        [(0, 0, 0, "uuu", True), (1, 1, 0, "uuu", True), (0, 1, 1, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    linear = o3.Linear(
        "2x0e+2x1o", "2x0e+2x1o", internal_weights=False, shared_weights=False
    )
    shared = o3.Linear(irreps, irreps, internal_weights=False, shared_weights=False)
    module, reference = [
        BilinearContraction(product, linear, 2, backend, shared).cuda()
        for backend in ("cuda", "torch")
    ]
    values = (
        torch.randn(5, module.input_dim, device="cuda"),
        torch.randn(5, module.base_dim, device="cuda"),
        torch.randn(5, module.gate_dim, device="cuda"),
        torch.randn(3, 2, linear.weight_numel, device="cuda"),
        torch.randn(shared.weight_numel, device="cuda"),
        torch.tensor([0, 1, 1, 2, 0], device="cuda"),
    )
    assert_functional_derivatives(
        lambda x, y, g, w, s, t: module(x, y, g, w, t, s),
        lambda x, y, g, w, s, t: reference(x, y, g, w, t, s),
        values,
        compiler,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA products")
def test_wide_bilinear_input_adjoints(monkeypatch):
    from equivariantx.ops.common.cuda import bilinear

    launches = set()
    generate = bilinear.inputs_source

    def source(dtype, metadata, branches, roles, large_batch):
        result = generate(dtype, metadata, branches, roles, large_batch)
        launches.add((roles, result[-1]))
        return result

    monkeypatch.setattr(bilinear, "inputs_source", source)
    nodes = 1025
    tp = o3.TensorProduct(
        "256x1o",
        "256x0e",
        "256x1o",
        [(0, 0, 0, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    linear = o3.Linear("256x1o", "64x1o", internal_weights=False, shared_weights=False)
    module = BilinearContraction(tp, linear, backend="cuda").cuda()
    types = torch.arange(nodes, device="cuda") % 2
    inputs = tuple(
        (0.3 * torch.randn(shape, device="cuda", dtype=torch.float32)).requires_grad_()
        for shape in ((nodes, 768), (nodes, 256), (nodes, 256), (2, 256 * 64))
    )
    seed = torch.randn(nodes, 192, device="cuda", dtype=torch.float32).requires_grad_()
    direction = torch.randn_like(inputs[0])

    def derivatives(output, values, cotangent):
        first = torch.autograd.grad(output, values, cotangent, create_graph=True)
        mixed = torch.autograd.grad(first[0], (*values[1:], cotangent), direction)
        return output, *first, *mixed

    actual = derivatives(module(*inputs, types), inputs, seed)
    assert ((0, 1, 4), 256) in launches
    x, y, gates, weights = values = tuple(
        v.detach().double().requires_grad_() for v in inputs
    )
    product = tp.cuda().double()(x, y, gates)
    linear = linear.cuda().double()
    reference = torch.where(
        types[:, None] == 0, linear(product, weights[0]), linear(product, weights[1])
    )
    expected = derivatives(reference, values, seed.detach().double().requires_grad_())
    for result, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(result.double(), ref, atol=2e-5, rtol=3e-5)
