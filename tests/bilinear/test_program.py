"""Functional derivatives of fused product, gates and expert maps."""

import pytest
import torch
from e3nn import o3

from equivariantx.tests.ops.testing import assert_functional_derivatives
from tace.ops.bilinear import BilinearContraction


def test_factor_groups_separates_projection_operands_and_slots():
    from tace.ops.bilinear_kernel import factor_groups

    def group(x, dy, slot):
        return (x, 1, 4), ((0, (x, 1, 2, dy, 4), 2, slot),)

    left, right = group(7, 6, 1), group(8, 6, 1)
    assert factor_groups((left, right), 3) == ((left[1], (right[0],)),)
    for different in (group(8, 9, 1), group(8, 6, 0)):
        assert all(not extra for _, extra in factor_groups((left, different), 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA products")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_dirs", [2, 3])
def test_factored_coefficient_mixed_adjoints(dtype, n_dirs, monkeypatch):
    from equivariantx.ops import config
    from tace.ops import bilinear_kernel as bilinear

    torch.manual_seed(521)
    fused = {}
    for name in ("weights_source", "forward_source"):
        original = getattr(bilinear, name)

        def generate(dtype, meta, terms, extra=(), name=name, original=original):
            fused.setdefault(name, set()).add(len(extra))
            return original(dtype, meta, terms, extra)

        monkeypatch.setattr(bilinear, name, generate)
    irreps = "64x0e+64x1o"
    tp = o3.TensorProduct(
        irreps,
        irreps,
        irreps,
        [(0, 0, 0, "uuu", True), (1, 1, 0, "uuu", True), (0, 1, 1, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    lin = o3.Linear(
        "32x0e+32x1o", "32x0e+32x1o", internal_weights=False, shared_weights=False
    )
    shared = o3.Linear(irreps, irreps, internal_weights=False, shared_weights=False)
    native, ref = [
        BilinearContraction(tp, lin, 2, backend, shared).to(device="cuda", dtype=dtype)
        for backend in ("cuda", "torch")
    ]
    n = 13
    xs = tuple(
        (0.2 * torch.randn(sz, device="cuda", dtype=dtype)).requires_grad_()
        for sz in (
            (n, native.input_dim),
            (n, native.base_dim),
            (n, native.gate_dim),
            (3, 2, lin.weight_numel),
            (shared.weight_numel,),
        )
    )
    types = torch.arange(n, device="cuda") % 3

    def run(m, seed, directions):
        y = m(*xs[:4], types, xs[4])
        gs = torch.autograd.grad(y, xs[:n_dirs], seed, create_graph=True)
        more = torch.autograd.grad(gs, (*xs, seed), directions, create_graph=True)
        third = torch.autograd.grad(sum(t.square().mean() for t in more), (*xs, seed))
        return y, *gs, *more, *third

    seed = torch.randn_like(native(*xs[:4], types, xs[4])).requires_grad_()
    ds = tuple(torch.randn_like(x) for x in xs[:n_dirs])
    a = run(native, seed, ds)
    assert all(
        n_dirs - 1 in fused[name] for name in ("weights_source", "forward_source")
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expect = run(ref, seed, ds)
    tol = 2e-5 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(a, expect, atol=tol, rtol=tol)


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
    from tace.ops import bilinear_kernel as bilinear

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
@pytest.mark.parametrize("inline_limit", [0, 256])
def test_force_training_package(inline_limit, tmp_path, double_precision, monkeypatch):
    from equivariantx.tests.ops.common.test_polynomial import (
        check_force_gradient_updates,
    )

    check_force_gradient_updates(
        "bilinear", False, inline_limit, tmp_path, monkeypatch, BilinearContraction
    )
