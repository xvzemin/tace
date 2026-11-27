"""Numerical derivatives and standalone compilation for the matching kernel."""

import pytest
import torch

from equivariantx.o3.wigner6j import Wigner6jTensorProductConv
from equivariantx.ops import config
from equivariantx.tests.o3.wigner6j.testing import (
    check_derivatives,
    check_dynamic_convolution,
    check_package,
    problem,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native Wigner-6j requires CUDA"
)


def test_fused_node_reduction(monkeypatch):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", True)
    monkeypatch.setattr(config, "WIGNER6J_INLINE", False)
    layer, values, index, _ = problem("convolution", "cuda", "cuda")

    def materialized(*args, **kwargs):
        raise AssertionError(
            "The fused convolution must not materialize edge messages."
        )

    monkeypatch.setattr(layer.schedule, "evaluate", materialized)
    x1, x2, x3, w1, w2 = values
    actual = layer(x1, x2, x3, index, w1, w2)
    expected = layer.forward_reference(x1, x2, x3, index, w1, w2)
    torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)


def _standard_problem():
    torch.manual_seed(881)
    op = (
        Wigner6jTensorProductConv(
            "2x1o",
            "1o",
            "1e",
            "2x0e+2x1e",
            "2x1e+2x1e",
            [(0, 0, i, "uvu", True) for i in range(2)],
            [(i % 2, 0, (i // 2) % 2, "uvu", True, 0.3 + 0.02 * i) for i in range(32)],
            shared_weights=False,
            internal_weights=False,
            backend="cuda",
        )
        .cuda()
        .double()
    )
    assert not any(name.startswith("standard") for name in op.state_dict())
    assert not tuple(op.parameters())
    operands = tuple(
        torch.randn(shape, device="cuda", dtype=torch.float64).requires_grad_()
        for shape in (
            (3, 6),
            (7, 3),
            (3, 3),
            (7, op.weight_numel1),
            (7, op.weight_numel2),
        )
    )
    edges = torch.tensor([[0, 2, 1, 0, 2, 2, 1], [3, 0, 0, 1, 1, 3, 3]], device="cuda")
    return op, operands, edges


def test_bounded_standard_tree_derivatives(monkeypatch):
    """Repeated independent paths, both weights and the extra node input stay live."""
    from equivariantx.ops.cuda.o3.wigner6j import convolution

    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", False)
    monkeypatch.setattr(config, "WIGNER6J_INLINE", False)
    op, operands, edges = _standard_problem()
    x, y, spin, w1, w2 = operands
    cot = torch.randn(5, op.output_dim, device="cuda", dtype=x.dtype)
    direction = tuple(torch.randn_like(a) for a in operands)

    def run(fn):
        out = fn(x, y, spin, edges, w1, w2, 5)
        first = torch.autograd.grad((out * cot).sum(), operands, create_graph=True)
        second = torch.autograd.grad(
            sum((g * d).sum() for g, d in zip(first, direction)), operands
        )
        assert all(g.count_nonzero() for g in (*first, *second))
        return out, first, second

    reference = run(op.forward_reference)
    seen = []
    calls = []
    # Force two edges per tile without building a huge problem.
    monkeypatch.setattr(
        convolution, "STANDARD_TILE_BYTES", 2 * op.irreps_mid.dim * x.element_size()
    )
    with (
        op.standard12.register_forward_hook(
            lambda _, args, out: seen.append(out.shape[0])
        ),
        op.standard123.register_forward_hook(
            lambda _, args, out: calls.append(out.shape[0])
        ),
    ):
        torch.testing.assert_close(run(op), reference, rtol=5e-10, atol=3e-10)
    assert seen == [2, 2, 2, 1] and calls == [5] * 4
    # The original standard implementation also accepts int32 graph indices.
    with op.standard12.register_forward_pre_hook(
        lambda *_: pytest.fail("int32 native index")
    ):
        torch.testing.assert_close(
            op(x, y, spin, edges.int(), w1, w2, 5), reference[0], rtol=5e-10, atol=3e-10
        )
    assert not convolution.use_standard_contractions(op, (x, y, spin, w1[:1], w2))
    assert not convolution.use_standard_contractions(
        op, (x, y[:0], spin, w1[:0], w2[:0])
    )


def test_standard_tree_broadcast_and_empty_edges_keep_original_schedule(monkeypatch):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", False)
    monkeypatch.setattr(config, "WIGNER6J_INLINE", False)
    op, (x, y, spin, w1, w2), edges = _standard_problem()
    calls = []
    original = op.schedule.evaluate

    def observe(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(op.schedule, "evaluate", observe)
    for count, a, b in [(7, w1[:1], w2), (7, w1, w2[:1]), (0, w1[:0], w2[:0])]:
        inputs = (x, y[:count], spin, edges[:, :count])
        actual = op(*inputs, a, b, 5)
        expected = (
            op.forward_reference(*inputs, a.expand(count, -1), b.expand(count, -1), 5)
            if count
            else x.new_zeros(5, op.output_dim)
        )
        torch.testing.assert_close(actual, expected, rtol=5e-10, atol=3e-10)
    assert len(calls) == 3


@pytest.mark.parametrize("dynamic", [False, True])
def test_standard_tree_export_operators_and_values(monkeypatch, tmp_path, dynamic):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", False)
    monkeypatch.setattr(config, "WIGNER6J_INLINE", False)
    op, (x, y, spin, w1, w2), edges = _standard_problem()

    class Layer(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x, y, spin, edges, w1, w2):
            return self.op(x, y, spin, edges, w1, w2, 5)

    args = (x, y, spin, edges, w1, w2)
    dim = torch.export.Dim("edges", min=2)
    shapes = ({}, {0: dim}, {}, {1: dim}, {0: dim}, {0: dim}) if dynamic else None
    graph = torch.export.export(Layer(op), args, dynamic_shapes=shapes, strict=True)
    targets = [n.target for n in graph.graph.nodes if n.op == "call_function"]
    if dynamic:
        # A single streamed tree accepts unbounded E; it must not materialize
        # the intermediate per-edge messages or specialize the tile count.
        assert [t for t in targets if getattr(t, "namespace", None) == "eqx"] == [
            torch.ops.eqx.edge_program.default
        ]
    else:
        assert targets.count(torch.ops.eqx.o3_contraction.default) == 2
    path = tmp_path / "standard.pt2"
    torch.export.save(graph, path)
    exported = torch.export.load(path).module()
    from equivariantx.ops.compiler import compile_and_package

    compiled_path = compile_and_package(
        graph, package_path=str(tmp_path / "standard_aoti.pt2")
    )
    compiled = torch._inductor.aoti_load_package(compiled_path)
    reference = op.cpu().forward_reference  # Independent e3nn standard tree.
    for count in (7, 4, 11) if dynamic else (7,):
        rows = torch.arange(count, device=x.device) % y.size(0)
        inputs = (x, y[rows], spin, edges.index_select(1, rows), w1[rows], w2[rows])
        cpu = tuple(v.detach().cpu().requires_grad_(v.requires_grad) for v in inputs)
        expected = reference(*cpu, 5)
        # The AOTI inference runner itself has no autograd tape.
        with torch.no_grad():
            torch.testing.assert_close(
                compiled(*inputs).cpu(), expected.detach(), rtol=5e-10, atol=3e-10
            )
        if dynamic:
            out = exported(*inputs)
            torch.testing.assert_close(out.cpu(), expected, rtol=5e-10, atol=3e-10)
            cot = torch.randn_like(expected)
            directions = [torch.randn_like(v) for v in cpu if v.requires_grad]

            def derivatives(output, values):
                variables = tuple(v for v in values if v.requires_grad)
                first = torch.autograd.grad(
                    output, variables, cot.to(output), create_graph=True
                )
                second = torch.autograd.grad(
                    sum((g * d.to(g)).sum() for g, d in zip(first, directions)),
                    variables,
                )
                return [g.detach().cpu() for g in (*first, *second)]

            grads = derivatives(expected, cpu)
            assert all(g.count_nonzero() for g in grads)
            torch.testing.assert_close(
                derivatives(out, inputs), grads, rtol=5e-10, atol=3e-10
            )


def test_compensated_node_reduction():
    from equivariantx.utils.module import default_dtype

    dtype = torch.float32
    with default_dtype(dtype):
        layer = Wigner6jTensorProductConv(
            "0e",
            "0e",
            "0e",
            "0e",
            "0e",
            [(0, 0, 0, "uvu", False)],
            [(0, 0, 0, "uvu", False)],
            backend="cuda",
            recouple=True,
        ).cuda()
    large = 4 / torch.finfo(dtype).eps
    attrs = torch.tensor([large, 1, 1, 1, 1, 1, 1, -large], device="cuda", dtype=dtype)
    attrs = attrs.repeat(128).reshape(-1, 1).requires_grad_()
    nodes = torch.ones(1, 1, device="cuda", dtype=dtype, requires_grad=True)
    edges = torch.zeros(2, attrs.shape[0], device="cuda", dtype=torch.long)
    result = layer(nodes, attrs, nodes, edges)
    torch.testing.assert_close(result, torch.full_like(result, 768), atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(result.sum(), attrs)[0], torch.ones_like(attrs)
    )


@pytest.mark.parametrize("recouple", [False, True])
def test_fifth_derivatives_and_transforms(monkeypatch, recouple):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    check_derivatives("convolution", "cuda", "cuda")


@pytest.mark.parametrize("recouple", [None, True])
def test_joint_training_package(tmp_path, monkeypatch, recouple):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    check_package("convolution", tmp_path, "cuda")


def test_dynamic_inductor(monkeypatch):
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    layer, values, index, _ = problem("convolution", "cuda", "cuda")
    compiled = torch.compile(layer, fullgraph=True, dynamic=True)
    for count, nodes in ((4, 3), (7, 5), (0, 2)):
        _, values, index, _ = problem("convolution", "cuda", "cuda", count, nodes)
        x1, x2, x3, w1, w2 = values
        actual = compiled(x1, x2, x3, index, w1, w2, nodes + 1)
        expected = layer.forward_reference(x1, x2, x3, index, w1, w2, nodes + 1)
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("shared", [False, True])
def test_unbacked_joint_aoti(tmp_path, monkeypatch, shared):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", None)
    check_dynamic_convolution(tmp_path, "cuda", "cuda", shared)


@pytest.mark.parametrize("recouple", [False, True])
def test_shared_scalar_derivatives(monkeypatch, recouple):
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    check_derivatives(
        "convolution", "cuda", "cuda", shared=True, multiplicity=1, edge_degree=0
    )


@pytest.mark.parametrize("recouple", [None, True])
def test_shared_scalar_package(tmp_path, monkeypatch, recouple):
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    check_package(
        "convolution", tmp_path, "cuda", shared=True, multiplicity=1, edge_degree=0
    )
