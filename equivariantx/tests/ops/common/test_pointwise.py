"""Shared row-local expression evaluation and repeated-column adjoints."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.pointwise import evaluate, torch_evaluate
from equivariantx.utils.program import Program


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("shape", [(3, 3), (2, 4, 3), (0, 3)])
def test_row_expressions(backend, shape, device):
    if backend == "cuda" and device != "cuda":
        pytest.skip("Native pointwise kernels require CUDA.")
    graph = Program()
    source = graph.input(0, "edge", 3)
    a, b = graph.gather(source, (2, 0, 0)), graph.gather(source, (1, 2, 1))
    output = graph.binary("mul", a, graph.binary("add", a, b))
    metadata = repr((tuple(graph.nodes), ((output, 0, "edge"),)))
    value = torch.randn(shape, dtype=torch.float64, device=device, requires_grad=True)
    native = evaluate if backend == "cuda" else torch_evaluate
    run = lambda x: native(x, metadata, 3)
    reference = lambda x: x[..., [2, 0, 0]] * (x[..., [2, 0, 0]] + x[..., [1, 2, 1]])
    torch.testing.assert_close(run(value), reference(value), atol=2e-11, rtol=2e-11)
    if value.numel():
        assert torch.autograd.gradcheck(run, (value,))
        assert torch.autograd.gradgradcheck(run, (value,))
    traced = make_fx(run)(value)
    assert not any(str(node.target).startswith("eqx.") for node in traced.graph.nodes)
    torch.testing.assert_close(traced(value), reference(value), atol=2e-11, rtol=2e-11)
