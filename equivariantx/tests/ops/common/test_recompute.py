"""Tests for the corresponding EQX source module."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.recompute import Replay


def test_replay_tf32_cache():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.common.recompute import Replay

    previous = torch.backends.cuda.matmul.allow_tf32
    program = Replay(lambda values, create_graph: (values[0] @ values[1],))
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    try:
        for enabled in (False, True, False):
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.testing.assert_close(program(x, y)[0], x @ y)
        assert len(program.graphs) == 2
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_replay_functional_and_tracing(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    def function(x, w):
        return [(x @ w).sin()]

    program = Replay(lambda values, graph: function(*values), capture=False)
    values = tuple(
        torch.randn(*shape, device=device, dtype=torch.float64, requires_grad=True)
        for shape in ((7, 3), (3, 5))
    )
    directions = tuple(torch.randn_like(value) for value in values)
    torch.testing.assert_close(
        torch.func.jvp(program, values, directions),
        torch.func.jvp(function, values, directions),
    )
    batched = tuple(torch.stack((value, value * 0.8)) for value in values)
    torch.testing.assert_close(
        torch.vmap(program)(*batched), torch.vmap(function)(*batched)
    )

    def joint(x, w):
        result = program(x, w)[0].square().mean()
        outputs = [result]
        for _ in range(4):
            gradients = torch.autograd.grad(result, (x, w), create_graph=True)
            outputs.extend(gradients)
            result = (
                sum(
                    (gradient * direction).sum()
                    for gradient, direction in zip(gradients, directions)
                )
                / 10
            )
        return outputs

    graph = make_fx(joint)(*values)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    compiled = torch.compile(graph, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(*values), joint(*values), atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor package")
def test_streaming_attention_aoti(tmp_path):
    import os
    import subprocess
    import sys

    from equivariantx.conv.attention import StreamingGraphAttention, graph_softmax

    def shapes(inputs):
        return tuple(
            inputs[0].new_empty(shape) for shape in ((4, 1, 4), (4, 2), (4, 2))
        )

    def edge_function(inputs):
        return inputs[1] @ inputs[6], (inputs[1] @ inputs[7]).unsqueeze(1)

    program = StreamingGraphAttention(
        edge_function,
        shapes,
        {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        2,
        3,
        4,
        5,
        3,
        1e-8,
        backend="cuda",
    )
    x = torch.randn(7, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    score = torch.randn(3, 2, device="cuda", dtype=x.dtype, requires_grad=True)
    value = torch.randn(3, 4, device="cuda", dtype=x.dtype, requires_grad=True)
    target = torch.arange(7, device="cuda") % 3
    normalizer = torch.rand(7, 1, device="cuda", dtype=x.dtype)
    weight = torch.rand_like(normalizer)
    valid = torch.arange(7, device="cuda") % 4 != 0

    def joint(x, score, value, reference=False):
        if reference:
            logits = torch.where(valid[:, None], x @ score, -torch.inf)
            scale = graph_softmax(logits, target, 4, normalizer, 1e-8) * weight
            messages = ((x @ value).reshape(7, 2, 2) * scale[:, :, None]).reshape(
                7, 1, 4
            )
            result = x.new_zeros((4, 1, 4)).index_add(0, target, messages)
        else:
            result = program(
                x.new_empty((4, 1)), x, target, normalizer, weight, valid, score, value
            )[0]
        energy = result.square().mean()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        gradients = torch.autograd.grad(
            energy + (force - 0.1).square().sum(), (score, value)
        )
        return energy, force, *gradients

    inputs = x, score, value
    torch.testing.assert_close(
        joint(*inputs), joint(*inputs, reference=True), atol=1e-9, rtol=1e-9
    )
    graph = make_fx(lambda x, score, value: joint(x, score, value))(*inputs)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    exported = torch.export.export(graph, tuple(value.detach() for value in inputs))
    package = tmp_path / "attention.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    for scale in (1.0, 0.8):
        values = tuple((value.detach() * scale).requires_grad_() for value in inputs)
        cases.append(
            {
                "inputs": tuple(value.detach().cpu() for value in values),
                "expected": tuple(
                    value.detach().cpu() for value in joint(*values, reference=True)
                ),
            }
        )
    data = tmp_path / "cases.pt"
    torch.save(cases, data)
    script = """
import sys
import torch
runner = torch._inductor.aoti_load_package(sys.argv[1])
for case in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = runner(*(value.cuda() for value in case['inputs']))
    for value, expected in zip(actual, case['expected']):
        torch.testing.assert_close(value.cpu(), expected, atol=1e-9, rtol=1e-9)
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(package), str(data)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
