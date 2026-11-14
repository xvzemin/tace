"""Segmented normalization against independent node-wise sums."""

import pytest
import torch

from equivariantx.conv.attention import graph_softmax
from equivariantx.ops import config, get_kernel
from equivariantx.ops.common.torch.attention import normalized_exp_reference


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("edges", [0, 7])
@pytest.mark.parametrize("weight_heads", [1, 3])
def test_normalization_derivatives(device, edges, weight_heads, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(config, "TORCH_ATTENTION_MIN_CPU_ELEMENTS", 0)
    scores = torch.randn(
        edges, 3, device=device, dtype=torch.float64, requires_grad=True
    )
    weight = torch.rand(edges, weight_heads, device=device, dtype=torch.float64)
    weight[:1] = 0
    weight.requires_grad_()
    target = torch.arange(edges, device=device) % 3
    eps = 1e-5

    def reference(scores, weight):
        result = scores * 0 + weight * 0
        for node in range(5):
            index = torch.where(target == node)[0]
            if index.numel():
                value = scores[index]
                value = (value - value.detach().amax(0, keepdim=True)).exp()
                value = value * weight[index]
                result = result.index_copy(
                    0, index, value / (value.sum(0, keepdim=True) + eps)
                )
        return result

    def optimized(scores, weight):
        return graph_softmax(scores, target, 5, weight, eps)

    actual, expected = optimized(scores, weight), reference(scores, weight)
    torch.testing.assert_close(actual, expected)
    for _ in range(4):
        seed = torch.randn_like(actual) * 0.1
        gradients = [
            torch.autograd.grad(
                (value.sin() * seed).sum(), (scores, weight), create_graph=True
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
        actual, expected = [
            torch.cat([value.flatten() for value in values]) for values in gradients
        ]
    inputs = scores, weight
    tangents = tuple(torch.randn_like(value) for value in inputs)
    torch.testing.assert_close(
        torch.func.jvp(optimized, inputs, tangents),
        torch.func.jvp(reference, inputs, tangents),
    )
    batched = tuple(torch.stack((value, value * 0.8)) for value in inputs)
    expected = torch.stack([reference(*values) for values in zip(*batched)])
    torch.testing.assert_close(torch.vmap(optimized)(*batched), expected)


@pytest.mark.parametrize("active", [(True, False), (False, True), (True, True)])
def test_reference_switch_and_active_inputs(active, monkeypatch):
    monkeypatch.setattr(config, "TORCH_ATTENTION_MIN_CPU_ELEMENTS", 0)
    torch.manual_seed(8)
    scores = torch.randn(7, 3, dtype=torch.float64, requires_grad=active[0])
    weight = torch.zeros(7, 1, dtype=torch.float64, requires_grad=active[1])
    target = torch.arange(7) % 3
    inputs = tuple(value for value in (scores, weight) if value.requires_grad)
    actual = graph_softmax(scores, target, 5, weight, eps=0.1)
    with monkeypatch.context() as reference:
        reference.setattr(config, "TORCH_REFERENCE", True)
        expected = graph_softmax(scores, target, 5, weight, eps=0.1)
        assert (
            get_kernel("graph_normalized_exp").__wrapped__ is normalized_exp_reference
        )
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        actual, expected = [
            torch.cat(
                [
                    grad.flatten()
                    for grad in torch.autograd.grad(
                        value.sin().sum(), inputs, create_graph=True
                    )
                ]
            )
            / 100
            for value in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


def test_dynamic_normalization():
    torch.compiler.reset()
    graphs = []

    def compiler(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def function(scores, weight, target):
        return graph_softmax(scores, target, 5, weight)

    run = torch.compile(function, backend=compiler, fullgraph=True, dynamic=True)
    for edges in (7, 11, 9):
        scores = torch.randn(edges, 3, dtype=torch.float64, requires_grad=True)
        weight = torch.rand(edges, 1, dtype=torch.float64, requires_grad=True)
        target = torch.arange(edges) % 3
        inputs = scores, weight, target
        for value in inputs:
            torch._dynamo.mark_dynamic(value, 0, min=2)
        actual = run(*inputs)
        expected = normalized_exp_reference(scores, weight, target, 5, 1e-16) * weight
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().sum(), inputs[:2]),
            torch.autograd.grad(expected.square().sum(), inputs[:2]),
        )
    assert len(graphs) == 1


def test_forward_ad():
    inputs = (
        torch.randn(7, 3, dtype=torch.float64, requires_grad=True),
        torch.rand(7, 1, dtype=torch.float64, requires_grad=True),
    )
    tangents = tuple(torch.randn_like(value) for value in inputs)
    target = torch.arange(7) % 3
    with torch.autograd.forward_ad.dual_level():
        scores, weight = tuple(
            torch.autograd.forward_ad.make_dual(value, tangent)
            for value, tangent in zip(inputs, tangents)
        )
        actual = graph_softmax(scores, target, 5, weight, 1e-5)
        expected = normalized_exp_reference(scores, weight, target, 5, 1e-5) * weight
        torch.testing.assert_close(
            torch.autograd.forward_ad.unpack_dual(actual),
            torch.autograd.forward_ad.unpack_dual(expected),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor package")
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_force_gradient_aoti(backend, tmp_path, double_precision):
    import subprocess
    import sys

    from torch.fx.experimental.proxy_tensor import make_fx

    x = torch.randn(11, 4, device="cuda", requires_grad=True)
    projection = (torch.randn(4, 3, device="cuda") * 0.2).requires_grad_()
    target = torch.arange(11, device="cuda") % 3

    def joint(x, projection, target, reference=False):
        scores = torch.nn.functional.silu(x @ projection)
        weight = torch.nn.functional.softplus(x[:, :1])
        value = (
            normalized_exp_reference(scores, weight, target, 5, 1e-5) * weight
            if reference
            else graph_softmax(scores, target, 5, weight, 1e-5, backend=backend)
        )
        energy = (value * x[:, :3].sin()).square().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy + (force - 0.01).square().sum()
        return loss, force, torch.autograd.grad(loss, projection)[0]

    inputs = x, projection, target
    graph = make_fx(lambda *values: joint(*values))(*inputs)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    exported = torch.export.export(graph, tuple(value.detach() for value in inputs))
    package = tmp_path / "attention.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    for scale, index in ((1.0, target), (0.7, target.roll(1))):
        changed = tuple(
            (value.detach() * scale).requires_grad_() for value in inputs[:2]
        )
        cases.append(
            {
                "inputs": (*[value.detach().cpu() for value in changed], index.cpu()),
                "expected": tuple(
                    value.detach().cpu() for value in joint(*changed, index, True)
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
    torch.testing.assert_close(tuple(value.cpu() for value in actual), case['expected'], atol=1e-9, rtol=1e-9)
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(package), str(data)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
