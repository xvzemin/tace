"""Native normalization, arbitrary derivatives and independent AOTI packages."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops.common.torch.attention import normalized_exp_reference
from equivariantx.tests.utils.testing import check_forward_ad, package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native attention requires CUDA"
)


def force_loss(kernel, scores, parameters, target):
    """Differentiate position-dependent edge weights and a shared parameter map."""
    weights = (scores.unsqueeze(-1) * parameters).sum(-2).sigmoid() + 0.5
    values = kernel(scores, weights, target, 5, 1e-5) * weights
    energy = (values * scores.sin()).sum() / scores.shape[1]
    force = -torch.autograd.grad(energy, scores, create_graph=True)[0]
    loss = (energy - 0.2).square() + (force - 0.01).square().sum()
    return energy, force, loss, torch.autograd.grad(loss, parameters)[0]


def dynamic_force_graph(scores, parameters, target):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    from equivariantx.ops.common.cuda.attention import evaluate

    edges = ShapeVar("edges", min=0, max=64, optimization_hint=7)
    shapes = dict(
        x=TensorSpec((edges, None)),
        p=TensorSpec((None, None)),
        t=TensorSpec((edges,)),
    )
    graph = make_fx(
        lambda x, p, t: force_loss(evaluate, x, p, t),
        tracing_mode="fake",
        dynamic_shapes=ShapesSpec(ParamsSpec(shapes)),
    )(scores, parameters, target)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    assert any(
        str(node.target) == "eqx.attention_normalized_exp.default"
        for node in graph.graph.nodes
    )
    return graph


@pytest.mark.parametrize("heads,weight_heads", [(3, 1), (8, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unbacked_force_loss(heads, weight_heads, dtype):
    parameters = (
        torch.randn(heads, weight_heads, device="cuda", dtype=dtype) * 0.1
    ).requires_grad_()
    scores = (torch.randn(7, heads, device="cuda", dtype=dtype) * 0.2).requires_grad_()
    target = torch.arange(7, device="cuda") % 3
    graph = dynamic_force_graph(scores, parameters, target)
    for edges in (7, 13, 0, 1, 33):
        scores = torch.randn(edges, heads, device="cuda", dtype=dtype) * 0.2
        target = torch.arange(edges, device="cuda") % 3
        actual = graph(scores, parameters.detach(), target)
        expected = force_loss(
            normalized_exp_reference,
            scores.cpu().double().requires_grad_(),
            parameters.detach().cpu().double().requires_grad_(),
            target.cpu(),
        )
        tolerance = 3e-5 if dtype == torch.float32 else 3e-10
        torch.testing.assert_close(
            tuple(value.detach().cpu().double() for value in actual),
            expected,
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dynamic_force_aoti_updates(dtype, tmp_path):
    parameters = torch.randn(3, 1, dtype=dtype) * 0.1
    initial = parameters.clone()
    records = []
    for edges in (7, 13, 3, 33):
        scores = torch.randn(edges, 3, dtype=dtype) * 0.2
        target = torch.arange(edges) % 2
        expected = force_loss(
            normalized_exp_reference,
            scores.double().requires_grad_(),
            parameters.double().requires_grad_(),
            target,
        )
        expected = tuple(value.detach() for value in expected)
        before = parameters
        parameters = parameters - 0.01 * expected[-1].to(dtype)
        assert not torch.equal(parameters, before)
        records.append(
            dict(scores=scores, target=target, expected=expected, updated=parameters)
        )
    inputs = (
        records[0]["scores"].cuda().requires_grad_(),
        initial.cuda().requires_grad_(),
        records[0]["target"].cuda(),
    )
    graph = dynamic_force_graph(*inputs)
    edges = torch.export.Dim("edges", min=2, max=64)
    exported = torch.export.export(
        graph,
        tuple(value.detach() for value in inputs),
        dynamic_shapes=({0: edges}, {}, {0: edges}),
        strict=True,
    )
    package = tmp_path / "dynamic_attention.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    data = tmp_path / "records.pt"
    torch.save(dict(initial=initial, cases=records), data)
    script = """
import sys
import torch
records = torch.load(sys.argv[2], weights_only=True)
import equivariantx
run = torch._inductor.aoti_load_package(sys.argv[1])
parameters = records['initial'].cuda()
tolerance = 3e-5 if parameters.dtype == torch.float32 else 3e-10
for case in records['cases']:
    scores, target = case['scores'].cuda(), case['target'].cuda()
    actual = tuple(run(scores, parameters, target))
    torch.testing.assert_close(tuple(v.cpu().double() for v in actual), case['expected'], atol=tolerance, rtol=tolerance)
    assert actual[1].abs().max() > 0 and actual[-1].abs().max() > 0
    assert len({v.untyped_storage().data_ptr() for v in (*actual, scores, parameters, target)}) == 7
    predictions = tuple(v.clone() for v in actual[:-1])
    before = parameters.clone()
    actual[-1].mul_(0.01)
    torch.testing.assert_close(actual[:-1], predictions, atol=0, rtol=0)
    torch.testing.assert_close(parameters, before, atol=0, rtol=0)
    parameters.sub_(actual[-1])
    assert not torch.equal(parameters, before)
    torch.testing.assert_close(parameters.cpu(), case['updated'], atol=tolerance, rtol=tolerance)
for count in (0, 1, 65):
    try:
        run(scores.new_zeros(count, 3), parameters, target.new_zeros(count))
    except RuntimeError:
        pass
    else:
        raise AssertionError(f'Accepted out-of-contract edges={count}')
assert 'tace' not in sys.modules
print('Four sequential dynamic attention updates passed with registered EQX operators.')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            package_script(script),
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={**os.environ, "AOTI_RUNTIME_CHECK_INPUTS": "1"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    (tmp_path / "loader.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("heads,weight_heads", [(1, 1), (3, 1), (8, 8), (33, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_derivatives(heads, weight_heads, dtype):
    from equivariantx.ops.common.cuda.attention import evaluate

    torch.manual_seed(7)
    scores = (
        torch.randn(17, heads * 2, device="cuda", dtype=dtype)[:, ::2] * 0.2
    ).requires_grad_()
    weight = torch.rand(17, weight_heads, device="cuda", dtype=dtype) + 0.5
    weight[::3] = 0
    weight.requires_grad_()
    target = torch.arange(34, device="cuda")[::2] % 5
    inputs = scores, weight
    actual = evaluate(*inputs, target, 7, 1e-5)
    expected = normalized_exp_reference(*inputs, target, 7, 1e-5)
    tolerance = 3e-5 if dtype == torch.float32 else 3e-10
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        actual, expected = [
            torch.cat(
                [
                    gradient.flatten()
                    for gradient in torch.autograd.grad(
                        value.sin().sum(), inputs, create_graph=True
                    )
                ]
            )
            * 0.1
            for value in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_forward_ad(double_precision):
    from equivariantx.ops.common.cuda.attention import evaluate

    values = (
        torch.randn(7, 3, device="cuda", requires_grad=True),
        (torch.rand(7, 1, device="cuda") + 0.5).requires_grad_(),
    )
    target = torch.arange(7, device="cuda") % 3
    check_forward_ad(
        lambda *args: evaluate(*args, target, 5, 1e-5),
        lambda *args: normalized_exp_reference(*args, target, 5, 1e-5),
        values,
    )


def test_dynamic_inductor(double_precision):
    from equivariantx.ops.common.cuda.attention import evaluate

    compiled = torch.compile(evaluate, fullgraph=True, dynamic=True)
    for edges in (7, 13, 0):
        scores = torch.randn(edges, 3, device="cuda", requires_grad=True)
        weight = (torch.rand(edges, 1, device="cuda") + 0.5).requires_grad_()
        target = torch.arange(edges, device="cuda") % 3
        actual = compiled(scores, weight, target, 5, 1e-5)
        expected = normalized_exp_reference(scores, weight, target, 5, 1e-5)
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sin().sum(), (scores, weight)),
            torch.autograd.grad(expected.sin().sum(), (scores, weight)),
            atol=3e-10,
            rtol=3e-10,
        )
