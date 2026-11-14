"""Native O3 dispatch and projection schedules under higher-order AD."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops import config
from equivariantx.ops.common.torch.o3 import reference
from equivariantx.ops.common.o3 import contraction
from equivariantx.tests.ops.common.torch.test_o3 import convolution_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA O3")


def projected_case(width, geometric=True, edges=7, shared=False):
    _, metadata, values, source, target = convolution_case(geometric, edges, shared)
    values = (
        values[0],
        (torch.randn(values[1].shape[0], width, device="cuda") * 0.2).requires_grad_(),
        (
            torch.randn(width, values[2].shape[1], device="cuda") / width**0.5
        ).requires_grad_(),
        *values[3:],
    )
    program = repr(((tuple(range(len(values))), False, ((4, 0),)),))
    return metadata, program, values, source, target


@pytest.mark.parametrize("geometric", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("edges", [0, 7])
def test_projected_derivatives(geometric, shared, edges, double_precision, monkeypatch):
    monkeypatch.setattr(config, "RADIAL_PROJECT_MIN_WIDTH", 4)
    metadata, program, values, source, target = projected_case(
        4, geometric, edges, shared
    )
    actual = contraction(metadata, program, source, target, list(values))[0]
    expected = reference(metadata, program, source, target, values)[0]
    active = (*values[:4], *values[5:])
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        gradients = [
            torch.autograd.grad(
                output.sin().sum(), active, create_graph=True, allow_unused=True
            )
            for output in (actual, expected)
        ]
        gradients = [
            tuple(torch.zeros_like(x) if g is None else g for x, g in zip(active, gs))
            for gs in gradients
        ]
        torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
        actual, expected = [sum(g.sum() for g in gs) / 20 for gs in gradients]
        if not actual.requires_grad:
            break


@pytest.mark.parametrize(
    "width,limit,expected",
    [(64, 256 << 20, 0), (128, 256 << 20, 0), (1024, 256 << 20, 1), (1024, 1, 0)],
)
def test_projection_policy(width, limit, expected, monkeypatch, double_precision):
    monkeypatch.setattr(config, "RADIAL_PROJECT_MAX_BYTES", limit)
    metadata, program, values, source, target = projected_case(width)
    graph = make_fx(
        lambda *values: contraction(metadata, program, source, target, list(values))
    )(*values)
    assert (
        sum(node.target == torch.ops.aten.mm.default for node in graph.graph.nodes)
        == expected
    )
    torch.testing.assert_close(
        graph(*values),
        reference(metadata, program, source, target, values),
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("compiler", ["inductor", "aoti"])
def test_projected_joint(compiler, tmp_path, double_precision):
    metadata, program, values, source, target = projected_case(1024)

    def joint(function, *values):
        output = function(metadata, program, source, target, list(values))[0]
        energy = output.square().mean()
        force = -torch.autograd.grad(energy, values[5], create_graph=True)[0]
        loss = energy + (force - 0.1).square().sum()
        return loss, force, *torch.autograd.grad(loss, values[:3])

    graph = make_fx(lambda *values: joint(contraction, *values))(*values)
    inputs = tuple(value.detach() for value in values)
    if compiler == "aoti":
        exported = torch.export.export(graph, inputs)
        package = torch._inductor.aoti_compile_and_package(
            exported, package_path=str(tmp_path / "projected.pt2")
        )
        run = torch._inductor.aoti_load_package(package)
    else:
        run = torch.compile(graph, fullgraph=True)
    for scale in (1.0, 0.7):
        changed = tuple((value.detach() * scale).requires_grad_() for value in values)
        expected = joint(reference, *changed)
        with torch.no_grad():
            actual = run(*(value.detach() for value in changed))
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
