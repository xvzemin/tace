"""Numerical force-loss problems shared by Wigner-6j kernel tests."""

import os
import subprocess
import sys

import torch

from equivariantx import o3
from equivariantx.conv import Wigner6jTensorProductConv


def problem(
    kind,
    backend,
    device,
    count=4,
    nodes=3,
    dtype=torch.float64,
    *,
    shared=False,
    multiplicity=2,
    edge_degree=1,
):
    cls = (
        o3.Wigner6jTensorProduct
        if kind == "tensor_product"
        else Wigner6jTensorProductConv
    )
    parity = "e" if edge_degree else "o"
    layer = cls(
        "2x1o",
        f"{multiplicity}x1o" if edge_degree else f"{multiplicity}x0e",
        f"{multiplicity}x1e",
        f"2x1{parity}",
        f"2x0{parity}+2x1{parity}+2x2{parity}",
        [(0, 0, 0, "uvu", True)],
        [(0, 0, i, "uvu", True) for i in range(3)],
        internal_weights=False,
        shared_weights=shared,
        backend=backend,
    ).to(device=device, dtype=dtype)
    node_rows = count if kind == "tensor_product" else nodes
    values = tuple(
        (torch.randn(*shape, dtype=dtype, device=device) * 0.15).requires_grad_()
        for shape in (
            (node_rows, 6),
            (count, (2 * edge_degree + 1) * multiplicity),
            (node_rows, 3 * multiplicity),
            (layer.weight_numel1,) if shared else (count, layer.weight_numel1),
            (layer.weight_numel2,) if shared else (count, layer.weight_numel2),
        )
    )
    index = torch.randint(nodes, (2, count), device=device)

    def forward(*args, reference=False):
        operation = layer.forward_reference if reference else layer
        if kind == "tensor_product":
            return operation(*args)
        x1, x2, x3, w1, w2 = args
        return operation(x1, x2, x3, index, w1, w2, nodes + 1)

    return layer, values, index, forward


def check_derivatives(kind, backend, device, **options):
    _, values, _, forward = problem(kind, backend, device, count=2, nodes=2, **options)
    actual = forward(*values)
    expected = forward(*values, reference=True)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    directions = tuple(
        torch.randn_like(value) / value.numel() ** 0.5 for value in values
    )
    outputs = [actual, expected]
    for order in range(5):
        gradients = [
            torch.autograd.grad(y.sin().mean(), values, create_graph=True)
            for y in outputs
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=3e-10, rtol=3e-10)
        outputs = [
            sum((g * d).sum() for g, d in zip(gradient, directions)).reshape(1)
            for gradient in gradients
        ]
    expected_jvp = torch.func.jvp(
        lambda *v: forward(*v, reference=True), values, directions
    )
    actual_jvp = torch.func.jvp(forward, values, directions)
    torch.testing.assert_close(actual_jvp, expected_jvp, atol=3e-11, rtol=3e-11)
    with torch.autograd.forward_ad.dual_level():
        duals = tuple(
            torch.autograd.forward_ad.make_dual(x, dx)
            for x, dx in zip(values, directions)
        )
        dual = torch.autograd.forward_ad.unpack_dual(forward(*duals))
        torch.testing.assert_close(
            dual.tangent, expected_jvp[1], atol=3e-11, rtol=3e-11
        )
    batched = tuple(torch.stack((value, value * 0.8)) for value in values)
    torch.testing.assert_close(
        torch.vmap(forward)(*batched),
        torch.stack(
            [
                forward(*(value[i] for value in batched), reference=True)
                for i in range(2)
            ]
        ),
        atol=3e-11,
        rtol=3e-11,
    )


def check_package(kind, tmp_path, device="cuda", **options):
    from torch.fx.experimental.proxy_tensor import make_fx

    _, values, _, forward = problem(
        kind, "cuda" if device == "cuda" else "torch", device, **options
    )

    def training(*args, reference=False):
        prediction = forward(*args, reference=reference)
        energy = 1 + prediction.sin().mean()
        force = -torch.autograd.grad(energy, args[0], create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        gradients = torch.autograd.grad(loss, args[3:])
        return loss, force, *gradients

    graph = make_fx(training)(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph, tuple(value.detach() for value in values), strict=True
    )
    package = tmp_path / (kind + ".pt2")
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for _ in range(3):
        expected = training(*values, reference=True)
        records.append(
            {
                "inputs": tuple(t.detach().cpu() for t in values),
                "outputs": tuple(t.detach().cpu() for t in expected),
            }
        )
        values = tuple(
            ((value - 0.01 * expected[i - 1]) if i >= 3 else value * 0.95 + 0.01)
            .detach()
            .requires_grad_()
            for i, value in enumerate(values)
        )
    data = tmp_path / "data.pt"
    torch.save(records, data)
    from equivariantx.tests.utils.testing import package_script

    script = """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad(): actual = run(*(value.to(sys.argv[3]) for value in row['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual), row['outputs'], atol=3e-9, rtol=3e-9)
assert ('equivariantx' in sys.modules) == (sys.argv[3] == 'cuda')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            package_script(script) if device == "cuda" else script,
            str(package),
            str(data),
            device,
        ],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def check_dynamic_convolution(tmp_path, backend, device, shared):
    """Compile independent node/edge sizes across both recoupling cost regimes."""
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    options = dict(shared=shared, multiplicity=1, edge_degree=0)
    layer, values, index, _ = problem(
        "convolution", backend, device, count=7, nodes=5, **options
    )

    def training(*args, reference=False):
        x1, x2, x3, w1, w2, indices = args
        operation = layer.forward_reference if reference else layer
        energy = (
            1 + operation(x1, x2, x3, indices, w1, w2, x1.shape[0] + 1).square().sum()
        )
        force = -torch.autograd.grad(energy, x1, create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        return loss, force, *torch.autograd.grad(loss, (w1, w2))

    nodes = ShapeVar("nodes", min=2, optimization_hint=5)
    edges = ShapeVar("edges", min=2, optimization_hint=7)
    specs = [
        TensorSpec([nodes, None]),
        TensorSpec([edges, None]),
        TensorSpec([nodes, None]),
    ]
    specs += [
        TensorSpec([None]) if shared else TensorSpec([edges, None]) for _ in range(2)
    ]
    specs += [TensorSpec([None, edges])]
    graph = make_fx(
        training,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(ParamsSpec({"*args": specs})),
    )(*values, index)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    nodes, edges = torch.export.Dim("nodes", min=2), torch.export.Dim("edges", min=2)
    shapes = [{0: nodes}, {0: edges}, {0: nodes}]
    shapes += [{} if shared else {0: edges} for _ in range(2)]
    shapes += [{1: edges}]
    exported = torch.export.export(
        graph,
        (*[value.detach() for value in values], index),
        dynamic_shapes=tuple(shapes),
        strict=True,
    )
    package = tmp_path / "wigner-dynamic.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for count, nodes in ((7, 5), (15, 3), (3, 15)):
        _, values, index, _ = problem(
            "convolution", backend, device, count, nodes, **options
        )
        # The two concrete regimes retain their measured eager choice.
        expected_choice = (
            layer.schedule.node_cost * nodes
            < (layer.schedule.standard_cost - layer.schedule.recoupled_cost) * count
        )
        assert layer.schedule.use_recoupling(values, index[0]) == expected_choice
        for _ in range(3):
            expected = training(*values, index, reference=True)
            records.append(
                {
                    "inputs": tuple(value.detach().cpu() for value in (*values, index)),
                    "outputs": tuple(value.detach().cpu() for value in expected),
                }
            )
            values = tuple(
                ((value - 0.01 * expected[i - 1]) if i >= 3 else value * 0.95 + 0.01)
                .detach()
                .requires_grad_()
                for i, value in enumerate(values)
            )
    data = tmp_path / "dynamic-data.pt"
    torch.save(records, data)
    script = """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2], weights_only=True):
    actual = run(*(value.to(sys.argv[3]) for value in record['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual), record['outputs'], atol=3e-9, rtol=3e-9)
assert ('equivariantx' in sys.modules) == (sys.argv[3] == 'cuda')
"""
    from equivariantx.tests.utils.testing import package_script

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            package_script(script) if device == "cuda" else script,
            str(package),
            str(data),
            device,
        ],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
