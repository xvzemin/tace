"""Tests for the corresponding EQX source module."""

import subprocess
import sys
import pytest
import torch
from equivariantx.ops import get_kernel
from equivariantx.ops import layout
from equivariantx.tests.ops.kernels.torch.test_layout import column_map
from equivariantx.tests.ops.testing import assert_functional_derivatives


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA layout transforms")
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
@pytest.mark.parametrize("operation", ["gather", "permute", "sum", "sum_transpose"])
def test_functional_transforms(operation, compiler, double_precision):
    if operation == "gather":
        values = (
            torch.randn(5, 3, device="cuda"),
            torch.tensor([2, 0, 2, 1, 4], device="cuda"),
        )

        def function(x, index):
            return layout.gather_sum([x, x], [index, None])

        def reference(x, index):
            return x[index] + x
    elif operation == "permute":
        values = (
            torch.randn(5, 5, device="cuda"),
            torch.tensor([3, 0, 4, 2, 1], device="cuda"),
        )

        def function(x, index):
            return layout.grouped_permute(x, index, index.argsort(), repr((2, 0, 3)))

        def reference(x, index):
            return get_kernel("grouped_permute", "torch")(
                x, index, index.argsort(), repr((2, 0, 3))
            )
    else:
        transpose = operation == "sum_transpose"
        index, ptr, ti, tp, count = column_map("cuda")
        if transpose:
            index, ptr, ti, tp = ti, tp, index, ptr
        values = (torch.randn(5, 3 if transpose else 5, device="cuda"), index)

        def function(x, index):
            return layout.indexed_sum(x, index, ptr, ti, tp, count, transpose)

        def reference(x, index):
            return get_kernel("indexed_sum", "torch")(
                x, index, ptr, ti, tp, count, transpose
            )

    assert_functional_derivatives(function, reference, values, compiler)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_aoti_reload(tmp_path):
    from equivariantx.ops.layout import gather_sum

    class Model(torch.nn.Module):
        def forward(self, features, indices):
            return gather_sum([features, features], [indices, indices])

    inputs = (
        torch.randn(4, 3, device="cuda", dtype=torch.float64),
        torch.tensor([2, 0, 3, 1, 0], device="cuda"),
    )
    model = Model().eval()
    torch.testing.assert_close(model(*inputs), 2 * inputs[0][inputs[1]])
    exported = torch.export.export(
        model,
        inputs,
        dynamic_shapes={
            "features": {0: torch.export.Dim("nodes", min=2)},
            "indices": {0: torch.export.Dim("edges", min=1)},
        },
    )
    path = tmp_path / "model.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(path))
    script = """
import sys
import torch
import equivariantx as eqx

assert "tace" not in sys.modules
model = torch._inductor.aoti_load_package(sys.argv[1])
for nodes, edges in ((4, 5), (7, 11)):
    features = torch.randn(nodes, 3, device="cuda", dtype=torch.float64)
    indices = torch.arange(edges, device="cuda") % nodes
    torch.testing.assert_close(model(features, indices), 2 * features[indices])
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("edges", [0, 17])
@pytest.mark.parametrize("first_indexed", [False, True])
def test_gather_sum(dtype, edges, first_indexed):
    from equivariantx.ops.layout import gather_sum

    indices = [torch.randint(5, (2 * edges,), device="cuda")[::2] for _ in range(3)]
    if not first_indexed:
        indices[0] = None
    inputs = [
        torch.randn(9, 2 * (edges if index is None else 5), device="cuda", dtype=dtype)
        .T[::2]
        .requires_grad_()
        for index in indices
    ]
    reference = sum(
        value if index is None else value.index_select(0, index)
        for value, index in zip(inputs, indices)
    )
    actual = gather_sum(inputs, indices)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    outputs = [
        value.sin().sum() / max(value.numel(), 1) for value in (actual, reference)
    ]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(output, inputs, create_graph=True, retain_graph=True)
            for output in outputs
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)
        outputs = [sum(value.square().sum() for value in grad) for grad in gradients]
    compiled = torch.compile(gather_sum, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(inputs, indices), reference, rtol=0, atol=0)
