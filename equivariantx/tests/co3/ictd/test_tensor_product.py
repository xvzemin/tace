"""Native Cartesian products, independent paths and reference switching."""

import pytest
import torch

from equivariantx import co3
from equivariantx.ops import config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native Cartesian products")
@pytest.mark.parametrize("kind", ["ictd", "ictp"])
@pytest.mark.parametrize(
    "mode,channels",
    [
        ("uuu", (2, 2, 2)),
        ("uvw", (2, 3, 4)),
        ("uvu", (2, 3, 2)),
        ("uvv", (2, 3, 3)),
        ("uuw", (2, 2, 4)),
        ("uvuv", (2, 3, 6)),
    ],
)
def test_backends(kind, mode, channels, monkeypatch, double_precision):
    u, v, w = channels
    irreps = (f"{u}x1o+{u}x2e", f"{v}x1o", f"{w}x1e+{w}x2e+{w}x1o+{w}x3o")
    paths = [
        (0, 0, 0, mode, True),
        (0, 0, 1, mode, True),
        (1, 0, 2, mode, True, 0.7),
        (1, 0, 3, mode, True),
        (1, 0, 2, mode, True, 1.3),
    ]
    constructor = getattr(co3, kind).TensorProduct
    native, ordinary = [
        constructor(
            *irreps,
            paths,
            internal_weights=False,
            shared_weights=False,
            backend=backend,
        ).cuda()
        for backend in ("cuda", "torch")
    ]
    values = (
        co3.Irreps(irreps[0]).randn(2, 1, -1, device="cuda").requires_grad_(),
        co3.Irreps(irreps[1]).randn(1, 3, -1, device="cuda").requires_grad_(),
        torch.randn(2, 3, native.weight_numel, device="cuda", requires_grad=True),
    )
    expected = ordinary(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    torch.testing.assert_close(ordinary(*values), expected, atol=2e-12, rtol=2e-12)
    actual = native(*values)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for _ in range(4 if mode in ("uuu", "uvw") else 2):
        actual, expected = [
            torch.cat(
                [
                    part.flatten()
                    for part in torch.autograd.grad(
                        result.sin().sum(), values, create_graph=True
                    )
                ]
            )
            / 30
            for result in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    directions = tuple(torch.randn_like(value) for value in values)
    torch.testing.assert_close(
        torch.func.jvp(native, values, directions),
        torch.func.jvp(ordinary, values, directions),
        atol=2e-11,
        rtol=2e-11,
    )
    if mode == "uuu":
        torch.compiler.reset()

        def jvp(*values):
            return torch.func.jvp(native, values, directions)

        torch.testing.assert_close(
            torch.compile(jvp, backend="aot_eager", fullgraph=True)(*values),
            torch.func.jvp(ordinary, values, directions),
            atol=2e-11,
            rtol=2e-11,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native Cartesian products")
@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("nodes", [0, 5])
def test_unweighted_channel_program(project, nodes, double_precision):
    paths = [(0, 0, 0, "uuu", False), (0, 0, 1, "uuu", False)]
    modules = [
        co3.TensorProduct(
            "3x1o", "3x1o", "3x0e+3x2e", paths, project=project, backend=backend
        ).cuda()
        for backend in ("cuda", "torch")
    ]
    values = tuple(
        torch.randn(nodes, 9, device="cuda", requires_grad=True) for _ in range(2)
    )
    actual, expected = [module(*values) for module in modules]
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for _ in range(4):
        actual, expected = [
            torch.cat(
                [
                    gradient.flatten()
                    for gradient in torch.autograd.grad(
                        result.sin().sum(), values, create_graph=True
                    )
                ]
            )
            / 20
            for result in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)


def test_native_requires_cuda():
    module = co3.ElementwiseTensorProduct("1o", "1o", backend="cuda")
    with pytest.raises(ValueError, match="CUDA"):
        module(torch.ones(2, 3), torch.ones(2, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Captured CUDA-device Torch")
@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_compiled_torch_channel_program(project, shared, monkeypatch, double_precision):
    from torch.fx.experimental.proxy_tensor import make_fx

    module = co3.TensorProduct(
        "3x1o",
        "3x1o",
        "3x0e+3x2e",
        [(0, 0, 0, "uuu", True), (0, 0, 1, "uuu", True), (0, 0, 0, "uuu", False, 0.7)],
        internal_weights=False,
        shared_weights=shared,
        project=project,
    ).cuda()
    x, y = [torch.randn(5, 9, device="cuda", requires_grad=True) for _ in range(2)]
    weight = torch.randn(
        *(() if shared else (5,)),
        module.weight_numel,
        device="cuda",
        requires_grad=True,
    )
    values = x, y, weight
    graph = make_fx(module)(*values)
    assert all(not str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = module(*values)
    actual = graph(*values)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for _ in range(4):
        actual, expected = [
            torch.cat(
                [
                    part.flatten()
                    for part in torch.autograd.grad(
                        result.sin().sum(), values, create_graph=True
                    )
                ]
            )
            / 20
            for result in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.compiler.reset()
    compiled = torch.compile(module, fullgraph=True, dynamic=True)
    for nodes in (7, 11):
        values = (
            torch.randn(nodes, 9, device="cuda"),
            torch.randn(nodes, 9, device="cuda"),
            torch.randn(
                *(() if shared else (nodes,)), module.weight_numel, device="cuda"
            ),
        )
        torch.testing.assert_close(
            compiled(*values), module(*values), atol=1e-11, rtol=1e-11
        )
