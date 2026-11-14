"""Bounded native ACE polynomials, mixed derivatives and standalone AOTI."""

import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.dynamic_spec import (
    ParamsSpec,
    ShapesSpec,
    ShapeVar,
    TensorSpec,
)
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.mbe.ace import UuuSymmetricContraction
from equivariantx.ops import config
from equivariantx.ops.common.polynomial import polynomial_prefix
from equivariantx.ops.common.symmetric import symmetric_inputs
from equivariantx.tests.utils.testing import check_forward_ad

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native ACE polynomials require CUDA"
)


def contraction(degree=4, channels=2, backend="cuda"):
    entries = [((0,), 0, 0, 0.7)]
    path = 1
    for order in range(2, degree + 1):
        entries.append(((0,) * order, path, 0, 0.3))
        for components in (range(1, 4), range(4, 9)):
            path += 1
            entries.extend(
                ((0,) * (order - 2) + (i, i), path, 0, 1 / len(components) ** 0.5)
                for i in components
            )
        path += 1
    return UuuSymmetricContraction.from_entries(
        f"{channels}x0e+{channels}x1o+{channels}x2e",
        f"{channels}x0e",
        entries,
        path,
        backend=backend,
    ).cuda()


@pytest.mark.parametrize("degree", [2, 4])
@pytest.mark.parametrize("budget", [4, 2048])
def test_recursive_derivatives(degree, budget, monkeypatch):
    monkeypatch.setattr(config, "SYMMETRIC_NATIVE", True)
    monkeypatch.setattr(config, "SYMMETRIC_NATIVE_MAX_NODES", budget)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    module = contraction(degree)
    reference = contraction(degree, backend="torch")
    x = (
        torch.randn(3, 2, 9, dtype=torch.float64, device="cuda") * 0.2
    ).requires_grad_()
    w = (
        torch.randn(2, module.num_paths, 2, dtype=torch.float64, device="cuda") * 0.2
    ).requires_grad_()
    types = torch.tensor([0, 1, 0], device="cuda")
    actual, expected = module(x, w, types), reference(x, w, types)
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        actual, expected = [
            torch.cat(
                [
                    value.flatten()
                    for value in torch.autograd.grad(
                        result.sin().sum(), (x, w), create_graph=True
                    )
                ]
            )
            * 0.2
            for result in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    if budget == 2048:
        assert torch.autograd.gradcheck(
            lambda x, w: module(x, w, types), (x, w), fast_mode=True
        )
        assert torch.autograd.gradgradcheck(
            lambda x, w: module(x, w, types), (x, w), fast_mode=True
        )


def test_functional_transforms(monkeypatch):
    monkeypatch.setattr(config, "SYMMETRIC_NATIVE", True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    module, reference = contraction(), contraction(backend="torch")
    x = torch.randn(3, 2, 9, dtype=torch.float64, device="cuda") * 0.2
    w = torch.randn(2, module.num_paths, 2, dtype=torch.float64, device="cuda") * 0.2
    types = torch.tensor([0, 1, 0], device="cuda")
    check_forward_ad(
        lambda x, w: module(x, w, types),
        lambda x, w: reference(x, w, types),
        (x, w),
    )
    actual, expected = [
        torch.vmap(lambda x: layer(x, w, types))(torch.stack((x, x * 0.5)))
        for layer in (module, reference)
    ]
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("nodes", [0, 17, 37])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strides_indices_and_empty(nodes, dtype, monkeypatch):
    monkeypatch.setattr(config, "SYMMETRIC_NATIVE", True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    module, reference = contraction(), contraction(backend="torch")
    x = (torch.randn(nodes, 4, 18, dtype=dtype, device="cuda") * 0.2)[
        :, ::2, ::2
    ].requires_grad_()
    w = (torch.randn(3, module.num_paths * 2, 4, dtype=dtype, device="cuda") * 0.2)[
        :, ::2, ::2
    ].requires_grad_()
    types = (torch.randperm(nodes, device="cuda") % 3).int()
    values = []
    for layer in (module, reference):
        y = layer(x, w, types)
        first = torch.autograd.grad(y.square().sum(), (x, w), create_graph=True)
        second = torch.autograd.grad(sum(v.square().sum() for v in first), (x, w))
        values.append((y, *first, *second))
    tolerance = 3e-6 if dtype == torch.float32 else 2e-9
    torch.testing.assert_close(*values, atol=tolerance, rtol=tolerance)


def test_packet_budget(monkeypatch):
    from equivariantx.ops.common.cuda.symmetric import compiled_packets

    monkeypatch.setattr(config, "SYMMETRIC_NATIVE", True)
    module = contraction()
    x = torch.randn(3, 2, 9, device="cuda")
    w = torch.randn(2, module.num_paths, 2, device="cuda")
    program, _ = symmetric_inputs(x, w, module.correlation, module.output_dim)
    groups = compiled_packets(module.metadata, program, 2048)
    assert groups is not None
    for chunks in groups:
        for packets in chunks:
            assert sum(len(polynomial_prefix(packet)[0]) for packet in packets) <= 2048
    assert compiled_packets(module.metadata, program, 4) is None


def test_split_output_force_loss(monkeypatch):
    from collections import Counter

    from equivariantx.ops.common.cuda.symmetric import compiled_packets

    monkeypatch.setattr(config, "SYMMETRIC_NATIVE", True)
    monkeypatch.setattr(config, "SYMMETRIC_NATIVE_MAX_NODES", 32)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    module, reference = contraction(6), contraction(6, backend="torch")
    x = (
        torch.randn(3, 2, 9, device="cuda", dtype=torch.float64) * 0.1
    ).requires_grad_()
    weight = (
        torch.randn(3, module.num_paths, 2, device="cuda", dtype=torch.float64) * 0.1
    ).requires_grad_()
    types = torch.tensor([2, 0, 1, 0, 0, 0], device="cuda")[::2]
    program, _ = symmetric_inputs(x, weight, module.correlation, module.output_dim)
    groups = compiled_packets(module.metadata, program, 32)
    assert groups is not None
    destinations = Counter()
    for chunks in groups:
        for packets in chunks:
            assert sum(len(polynomial_prefix(packet)[0]) for packet in packets) <= 32
            for packet in packets:
                destinations.update(key for key, _ in packet)
    assert max(destinations.values()) > 1

    def evaluate(layer):
        energy = layer(x, weight, types).sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + (force + 0.01).square().sum()
        return energy, force, torch.autograd.grad(loss, weight)[0]

    torch.testing.assert_close(
        evaluate(module), evaluate(reference), atol=2e-10, rtol=2e-9
    )


@pytest.mark.parametrize("trainable", [0, 1])
def test_partial_gradient(trainable, monkeypatch):
    monkeypatch.setattr(config, "SYMMETRIC_NATIVE", True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    module, reference = contraction(), contraction(backend="torch")
    values = (
        torch.randn(5, 2, 9, dtype=torch.float64, device="cuda") * 0.1,
        torch.randn(3, module.num_paths, 2, dtype=torch.float64, device="cuda") * 0.1,
    )
    values[trainable].requires_grad_()
    types = torch.arange(5, device="cuda") % 3
    gradients = [
        torch.autograd.grad(layer(*values, types).sin().sum(), values[trainable])[0]
        for layer in (module, reference)
    ]
    torch.testing.assert_close(*gradients, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize("budget,strided", [(4, False), (2048, False), (2048, True)])
def test_dynamic_force_loss_aoti(budget, strided, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SYMMETRIC_NATIVE", True)
    monkeypatch.setattr(config, "SYMMETRIC_NATIVE_MAX_NODES", budget)
    module, reference = (
        contraction(channels=64),
        contraction(channels=64, backend="torch"),
    )

    def joint(layer, x, w, types):
        energy = layer(x, w, types).sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + (force + 0.01).square().sum()
        return energy, force, torch.autograd.grad(loss, w)[0]

    def features(nodes):
        shape = (64, nodes, 9) if strided else (nodes, 64, 9)
        value = torch.randn(*shape, dtype=torch.float64, device="cuda") * 0.1
        return (value.transpose(0, 1) if strided else value).requires_grad_()

    count = ShapeVar("nodes", min=2, optimization_hint=7)
    x = features(7)
    w = (
        torch.randn(3, module.num_paths, 64, dtype=torch.float64, device="cuda") * 0.1
    ).requires_grad_()
    types = torch.arange(7, device="cuda") % 3
    specs = [
        TensorSpec([count, None, None]),
        TensorSpec([None, None, None]),
        TensorSpec([count]),
    ]
    graph = make_fx(
        lambda *values: joint(module, *values),
        tracing_mode="fake",
        dynamic_shapes=ShapesSpec(ParamsSpec({"*args": specs})),
    )(x, w, types)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    assert any(
        "eqx.symmetric_compiled" in str(node.target) for node in graph.graph.nodes
    ) == (budget == 2048)
    nodes = torch.export.Dim("nodes", min=2, max=8192)
    exported = torch.export.export(
        graph,
        (x.detach(), w.detach(), types),
        strict=True,
        dynamic_shapes=({0: nodes}, None, {0: nodes}),
    )
    package = tmp_path / "symmetric.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (7, 11, 9):
        x = features(count)
        types = torch.arange(count, device="cuda") % 3
        for _ in range(3):
            expected = joint(reference, x, w, types)
            records.append(
                dict(
                    inputs=tuple(v.detach().cpu() for v in (x, w, types)),
                    outputs=tuple(v.detach().cpu() for v in expected),
                )
            )
            w = (w - 0.001 * expected[-1]).detach().requires_grad_()
    data = tmp_path / "cases.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys, torch
run=torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2],weights_only=True):
    actual=run(*(value.cuda() for value in row['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual),row['outputs'],atol=2e-9,rtol=2e-9)
assert 'equivariantx' not in sys.modules
""",
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strided_node_type_force_loss(native, index_dtype, dtype, monkeypatch):
    monkeypatch.setattr(config, "SYMMETRIC_NATIVE", native)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    torch.manual_seed(3711)
    module, reference = contraction(degree=2), contraction(degree=2, backend="torch")
    x = (torch.randn(17, 4, 18, dtype=dtype, device="cuda") * 0.15)[
        :, ::2, ::2
    ].requires_grad_()
    w = (torch.randn(3, module.num_paths * 2, 4, dtype=dtype, device="cuda") * 0.15)[
        :, ::2, ::2
    ].requires_grad_()
    types = (torch.arange(34, device="cuda") % 3).to(index_dtype)[::2]
    assert types.stride() == (2,)
    original = tuple(value.detach().clone() for value in (x, w, types))
    strides = x.stride(), w.stride()

    def evaluate(layer):
        output = layer(x, w, types)
        energy = output.sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + (force + 0.01).square().sum()
        gradient = torch.autograd.grad(loss, w)[0]
        assert all(value.dtype == dtype for value in (output, energy, force, gradient))
        return output, energy, force, gradient

    tolerance = 3e-6 if dtype == torch.float32 else 2e-9
    torch.testing.assert_close(
        evaluate(module), evaluate(reference), rtol=tolerance, atol=tolerance
    )
    assert (x.stride(), w.stride()) == strides
    for actual, expected in zip((x, w, types), original):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("native", [False, True])
def test_contiguous_node_type_reuses_storage(native, monkeypatch):
    monkeypatch.setattr(config, "SYMMETRIC_NATIVE", native)
    module = contraction(degree=2)
    x = torch.randn(3, 2, 9, dtype=torch.float64, device="cuda") * 0.1
    w = torch.randn(3, module.num_paths, 2, dtype=torch.float64, device="cuda") * 0.1
    types = torch.tensor([2, 0, 1], device="cuda")
    original, results = torch.Tensor.contiguous, []

    def contiguous(value, *args, **kwargs):
        output = original(value, *args, **kwargs)
        if value is types:
            results.append(output is value)
        return output

    monkeypatch.setattr(torch.Tensor, "contiguous", contiguous)
    module(x, w, types)
    assert results and all(results)
