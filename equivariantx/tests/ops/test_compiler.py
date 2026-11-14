"""Explicit compiler passes preserve state, derivatives and storage semantics."""

import copy
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.ops import compiler
from equivariantx.ops.compiler import JointCSE


@pytest.fixture
def compiler_pass():
    version = re.match(r"(\d+)\.(\d+)", str(torch.__version__))
    if version is None or tuple(map(int, version.groups())) < (2, 15):
        pytest.skip("JointCSE requires PyTorch >= 2.15.")
    return JointCSE()


@pytest.mark.parametrize("version", ["2.4.0+cpu", "2.9.0", "2.14.1", "unknown"])
def test_unsupported_version(version, monkeypatch):
    monkeypatch.setattr(torch, "__version__", version)
    with pytest.raises(RuntimeError, match=r"PyTorch >= 2\.15"):
        JointCSE()


def test_missing_private_api(monkeypatch):
    monkeypatch.setattr(compiler, "CustomGraphPass", object)
    with pytest.raises(RuntimeError, match="CustomGraphPass"):
        JointCSE()


def test_optional_import_and_serialization(compiler_pass, tmp_path):
    from torch._inductor import config

    settings = {
        name: getattr(config, name)
        for name in ("joint_custom_pre_pass", "post_grad_custom_pre_pass")
    }
    restored = pickle.loads(pickle.dumps(compiler_pass))
    assert not vars(restored)
    assert restored.uuid() == compiler_pass.uuid()
    payload = tmp_path / "pass.pkl"
    payload.write_bytes(pickle.dumps((restored, restored.uuid())))
    code = """
import importlib.util
import pickle
import sys
import torch
import equivariantx
assert 'equivariantx.ops.compiler' not in sys.modules
from torch._inductor import config
settings = (config.joint_custom_pre_pass, config.post_grad_custom_pre_pass)
from equivariantx.ops.compiler import JointCSE
value, expected = pickle.load(open(sys.argv[1], 'rb'))
assert isinstance(value, JointCSE) and not vars(value)
assert value.uuid() == expected
assert settings == (config.joint_custom_pre_pass, config.post_grad_custom_pre_pass)
# An old installation can import the optional file even without the private API.
torch.__version__ = '2.4.0'
sys.modules['torch._inductor.custom_graph_pass'] = None
spec = importlib.util.spec_from_file_location('compiler_old', sys.argv[2])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.JointCSE()
except RuntimeError as error:
    assert 'PyTorch >= 2.15' in str(error)
else:
    raise AssertionError('unsupported compiler must fail when requested')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(payload), str(Path(compiler.__file__))],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(getattr(config, name) is value for name, value in settings.items())


def test_mutable_parameters_and_buffers(compiler_pass):
    from torch._inductor import config

    torch.manual_seed(519)
    left = torch.randn(4, 2, dtype=torch.float64)
    right = left.clone()
    indices_left = torch.tensor([0, 1, 2, 0])
    indices_right = indices_left.clone()

    def operation(x, w, v):
        a = x.index_select(1, indices_left) @ left
        b = x.index_select(1, indices_right) @ right
        output = a.sin() @ w + b.cos() @ v
        force = torch.autograd.grad(output.sum(), x, create_graph=True)[0]
        loss = output.square().mean() + force.square().mean()
        return loss, force, *torch.autograd.grad(loss, (w, v))

    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    v = w.detach().clone().requires_grad_()
    original = make_fx(operation)(x, w, v)
    candidate = copy.deepcopy(original)
    attributes = {
        node.target: getattr(original, node.target)
        for node in original.graph.nodes
        if node.op == "get_attr"
    }
    targets = [
        next(name for name, value in attributes.items() if value is tensor)
        for tensor in (left, indices_left)
    ]
    previous = config.joint_custom_pre_pass
    run = torch.compile(
        candidate, fullgraph=True, options={"joint_custom_pre_pass": compiler_pass}
    )
    for step in range(4):
        expected = operation(x, w, v)
        actual = run(x.detach(), w.detach(), v.detach())
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        w = (w - 0.01 * expected[2]).detach().requires_grad_()
        for graph in (original, candidate):
            getattr(graph, targets[0]).add_(0.01)
            getattr(graph, targets[1]).copy_(torch.tensor([(step + 1) % 3, 0, 1, 2]))
    assert config.joint_custom_pre_pass is previous


def test_output_storage_and_random_operations(compiler_pass):
    def operation(x):
        return x.clone(), x.clone(), torch.rand_like(x), torch.rand_like(x)

    run = torch.compile(
        operation, fullgraph=True, options={"joint_custom_pre_pass": compiler_pass}
    )
    x = torch.randn(4, 3)
    left, right, a, b = run(x)
    assert left.data_ptr() != right.data_ptr()
    left.add_(1)
    torch.testing.assert_close(right, x)
    assert not torch.equal(a, b)


def test_high_order_graph(compiler_pass):
    x = (torch.randn(3, 2, dtype=torch.float64) * 0.1).requires_grad_()
    weight = (torch.randn(2, 2, dtype=torch.float64) * 0.1).requires_grad_()
    directions = (torch.randn_like(x) / 6**0.5, torch.randn_like(weight) / 2)

    def operation(x, weight):
        first, second = (x @ weight).sin(), (x @ weight).sin()
        value = (first * second + first.cos()).mean()
        result = [value]
        for _ in range(5):
            gradients = torch.autograd.grad(value, (x, weight), create_graph=True)
            value = sum((g * d).sum() for g, d in zip(gradients, directions))
            result.append(value)
        return tuple(result)

    graph = make_fx(operation)(x, weight)
    run = torch.compile(
        graph, fullgraph=True, options={"joint_custom_pre_pass": compiler_pass}
    )
    torch.testing.assert_close(
        run(x.detach(), weight.detach()), operation(x, weight), atol=3e-10, rtol=3e-10
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("width", [64, 128, 1024])
def test_dynamic_aoti_state_updates(compiler_pass, width, tmp_path):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    def operation(x, w1, b1, w2, b2, left, right, indices):
        hidden = torch.nn.functional.silu(x @ w1 + b1)
        output = hidden @ w2 + b2
        energy = (output.sin() * left + output[indices].cos() * right).mean()
        force = torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + (force + 0.01).square().sum()
        return loss, force, *torch.autograd.grad(loss, (w1, b1, w2, b2))

    def tensor(shape, trainable=True):
        return (
            torch.randn(*shape, dtype=torch.float64, device="cuda") * 0.1
        ).requires_grad_(trainable)

    values = (
        tensor((7, 3)),
        tensor((3, width)),
        tensor((width,)),
        tensor((width, 2)),
        tensor((2,)),
        torch.ones(2, dtype=torch.float64, device="cuda"),
        torch.ones(2, dtype=torch.float64, device="cuda"),
        torch.arange(7, device="cuda"),
    )
    rows = ShapeVar("rows", min=2)
    shapes = ShapesSpec(
        ParamsSpec(
            {
                "x": TensorSpec((rows, None)),
                "indices": TensorSpec((rows,)),
            }
        )
    )
    graph = make_fx(
        operation,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=shapes,
    )(*values)
    rows = torch.export.Dim("rows", min=2)
    exported = torch.export.export(
        graph,
        tuple(v.detach() for v in values),
        strict=True,
        dynamic_shapes=({0: rows}, None, None, None, None, None, None, {0: rows}),
    )
    package = tmp_path / "training.pt2"
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(package),
        inductor_configs={"joint_custom_pre_pass": compiler_pass},
    )
    records = []
    for count in (7, 11, 9):
        values = (
            tensor((count, 3)),
            *values[1:7],
            torch.randperm(count, device="cuda"),
        )
        for _ in range(3):
            expected = operation(*values)
            records.append(
                {
                    "inputs": tuple(v.detach().cpu() for v in values),
                    "outputs": tuple(v.detach().cpu() for v in expected),
                }
            )
            parameters = tuple(
                (v - 0.01 * g).detach().requires_grad_()
                for v, g in zip(values[1:5], expected[2:])
            )
            values = (
                values[0],
                *parameters,
                values[5] + 0.001,
                values[6],
                (values[7] + 1) % count,
            )
    data = tmp_path / "states.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = run(*(v.cuda() for v in row['inputs']))
    torch.testing.assert_close(tuple(v.cpu() for v in actual), row['outputs'], atol=3e-9, rtol=3e-9)
assert 'equivariantx' not in sys.modules
""",
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
