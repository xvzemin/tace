"""Symmetric matrix actions and their complete compiled derivatives."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.co3.symmetric import SymmetricBasis
from equivariantx.ops import config
from equivariantx.ops.torch.co3 import symmetric as kernel


@pytest.mark.parametrize("rank", [2, 4])
def test_joint_inductor(rank, monkeypatch, device):
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_KERNELS", True)
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_ADJOINT", True)
    module = SymmetricBasis(rank).to(device=device, dtype=torch.float64)
    x = (
        torch.randn(2, 2, 3**rank, device=device, dtype=torch.float64) * 0.1
    ).requires_grad_()
    v = (
        torch.randn(2, 1, 3, device=device, dtype=torch.float64) + 0.5
    ).requires_grad_()
    w = (torch.randn(2, 1, device=device, dtype=torch.float64) * 0.1).requires_grad_()

    def operation(x, v, w):
        n = torch.nn.functional.normalize(v, dim=-1)
        plane = torch.eye(3, device=x.device, dtype=x.dtype) - n.unsqueeze(
            -1
        ) * n.unsqueeze(-2)
        y = module.unpack(module(module.pack(x * w), plane))
        e = 1 + y.sin().sum()
        force = -torch.autograd.grad(e, v, create_graph=True)[0]
        loss = force.square().sum() + e.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    graph = make_fx(operation)(x, v, w)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    compiled = torch.compile(graph, fullgraph=True)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for _ in range(3):
        expected = operation(x, v, w)
        actual = compiled(x.detach(), v.detach(), w.detach())
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()


def directional_derivatives(operation, values, directions):
    scalar = operation(*values).sin().sum()
    result = []
    for _ in range(5):
        gradients = torch.autograd.grad(scalar, values, create_graph=True)
        scalar = sum((g * d).sum() for g, d in zip(gradients, directions))
        result.append(scalar)
    return result


@pytest.mark.parametrize("rank", [2, 4, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_transform_derivatives(rank, dtype, monkeypatch):
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_MATRIX", True)
    module = SymmetricBasis(rank).to(dtype=dtype)
    x = (torch.randn(1, 64, module.dim, dtype=dtype) * 0.1).requires_grad_()
    a = (
        torch.randn(1, 1, 3, 3, dtype=dtype) * 0.1 + torch.eye(3, dtype=dtype)
    ).requires_grad_()
    values = x, a
    graph = make_fx(module)(*values)
    directions = tuple(torch.randn_like(value) * 0.1 for value in values)

    def reference(x, a):
        return kernel.transform_reference(module, x, a)

    tolerance = (
        {"atol": 3e-6, "rtol": 3e-5}
        if dtype == torch.float32
        else {"atol": 3e-10, "rtol": 3e-9}
    )
    torch.testing.assert_close(graph(*values), reference(*values), **tolerance)
    torch.testing.assert_close(
        directional_derivatives(graph, values, directions),
        directional_derivatives(reference, values, directions),
        **tolerance,
    )
    torch.testing.assert_close(
        torch.func.jvp(graph, values, directions),
        torch.func.jvp(reference, values, directions),
        **tolerance,
    )
    with torch.autograd.forward_ad.dual_level():
        duals = tuple(
            torch.autograd.forward_ad.make_dual(value, direction)
            for value, direction in zip(values, directions)
        )
        torch.testing.assert_close(
            torch.autograd.forward_ad.unpack_dual(graph(*duals)),
            torch.autograd.forward_ad.unpack_dual(reference(*duals)),
            **tolerance,
        )

    def nested(operation):
        return torch.func.jvp(
            lambda *v: torch.func.jvp(operation, v, directions)[1], values, directions
        )

    torch.testing.assert_close(nested(graph), nested(reference), **tolerance)
    batched = tuple(torch.stack((value, value * 0.9)) for value in values)
    torch.testing.assert_close(
        torch.vmap(graph)(*batched), torch.vmap(reference)(*batched), **tolerance
    )


@pytest.mark.parametrize(
    "case, selected",
    [
        ("geometry", True),
        ("global", True),
        ("leading", True),
        ("strided", True),
        ("empty", True),
        ("wide", True),
        ("per_channel", False),
        ("small_channels", False),
        ("two_dimensional", False),
    ],
)
def test_shared_transform_layout(case, selected, monkeypatch):
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_MATRIX", True)
    module = SymmetricBasis(4).double()
    count = 0 if case == "empty" else 3
    width = 8 if case == "small_channels" else 1024 if case == "wide" else 64
    shape = (
        (2, count, width, module.dim)
        if case == "leading"
        else (count, width, module.dim)
    )
    x = torch.randn(*shape, dtype=torch.float64) * 0.1
    a_shape = (
        (2, count, 1, 3, 3)
        if case == "leading"
        else (count, width if case == "per_channel" else 1, 3, 3)
    )
    a = torch.randn(*a_shape, dtype=torch.float64) * 0.1 + torch.eye(
        3, dtype=torch.float64
    )
    if case in ("global", "two_dimensional"):
        a = a[0, 0]
    if case == "two_dimensional":
        x = x[0]
    if case == "strided":
        x = (torch.randn(count, width, 2 * module.dim, dtype=torch.float64) * 0.1)[
            ..., ::2
        ]
        a = a.transpose(-1, -2)
    assert kernel.use_shared_matrix(module, x, a) == selected
    graph = make_fx(module)(x, a)
    torch.testing.assert_close(
        graph(x, a), kernel.transform_reference(module, x, a), atol=3e-10, rtol=3e-9
    )


def test_shared_transform_buffers_and_switches(monkeypatch):
    module = SymmetricBasis(4).float().double()
    torch.testing.assert_close(
        module.transform_identity,
        torch.eye(module.dim, dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    assert "transform_identity" not in module.state_dict()
    assert not hasattr(SymmetricBasis(3), "transform_identity")
    x = torch.randn(2, 64, module.dim, dtype=torch.float64) * 0.1
    a = torch.randn(2, 1, 3, 3, dtype=torch.float64) * 0.1 + torch.eye(
        3, dtype=torch.float64
    )
    expected = kernel.transform_reference(module, x, a)
    for flag, value in (
        ("TORCH_REFERENCE", True),
        ("TORCH_SYMMETRIC_KERNELS", False),
        ("TORCH_SYMMETRIC_MATRIX", False),
    ):
        with monkeypatch.context() as context:
            context.setattr(config, flag, value)
            graph = make_fx(module)(x, a)
            torch.testing.assert_close(graph(x, a), expected, atol=3e-10, rtol=3e-9)


@pytest.mark.parametrize("unknown_channels", [False, True])
def test_shared_transform_symbolic_channels(unknown_channels, monkeypatch):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_MATRIX", True)
    module = SymmetricBasis(2).double()
    rows = ShapeVar("rows", min=0, max=16)
    width = ShapeVar("width", min=2, max=1024) if unknown_channels else None
    x = torch.randn(7, 64, module.dim, dtype=torch.float64) * 0.1
    a = torch.randn(7, 1, 3, 3, dtype=torch.float64) * 0.1 + torch.eye(
        3, dtype=torch.float64
    )
    decisions = []

    def operation(x, a):
        decisions.append(kernel.use_shared_matrix(module, x, a))
        return module(x, a)

    graph = make_fx(
        operation,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(
            ParamsSpec(
                {
                    "x": TensorSpec((rows, width, None)),
                    "a": TensorSpec((rows, None, None, None)),
                }
            )
        ),
    )(x, a)
    assert decisions == [not unknown_channels]
    for count, channels in (
        (7, 64),
        (0, 64),
        (1, 64),
        (3, 128 if unknown_channels else 64),
        (11, 8 if unknown_channels else 64),
    ):
        x = torch.randn(count, channels, module.dim, dtype=torch.float64) * 0.1
        a = torch.randn(count, 1, 3, 3, dtype=torch.float64) * 0.1 + torch.eye(
            3, dtype=torch.float64
        )
        torch.testing.assert_close(
            graph(x, a), kernel.transform_reference(module, x, a), atol=3e-10, rtol=3e-9
        )


@pytest.mark.parametrize("rank", [2, 4])
def test_shared_transform_joint_inductor(rank, monkeypatch):
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_MATRIX", True)
    module = SymmetricBasis(rank).double()
    positions = (torch.randn(2, 3, 3, dtype=torch.float64) * 0.1).requires_grad_()
    features = torch.randn(2, 64, module.dim, dtype=torch.float64) * 0.1
    weight = (torch.randn(64, module.dim, dtype=torch.float64) * 0.1).requires_grad_()

    def operation(v, x, w, reference=False):
        a = (torch.eye(3, dtype=v.dtype) + 0.1 * v).unsqueeze(-3)
        output = (
            kernel.transform_reference(module, x * w, a)
            if reference
            else module(x * w, a)
        )
        energy = output.sin().sum((-1, -2)) / (64 * module.dim) ** 0.5
        force = -torch.autograd.grad(energy.sum(), v, create_graph=True)[0]
        loss = (energy - 0.2).square().sum() + 10 * (force - 0.1).square().sum()
        return energy, force, loss, torch.autograd.grad(loss, w)[0]

    graph = make_fx(lambda v, x, w: operation(v, x, w))(positions, features, weight)
    compiled = torch.compile(graph, fullgraph=True)
    for _ in range(3):
        expected = operation(positions, features, weight, True)
        inputs = positions.detach(), features.detach(), weight.detach()
        actual = compiled(*inputs)
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-9)
        saved = tuple(value.clone() for value in inputs)
        predictions = tuple(value.clone() for value in actual[:-1])
        gradient = actual[-1].clone()
        actual[-1].mul_(0.125)
        torch.testing.assert_close(inputs, saved, atol=0, rtol=0)
        torch.testing.assert_close(actual[:-1], predictions, atol=0, rtol=0)
        weight = (weight - 0.001 * gradient).detach().requires_grad_()


def test_shared_transform_aoti_checked_updates(tmp_path, monkeypatch):
    from torch.fx.experimental import _config as shape_config
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_SYMMETRIC_MATRIX", True)
    module = SymmetricBasis(4).double()
    weight = (torch.randn(64, module.dim, dtype=torch.float64) * 0.1).requires_grad_()

    def inputs(count):
        v = (torch.randn(count, 3, 3, dtype=torch.float64) * 0.1).requires_grad_()
        x = torch.randn(count, 64, module.dim, dtype=torch.float64) * 0.1
        return v, x

    def operation(v, x, w, reference=False):
        matrix = (torch.eye(3, dtype=v.dtype) + 0.1 * v).unsqueeze(-3)
        output = (
            kernel.transform_reference(module, x * w, matrix)
            if reference
            else module(x * w, matrix)
        )
        energy = output.sin().sum((-1, -2)) / (64 * module.dim) ** 0.5
        force = -torch.autograd.grad(energy.sum(), v, create_graph=True)[0]
        loss = (energy - 0.2).square().sum() + 10 * (force - 0.1).square().sum()
        return energy, force, loss, torch.autograd.grad(loss, w)[0]

    v, x = inputs(3)
    with (
        shape_config.patch(backed_size_oblivious=True),
        torch._inductor.config.patch(strict_output_strides=True),
    ):
        rows = ShapeVar("rows", min=2, max=8)
        graph = make_fx(
            lambda v, x, w: operation(v, x, w),
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
            dynamic_shapes=ShapesSpec(
                ParamsSpec(
                    {
                        "v": TensorSpec((rows, None, None)),
                        "x": TensorSpec((rows, None, None)),
                    }
                )
            ),
        )(v, x, weight)
        count = torch.export.Dim("rows", min=2, max=8)
        exported = torch.export.export(
            graph,
            (v.detach(), x.detach(), weight.detach()),
            strict=True,
            dynamic_shapes=({0: count}, {0: count}, None),
        )
        package = tmp_path / "shared-transform.pt2"
        torch._inductor.aoti_compile_and_package(
            exported,
            package_path=str(package),
            inductor_configs={
                "aot_inductor.check_lowerbound": True,
                "aot_inductor.check_upperbound": True,
            },
        )
    initial = weight.detach().clone()
    records = []
    for count in (3, 7, 5):
        v, x = inputs(count)
        for _ in range(3):
            expected = operation(v, x, weight, True)
            records.append(
                (
                    v.detach(),
                    x.detach(),
                    tuple(value.detach().clone() for value in expected),
                )
            )
            weight = (weight - 0.001 * expected[-1]).detach().requires_grad_()
    payload = tmp_path / "records.pt"
    torch.save(
        {"initial": initial, "records": records, "final": weight.detach()}, payload
    )
    child = r"""
import sys, torch
torch.set_num_threads(4)
run = torch._inductor.aoti_load_package(sys.argv[1], run_single_threaded=True)
data = torch.load(sys.argv[2], weights_only=True)
weight = data["initial"].clone()
for v, x, expected in data["records"]:
    saved = (v.clone(), x.clone(), weight.clone())
    actual = run(v, x, weight)
    torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-9)
    predictions = tuple(value.clone() for value in actual[:-1])
    gradient = actual[-1].clone()
    actual[-1].mul_(0.125)
    torch.testing.assert_close((v, x, weight), saved, atol=0, rtol=0)
    torch.testing.assert_close(actual[:-1], predictions, atol=0, rtol=0)
    weight = weight - 0.001 * gradient
torch.testing.assert_close(weight, data["final"], atol=3e-10, rtol=3e-9)
assert not torch.equal(weight, data["initial"])
for count in (0, 1, 9):
    try:
        run(torch.zeros(count,3,3,dtype=weight.dtype), torch.zeros(count,*x.shape[1:],dtype=weight.dtype), weight)
    except RuntimeError as error:
        assert ("too small" if count < 2 else "too large") in str(error), str(error)
    else:
        raise AssertionError("AOTI accepted a shape outside the explicit [2,8] contract")
assert "equivariantx" not in sys.modules and "e3nn" not in sys.modules
print("9 updates; checked [2,8]; Torch-only")
"""
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    environment["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", "-c", child, str(package), str(payload)],
        env=environment,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=180,
    )
    (tmp_path / "loader.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "9 updates; checked [2,8]; Torch-only" in result.stdout
