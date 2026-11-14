"""Shared CPU frame maps through the public Cartesian LocalFrame module."""

import os
import subprocess
import sys
import zipfile

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.dynamic_spec import (
    ParamsSpec,
    ShapesSpec,
    ShapeVar,
    TensorSpec,
)
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co2, o2
from equivariantx.ops import config
from equivariantx.ops.common.torch import tensor_axis
from equivariantx.ops.torch.co2 import local_frame as kernel
from equivariantx.tests.utils.testing import check_forward_ad


@pytest.fixture(autouse=True)
def matrix_schedule(monkeypatch):
    monkeypatch.setattr(config, "TORCH_CARTESIAN_FRAME_MATRIX", True)
    monkeypatch.setattr(config, "CARTESIAN_FRAME_MATRIX_MIN_CHANNELS", 64)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_TENSOR_AXIS", True)
    monkeypatch.setattr(config, "TORCH_TENSOR_AXIS_ADJOINT", False)


def operation(module, inverse):
    return module.to_global if inverse else module.to_local


def reference(module, inverse, features, rotation):
    direct = kernel.to_global if inverse else kernel.to_local
    return direct(module, features, rotation, contract=tensor_axis.reference)


def inputs(
    module, inverse, feature_shape=(3,), rotation_shape=(3,), dtype=torch.float64
):
    width = module.output_dim if inverse else module.input_dim
    x = (torch.randn(*feature_shape, 2 * width, dtype=dtype) * 0.1)[..., ::2]
    r = (torch.randn(*rotation_shape, 6, 6, dtype=dtype) * 0.2)[..., ::2, ::2]
    return x.requires_grad_(), r.requires_grad_()


def observe_matrix(monkeypatch, inverse):
    name = "global_matrix" if inverse else "local_matrix"
    original = getattr(kernel, name)
    calls = []

    def observed(module, rotation, rank, *args):
        calls.append(rank)
        return original(module, rotation, rank, *args)

    monkeypatch.setattr(kernel, name, observed)
    return calls


def capture(module, inverse, values):
    method = operation(module, inverse)
    return make_fx(
        lambda x, r: method(x, r),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
    )(*values)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    "rank,channels,enabled",
    [
        (0, 64, False),
        (1, 64, False),
        (2, 63, False),
        (2, 64, True),
        (3, 128, True),
        (4, 1024, True),
        (4, 1025, False),
        (5, 64, False),
    ],
)
def test_matrix_policy(rank, channels, enabled, inverse, monkeypatch):
    layer = co2.LocalFrame(f"{channels}x{rank}e").double()
    values = inputs(layer, inverse, (1,), (3,))
    calls = observe_matrix(monkeypatch, inverse)
    operation(layer, inverse)(*values)
    assert calls == []
    graph = capture(layer, inverse, values)
    assert calls == ([rank] if enabled else [])
    torch.testing.assert_close(
        graph(*values), reference(layer, inverse, *values), atol=3e-10, rtol=3e-10
    )


@pytest.mark.parametrize("inverse", [False, True])
def test_matrix_switches_and_explicit_contracts(inverse, monkeypatch):
    layer = co2.LocalFrame("64x3e").double()
    values = inputs(layer, inverse)
    calls = observe_matrix(monkeypatch, inverse)
    for name, setting in (
        ("TORCH_CARTESIAN_FRAME_MATRIX", False),
        ("CARTESIAN_FRAME_MATRIX_MIN_CHANNELS", 128),
        ("TORCH_REFERENCE", True),
        ("TORCH_TENSOR_AXIS", False),
        ("TORCH_TENSOR_AXIS_ADJOINT", True),
    ):
        with monkeypatch.context() as flags:
            flags.setattr(config, name, setting)
            graph = capture(layer, inverse, values)
            torch.testing.assert_close(
                graph(*values),
                reference(layer, inverse, *values),
                atol=3e-10,
                rtol=3e-10,
            )
    direct = kernel.to_global if inverse else kernel.to_local

    def explicit(contract):
        return make_fx(lambda x, r: direct(layer, x, r, contract=contract))(*values)

    for contract in (tensor_axis.reference, tensor_axis.primal, tensor_axis.evaluate):
        graph = explicit(contract)
        torch.testing.assert_close(
            graph(*values), reference(layer, inverse, *values), atol=3e-10, rtol=3e-10
        )
    assert calls == []


@pytest.mark.parametrize("method", ["rotation", "projection"])
@pytest.mark.parametrize("inverse", [False, True])
def test_other_methods_keep_axis_schedule(method, inverse, monkeypatch):
    layer = co2.LocalFrame("64x2o", mmax=1, method=method).double()
    values = inputs(layer, inverse)
    calls = observe_matrix(monkeypatch, inverse)
    graph = capture(layer, inverse, values)
    assert calls == []
    torch.testing.assert_close(
        graph(*values), reference(layer, inverse, *values), atol=3e-10, rtol=3e-10
    )


@pytest.mark.parametrize("inverse", [False, True])
def test_cuda_keeps_axis_schedule(inverse, monkeypatch):
    layer = co2.LocalFrame("64x3e").double()
    calls = observe_matrix(monkeypatch, inverse)
    width = layer.output_dim if inverse else layer.input_dim
    with FakeTensorMode(allow_non_fake_inputs=True):
        x = torch.empty(3, width, device="cuda", dtype=torch.float64)
        r = torch.empty(3, 3, 3, device="cuda", dtype=torch.float64)
        capture(layer, inverse, (x, r))
    assert calls == []


@pytest.mark.parametrize("inverse", [False, True])
def test_mixed_entries_share_rank_maps(inverse, monkeypatch):
    layer = co2.LocalFrame(
        "64x0e+8x2e+64x2o+128x2e+64x3e+1025x2e+64x5o", mmax=2
    ).double()
    values = inputs(layer, inverse, (1,), (3,))
    calls = observe_matrix(monkeypatch, inverse)
    graph = capture(layer, inverse, values)
    assert calls == [2, 3]
    torch.testing.assert_close(
        graph(*values), reference(layer, inverse, *values), atol=3e-10, rtol=3e-10
    )


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("basis_change", [False, True])
@pytest.mark.parametrize("mmax", [0, 1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixed_values(inverse, basis_change, mmax, dtype):
    layer = co2.LocalFrame(
        "64x0e+64x0o+64x1e+64x2o+128x3e+64x4o", mmax, basis_change
    ).to(dtype=dtype)
    values = inputs(layer, inverse, (2, 1), (1, 3), dtype)
    graph = capture(layer, inverse, values)
    tolerance = 3e-5 if dtype == torch.float32 else 3e-10
    torch.testing.assert_close(
        graph(*values),
        reference(layer, inverse, *values),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("nodes", [0, 3])
def test_matrix_five_derivatives_and_forward_ad(inverse, nodes):
    layer = co2.LocalFrame("64x0e+64x2o+64x3e+64x4e", mmax=2).double()
    values = inputs(layer, inverse, (nodes,), (nodes,))
    graph = capture(layer, inverse, values)

    def direct(*args):
        return reference(layer, inverse, *args)

    losses = [function(*values).sin().sum() for function in (graph, direct)]
    tangents = tuple(torch.randn_like(v) * 0.03 for v in values)
    for _ in range(5):
        gradients = [
            tuple(
                value * 0 if gradient is None else gradient + value * 0
                for gradient, value in zip(
                    torch.autograd.grad(
                        loss, values, create_graph=True, allow_unused=True
                    ),
                    values,
                )
            )
            for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-9, rtol=3e-9)
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]
    check_forward_ad(graph, direct, values)

    def nested(function):
        return torch.func.jvp(
            lambda *args: torch.func.jvp(function, args, tangents)[1], values, tangents
        )

    torch.testing.assert_close(nested(graph), nested(direct), atol=3e-9, rtol=3e-9)
    stacked = tuple(torch.stack((v, v * 0.7)) for v in values)
    for dims, batch in (
        ((0, 0), stacked),
        ((0, None), (stacked[0], values[1])),
        ((None, 0), (values[0], stacked[1])),
    ):
        torch.testing.assert_close(
            torch.vmap(graph, in_dims=dims)(*batch),
            torch.vmap(direct, in_dims=dims)(*batch),
            atol=3e-9,
            rtol=3e-9,
        )


def test_empty_irreps_and_legacy_object(tmp_path):
    empty = co2.LocalFrame("").double()
    for inverse in (False, True):
        values = inputs(empty, inverse, (1,), (3,))
        torch.testing.assert_close(capture(empty, inverse, values)(*values), values[0])
    layer = co2.LocalFrame("64x0e+64x3e", mmax=1).double()
    del layer.backend
    path = tmp_path / "old-frame.pt"
    torch.save(layer, path)
    restored = torch.load(path, weights_only=False)
    for inverse in (False, True):
        values = inputs(restored, inverse, (1,), (3,))
        graph = capture(restored, inverse, values)
        torch.testing.assert_close(
            graph(*values),
            reference(restored, inverse, *values),
            atol=3e-10,
            rtol=3e-10,
        )
    assert not hasattr(restored, "backend")


@pytest.mark.parametrize("inverse", [False, True])
def test_dynamic_cpu_force_loss_aoti(inverse, monkeypatch, tmp_path):
    layer = co2.LocalFrame("64x2o+64x3e+64x4e", mmax=2).double()
    width = layer.output_dim if inverse else layer.input_dim

    def joint(x, vector, weight):
        rotation = o2.rotation_matrix_to_y_axis(vector)
        y = operation(layer, inverse)(x * weight, rotation)
        energy = 1 + (y.sin() + 0.1 * y.square()).sum() / (x.shape[0] + 1)
        force = -torch.autograd.grad(energy, vector, create_graph=True)[0]
        loss = energy.square() + (force + 0.01).square().sum()
        return loss, force, torch.autograd.grad(loss, weight)[0]

    def batch(nodes):
        x = (torch.randn(nodes, width * 2, dtype=torch.float64) * 0.1)[:, ::2]
        vector = (
            torch.randn(3, nodes, dtype=torch.float64) * 0.1
            + torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        ).T.requires_grad_()
        return x, vector

    x, vector = batch(7)
    weight = (torch.randn(width, dtype=torch.float64) * 0.1).requires_grad_()
    nodes = ShapeVar("nodes", min=0, optimization_hint=7)
    shapes = ShapesSpec(
        ParamsSpec(
            {
                "x": TensorSpec([nodes, None]),
                "vector": TensorSpec([nodes, None]),
                "weight": TensorSpec([None]),
            }
        )
    )
    graph = make_fx(
        joint, tracing_mode="fake", dynamic_shapes=shapes, _allow_non_fake_inputs=True
    )(x, vector, weight)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    assert not any("eqx." in str(node.target) for node in graph.graph.nodes)
    count = torch.export.Dim("nodes", min=2, max=16)
    exported = torch.export.export(
        graph,
        tuple(v.detach() for v in (x, vector, weight)),
        strict=True,
        dynamic_shapes=({0: count}, {0: count}, None),
    )
    package = tmp_path / "frame.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        assert any(name.endswith(".so") for name in names)
        assert not any(name.endswith((".cubin", ".ptx")) for name in names)
        source = "\n".join(
            archive.read(name).decode() for name in names if name.endswith(".cpp")
        )
        assert source and "eqx::" not in source
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    oracle = make_fx(
        joint, tracing_mode="fake", dynamic_shapes=shapes, _allow_non_fake_inputs=True
    )(x, vector, weight)
    for count in (0, 1, 7, 11):
        values = (*batch(count), weight)
        torch.testing.assert_close(
            graph(*values), oracle(*values), atol=2e-9, rtol=2e-9
        )
    torch.save(oracle, tmp_path / "oracle.pt")
    torch.save(
        (
            weight.detach(),
            [tuple(v.detach() for v in batch(n)) for n in (7, 11, 5)],
            [tuple(v.detach() for v in batch(n)) for n in (0, 1, 17)],
        ),
        tmp_path / "inputs.pt",
    )
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys, torch
torch.set_num_threads(2)
run = torch._inductor.aoti_load_package(sys.argv[1], run_single_threaded=True)
oracle = torch.load(sys.argv[2], weights_only=False)
weight, cases, invalid = torch.load(sys.argv[3], weights_only=True)
expected_weight = weight.clone()
for x, vector in cases:
    for _ in range(3):
        actual = run(x, vector, weight)
        expected = oracle(x, vector, expected_weight)
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        assert all(v.device.type == 'cpu' for v in actual)
        assert torch.count_nonzero(actual[1]) > 0
        previous = weight.clone()
        weight = weight - 0.001 * actual[-1]
        expected_weight = expected_weight - 0.001 * expected[-1]
        torch.testing.assert_close(weight, expected_weight, atol=2e-9, rtol=2e-9)
        assert not torch.equal(weight, previous)
for x, vector in invalid:
    try:
        run(x, vector, weight)
    except RuntimeError as error:
        assert ('too small' if x.shape[0] < 2 else 'too large') in str(error)
    else:
        raise AssertionError('The exported node bounds were not enforced')
assert 'equivariantx' not in sys.modules and 'e3nn' not in sys.modules
""",
            str(package),
            str(tmp_path / "oracle.pt"),
            str(tmp_path / "inputs.pt"),
        ],
        cwd=tmp_path,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "AOTI_RUNTIME_CHECK_INPUTS": "1",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert child.returncode == 0, child.stdout + child.stderr
