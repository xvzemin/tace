"""All native symmetric methods in independently loaded force-loss packages."""

import os
import subprocess
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.co3.symmetric import SymmetricBasis
from equivariantx.ops import config
from equivariantx.tests.utils.testing import package_script

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native symmetric maps require CUDA"
)


@pytest.mark.parametrize("rank", [2, 4])
def test_joint_aoti(rank, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SYMMETRIC_INLINE", False)
    layers = {
        backend: SymmetricBasis(rank, backend=backend).double().cuda()
        for backend in ("torch", "cuda")
    }

    def operation(x, v, w, ordinary=False):
        layer = layers["torch" if ordinary else "cuda"]
        n = torch.nn.functional.normalize(v, dim=-1)
        plane = torch.eye(3, device=x.device, dtype=x.dtype) - n.unsqueeze(
            -1
        ) * n.unsqueeze(-2)
        entries = torch.cat(
            (plane[..., 0, :], plane[..., 1, 1:], plane[..., 2, 2:]), -1
        )
        y = layer(layer.pack(x * w), plane)
        lifted = layer.multiply(layer.trace(y), entries)
        energy = (
            (y + lifted).sin().sum()
            + layer.unpack(y).square().sum() * 0.01
            + layer.contract(y, n).sin().sum() * 0.01
        )
        force = -torch.autograd.grad(energy, v, create_graph=True)[0]
        loss = force.square().mean() + energy.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    x = (
        torch.randn(2, 2, 3**rank, device="cuda", dtype=torch.float64) * 0.1
    ).requires_grad_()
    v = (
        torch.randn(2, 1, 3, device="cuda", dtype=torch.float64) + 0.5
    ).requires_grad_()
    w = (torch.randn(2, 1, device="cuda", dtype=torch.float64) * 0.1).requires_grad_()
    values = x, v, w
    graph = make_fx(lambda *args: operation(*args))(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph, tuple(t.detach() for t in values), strict=True
    )
    package = tmp_path / "symmetric.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for _ in range(3):
        expected = operation(x, v, w, True)
        records.append(
            {
                "inputs": tuple(t.detach().cpu() for t in (x, v, w)),
                "outputs": tuple(t.detach().cpu() for t in expected),
            }
        )
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()
        v = (v.detach() * 0.9 + 0.1).requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            package_script("""
import sys
import torch
run=torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2],weights_only=True):
    with torch.no_grad():actual=run(*(v.cuda() for v in row['inputs']))
    torch.testing.assert_close(tuple(v.cpu() for v in actual),row['outputs'],atol=3e-9,rtol=3e-9)
assert 'equivariantx' in sys.modules
"""),
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("rank", [2, 3])
def test_native_dynamic_inductor(rank, monkeypatch):
    from equivariantx.ops.torch.co3 import symmetric as reference

    monkeypatch.setattr(config, "SYMMETRIC_INLINE", False)
    layer = SymmetricBasis(rank, backend="cuda").double().cuda()

    def run(x, p):
        return layer.unpack(layer(layer.pack(x), p))

    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = torch.randn(count, 2, 3**rank, device="cuda", dtype=torch.float64) * 0.1
        p = torch.randn(count, 1, 3, 3, device="cuda", dtype=torch.float64) * 0.1
        expected = reference.unpack_factorized(
            layer,
            reference.transform_compact(layer, reference.pack_factorized(layer, x), p),
        )
        torch.testing.assert_close(compiled(x, p), expected, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("rank", [2, 4])
def test_unbacked_joint_and_aoti_bounds(rank, monkeypatch, tmp_path):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    monkeypatch.setattr(config, "SYMMETRIC_INLINE", False)
    layer = SymmetricBasis(rank, backend="cuda").double().cuda()

    def joint(x, v, w):
        direction = torch.nn.functional.normalize(v, dim=-1)
        plane = torch.eye(3, device=x.device, dtype=x.dtype) - direction.unsqueeze(
            -1
        ) * direction.unsqueeze(-2)
        y = layer.unpack(layer(layer.pack(x * w), plane.unsqueeze(-3)))
        energy = 1 + y.sin().sum() * 0.01
        force = -torch.autograd.grad(energy, v, create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        return energy, force, loss, torch.autograd.grad(loss, w)[0]

    def inputs(count):
        return (
            (
                torch.randn(count, 2, 3**rank, device="cuda", dtype=torch.float64) * 0.1
            ).requires_grad_(),
            (
                torch.randn(count, 3, device="cuda", dtype=torch.float64) + 0.5
            ).requires_grad_(),
        )

    x, v = inputs(7)
    weight = (
        torch.randn(2, 3**rank, device="cuda", dtype=torch.float64) * 0.1
    ).requires_grad_()
    package = tmp_path / "dynamic-symmetric.pt2"
    rows = ShapeVar("rows", min=0)
    graph = make_fx(
        joint,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(
            ParamsSpec(
                {
                    "x": TensorSpec((rows, None, None)),
                    "v": TensorSpec((rows, None)),
                }
            )
        ),
    )(x, v, weight)
    assert any(
        str(node.target) == "eqx.symmetric_map.default" for node in graph.graph.nodes
    )
    count = torch.export.Dim("rows", min=2, max=8192)
    exported = torch.export.export(
        graph,
        tuple(value.detach() for value in (x, v, weight)),
        strict=True,
        dynamic_shapes=({0: count}, {0: count}, None),
    )
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    initial = weight.detach().cpu().clone()
    records = []
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (7, 0, 1, 11):
        x, v = inputs(count)
        for _ in range(3):
            expected = joint(x, v, weight)
            torch.testing.assert_close(
                graph(x, v, weight), expected, atol=3e-10, rtol=3e-10
            )
            if count >= 2:
                records.append(
                    {
                        "inputs": tuple(value.detach().cpu() for value in (x, v)),
                        "outputs": tuple(value.detach().cpu() for value in expected),
                    }
                )
                weight = (weight - 0.01 * expected[-1]).detach().requires_grad_()
    data = tmp_path / "dynamic-symmetric-data.pt"
    torch.save((initial, records), data)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            package_script("""
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
weight, records = torch.load(sys.argv[2], weights_only=True)
weight = weight.cuda()
for row in records:
    x, v = (value.cuda() for value in row['inputs'])
    saved = tuple(value.clone() for value in (x, v, weight))
    actual = run(x, v, weight)
    torch.testing.assert_close(tuple(value.cpu() for value in actual), row['outputs'],
                               atol=3e-10, rtol=3e-10)
    gradient = actual[-1].clone()
    previous = tuple(value.clone() for value in actual[:-1])
    actual[-1].mul_(0.125)
    torch.testing.assert_close((x, v, weight), saved, atol=0, rtol=0)
    torch.testing.assert_close(actual[:-1], previous, atol=0, rtol=0)
    updated = weight - 0.01 * gradient
    if x.numel():
        assert torch.count_nonzero(actual[1]) > 0
        assert not torch.equal(updated, weight)
    else:
        torch.testing.assert_close(updated, weight, atol=0, rtol=0)
    weight = updated
assert torch.isfinite(weight).all()
for count in (0, 1, 8193):
    try:
        run(x.new_empty(count, *x.shape[1:]), v.new_empty(count, *v.shape[1:]), weight)
    except RuntimeError as error:
        expected = 'too small' if count < 2 else 'too large'
        assert expected in str(error)
    else:
        raise AssertionError('The exported input bounds were not enforced')
assert 'equivariantx' in sys.modules
"""),
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={
            **{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            "AOTI_RUNTIME_CHECK_INPUTS": "1",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_forward_ad():
    from equivariantx.tests.utils.testing import check_forward_ad

    layers = {
        backend: SymmetricBasis(3, backend=backend).double().cuda()
        for backend in ("torch", "cuda")
    }

    def operation(x, p, backend):
        layer = layers[backend]
        return layer.unpack(layer(layer.pack(x), p))

    x = torch.randn(2, 2, 27, device="cuda", dtype=torch.float64).requires_grad_()
    p = torch.randn(2, 1, 3, 3, device="cuda", dtype=torch.float64).requires_grad_()
    check_forward_ad(
        lambda x, p: operation(x, p, "cuda"),
        lambda x, p: operation(x, p, "torch"),
        (x, p),
    )
