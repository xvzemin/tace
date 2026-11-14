"""CG providers preserve e3nn paths, weights and mixed derivatives."""

import os
import subprocess
import sys

import pytest
import torch
from e3nn import o3

from equivariantx import o3 as eqx_o3


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("rows", [0, 7])
def test_unweighted_without_weights(backend, rows, device, double_precision):
    if backend == "cuda" and device != "cuda":
        pytest.skip("CUDA unavailable")
    torch.compiler.reset()
    arguments = ("2x1o", "2x1o", "2x0e", [(0, 0, 0, "uuu", False)])
    options = dict(internal_weights=False, shared_weights=False)
    module = eqx_o3.TensorProduct(*arguments, backend=backend, **options).to(device)
    reference = o3.TensorProduct(*arguments, **options).to(device)
    inputs = tuple(
        torch.randn(rows, 6, device=device, requires_grad=True) for _ in range(2)
    )
    actual, expected = module(*inputs), reference(*inputs)
    for _ in range(4):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        actual, expected = [
            torch.cat(
                [
                    value.flatten()
                    for value in torch.autograd.grad(
                        result.sin().sum(), inputs, create_graph=True
                    )
                ]
            )
            / 10
            for result in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    tangents = tuple(torch.randn_like(value) for value in inputs)
    torch.testing.assert_close(
        torch.func.jvp(module, inputs, tangents),
        torch.func.jvp(reference, inputs, tangents),
        atol=2e-10,
        rtol=2e-10,
    )
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(*inputs), reference(*inputs), atol=2e-12, rtol=2e-12
    )


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize(
    "mode,weighted",
    [
        ("uvw", True),
        ("uvu", True),
        ("uvu", False),
        ("uvv", True),
        ("uuw", True),
        ("uuw", False),
        ("uuu", True),
        ("uuu", False),
        ("uvuv", True),
        ("uvuv", False),
        ("uvu<v", True),
        ("uvu<v", False),
        ("u<vw", True),
    ],
)
def test_connection_modes(backend, mode, weighted, device, double_precision):
    if backend == "cuda" and device != "cuda":
        pytest.skip("CUDA unavailable")
    channels1 = 3
    channels2 = 3 if mode in ("uuu", "uuw", "uvu<v", "u<vw") else 2
    channels_out = {
        "uvw": 4,
        "uvu": 3,
        "uvv": 2,
        "uuu": 3,
        "uuw": 4 if weighted else 1,
        "uvuv": 6,
        "uvu<v": 3,
        "u<vw": 4,
    }[mode]
    args = (
        f"{channels1}x1o",
        f"{channels2}x1e",
        f"{channels_out}x2o",
        [(0, 0, 0, mode, weighted)],
    )
    options = dict(internal_weights=False, shared_weights=False)
    reference = o3.TensorProduct(*args, **options).to(device)
    module = eqx_o3.TensorProduct(*args, backend=backend, **options).to(device)
    inputs = tuple(
        torch.randn(*shape, device=device, requires_grad=True) / 3
        for shape in (
            (2, 1, reference.irreps_in1.dim),
            (1, 3, reference.irreps_in2.dim),
            (2, 3, reference.weight_numel),
        )
    )
    actual, expected = module(*inputs), reference(*inputs)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        module.contraction(*inputs), expected, atol=2e-12, rtol=2e-12
    )
    a, b = actual.sin().sum(), expected.sin().sum()
    for _ in range(4):
        ga, gb = [
            torch.autograd.grad(value, inputs, create_graph=True, allow_unused=True)
            for value in (a, b)
        ]
        ga, gb = [
            tuple(
                torch.zeros_like(x) if g is None else g
                for x, g in zip(inputs, gradients)
            )
            for gradients in (ga, gb)
        ]
        torch.testing.assert_close(ga, gb, atol=2e-10, rtol=2e-10)
        a, b = [sum(g.sin().sum() for g in gradients) / 10 for gradients in (ga, gb)]
    tangents = tuple(torch.randn_like(value) for value in inputs)
    torch.testing.assert_close(
        torch.func.jvp(module.contraction, inputs, tangents),
        torch.func.jvp(reference, inputs, tangents),
        atol=2e-10,
        rtol=2e-10,
    )


@pytest.mark.parametrize(
    "name,args",
    [
        ("FullyConnectedTensorProduct", ("2x0e+2x1o", "2x0e+2x1o", "2x0e+2x1o")),
        ("FullTensorProduct", ("2x0e+2x1o", "2x0e+2x1o")),
        ("ElementwiseTensorProduct", ("2x0e+2x1o", "2x0e+2x1o")),
    ],
)
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_factories_and_state(name, args, backend, device, double_precision):
    torch.compiler.reset()
    if backend == "cuda" and device != "cuda":
        pytest.skip("CUDA unavailable")
    reference = getattr(o3, name)(*args).to(device)
    module = getattr(eqx_o3, name)(*args, backend=backend).to(device)
    assert isinstance(module, eqx_o3.TensorProduct)
    module.load_state_dict(reference.state_dict())
    x = torch.randn(4, module.irreps_in1.dim, device=device)
    y = torch.randn(4, module.irreps_in2.dim, device=device)
    expected = reference(x, y)
    torch.testing.assert_close(module(x, y), expected, atol=2e-12, rtol=2e-12)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(x, y), expected, atol=2e-12, rtol=2e-12)
    assert compiled(x[:0], y[:0]).shape == (0, module.irreps_out.dim)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_precision(dtype):
    module = eqx_o3.FullyConnectedTensorProduct("0e", "0e", "0e")
    with pytest.raises(TypeError, match="float32 or float64"):
        module(torch.ones(2, 1, dtype=dtype), torch.ones(2, 1, dtype=dtype))


def test_dtype_roundtrip(double_precision):
    module = eqx_o3.FullyConnectedTensorProduct("2x1o", "2x1e", "2x2o")
    x, y = torch.randn(2, 3, 6)
    weights = module.weight.detach().clone()
    expected = module(x, y)
    module.float().double()
    with torch.no_grad():
        module.weight.copy_(weights)
    torch.testing.assert_close(module(x, y), expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        module.contraction(x, y, module.weight), expected, atol=2e-12, rtol=2e-12
    )


def test_grouped_reference_switch(monkeypatch, device, double_precision):
    if device != "cuda":
        pytest.skip("Compiled CUDA schedule")
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config

    module = eqx_o3.FullyConnectedTensorProduct(
        "2x0e+2x1o", "2x0e+2x1o", "2x0e+2x1o", internal_weights=False
    ).cuda()
    x = torch.randn(512, module.irreps_in1.dim, device="cuda")
    y = torch.randn_like(x)
    weights = torch.randn(module.weight_numel, device="cuda")
    expected = module(x, y, weights)
    graph = make_fx(module)(x, y, weights)
    torch.testing.assert_close(graph(x, y, weights), expected, atol=2e-12, rtol=2e-12)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    reference = make_fx(module)(x, y, weights)
    torch.testing.assert_close(
        reference(x, y, weights), expected, atol=2e-12, rtol=2e-12
    )
    assert str(graph.graph) != str(reference.graph)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("shared", [False, True])
def test_unbacked_joint_package(backend, shared, tmp_path, device, double_precision):
    """One force-loss graph crosses the grouped schedule's batch threshold."""
    if device != "cuda":
        pytest.skip("The grouped CG schedule applies to CUDA tensors")
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    args = ("2x0e+2x1o",) * 3
    options = dict(internal_weights=False, shared_weights=shared)
    module = eqx_o3.FullyConnectedTensorProduct(
        *args, backend=backend, **options
    ).cuda()
    reference = o3.FullyConnectedTensorProduct(*args, **options).cuda()

    def values(rows):
        shapes = [(rows, 8), (rows, 8)]
        shapes.append((module.weight_numel,) if shared else (rows, module.weight_numel))
        return tuple(
            (torch.randn(shape, device="cuda") * 0.1).requires_grad_()
            for shape in shapes
        )

    def training(x, y, weight, ordinary=False):
        operation = reference if ordinary else module
        energy = 1 + operation(x, y, weight).sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + (force + 0.01).square().sum()
        return loss, force, torch.autograd.grad(loss, weight)[0]

    rows = ShapeVar("rows", min=2, optimization_hint=7)
    specs = [TensorSpec([rows, None]), TensorSpec([rows, None])]
    specs.append(TensorSpec([None]) if shared else TensorSpec([rows, None]))
    inputs = values(7)
    graph = make_fx(
        lambda x, y, w: training(x, y, w),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(ParamsSpec(dict(zip(("x", "y", "w"), specs)))),
    )(*inputs)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    rows = torch.export.Dim("rows", min=2)
    shapes = ({0: rows}, {0: rows}, {} if shared else {0: rows})
    exported = torch.export.export(
        graph,
        tuple(value.detach() for value in inputs),
        dynamic_shapes=shapes,
        strict=True,
    )
    package = tmp_path / "cg-dynamic.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    for count in (7, 11, 513):
        inputs = values(count)
        for _ in range(3):
            expected = training(*inputs, ordinary=True)
            records.append(
                dict(
                    inputs=tuple(value.detach().cpu() for value in inputs),
                    outputs=tuple(value.detach().cpu() for value in expected),
                )
            )
            x, y, weight = inputs
            inputs = (
                (x * 0.95).detach().requires_grad_(),
                (y * 0.95).detach().requires_grad_(),
                (weight - 0.01 * expected[-1]).detach().requires_grad_(),
            )
    data = tmp_path / "records.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for record in torch.load(sys.argv[2], weights_only=True):
    actual = run(*(value.cuda() for value in record['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual), record['outputs'], atol=3e-9, rtol=3e-9)
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
