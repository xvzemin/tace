"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx import co3, o2
from equivariantx import o3 as eqx_o3
from equivariantx.tests.o2.testing import _transform
from equivariantx.utils.module import copy_model


@pytest.mark.parametrize("mode", ["single", "pair"])
def test_element_linear_autocast(mode, device):
    linear = o2.Linear(
        "3x0e+3x1m", "2x0e+2x1m", internal_weights=False, shared_weights=False
    ).to(device=device, dtype=torch.float32)
    layer = o2.ElementLinear(linear, backend="torch", element_mode=mode)
    x = torch.randn(
        9,
        linear.irreps_in.dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    shape = (3, 3, linear.weight_numel) if mode == "pair" else (3, linear.weight_numel)
    weight = torch.randn(*shape, device=device, dtype=torch.float32, requires_grad=True)
    source = torch.arange(9, device=device) % 3
    target = source.roll(1) if mode == "pair" else None
    with pytest.raises(TypeError, match="float32 or float64"):
        layer(x, weight, source, target)
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        with pytest.raises(TypeError, match="autocast"):
            layer(x.float(), weight, source, target)
    if x.is_cuda:
        layer.backend = "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with pytest.raises(TypeError, match="autocast"):
                layer(x.float(), weight, source, target)


def test_element_linear_parameters():
    linear = o2.Linear("0e", "0e")
    layer = o2.ElementLinear(linear, backend="torch")
    weight = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    source = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 1, 0, 1])
    x = torch.ones(4, 1)
    torch.testing.assert_close(
        layer(x, weight, source, target), torch.arange(1.0, 5.0).reshape(4, 1)
    )
    source = torch.tensor([0] * 16 + [0, 1, 1])
    target = torch.tensor([0] * 16 + [1, 0, 1])
    x = torch.ones(19, 1, requires_grad=True)
    weight.requires_grad_()
    actual, expected = layer(x, weight, source, target), x * weight[source, target]
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (x, weight)),
        torch.autograd.grad(expected.square().sum(), (x, weight)),
    ):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="target_type"):
        layer(x, weight, source)
    with pytest.raises(ValueError, match="Weights"):
        layer(x, weight.flatten(0, 1), source, target)
    with pytest.raises(ValueError, match="unbiased"):
        o2.ElementLinear(o2.Linear("0e", "0e", biases=True))


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["single", "pair"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("expand_weights", [False, True])
def test_element_linear(backend, device, mode, dtype, expand_weights):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    linear = o2.Linear(
        "2x0ee+0ee+2x0oo+3x1me+2x1mo+2x3me",
        "3x0ee+2x0oo+2x1me+1me+2x1mo+2x3me+2me",
        internal_weights=False,
        shared_weights=False,
    ).to(device=device, dtype=dtype)
    module = o2.ElementLinear(linear, backend, element_mode=mode)
    assert not module.expand_weights
    module.expand_weights = expand_weights
    assert o2.ElementLinear(linear).element_mode == "pair"
    assert o2.ElementLinear(linear).backend == "torch"
    shape = (3, 4, linear.weight_numel) if mode == "pair" else (3, linear.weight_numel)
    weight = torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)
    for count in (0, 7):
        x = torch.randn(
            linear.irreps_in.dim, count, dtype=dtype, device=device
        ).T.requires_grad_()
        source = (torch.arange(2 * count, device=device) % 3)[::2]
        target = torch.arange(count, device=device) % 4 if mode == "pair" else None
        actual = module(x, weight, source, target)
        selected = weight[source, target] if mode == "pair" else weight[source]
        expected = linear(x, selected)
        tolerance = 3e-5 if dtype == torch.float32 else 2e-11
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        for reflected, time_reversal in ((False, False), (True, False), (False, True)):
            torch.testing.assert_close(
                module(
                    _transform(x, linear.irreps_in, 0.37, reflected, time_reversal),
                    weight,
                    source,
                    target,
                ),
                _transform(actual, linear.irreps_out, 0.37, reflected, time_reversal),
                atol=tolerance,
                rtol=tolerance,
            )
        for _ in range(3):
            seed = torch.randn_like(actual)
            gradients = [
                torch.autograd.grad(
                    (y.sin() * seed).sum(), (x, weight), create_graph=True
                )
                for y in (actual, expected)
            ]
            for a, b in zip(*gradients):
                torch.testing.assert_close(a, b, atol=5 * tolerance, rtol=5 * tolerance)
            actual, expected = [
                torch.cat([g.flatten() for g in gs]) for gs in gradients
            ]


@pytest.mark.parametrize("space,experts", [("o3", 1), ("o3", 2), ("co3", 1), ("o2", 1)])
@pytest.mark.parametrize("mode", ["single", "pair"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_element_linear_compile(
    space, experts, mode, backend, device, double_precision
):
    from torch.fx.experimental.proxy_tensor import make_fx

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    group = {"o3": eqx_o3, "co3": co3, "o2": o2}[space]
    linear_cls = o3.Linear if space == "o3" else group.Linear
    linear = linear_cls(
        "2x0e+0e+2x1m+2x2m" if space == "o2" else "2x0e+0e+2x1o+2x2e",
        "3x0e+2x1m+2m+3m" if space == "o2" else "3x0e+2x1o+1o+3o",
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    cls = group.ElementLinear if experts == 1 else group.MoEElementLinear
    module = cls(
        linear,
        backend=backend,
        element_mode=mode,
        **({} if experts == 1 else {"num_experts": experts}),
    )
    reference = copy_model(module)
    reference.backend = "torch"
    reference.expand_weights = True
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    shape = (3, 4) if mode == "pair" else (3,)
    weight = torch.randn(
        *shape, experts, linear.weight_numel, device=device, requires_grad=True
    )
    traced = None
    for count in (5, 9, 9, 0):
        features = torch.randn(
            module.input_dim, count, device=device
        ).T.requires_grad_()
        source = torch.randint(3, (count,), device=device)
        target = torch.randint(4, (count,), device=device) if mode == "pair" else None
        args = (features, weight, source, target)
        if traced is None:
            traced = make_fx(module, tracing_mode="symbolic")(*args)
        expected = reference(*args)
        for evaluate in (module, traced, compiled):
            actual = evaluate(*args)
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
            for a, b in zip(
                torch.autograd.grad(actual.square().sum(), (features, weight)),
                torch.autograd.grad(
                    expected.square().sum(), (features, weight), retain_graph=True
                ),
            ):
                torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "channels,rows,dtype", [(64, 256, torch.float64), (128, 257, torch.float32)]
)
def test_moe_cpu_force_aoti_updates(
    tmp_path, monkeypatch, double_precision, dynamic, channels, rows, dtype
):
    import os
    import subprocess
    import sys

    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config
    from equivariantx.ops.compiler import JointCSE
    from equivariantx.ops.torch.nn.element_linear import reference

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_GROUPED_LINEAR", True)
    module = eqx_o3.MoEElementLinear(
        o3.Linear(
            f"{channels}x0e+{channels}x1o",
            f"{channels}x0e+{channels}x1o",
            internal_weights=False,
            shared_weights=False,
        ),
        4,
        element_mode="pair",
    )
    features = (
        torch.randn(module.input_dim, rows, dtype=dtype).T * 0.1
    ).requires_grad_()
    weight = (
        torch.randn(module.weight_numel, 4, 4, 4, dtype=dtype).permute(1, 2, 3, 0) * 0.1
    ).requires_grad_()
    source = torch.arange(rows) % 4
    target = (torch.arange(rows) // 4) % 4

    def joint(function, x, w, source, target):
        prediction = function(x, w, source, target)
        energy = prediction.sin().sum() / (x.shape[0] + 1)
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + (force - 0.01).square().sum()
        return prediction, force, loss, torch.autograd.grad(loss, w)[0]

    options = {}
    if dynamic:
        count = ShapeVar("nodes", min=256, max=1024, optimization_hint=rows)
        options = {
            "tracing_mode": "fake",
            "dynamic_shapes": ShapesSpec(
                ParamsSpec(
                    {
                        "*args": [
                            TensorSpec((count, None)),
                            TensorSpec((None, None, None, None)),
                            TensorSpec((count,)),
                            TensorSpec((count,)),
                        ]
                    }
                )
            ),
        }
    graph = make_fx(lambda *args: joint(module, *args), **options)(
        features, weight, source, target
    )
    sorted_rows = [
        node.args[0].meta["val"].shape[0]
        for node in graph.graph.nodes
        if node.target == torch.ops.aten.sort.default
    ]
    if dynamic:
        assert len(sorted_rows) == 1 and str(sorted_rows[0]) == "4*u0"
    else:
        assert sorted_rows == [rows]
    inputs = tuple(value.detach() for value in (features, weight, source, target))
    shapes = None
    if dynamic:
        count = torch.export.Dim("nodes", min=256, max=1024)
        shapes = ({0: count}, {}, {0: count}, {0: count})
    exported = torch.export.export(graph, inputs, dynamic_shapes=shapes, strict=True)
    package = tmp_path / "moe.pt2"
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(package),
        inductor_configs={"joint_custom_pre_pass": JointCSE()},
    )

    parameter = torch.nn.Parameter(weight.detach().clone())
    cases = []
    for step, count in enumerate((256, 511, 1024) if dynamic else (rows,) * 3):
        case_features = (
            features
            if count == rows
            else (
                torch.randn(module.input_dim, count, dtype=dtype).T * 0.1
            ).requires_grad_()
        )
        sources = torch.arange(count) % 4
        targets = (torch.arange(count) // 4) % 4
        if step == 1:
            sources = torch.zeros_like(sources)
        elif step == 2:
            sources, targets = sources.roll(3), targets.roll(7)
        values = joint(
            lambda *args: reference(module, *args),
            case_features.double(),
            parameter.double(),
            sources,
            targets,
        )
        expected = tuple(value.detach().clone() for value in values)
        parameter.grad = values[-1].to(parameter)
        torch.nn.utils.clip_grad_norm_([parameter], 0.05, foreach=True)
        with torch.no_grad():
            parameter.add_(parameter.grad, alpha=-0.01)
        cases.append(
            {
                "features": case_features.detach(),
                "source": sources,
                "target": targets,
                "expected": expected,
                "updated": parameter.detach().clone(),
            }
        )
    data = tmp_path / "cases.pt"
    torch.save(
        {"inputs": inputs, "cases": cases, "dynamic": dynamic, "rows": rows}, data
    )
    script = """
import sys
import torch
torch.set_num_threads(4)
runner = torch._inductor.aoti_load_package(sys.argv[1], run_single_threaded=True)
data = torch.load(sys.argv[2], weights_only=True)
x, w, _, _ = data['inputs']
parameter = torch.nn.Parameter(w.clone())
atol, rtol = (3e-6, 3e-4) if w.dtype == torch.float32 else (5e-10, 5e-10)

def interval(value):
    size = 1 + sum((n - 1) * stride for n, stride in zip(value.shape, value.stride()))
    return value.data_ptr(), value.data_ptr() + size * value.element_size()

for case in data['cases']:
    args = case['features'], parameter.detach(), case['source'], case['target']
    saved_inputs = tuple(value.clone() for value in args)
    actual = tuple(runner(*args))
    torch.testing.assert_close(args, saved_inputs, atol=0, rtol=0)
    torch.testing.assert_close(actual, case['expected'], atol=atol, rtol=rtol, check_dtype=False)
    intervals = [interval(value) for value in actual]
    for i, bounds in enumerate(intervals):
        for other in [interval(value) for value in args] + intervals[:i]:
            assert bounds[1] <= other[0] or other[1] <= bounds[0]
    saved_outputs = tuple(value.clone() for value in actual[:-1])
    parameter.grad = actual[-1]
    torch.nn.utils.clip_grad_norm_([parameter], 0.05, foreach=True)
    torch.testing.assert_close(actual[:-1], saved_outputs, atol=0, rtol=0)
    with torch.no_grad():
        parameter.add_(parameter.grad, alpha=-0.01)
    torch.testing.assert_close(parameter, case['updated'], atol=atol, rtol=rtol)
for nodes in (255, 1025) if data['dynamic'] else (data['rows']-1, data['rows']+1):
    bad_x = x.new_empty((x.shape[1], nodes)).T
    bad_types = data['cases'][0]['source'].new_zeros(nodes)
    try:
        runner(bad_x, parameter.detach(), bad_types, bad_types)
    except RuntimeError:
        pass
    else:
        raise AssertionError('The checked package accepted a node count outside its contract')
assert not any(name.startswith(('equivariantx', 'e3nn', 'tace')) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(package), str(data)],
        cwd=tmp_path,
        env={**os.environ, "AOTI_RUNTIME_CHECK_INPUTS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
