"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx.mbe.ace import (
    BilinearContraction,
)

from .testing import (
    check_cpu_dynamic_coefficients,
    check_parent_aoti,
    check_parent_provider,
)


@pytest.mark.parametrize("space", ["o3", "co3"])
@pytest.mark.parametrize("experts", [1, 2])
def test_cpu_dynamic_coefficients(space, experts, monkeypatch):
    check_cpu_dynamic_coefficients(space, "bilinear", monkeypatch, experts)


def test_cpu_bounded_coefficients(monkeypatch):
    check_cpu_dynamic_coefficients("co3", "bilinear", monkeypatch, 2, dynamic=False)


@pytest.mark.parametrize("shared_weights", [False, True])
def test_force_gradient_aoti(shared_weights, tmp_path, double_precision, monkeypatch):
    check_parent_aoti(
        "o3", "bilinear", tmp_path, monkeypatch, shared_weights=shared_weights
    )


@pytest.mark.parametrize("device,nodes", [("cpu", 0), ("cpu", 7), ("cuda", 7)])
@pytest.mark.parametrize("reference", [False, True])
def test_shared_descriptors(device, nodes, reference, double_precision, monkeypatch):
    check_parent_provider(
        "o3",
        "bilinear",
        device,
        nodes,
        "torch",
        reference,
        monkeypatch,
        shared_weights=True,
    )


@pytest.mark.parametrize("device,nodes", [("cpu", 0), ("cpu", 7), ("cuda", 7)])
@pytest.mark.parametrize("child_backend", ["torch", "cuda"])
@pytest.mark.parametrize("reference", [False, True])
def test_parent_provider(
    device, nodes, child_backend, reference, double_precision, monkeypatch
):
    check_parent_provider(
        "o3", "bilinear", device, nodes, child_backend, reference, monkeypatch
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("experts", [1, 2, 4])
@pytest.mark.parametrize("nodes", [0, 3, 9])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("degree", [1, 3])
def test_bilinear_contraction(device, experts, nodes, shared, degree):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = o3.TensorProduct(
        f"4x0e+4x{degree}o",
        f"4x0e+4x{degree}o+4x0e",
        f"4x0e+4x0e+4x{degree}o",
        [(0, 0, 0, "uuu", True), (1, 1, 1, "uuu", True), (1, 2, 2, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    linear = o3.Linear(
        [(mul // experts, ir) for mul, ir in tp.irreps_out.simplify()],
        f"{4 // experts}x0e+{4 // experts}x{degree}o",
        internal_weights=False,
        shared_weights=False,
    )
    shared_linear = (
        o3.Linear(
            tp.irreps_out.simplify(),
            f"4x0e+4x{degree}o",
            internal_weights=False,
            shared_weights=False,
        )
        if shared
        else None
    )
    module = BilinearContraction(
        tp,
        linear,
        experts,
        backend="cuda" if device == "cuda" else "torch",
        shared_linear=shared_linear,
    ).to(device=device, dtype=torch.float64)
    x, y, gates = [
        torch.randn(nodes, 2 * size, dtype=torch.float64, device=device)[
            :, ::2
        ].requires_grad_()
        for size in (tp.irreps_in1.dim, tp.irreps_in2.dim, tp.weight_numel)
    ]
    weight = torch.randn(
        3,
        experts,
        linear.weight_numel,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    types = torch.arange(nodes, device=device) % 3
    shared_weight = (
        torch.randn(
            shared_linear.weight_numel,
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        if shared
        else None
    )
    actual = module(x, y, gates, weight, types, shared_weight)
    # Compare against the existing tensor product and independent coefficient map.
    module.coefficients.backend = "torch"
    expected = module.coefficients(tp(x, y, gates), weight, types)
    inputs = (x, y, gates, weight)
    if shared:
        expected = expected + shared_linear(
            tp(x, y, gates), shared_weight.expand(nodes, -1)
        )
        inputs = (*inputs, shared_weight)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    gradients = [
        torch.autograd.grad(
            value.sin().sum(), (x, y), create_graph=True, retain_graph=True
        )
        for value in (actual, expected)
    ]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    mixed = [
        torch.autograd.grad(
            sum(value.square().sum() for value in grad),
            inputs,
            retain_graph=True,
        )
        for grad in gradients
    ]
    for a, b in zip(*mixed):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-9)
    losses = [value.square().sum() for value in (actual, expected)]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(
                a, b, atol=1e-8, rtol=1e-9, msg=f"Derivative order {_ + 1}"
            )
        losses = [sum(value.square().sum() for value in grad) for grad in gradients]
    assert not module.state_dict()
    if device == "cuda" and nodes and experts == 2:
        compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(x, y, gates, weight, types, shared_weight),
            expected,
            atol=1e-11,
            rtol=1e-11,
        )


def shared_coefficient_module(channels=64):
    irreps = f"{channels}x0e"
    product = o3.TensorProduct(
        irreps,
        irreps,
        irreps,
        [(0, 0, 0, "uuu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    coefficient = o3.Linear(
        irreps, irreps, internal_weights=False, shared_weights=False
    )
    return BilinearContraction(
        product, coefficient, backend="torch", shared_linear=coefficient
    ).double()


@pytest.mark.parametrize(
    "nodes,elements,channels,expected",
    [
        (7, 1, 64, False),
        (256, 1, 32, False),
        (256, 1, 64, True),
        (256, 16, 64, True),
        (256, 17, 64, False),
    ],
)
def test_shared_coefficient_policy(nodes, elements, channels, expected, monkeypatch):
    from equivariantx.ops import config
    from equivariantx.ops.torch.mbe.ace import bilinear

    module = shared_coefficient_module(channels)
    features = torch.empty(nodes, channels, dtype=torch.float32)
    weight = torch.empty(elements, module.weight_numel, dtype=torch.float32)
    monkeypatch.setattr(config, "TORCH_SHARED_COEFFICIENTS", True)
    assert not bilinear.shared_coefficients_available(module, features, weight)
    monkeypatch.setattr(bilinear, "get_proxy_mode", lambda: object())
    assert bilinear.shared_coefficients_available(module, features, weight) is expected
    monkeypatch.setattr(config, "SHARED_COEFFICIENTS_MAX_BYTES", weight.numel() * 4)
    assert not bilinear.shared_coefficients_available(module, features, weight.double())
    monkeypatch.setattr(config, "TORCH_SHARED_COEFFICIENTS", False)
    assert not bilinear.shared_coefficients_available(module, features, weight)


def test_shared_coefficient_unbacked_fallback(monkeypatch):
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config
    from equivariantx.ops.torch.mbe.ace import bilinear

    module = shared_coefficient_module()
    count = ShapeVar("nodes", min=2, max=1024, optimization_hint=7)
    weight = torch.randn(1, module.weight_numel)
    monkeypatch.setattr(config, "TORCH_SHARED_COEFFICIENTS", True)

    def dispatch(x, w):
        return x + 1 if bilinear.shared_coefficients_available(module, x, w) else x * 2

    graph = make_fx(
        dispatch,
        tracing_mode="fake",
        dynamic_shapes=ShapesSpec(
            ParamsSpec(
                {
                    "x": TensorSpec([count, None]),
                    "w": TensorSpec([None, None]),
                }
            )
        ),
    )(torch.randn(7, 64), weight)
    for nodes in (7, 11, 256, 513):
        value = torch.randn(nodes, 64)
        torch.testing.assert_close(graph(value, weight), value * 2)


@pytest.mark.parametrize("elements", [1, 16])
def test_shared_coefficient_noncontiguous_capture(
    elements, double_precision, monkeypatch
):
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config

    module = shared_coefficient_module()
    shapes = ((256, 64), (256, 64), (256, 64), (elements, 1, 4096), (64, 64))
    values = tuple(
        (torch.randn(*shape, 2) * 0.1)[..., 0].requires_grad_() for shape in shapes
    )
    types = torch.arange(256) % elements

    def joint(*args):
        x, base, gates, weight, shared = args
        value = module(x, base, gates, weight, types, shared)
        energy = value.square().mean()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy + (force - 0.01).square().sum()
        return energy, force, *torch.autograd.grad(loss, args)

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_SHARED_COEFFICIENTS", True)
    graph = make_fx(joint)(*values)
    actual = graph(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = joint(*values)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)


def test_shared_coefficient_aoti_storage_clip(tmp_path, double_precision, monkeypatch):
    import subprocess
    import sys

    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config
    from equivariantx.ops.compiler import JointCSE

    module = shared_coefficient_module()
    shapes = ((256, 64), (256, 64), (256, 64), (1, 1, 4096), (64, 64))
    initial = tuple(torch.randn(shape) * 0.1 for shape in shapes)
    types = torch.zeros(256, dtype=torch.long)

    def joint(x, base, gates, weight, shared, indices):
        value = module(x, base, gates, weight, indices, shared)
        energy = value.square().mean()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy + (force - 0.01).square().sum()
        return energy, force, *torch.autograd.grad(loss, (weight, shared))

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_SHARED_COEFFICIENTS", True)
    values = tuple(value.clone().requires_grad_() for value in initial)
    graph = make_fx(joint)(*values, types)
    inputs = (*[value.detach() for value in values], types)
    exported = torch.export.export(graph, inputs, strict=True)
    package = tmp_path / "bilinear.pt2"
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(package),
        inductor_configs={"joint_custom_pre_pass": JointCSE()},
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    cases = []
    for method in ("sequential", "foreach", "clip_grad_norm"):
        state = tuple(value.clone().requires_grad_() for value in initial)
        steps = []
        for _ in range(3):
            result = joint(*state, types)
            expected = tuple(value.detach().clone() for value in result)
            gradients = list(result[2:])
            maximum = (
                torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]))
                * 0.37
            )
            for parameter, gradient in zip(state[3:], gradients):
                parameter.grad = gradient
            if method == "sequential":
                for gradient in gradients:
                    gradient.mul_(0.37)
            elif method == "foreach":
                torch._foreach_mul_(gradients, 0.37)
            else:
                torch.nn.utils.clip_grad_norm_(state[3:], maximum, foreach=True)
            with torch.no_grad():
                for parameter, gradient in zip(state[3:], gradients):
                    parameter.add_(gradient, alpha=-0.03)
            steps.append(
                {
                    "expected": expected,
                    "maximum": maximum,
                    "gradients": tuple(g.detach().clone() for g in gradients),
                    "parameters": tuple(p.detach().clone() for p in state[3:]),
                }
            )
        cases.append({"method": method, "steps": steps})
    data = tmp_path / "cases.pt"
    torch.save({"initial": initial, "types": types, "cases": cases}, data)
    script = """
import sys
import torch
torch.set_num_threads(4)
runner = torch._inductor.aoti_load_package(sys.argv[1])
data = torch.load(sys.argv[2], weights_only=True)
def overlap(a, b):
    if a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr():
        return False
    def extent(x):
        start = x.storage_offset() * x.element_size()
        end = start + (1 + sum((n - 1) * s for n, s in zip(x.shape, x.stride()))) * x.element_size()
        return start, end
    a0, a1 = extent(a)
    b0, b1 = extent(b)
    return max(a0, b0) < min(a1, b1)
for case in data['cases']:
    state = [value.clone() for value in data['initial']]
    parameters = [torch.nn.Parameter(value) for value in state[3:]]
    for step in case['steps']:
        with torch.no_grad():
            actual = runner(*state, data['types'])
        torch.testing.assert_close(tuple(actual), step['expected'], atol=2e-10, rtol=2e-9)
        gradients = list(actual[2:])
        assert not overlap(*gradients)
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient
        if case['method'] == 'sequential':
            for gradient in gradients:
                gradient.mul_(0.37)
        elif case['method'] == 'foreach':
            torch._foreach_mul_(gradients, 0.37)
        else:
            torch.nn.utils.clip_grad_norm_(parameters, step['maximum'], foreach=True)
        torch.testing.assert_close(tuple(gradients), step['gradients'], atol=2e-10, rtol=2e-9)
        for parameter, gradient in zip(state[3:], gradients):
            parameter.add_(gradient, alpha=-0.03)
        torch.testing.assert_close(tuple(state[3:]), step['parameters'], atol=2e-10, rtol=2e-9)
assert not any(name.startswith('equivariantx') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(package), str(data)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
