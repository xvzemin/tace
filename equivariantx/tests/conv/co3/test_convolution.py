"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx import conv as eqx_conv
from equivariantx.ops import config
from equivariantx.tests.conv.testing import assert_native_autograd


@pytest.mark.parametrize(
    "geometry,shared,projected,radial_channels",
    [
        (True, False, False, 2),
        (False, False, False, 2),
        (True, True, True, 2),
        (True, False, False, 1024),
        (False, True, True, 1024),
    ],
)
def test_cartesian_convolution_derivatives(
    geometry, shared, projected, radial_channels, double_precision, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3

    tp = co3.TensorProduct(
        "3x1o+2x2e",
        "2x1o+1x2e",
        "3x0e+3x1e+3x2e+2x1o+2x2o+2x3o+3x0e+3x1o+3x2o+3x3o",
        [
            (0, 0, 0, "uvu", True),
            (0, 0, 1, "uvu", True),
            (0, 0, 2, "uvu", True),
            (1, 0, 3, "uvu", True),
            (1, 0, 4, "uvu", True),
            (1, 0, 5, "uvu", True),
            (0, 0, 6, "uvu", False),
            (0, 1, 7, "uvu", True),
            (0, 1, 8, "uvu", True),
            (0, 1, 9, "uvu", True),
        ],
        project=projected,
        shared_weights=False,
        internal_weights=False,
    ).cuda()
    conv = eqx_conv.CartesianTensorProductConv(tp, backend="cuda").cuda()
    reference = eqx_conv.CartesianTensorProductConv(tp, backend="torch").cuda()
    edges = torch.randint(4, (2, 7), device="cuda")
    x = co3.ChangeOfBasis(tp.irreps_in1).cuda()(
        torch.randn(4, tp.irreps_in1.spherical().dim, device="cuda")
    )
    attrs = torch.randn(1 if shared else 7, tp.irreps_in2.dim, device="cuda")
    inputs = [
        x,
        torch.randn(7, 3, device="cuda") if geometry else attrs,
        torch.randn(1 if shared else 7, radial_channels, device="cuda"),
        torch.randn(radial_channels, tp.weight_numel, device="cuda")
        / radial_channels**0.5,
    ]
    if geometry:
        inputs.append(torch.randn(1 if shared else 7, 1, device="cuda"))
    inputs = [x.requires_grad_() for x in inputs]
    outputs = []
    for module, ordinary in ((reference, True), (reference, False), (conv, False)):
        monkeypatch.setattr(config, "TORCH_REFERENCE", ordinary)
        x, y, radial, projection = inputs[:4]
        outputs.append(
            module(
                x,
                None if geometry else y,
                radial,
                projection,
                edges,
                vectors=y if geometry else None,
                amplitudes=inputs[-1] if geometry else None,
            )
        )
    for output in outputs[1:]:
        torch.testing.assert_close(outputs[0], output, atol=2e-12, rtol=2e-12)
    assert_native_autograd(outputs[0])
    losses = [out.sin().sum() for out in outputs]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for a in gradients[0]:
            assert_native_autograd(a)
        for actual in gradients[1:]:
            torch.testing.assert_close(gradients[0], actual, atol=5e-9, rtol=5e-9)
        tangents = [torch.randn_like(x) for x in inputs]
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_cartesian_analytic_convolution(backend, double_precision):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = (
        "2x2e",
        "1x1o",
        "2x1o+2x2o+2x3o",
        [(0, 0, k, "uvu", True) for k in range(3)],
    )
    kwargs = dict(internal_weights=False, shared_weights=False)
    reference = eqx_conv.CartesianTensorProductConv(
        co3.TensorProduct(*args, **kwargs), backend="torch"
    ).to(device)
    module = eqx_conv.CartesianTensorProductConv(
        co3.ictp.TensorProduct(*args, **kwargs), backend=backend
    ).to(device)
    x = torch.randn(4, 10, device=device, requires_grad=True)
    y = torch.randn(7, 3, device=device, requires_grad=True)
    w = torch.randn(7, module.weight_numel, device=device, requires_grad=True)
    cx = co3.ChangeOfBasis(args[0]).to(device)(x)
    edges = torch.randint(4, (2, 7), device=device)
    projection = x.new_empty(0, module.weight_numel)
    actual, expected = [
        layer(cx, y, w, projection, edges) for layer in (module, reference)
    ]
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    losses = [value.sin().sum() for value in (actual, expected)]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, (x, y, w), create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=5e-9, rtol=5e-9)
        tangents = [torch.randn_like(t) for t in (x, y, w)]
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]


@pytest.mark.parametrize("symmetric_inputs", [False, True])
def test_cartesian_convolution_angular_tiles(double_precision, symmetric_inputs):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3

    tp = co3.TensorProduct(
        "2x4e",
        "1o+4e",
        "2x3o+2x4o+2x5o+2x0e",
        [
            (0, 0, 0, "uvu", True),
            (0, 0, 1, "uvu", True),
            (0, 0, 2, "uvu", True),
            (0, 1, 3, "uvu", True),
        ],
        internal_weights=False,
        shared_weights=False,
        project=False,
    ).cuda()
    modules = [
        eqx_conv.CartesianTensorProductConv(
            tp, backend=backend, normalize=False, symmetric_inputs=symmetric_inputs
        ).cuda()
        for backend in ("torch", "cuda")
    ]
    edges = torch.tensor([[0, 1, 0], [1, 0, 1]], device="cuda")
    inputs = [
        torch.randn(2, tp.irreps_in1.dim, device="cuda"),
        torch.randn(3, 3, device="cuda") * 0.3,
        torch.randn(3, 2, device="cuda"),
        torch.randn(2, tp.weight_numel, device="cuda"),
    ]
    inputs[1][0] = 0
    inputs = [x.requires_grad_() for x in inputs]
    x, vectors, radial, projection = inputs
    outputs = [m(x, None, radial, projection, edges, vectors=vectors) for m in modules]
    torch.testing.assert_close(*outputs, atol=2e-12, rtol=2e-12)
    losses = [out.square().sum() for out in outputs]
    for _ in range(2):
        grads = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-10)
        seed = [torch.randn_like(x) for x in inputs]
        losses = [sum((g * s).sum() for g, s in zip(gs, seed)) for gs in grads]


@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("shared_output", [False, True])
@pytest.mark.parametrize("symmetric_inputs", [False, True])
def test_cartesian_output_components(
    degree, shared_output, symmetric_inputs, double_precision
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3

    ir = o3.Irrep(degree, (-1) ** degree)
    tp = co3.TensorProduct(
        f"2x0e+2x{ir}",
        f"{ir}+0e",
        f"0x0e+2x{ir}",
        (
            [(0, 0, 1, "uvu", True), (1, 1, 1, "uvu", True)]
            if shared_output
            else [(0, 0, 1, "uvu", True)]
        ),
        internal_weights=False,
        shared_weights=False,
        project=False,
    ).cuda()
    modules = [
        eqx_conv.CartesianTensorProductConv(
            tp, backend=b, normalize=False, symmetric_inputs=symmetric_inputs
        ).cuda()
        for b in ("torch", "cuda")
    ]
    if shared_output and not symmetric_inputs:
        assert modules[1].harmonic_output_dim == tp.irreps_out.dim
    else:
        assert modules[1].harmonic_output_dim == 2 * (degree + 1) * (degree + 2) // 2
    edges = torch.randint(4, (2, 1025), device="cuda")
    inputs = [
        torch.randn(4, tp.irreps_in1.dim, device="cuda"),
        torch.randn(1025, 3, device="cuda") * 0.2,
        torch.randn(1025, tp.weight_numel, device="cuda"),
    ]
    inputs[1][0] = 0
    inputs = [x.requires_grad_() for x in inputs]
    x, vectors, radial = inputs
    projection = x.new_empty(0, tp.weight_numel)
    values = [m(x, None, radial, projection, edges, vectors=vectors) for m in modules]
    torch.testing.assert_close(*values, atol=2e-11, rtol=2e-11)
    losses = [y.sin().sum() for y in values]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)
        tangents = [torch.randn_like(x) for x in inputs]
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]


@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("symmetric_inputs", [False, True])
def test_cartesian_input_components(degree, symmetric_inputs, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3

    ir = o3.Irrep(degree, (-1) ** degree)
    tp = co3.TensorProduct(
        f"2x{ir}",
        f"2x{ir}",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
        internal_weights=False,
        shared_weights=False,
        project=False,
    ).cuda()
    modules = [
        eqx_conv.CartesianTensorProductConv(
            tp, backend=b, normalize=False, symmetric_inputs=symmetric_inputs
        ).cuda()
        for b in ("torch", "cuda")
    ]
    assert modules[1].pack_inputs
    edges = torch.randint(4, (2, 19), device="cuda")
    # Deliberately not symmetric: packing must also preserve ambient derivatives.
    inputs = [
        torch.randn(tp.irreps_in1.dim, 4, device="cuda").T,
        torch.randn(19, 3, device="cuda") * 0.2,
        torch.randn(19, tp.weight_numel, device="cuda"),
        torch.randn(19, 2, device="cuda"),
    ]
    inputs[1][0] = 0
    inputs = [x.requires_grad_() for x in inputs]
    x, vectors, radial, amplitudes = inputs
    projection = x.new_empty(0, tp.weight_numel)
    values = [
        m(x, None, radial, projection, edges, vectors=vectors, amplitudes=amplitudes)
        for m in modules
    ]
    torch.testing.assert_close(*values, atol=2e-12, rtol=2e-12)
    losses = [y.sin().sum() for y in values]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
        tangents = [torch.randn_like(x) for x in inputs]
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]


@pytest.mark.parametrize("variant", ["ictd", "ictp", "raw"])
@pytest.mark.parametrize("child_backend", ["torch", "cuda"])
@pytest.mark.parametrize("indexed", [False, True])
def test_cartesian_reference_independent(
    variant, child_backend, indexed, double_precision, monkeypatch
):
    from equivariantx import co3

    factory = co3.ictp.TensorProduct if variant == "ictp" else co3.TensorProduct
    options = {} if variant == "ictp" else {"project": variant != "raw"}
    tp = factory(
        "2x1o",
        "1x1o",
        "2x0e+2x1e+2x2e",
        [
            (0, 0, 0, "uvu", True),
            (0, 0, 1, "uvu", True),
            (0, 0, 2, "uvu", True),
            (0, 0, 0, "uvu", False),
        ],
        backend=child_backend,
        internal_weights=False,
        shared_weights=False,
        **options,
    )
    module = eqx_conv.CartesianTensorProductConv(tp, backend="torch")
    values = tuple(
        (torch.randn(*shape) * 0.1).requires_grad_()
        for shape in ((4, 6), (1, 3), (1, tp.weight_numel))
    )
    x, attrs, weights = values
    projection = x.new_empty(0, tp.weight_numel)
    index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]]) if indexed else None
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    optimized = module(x, attrs, weights, projection, index)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    reference = module(x, attrs, weights, projection, index)
    assert tp.backend == child_backend
    assert_native_autograd(reference)
    torch.testing.assert_close(reference, optimized, atol=2e-12, rtol=2e-12)
    losses = []
    for value in (reference, optimized):
        energy = 1 + value.square().mean()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        losses.append(energy.square() + (force - 0.01).square().sum())
    directions = tuple(torch.randn_like(value) * 0.1 for value in values)
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, values, create_graph=True) for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=2e-10, rtol=2e-10)
        for gradient in gradients[0]:
            assert_native_autograd(gradient)
        losses = [
            sum((g * d).sum() for g, d in zip(gs, directions)) for gs in gradients
        ]


def test_cartesian_reference_dynamic_aoti(tmp_path, double_precision, monkeypatch):
    import subprocess
    import sys

    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx import co3

    tp = co3.TensorProduct(
        "2x1o",
        "1x1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, k, "uvu", True) for k in range(3)],
        backend="cuda",
        internal_weights=False,
        shared_weights=False,
    )
    module = eqx_conv.CartesianTensorProductConv(tp, backend="torch")
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)

    def joint(x, attrs, weight):
        value = module(x, attrs, weight, x.new_empty(0, tp.weight_numel), None)
        energy = 1 + value.square().mean()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + (force - 0.01).square().sum()
        return loss, force, torch.autograd.grad(loss, weight)[0]

    def inputs(rows):
        return tuple(
            (torch.randn(*shape) * 0.1).requires_grad_()
            for shape in ((rows, 6), (rows, 3), (1, tp.weight_numel))
        )

    values = inputs(7)
    count = ShapeVar("rows", min=2, max=16, optimization_hint=7)
    graph = make_fx(
        joint,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(
            ParamsSpec(
                {
                    "x": TensorSpec([count, None]),
                    "attrs": TensorSpec([count, None]),
                    "weight": TensorSpec([None, None]),
                }
            )
        ),
    )(*values)
    dynamic = torch.export.Dim("rows", min=2, max=16)
    exported = torch.export.export(
        graph,
        tuple(value.detach() for value in values),
        strict=True,
        dynamic_shapes=({0: dynamic}, {0: dynamic}, None),
    )
    package = tmp_path / "cartesian-reference.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    cases = []
    weight = values[-1]
    for rows in (7, 11, 5):
        x, attrs, _ = inputs(rows)
        with torch.no_grad():
            weight.add_(0.001)
        expected = joint(x, attrs, weight)
        actual = graph(x, attrs, weight)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        cases.append(
            {
                "inputs": tuple(value.detach().clone() for value in (x, attrs, weight)),
                "expected": tuple(value.detach().clone() for value in expected),
            }
        )
    data = tmp_path / "cases.pt"
    torch.save(cases, data)
    script = """
import sys
import torch
torch.set_num_threads(4)
runner = torch._inductor.aoti_load_package(sys.argv[1])
for case in torch.load(sys.argv[2], weights_only=True):
    actual = runner(*case['inputs'])
    torch.testing.assert_close(tuple(actual), case['expected'], atol=2e-10, rtol=2e-10)
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
