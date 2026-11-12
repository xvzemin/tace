"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import conv as eqx_conv
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
    geometry, shared, projected, radial_channels, double_precision
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
    for module in (reference, conv):
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
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-12, rtol=2e-12)
    assert_native_autograd(outputs[0])
    losses = [out.sin().sum() for out in outputs]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=5e-9, rtol=5e-9)
            assert_native_autograd(a)
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
