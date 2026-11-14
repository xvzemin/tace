"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx import co2, co3, o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", ["direct", "projection"])
def test_reference_backend_routing(device, method, monkeypatch, double_precision):
    import copy
    from unittest.mock import patch

    from equivariantx.ops import config
    from equivariantx.ops.cuda.co2 import local_frame as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = co2.LocalFrame("2x2e", method=method, backend="cuda").to(device)
    reference = copy.deepcopy(module)
    reference.backend = "torch"
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (3, 0):
        x = torch.randn(count, 18, device=device)
        rotation = torch.randn(count, 3, 3, device=device)
        with patch.object(kernel, "to_local", wraps=kernel.to_local) as selected:
            actual = module(x, rotation)
        assert selected.call_count == (device == "cuda")
        torch.testing.assert_close(actual, reference(x, rotation))
        with patch.object(kernel, "to_global", wraps=kernel.to_global) as selected:
            actual = module.to_global(actual, rotation)
        assert selected.call_count == (device == "cuda")
        torch.testing.assert_close(
            actual, reference.to_global(reference(x, rotation), rotation)
        )
    assert module.backend == "cuda" and config.TORCH_REFERENCE is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_projection_reference_bypasses_children(device, monkeypatch, double_precision):
    from equivariantx.ops import config
    from equivariantx.tests.conv.testing import assert_native_autograd

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = co2.LocalFrame("2x2e", method="projection").to(device)
    reference = co2.LocalFrame("2x2e", method="projection").to(device)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)

    def forbidden(*args, **kwargs):
        raise AssertionError("A reference dispatched its restriction child.")

    for child in module.restrictions.values():
        child.backend = "cuda"
        monkeypatch.setattr(child, "forward", forbidden)
        monkeypatch.setattr(child, "inverse", forbidden)
    for count in (3, 0):
        values = (
            (torch.randn(count, 18, device=device) * 0.1).requires_grad_(),
            (torch.randn(count, 3, 3, device=device) * 0.1).requires_grad_(),
        )

        def run(layer, x, rotation):
            return layer.to_global(layer(x, rotation).sin(), rotation)

        actual, expected = run(module, *values), run(reference, *values)
        for _ in range(4):
            torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
            assert_native_autograd(actual)
            actual, expected = [
                sum(
                    g.sin().sum()
                    for g in torch.autograd.grad(
                        v.sin().sum(), values, create_graph=True
                    )
                )
                for v in (actual, expected)
            ]
        tangents = tuple(torch.randn_like(value) for value in values)
        torch.testing.assert_close(
            torch.func.jvp(lambda *xs: run(module, *xs), values, tangents),
            torch.func.jvp(lambda *xs: run(reference, *xs), values, tangents),
        )
    assert all(child.backend == "cuda" for child in module.restrictions.values())
    assert config.TORCH_REFERENCE is True


def test_transform_reference_preserves_cuda(monkeypatch, double_precision):
    from unittest.mock import patch

    from equivariantx.co2.local_frame import transform_tensor
    from equivariantx.ops import config
    from equivariantx.ops.cuda.co2 import local_frame as kernel

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    x, rotation = torch.randn(3, 9, device="cuda"), torch.randn(3, 2, 3, device="cuda")
    with patch.object(
        kernel, "transform_tensor", wraps=kernel.transform_tensor
    ) as selected:
        actual = transform_tensor(x, rotation, 2, backend="cuda")
    assert selected.call_count == 1
    torch.testing.assert_close(actual, transform_tensor(x, rotation, 2))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_method_precision(device, double_precision):
    from equivariantx.tests.utils.testing import check_method_precision

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layer = co2.LocalFrame("2x1o+2x2e").to(device)
    rotation = torch.eye(3, device=device).repeat(3, 1, 1)
    features = co3.Irreps("2x1o+2x2e").randn(3, -1, device=device)
    local = layer(features, rotation)
    check_method_precision(
        layer,
        [
            ("to_local", (features, rotation), {}),
            ("to_global", (local, rotation), {}),
            ("rotate", (features, rotation), {}),
        ],
    )


@pytest.mark.parametrize("basis_change", [False, True])
@pytest.mark.parametrize("mmax", [1, 4])
@pytest.mark.parametrize("method", ["direct", "rotation", "projection"])
@pytest.mark.parametrize("time_reversal", [False, True])
def test_cartesian_local_frame(
    double_precision, basis_change, mmax, method, time_reversal
):
    if time_reversal and not hasattr(o3.Irrep("0e"), "t"):
        pytest.skip("Requires time-reversal e3nn")
    irreps = o3.Irreps(
        "2x0eo+2x1oo+1x1ee+2x2oe+1x4eo" if time_reversal else "2x0e+2x1o+1x1e+2x2o+1x4e"
    )
    frame = co2.LocalFrame(irreps, mmax, basis_change, method=method).to(DEVICE)
    reference = o2.LocalFrame(irreps, mmax=mmax, basis_change=basis_change).to(DEVICE)
    embed = co3.ChangeOfBasis(irreps).to(DEVICE)
    circular = co2.ChangeOfBasis(frame.irreps_out, inverse=True).to(DEVICE)
    x = torch.randn(3, irreps.dim, device=DEVICE, requires_grad=True)
    vectors = torch.randn(3, 3, device=DEVICE, requires_grad=True)
    rotation = o2.rotation_matrix_to_y_axis(vectors)
    wigner = o2.WignerD(4, 4, method="recursive").to(DEVICE).forward_packed(vectors)
    ir_mul = torch.cat(
        [
            x[:, section].reshape(3, mul, ir.dim).transpose(-1, -2).flatten(1)
            for (mul, ir), section in zip(irreps, irreps.slices())
        ],
        -1,
    )
    local = frame(embed(x), rotation)
    if time_reversal:
        signs = x.new_tensor([ir.t for mul, ir in irreps for _ in range(mul * ir.dim)])
        transformed = frame(embed(x * signs), rotation)
        reverse = frame.irreps_out.D_from_angle(x.new_tensor(0), time_reversal=True)
        torch.testing.assert_close(
            transformed, local @ reverse.T, atol=3e-11, rtol=3e-11
        )
    actual, expected = circular(local), reference(ir_mul, wigner)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    if mmax == 4:
        torch.testing.assert_close(
            frame.to_global(local, rotation), embed(x), atol=3e-11, rtol=3e-11
        )
    projection = co2.LocalFrame(irreps, mmax, basis_change, method="projection").to(
        DEVICE
    )
    torch.testing.assert_close(
        frame.to_global(local, rotation),
        projection.to_global(local, rotation),
        atol=3e-11,
        rtol=3e-11,
    )
    torch.testing.assert_close(
        frame.rotate(frame.rotate(embed(x), rotation), rotation.transpose(-1, -2)),
        embed(x),
        atol=3e-11,
        rtol=3e-11,
    )
    for _ in range(2):
        probe = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    grad.flatten()
                    for grad in torch.autograd.grad(
                        value.sin(), (x, vectors), probe, create_graph=True
                    )
                ]
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
    assert frame(embed(x[:0]), rotation[:0]).shape == (0, frame.output_dim)
    assert frame.to_global(local[:0], rotation[:0]).shape == (0, frame.input_dim)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cartesian_frame_methods(dtype):
    irreps = co3.Irreps("1x0e+1x1o+2x3e+1x6e")
    features = irreps.randn(7, -1, device=DEVICE, dtype=dtype)
    vectors = torch.cat((torch.eye(3), -torch.eye(3), torch.ones(1, 3))).to(
        DEVICE, torch.float64
    )
    rotation = o2.rotation_matrix_to_y_axis(vectors)
    outputs = []
    for method in ("direct", "rotation", "projection"):
        frame = co2.LocalFrame(irreps, method=method).float().to(DEVICE, dtype)
        output = frame(features, rotation)
        torch.testing.assert_close(
            frame.to_global(output, rotation),
            features,
            atol=3e-5 if dtype == torch.float32 else 3e-11,
            rtol=3e-5 if dtype == torch.float32 else 3e-11,
        )
        outputs.append(output)
    for output in outputs[1:]:
        torch.testing.assert_close(
            output,
            outputs[0],
            atol=3e-5 if dtype == torch.float32 else 3e-11,
            rtol=3e-5 if dtype == torch.float32 else 3e-11,
        )
    for method in ("direct", "rotation", "projection"):
        scalar = co2.LocalFrame("2x0e", method=method).to(DEVICE, dtype)
        value = torch.randn(1, 2, device=DEVICE, dtype=dtype)
        expected = value.expand(vectors.shape[0], -1)
        torch.testing.assert_close(scalar(value, rotation), expected)
        torch.testing.assert_close(scalar.to_global(value, rotation), expected)
        torch.testing.assert_close(scalar.rotate(value, rotation), expected)
    with pytest.raises(ValueError, match="method"):
        co2.LocalFrame(irreps, method="unknown")


@pytest.mark.parametrize("method", ["direct", "rotation", "projection"])
def test_cartesian_frame_export(double_precision, method):
    frame = co2.LocalFrame("2x0e+2x1e+1x3o", method=method).to(DEVICE)
    features = frame.irreps_in.randn(3, -1, device=DEVICE).requires_grad_()
    rotation = o2.rotation_matrix_to_y_axis(torch.randn(3, 3, device=DEVICE))
    exported = torch.export.export(frame, (features, rotation)).module()
    actual, expected = exported(features, rotation), frame(features, rotation)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), features)[0],
        torch.autograd.grad(expected.square().sum(), features)[0],
    )


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("method", ["direct", "rotation", "projection"])
def test_backend_derivatives(backend, method, double_precision, monkeypatch):
    from equivariantx.ops import config

    module = co2.LocalFrame("2x1o+1x2e+1x3o", method=method, backend=backend).to(DEVICE)
    reference = co2.LocalFrame("2x1o+1x2e+1x3o", method=method).to(DEVICE)
    x = (torch.randn(2, module.input_dim, device=DEVICE) * 0.1).requires_grad_()
    rotation = (torch.randn(2, 3, 3, device=DEVICE) * 0.1).requires_grad_()
    values = x, rotation
    for name in ("to_local", "rotate"):
        actual = getattr(module, name)(*values)
        monkeypatch.setattr(config, "TORCH_REFERENCE", True)
        expected = getattr(reference, name)(*values)
        monkeypatch.setattr(config, "TORCH_REFERENCE", False)
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
        losses = [v.sin().sum() for v in (actual, expected)]
        for _ in range(3):
            gradients = [
                torch.autograd.grad(loss, values, create_graph=True) for loss in losses
            ]
            torch.testing.assert_close(*gradients, atol=3e-9, rtol=3e-9)
            tangent = tuple(torch.randn_like(v) * 0.1 for v in values)
            losses = [
                sum((g * t).sum() for g, t in zip(gs, tangent)) for gs in gradients
            ]
    local = module(x, rotation).detach().requires_grad_()
    tangent = torch.randn_like(local), torch.randn_like(rotation)
    actual = torch.func.jvp(module.to_global, (local, rotation), tangent)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = torch.func.jvp(reference.to_global, (local, rotation), tangent)
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
    del module.backend
    torch.testing.assert_close(module(x, rotation), reference(x, rotation))
