"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import co2, co3, o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
