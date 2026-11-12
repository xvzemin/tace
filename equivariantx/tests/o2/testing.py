"""Tests for the corresponding EQX source module."""

import torch
from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def _transform(features, irreps, angle, reflected=False, time_reversal=False):
    matrix = irreps.D_from_angle(
        angle,
        reflected=reflected,
        time_reversal=time_reversal,
        dtype=features.dtype,
        device=features.device,
    )
    return torch.matmul(features, matrix.transpose(-1, -2))


def _time_reverse(features: torch.Tensor, irreps) -> torch.Tensor:
    output = features.clone()
    for ir_mul, ir_slice in zip(irreps, irreps.slices()):
        output[..., ir_slice] *= ir_mul.ir.t
    return output


def _tensor_product_case(mode):
    if mode == "u1u":
        return "2x1m", "0e", "2x1m"
    if mode == "uuu":
        return "2x1m", "2x1m", "2x0e"
    return "2x1m", "3x1m", "4x0e"


def _asymmetric_contractions(correlation=3, path_mode="sum"):
    irreps_in = o2.Irreps("2x0e+2x0o+2x1m")
    irreps_out = o2.Irreps("2x0e+2x0o+2x1m+2x2m")
    recursive = o2.AsymmetricContraction(
        irreps_in,
        irreps_out,
        correlation,
        algorithm="recursive",
        path_mode=path_mode,
    ).to(DEVICE, DTYPE)
    dense = o2.AsymmetricContraction(
        irreps_in,
        irreps_out,
        correlation,
        algorithm="dense",
        path_mode=path_mode,
    ).to(DEVICE, DTYPE)
    return recursive, dense


def spherical_layout(features, irreps, inverse=False):
    values = []
    for (mul, ir), section in zip(irreps, irreps.slices()):
        shape = (ir.dim, mul) if inverse else (mul, ir.dim)
        values.append(
            features[..., section]
            .reshape(*features.shape[:-1], *shape)
            .transpose(-1, -2)
            .flatten(-2)
        )
    return torch.cat(values, dim=-1)
