################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union

import torch


def satisfy(l1: int, l2: int, restriction: Union[str, None] = None) -> bool:
    if restriction == None:
        return True
    elif restriction == "<":
        return l1 < l2
    elif restriction == "<=":
        return l1 <= l2
    elif restriction == ">":
        return l1 > l2
    elif restriction == ">=":
        return l1 >= l2
    elif restriction == "==":
        return l1 == l2
    elif restriction == "!=":
        return l1 != l2
    else:
        raise ValueError(f"Unknown restriction: {restriction}")


def so2_expand_index(mmax: int, lmax: int, start: int = 0) -> tuple[int, torch.Tensor]:
    expand_index = []
    offset = 0
    for m in range(start, mmax + 1):
        index = torch.arange((lmax + 1 - m))
        index = index + offset
        expand_index.append(index)
        if m > 0:
            expand_index.append(index)  # +- m
        offset = offset + len(index)
    expand_index = torch.cat(expand_index, dim=0)
    expand_index = expand_index.long()
    num_m_components = offset
    return num_m_components, expand_index


def so3_expand_index(mmax: int, lmax: int) -> tuple[int, torch.Tensor]:
    assert mmax == lmax
    expand_index = torch.zeros([((lmax + 1) ** 2)]).long()
    start_idx = 0
    for l in range(lmax + 1):
        length = 2 * l + 1
        expand_index[start_idx : (start_idx + length)] = l
        start_idx = start_idx + length
    num_l_components = lmax + 1
    return num_l_components, expand_index


def rot(m: int, theta: float) -> torch.Tensor:
    c = math.cos(m * theta)
    s = math.sin(m * theta)
    R = torch.tensor(
        [
            [c, -s],
            [s, c],
        ]
    )
    return R


def complex_mul(x, y):
    a, b = x
    c, d = y
    return torch.tensor(
        [
            a * c - b * d,
            a * d + b * c,
        ]
    )


def complex_mul_conj(x, y):
    a, b = x
    c, d = y
    return torch.tensor(
        [
            a * c + b * d,
            b * c - a * d,
        ]
    )


def num_so2_components(
    lmax: int,
    mmax: int,
):
    total = lmax + 1
    for m in range(1, mmax + 1):
        total += 2 * (lmax + 1 - m)
    return total


def num_uuu_so2_components(
    lmax: int,
    mmax: int,
):
    return (lmax + 1) + (lmax + 1) * (mmax) * 2


def rotate_real_irrep(
    x: torch.Tensor,
    theta: float,
    m: int,
):
    # x: [B, 2, n, C]
    c = math.cos(m * theta)
    s = math.sin(m * theta)

    xr = x[:, 0]
    xi = x[:, 1]

    yr = c * xr - s * xi
    yi = s * xr + c * xi

    out = torch.stack([yr, yi], dim=1)

    return out


def rotate_so2_features(
    x: torch.Tensor,
    theta: float,
    lmax: int,
    mmax: int,
    num_channels: int,
):
    # x: [B, num_components, C]
    B = x.shape[0]
    outputs = []
    offset = 0

    # m = 0
    n0 = lmax + 1
    x0 = x[:, offset : offset + n0]
    outputs.append(x0)
    offset += n0

    # m > 0
    for m in range(1, mmax + 1):
        n = lmax + 1 - m
        xm = x[:, offset : offset + 2 * n]
        xm = xm.view(B, 2, n, num_channels)
        xm = rotate_real_irrep(
            xm,
            theta,
            m,
        )
        xm = xm.reshape(B, 2 * n, num_channels)
        outputs.append(xm)
        offset += 2 * n

    out = torch.cat(outputs, dim=1)

    return out


def rotate_uuu_so2_features(
    x: torch.Tensor,
    theta: float,
    lmax: int,
    mmax: int,
    num_channels: int,
):
    # x: [B, num_components, C]
    B = x.shape[0]
    outputs = []
    offset = 0

    n = lmax + 1
    # m = 0
    x0 = x[:, offset : offset + n]
    outputs.append(x0)
    offset += n

    # m > 0
    for m in range(1, mmax + 1):
        xm = x[:, offset : offset + 2 * n]
        xm = xm.view(B, 2, n, num_channels)
        xm = rotate_real_irrep(
            xm,
            theta,
            m,
        )
        xm = xm.reshape(B, 2 * n, num_channels)
        outputs.append(xm)
        offset += 2 * n

    out = torch.cat(outputs, dim=1)

    return out
