"""Shared angular coefficients and their derivatives."""

from functools import lru_cache

import torch
from e3nn import o3

from ..o3.rotation import so3_generators
from .metadata import kernel_plan


@lru_cache(maxsize=256)
def harmonic_derivatives(degree, rank):
    """Return angular derivatives at the pole, including rotating vector indices."""
    if rank == 1:
        return so3_generators(degree)[:, :, degree].T.contiguous()
    previous = harmonic_derivatives(degree, rank - 1)
    value = torch.tensordot(so3_generators(degree), previous, dims=([2], [0]))
    value = value.movedim(0, -1).contiguous()
    for axis in range(1, rank):
        term = torch.tensordot(previous, -so3_generators(1), dims=([axis], [1]))
        value += term.movedim(-1, axis)
    value[..., 1] = 0
    return value


@lru_cache(maxsize=256)
def angular_coefficients(metadata, rank):
    """Differentiate fixed angular tensors, including their vector indices.

    A rank-zero tensor is the order-zero CG slice. Derivatives act on the
    harmonic at the pole and its vector indices before coupling to features.
    General order-zero maps without harmonic metadata differentiate every
    angular index instead.
    """
    plan = kernel_plan(metadata)
    coefficients = []
    direct = rank > 0 and bool(plan.harmonic_degrees)
    previous = angular_coefficients(metadata, rank - 1) if rank and not direct else None
    for index, ((_, path), entries) in enumerate(
        zip(plan.path_data, plan.sparse_paths)
    ):
        dim, dim_out = path[4:6]
        if not rank:
            value = torch.zeros(dim, dim_out, dtype=torch.float64, device="cpu")
            for a, b, coefficient in entries:
                value[a, b] = coefficient
        elif direct:
            degree = plan.harmonic_degrees[index]
            cg = o3.wigner_3j(
                (dim - 1) // 2,
                degree,
                (dim_out - 1) // 2,
                dtype=torch.float64,
                device="cpu",
            )
            base = angular_coefficients(metadata, 0)[index]
            scale = (base * cg[:, degree, :]).sum() / cg[:, degree, :].square().sum()
            # Differentiate the harmonic at the pole before coupling it. This
            # preserves its sparse support without subtracting large, nearly
            # cancelling input/output generator contractions.
            harmonic = harmonic_derivatives(degree, rank)
            value = torch.einsum("aqb,q...->ab...", cg, harmonic) * scale
        else:
            value = torch.zeros(
                dim, dim_out, *([3] * rank), dtype=torch.float64, device="cpu"
            )
            if index not in plan.scalar_paths:
                base = previous[index]
                for axis, degree in enumerate(
                    ((dim - 1) // 2, (dim_out - 1) // 2, *([1] * (rank - 1)))
                ):
                    # F(Q n; x) = F(n; D(Q)^T x). The generators are skew.
                    term = torch.tensordot(
                        base, -so3_generators(degree), dims=([axis], [1])
                    )
                    value += term.movedim(-1, axis)
                # Rotation about the aligned direction fixes the direction.
                # Its generator vanishes after summing all angular indices.
                value[..., 1] = 0
        coefficients.append(value)
    return tuple(coefficients)
