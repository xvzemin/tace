"""Exact O(3) path conversion for a geometry plan shared by the node stack."""

import torch

from equivariantx.conv.o2_o3 import O2O3TensorProductConv
from equivariantx.o2 import O3TensorProduct


def aligned_convolution(tensor_product):
    """Preserve every CG path and trainable weight in an aligned convolution.

    Parameters
    ----------
    tensor_product : e3nn.o3.TensorProduct
        Externally weighted ``uvu`` tensor product. Resolved path coefficients
        are copied without applying a second normalization.

    Returns
    -------
    equivariantx.conv.O2O3TensorProductConv
        Exact O(2) restriction. Call with a shared packed Wigner tensor so its
        geometry and derivatives can be reused across the entire node stack.
    """
    if any(ins.connection_mode != "uvu" for ins in tensor_product.instructions):
        raise ValueError("Aligned OAM convolution requires uvu paths")
    instructions = [
        (
            ins.i_in1,
            ins.i_in2,
            ins.i_out,
            ins.connection_mode,
            ins.has_weight,
            ins.path_weight**2,
        )
        for ins in tensor_product.instructions
    ]
    converted = O3TensorProduct(
        tensor_product.irreps_in1,
        tensor_product.irreps_in2,
        tensor_product.irreps_out,
        instructions,
        internal_weights=False,
        shared_weights=False,
        irrep_normalization="none",
        path_normalization="none",
    )
    return O2O3TensorProductConv(converted, method="wigner", backend="cuda")


def packed_frame(frame, vectors):
    """Construct an orthonormal frame without regularizing nonzero lengths.

    The least-aligned Cartesian axis gives a well-conditioned local gauge.
    No physical result depends on that gauge. Inputs must be nonzero.
    """
    direction = vectors / vectors.norm(dim=-1, keepdim=True)
    axis = torch.nn.functional.one_hot(direction.abs().argmin(-1), 3).to(direction)
    first = torch.linalg.cross(axis, direction)
    first = first / first.norm(dim=-1, keepdim=True)
    third = torch.linalg.cross(first, direction)
    rotation = torch.stack((first, direction, third), dim=-2)
    return torch.cat(
        [block.flatten(1) for block in frame.from_matrix(rotation)], dim=-1
    )
