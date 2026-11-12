"""Channel-contiguous execution contracts for all node products in the model."""

import torch

from equivariantx.utils.metadata import channel_product_layout


def ir_mul_order(irreps):
    """Return old column indices in magnetic-component/channel order."""
    result, offset = [], 0
    for channels, irrep in irreps:
        result.extend(
            offset + c * irrep.dim + m
            for m in range(irrep.dim)
            for c in range(channels)
        )
        offset += channels * irrep.dim
    return torch.tensor(result, dtype=torch.int64)


def channel_contiguous_metadata(metadata, orders, channels):
    """Rewrite a strided CG program into a uniform contiguous channel program.

    Parameters
    ----------
    metadata : str
        Original EQX channel-product contract.
    orders : sequence of torch.Tensor
        Old indices in the new order, for both inputs and the output.
    channels : int
        Shared multiplicity of every retained irrep path.
    """
    dimensions, rows = channel_product_layout(metadata)
    if len(orders) != len(dimensions) or any(
        len(order) != dim for order, dim in zip(orders, dimensions)
    ):
        raise ValueError("Layout orders do not match the tensor-product dimensions")
    inverse = [order.argsort().tolist() for order in orders]
    converted = []
    for indices, coefficient, width, strides in rows:
        if width != channels:
            raise ValueError("OAM node products require a uniform channel count")
        bases = []
        for table, index, stride in zip(inverse, indices, strides):
            if index < 0:
                bases.append(-1)
                continue
            base = table[index]
            if base % channels or any(
                table[index + c * stride] != base + c for c in range(width)
            ):
                raise ValueError("The requested layout is not channel contiguous")
            bases.append(base // channels)
        converted.append((tuple(bases), coefficient))
    return repr(
        (channels, tuple(dim // channels for dim in dimensions), tuple(converted))
    )
