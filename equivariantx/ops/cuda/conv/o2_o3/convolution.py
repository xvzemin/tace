"""Aligned and frame-free contraction schedules."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .....utils.autograd import forward_ad_active
from .... import config
from ....common.o2_o3 import contraction


def evaluate(
    module,
    features,
    radial,
    projection,
    wigner,
    amplitudes,
    edge_index,
    num_nodes,
    vectors,
    radial_network,
):
    """Evaluate validated aligned-convolution operands."""
    if module.channel_mixing:
        from equivariantx.ops.common.convolution import convolve

        return convolve(
            module.kernel_metadata,
            features,
            wigner,
            radial,
            projection,
            edge_index,
            num_nodes,
            module.output_dim,
            network=radial_network,
            amplitudes=amplitudes,
            backend="cuda",
        )
    output = features.new_empty(1).expand(num_nodes, module.output_dim)
    program = (((0, 1, 2, 3, 3, 4, 5), False, ((6, 0),)),)
    operands = [features, radial, projection, wigner, amplitudes, output]
    if radial_network is not None:
        from equivariantx.ops.common.radial import convolve

        return convolve(
            "o2",
            module.kernel_metadata,
            [features, radial, projection, wigner, wigner, amplitudes, output],
            edge_index,
            radial_network,
        )
    return contraction(
        module.kernel_metadata, repr(program), edge_index[0], edge_index[1], operands
    )[0]


def transverse(
    module,
    features,
    radial,
    projection,
    amplitudes,
    edge_index,
    num_nodes,
    vectors,
    method,
    radial_network,
):
    """Evaluate validated aligned-convolution operands."""
    metadata = (
        module.transverse_metadata
        if method == "baseline"
        else getattr(module, f"{method}_metadata")
    )
    from equivariantx.ops.common.o3 import contraction as spherical_contraction
    from equivariantx.ops.common.wigner import alignment_cuda

    if torch.compiler.is_compiling():
        torch._dynamo.mark_static(vectors, -1)
    expanded = (
        module.channel_mixing
        and module.radial_mode == "last"
        and (method in ("baseline", "cg"))
        and (
            max((max(path[5:8]) // 2 for path in module.transverse_paths), default=0)
            <= config.TRANSVERSE_POLYNOMIAL_DEGREE
        )
        and (
            torch.compiler.is_compiling()
            or get_proxy_mode() is not None
            or forward_ad_active()
        )
    )
    direction = (
        vectors / vectors.square().sum(dim=-1, keepdim=True).sqrt()
        if expanded
        else alignment_cuda(repr("normalize"), [vectors])[0]
    )
    if module.channel_mixing:
        from equivariantx.ops.common.convolution import convolve

        return convolve(
            metadata,
            features,
            direction,
            radial,
            projection,
            edge_index,
            num_nodes,
            module.output_dim,
            network=radial_network,
            amplitudes=amplitudes,
            transverse=module.transverse_layout,
            backend="torch" if expanded else "cuda",
        )
    operands = [
        features,
        radial,
        projection,
        amplitudes,
        features.new_empty(1).expand(num_nodes, module.output_dim),
        direction,
    ]
    if radial_network is not None:
        from equivariantx.ops.common.radial import convolve

        return convolve("o3", metadata, operands, edge_index, radial_network)
    return spherical_contraction(
        metadata,
        repr(((tuple(range(6)), False, ((4, 0),)),)),
        edge_index[0],
        edge_index[1],
        operands,
    )[0]
