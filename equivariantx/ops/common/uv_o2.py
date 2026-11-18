"""Operands and expression schedules for local O(2) convolutions."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ...utils.metadata import static_dimensions
from .. import config


def rotation_layout(wigner):
    """Specialize rotation block widths without specializing the edge count."""
    dims = static_dimensions(wigner, range(1, wigner.ndim))
    return (dims[0], 1, True) if len(dims) == 1 else (*dims, False)


def prepare(
    module, features, edge_features, conv_weights, edge_index, vectors, radial_network
):
    """Prepare the radial partition and select explicit polynomial expansion."""
    source, target = edge_index
    transverse = vectors is not None
    expanded = (
        module.backend == "torch"
        and transverse
        and (radial_network is None or module.radial_mode == "last")
        and (not config.TORCH_REFERENCE)
        and (module.degree <= config.TRANSVERSE_POLYNOMIAL_DEGREE)
        and (
            torch.compiler.is_compiling()
            or get_proxy_mode() is not None
            or torch._C._are_functorch_transforms_active()
        )
    )
    if radial_network is not None:
        from .radial import radial_features, split_radial

        if module.backend == "torch" and config.TORCH_REFERENCE:
            conv_weights = radial_features(radial_network, conv_weights, edge_index)
            radial_network = None
        elif module.radial_mode == "last":
            hidden, projection = split_radial(radial_network)
            if hidden or not projection:
                conv_weights = radial_features(hidden, conv_weights, edge_index)
            radial_network = projection or None
    if edge_features is None:
        edge_features = features.new_empty((source.shape[0], 0))
    return (edge_features, conv_weights, radial_network, expanded)


def program_inputs(
    module,
    features,
    edge_features,
    conv_weights,
    wigner,
    wigner_inv,
    cutoff,
    parameters,
    source,
    target,
    vectors,
    radial_network,
    expanded,
    normalized=False,
):
    """Construct the local convolution program and its tensor operands."""
    transverse = vectors is not None
    if transverse:
        if any(
            (
                f is None and description is not None
                for f, description in zip(module.transverse_frames, module.frames)
            )
        ):
            raise ValueError("Transverse evaluation requires canonical O(2) frames.")
        if module.backend == "cuda":
            from .wigner import alignment_cuda

            geometry = alignment_cuda(repr("normalize"), [vectors])[0]
        else:
            geometry = vectors / vectors.norm(dim=-1, keepdim=True)
        inverse = features.new_empty((source.numel(), 0))
        local_dim = global_dim = 0
        packed = False
    else:
        if wigner is None:
            raise ValueError("Supply Wigner matrices or vectors.")
        local_dim, global_dim, packed = rotation_layout(wigner)
        geometry, inverse = (
            wigner,
            features.new_empty((1, 0)) if packed else wigner_inv,
        )
    metadata = module.convolution_program(
        local_dim,
        global_dim,
        packed,
        transverse,
        native=not expanded,
        normalized=normalized,
    )
    inputs = [features, edge_features, conv_weights, geometry, inverse, cutoff]
    inputs.extend((value.reshape(1, -1) for value in parameters))
    if radial_network is not None:
        from .radial import compose_radial

        metadata, inputs = compose_radial(metadata, inputs, 2, radial_network)
    return (metadata, inputs)
