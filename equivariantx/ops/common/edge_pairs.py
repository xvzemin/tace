"""Reciprocal-edge indexing for invariant convolution weights."""

from typing import NamedTuple

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from .. import config
from .radial import materialize, radial_layers, radial_reference


def project_rows(radial, projection, network, bias, backend, mode, rows):
    """Evaluate an invariant radial program once per row, replaying it in full mode."""
    from ...utils.program import Program
    from .convolution import radial_input
    from .edge import edge_program
    from .weights import IndexedProjection, RadialWithContext

    optimized = not (
        config.TORCH_REFERENCE and (backend == "torch" or not radial.is_cuda)
    )
    if optimized and hasattr(network, "stream_rows"):
        value = network.stream_rows(radial, backend, mode)
        if value is not None:
            return project_rows(value, projection, None, bias, backend, mode, rows)
    if (
        network is not None
        and mode == "full"
        and optimized
        and (
            torch.is_grad_enabled()
            or projection is None
            or hasattr(
                network.network if isinstance(network, RadialWithContext) else network,
                "to_edge_program",
            )
            or not statically_known_true(
                pair_workspace(
                    radial.shape[0],
                    0,
                    projection.shape[1],
                    radial.element_size(),
                    network,
                    compact=True,
                )
                <= min(config.RADIAL_PROJECT_MAX_BYTES, config.PAIR_EAGER_MAX_BYTES)
            )
        )
    ):
        program, inputs = Program(), []
        root = radial_input(program, inputs, radial, network)
        if projection is not None:
            if projection.shape[0] == program.size(root) + 1:
                root = program.concatenate((root, program.constant(1, 1)))
            root = IndexedProjection(projection, (), None, bias, None).to_edge_program(
                inputs, program, root
            )
        metadata = repr((tuple(program.nodes), ((root, 0, "edge"),)))
        return edge_program(
            backend if radial.is_cuda else "torch",
            metadata,
            0,
            rows,
            rows,
            *inputs,
            inline=False,
        )[0]
    if network is not None:
        radial = radial_reference(network, radial)
    if projection is not None and projection.shape[0]:
        if projection.shape[0] == radial.shape[1] + 1:
            bias = projection[-1] if bias is None else projection[-1] + bias
            projection = projection[:-1]
        return (
            radial @ projection
            if bias is None
            else torch.addmm(bias, radial, projection)
        )
    return radial if bias is None else radial + bias


class EdgePairs(NamedTuple):
    """Representative edge indices and the pair index of each directed edge."""

    representatives: torch.Tensor
    inverse: torch.Tensor

    def expand(self, values):
        """Gather one row per directed edge from a pair table."""
        if values.ndim == 0 or values.shape[0] != self.representatives.numel():
            raise ValueError("Expected one row per undirected pair.")
        return values.index_select(0, self.inverse)


def undirected_edge_pairs(edge_index, edge_shifts=None, *, num_nodes=None):
    """Index bonds independently of their orientation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Source and target indices of shape ``(2, edges)`` in a common node set.
    edge_shifts : torch.Tensor, optional
        Integer lattice translations of shape ``(edges, 3)``. Required for
        periodic graphs. Opposite translations are paired on edge reversal.
    num_nodes : int, optional
        Node count. Enables scalar integer keys for nonperiodic graphs.

    Returns
    -------
    EdgePairs
        Indices reusable with all convolutions on the same graph. An edge
        without a reverse is a singleton. Repeated messages remain distinct.

    Examples
    --------
    >>> index = torch.tensor([[1, 0, 2], [0, 1, 0]])
    >>> pairs = undirected_edge_pairs(index, num_nodes=3)
    >>> pairs.representatives.tolist(), pairs.inverse.tolist()
    ([0, 2], [0, 0, 1])
    """
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
    ):
        raise ValueError("edge_index must be an int64 matrix of shape (2, edges).")
    source, target = edge_index
    low, high = torch.minimum(source, target), torch.maximum(source, target)
    shift = None
    span = 1
    if edge_shifts is not None:
        if (
            edge_shifts.shape != (source.numel(), 3)
            or edge_shifts.device != source.device
        ):
            raise ValueError(
                "edge_shifts must have shape (edges, 3) on the graph device."
            )
        if edge_shifts.is_floating_point():
            torch._assert_async(
                (edge_shifts == edge_shifts.round()).all(),
                "edge_shifts must be integer lattice translations.",
            )
        shift = edge_shifts.to(torch.long)
        if num_nodes is not None and source.numel():
            # Topology is prepared outside the differentiable graph. Use
            # collision-free scalar sorting for ordinary periodic cells.
            span = 2 * int(shift.abs().amax()) + 1
        if span == 1 and num_nodes is not None:
            shift = None
    if shift is not None:
        a, b, c = shift.unbind(-1)
        negative = (a < 0) | ((a == 0) & ((b < 0) | ((b == 0) & (c < 0))))
        flip = (source > target) | ((source == target) & negative)
        shift = torch.where(flip[:, None], -shift, shift)
    if num_nodes is not None and num_nodes >= 0 and num_nodes**2 * span**3 <= 2**63 - 1:
        keys = low * num_nodes + high
        if shift is not None:
            for component in shift.unbind(-1):
                keys = keys * span + (component + span // 2)
        unique, inverse = torch.unique(keys, return_inverse=True)
    else:
        keys = (low, high)
        if shift is not None:
            keys += shift.unbind(-1)
        unique, inverse = torch.unique(
            torch.stack(keys, -1), dim=0, return_inverse=True
        )
    order = torch.arange(source.numel(), device=source.device)
    representatives = order.new_full((unique.shape[0],), source.numel())
    representatives.scatter_reduce_(0, inverse, order, reduce="amin", include_self=True)
    return EdgePairs(representatives, inverse)


def pair_workspace(rows, edges, width, itemsize, network, compact=False, mode="last"):
    """Estimate retained pair features, excluding the separately tiled full-mode workspace."""
    from .weights import IndexedProjection, RadialWithContext

    retained = (rows + (0 if compact else edges)) * width
    if mode == "full":
        return retained * itemsize
    while isinstance(network, (IndexedProjection, RadialWithContext)):
        network = network.network
    hidden = 0
    if network is not None:
        layers = (
            (network,)
            if hasattr(network, "to_edge_program")
            else radial_layers(network)
        )
        for layer in layers:
            width_out = getattr(
                layer,
                "out_features",
                getattr(layer, "h_out", getattr(layer, "out_dim", 0)),
            )
            if isinstance(width_out, int):
                hidden += width_out
    return (retained + rows * hidden) * itemsize


def reciprocal_weights(
    module,
    pairs,
    radial,
    projection,
    network,
    edge_index,
    *,
    compact=False,
    local=False,
    dtype=None,
):
    """Evaluate invariant weights on pair rows before the directed contraction."""
    from .weights import ConvolutionWeights, IndexedProjection, radial_policy

    source, target = edge_index
    rows = pairs.representatives
    if (
        pairs.inverse.shape != source.shape
        or pairs.inverse.dtype != torch.long
        or pairs.inverse.device != source.device
        or rows.dtype != torch.long
        or rows.ndim != 1
        or rows.device != source.device
    ):
        raise ValueError("The pair map must contain one index per directed edge.")
    if isinstance(radial, tuple):
        radial = materialize(radial, (source[rows], target[rows]))
    if radial.ndim != 2 or radial.shape[0] != rows.numel():
        raise ValueError("Radial inputs must contain one row per undirected pair.")
    if dtype is not None and (
        radial.dtype != dtype or projection is not None and projection.dtype != dtype
    ):
        from ..precision import validate_precision

        validate_precision((radial, projection))
        radial = radial.to(dtype)
        if projection is not None:
            projection = projection.to(dtype)
    width = module.weight_numel
    # Preserve bounded streaming when even the directed weight gather would
    # exceed the existing projection budget. Expand the narrow input instead.
    if network is not None or (projection is not None and projection.shape[0]):
        mode = radial_policy(module, network)
        size = pair_workspace(
            rows.numel(),
            source.numel(),
            width,
            radial.element_size(),
            network,
            compact,
            mode,
        )
        bounded = (
            not statically_known_true(size <= config.RADIAL_PROJECT_MAX_BYTES)
            if mode == "full"
            else statically_known_true(size > config.RADIAL_PROJECT_MAX_BYTES)
        )
        if bounded:
            if (
                compact
                and projection is not None
                and projection.shape[0]
                and not isinstance(network, IndexedProjection)
            ):
                from .weights import indexed_weights

                return indexed_weights(
                    radial,
                    projection,
                    (),
                    radial_network=network,
                    radial_mode="full",
                    edge_pairs=pairs,
                )
            if isinstance(network, IndexedProjection):
                network = network._replace(
                    terms=tuple((t, i[pairs.inverse]) for t, i in network.terms),
                    term_envelope=None
                    if network.term_envelope is None
                    else pairs.expand(network.term_envelope),
                )
            return ConvolutionWeights(pairs.expand(radial), projection, network)
    if (
        local
        and torch.is_grad_enabled()
        and network is not None
        and not isinstance(network, IndexedProjection)
        and radial_policy(module, network) == "last"
    ):
        from .radial import split_radial

        hidden, tail = (
            (network, ()) if projection is not None else split_radial(network)
        )
        if projection is not None:
            hidden_width = projection.shape[0]
        elif tail:
            last = tail[0]
            hidden_width = getattr(last, "in_features", getattr(last, "h_in", 0))
        else:
            hidden_width = 0
        # Keep narrow hidden activations, not the adjoint of an E x W output
        # table, when the local frame kernel can fuse the final projection.
        if 0 < hidden_width <= 128 and hidden_width < width and hidden:
            value = pairs.expand(radial_reference(hidden, radial))
            return ConvolutionWeights(value, projection, tail or None)
    radial = project_rows(
        radial,
        projection,
        network,
        None,
        module.backend,
        radial_policy(module, network),
        rows,
    )
    if radial.shape[1] != width:
        raise ValueError("The radial output width must equal weight_numel.")
    direct = radial if compact else pairs.expand(radial)
    return ConvolutionWeights(
        direct, None if projection is None else projection.new_empty((0, width)), None
    )
