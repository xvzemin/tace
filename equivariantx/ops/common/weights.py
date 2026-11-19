"""Indexed and factorized path weights for graph convolutions."""

from typing import NamedTuple

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ...utils.metadata import static_dimensions
from .. import config
from ..precision import validate_precision
from ..registry import resolve_backend
from .radial import materialize, radial_features, radial_program, radial_reference


class IndexedRows(NamedTuple):
    indices: torch.Tensor
    num_rows: int
    scale: float
    coefficients: torch.Tensor | None


def row_coefficients(indices, rows, dtype, scale):
    coefficients = torch.zeros(
        (indices.shape[1], rows), device=indices.device, dtype=dtype
    )
    return coefficients.scatter_add_(
        1, indices.T, coefficients.new_full((), scale).expand_as(indices.T)
    )


def indexed_rows(indices, num_rows, *, dtype, reduce="sum", backend=None):
    """Prepare a row map reusable across convolution layers.

    Parameters
    ----------
    indices : torch.Tensor
        Int64 row indices of shape ``(lookups, edges)``.
    num_rows : int
        Number of rows in each table using this map.
    dtype : torch.dtype
        Floating dtype of the tables.
    reduce : {"sum", "mean"}
        Reduction over lookups for each edge, counting repeated indices.
    backend : {None, "torch", "cuda"}, optional
        Convolution backend used to choose the row-map layout.

    Examples
    --------
    >>> ids = torch.tensor([[0, 1], [1, 1]])
    >>> rows = indexed_rows(ids, 2, dtype=torch.float32, reduce="mean")
    """
    if indices.dtype != torch.long or indices.ndim != 2 or not indices.shape[0]:
        raise ValueError("indices must be an int64 matrix with at least one lookup.")
    if (
        num_rows < 1
        or reduce not in ("sum", "mean")
        or dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("Provide a positive row count, sum or mean, and FP32 or FP64.")
    scale = 1 / indices.shape[0] if reduce == "mean" else 1.0
    coefficients = None
    backend = resolve_backend(backend, indices.device)
    limit = (
        config.INDEXED_PROJECTION_MAX_ROWS
        if backend == "cuda" and indices.is_cuda
        else config.INDEXED_TORCH_PROJECTION_MAX_ROWS
    )
    if num_rows <= limit and statically_known_true(
        indices.shape[1] * num_rows * dtype.itemsize
        <= config.INDEXED_PROJECTION_MAP_BYTES
    ):
        coefficients = row_coefficients(indices, num_rows, dtype, scale)
    return IndexedRows(indices, num_rows, scale, coefficients)


class RadialWithContext(torch.nn.Module):
    """Pass trailing conditioning columns unchanged through a radial network."""

    def __init__(self, network, width, mode=None):
        super().__init__()
        self.network, self.width, self.mode = network, width, mode

    def forward(self, value):
        return torch.cat(
            (
                radial_reference(self.network, value[:, : -self.width]),
                value[:, -self.width :],
            ),
            -1,
        )

    def to_edge_program(self, inputs, program, value):
        width = program.size(value) - self.width
        hidden = radial_program(
            self.network, inputs, program, program.slice(value, 0, width)
        )
        return program.concatenate((hidden, program.slice(value, width, self.width)))


class IndexedProjection(NamedTuple):
    projection: torch.Tensor
    terms: tuple
    network: object
    bias: torch.Tensor | None
    mode: str | None
    scales: tuple = ()
    lookup_only: bool = False

    def narrow_edges(self, start, stop):
        return self._replace(terms=tuple((t, i[start:stop]) for t, i in self.terms))

    def forward(self, value):
        if self.network is not None:
            value = radial_reference(self.network, value)
        if self.projection.shape[0]:
            value = value @ self.projection
        if self.bias is not None:
            value = value + self.bias
        if self.lookup_only:
            table, index = self.terms[0]
            value = table.index_select(0, index)
            if self.scales:
                value = value * self.scales[0]
        for i, (table, index) in enumerate(self.terms):
            if self.lookup_only and i == 0:
                continue
            value = torch.add(
                value,
                table.index_select(0, index),
                alpha=self.scales[i] if self.scales else 1,
            )
        return value

    def project(self, value, backend):
        """Accumulate indexed rows in the destination of the final GEMM."""
        from .layout import gather_sum
        from .torch.layout import gather_sum as torch_gather_sum

        if not self.terms or self.scales:
            return self.forward(value)
        tables, indices = map(list, zip(*self.terms))
        rows = indices[0].numel()
        table = tables[0]
        if (
            backend == "cuda"
            and value.is_cuda
            and torch.is_grad_enabled()
            and table.requires_grad
            and config.RADIAL_PROJECT_MIN_WIDTH > 0
            and self.projection.shape[0] >= config.RADIAL_PROJECT_MIN_WIDTH
            and all(t is table for t in tables)
            and statically_known_true(rows >= 8 * table.shape[0])
            and statically_known_true(
                rows * table.shape[0] * table.element_size()
                <= config.INDEXED_PROJECTION_MAP_BYTES
            )
        ):
            # A small shared table has many repeated rows. Its adjoint is a
            # single GEMM instead of a contended scatter for each lookup.
            coefficients = row_coefficients(
                torch.stack(indices), table.shape[0], table.dtype, 1.0
            )
            offsets = coefficients @ table
            if self.bias is not None:
                offsets = offsets + self.bias
        else:
            if self.bias is not None:
                tables.append(self.bias.expand(rows, -1))
                indices.append(None)
            offsets = (
                gather_sum(tables, indices)
                if backend == "cuda" and value.is_cuda
                else torch_gather_sum(tables, indices)
            )
        if not self.projection.shape[0]:
            return offsets if self.lookup_only else offsets + value
        if value.shape[0] == 1:
            return offsets + value @ self.projection
        return offsets.addmm_(value, self.projection)

    def to_edge_program(self, inputs, program, value):
        if self.network is not None:
            value = radial_program(self.network, inputs, program, value)
        rows, width = static_dimensions(self.projection)
        if program.size(value) != (rows or width):
            raise ValueError("Radial width does not match the indexed projection.")
        if rows:
            matrix = program.input(len(inputs), "shared", rows * width)
            if self.projection.stride(0) == 1 and self.projection.stride(1) != 1:
                inputs.append(self.projection.T.reshape(1, -1))
                matrix = program.transpose(matrix, width, rows)
            else:
                inputs.append(self.projection.reshape(1, -1))
            value = program.matmul(value, matrix, 1, rows, width)
        if self.bias is not None:
            bias = program.input(len(inputs), "shared", width)
            inputs.append(self.bias.reshape(1, width))
            value = program.binary("add", value, bias)
        slots = []
        for i, (table, index) in enumerate(self.terms):
            for previous, slot in slots:
                if previous is table:
                    break
            else:
                slot = len(inputs)
                inputs.append(table)
                slots.append((table, slot))
            term = program.input(slot, ("indexed", len(inputs), slot), width)
            inputs.append(index)
            if self.scales:
                term = program.scale(term, self.scales[i])
            value = (
                term
                if self.lookup_only and i == 0
                else program.binary("add", value, term)
            )
        return value


def radial_policy(module, network):
    if isinstance(network, (IndexedProjection, RadialWithContext)) and network.mode:
        return network.mode
    return module.radial_mode


class ConvolutionWeights(NamedTuple):
    """Radial input, final projection and optional network for a convolution."""

    radial: torch.Tensor | tuple
    projection: torch.Tensor
    network: object

    def as_network(self):
        """Return the two operands for convolutions without a projection argument."""
        if self.projection.shape[0]:
            mode = (
                self.network.mode
                if isinstance(self.network, RadialWithContext)
                else None
            )
            return self.radial, IndexedProjection(
                self.projection, (), self.network, None, mode
            )
        return self.radial, self.network if self.network is not None else ()


def indexed_weights(
    radial,
    projection,
    terms,
    *,
    radial_network=None,
    bias=None,
    radial_mode=None,
    edge_pairs=None,
):
    """Add indexed rows to a convolution's radial path weights.

    Parameters
    ----------
    radial : torch.Tensor or tuple
        Edge features or radial input partitions.
    projection : torch.Tensor
        Matrix of shape ``(hidden, weights)``, or ``(0, weights)`` if the
        radial network already returns path weights.
    terms : sequence of pairs
        Each pair contains a table of shape ``(rows, weights)`` and one
        int64 row index per edge, or a map from :func:`indexed_rows`. Tables
        may be shared or computed by a differentiable function. Rows are
        added in the given path order.
    radial_network : torch.nn.Module, optional
        Network preceding the projection.
    bias : torch.Tensor, optional
        Shared output bias of shape ``(weights,)``.
    radial_mode : {"last", "full"}, optional
        Override the schedule for a deferred network or indexed projection.
        A plain projection uses the convolution's native schedule.
    edge_pairs : equivariantx.conv.EdgePairs, optional
        Evaluate the invariant radial tensor on pair rows. Indices in
        ``terms`` are still directed, and may select different rows on reversal.

    Returns
    -------
    ConvolutionWeights
        Unpack as ``radial, projection, radial_network`` or call
        ``as_network()`` for a convolution that accepts no separate projection.

    Examples
    --------
    >>> from equivariantx.conv import indexed_weights
    >>> x, p, table = torch.randn(3, 4), torch.randn(4, 6), torch.randn(2, 6)
    >>> ids = torch.tensor([1, 0, 1])
    >>> radial, projection, network = indexed_weights(x, p, [(table, ids)])
    >>> torch.testing.assert_close(network.forward(radial), x @ p + table[ids])
    """
    terms = tuple(terms)
    if radial_mode not in (None, "last", "full"):
        raise ValueError("radial_mode must be last or full.")
    if projection.ndim != 2:
        raise ValueError("projection must be a matrix.")
    width = projection.shape[1]
    validate_precision((projection,))
    if bias is not None and bias.shape != (width,):
        raise ValueError("bias must contain one value per path weight.")
    if bias is not None and (
        bias.device != projection.device or bias.dtype != projection.dtype
    ):
        raise ValueError("bias and projection must have the same dtype and device.")
    for table, index in terms:
        if table.ndim != 2 or table.shape[1] != width:
            raise ValueError("Each table must have one column per path weight.")
        if isinstance(index, IndexedRows):
            if table.shape[0] != index.num_rows or (
                index.coefficients is not None
                and table.dtype != index.coefficients.dtype
            ):
                raise ValueError("The row map does not match the table.")
            index = index.indices[0]
        if index.ndim != 1 or index.dtype != torch.long:
            raise ValueError("Each index must be an int64 vector.")
        if index.device != projection.device or table.device != index.device:
            raise ValueError(
                "Tables, indices and projection must be on the same device."
            )
        if table.dtype != projection.dtype:
            raise ValueError("Tables and projection must have the same dtype.")
    lookup_only = False
    if edge_pairs is not None:
        from .edge_pairs import pair_workspace

        if (
            not isinstance(radial, torch.Tensor)
            or radial.ndim != 2
            or radial.shape[0] != edge_pairs.representatives.numel()
        ):
            raise ValueError("Supply a radial matrix with one row per undirected pair.")
        inverse = edge_pairs.inverse
        if (
            inverse.ndim != 1
            or inverse.dtype != torch.long
            or inverse.device != radial.device
        ):
            raise ValueError(
                "The pair map must be an int64 vector on the radial input device."
            )
        if any(
            (i.indices.shape[1] if isinstance(i, IndexedRows) else i.numel())
            != inverse.numel()
            for _, i in terms
        ):
            raise ValueError("Endpoint terms must contain one index per directed edge.")
        if statically_known_true(
            pair_workspace(
                radial.shape[0],
                inverse.numel(),
                width,
                projection.element_size(),
                radial_network,
            )
            <= config.RADIAL_PROJECT_MAX_BYTES
        ):
            value = (
                radial_reference(radial_network, radial)
                if radial_network is not None
                else radial
            )
            if projection.shape[0]:
                value = (
                    value @ projection
                    if bias is None
                    else torch.addmm(bias, value, projection)
                )
            elif bias is not None:
                value = value + bias
            if value.shape[1] != width:
                raise ValueError("Radial output width must match the path weights.")
            if len(terms) == 2 and all(
                isinstance(ids, torch.Tensor) for _, ids in terms
            ):
                (a, ia), (b, ib) = terms
                count = a.shape[0] * b.shape[0]
                if statically_known_true(
                    8 * count <= inverse.numel()
                ) and statically_known_true(
                    count * width * value.element_size()
                    <= config.INDEXED_PROJECTION_MAP_BYTES
                ):
                    # Small ordered vocabularies: form each distinct sum once.
                    terms = (((a[:, None] + b).flatten(0, 1), ia * b.shape[0] + ib),)
            # One multi-table gather sums the invariant and oriented terms.
            terms = ((value, inverse), *terms)
            projection = projection.new_empty((0, width))
            radial = value.new_zeros((1, 1)).expand(1, width)
            radial_network, bias, lookup_only = None, None, True
        else:
            radial = edge_pairs.expand(radial)
    if bias is None and not terms:
        return ConvolutionWeights(radial, projection, radial_network)
    if (
        terms
        and projection.shape[0]
        and all(
            isinstance(rows, IndexedRows) and rows.coefficients is not None
            for _, rows in terms
        )
        and statically_known_true(
            (
                projection.shape[0]
                + sum(rows.num_rows for _, rows in terms)
                + int(bias is not None)
            )
            * (
                width
                + (
                    terms[0][1].indices.shape[1]
                    if radial_network is None and not isinstance(radial, tuple)
                    else 0
                )
            )
            * projection.element_size()
            <= config.RADIAL_PROJECT_MAX_BYTES
        )
    ):
        contexts = tuple(rows.coefficients for _, rows in terms)
        matrices = (projection, *(table for table, _ in terms))
        if bias is not None:
            matrices = (*matrices, bias.unsqueeze(0))
            contexts = (*contexts, bias.new_ones(1, 1).expand(contexts[0].shape[0], 1))
        if radial_network is None and not isinstance(radial, tuple):
            if radial.shape[0] == 1:
                radial = radial.expand(contexts[0].shape[0], -1)
            radial = torch.cat((radial, *contexts), -1)
        else:
            parts = radial if isinstance(radial, tuple) else ((radial, "edge"),)
            radial = (*parts, *((x, "edge") for x in contexts))
            radial_network = RadialWithContext(
                radial_network if radial_network is not None else (),
                sum(x.shape[1] for x in contexts),
                radial_mode,
            )
        return ConvolutionWeights(radial, torch.cat(matrices, 0), radial_network)
    sparse, scales = [], []
    for table, rows in terms:
        if isinstance(rows, IndexedRows):
            scale = rows.scale
            if (
                scale != 1
                and table.numel() * table.element_size()
                <= config.INDEXED_PROJECTION_MAP_BYTES
            ):
                table, scale = table * scale, 1.0
            sparse.extend(
                (table, rows.indices[i]) for i in range(rows.indices.shape[0])
            )
            scales.extend([scale] * rows.indices.shape[0])
        else:
            sparse.append((table, rows))
            scales.append(1.0)
    if sparse and any(index.shape != sparse[0][1].shape for _, index in sparse):
        raise ValueError("All row index vectors must have the same length.")
    return ConvolutionWeights(
        radial,
        projection.new_empty((0, width)),
        IndexedProjection(
            projection,
            tuple(sparse),
            radial_network,
            bias,
            radial_mode,
            tuple(scales) if any(s != 1.0 for s in scales) else (),
            lookup_only,
        ),
    )


def prepare_indexed(module, radial, network, edge_index, projection=None):
    """Resolve the reference, bounded last-projection or streamed schedule."""
    from ...utils.autograd import forward_ad_active

    if projection is not None and projection.shape != (0, network.projection.shape[1]):
        raise ValueError("Indexed weights already include the output projection.")
    edges = edge_index[0].numel()
    if network.terms and network.terms[0][1].numel() != edges:
        raise ValueError("Row indices must have one entry per edge.")
    if (
        config.TORCH_REFERENCE
        and (module.backend == "torch" or not network.projection.is_cuda)
    ) or forward_ad_active():
        return network.forward(materialize(radial, edge_index)), None
    if network.lookup_only and statically_known_true(
        edges * network.projection.shape[1] * network.projection.element_size()
        <= config.RADIAL_PROJECT_MAX_BYTES
    ):
        return network.project(radial, module.backend), None
    if radial_policy(module, network) == "full":
        return radial, network
    if network.network is not None:
        radial = radial_features(network.network, radial, edge_index)
        network = network._replace(network=None)
    elif isinstance(radial, tuple):
        radial = materialize(radial, edge_index)
    size = edges * network.projection.shape[1] * network.projection.element_size()
    if statically_known_true(size <= config.RADIAL_PROJECT_MAX_BYTES):
        return network.project(radial, module.backend), None
    return radial, network._replace(mode="full")
