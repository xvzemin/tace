"""Symmetric powers of planar Cartesian tensors."""

from collections import defaultdict
from functools import lru_cache

import torch

from ..mbe.ace.generalized import channel_permutation
from ..o2.contraction import SymmetricContraction as CircularContraction
from ..o2.contraction import contraction_metadata
from ..ops import resolve_backend
from ..ops.precision import validate_module_precision
from ..ops.torch.o2.contraction import Contraction
from .basis import projector_matrix
from .irreps import Irrep, Irreps
from .tensor_product import coupling_entries


@lru_cache(maxsize=128)
def projected_entries(a, b, c):
    """Return Cartesian coupling entries including the sum-order STF projection."""
    entries = coupling_entries(a, b, c)
    if not a.m or not b.m or c.m != a.m + b.m:
        return entries
    matrix = projector_matrix(c.m).tolist()
    rows = defaultdict(float)
    for i, j, k, weight in entries:
        for o, coefficient in enumerate(matrix[k]):
            if coefficient:
                rows[i, j, o] += weight * coefficient
    return tuple((*key, w) for key, w in rows.items() if w)


class SymmetricContraction(torch.nn.Module):
    """Contract repeated planar STF features with external path weights.

    Parameters
    ----------
    irreps_in, irreps_out : co2.Irreps or str
        Cartesian representations with a common multiplicity in ``mul_ir`` layout.
    correlation : int
        Maximum polynomial degree.
    path_mode : {"sum", "expand"}, optional
        Sum or retain the independent paths within each output type.
    backend : {"torch", "cuda"}, optional
        Execution backend.

    Examples
    --------
    >>> from equivariantx import co2
    >>> import torch
    >>> layer = co2.SymmetricContraction("4x1m", "4x0e", 2)
    >>> layer(torch.randn(3, 8), torch.ones(layer.weight_numel)).shape
    torch.Size([3, 4])
    """

    def __init__(
        self, irreps_in, irreps_out, correlation, *, path_mode="sum", backend=None
    ):
        super().__init__()
        self.backend = resolve_backend(backend)
        self.irreps_in = Irreps(irreps_in)
        labels = CircularContraction(
            self.irreps_in.circular(),
            Irreps(irreps_out).circular(),
            correlation,
            path_mode=path_mode,
            backend="torch",
        )
        self.irreps_out = Irreps([(n, Irrep(ir)) for ir, n in labels.irreps_out])
        self.correlation = correlation
        self.weight_numel = labels.weight_numel
        self.num_paths = labels.num_paths
        self.order_num_paths = labels.order_num_paths
        self.num_channels = labels.num_channels
        self.weight_splits = tuple(n * self.num_channels for n in self.order_num_paths)
        inputs = tuple(ir for _, ir in self.irreps_in)
        outputs = tuple(Irrep(ir) for ir in labels._base_irreps_out.expanded())
        paths = tuple(
            tuple(
                p._replace(intermediates=tuple(Irrep(ir) for ir in p.intermediates))
                for p in level
            )
            for level in labels._paths_by_order
        )
        base_in = Irreps([(1, ir) for ir in inputs])
        base_out = Irreps([(1, ir) for ir in outputs])
        self.input_width, self.output_width = base_in.dim, base_out.dim
        self.input_dim, self.output_dim = self.irreps_in.dim, self.irreps_out.dim
        self.contraction = Contraction(
            inputs,
            paths,
            labels._path_scales,
            base_in.slices(),
            base_out.slices(),
            base_out.dim,
            shared=True,
            coupling=projected_entries,
        )
        self.kernel_metadata = None
        if (
            self.backend == "cuda"
            and correlation <= 4
            and self.input_width <= 12
            and max((ir.m for ir in outputs), default=0) <= 4
            and labels.num_paths <= 64
        ):

            def layout(irreps):
                offset, coordinates = 0, []
                for ir in irreps:
                    coordinates.extend((offset + k, ir.dim) for k in range(ir.dim))
                    offset += self.num_channels * ir.dim
                return tuple(coordinates)

            self.kernel_metadata = contraction_metadata(
                inputs,
                paths,
                labels._path_scales,
                base_in.slices(),
                base_out.slices(),
                base_in.dim,
                base_out.dim,
                self.num_channels,
                shared=True,
                coupling=projected_entries,
                component_layout=(layout(inputs), layout(outputs)),
            )
        self.register_buffer(
            "input_order", channel_permutation(self.irreps_in), persistent=False
        )
        expanded = Irreps([(self.num_channels, ir) for ir in outputs])
        self.register_buffer(
            "output_order", channel_permutation(expanded).argsort(), persistent=False
        )

    def forward(self, features, weight):
        """Evaluate the contraction with broadcastable leading dimensions."""
        validate_module_precision(self, (features, weight))
        if (
            features.shape[-1] != self.input_dim
            or weight.shape[-1] != self.weight_numel
        ):
            raise ValueError(
                "Features and weights must match the declared trailing dimensions."
            )
        leading = torch.broadcast_shapes(features.shape[:-1], weight.shape[:-1])
        if self.kernel_metadata is not None and features.is_cuda:
            from ..ops.common.channel_product import polynomial_contraction

            return polynomial_contraction(
                self.kernel_metadata,
                (features,) * self.correlation,
                weight,
                leading,
                self.output_dim,
                shared=True,
            )
        x = features.index_select(-1, self.input_order).expand(*leading, self.input_dim)
        x = x.reshape(*leading, self.input_width, self.num_channels)
        weights = weight.split(self.weight_splits, -1)
        weights = tuple(
            w.reshape(*w.shape[:-1], n, self.num_channels)
            for w, n in zip(weights, self.order_num_paths)
        )
        cuda = self.backend == "cuda" and x.is_cuda
        values = self.contraction(
            (x,) * self.correlation,
            weights,
            "cuda" if cuda else "torch",
            cuda,
            all_weights=weight.unflatten(-1, (self.num_paths, self.num_channels))
            if self.contraction.joint is not None
            else None,
        )
        return values.flatten(-2).index_select(-1, self.output_order)
