"""Cartesian graph convolutions with deferred node-level projection."""

import math
from collections import defaultdict

import torch

from ... import co3
from ..._layout import _Permute
from ...co3.basis import path_normalization
from ...co3.tensor_product import coupling_phase
from ..o3.convolution import O3TensorProductConv
from .polynomials import (
    ANGULAR_TILE_SIZE,
    coupling_coefficients,
    input_components,
    output_components,
    symmetric_indices,
)


class CartesianTensorProductConv(O3TensorProductConv):
    """Gather, couple and sum Cartesian tensors without storing edge messages.

    Parameters
    ----------
    tensor_product : eqx.co3.TensorProduct
        Defines the irreps, ``uvu`` instructions and normalization.
    backend : {"cuda", "torch"}, optional
        CUDA fuses delta/epsilon contractions and graph reduction. CPU inputs
        use ordinary PyTorch operations.
    normalization : {"component", "integral", "norm"}, optional
        Cartesian-harmonic normalization when vectors are supplied.
    normalize : bool, optional
        Normalize vector inputs before evaluating harmonics.
    symmetric_inputs : bool, optional
        Average input index permutations before coupling. Enables shared
        symmetric storage across paths when vectors are supplied.

    Notes
    -----
    Inputs and outputs use flattened ``mul_ir`` layout. Inputs must be
    symmetric traceless tensors. Every output path is retained.
    ``tensor_product.project`` selects STF or raw outputs. CUDA applies
    projection after aggregation.
    At low degree, equivalent input entries are summed on nodes, and
    identical harmonic polynomials and output entries are shared per path.
    The full output layout is restored on nodes.
    Radial projections use bounded temporary workspaces, recomputed during
    backward. Vector inputs evaluate harmonics and their derivatives inside
    the CUDA contraction. The same contraction supports higher derivatives.
    """

    def __init__(
        self,
        tensor_product,
        *,
        backend="cuda",
        normalization="component",
        normalize=True,
        symmetric_inputs=False,
    ):
        torch.nn.Module.__init__(self)
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be torch or cuda.")
        if normalization not in ("component", "integral", "norm"):
            raise ValueError("normalization must be integral, component or norm.")
        if any(ins.connection_mode != "uvu" for ins in tensor_product.instructions):
            raise NotImplementedError(
                "CartesianTensorProductConv supports only 'uvu' instructions."
            )
        self.backend, self.normalization, self.normalize = (
            backend,
            normalization,
            normalize,
        )
        self.symmetric_inputs = symmetric_inputs
        self.irreps_in1 = tensor_product.irreps_in1
        self.irreps_in2 = tensor_product.irreps_in2
        self.irreps_out = tensor_product.irreps_out
        self.input_dim, self.edge_dim, self.output_dim = (
            ir.dim for ir in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        )
        self.instructions = tuple(tensor_product.instructions)
        self.weight_numel = tensor_product.weight_numel
        self.tp = tensor_product
        self.projector = (
            co3.Projector(self.irreps_out)
            if tensor_product.project
            else torch.nn.Identity()
        )
        self.harmonics = torch.nn.ModuleDict(
            {
                str(ir.l): co3.CartesianHarmonics(
                    ir.l, normalize=False, normalization=normalization
                )
                for _, ir in self.irreps_in2
            }
        )
        slices = [
            ir.slices() for ir in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        ]
        for name, irreps in (
            ("input", self.irreps_in1),
            ("attrs", self.irreps_in2),
            ("output", self.irreps_out),
        ):
            indices, offset = [], 0
            for mul, ir in irreps:
                indices.extend(
                    (
                        torch.arange(mul * ir.dim).reshape(mul, ir.dim).T.flatten()
                        + offset
                    ).tolist()
                )
                offset += mul * ir.dim
            index = torch.tensor(indices, dtype=torch.long)
            self.register_buffer(f"{name}_index", index, persistent=False)
            self.register_buffer(f"{name}_inverse", index.argsort(), persistent=False)
            setattr(self, f"{name}_identity", indices == list(range(offset)))
        if symmetric_inputs:
            indices, counts, offset = [], [], 0
            for mul, ir in self.irreps_in1:
                entries = symmetric_indices(ir.dim)
                unique = {group: i for i, group in enumerate(dict.fromkeys(entries))}
                indices.extend(
                    offset + u * len(unique) + unique[group]
                    for u in range(mul)
                    for group in entries
                )
                counts.extend(len(group) for _ in range(mul) for group in unique)
                offset += mul * len(unique)
            self.symmetric_dim = offset
            self.register_buffer(
                "symmetric_index",
                torch.tensor(indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "symmetric_count",
                torch.tensor(counts, dtype=torch.long),
                persistent=False,
            )
        paths, offset = [], 0
        for ins in self.instructions:
            mul1, ir1 = self.irreps_in1[ins.i_in1]
            mul2, ir2 = self.irreps_in2[ins.i_in2]
            _, ir_out = self.irreps_out[ins.i_out]
            weight = offset if ins.has_weight else -1
            if ins.has_weight:
                offset += math.prod(ins.path_shape)
            if not mul1 or not mul2 or not ins.path_weight:
                continue
            degrees = ir1.l, ir2.l, ir_out.l
            k, odd = divmod(ir1.l + ir2.l - ir_out.l, 2)
            factor = (
                ins.path_weight
                * coupling_phase(*degrees)
                / path_normalization(*degrees)
                / math.sqrt(3**k * (2 if odd else 1))
            )
            paths.append(
                (
                    slices[0][ins.i_in1].start,
                    slices[1][ins.i_in2].start,
                    slices[2][ins.i_out].start,
                    mul1,
                    mul2,
                    ir1.dim,
                    ir2.dim,
                    ir_out.dim,
                    weight,
                    factor,
                    coupling_coefficients(*degrees),
                )
            )
        self.paths = tuple(paths)
        unweighted = any(p[8] < 0 for p in paths)
        self.kernel_metadata = repr((self.paths, self.weight_numel, unweighted))
        offsets, offset = {}, 0
        for (mul, _), section in zip(self.irreps_in2, slices[1]):
            offsets[section.start] = offset
            offset += mul
        self.amplitude_dim = offset
        harmonic_paths, expansion, self.harmonic_output_dim = output_components(
            paths, self.irreps_out, normalization, symmetric_inputs
        )
        harmonic_paths, sources, targets, self.harmonic_input_dim = input_components(
            harmonic_paths, self.input_dim, symmetric_inputs
        )
        self.pack_inputs = sources != tuple(range(self.input_dim)) or targets != sources
        sources = self.input_index[torch.tensor(sources, dtype=torch.long)]
        targets = torch.tensor(targets, dtype=torch.long)
        for name, columns, rows, width in (
            ("harmonic_input", sources, targets, self.harmonic_input_dim),
            ("harmonic_input_transpose", targets, sources, self.input_dim),
        ):
            self.register_buffer(
                f"{name}_index", columns[rows.argsort(stable=True)], persistent=False
            )
            ptr = torch.cat(
                (rows.new_zeros(1), rows.bincount(minlength=width).cumsum(0))
            )
            self.register_buffer(f"{name}_ptr", ptr, persistent=False)
        self.register_buffer(
            "harmonic_input_count",
            (
                self.harmonic_input_ptr.diff()
                if symmetric_inputs
                else targets.new_empty(0)
            ),
            persistent=False,
        )
        self.register_buffer(
            "harmonic_output_index",
            torch.tensor(expansion, dtype=torch.long)[self.output_inverse],
            persistent=False,
        )
        tiles = []
        for path in harmonic_paths:
            coefficients = defaultdict(list)
            for a, b, c, value in path[-1]:
                tile = tuple(index // ANGULAR_TILE_SIZE for index in (a, b, c))
                coefficients[tile].append((a, b, c, value))
            tiles.extend(
                (path[0], offsets[path[1]], *path[2:-1], tuple(entries))
                for entries in coefficients.values()
            )
        self.harmonic_metadata = repr(
            (
                tuple(tiles),
                self.weight_numel,
                unweighted,
                ("cartesian", normalization),
            )
        )

    def forward(
        self,
        features,
        edge_attrs,
        radial,
        projection,
        edge_index,
        num_nodes=None,
        *,
        vectors=None,
        amplitudes=None,
    ):
        """Convolve node tensors with edge attributes or Cartesian harmonics.

        Parameters follow :meth:`eqx.conv.O3TensorProductConv.forward`, with all tensor
        features in flattened ``mul_ir`` layout and Cartesian dimensions.
        """
        if self.symmetric_inputs and (
            vectors is None or self.backend == "torch" or not features.is_cuda
        ):
            values = features.new_zeros(
                (features.size(0), self.symmetric_dim)
            ).index_add(-1, self.symmetric_index, features)
            features = (values / self.symmetric_count).index_select(
                -1, self.symmetric_index
            )
        if vectors is not None and (self.backend == "torch" or not features.is_cuda):
            vectors = (
                torch.nn.functional.normalize(vectors, dim=-1)
                if self.normalize
                else vectors
            )
            attrs, offset = [], 0
            for mul, ir in self.irreps_in2:
                harmonic = self.harmonics[str(ir.l)](vectors)
                amplitude = (
                    features.new_ones((1, mul))
                    if amplitudes is None
                    else amplitudes.expand(amplitudes.size(0), self.amplitude_dim)[
                        :, offset : offset + mul
                    ]
                )
                attrs.append(
                    (amplitude.unsqueeze(-1) * harmonic.unsqueeze(-2)).flatten(1)
                )
                offset += mul
            edge_attrs = torch.cat(attrs, dim=-1) if attrs else vectors[:, :0]
            vectors = None
        if vectors is not None and self.pack_inputs:
            from ...kernels.layout import indexed_sum

            if features.size(-1) != self.input_dim:
                raise ValueError(
                    "Feature dimensions do not match the tensor-product irreps."
                )
            features = indexed_sum(
                features,
                self.harmonic_input_index,
                self.harmonic_input_ptr,
                self.harmonic_input_transpose_index,
                self.harmonic_input_transpose_ptr,
                self.harmonic_input_count,
            )
        elif not self.input_identity:
            features = _Permute.apply(features, self.input_index, self.input_inverse)
        if vectors is None and not self.attrs_identity:
            edge_attrs = _Permute.apply(
                edge_attrs, self.attrs_index, self.attrs_inverse
            )
        output = super().forward(
            features,
            edge_attrs,
            radial,
            projection,
            edge_index,
            num_nodes,
            vectors=vectors,
            amplitudes=amplitudes,
        )
        if vectors is not None:
            output = output.index_select(-1, self.harmonic_output_index)
        elif not self.output_identity:
            output = _Permute.apply(output, self.output_inverse, self.output_index)
        return (
            self.projector(output)
            if self.backend == "cuda" and features.is_cuda
            else output
        )

    def reference(
        self, features, edge_attrs, radial, projection, edge_index, num_nodes
    ):
        """Evaluate the same instructions using native Cartesian operations."""
        features = (
            features
            if self.input_identity
            else features.index_select(-1, self.input_inverse)
        )
        edge_attrs = (
            edge_attrs
            if self.attrs_identity
            else edge_attrs.index_select(-1, self.attrs_inverse)
        )
        weights = radial @ projection if projection.numel() else radial
        message = self.tp(features[edge_index[0]], edge_attrs, weights)
        output = message.new_zeros((num_nodes, self.output_dim)).index_add(
            0, edge_index[1], message
        )
        output = output + sum(
            x.sum() * 0 for x in (features, edge_attrs, radial, projection)
        )
        return (
            output
            if self.output_identity
            else output.index_select(-1, self.output_index)
        )
