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
    symmetric_components,
    symmetric_indices,
    symmetric_path_matrix,
)


class CartesianTensorProductConv(O3TensorProductConv):
    """Gather, couple and sum Cartesian tensors.

    Parameters
    ----------
    tensor_product : eqx.co3.TensorProduct
        Defines the irreps, ``uvu`` instructions and normalization.
    backend : {"cuda", "torch"}, optional
        CUDA fuses delta/epsilon contractions and graph reduction. ``"torch"``
        uses PyTorch Cartesian operations and automatic differentiation.
    normalization : {"component", "integral", "norm"}, optional
        Cartesian-harmonic normalization when vectors are supplied.
    normalize : bool, optional
        Normalize vector inputs before evaluating harmonics.
    symmetric_inputs : bool, optional
        Average input index permutations before coupling. Enables shared
        symmetric storage across paths when vectors are supplied.
    input_basis : {"cartesian", "spherical"}, optional
        Spherical inputs are converted directly to unique Cartesian entries.
    compact_output : bool, optional
        Return partially symmetric raw tensors in ``ir_mul`` order. Requires
        symmetric inputs and ``tensor_product.project=False``. The expansion
        map is available as ``harmonic_output_index`` for node-level Linear.

    Notes
    -----
    Cartesian inputs must be symmetric traceless and use flattened ``mul_ir``
    storage. Every output path is retained. ``tensor_product.project`` selects
    STF or raw outputs; CUDA projects after aggregation. Both backends support
    higher derivatives.
    """

    def __init__(
        self,
        tensor_product,
        *,
        backend="cuda",
        normalization="component",
        normalize=True,
        symmetric_inputs=False,
        input_basis="cartesian",
        compact_output=False,
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
        if input_basis not in ("cartesian", "spherical"):
            raise ValueError("input_basis must be cartesian or spherical.")
        if input_basis == "spherical" and not symmetric_inputs:
            raise ValueError("Spherical input requires symmetric_inputs=True.")
        if compact_output and (not symmetric_inputs or tensor_product.project):
            raise ValueError(
                "Compact output requires symmetric inputs and raw outputs."
            )
        self.input_basis = input_basis
        self.compact_output = compact_output
        self.irreps_in1 = tensor_product.irreps_in1
        self.irreps_in2 = tensor_product.irreps_in2
        self.irreps_out = tensor_product.irreps_out
        self.input_dim, self.edge_dim, self.output_dim = (
            ir.dim for ir in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        )
        self.instructions = tuple(tensor_product.instructions)
        self.weight_numel = tensor_product.weight_numel
        self.tp = tensor_product
        if input_basis == "spherical":
            self.to_cartesian = co3.ChangeOfBasis(self.irreps_in1)
            for _, ir in self.irreps_in1:
                self.register_buffer(
                    f"input_basis_{ir.l}",
                    symmetric_path_matrix(ir.l).to(torch.get_default_dtype()).clone(),
                    persistent=False,
                )
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
        if symmetric_inputs:
            (
                harmonic_paths,
                sources,
                targets,
                self.harmonic_input_dim,
                expansion,
                self.harmonic_output_dim,
            ) = symmetric_components(paths, self.irreps_in1, self.irreps_out)
        else:
            harmonic_paths, expansion, self.harmonic_output_dim = output_components(
                paths, self.irreps_out, normalization
            )
            harmonic_paths, sources, targets, self.harmonic_input_dim = (
                input_components(harmonic_paths, self.input_dim)
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
        representatives = [0] * self.harmonic_output_dim
        for i, j in reversed(list(enumerate(self.harmonic_output_index.tolist()))):
            representatives[j] = i
        self.register_buffer(
            "output_pack_index",
            torch.tensor(representatives, dtype=torch.long),
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
                (
                    "cartesian_symmetric" if symmetric_inputs else "cartesian",
                    normalization,
                ),
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
        fused = self.backend == "cuda" and features.is_cuda
        packed_input = self.input_basis == "spherical" and vectors is not None and fused
        if self.input_basis == "spherical":
            if packed_input:
                if features.size(-1) != self.to_cartesian.input_dim:
                    raise ValueError(
                        "The spherical feature dimension does not match irreps_in1."
                    )
                values = []
                for mul, l, dim, section in self.to_cartesian.paths:
                    value = features[:, section].reshape(features.size(0), mul, dim)
                    value = value @ getattr(self, f"input_basis_{l}").T
                    values.append(value.transpose(-1, -2).flatten(1))
                features = torch.cat(values, -1) if values else features[:, :0]
            else:
                features = self.to_cartesian(features)
        if self.symmetric_inputs and (vectors is None or not fused):
            values = features.new_zeros(
                (features.size(0), self.symmetric_dim)
            ).index_add(-1, self.symmetric_index, features)
            features = (values / self.symmetric_count).index_select(
                -1, self.symmetric_index
            )
        if vectors is not None and not fused:
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
            edge_attrs = edge_attrs + vectors.sum() * 0
            vectors = None
        if vectors is not None and self.pack_inputs and not packed_input:
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
        elif not packed_input and not self.input_identity:
            features = (
                _Permute.apply(features, self.input_index, self.input_inverse)
                if fused
                else features.index_select(-1, self.input_index)
            )
        if vectors is None and not self.attrs_identity:
            edge_attrs = (
                _Permute.apply(edge_attrs, self.attrs_index, self.attrs_inverse)
                if fused
                else edge_attrs.index_select(-1, self.attrs_index)
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
        if vectors is not None and self.compact_output:
            return output
        if vectors is not None:
            output = output.index_select(-1, self.harmonic_output_index)
        elif not self.output_identity:
            output = (
                _Permute.apply(output, self.output_inverse, self.output_index)
                if fused
                else output.index_select(-1, self.output_inverse)
            )
        if self.compact_output:
            return output.index_select(-1, self.output_pack_index)
        return self.projector(output) if fused else output

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        if self.input_basis == "spherical":
            for _, ir in self.irreps_in1:
                name = f"input_basis_{ir.l}"
                self._buffers[name] = (
                    symmetric_path_matrix(ir.l).to(self._buffers[name]).clone()
                )
        return self

    def reference(
        self, features, edge_attrs, radial, projection, edge_index, num_nodes
    ):
        """Evaluate the same instructions using PyTorch Cartesian operations."""
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
