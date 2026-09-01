################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from dataclasses import dataclass
from typing import Sequence

import torch

from .irreps import Irrep, Irreps, IrrepsLike
from .tensor_product import _cg_product, _cg_product_uuu


@dataclass(frozen=True)
class _Path:
    leaves: tuple[int, ...]
    intermediates: tuple[Irrep, ...]
    output_index: int


def _cg_tensor(irrep1: Irrep, irrep2: Irrep, irrep_out: Irrep) -> torch.Tensor:
    input1 = torch.eye(irrep1.dim, dtype=torch.float64)
    input2 = torch.eye(irrep2.dim, dtype=torch.float64)
    product = _cg_product(input1, irrep1, input2, irrep2, irrep_out)
    return product.permute(1, 2, 0).contiguous()


class AsymmetricContraction(torch.nn.Module):
    """Contract independent O(2) inputs into many-body features.

    Parameters
    ----------
    irreps_in : IrrepsLike
        Representation of every independent input. All entries must have one
        common multiplicity, which is interpreted as the channel count.
    irreps_out : IrrepsLike
        Requested output types. Their multiplicities must equal the common
        input multiplicity.
    correlation : int
        Highest correlation order to enumerate. The module consumes one
        independent input tensor for every order up to this value.
    algorithm : {"edge", "node"}
        Evaluation strategy. ``"edge"`` recursively contracts individual
        paths; ``"node"`` evaluates precomputed generalized coupling tensors.
    path_mode : {"sum", "expand"}, optional
        ``"sum"`` accumulates equivalent paths into each requested output
        type with variance normalization. ``"expand"`` preserves every path
        as a separate output multiplicity.

    Notes
    -----
    Inputs and outputs use flattened ``ir_mul`` layout. Weights are always
    supplied externally and are applied directly to the enumerated paths.
    ``weight_numel`` gives the required trailing weight dimension.
    """

    def __init__(
        self,
        irreps_in: IrrepsLike,
        irreps_out: IrrepsLike,
        correlation: int,
        *,
        algorithm: str,
        path_mode: str = "sum",
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.channels = Irreps.common_multiplicity(self.irreps_in)
        self._input_irreps = tuple(ir for ir, _ in self.irreps_in)
        requested_irreps_out = Irreps(irreps_out)
        if any(mul != self.channels for _, mul in requested_irreps_out):
            raise ValueError(
                "AsymmetricContraction input and requested output entries must "
                "use the same multiplicity."
            )
        if not isinstance(correlation, int) or isinstance(correlation, bool):
            raise TypeError("correlation must be an integer.")
        if correlation < 1:
            raise ValueError("correlation must be positive.")
        if algorithm not in ("edge", "node"):
            raise ValueError("algorithm must be 'edge' or 'node'.")
        if path_mode not in ("sum", "expand"):
            raise ValueError("path_mode must be 'sum' or 'expand'.")
        self.correlation = correlation
        self.algorithm = algorithm
        self.path_mode = path_mode

        output_types = []
        for ir, _ in requested_irreps_out:
            if ir not in output_types:
                output_types.append(ir)
        self.irreps_out_types = Irreps([(ir, self.channels) for ir in output_types])
        allowed_outputs = set(output_types)
        filtered_states = tuple(
            tuple(
                (leaves, intermediates)
                for leaves, intermediates in states
                if intermediates[-1] in allowed_outputs
            )
            for states in self._enumerate_states()
        )

        if path_mode == "sum":
            base_irreps_out = Irreps(output_types)
            output_indices = {ir: index for index, ir in enumerate(output_types)}
            paths_by_order = [
                tuple(
                    _Path(leaves, intermediates, output_indices[intermediates[-1]])
                    for leaves, intermediates in states
                )
                for states in filtered_states
            ]
        else:
            output_counts = {ir: 0 for ir in output_types}
            for states in filtered_states:
                for _, intermediates in states:
                    output_counts[intermediates[-1]] += 1
            base_irreps_out = Irreps(
                [(ir, output_counts[ir]) for ir in output_types if output_counts[ir]]
            )
            next_output_index = {}
            offset = 0
            for ir, mul in base_irreps_out:
                next_output_index[ir] = offset
                offset += mul
            paths_by_order = []
            for states in filtered_states:
                paths = []
                for leaves, intermediates in states:
                    ir_out = intermediates[-1]
                    paths.append(
                        _Path(leaves, intermediates, next_output_index[ir_out])
                    )
                    next_output_index[ir_out] += 1
                paths_by_order.append(tuple(paths))

        self._base_irreps_out = base_irreps_out
        self.irreps_out = Irreps(
            [(ir, mul * self.channels) for ir, mul in base_irreps_out]
        )
        self._paths_by_order = tuple(paths_by_order)
        self.order_num_paths = tuple(len(paths) for paths in self._paths_by_order)
        self.num_paths = sum(self.order_num_paths)
        self.weight_numel = self.num_paths * self.channels
        self.weight_shape = (self.weight_numel,)

        if path_mode == "sum":
            output_path_counts = [0] * base_irreps_out.num_irreps
            for paths in self._paths_by_order:
                for path in paths:
                    output_path_counts[path.output_index] += 1
            self._path_scales = tuple(
                tuple(output_path_counts[path.output_index] ** -0.5 for path in paths)
                for paths in self._paths_by_order
            )
        else:
            self._path_scales = tuple(
                tuple(1.0 for _ in paths) for paths in self._paths_by_order
            )

        order_weight_slices = []
        offset = 0
        for num_paths in self.order_num_paths:
            width = num_paths * self.channels
            order_weight_slices.append(slice(offset, offset + width))
            offset += width
        self._order_weight_slices = tuple(order_weight_slices)
        self._input_slices = self.irreps_in.slices()
        self._base_input_slices = Irreps(self._input_irreps).slices()
        self._base_output_slices = tuple(
            slice(start, start + ir.dim)
            for start, ir in self._expanded_offsets(base_irreps_out)
        )
        paths_by_output = [[] for _ in range(base_irreps_out.num_irreps)]
        for order_index, paths in enumerate(self._paths_by_order):
            for path_index, path in enumerate(paths):
                paths_by_output[path.output_index].append(
                    (order_index, path_index, path)
                )
        self._paths_by_output = tuple(tuple(paths) for paths in paths_by_output)
        if algorithm == "node":
            self._setup_generalized_cg()

    @staticmethod
    def _expanded_offsets(irreps: Irreps):
        offset = 0
        for ir in irreps.expanded():
            yield offset, ir
            offset += ir.dim

    def _enumerate_states(self):
        previous = tuple(
            (((index,), (ir,))) for index, ir in enumerate(self._input_irreps)
        )
        states_by_order = [previous]
        for _ in range(2, self.correlation + 1):
            current = []
            for leaves, intermediates in previous:
                for input_index, ir in enumerate(self._input_irreps):
                    for ir_out in intermediates[-1] * ir:
                        current.append(
                            (leaves + (input_index,), intermediates + (ir_out,))
                        )
            previous = tuple(current)
            states_by_order.append(previous)
        return tuple(states_by_order)

    def _compact_generalized_cg(self, path: _Path) -> torch.Tensor:
        first_ir = self._input_irreps[path.leaves[0]]
        coefficient = torch.eye(first_ir.dim, dtype=torch.float64)
        for order_index in range(1, len(path.leaves)):
            pair = _cg_tensor(
                path.intermediates[order_index - 1],
                self._input_irreps[path.leaves[order_index]],
                path.intermediates[order_index],
            )
            coefficient = torch.tensordot(
                coefficient,
                pair,
                dims=([-1], [0]),
            ).contiguous()
        return coefficient

    def _setup_generalized_cg(self) -> None:
        input_dim = sum(ir.dim for ir in self._input_irreps)
        output_dim = sum(ir.dim for ir in self._base_irreps_out.expanded())
        for order_index, paths in enumerate(self._paths_by_order):
            order = order_index + 1
            coefficient = torch.zeros(
                (output_dim,) + (input_dim,) * order + (len(paths),),
                dtype=torch.float64,
            )
            for path_index, (path, scale) in enumerate(
                zip(paths, self._path_scales[order_index])
            ):
                compact = self._compact_generalized_cg(path)
                compact = compact.permute(
                    compact.ndim - 1,
                    *range(compact.ndim - 1),
                )
                coefficient[
                    (
                        self._base_output_slices[path.output_index],
                        *(self._base_input_slices[index] for index in path.leaves),
                        path_index,
                    )
                ] = compact * scale
            self.register_buffer(
                f"generalized_cg_{order}",
                coefficient,
                persistent=False,
            )

    def _validate_inputs(self, inputs, weights):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("inputs must be a list or tuple of tensors.")
        if len(inputs) != self.correlation:
            raise ValueError(
                f"Expected {self.correlation} independent input tensors, "
                f"got {len(inputs)}."
            )
        for features in inputs:
            if not isinstance(features, torch.Tensor):
                raise TypeError("Every contraction input must be a torch.Tensor.")
            if features.ndim < 1 or features.size(-1) != self.irreps_in.dim:
                raise ValueError(
                    "Contraction input trailing dimension must be "
                    f"{self.irreps_in.dim}, got {tuple(features.shape)}."
                )
        if not isinstance(weights, torch.Tensor):
            raise TypeError("External weights must be a torch.Tensor.")
        if weights.is_complex():
            raise TypeError("AsymmetricContraction supports real weights only.")
        if weights.ndim < 1 or weights.size(-1) != self.weight_numel:
            raise ValueError(
                "External weights must have trailing dimension "
                f"{self.weight_numel}, got {tuple(weights.shape)}."
            )
        try:
            leading_shape = torch.broadcast_shapes(
                *(features.shape[:-1] for features in inputs),
                weights.shape[:-1],
            )
        except RuntimeError as error:
            raise ValueError(
                "Contraction input and weight batch dimensions do not broadcast."
            ) from error
        inputs = tuple(
            features.expand(*leading_shape, self.irreps_in.dim) for features in inputs
        )
        weights = weights.expand(*leading_shape, self.weight_numel)
        return inputs, weights, leading_shape

    def _order_weights(self, weights: torch.Tensor, order_index: int):
        return weights[..., self._order_weight_slices[order_index]].reshape(
            *weights.shape[:-1],
            self.order_num_paths[order_index],
            self.channels,
        )

    def _entry(self, features: torch.Tensor, index: int) -> torch.Tensor:
        ir = self._input_irreps[index]
        return features[..., self._input_slices[index]].reshape(
            *features.shape[:-1], ir.dim, self.channels
        )

    def _contract_edge_path(self, inputs, path):
        value = self._entry(inputs[0], path.leaves[0])
        for order_index in range(1, len(path.leaves)):
            input_index = path.leaves[order_index]
            value = _cg_product_uuu(
                value,
                path.intermediates[order_index - 1],
                self._entry(inputs[order_index], input_index),
                self._input_irreps[input_index],
                path.intermediates[order_index],
            )
        return value

    def _flatten_base(self, features: torch.Tensor) -> torch.Tensor:
        outputs = []
        offset = 0
        for ir, mul in self._base_irreps_out:
            values = []
            for _ in range(mul):
                values.append(features[..., offset : offset + ir.dim, :])
                offset += ir.dim
            outputs.append(
                torch.cat(values, dim=-1).reshape(
                    *features.shape[:-2], ir.dim * mul * self.channels
                )
            )
        return (
            torch.cat(outputs, dim=-1)
            if outputs
            else features.new_empty(*features.shape[:-2], 0)
        )

    def _forward_edge(self, inputs, weights, leading_shape):
        outputs = []
        zero = sum(features.sum() * 0 for features in inputs) + weights.sum() * 0
        order_weights = tuple(
            self._order_weights(weights, index) for index in range(self.correlation)
        )
        for output_index, ir_out in enumerate(self._base_irreps_out.expanded()):
            output = (
                inputs[0].new_zeros(*leading_shape, ir_out.dim, self.channels) + zero
            )
            for order_index, path_index, path in self._paths_by_output[output_index]:
                value = self._contract_edge_path(inputs, path)
                output = output + (
                    value
                    * order_weights[order_index][..., path_index, :].unsqueeze(-2)
                    * self._path_scales[order_index][path_index]
                )
            outputs.append(output)
        base = (
            torch.cat(outputs, dim=-2)
            if outputs
            else inputs[0].new_empty(*leading_shape, 0, self.channels)
        )
        return self._flatten_base(base)

    def _to_base_layout(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self._entry(features, index) for index in range(len(self.irreps_in))],
            dim=-2,
        )

    def _forward_node(self, inputs, weights):
        letters = "abdefghijklmnqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if self.correlation > len(letters):
            raise RuntimeError("correlation is too large for the CG einsum.")
        base_inputs = tuple(self._to_base_layout(features) for features in inputs)
        output = None
        for order_index in range(self.correlation):
            order = order_index + 1
            indices = letters[:order]
            equation = (
                ",".join(
                    [f"o{indices}p"] + [f"...{index}c" for index in indices] + ["...pc"]
                )
                + "->...oc"
            )
            coefficient = getattr(self, f"generalized_cg_{order}").to(inputs[0])
            contribution = torch.einsum(
                equation,
                coefficient,
                *base_inputs[:order],
                self._order_weights(weights, order_index),
            )
            output = contribution if output is None else output + contribution
        if output is None:
            raise RuntimeError("AsymmetricContraction produced no orders.")
        return self._flatten_base(output)

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate all correlation orders.

        Parameters
        ----------
        inputs : sequence of torch.Tensor
            Exactly ``correlation`` independent tensors. Every tensor has
            shape ``(..., irreps_in.dim)``; their leading dimensions must
            broadcast.
        weights : torch.Tensor
            Real external path weights with shape ``(..., weight_numel)``.
            Leading dimensions broadcast with all inputs.

        Returns
        -------
        torch.Tensor
            Contracted features with shape ``(..., irreps_out.dim)`` over the
            broadcast leading shape.
        """
        inputs, weights, leading_shape = self._validate_inputs(inputs, weights)
        if self.algorithm == "edge":
            return self._forward_edge(inputs, weights, leading_shape)
        return self._forward_node(inputs, weights)

    def extra_repr(self) -> str:
        return (
            f"irreps_in={self.irreps_in}, irreps_out={self.irreps_out}, "
            f"correlation={self.correlation}, algorithm={self.algorithm!r}, "
            f"path_mode={self.path_mode!r}, num_paths={self.num_paths}"
        )
