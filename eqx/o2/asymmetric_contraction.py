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
    """Contract independent O(2) features to increasing correlation.

    ``correlation`` input tensors are required. The contribution of order
    ``nu`` uses the first ``nu`` tensors. Every input irrep group must have
    multiplicity one, although the same irrep may occur in multiple groups.
    Every input has shape ``(..., irreps_in.dim, channels)``.
    ``path_mode="sum"`` accumulates paths
    with the same output irrep, while ``path_mode="expand"`` retains every
    path as a separate output multiplicity for a later linear contraction.

    All channel-wise path weights are external and have trailing dimension
    :attr:`weight_numel`. ``algorithm="edge"`` recursively evaluates one path
    at a time and avoids materializing every generalized-CG path. It uses less
    memory but launches many small contractions. ``algorithm="node"`` stores
    dense generalized O(2) Clebsch--Gordan tensors and evaluates one contraction
    per correlation order. It is faster for node-level work but its coefficient
    storage grows rapidly with the representation dimension and correlation.

    The two algorithms use identical paths, external weights, and orthonormal
    CG maps. Summed paths use per-output normalization; expanded paths retain
    unit path scale for a following normalized linear layer.
    """

    def __init__(
        self,
        irreps_in: IrrepsLike,
        irreps_out: IrrepsLike,
        channels: int,
        correlation: int,
        *,
        algorithm: str,
        path_mode: str = "sum",
    ) -> None:
        super().__init__()

        self.irreps_in = Irreps(irreps_in)
        if any(mul != 1 for mul, ir in self.irreps_in):
            raise ValueError(
                "Every AsymmetricContraction input irrep group must have "
                "multiplicity one. Repeat an irrep as separate groups when "
                "multiple copies are required."
            )
        requested_irreps_out = Irreps(irreps_out)
        if not isinstance(channels, int) or isinstance(channels, bool):
            raise TypeError("channels must be an integer.")
        if channels < 1:
            raise ValueError("channels must be positive.")
        if not isinstance(correlation, int) or isinstance(correlation, bool):
            raise TypeError("correlation must be an integer.")
        if correlation < 1:
            raise ValueError("correlation must be positive.")
        if algorithm not in ("edge", "node"):
            raise ValueError("algorithm must be 'edge' or 'node'.")
        if path_mode not in ("sum", "expand"):
            raise ValueError("path_mode must be 'sum' or 'expand'.")
        self.channels = channels
        self.correlation = correlation
        self.algorithm = algorithm
        self.path_mode = path_mode

        output_types = []
        for mul, ir in requested_irreps_out:
            if ir not in output_types:
                output_types.append(ir)
        self.irreps_out_types = Irreps(output_types)

        states_by_order = self._enumerate_states()
        allowed_outputs = set(output_types)
        filtered_states = tuple(
            tuple(
                (leaves, intermediates)
                for leaves, intermediates in states
                if intermediates[-1] in allowed_outputs
            )
            for states in states_by_order
        )

        if path_mode == "sum":
            self.irreps_out = self.irreps_out_types
            output_indices = {
                ir: index for index, ir in enumerate(output_types)
            }
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
            self.irreps_out = Irreps(
                [
                    (output_counts[ir], ir)
                    for ir in output_types
                    if output_counts[ir] > 0
                ]
            )
            next_output_index = {}
            offset = 0
            for mul, ir in self.irreps_out:
                next_output_index[ir] = offset
                offset += mul
            paths_by_order = []
            for states in filtered_states:
                paths = []
                for leaves, intermediates in states:
                    output_irrep = intermediates[-1]
                    paths.append(
                        _Path(
                            leaves,
                            intermediates,
                            next_output_index[output_irrep],
                        )
                    )
                    next_output_index[output_irrep] += 1
                paths_by_order.append(tuple(paths))
        self._paths_by_order = tuple(paths_by_order)
        self.order_num_paths = tuple(len(paths) for paths in self._paths_by_order)
        self.num_paths = sum(self.order_num_paths)
        self.weight_numel = self.num_paths * channels
        self.weight_shape = (self.weight_numel,)

        if path_mode == "sum":
            output_path_counts = [0] * self.irreps_out.num_irreps
            for paths in self._paths_by_order:
                for path in paths:
                    output_path_counts[path.output_index] += 1
            self._path_scales = tuple(
                tuple(
                    output_path_counts[path.output_index] ** -0.5
                    for path in paths
                )
                for paths in self._paths_by_order
            )
        else:
            self._path_scales = tuple(
                tuple(1.0 for _ in paths) for paths in self._paths_by_order
            )

        order_weight_slices = []
        offset = 0
        for num_paths in self.order_num_paths:
            width = num_paths * channels
            order_weight_slices.append(slice(offset, offset + width))
            offset += width
        self._order_weight_slices = tuple(order_weight_slices)
        self._input_irreps = self.irreps_in.expanded()
        self._input_slices = self.irreps_in.expanded_slices()
        self._output_slices = self.irreps_out.expanded_slices()
        paths_by_output = [[] for _ in range(self.irreps_out.num_irreps)]
        for order_index, paths in enumerate(self._paths_by_order):
            for path_index, path in enumerate(paths):
                paths_by_output[path.output_index].append(
                    (order_index, path_index, path)
                )
        self._paths_by_output = tuple(tuple(paths) for paths in paths_by_output)

        if algorithm == "node":
            self._setup_generalized_cg()

    def _enumerate_states(
        self,
    ) -> tuple[tuple[tuple[tuple[int, ...], tuple[Irrep, ...]], ...], ...]:
        irrep_list = self.irreps_in.expanded()
        previous = tuple(
            (((index,), (ir,))) for index, ir in enumerate(irrep_list)
        )
        states_by_order = [previous]
        for _ in range(2, self.correlation + 1):
            current = []
            for leaves, intermediates in previous:
                for input_index, input_ir in enumerate(irrep_list):
                    for output_ir in intermediates[-1] * input_ir:
                        current.append(
                            (
                                leaves + (input_index,),
                                intermediates + (output_ir,),
                            )
                        )
            previous = tuple(current)
            states_by_order.append(previous)
        return tuple(states_by_order)

    def _compact_generalized_cg(self, path: _Path) -> torch.Tensor:
        irrep_list = self.irreps_in.expanded()
        first_ir = irrep_list[path.leaves[0]]
        coefficient = torch.eye(first_ir.dim, dtype=torch.float64)
        for order_index in range(1, len(path.leaves)):
            previous_ir = path.intermediates[order_index - 1]
            input_ir = irrep_list[path.leaves[order_index]]
            output_ir = path.intermediates[order_index]
            pair = _cg_tensor(previous_ir, input_ir, output_ir)
            coefficient = torch.tensordot(
                coefficient,
                pair,
                dims=([-1], [0]),
            ).contiguous()
        return coefficient

    def _setup_generalized_cg(self) -> None:
        input_dim = self.irreps_in.dim
        output_dim = self.irreps_out.dim
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
                indices = (
                    self._output_slices[path.output_index],
                    *(self._input_slices[index] for index in path.leaves),
                    path_index,
                )
                coefficient[indices] = compact * scale
            self.register_buffer(
                f"generalized_cg_{order}",
                coefficient,
                persistent=False,
            )

    def _validate_inputs(
        self,
        inputs: Sequence[torch.Tensor],
        weights: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Size]:
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("inputs must be a list or tuple of tensors.")
        if len(inputs) != self.correlation:
            raise ValueError(
                f"Expected {self.correlation} independent input tensors, "
                f"got {len(inputs)}."
            )
        expected = (self.irreps_in.dim, self.channels)
        for input in inputs:
            if not isinstance(input, torch.Tensor):
                raise TypeError("Every contraction input must be a torch.Tensor.")
            if input.is_complex():
                raise TypeError("AsymmetricContraction supports real inputs only.")
            if input.ndim < 2 or tuple(input.shape[-2:]) != expected:
                raise ValueError(
                    "Contraction input trailing shape must be "
                    f"{expected}, got {tuple(input.shape)}."
                )
        if not isinstance(weights, torch.Tensor):
            raise TypeError("External weights must be a torch.Tensor.")
        if weights.is_complex():
            raise TypeError("AsymmetricContraction supports real weights only.")
        if weights.ndim < 1 or weights.shape[-1] != self.weight_numel:
            raise ValueError(
                "External weights must have trailing dimension "
                f"{self.weight_numel}, got {tuple(weights.shape)}."
            )
        try:
            leading_shape = torch.broadcast_shapes(
                *(input.shape[:-2] for input in inputs),
                weights.shape[:-1],
            )
        except RuntimeError as error:
            raise ValueError(
                "Contraction input and weight batch dimensions do not broadcast."
            ) from error
        expanded_inputs = tuple(
            input.expand(leading_shape + expected) for input in inputs
        )
        weights = weights.expand(leading_shape + (self.weight_numel,))
        return expanded_inputs, weights, leading_shape

    def _order_weights(
        self,
        weights: torch.Tensor,
        order_index: int,
    ) -> torch.Tensor:
        return weights[..., self._order_weight_slices[order_index]].reshape(
            *weights.shape[:-1],
            self.order_num_paths[order_index],
            self.channels,
        )

    def _contract_edge_path(
        self,
        inputs: tuple[torch.Tensor, ...],
        path: _Path,
    ) -> torch.Tensor:
        value = inputs[0][..., self._input_slices[path.leaves[0]], :]
        for order_index in range(1, len(path.leaves)):
            input_index = path.leaves[order_index]
            value = _cg_product_uuu(
                value,
                path.intermediates[order_index - 1],
                inputs[order_index][..., self._input_slices[input_index], :],
                self._input_irreps[input_index],
                path.intermediates[order_index],
            )
        return value

    def _forward_edge(
        self,
        inputs: tuple[torch.Tensor, ...],
        weights: torch.Tensor,
        leading_shape: torch.Size,
    ) -> torch.Tensor:
        output_blocks = []
        zero_dependency = sum(input.sum() * 0 for input in inputs) + weights.sum() * 0
        order_weights = tuple(
            self._order_weights(weights, order_index)
            for order_index in range(self.correlation)
        )
        for output_index, output_ir in enumerate(self.irreps_out.expanded()):
            output = inputs[0].new_zeros(
                leading_shape + (output_ir.dim, self.channels)
            )
            output = output + zero_dependency
            for order_index, path_index, path in self._paths_by_output[output_index]:
                value = self._contract_edge_path(inputs, path)
                path_weight = order_weights[order_index][
                    ..., path_index, :
                ].unsqueeze(-2)
                output = output + (
                    value * path_weight * self._path_scales[order_index][path_index]
                )
            output_blocks.append(output)
        return torch.cat(output_blocks, dim=-2)

    def _forward_node(
        self,
        inputs: tuple[torch.Tensor, ...],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        letters = "abdefghijklmnqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if self.correlation > len(letters):
            raise RuntimeError(
                "correlation is too large for the generalized CG einsum."
            )
        output = None
        for order_index in range(self.correlation):
            order = order_index + 1
            indices = letters[:order]
            equation = ",".join(
                [f"o{indices}p"] + [f"...{index}c" for index in indices] + ["...pc"]
            )
            equation = f"{equation}->...oc"
            coefficient = getattr(self, f"generalized_cg_{order}").to(inputs[0])
            contribution = torch.einsum(
                equation,
                coefficient,
                *inputs[:order],
                self._order_weights(weights, order_index),
            )
            output = contribution if output is None else output + contribution
        if output is None:
            raise RuntimeError(
                "AsymmetricContraction produced no correlation orders."
            )
        return output

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        inputs, weights, leading_shape = self._validate_inputs(inputs, weights)
        if self.algorithm == "edge":
            return self._forward_edge(inputs, weights, leading_shape)
        return self._forward_node(inputs, weights)

    def extra_repr(self) -> str:
        return (
            f"irreps_in={self.irreps_in}, irreps_out={self.irreps_out}, "
            f"channels={self.channels}, correlation={self.correlation}, "
            f"algorithm={self.algorithm!r}, path_mode={self.path_mode!r}, "
            f"num_paths={self.num_paths}"
        )
