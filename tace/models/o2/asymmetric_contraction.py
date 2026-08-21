################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from dataclasses import dataclass
from typing import Sequence

import torch

from .irreps import Irrep, IrrepsLike, check_o2_irreps, tensor_product_irreps
from .tensor_product import _cg_product


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


class O2AsymmetricContraction(torch.nn.Module):
    """Contract independent complete-O(2) features to increasing correlation.

    ``correlation`` input tensors are required. The contribution of order
    ``nu`` uses the first ``nu`` tensors, so the output is the normalized sum
    of correlations from one through ``correlation``. Every input has shape
    ``(..., irreps_in.dim, channels)`` and the output has shape
    ``(..., irreps_out.dim, channels)``.

    All channel-wise path weights are external and have trailing dimension
    :attr:`weight_numel`. ``algorithm="edge"`` recursively evaluates one path
    at a time and avoids materializing every generalized-CG path. It uses less
    memory but launches many small contractions. ``algorithm="node"`` stores
    dense generalized O(2) Clebsch--Gordan tensors and evaluates one contraction
    per correlation order. It is faster for node-level work but its coefficient
    storage grows rapidly with the representation dimension and correlation.

    The two algorithms use identical paths, external weights, orthonormal CG
    maps, and per-output path normalization.
    """

    def __init__(
        self,
        irreps_in: IrrepsLike,
        irreps_out: IrrepsLike,
        channels: int,
        correlation: int,
        *,
        algorithm: str,
    ) -> None:
        super().__init__()

        self.irreps_in = check_o2_irreps(irreps_in)
        self.irreps_out = check_o2_irreps(irreps_out)
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
        self.channels = channels
        self.correlation = correlation
        self.algorithm = algorithm

        states_by_order = self._enumerate_states()
        outputs = self.irreps_out.expanded()
        paths_by_order = []
        for states in states_by_order:
            paths = []
            for output_index, output_irrep in enumerate(outputs):
                paths.extend(
                    _Path(leaves, intermediates, output_index)
                    for leaves, intermediates in states
                    if intermediates[-1] == output_irrep
                )
            paths_by_order.append(tuple(paths))
        self._paths_by_order = tuple(paths_by_order)
        self.order_num_paths = tuple(len(paths) for paths in self._paths_by_order)
        self.num_paths = sum(self.order_num_paths)
        self.weight_numel = self.num_paths * channels
        self.weight_shape = (self.weight_numel,)

        output_path_counts = [0] * self.irreps_out.num_irreps
        for paths in self._paths_by_order:
            for path in paths:
                output_path_counts[path.output_index] += 1
        self._path_scales = tuple(
            tuple(output_path_counts[path.output_index] ** -0.5 for path in paths)
            for paths in self._paths_by_order
        )

        order_weight_slices = []
        offset = 0
        for num_paths in self.order_num_paths:
            width = num_paths * channels
            order_weight_slices.append(slice(offset, offset + width))
            offset += width
        self._order_weight_slices = tuple(order_weight_slices)
        self._input_slices = self.irreps_in.expanded_slices()
        self._output_slices = self.irreps_out.expanded_slices()

        if algorithm == "node":
            self._setup_generalized_cg()

    def _enumerate_states(
        self,
    ) -> tuple[tuple[tuple[tuple[int, ...], tuple[Irrep, ...]], ...], ...]:
        inputs = self.irreps_in.expanded()
        previous = tuple((((index,), (irrep,))) for index, irrep in enumerate(inputs))
        states_by_order = [previous]
        for _ in range(2, self.correlation + 1):
            current = []
            for leaves, intermediates in previous:
                for input_index, input_irrep in enumerate(inputs):
                    for output_irrep in tensor_product_irreps(
                        intermediates[-1],
                        input_irrep,
                    ):
                        current.append(
                            (
                                leaves + (input_index,),
                                intermediates + (output_irrep,),
                            )
                        )
            previous = tuple(current)
            states_by_order.append(previous)
        return tuple(states_by_order)

    def _compact_generalized_cg(self, path: _Path) -> torch.Tensor:
        inputs = self.irreps_in.expanded()
        first_irrep = inputs[path.leaves[0]]
        coefficient = torch.eye(first_irrep.dim, dtype=torch.float64)
        for order_index in range(1, len(path.leaves)):
            previous_irrep = path.intermediates[order_index - 1]
            input_irrep = inputs[path.leaves[order_index]]
            output_irrep = path.intermediates[order_index]
            pair = _cg_tensor(previous_irrep, input_irrep, output_irrep)
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
                raise TypeError("O2AsymmetricContraction supports real inputs only.")
            if input.ndim < 2 or tuple(input.shape[-2:]) != expected:
                raise ValueError(
                    "Contraction input trailing shape must be "
                    f"{expected}, got {tuple(input.shape)}."
                )
        if not isinstance(weights, torch.Tensor):
            raise TypeError("External weights must be a torch.Tensor.")
        if weights.is_complex():
            raise TypeError("O2AsymmetricContraction supports real weights only.")
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
        input_irreps = self.irreps_in.expanded()
        value = inputs[0][..., self._input_slices[path.leaves[0]], :]
        for order_index in range(1, len(path.leaves)):
            input_index = path.leaves[order_index]
            product = _cg_product(
                value,
                path.intermediates[order_index - 1],
                inputs[order_index][..., self._input_slices[input_index], :],
                input_irreps[input_index],
                path.intermediates[order_index],
            )
            value = torch.diagonal(product, dim1=-2, dim2=-1)
        return value

    def _forward_edge(
        self,
        inputs: tuple[torch.Tensor, ...],
        weights: torch.Tensor,
        leading_shape: torch.Size,
    ) -> torch.Tensor:
        output_blocks = []
        zero_dependency = sum(input.sum() * 0 for input in inputs) + weights.sum() * 0
        for output_index, output_irrep in enumerate(self.irreps_out.expanded()):
            output = inputs[0].new_zeros(
                leading_shape + (output_irrep.dim, self.channels)
            )
            output = output + zero_dependency
            for order_index, paths in enumerate(self._paths_by_order):
                order_weights = self._order_weights(weights, order_index)
                for path_index, path in enumerate(paths):
                    if path.output_index != output_index:
                        continue
                    value = self._contract_edge_path(inputs, path)
                    path_weight = order_weights[..., path_index, :].unsqueeze(-2)
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
                "O2AsymmetricContraction produced no correlation orders."
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
            f"algorithm={self.algorithm!r}, num_paths={self.num_paths}"
        )
