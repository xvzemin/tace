################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from collections import Counter
from typing import Optional, Sequence, Tuple

import torch

from .irreps import Irrep, Irreps, IrrepsLike


class Linear(torch.nn.Module):
    """An O(3)-equivariant linear map between Cartesian irreps.

    Inputs and outputs use ``(..., irreps.dim, channels)``. Every valid path
    uses a dense input-output channel matrix (UV mode). Only equal ``(l, p)``
    types mix, and only ``0e`` outputs receive a bias.
    """

    def __init__(
        self,
        irreps_in: IrrepsLike,
        irreps_out: IrrepsLike,
        channels_in: int,
        channels_out: Optional[int] = None,
        *,
        internal_weights: bool = True,
        bias: bool = True,
        path_norm: bool = True,
        path: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        if channels_out is None:
            channels_out = channels_in
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (channels_in, channels_out)
        ):
            raise TypeError("Linear channel counts must be integers.")
        if channels_in < 1 or channels_out < 1:
            raise ValueError("Linear channel counts must be positive.")
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.internal_weights = bool(internal_weights)
        self.path_norm = bool(path_norm)
        # Internal weights have unit variance. This fixed factor preserves
        # feature variance under the dense channel contraction.
        self.weight_scale = channels_in**-0.5

        inputs = self.irreps_in.expanded()
        outputs = self.irreps_out.expanded()
        if path is None:
            paths = tuple(
                (output_index, input_index)
                for output_index, output in enumerate(outputs)
                for input_index, input_irrep in enumerate(inputs)
                if output == input_irrep
            )
        else:
            paths = tuple(path)
        for output_index, input_index in paths:
            if not 0 <= output_index < len(outputs):
                raise ValueError(f"Invalid Linear output index: {output_index}.")
            if not 0 <= input_index < len(inputs):
                raise ValueError(f"Invalid Linear input index: {input_index}.")
            if outputs[output_index] != inputs[input_index]:
                raise ValueError("Linear paths must connect identical O(3) irreps.")
        self.path = paths
        self._input_slices = self.irreps_in.expanded_slices()
        paths_by_output = [[] for _ in outputs]
        for path_index, (output_index, input_index) in enumerate(paths):
            paths_by_output[output_index].append((path_index, input_index))
        self._paths_by_output = tuple(tuple(group) for group in paths_by_output)

        counts = Counter(output_index for output_index, _ in paths)
        self.register_buffer(
            "path_scales",
            torch.tensor(
                [
                    counts[output_index] ** -0.5 if path_norm else 1.0
                    for output_index, _ in paths
                ],
                dtype=torch.get_default_dtype(),
            ),
            persistent=False,
        )

        self.weight_shape = (len(paths), channels_in, channels_out)
        self.weight_numel = math.prod(self.weight_shape)
        if self.internal_weights:
            self.weight = torch.nn.Parameter(torch.empty(self.weight_shape))
        else:
            self.register_parameter("weight", None)

        bias_indices = []
        num_biases = 0
        for output in outputs:
            if output == Irrep("0e"):
                bias_indices.append(num_biases)
                num_biases += 1
            else:
                bias_indices.append(-1)
        self._bias_indices = tuple(bias_indices)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(num_biases, channels_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _resolve_weight(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if self.internal_weights:
            if weight is not None:
                raise ValueError("Do not pass weight when using internal weights.")
            weight = self.weight
        elif weight is None:
            raise ValueError("Linear requires external weight.")
        if weight is None:
            raise RuntimeError("Linear weight resolution failed.")
        if weight.is_complex():
            raise TypeError("Cartesian O(3) Linear supports real weights only.")
        ndim = len(self.weight_shape)
        if weight.ndim < ndim or tuple(weight.shape[-ndim:]) != self.weight_shape:
            raise ValueError(
                f"Linear weight trailing shape must be {self.weight_shape}, "
                f"got {tuple(weight.shape)}."
            )
        return weight * self.weight_scale

    def forward(
        self,
        input: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        expected = (self.irreps_in.dim, self.channels_in)
        if input.is_complex():
            raise TypeError("Cartesian O(3) Linear supports real inputs only.")
        if input.ndim < 2 or tuple(input.shape[-2:]) != expected:
            raise ValueError(
                f"Linear input trailing shape must be {expected}, "
                f"got {tuple(input.shape)}."
            )
        weight = self._resolve_weight(weight)
        weight_ndim = len(self.weight_shape)
        try:
            leading_shape = torch.broadcast_shapes(
                input.shape[:-2], weight.shape[:-weight_ndim]
            )
        except RuntimeError as error:
            raise ValueError(
                "Linear input and weight batches do not broadcast."
            ) from error
        input = input.expand(leading_shape + expected)
        weight = weight.expand(leading_shape + self.weight_shape)
        outputs = self.irreps_out.expanded()
        zero = input.sum() * 0 + weight.sum() * 0
        blocks = []
        for output_index, output_irrep in enumerate(outputs):
            contributions = []
            for path_index, input_index in self._paths_by_output[output_index]:
                block = input[..., self._input_slices[input_index], :]
                value = torch.einsum(
                    "...di,...io->...do", block, weight[..., path_index, :, :]
                )
                contributions.append(value * self.path_scales[path_index])
            if contributions:
                block = sum(contributions[1:], contributions[0])
            else:
                block = input.new_zeros(
                    leading_shape + (output_irrep.dim, self.channels_out)
                )
                block = block + zero
            bias_index = self._bias_indices[output_index]
            if self.bias is not None and bias_index >= 0:
                block = block + self.bias[bias_index]
            blocks.append(block)
        if blocks:
            return torch.cat(blocks, dim=-2)
        return input.new_empty(leading_shape + (0, self.channels_out)) + zero

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({self.channels_in * self.irreps_in} -> "
            f"{self.channels_out * self.irreps_out} | {self.weight_numel} weights)"
            f"(bias={self.bias is not None})"
        )
