################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Iterator, NamedTuple, Optional, Sequence, Union

import torch

from .irreps import Irreps, IrrepsLike


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple[int, ...]
    path_weight: float


class Linear(torch.nn.Module):
    """Apply an O(3)-equivariant linear map to flattened features.

    Parameters
    ----------
    irreps_in : IrrepsLike
        Representation carried by the input feature axis.
    irreps_out : IrrepsLike
        Representation carried by the output feature axis.
    internal_weights : bool, optional
        Store trainable weights and biases in the module. If ``False``, they
        must be supplied to :meth:`forward`.
    shared_weights : bool, optional
        Share one weight vector across all leading dimensions.
    instructions : sequence of tuple of int, optional
        Entry-level ``(i_in, i_out)`` paths. Every path must connect identical
        irreps. All compatible entries are connected by default.
    biases : bool or sequence of bool, optional
        Enable biases globally or per output entry. Only ``0e`` supports bias.
    path_normalization : {"element", "path"}, optional
        ``"element"`` normalizes by the total input multiplicity feeding an
        output. ``"path"`` assigns equal variance to every incoming path.

    Notes
    -----
    Features have shape ``(..., irreps.dim)``. Each entry uses flattened
    ``ir_mul`` order and is viewed internally as ``(..., ir.dim, mul)``.
    """

    def __init__(
        self,
        irreps_in: IrrepsLike,
        irreps_out: IrrepsLike,
        *,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[Sequence[tuple[int, int]]] = None,
        biases: Union[bool, Sequence[bool]] = False,
        path_normalization: str = "element",
    ) -> None:
        super().__init__()
        if path_normalization not in ("element", "path"):
            raise ValueError("path_normalization must be 'element' or 'path'.")
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.path_normalization = path_normalization

        if instructions is None:
            instructions = [
                (i_in, i_out)
                for i_in, (ir_in, _) in enumerate(self.irreps_in)
                for i_out, (ir_out, _) in enumerate(self.irreps_out)
                if ir_in == ir_out
            ]
        else:
            instructions = list(instructions)
        for instruction in instructions:
            if not isinstance(instruction, tuple) or len(instruction) != 2:
                raise TypeError("Each Linear instruction must be (i_in, i_out).")
            i_in, i_out = instruction
            if not isinstance(i_in, int) or isinstance(i_in, bool):
                raise TypeError("Linear instruction indices must be integers.")
            if not isinstance(i_out, int) or isinstance(i_out, bool):
                raise TypeError("Linear instruction indices must be integers.")
            if not 0 <= i_in < len(self.irreps_in):
                raise IndexError(f"{i_in} is not a valid index for irreps_in.")
            if not 0 <= i_out < len(self.irreps_out):
                raise IndexError(f"{i_out} is not a valid index for irreps_out.")
            if self.irreps_in[i_in].ir != self.irreps_out[i_out].ir:
                raise ValueError(f"{i_in} and {i_out} do not have the same irrep.")

        weighted_instructions = []
        for i_in, i_out in instructions:
            mul_in = self.irreps_in[i_in].mul
            if path_normalization == "element":
                denominator = sum(
                    self.irreps_in[j_in].mul
                    for j_in, j_out in instructions
                    if j_out == i_out
                )
            else:
                denominator = mul_in * sum(j_out == i_out for _, j_out in instructions)
            weighted_instructions.append(
                Instruction(
                    i_in,
                    i_out,
                    (mul_in, self.irreps_out[i_out].mul),
                    1.0 if denominator == 0 else denominator**-0.5,
                )
            )

        if isinstance(biases, bool):
            bias_list = [biases and ir.is_even_scalar() for ir, _ in self.irreps_out]
        else:
            bias_list = list(biases)
            if len(bias_list) != len(self.irreps_out):
                raise ValueError("biases must have one value per output entry.")
        for bias, (ir, _) in zip(bias_list, self.irreps_out):
            if bias and not ir.is_even_scalar():
                raise ValueError("Only inversion-even scalars can have biases.")

        bias_instructions = [
            Instruction(-1, i_out, (mul,), 1.0)
            for i_out, (bias, (_, mul)) in enumerate(zip(bias_list, self.irreps_out))
            if bias
        ]
        self.instructions = tuple(weighted_instructions + bias_instructions)
        self._weight_instructions = tuple(weighted_instructions)
        self._bias_instructions = tuple(bias_instructions)

        if shared_weights is False and internal_weights is None:
            internal_weights = False
        if shared_weights is None:
            shared_weights = True
        if internal_weights is None:
            internal_weights = True
        if internal_weights and not shared_weights:
            raise ValueError("Internal weights require shared_weights=True.")
        self.internal_weights = bool(internal_weights)
        self.shared_weights = bool(shared_weights)

        self.weight_numel = sum(
            math.prod(instruction.path_shape)
            for instruction in self._weight_instructions
        )
        self.bias_numel = sum(
            math.prod(instruction.path_shape) for instruction in self._bias_instructions
        )
        self.weight_shape = (self.weight_numel,)
        if self.internal_weights and self.weight_numel > 0:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.empty(0))
        if self.internal_weights and self.bias_numel > 0:
            self.bias = torch.nn.Parameter(torch.zeros(self.bias_numel))
        else:
            self.register_buffer("bias", torch.empty(0))

        self._input_slices = self.irreps_in.slices()
        weight_offsets = []
        offset = 0
        for instruction in self._weight_instructions:
            size = math.prod(instruction.path_shape)
            weight_offsets.append((offset, size))
            offset += size
        self._weight_offsets = tuple(weight_offsets)
        bias_offsets = {}
        offset = 0
        for instruction in self._bias_instructions:
            size = math.prod(instruction.path_shape)
            bias_offsets[instruction.i_out] = (offset, size)
            offset += size
        self._bias_offsets = bias_offsets

        output_mask = []
        for i_out, ir_mul in enumerate(self.irreps_out):
            connected = any(
                instruction.i_out == i_out for instruction in self.instructions
            )
            output_mask.append(
                torch.ones(ir_mul.dim) if connected else torch.zeros(ir_mul.dim)
            )
        self.register_buffer(
            "output_mask",
            torch.cat(output_mask) if output_mask else torch.ones(0),
            persistent=False,
        )

    def _resolve_weight(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Weights must be provided when internal_weights=False."
                )
            weight = self.weight
        if weight.is_complex():
            raise TypeError("O(3) Linear supports real weights only.")
        if weight.ndim < 1 or weight.size(-1) != self.weight_numel:
            raise ValueError(
                f"Linear weight trailing dimension must be {self.weight_numel}, "
                f"got {tuple(weight.shape)}."
            )
        return weight

    def _resolve_bias(self, bias: Optional[torch.Tensor]) -> torch.Tensor:
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Biases must be provided when internal_weights=False."
                )
            bias = self.bias
        if bias.is_complex():
            raise TypeError("O(3) Linear supports real biases only.")
        if bias.ndim < 1 or bias.size(-1) != self.bias_numel:
            raise ValueError(
                f"Linear bias trailing dimension must be {self.bias_numel}, "
                f"got {tuple(bias.shape)}."
            )
        return bias

    def forward(
        self,
        features: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the equivariant linear map.

        Parameters
        ----------
        features : torch.Tensor
            Input with shape ``(..., irreps_in.dim)``.
        weight : torch.Tensor, optional
            Flattened external weights with trailing dimension
            ``weight_numel``. Leading dimensions must broadcast with the
            input. Omit when internal weights are enabled.
        bias : torch.Tensor, optional
            Flattened external scalar biases with trailing dimension
            ``bias_numel``. Omit when internal biases are enabled.

        Returns
        -------
        torch.Tensor
            Output with shape ``(..., irreps_out.dim)`` over the broadcast
            leading dimensions.
        """
        if features.is_complex():
            raise TypeError("O(3) Linear supports real features only.")
        if features.ndim < 1 or features.size(-1) != self.irreps_in.dim:
            raise ValueError(
                f"Linear feature trailing dimension must be {self.irreps_in.dim}, "
                f"got {tuple(features.shape)}."
            )
        weight = self._resolve_weight(weight)
        bias = self._resolve_bias(bias)
        try:
            leading_shape = torch.broadcast_shapes(
                features.shape[:-1], weight.shape[:-1], bias.shape[:-1]
            )
        except RuntimeError as error:
            raise ValueError(
                "Linear feature, weight, and bias batch dimensions do not broadcast."
            ) from error

        outputs = []
        zero = features.sum() * 0 + weight.sum() * 0 + bias.sum() * 0
        for i_out, (ir_out, mul_out) in enumerate(self.irreps_out):
            contributions = []
            for instruction_index, instruction in enumerate(self._weight_instructions):
                if instruction.i_out != i_out:
                    continue
                mul_in, _ = instruction.path_shape
                values = features[..., self._input_slices[instruction.i_in]].reshape(
                    *features.shape[:-1], ir_out.dim, mul_in
                )
                offset, size = self._weight_offsets[instruction_index]
                matrix = weight.narrow(-1, offset, size).reshape(
                    *weight.shape[:-1], mul_in, mul_out
                )
                contributions.append(
                    torch.matmul(values, matrix) * instruction.path_weight
                )
            if contributions:
                output = sum(contributions[1:], contributions[0])
            else:
                output = features.new_zeros(*leading_shape, ir_out.dim, mul_out) + zero
            bias_specification = self._bias_offsets.get(i_out)
            if bias_specification is not None:
                offset, size = bias_specification
                output = output + bias.narrow(-1, offset, size).unsqueeze(-2)
            outputs.append(output.reshape(*leading_shape, ir_out.dim * mul_out))
        if outputs:
            return torch.cat(outputs, dim=-1)
        return features.new_empty(*leading_shape, 0) + zero

    def weight_view_for_instruction(
        self,
        instruction: int,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the ``(mul_in, mul_out)`` weight view for one instruction."""
        weight = self._resolve_weight(weight)
        weighted_instruction = self._weight_instructions[instruction]
        offset, size = self._weight_offsets[instruction]
        return weight.narrow(-1, offset, size).view(
            *weight.shape[:-1], *weighted_instruction.path_shape
        )

    def weight_views(
        self,
        weight: Optional[torch.Tensor] = None,
        yield_instruction: bool = False,
    ) -> Iterator:
        """Iterate over instruction-shaped weight views."""
        for index, instruction in enumerate(self._weight_instructions):
            view = self.weight_view_for_instruction(index, weight)
            yield (index, instruction, view) if yield_instruction else view

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.irreps_in} -> "
            f"{self.irreps_out} | {self.weight_numel} weights)"
        )
