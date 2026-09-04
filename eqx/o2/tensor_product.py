################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Iterator, NamedTuple, Optional, Sequence

import torch

from .irreps import Irrep, Irreps, IrrepsLike


def _quarter_turn(features: torch.Tensor) -> torch.Tensor:
    return torch.stack((-features[..., 1, :], features[..., 0, :]), dim=-2)


def _cg_product(
    input1: torch.Tensor,
    irrep1: Irrep,
    input2: torch.Tensor,
    irrep2: Irrep,
    irrep_out: Irrep,
) -> torch.Tensor:
    """Evaluate one component-normalized real O(2) Clebsch--Gordan map."""
    if irrep_out not in irrep1 * irrep2:
        raise ValueError(
            f"Illegal O(2) tensor-product path: {irrep1} x {irrep2} -> {irrep_out}."
        )
    if irrep1.m == 0 and irrep2.m == 0:
        value = input1[..., 0, :, None] * input2[..., 0, None, :]
        return value.unsqueeze(-3)
    if irrep1.m == 0:
        vector = _quarter_turn(input2) if irrep1.p == -1 else input2
        return input1[..., 0, None, :, None] * vector.unsqueeze(-2)
    if irrep2.m == 0:
        vector = _quarter_turn(input1) if irrep2.p == -1 else input1
        return vector.unsqueeze(-1) * input2[..., 0, None, None, :]

    first_real = input1[..., 0, :, None]
    first_imag = input1[..., 1, :, None]
    second_real = input2[..., 0, None, :]
    second_imag = input2[..., 1, None, :]
    scale = math.sqrt(0.5)
    if irrep_out.m == irrep1.m + irrep2.m:
        real = first_real * second_real - first_imag * second_imag
        imaginary = first_real * second_imag + first_imag * second_real
        return torch.stack((real, imaginary), dim=-3) * scale

    real = first_real * second_real + first_imag * second_imag
    imaginary = first_imag * second_real - first_real * second_imag
    if irrep2.m > irrep1.m:
        imaginary = -imaginary
    if irrep_out.m > 0:
        return torch.stack((real, imaginary), dim=-3) * scale
    if irrep_out.is_even_scalar():
        return real.unsqueeze(-3) * scale
    return imaginary.unsqueeze(-3) * scale


def _cg_product_uuu(
    input1: torch.Tensor,
    irrep1: Irrep,
    input2: torch.Tensor,
    irrep2: Irrep,
    irrep_out: Irrep,
) -> torch.Tensor:
    if input1.size(-1) != input2.size(-1):
        raise ValueError("uuu tensor products require equal multiplicities.")
    return _cg_product(
        input1,
        irrep1,
        input2,
        irrep2,
        irrep_out,
    ).diagonal(dim1=-2, dim2=-1)


class Instruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple[int, ...]


class TensorProduct(torch.nn.Module):
    """O(2) tensor product with time-reversal-aware coupling instructions.

    Parameters
    ----------
    irreps_in1 : IrrepsLike
        Representation of the first input.
    irreps_in2 : IrrepsLike
        Representation of the second input.
    irreps_out : IrrepsLike
        Requested output representation.
    instructions : sequence of tuple
        Coupling paths written as ``(i_in1, i_in2, i_out, mode, train)`` or
        ``(i_in1, i_in2, i_out, mode, train, path_weight)``. ``mode`` is one
        of ``"u1u"``, ``"uuu"``, or ``"uvw"``; ``train`` selects whether the
        path has a weight.
    in1_var : sequence of float, optional
        Expected variance for each first-input entry. Used only to calculate
        path normalization. Defaults to one.
    in2_var : sequence of float, optional
        Expected variance for each second-input entry. Defaults to one.
    out_var : sequence of float, optional
        Requested variance for each output entry. Defaults to one.
    irrep_normalization : {"component", "norm", "none"}, optional
        Normalization of each irreducible coupling map.
    path_normalization : {"element", "path", "none"}, optional
        Normalization across paths contributing to the same output entry.
    internal_weights : bool, optional
        If ``True``, trainable weights are stored by the module. The default
        is inferred from ``shared_weights`` and the weighted instructions.
    shared_weights : bool, optional
        Whether one weight vector is shared over all leading dimensions.

    Notes
    -----
    Inputs and outputs use ``(..., irreps.dim)`` tensors in flattened
    ``ir_mul`` order. ``u1u`` couples a single second-input channel to matched
    first/output channels, ``uuu`` couples matching channels, and ``uvw`` uses
    a dense ``mul1 x mul2 x mul_out`` weight tensor.
    """

    def __init__(
        self,
        irreps_in1: IrrepsLike,
        irreps_in2: IrrepsLike,
        irreps_out: IrrepsLike,
        instructions: Sequence[tuple],
        in1_var: Optional[Sequence[float]] = None,
        in2_var: Optional[Sequence[float]] = None,
        out_var: Optional[Sequence[float]] = None,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if irrep_normalization not in ("component", "norm", "none"):
            raise ValueError(
                "irrep_normalization must be 'component', 'norm', or 'none'."
            )
        if path_normalization not in ("element", "path", "none"):
            raise ValueError("path_normalization must be 'element', 'path', or 'none'.")
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization

        raw_instructions = [
            instruction if len(instruction) == 6 else instruction + (1.0,)
            for instruction in instructions
        ]
        parsed = []
        for instruction in raw_instructions:
            if len(instruction) != 6:
                raise TypeError(
                    "TensorProduct instructions must be "
                    "(i_in1, i_in2, i_out, mode, train[, path_weight])."
                )
            i_in1, i_in2, i_out, mode, train, path_weight = instruction
            if mode not in ("u1u", "uuu", "uvw"):
                raise ValueError("connection_mode must be 'u1u', 'uuu', or 'uvw'.")
            if not 0 <= i_in1 < len(self.irreps_in1):
                raise IndexError(f"{i_in1} is not a valid irreps_in1 index.")
            if not 0 <= i_in2 < len(self.irreps_in2):
                raise IndexError(f"{i_in2} is not a valid irreps_in2 index.")
            if not 0 <= i_out < len(self.irreps_out):
                raise IndexError(f"{i_out} is not a valid irreps_out index.")
            ir1, mul1 = self.irreps_in1[i_in1]
            ir2, mul2 = self.irreps_in2[i_in2]
            ir_out, mul_out = self.irreps_out[i_out]
            if ir_out not in ir1 * ir2:
                raise ValueError(
                    f"Illegal O(2) TensorProduct instruction: "
                    f"{ir1} x {ir2} -> {ir_out}."
                )
            if mode == "u1u" and not (mul2 == 1 and mul1 == mul_out):
                raise ValueError(
                    "connection_mode='u1u' requires mul_in2=1 and mul_in1=mul_out."
                )
            if mode == "uuu" and not (mul1 == mul2 == mul_out):
                raise ValueError("connection_mode='uuu' requires equal multiplicities.")
            path_shape = {
                "u1u": (mul1,),
                "uuu": (mul1,),
                "uvw": (mul1, mul2, mul_out),
            }[mode]
            parsed.append(
                Instruction(
                    i_in1,
                    i_in2,
                    i_out,
                    mode,
                    bool(train),
                    float(path_weight),
                    path_shape,
                )
            )

        def variances(
            values: Optional[Sequence[float]],
            size: int,
            name: str,
        ) -> tuple[float, ...]:
            if values is None:
                return (1.0,) * size
            values = tuple(float(value) for value in values)
            if len(values) != size:
                raise ValueError(f"{name} must have one value per irrep entry.")
            return values

        in1_var = variances(in1_var, len(self.irreps_in1), "in1_var")
        in2_var = variances(in2_var, len(self.irreps_in2), "in2_var")
        out_var = variances(out_var, len(self.irreps_out), "out_var")

        def num_elements(instruction: Instruction) -> int:
            mul1 = self.irreps_in1[instruction.i_in1].mul
            mul2 = self.irreps_in2[instruction.i_in2].mul
            return mul1 * mul2 if instruction.connection_mode == "uvw" else 1

        normalized = []
        for instruction in parsed:
            ir1 = self.irreps_in1[instruction.i_in1].ir
            ir2 = self.irreps_in2[instruction.i_in2].ir
            ir_out = self.irreps_out[instruction.i_out].ir
            if irrep_normalization == "component":
                coefficient = 1.0
            elif irrep_normalization == "norm":
                coefficient = ir1.dim * ir2.dim / ir_out.dim
            else:
                coefficient = 1.0

            if path_normalization == "element":
                denominator = sum(
                    in1_var[item.i_in1] * in2_var[item.i_in2] * num_elements(item)
                    for item in parsed
                    if item.i_out == instruction.i_out
                )
            elif path_normalization == "path":
                denominator = (
                    in1_var[instruction.i_in1]
                    * in2_var[instruction.i_in2]
                    * num_elements(instruction)
                    * sum(item.i_out == instruction.i_out for item in parsed)
                )
            else:
                denominator = 1.0
            if denominator > 0:
                coefficient /= denominator
            coefficient *= out_var[instruction.i_out]
            coefficient *= instruction.path_weight
            normalized.append(instruction._replace(path_weight=math.sqrt(coefficient)))
        self.instructions = tuple(normalized)

        if shared_weights is None:
            shared_weights = True
        if internal_weights is None:
            internal_weights = bool(shared_weights) and any(
                instruction.has_weight for instruction in self.instructions
            )
        if internal_weights and not shared_weights:
            raise ValueError("Internal weights require shared_weights=True.")
        self.internal_weights = bool(internal_weights)
        self.shared_weights = bool(shared_weights)
        self.weight_numel = sum(
            math.prod(instruction.path_shape)
            for instruction in self.instructions
            if instruction.has_weight
        )
        self.weight_shape = (self.weight_numel,)
        if self.internal_weights and self.weight_numel > 0:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.empty(0))

        self._input1_slices = self.irreps_in1.slices()
        self._input2_slices = self.irreps_in2.slices()
        offsets = []
        offset = 0
        for instruction in self.instructions:
            if instruction.has_weight:
                size = math.prod(instruction.path_shape)
                offsets.append((offset, size))
                offset += size
            else:
                offsets.append(None)
        self._weight_offsets = tuple(offsets)

        output_mask = []
        for i_out, ir_mul in enumerate(self.irreps_out):
            connected = any(
                instruction.i_out == i_out and instruction.path_weight != 0
                for instruction in self.instructions
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
            raise TypeError("O(2) TensorProduct supports real weights only.")
        if weight.ndim < 1 or weight.size(-1) != self.weight_numel:
            raise ValueError(
                "TensorProduct weight trailing dimension must be "
                f"{self.weight_numel}, got {tuple(weight.shape)}."
            )
        return weight

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate all tensor-product instructions.

        Parameters
        ----------
        input1 : torch.Tensor
            First input with shape ``(..., irreps_in1.dim)``.
        input2 : torch.Tensor
            Second input with shape ``(..., irreps_in2.dim)``.
        weight : torch.Tensor, optional
            External flattened weights with trailing shape ``weight_shape``.
            Leading dimensions broadcast with both inputs. Omit when internal
            weights are enabled.

        Returns
        -------
        torch.Tensor
            Coupled features with shape ``(..., irreps_out.dim)`` over the
            broadcast leading shape.
        """
        if input1.is_complex() or input2.is_complex():
            raise TypeError("O(2) TensorProduct supports real inputs only.")
        if input1.ndim < 1 or input1.size(-1) != self.irreps_in1.dim:
            raise ValueError(
                "TensorProduct input1 trailing dimension must be "
                f"{self.irreps_in1.dim}, got {tuple(input1.shape)}."
            )
        if input2.ndim < 1 or input2.size(-1) != self.irreps_in2.dim:
            raise ValueError(
                "TensorProduct input2 trailing dimension must be "
                f"{self.irreps_in2.dim}, got {tuple(input2.shape)}."
            )
        weight = self._resolve_weight(weight)
        try:
            leading_shape = torch.broadcast_shapes(
                input1.shape[:-1],
                input2.shape[:-1],
                weight.shape[:-1],
            )
        except RuntimeError as error:
            raise ValueError(
                "TensorProduct input and weight batch dimensions do not broadcast."
            ) from error

        outputs = []
        zero = input1.sum() * 0 + input2.sum() * 0 + weight.sum() * 0
        for i_out, (ir_out, mul_out) in enumerate(self.irreps_out):
            contributions = []
            for instruction_index, instruction in enumerate(self.instructions):
                if instruction.i_out != i_out:
                    continue
                ir1, mul1 = self.irreps_in1[instruction.i_in1]
                ir2, mul2 = self.irreps_in2[instruction.i_in2]
                values1 = input1[..., self._input1_slices[instruction.i_in1]].reshape(
                    *input1.shape[:-1], ir1.dim, mul1
                )
                values2 = input2[..., self._input2_slices[instruction.i_in2]].reshape(
                    *input2.shape[:-1], ir2.dim, mul2
                )
                product = _cg_product(values1, ir1, values2, ir2, ir_out)
                if instruction.connection_mode == "uvw":
                    if instruction.has_weight:
                        offset, size = self._weight_offsets[instruction_index]
                        path_weight = weight.narrow(-1, offset, size).reshape(
                            *weight.shape[:-1], mul1, mul2, mul_out
                        )
                        contribution = torch.einsum(
                            "...duv,...uvw->...dw",
                            product,
                            path_weight,
                        )
                    else:
                        raise ValueError("uvw instructions require weights.")
                elif instruction.connection_mode == "u1u":
                    contribution = product[..., 0]
                    if instruction.has_weight:
                        offset, size = self._weight_offsets[instruction_index]
                        path_weight = weight.narrow(-1, offset, size).reshape(
                            *weight.shape[:-1], mul_out
                        )
                        contribution = contribution * path_weight.unsqueeze(-2)
                else:
                    contribution = product.diagonal(dim1=-2, dim2=-1)
                    if instruction.has_weight:
                        offset, size = self._weight_offsets[instruction_index]
                        path_weight = weight.narrow(-1, offset, size).reshape(
                            *weight.shape[:-1], mul_out
                        )
                        contribution = contribution * path_weight.unsqueeze(-2)
                contributions.append(contribution * instruction.path_weight)
            if contributions:
                output = sum(contributions[1:], contributions[0])
            else:
                output = (
                    input1.new_zeros(
                        *leading_shape,
                        ir_out.dim,
                        mul_out,
                    )
                    + zero
                )
            outputs.append(output.reshape(*leading_shape, ir_out.dim * mul_out))
        if outputs:
            return torch.cat(outputs, dim=-1)
        return input1.new_empty(*leading_shape, 0) + zero

    def weight_view_for_instruction(
        self,
        instruction: int,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return an instruction-shaped view of the weight storage.

        Parameters
        ----------
        instruction : int
            Index into :attr:`instructions`.
        weight : torch.Tensor, optional
            Weight storage to view. The module weight is used when omitted.

        Returns
        -------
        torch.Tensor
            View whose trailing dimensions equal the instruction path shape.
        """
        specification = self.instructions[instruction]
        if not specification.has_weight:
            raise ValueError("The selected instruction has no weights.")
        weight = self._resolve_weight(weight)
        offset, size = self._weight_offsets[instruction]
        return weight.narrow(-1, offset, size).view(
            *weight.shape[:-1],
            *specification.path_shape,
        )

    def weight_views(
        self,
        weight: Optional[torch.Tensor] = None,
        yield_instruction: bool = False,
    ) -> Iterator:
        """Iterate over the weighted instruction views.

        Parameters
        ----------
        weight : torch.Tensor, optional
            Weight storage to view. The module weight is used when omitted.
        yield_instruction : bool, optional
            If ``True``, also yield the instruction index and metadata.

        Yields
        ------
        torch.Tensor or tuple
            One weight view, optionally with its instruction metadata.
        """
        for index, instruction in enumerate(self.instructions):
            if not instruction.has_weight:
                continue
            view = self.weight_view_for_instruction(index, weight)
            yield (index, instruction, view) if yield_instruction else view

    def __repr__(self) -> str:
        num_paths = sum(math.prod(item.path_shape) for item in self.instructions)
        return (
            f"{self.__class__.__name__}({self.irreps_in1.simplify()} x "
            f"{self.irreps_in2.simplify()} -> {self.irreps_out.simplify()} | "
            f"{num_paths} paths | {self.weight_numel} weights)"
        )
