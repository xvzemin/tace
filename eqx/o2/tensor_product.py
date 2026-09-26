################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Iterator, NamedTuple, Optional, Sequence

import torch

from ._clebsch_gordan import clebsch_gordan_products
from .irreps import Irreps, IrrepsLike


class Instruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple[int, ...]


class TensorProduct(torch.nn.Module):
    """Tensor product with parametrized O(2) coupling paths.

    Parameters
    ----------
    irreps_in1 : Irreps, str, or sequence
        Representation of the first input.
    irreps_in2 : Irreps, str, or sequence
        Representation of the second input.
    irreps_out : Irreps, str, or sequence
        Requested output representation.
    instructions : sequence of tuple
        Coupling paths written as ``(i_in1, i_in2, i_out, mode, train)`` or
        ``(i_in1, i_in2, i_out, mode, train, path_weight)``. ``"u1u"`` couples
        one second-input channel to matching first-input/output channels;
        ``"uuu"`` couples matching channels; ``"uvw"`` mixes all channels.
        ``train`` selects whether the path has a weight.
    in1_var : sequence of float, optional
        Expected variance for each first-input entry. Used only to calculate
        path normalization. Defaults to one.
    in2_var : sequence of float, optional
        Expected variance for each second-input entry. Defaults to one.
    out_var : sequence of float, optional
        Requested variance for each output entry. Defaults to one.
    irrep_normalization : {"component", "norm", "none"}, optional
        ``"component"`` gives unit output variance for independent unit-variance
        input components. ``"norm"`` rescales by
        ``sqrt(dim_in1 * dim_in2 / dim_out)``. ``"none"`` uses coupling tensors
        with unit squared Frobenius norm.
    path_normalization : {"element", "path", "none"}, optional
        Normalization across paths contributing to the same output entry.
    internal_weights : bool, optional
        If ``True``, trainable weights are stored by the module. The default
        is inferred from ``shared_weights`` and the weighted instructions.
    shared_weights : bool, optional
        Whether one weight vector is shared over all leading dimensions.
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

        parsed = []
        for instruction in instructions:
            instruction = tuple(instruction)
            if len(instruction) == 5:
                instruction += (1.0,)
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
            if mode == "uvw" and not train:
                raise ValueError("uvw instructions require weights.")
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
            if irrep_normalization == "norm":
                coefficient = ir1.dim * ir2.dim / ir_out.dim
            elif irrep_normalization == "none":
                coefficient = 1.0 / ir_out.dim
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
        couplings = {}
        for instruction in self.instructions:
            key = (
                instruction.i_in1,
                instruction.i_in2,
                instruction.connection_mode != "uvw",
            )
            ir = self.irreps_out[instruction.i_out].ir
            outputs = couplings.setdefault(key, [])
            if ir not in outputs:
                outputs.append(ir)
        self._couplings = tuple((*key, tuple(irs)) for key, irs in couplings.items())
        indices = {
            (i1, i2, elementwise, ir): (i, j)
            for i, (i1, i2, elementwise, irs) in enumerate(self._couplings)
            for j, ir in enumerate(irs)
        }
        self._coupling_indices = tuple(
            indices[
                instruction.i_in1,
                instruction.i_in2,
                instruction.connection_mode != "uvw",
                self.irreps_out[instruction.i_out].ir,
            ]
            for instruction in self.instructions
        )
        self._instructions_by_coupling = tuple(
            tuple(
                j for j, (index, _) in enumerate(self._coupling_indices) if index == i
            )
            for i in range(len(self._couplings))
        )

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

        self._input1_sizes = tuple(ir_mul.dim for ir_mul in self.irreps_in1)
        self._input2_sizes = tuple(ir_mul.dim for ir_mul in self.irreps_in2)
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
        self._weight_sizes = tuple(0 if item is None else item[1] for item in offsets)
        self._instructions_by_output = tuple(
            tuple(i for i, ins in enumerate(self.instructions) if ins.i_out == i_out)
            for i_out in range(len(self.irreps_out))
        )

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

    def _get_weights(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
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
            First input with shape ``(..., irreps_in1.dim)`` in flattened
            ``ir_mul`` order.
        input2 : torch.Tensor
            Second input with shape ``(..., irreps_in2.dim)`` in the same layout.
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
        weight = self._get_weights(weight)
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

        values1 = [
            values.reshape(*input1.shape[:-1], ir.dim, mul)
            for (ir, mul), values in zip(
                self.irreps_in1, input1.split(self._input1_sizes, dim=-1)
            )
        ]
        values2 = [
            values.reshape(*input2.shape[:-1], ir.dim, mul)
            for (ir, mul), values in zip(
                self.irreps_in2, input2.split(self._input2_sizes, dim=-1)
            )
        ]
        weights = weight.split(self._weight_sizes, dim=-1)
        contributions = [None] * len(self.instructions)
        for (i1, i2, elementwise, irreps_out), indices in zip(
            self._couplings, self._instructions_by_coupling
        ):
            couplings = clebsch_gordan_products(
                values1[i1],
                self.irreps_in1[i1].ir,
                values2[i2],
                self.irreps_in2[i2].ir,
                irreps_out,
                elementwise=elementwise,
            )
            for instruction_index in indices:
                instruction = self.instructions[instruction_index]
                _, mul1 = self.irreps_in1[instruction.i_in1]
                _, mul2 = self.irreps_in2[instruction.i_in2]
                _, index = self._coupling_indices[instruction_index]
                contribution = couplings[index]
                if instruction.has_weight:
                    path_weight = weights[instruction_index]
                    if instruction.connection_mode == "uvw":
                        mul_out = self.irreps_out[instruction.i_out].mul
                        path_weight = path_weight.reshape(
                            *weight.shape[:-1], mul1, mul2, mul_out
                        )
                        contribution = torch.einsum(
                            "...duv,...uvw->...dw", contribution, path_weight
                        )
                    else:
                        contribution = contribution * path_weight.unsqueeze(-2)
                contributions[instruction_index] = (
                    contribution * instruction.path_weight
                )
        outputs = []
        zero = None
        for i_out, (ir_out, mul_out) in enumerate(self.irreps_out):
            values = [contributions[i] for i in self._instructions_by_output[i_out]]
            if values:
                output = sum(values[1:], values[0])
            else:
                if zero is None:
                    zero = (
                        input1[..., :0].sum()
                        + input2[..., :0].sum()
                        + weight[..., :0].sum()
                    )
                output = input1.new_zeros((*leading_shape, ir_out.dim, mul_out)) + zero
            outputs.append(
                output.expand(*leading_shape, ir_out.dim, mul_out).reshape(
                    *leading_shape, ir_out.dim * mul_out
                )
            )
        if outputs:
            return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        return input1.new_empty((*leading_shape, 0)) + (
            input1[..., :0].sum() + input2[..., :0].sum() + weight[..., :0].sum()
        )

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
        weight = self._get_weights(weight)
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
