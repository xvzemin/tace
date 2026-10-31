"""Tensor-product path metadata and variance normalization."""

import math
from typing import NamedTuple, Optional, Sequence


class Instruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple[int, ...]


def tensor_product_instructions(
    irreps_in1,
    irreps_in2,
    irreps_out,
    instructions,
    in1_var=None,
    in2_var=None,
    out_var=None,
    irrep_normalization="component",
    path_normalization="element",
    *,
    dimensions=None,
):
    """Validate paths and normalize weights using independent irrep dimensions."""
    if irrep_normalization not in ("component", "norm", "none"):
        raise ValueError("Invalid irrep_normalization.")
    if path_normalization not in ("element", "path", "none"):
        raise ValueError("Invalid path_normalization.")
    if dimensions is None:
        dimensions = tuple(
            tuple(entry.ir.dim for entry in irreps)
            for irreps in (irreps_in1, irreps_in2, irreps_out)
        )
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
        if not 0 <= i_in1 < len(irreps_in1):
            raise IndexError(f"{i_in1} is not a valid irreps_in1 index.")
        if not 0 <= i_in2 < len(irreps_in2):
            raise IndexError(f"{i_in2} is not a valid irreps_in2 index.")
        if not 0 <= i_out < len(irreps_out):
            raise IndexError(f"{i_out} is not a valid irreps_out index.")
        ir1, mul1 = irreps_in1[i_in1].ir, irreps_in1[i_in1].mul
        ir2, mul2 = irreps_in2[i_in2].ir, irreps_in2[i_in2].mul
        ir_out, mul_out = irreps_out[i_out].ir, irreps_out[i_out].mul
        if ir_out not in ir1 * ir2:
            raise ValueError(
                f"Illegal TensorProduct instruction: {ir1} x {ir2} -> {ir_out}."
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

    in1_var = variances(in1_var, len(irreps_in1), "in1_var")
    in2_var = variances(in2_var, len(irreps_in2), "in2_var")
    out_var = variances(out_var, len(irreps_out), "out_var")

    def num_elements(instruction: Instruction) -> int:
        mul1 = irreps_in1[instruction.i_in1].mul
        mul2 = irreps_in2[instruction.i_in2].mul
        return mul1 * mul2 if instruction.connection_mode == "uvw" else 1

    normalized = []
    for instruction in parsed:
        if irrep_normalization == "norm":
            coefficient = (
                dimensions[0][instruction.i_in1]
                * dimensions[1][instruction.i_in2]
                / dimensions[2][instruction.i_out]
            )
        elif irrep_normalization == "none":
            coefficient = 1.0 / dimensions[2][instruction.i_out]
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
    return tuple(normalized)
