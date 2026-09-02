################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from eqx import co3


def satisfy(l1: int, l2: int, rule) -> bool:
    if rule is None:
        return True
    if rule == "<=":
        return l1 <= l2
    if rule == ">=":
        return l1 >= l2
    raise ValueError(f"Unknown l1l2 restriction: {rule!r}.")


def generate_paths(
    irreps_in1,
    irreps_in2,
    irreps_out,
    *,
    mode: str,
    trainable: bool,
    l1l2=None,
    identical_inputs: bool = False,
) -> tuple[list[tuple[int, int, int, str, bool]], co3.Irreps]:
    """Return one output entry and instruction for every coupling path."""
    irreps_in1 = co3.Irreps(irreps_in1)
    irreps_in2 = co3.Irreps(irreps_in2)
    irreps_out = co3.Irreps(irreps_out)
    if identical_inputs and irreps_in1 != irreps_in2:
        raise ValueError("identical_inputs requires matching input irreps.")

    instructions = []
    output_entries = []
    for ir_out, _ in irreps_out:
        for i1, (ir1, mul1) in enumerate(irreps_in1):
            for i2, (ir2, mul2) in enumerate(irreps_in2):
                if not satisfy(ir1.l, ir2.l, l1l2) or ir_out not in ir1 * ir2:
                    continue
                if (
                    identical_inputs
                    and i1 == i2
                    and (ir1.l + ir2.l - ir_out.l) % 2 == 1
                ):
                    continue
                if mode == "u1u":
                    if mul2 != 1:
                        raise ValueError("u1u paths require multiplicity-one input2.")
                    mul_out = mul1
                elif mode == "uuu":
                    if mul1 != mul2:
                        raise ValueError("uuu paths require equal input multiplicities.")
                    mul_out = mul1
                elif mode == "uvw":
                    raise ValueError("Path-expanded uvw outputs require an explicit output.")
                else:
                    raise ValueError(f"Unknown tensor-product mode: {mode!r}.")
                i_out = len(output_entries)
                output_entries.append((ir_out, mul_out))
                instructions.append((i1, i2, i_out, mode, trainable))
    return instructions, co3.Irreps(output_entries)


__all__ = [
    "generate_paths",
    "satisfy",
]
