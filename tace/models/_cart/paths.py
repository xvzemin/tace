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


def tensor_product_instructions(
    irreps_in1,
    irreps_in2,
    irreps_out,
    *,
    mode: str,
    trainable: bool,
    l1l2=None,
) -> list[tuple[int, int, int, str, bool]]:
    irreps_in1 = co3.Irreps(irreps_in1)
    irreps_in2 = co3.Irreps(irreps_in2)
    irreps_out = co3.Irreps(irreps_out)
    return [
        (i1, i2, i_out, mode, trainable)
        for i1, (ir1, _) in enumerate(irreps_in1)
        for i2, (ir2, _) in enumerate(irreps_in2)
        for i_out, (ir_out, _) in enumerate(irreps_out)
        if satisfy(ir1.l, ir2.l, l1l2) and ir_out in ir1 * ir2
    ]


__all__ = ["satisfy", "tensor_product_instructions"]
