################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Union

from e3nn import o3


def satisfy(l1: int, l2: int, restriction: Union[str, None] = None) -> bool:
    if restriction == None:
        return True
    elif restriction == "<":
        return l1 < l2
    elif restriction == "<=":
        return l1 <= l2
    elif restriction == ">":
        return l1 > l2
    elif restriction == ">=":
        return l1 >= l2
    elif restriction == "==":
        return l1 == l2
    elif restriction == "!=":
        return l1 != l2
    else:
        raise ValueError(f"Unknown restriction: {restriction}")


def generate_paths(
    irreps_out: o3.Irreps,
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    *,
    l1l2: Union[str, None] = None,
    l2l3: Union[str, None] = None,
    l3l1: Union[str, None] = None,
    e3nn_mode="uvu",
    trainable: bool = False,
    identical_inputs: bool = False,
):

    e3nn_paths: list[tuple[int, int, int, str, bool]] = []
    e3nn_out_irreps: list[tuple[int, o3.Irrep]] = []

    if identical_inputs and irreps_in1 != irreps_in2:
        raise ValueError("identical_inputs requires matching input irreps")

    for _, (_, ir_out) in enumerate(irreps_out):
        for i, (mul, ir1) in enumerate(irreps_in1):
            for j, (_, ir2) in enumerate(irreps_in2):
                l1 = ir1.l
                l2 = ir2.l
                l3 = ir_out.l

                if (
                    ir_out in ir1 * ir2
                    and satisfy(l1, l2, l1l2)
                    and satisfy(l2, l3, l2l3)
                    and satisfy(l3, l1, l3l1)
                ):
                    if identical_inputs and i == j and (l1 + l2 - l3) % 2 == 1:
                        continue

                    k = len(e3nn_out_irreps)
                    e3nn_out_irreps.append((mul, ir_out))
                    e3nn_paths.append(
                        (i, j, k, e3nn_mode, e3nn_mode == "uvu" or trainable)
                    )

    return e3nn_paths, o3.Irreps(e3nn_out_irreps)
