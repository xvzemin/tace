################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import itertools
from typing import Union

from e3nn import o3

try:
    import cuequivariance as cue
    from cuequivariance.group_theory.experimental.e3nn import O3_e3nn
    from cuequivariance.group_theory.irreps_array.irrep_utils import into_list_of_irrep
except Exception:
    pass


def _to_cue_irreps(irreps: o3.Irreps):
    return cue.Irreps(
        O3_e3nn,
        [(mul, (ir.l, ir.p)) for mul, ir in o3.Irreps(irreps)],
    )


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


def generate_cueq_paths(
    irreps_out: o3.Irreps,
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    l1l2: Union[str, None] = None,
    l2l3: Union[str, None] = None,
    l3l1: Union[str, None] = None,
):
    irreps_in1: cue.Irreps = _to_cue_irreps(irreps_in1)
    irreps_in2: cue.Irreps = _to_cue_irreps(irreps_in2)
    irreps_out: cue.Irreps = _to_cue_irreps(irreps_out)
    G = irreps_in1.irrep_class
    target_irreps_out = into_list_of_irrep(G, irreps_out)

    d = cue.SegmentedTensorProduct.from_subscripts("uv,iu,jv,kuv+ijk")

    for mul1, ir1 in irreps_in1:
        d.add_segment(1, (ir1.dim, mul1))
    for mul2, ir2 in irreps_in2:
        d.add_segment(2, (ir2.dim, mul2))

    irreps_out_list = []
    for (i, (mul1, ir1)), (j, (mul2, ir2)) in itertools.product(
        enumerate(irreps_in1), enumerate(irreps_in2)
    ):
        for ir_out in ir1 * ir2:
            if ir_out not in target_irreps_out:
                continue

            l1 = ir1.l
            l2 = ir2.l
            l3 = ir_out.l

            if (
                satisfy(l1, l2, l1l2)
                and satisfy(l2, l3, l2l3)
                and satisfy(l3, l1, l3l1)
            ):
                for cg in cue.clebsch_gordan(ir1, ir2, ir_out):
                    d.add_path(None, i, j, None, c=cg, dims={"u": mul1, "v": mul2})
                    irreps_out_list.append((mul1 * mul2, ir_out))

    actual_irreps_out = cue.Irreps(G, irreps_out_list)
    actual_irreps_out, perm, inv = actual_irreps_out.sort()
    d = d.permute_segments(0, inv)
    d = d.permute_segments(3, inv)
    d = d.normalize_paths_for_operand(-1)
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps_in1.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps_in1, cue.ir_mul),
            cue.IrrepsAndLayout(irreps_in2, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(actual_irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def generate_cueq_uuu_paths(
    irreps_out: o3.Irreps,
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    l1l2: Union[str, None] = None,
    l2l3: Union[str, None] = None,
    l3l1: Union[str, None] = None,
    trainable: bool = False,
):
    """
    Cuequivariance descriptor for e3nn ``uuu`` tensor products.

    The output path order follows ``tace.models._e3nn.paths.generate_paths`` so
    this can be used as a drop-in replacement for ``o3.TensorProduct`` in ACE.
    """
    e3nn_irreps_out = o3.Irreps(irreps_out)
    irreps_in1: cue.Irreps = _to_cue_irreps(irreps_in1)
    irreps_in2: cue.Irreps = _to_cue_irreps(irreps_in2)
    irreps_out: cue.Irreps = _to_cue_irreps(e3nn_irreps_out)
    G = irreps_in1.irrep_class

    if trainable:
        d = cue.SegmentedTensorProduct.from_subscripts("u,iu,ju,ku+ijk")
        in1_operand = 1
        in2_operand = 2
    else:
        d = cue.SegmentedTensorProduct.from_subscripts("iu,ju,ku+ijk")
        in1_operand = 0
        in2_operand = 1

    for mul1, ir1 in irreps_in1:
        d.add_segment(in1_operand, (ir1.dim, mul1))
    for mul2, ir2 in irreps_in2:
        d.add_segment(in2_operand, (ir2.dim, mul2))

    irreps_out_list = []
    actual_e3nn_irreps_out = []
    for (_, e3nn_ir_out), (_, ir_out) in zip(e3nn_irreps_out, irreps_out):
        for i, (mul1, ir1) in enumerate(irreps_in1):
            for j, (mul2, ir2) in enumerate(irreps_in2):
                l1 = ir1.l
                l2 = ir2.l
                l3 = ir_out.l
                if (
                    ir_out in ir1 * ir2
                    and satisfy(l1, l2, l1l2)
                    and satisfy(l2, l3, l2l3)
                    and satisfy(l3, l1, l3l1)
                ):
                    if mul1 != mul2:
                        raise ValueError(
                            "cueq uuu tensor product requires equal "
                            f"multiplicities, got {mul1} and {mul2}"
                        )
                    for cg in cue.clebsch_gordan(ir1, ir2, ir_out):
                        if trainable:
                            d.add_path(None, i, j, None, c=cg, dims={"u": mul1})
                        else:
                            d.add_path(i, j, None, c=cg, dims={"u": mul1})
                        irreps_out_list.append((mul1, ir_out))
                        actual_e3nn_irreps_out.append((mul1, e3nn_ir_out))

    actual_irreps_out = cue.Irreps(G, irreps_out_list)
    d = d.normalize_paths_for_operand(-1)
    inputs = [
        cue.IrrepsAndLayout(irreps_in1, cue.ir_mul),
        cue.IrrepsAndLayout(irreps_in2, cue.ir_mul),
    ]
    if trainable:
        inputs.insert(
            0,
            cue.IrrepsAndLayout(
                irreps_in1.new_scalars(d.operands[0].size),
                cue.ir_mul,
            ),
        )
    descriptor = cue.EquivariantPolynomial(
        inputs,
        [cue.IrrepsAndLayout(actual_irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )
    return descriptor, o3.Irreps(actual_e3nn_irreps_out)
