################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from e3nn import o3

from equivariantx.mbe.ace.paths import (
    SymmetricProductPaths,
    asymmetric_contraction_paths,
    generate_paths,
    satisfy,
    symmetric_contraction_paths,
)

__all__ = [
    "SymmetricProductPaths",
    "asymmetric_contraction_paths",
    "generate_paths",
    "satisfy",
    "symmetric_contraction_paths",
    "to_possible_tp_irreps",
]


def to_possible_tp_irreps(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    parity: bool | None = None,
    lmax: int | None = None,
    *,
    target_irreps: o3.Irreps | None = None,
) -> o3.Irreps:
    """Return the allowed irreps reachable by a tensor product.

    Parameters
    ----------
    irreps_in1, irreps_in2 : o3.Irreps
        Input representations. Multiplicities do not affect selection.
    parity : bool or None
        False selects natural parity; True selects both spatial parities.
        None selects natural parity and the spatial irreps in ``target_irreps``.
    lmax : int, optional
        Maximum output degree. Defaults to the second input's maximum degree.
    target_irreps : o3.Irreps, optional
        Additional spatial irreps required by predicted or embedded properties.
        Time-reversal labels are determined by the tensor product.

    Returns
    -------
    o3.Irreps
        Sorted reachable irreps, with multiplicity one.
    """
    irreps_in1, irreps_in2 = o3.Irreps(irreps_in1), o3.Irreps(irreps_in2)
    if not irreps_in1 or not irreps_in2:
        return o3.Irreps([])
    lmax = irreps_in2.lmax if lmax is None else lmax
    targets = {(ir.l, ir.p) for _, ir in o3.Irreps(target_irreps or [])}
    irreps = {
        ir_out
        for ir1 in {ir for _, ir in irreps_in1}
        for ir2 in {ir for _, ir in irreps_in2}
        for ir_out in ir1 * ir2
        if ir_out.l <= lmax
        and (
            parity is True
            or ir_out.p == (-1) ** ir_out.l
            or (parity is None and (ir_out.l, ir_out.p) in targets)
        )
    }
    return o3.Irreps([(1, ir) for ir in irreps]).sort().irreps
