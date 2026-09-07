################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from e3nn import o3


def supports_time_reversal() -> bool:
    return hasattr(o3.Irrep("0e"), "t")


def make_irrep(l: int, p: int, t: int = 1) -> o3.Irrep:
    if supports_time_reversal():
        return o3.Irrep(l, p, t)
    return o3.Irrep(l, p)


def with_time_reversal(
    irreps: o3.Irreps,
    time_reversal: int,
) -> o3.Irreps:
    irreps = o3.Irreps(irreps)
    if not supports_time_reversal():
        return irreps
    return o3.Irreps(
        [(mul, make_irrep(ir.l, ir.p, time_reversal)) for mul, ir in irreps]
    )


def with_natural_parity(irreps: o3.Irreps) -> o3.Irreps:
    irreps = o3.Irreps(irreps)
    return o3.Irreps(
        [
            (
                mul,
                make_irrep(ir.l, (-1) ** ir.l, getattr(ir, "t", 1)),
            )
            for mul, ir in irreps
        ]
    )


def contains_time_odd_irreps(*irreps: o3.Irreps) -> bool:
    return supports_time_reversal() and any(
        ir.t == -1 for rep in irreps for _, ir in o3.Irreps(rep)
    )


def spherical_harmonics_irreps(
    lmax: int,
    p: int = -1,
    time_reversal: int = 1,
) -> o3.Irreps:
    irreps = o3.Irreps.spherical_harmonics(lmax, p=p)
    if not supports_time_reversal() or time_reversal == 1:
        return irreps
    return o3.Irreps(
        [(mul, make_irrep(ir.l, ir.p, time_reversal**ir.l)) for mul, ir in irreps]
    )
