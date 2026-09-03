################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from e3nn import o3


def supports_time_reversal() -> bool:
    """Return whether the installed e3nn represents time-reversal parity."""

    return hasattr(o3.Irrep("0e"), "t")


def make_irrep(l: int, p: int, time_reversal: int = 1) -> o3.Irrep:
    """Construct an irrep supported by the installed e3nn."""

    if supports_time_reversal():
        return o3.Irrep(l, p, time_reversal)
    return o3.Irrep(l, p)


def with_time_reversal(
    irreps: o3.Irreps,
    time_reversal: int,
) -> o3.Irreps:
    """Attach one time-reversal parity to every irrep when it is supported."""

    irreps = o3.Irreps(irreps)
    if not supports_time_reversal():
        return irreps
    return o3.Irreps(
        [(mul, make_irrep(ir.l, ir.p, time_reversal)) for mul, ir in irreps]
    )


def contains_time_odd_irreps(*irreps: o3.Irreps) -> bool:
    """Return whether any representation contains a time-odd irrep."""

    return supports_time_reversal() and any(
        ir.t == -1 for rep in irreps for _, ir in o3.Irreps(rep)
    )


def spherical_harmonics_irreps(
    lmax: int,
    p: int = -1,
    time_reversal: int = 1,
) -> o3.Irreps:
    """Return spherical-harmonic irreps for an input with parities ``p`` and ``t``."""

    irreps = o3.Irreps.spherical_harmonics(lmax, p=p)
    if not supports_time_reversal() or time_reversal == 1:
        return irreps
    return o3.Irreps(
        [(mul, make_irrep(ir.l, ir.p, time_reversal**ir.l)) for mul, ir in irreps]
    )
