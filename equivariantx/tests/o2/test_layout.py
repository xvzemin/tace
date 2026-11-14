"""O(2) feature blocks and Wigner row selection."""

import pytest

from equivariantx import o2
from equivariantx.o2.layout import irrep_layout, wigner_indices, wigner_orders


@pytest.mark.parametrize(
    "transverse,expected",
    [
        (False, ((1, 2, 0), (2, 3, 2), (1, 1, 8), (2, 2, 9))),
        (True, ((1, 2, 0), (5, 3, 2), (1, 1, 17), (3, 2, 18))),
    ],
)
def test_feature_blocks(transverse, expected):
    irreps = o2.Irreps("2x0e+3x2m+1x0o+2x1m")
    assert irrep_layout(irreps, transverse) == expected
    assert irrep_layout(o2.Irreps(), transverse) == ()


@pytest.mark.parametrize("lmax,mmax", [(l, m) for l in range(5) for m in range(l + 1)])
def test_wigner_rows(lmax, mmax):
    wigner = o2.WignerD(mmax, lmax)
    local = wigner.local_indices.tolist()
    covered = []
    for degree in range(lmax + 1):
        rows = wigner_indices(degree, mmax, lmax)
        orders = [
            0,
            *[signed for m in range(1, min(degree, mmax) + 1) for signed in (m, -m)],
        ]
        assert [local[row] for row in rows] == [degree**2 + degree + m for m in orders]
        covered.extend(rows)
    assert sorted(covered) == list(range(len(local)))
    assert wigner_orders((lmax + 1) ** 2, len(local)) == (lmax, mmax)
    assert wigner_orders((lmax + 1) ** 2, len(local), lmax=lmax, mmax=mmax) == (
        lmax,
        mmax,
    )
    with pytest.raises(ValueError):
        wigner_orders((lmax + 1) ** 2, len(local), lmax=lmax + 1)
    with pytest.raises(ValueError):
        wigner_orders((lmax + 1) ** 2, len(local), mmax=mmax + 1)


@pytest.mark.parametrize(
    "global_dim,local_dim", [(16, 15), (16, 13), (16, 17), (16, 0), (8, 8), (0, 0)]
)
def test_invalid_dimensions(global_dim, local_dim):
    with pytest.raises(ValueError):
        wigner_orders(global_dim, local_dim)
