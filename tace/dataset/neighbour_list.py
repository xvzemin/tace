################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional, Tuple, Union


import numpy as np
from matscipy.neighbours import neighbour_list


def filter_max_neighbors(source, target, shifts, distances, max_neighbors="inf"):

    if max_neighbors is None or max_neighbors == "inf":
        return source, target, shifts
    order = np.lexsort((distances, source))
    src_sorted = source[order]
    dst_sorted = target[order]
    shifts_sorted = shifts[order]

    unique_src, counts = np.unique(src_sorted, return_counts=True)
    cum_counts = np.cumsum(counts)  # [3, 2, 1] => [3, 5, 6]

    mask = np.zeros(len(src_sorted), dtype=bool)
    start_idx = 0
    for end_idx in cum_counts:
        count = end_idx - start_idx
        keep = min(max_neighbors, count)
        mask[start_idx : start_idx + keep] = True
        start_idx = end_idx

    return (
        src_sorted[mask],
        dst_sorted[mask],
        shifts_sorted[mask],
    )


def get_neighborhood(
    positions: np.ndarray,
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    lattice: Optional[np.ndarray] = None,  # [3, 3]
    max_neighbors: Union[str, int] = "inf",
) -> Tuple[np.ndarray, np.ndarray]:

    pbc = tuple(bool(i) for i in pbc)

    if pbc is not None:
        if not (all(pbc) or not any(pbc)):
            raise ValueError(
                "pbc must be either all True or all False, "
                "for author not check for 1D and 2D Periodic systems' neighbor lists."
            )

    if pbc is None:
        pbc = None
        lattice = None

    if pbc == (False, False, False):
        pbc = None

        if lattice is None:
            lattice = None
        elif lattice is not None:
            if lattice.any() == np.zeros((3, 3)).any():
                pbc = None
                lattice = None
        else:
            pbc == (True, True, True)

    if lattice is None:
        pbc = None
        lattice = None

    if lattice is not None:
        if lattice.any() == np.zeros((3, 3)).any():
            pbc = None
            lattice = None

    if pbc == (True, True, True):
        if lattice is None:
            raise ValueError("Lattice is required for periodic boundary conditions.")
        if np.allclose(lattice, np.zeros((3, 3))):
            raise ValueError(
                "Lattice matrix is zero, which is invalid for periodic systems."
            )

    # pbc_x = pbc[0]
    # pbc_y = pbc[1]
    # pbc_z = pbc[2]
    # identity = np.identity(3, dtype=float)
    # max_positions = np.max(np.absolute(positions)) + 1
    # if not pbc_x:
    #     lattice[0, :] = max_positions * 5 * cutoff * identity[0, :]
    # if not pbc_y:
    #     lattice[1, :] = max_positions * 5 * cutoff * identity[1, :]
    # if not pbc_z:
    #     lattice[2, :] = max_positions * 5 * cutoff * identity[2, :]

    edges = neighbour_list(
        quantities="ijSd",
        pbc=pbc,
        cell=lattice,
        positions=positions,
        cutoff=cutoff,
    )
    source, target, shifts = filter_max_neighbors(*edges, max_neighbors=max_neighbors)

    real_self_loop = source == target
    real_self_loop &= np.all(shifts == 0, axis=1)
    keep_edge = ~real_self_loop

    source = source[keep_edge]
    target = target[keep_edge]

    edge_shifts = shifts[keep_edge]
    edge_index = np.stack((source, target))

    return edge_index, edge_shifts, pbc, lattice


def get_neighborhood_for_calculator(
    positions: np.ndarray,
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    lattice: Optional[np.ndarray] = None,  # [3, 3]
    max_neighbors: Union[str, int] = "inf",
) -> Tuple[np.ndarray, np.ndarray]:

    edges = neighbour_list(
        quantities="ijSd",
        pbc=pbc,
        cell=lattice,
        positions=positions,
        cutoff=cutoff,
    )
    source, target, shifts = filter_max_neighbors(*edges, max_neighbors=max_neighbors)

    real_self_loop = source == target
    real_self_loop &= np.all(shifts == 0, axis=1)
    keep_edge = ~real_self_loop

    source = source[keep_edge]
    target = target[keep_edge]

    edge_shifts = shifts[keep_edge]
    edge_index = np.stack((source, target))

    return edge_index, edge_shifts
