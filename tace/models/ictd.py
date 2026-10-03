################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
see
https://www.jmlr.org/papers/v26/25-0134.html for ICTD
https://arxiv.org/abs/2509.14961 for TACE
https://arxiv.org/abs/2512.16882 for Cartesian-3j
"""

from typing import List, Tuple

import torch
from e3nn.o3 import wigner_3j


def ICTD(
    n_total: int,
    w: int = -1,  # if not -1, return first rank = n_total, weight = w
    decomposition: bool = True,
    dtype=torch.float64,
    device=None,
) -> Tuple[List[List[int]], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    n_now = 0
    j_now = 0
    path_list = []
    this_path = []
    this_pathmatrix = wigner_3j(0, 0, 0, dtype=dtype, device=device)
    pathmatrices_list = []
    cart2sph_list = []
    sph2cart_list = []
    stop_flag = {"stop": False}

    # generate paths and path matrices
    def paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total):
        if stop_flag["stop"]:
            return
        if n_now <= n_total:
            if stop_flag["stop"]:
                return
            this_path.append(j_now)
            for j in [j_now + 1, j_now, j_now - 1]:
                if not (j_now == 0 and (j != 1)) and n_now + 1 <= n_total:
                    cgmatrix = wigner_3j(1, j_now, j, dtype=dtype, device=device)
                    this_pathmatrix_ = torch.einsum(
                        "abc,dce->dabe", this_pathmatrix, cgmatrix
                    )
                    this_pathmatrix_ = this_pathmatrix_.reshape(
                        cgmatrix.shape[0], -1, cgmatrix.shape[-1]
                    )
                    paths_generate(
                        n_now + 1, j, this_path.copy(), this_pathmatrix_, n_total
                    )
            if n_now == n_total:
                this_pathmatrix = this_pathmatrix.reshape(-1, this_pathmatrix.shape[-1])
                this_pathmatrix = this_pathmatrix * (
                    1.0 / (this_pathmatrix**2).sum(0)[0] ** (0.5)
                )  # normalize
                if w == -1:
                    pathmatrices_list.append(this_pathmatrix)
                    path_list.append(this_path)
                else:
                    if path_list:
                        path_list[-1] = this_path
                        pathmatrices_list[-1] = this_pathmatrix
                    else:
                        pathmatrices_list.append(this_pathmatrix)
                        path_list.append(this_path)
                    if this_path[-1] == w:
                        stop_flag["stop"] = True
                        return
        return

    paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total)
    decomp_list = []
    for path_matrix in pathmatrices_list:
        if decomposition:
            decomp_list.append(path_matrix @ path_matrix.T)
        cart2sph_list.append(path_matrix)
        sph2cart_list.append(path_matrix.T)
    return path_list, decomp_list, cart2sph_list, sph2cart_list
