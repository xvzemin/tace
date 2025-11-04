################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
'''
Using ICT_decomposition Code from paper,
High-Rank Irreducible Cartesian Tensor Decomposition and Bases of Equivariant Spaces
'''
from typing import Tuple, List


import torch
from e3nn import o3


def ICT_decomposition(n_total: int) -> Tuple[List[List[int]], List[torch.Tensor]]:
    n_now = 0
    j_now = 0
    path_list = []
    this_path = []
    this_pathmatrix = o3.wigner_3j(0, 0, 0)
    pathmatrices_list = []

    # generate paths and path matrices
    def paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total):
        if n_now <= n_total:
            this_path.append(j_now)
            for j in [j_now + 1, j_now, j_now - 1]:
                if not (j_now == 0 and (j != 1)) and n_now + 1 <= n_total:
                    cgmatrix = o3.wigner_3j(1, j_now, j)
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
                pathmatrices_list.append(this_pathmatrix)
                path_list.append(this_path)
        return

    paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total)
    decomp_list = []
    for path_matrix in pathmatrices_list:
        decomp_list.append(path_matrix @ path_matrix.T)
    return path_list, decomp_list
