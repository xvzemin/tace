################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Tuple, List
from collections import Counter


from tqdm import tqdm
import torch


from ._wigner import wigner_3j


def ICTD(
        n_total: int, 
        w: int = -1, # if not -1, return first rank = n_total, weight = w
        decomposition: bool = True, 
        dtype=None, 
        device=None
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


def equivariant_basis_generation(n_total : int) -> List[torch.Tensor]:
    '''
    Endomorphism, Linear operation in one gct
    '''
    n_now = 0
    j_now = 0
    path_list = []
    this_path = []
    this_pathmatrix = wigner_3j(0,0,0)
    pathmatrices_list = []
    # generate paths and path matrices
    def paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total):
        if n_now<=n_total:
            this_path.append(j_now)
            for j in [j_now + 1, j_now, j_now - 1]:
                if not (j_now==0 and (j!=1) ) and n_now+1<=n_total:
                    cgmatrix = wigner_3j(1,j_now,j)
                    this_pathmatrix_ = torch.einsum("abc,dce->dabe",this_pathmatrix,cgmatrix)
                    this_pathmatrix_ = this_pathmatrix_.reshape(cgmatrix.shape[0],-1,cgmatrix.shape[-1])
                    paths_generate(n_now+1,j,this_path.copy(),this_pathmatrix_,n_total)
            if n_now == n_total:
                this_pathmatrix = this_pathmatrix.reshape(-1,this_pathmatrix.shape[-1])
                this_pathmatrix = this_pathmatrix*(1./(this_pathmatrix**2).sum(0)[0]**(0.5)) # normalize
                pathmatrices_list.append(this_pathmatrix)
                path_list.append(this_path)
        return     
    paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total)
    equiv_basis_matrix = torch.zeros([3**n_total,3**n_total])
    param_list = []
    for ind, path in tqdm(enumerate(path_list),total=len(path_list)):
        print(path[-1])
        for ind2, path2 in enumerate(path_list):
            if path[-1] == path2[-1] and ind !=ind2:
                parameter = torch.nn.Parameter(torch.rand(1))
                equiv_basis_matrix+=(pathmatrices_list[ind] @ pathmatrices_list[ind2].T)*parameter
                param_list.append(parameter)
    return equiv_basis_matrix, param_list # [equiv_basis_matrix, list of trainable weights]


def change_of_basis_generation(n_total):
    '''Endomorphism, equivariant_basis_generation的算法4版本，线性变换的实现略微不太一样，但是始终都在一个ICT内部.'''
    n_now = 0
    j_now = 0
    path_list = []
    this_path = []
    this_pathmatrix = wigner_3j(0,0,0)
    pathmatrices_list = []
    # generate paths
    def paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total):
        if n_now<=n_total:
            this_path.append(j_now)
            for j in [j_now + 1, j_now, j_now - 1]:
                if not j > n_now+1 and not (j_now==0 and (j!=1) ) and n_now+1<=n_total:
                    wigner3j = wigner_3j(1,j_now,j)
                    this_pathmatrix_ = torch.einsum("abc,dce->dabe",this_pathmatrix,wigner3j)
                    this_pathmatrix_ = this_pathmatrix_.reshape(wigner3j.shape[0],-1,wigner3j.shape[-1])
                    paths_generate(n_now+1,j,this_path.copy(),this_pathmatrix_,n_total)
            if n_now == n_total:
                this_pathmatrix = this_pathmatrix.reshape(-1,this_pathmatrix.shape[-1])
                this_pathmatrix = this_pathmatrix*(1./(this_pathmatrix**2).sum(0)[0]**(0.5)) # normalize
                pathmatrices_list.append(this_pathmatrix)
                path_list.append(this_path)
        return     
    paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total)
    # sort 
    def get_last_digit(s):
        return int(s[-1])  
    sorted_indices = sorted(range(len(path_list)), key=lambda i: get_last_digit(path_list[i]))
    sorted_path_list = [path_list[i][-1]*2+1 for i in sorted_indices] # 2l+1
    sorted_pathmatrices_list = [pathmatrices_list[i] for i in sorted_indices]
    CT_matrix = torch.concat(sorted_pathmatrices_list,-1)
    fc_list = torch.nn.ModuleList()
    for _, v in Counter(sorted_path_list).items(): # k, v = l, p
        print(_)
        print(v)
        fc_list.append(torch.nn.Linear(v,v,bias=False))
    return sorted_path_list, fc_list, CT_matrix



def general_equivariant_basis_generator(space_1: Tuple, space_2: Tuple) \
    -> Tuple[List[Tuple] ,List[torch.Tensor]]:
    '''Homomorphism'''
    path_list_composition = [] 
    path_list_decomposition = []
    def general_scheme(rank_list, this_path, current_rank, ind, path_list, parity, index):
        rank = rank_list[ind]
        for i in range(abs(current_rank - rank), current_rank + rank + 1):
            this_path_ = this_path.copy()
            this_path_.append([rank,i])
            if ind+1 >= len(rank_list):
                path_list.append({"path": this_path_, "parity": parity, "index": index})
            else:
                general_scheme(rank_list, this_path_, i, ind+1, path_list, parity, index)
    for n,i in enumerate(space_1["space"]):
        general_scheme(i, [[0,0]], 0, 0, path_list_decomposition, space_1["parity"], n)
    for n,i in enumerate(space_2["space"]):
        general_scheme(i, [[0,0]], 0, 0, path_list_composition, space_1["parity"], n)
    def path_matrices_generators(path):
        path_matrix = wigner_3j(0,0,0)
        current_j = 0
        for (bridge_j, next_j) in path:
            cg_matrix = wigner_3j(bridge_j,current_j,next_j)
            path_matrix = torch.einsum("abc,dce->dabe", path_matrix, cg_matrix)\
                .reshape(2*bridge_j+1, -1, 2*next_j+1)
            current_j = next_j
        path_matrix = path_matrix.reshape(-1,path_matrix.shape[-1])
        path_matrix = path_matrix*(1./(path_matrix**2).sum(0)[0]**(0.5)) # normalize
        return path_matrix
    equivariant_basis = []
    paths_and_spaces = []
    for path_c in path_list_composition:
        for path_d in path_list_decomposition:
            if path_d["path"][-1][-1] == path_c["path"][-1][-1] and \
                path_d["parity"] == path_c["parity"]:
                    paths_and_spaces.append({"path_c": path_c, "path_d": path_d})
                    path_matrix_c = path_matrices_generators(path_c["path"])
                    path_matrix_d = path_matrices_generators(path_d["path"])
                    equivariant_basis.append(path_matrix_c @ path_matrix_d.T)
    return paths_and_spaces, equivariant_basis