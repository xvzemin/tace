from typing import NamedTuple, Callable, Any, List, Tuple, Union
import functools
import bisect


import numpy as np
import sympy
from sympy import MutableDenseNDimArray, Matrix
from sympy.physics.wigner import clebsch_gordan
from sympy import S
import torch
from torch import tensor, Tensor


from ..irreps import Irreps, has_path, irrep_segments


def expand_left(source: torch.Tensor, target:torch.Tensor, dim:int):
    if dim < 0:
        dim = source.ndim + dim
    target = target.view([1]*dim+[-1])
    return target


def extract_batch_segments(keys: List[List[int]]):
    r"""
    Process sorted integer key lists to generate batch indices, boundary pointers, and key values.

    Parameters
    ----------
    keys : List[List[int]]
        A list of sorted integer key lists. All lists must have the same length.

    Returns
    -------
    batch : List[int]
        A list where each element indicates the batch index it belongs to.
    seg : List[int]
        A list of boundary pointers indicating the start and end of each batch.
    val : List[List[int]]
        A list of lists containing the key values at the boundary points for each key list.

    Notes
    -----
    - The input key lists must be sorted in ascending order.
    - If the input is empty, the function returns empty lists for `batch`, `seg`, and `val`.

    Examples
    --------
    >>> keys = [
    ...     [1, 1, 2, 2],
    ...     [1, 1, 2, 2]
    ... ]
    >>> extract_batch_seg_native(keys)
    ([0, 0, 1, 1], [0, 2, 4], [[1, 2], [1, 2]])

    >>> keys = [
    ...     [5, 5, 5],
    ...     [5, 5, 5]
    ... ]
    >>> extract_batch_seg_native(keys)
    ([0, 0, 0], [0, 3], [[5], [5]])

    >>> keys = [
    ...     [1, 1, 2, 3, 3],
    ...     [1, 2, 2, 3, 3]
    ... ]
    >>> extract_batch_seg_native(keys)
    ([0, 1, 2, 3, 3], [0, 1, 2, 3, 5], [[1, 1, 2, 3], [1, 2, 2, 3]])
    """
    if not keys or not keys[0]:
        return [], [], []
    
    length = len(keys[0])
    seg = [0]  # 初始化分界指针
    
    # 生成分界指针
    for i in range(length):
        last_idx = seg[-1]
        # 检查所有键在当前索引i处是否与上一个分界点的值不同
        if any(key[i] != key[last_idx] for key in keys):
            seg.append(i)
    
    seg.append(length)  # 添加最终边界
    
    # 生成批次索引
    batch = [0] * length
    for batch_idx in range(1, len(seg)):
        start = seg[batch_idx-1]
        end = seg[batch_idx]
        for i in range(start, end):
            batch[i] = batch_idx - 1
    
    # 提取分界点键值
    val = [
        [key[boundary] for boundary in seg[:-1]]  # 排除最后一个边界
        for key in keys
    ]
    
    return batch, seg, val


def sort_by_column_key(to_sort: List[List[Any]], key: List[List[Any]] = None) -> List[List[Any]]:
    """
    Sort the columns of the first 2D list based on the column-wise lexicographical order of the key 2D list.

    Parameters
    ----------
    to_sort : List[List[Any]]
        The first 2D list whose columns are to be sorted.
    key : List[List[Any]]
        The key 2D list used to determine the sorting order of columns.

    Returns
    -------
    List[List[Any]]
        The first 2D list with columns sorted according to the column-wise lexicographical order of the key.

    Raises
    ------
    ValueError
        If either `to_sort` or `key` is empty, or if their lengths do not match.

    Examples
    --------
    >>> to_sort = [[1, 2, 3], 
    ...            [4, 5, 6]]
    >>> key = [[2, 1, 3], 
    ...        [1, 3, 2]]
    >>> sort_by_column_key(to_sort, key)
    [[2, 1, 3], 
     [5, 4, 6]]
    """

    if key is None:
        key = to_sort

    # 将两个列表转置为列优先的形式
    to_sort_transposed = list(zip(*to_sort))
    key_transposed = list(zip(*key))
    # 将转置后的列表组合成 [(to_sort_col, key_col)] 的形式
    combined = list(zip(to_sort_transposed, key_transposed))
    # 根据 key 的列字典序进行排序
    sorted_combined = sorted(combined, key=lambda x: x[1])
    # 提取排序后的 to_sort 的列
    sorted_to_sort_transposed = [item[0] for item in sorted_combined]
    # 将转置后的结果还原为原始形式
    sorted_to_sort = list(zip(*sorted_to_sort_transposed))
    # 将元组转换为列表
    sorted_to_sort = [list(row) for row in sorted_to_sort]

    return sorted_to_sort


def extract_scatter_indices(keys: List[List[int]]) -> Tuple[List[int], List[List[int]]]:
    """
    Process integer key lists to generate scatter indices and sorted unique keys.

    Parameters
    ----------
    keys : List[List[int]]
        A list of integer key lists. All lists must have the same length.

    Returns
    -------
    indices : List[int]
        A list where each element is the index of the corresponding key tuple in the sorted unique list.
    scatter_keys : List[List[int]]
        A list of lists containing the sorted unique key values for each original key list.

    Notes
    -----
    - If the input is empty, the function returns empty lists for `indices` and `scatter_keys`.

    Examples
    --------
    >>> keys = [
    ...     [1, 1, 2, 2],
    ...     [1, 1, 2, 2]
    ... ]
    >>> extract_scatter_indices(keys)
    ([0, 0, 1, 1], [[1, 2], [1, 2]])

    >>> keys = [
    ...     [5, 5, 5],
    ...     [5, 5, 5]
    ... ]
    >>> extract_scatter_indices(keys)
    ([0, 0, 0], [[5], [5]])

    >>> keys = [
    ...     [1, 1, 2, 3, 3],
    ...     [1, 2, 2, 3, 3]
    ... ]
    >>> extract_scatter_indices(keys)
    ([0, 1, 2, 3, 3], [[1, 1, 2, 3], [1, 2, 2, 3]])
    """
    if not keys or not keys[0]:
        return [], []
    
    # 将每个位置的键打包成元组
    tuple_list = list(zip(*keys))
    
    # 排序并去重得到唯一的元组列表
    scatter = sorted(set(tuple_list))
    
    # 生成每个原始元组的索引
    indices = []
    for t in tuple_list:
        idx = bisect.bisect_left(scatter, t)
        indices.append(idx)
    
    # 解压唯一元组列表为各键列表
    if not scatter:
        scatter_keys = []
    else:
        scatter_keys = [list(sk) for sk in zip(*scatter)]
    
    return indices, scatter_keys


def add_operation_methods(cls):
    # """类装饰器：为 NamedTuple 动态添加 to/cuda/cpu 等方法"""
    def _apply(self, func: Callable[[Any], Any]):
        processed = []
        for field in self._fields:
            value = getattr(self, field)
            # 递归处理 Tensor 或同类型实例
            if isinstance(value, Tensor):
                processed.append(func(value))
            elif isinstance(value, self.__class__):
                processed.append(func(value))  # 递归处理同类型字段
            elif hasattr(value, '_apply'):
                processed.append(value._apply(func))
            else:
                processed.append(value)
        return self.__class__(*processed)
    
    def to(self, *args, **kwargs):
        # 解析参数
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        
        # 定义转换函数
        def convert(t):
            if isinstance(t, Tensor):
                # 只对浮点张量应用dtype转换
                target_dtype = dtype if (dtype is not None and t.is_floating_point()) else None
                return t.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
            return t
        
        return self._apply(convert)

    def cuda(self, *args, **kwargs):
        return self._apply(lambda x: x.cuda(*args, **kwargs))

    def cpu(self, *args, **kwargs):
        return self._apply(lambda x: x.cpu(*args, **kwargs))

    cls._apply = _apply
    cls.to = to
    cls.cuda = cuda
    cls.cpu = cpu
    return cls


@add_operation_methods
class SparseScaleInfo(NamedTuple):
    '''
        z_M = sum_{t in Ind*[M]} s_t * x_Ind'[t]

        or

        z_M = sum_{M'} s_{MM'} x_M'
    '''
    scale: Union[torch.Tensor, None] = None # (num_t,), floating
    index: Union[torch.Tensor, None] = None # (num_t,), int in [0, num_M')
    seg_out: Union[torch.Tensor, None] = None # (num_M_nonzero+1,), increasing int in [0, num_t]
    index_out: Union[torch.Tensor, None] = None # (num_M_nonzero,), int in [0, num_M)
    out_size: Union[int, None] = None # num_M


@add_operation_methods
class SparseProductInfo(NamedTuple):
    '''
        z_M = sum_{t in Ind*[M]} s_t * x_Ind1[t] * y_Ind2[t]

        or

        z_M = sum_{M1M2} s_{MM1M2} x_M1 * y_M2
    '''
    scale: Union[torch.Tensor, None] = None # (num_t,), floating
    index1: Union[torch.Tensor, None] = None # (num_t,), int in [0, num_M1)
    index2: Union[torch.Tensor, None] = None # (num_t,), int in [0, num_M2)
    seg_out: Union[torch.Tensor, None] = None # (num_M_nonzero+1,), increasing int in [0, num_t]
    segment_index: Union[torch.Tensor, None] = None # (num_t,), int in [0, num_M)
    gather_index: Union[torch.Tensor, None] = None # (num_M_nonzero,) int in [0, num_t)
    index_out: Union[torch.Tensor, None] = None # (num_M_nonzero,), int in [0, num_M)
    out_size: Union[int, None] = None # num_M


@add_operation_methods
class TensorProductInfo(NamedTuple):
    info_Mij_fwd: SparseProductInfo
    info_Mij_bwd1: SparseProductInfo
    info_Mij_bwd2: SparseProductInfo
    
    info_M_fwd: SparseProductInfo
    info_M_bwd1: SparseProductInfo
    info_M_bwd2: SparseProductInfo
    
    info_kM1j_fwd: SparseProductInfo
    info_kM1j_bwd1: SparseProductInfo
    info_kM1j_bwd2: SparseProductInfo
    info_kM1M2_fwd: SparseProductInfo
    info_kM1M2_bwd1: SparseProductInfo
    info_kM1M2_bwd2: SparseProductInfo
    info_M_kM1M2_fwd: SparseScaleInfo
    info_M_kM1M2_bwd: SparseScaleInfo
    out_size: int


@add_operation_methods
class IrrepsInfo(NamedTuple):
    rsqrt_dims: Tensor 
    rdims: Tensor 
    irrep_index: Tensor
    irrep_seg: Tensor
    num_irreps: int


@add_operation_methods
class IrrepsLinearInfo(NamedTuple):

    scale_MM0: Tensor
    M_seg_MM0: Tensor
    ii0_MM0: Tensor
    M0_MM0: Tensor
    M_MM0: Tensor
    M_out: Tensor

    # for grad on weight
    ii0_seg_ii0MM0: Tensor
    M_ii0MM0: Tensor
    M0_ii0MM0: Tensor
    scales_ii0: Tensor
    
    out_size: int


@functools.lru_cache(maxsize=None)
def j_matrix(l: int):
    r"""Computes the Wigner D-matrix for the rotation that exchanges y-z axes and reverses x-axis.
    
    This function calculates the Wigner D-matrix corresponding to a specific rotation
    transformation that exchanges the y and z axes while reversing the direction of the x-axis.
    It is used in general Wigner D-matrix calculations.
    
    Args:
        l: The angular momentum quantum number.
        
    Returns:
        sympy.Matrix: The Wigner D-matrix for the specified rotation transformation.
    """
    if l == 0:
        return Matrix([[1]])
    if l == 1:
        return Matrix(
            [[ 0,  1,  0],
             [ 1,  0,  0],
             [ 0,  0, -1]])
    else:
        cg = so3_clebsch_gordan(l,1,l-1)
        j = j_matrix(l-1)
        return np.einsum('iI,jJ,kij,KIJ->kK',j_matrix(1), j, cg, cg)
    

# adapted and modified from e3nn
@functools.lru_cache(maxsize=None)
def so3_clebsch_gordan(l, l1, l2):
    r"""Computes the Clebsch-Gordan coefficients for SO(3) group.
    
    This function calculates the Clebsch-Gordan coefficients for the SO(3) group,
    which are used to decompose the tensor product of two irreducible representations
    into a direct sum of irreducible representations.
    
    Args:
        l: An integer representing the angular momentum quantum number of the resulting representation.
        l1: An integer representing the angular momentum quantum number of the first representation.
        l2: An integer representing the angular momentum quantum number of the second representation.
        
    Returns:
        sympy.Array: The Clebsch-Gordan coefficients matrix for the specified angular momenta.
    """
    Q1 = _change_basis_real_to_complex(l1)
    Q2 = _change_basis_real_to_complex(l2)
    QT = sympy.conjugate(_change_basis_real_to_complex(l)).transpose()
    C = _su2_clebsch_gordan(l, l1, l2)
    C = np.einsum("mn,nik,ij,kl->mjl", QT, C, Q1, Q2)
    return sympy.sympify(C)


@functools.lru_cache(maxsize=None)
def _change_basis_real_to_complex(l: int):
    sqrt2 = sympy.sqrt(2)
    q = MutableDenseNDimArray.zeros(2 * l + 1, 2 * l + 1)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = sqrt2 / 2
        q[l + m, l - abs(m)] = -sqrt2 * sympy.I / 2
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** S(m) * sqrt2 / 2
        q[l + m, l - abs(m)] = sympy.I * (-1) ** S(m) * sqrt2 / 2
    q = (-sympy.I) ** S(l) * q  # Added factor of sympy.I**l to make the Clebsch-Gordan coefficients real
    return q


@functools.lru_cache(maxsize=None)
def _su2_clebsch_gordan(j3: Union[int, float], j1: Union[int, float], j2: Union[int, float]):
    r"""Calculates the Clebsch-Gordon matrix
    for SU(2) coupling j1 and j2 to give j3.
    Parameters
    ----------
    j3 : float
        Total angular momentum 3.
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    Returns
    -------
    cg_matrix : MutableDenseNDimArray
        Requested Clebsch-Gordan matrix.
    """
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = MutableDenseNDimArray.zeros(int(2 * j3 + 1), int(2 * j1 + 1), int(2 * j2 + 1))
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for twice_m1 in (x for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for twice_m2 in (x for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(twice_m1 + twice_m2) <= 2*j3:
                    mat[j3 + int(twice_m1/2 + twice_m2/2), int(j1 + twice_m1/2), int(j2 + twice_m2/2)] = clebsch_gordan(
                        j1, j2, j3, S(twice_m1)/2, S(twice_m2)/2, S(twice_m1+twice_m2)/2
                    )
    return mat

# The function _su2_clebsch_gordan is modified from
# QuTiP: Quantum Toolbox in Python.
# Key modifications are: (1) set the j3 to the first axis
#                        (2) use MutableDenseNDimArray instead of numpy array
#                        (3) use sympy.physics.wigner.clebsch_gordan
#                            instead of qutip.utilities._su2_clebsch_gordan_coeff
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################



def sparse_scale_info(index=None, index_out=None, scale=None, out_size=None):
    assert index is not None or index_out is not None, "At least one index should be not None"
    inter_size = len(index or index_out)
    index = index or list(range(inter_size))
    index_out = index_out or list(range(inter_size))
    if scale is not None:
        index_out_MM1, index_MM1, scale_MM1 = sort_by_column_key(
            [index_out, index, scale]
        )
        batch_M_MM1, seg_M_MM1, (index_out_M,) = extract_batch_segments(
            [index_out_MM1]
        )
    else:
        index_out_MM1, index_MM1 = sort_by_column_key(
            [index_out, index]
        )
        batch_M_MM1, seg_M_MM1, (index_out_M,) = extract_batch_segments(
            [index_out_MM1]
        )
        scale_MM1 = None

    
    if out_size is None:
        out_size = len(index_out_M)

    seg_new = [0]
    out_count = 0
    for out_M in range(out_size):
        if out_count < len(index_out_M) and index_out_M[out_count] == out_M:
            seg_new.append(seg_M_MM1[out_count+1])
            out_count+=1
        else:
            seg_new.append(seg_new[-1])

    index_out_M = None
    if index_MM1 == list(range(inter_size)):
        index_MM1 = None

    if seg_new == list(range(out_size+1)):#len(batch_M_MM1)+1:
        seg_new = None
    # if index_out_M == list(range(out_size)):
    #     index_out_M = None

    return SparseScaleInfo(
        tensor(scale_MM1) if scale_MM1 is not None else None, 
        tensor(index_MM1) if index_MM1 is not None else None, 
        # tensor(seg_M_MM1) if seg_M_MM1 is not None else None, 
        tensor(seg_new) if seg_new is not None else None, 
        tensor(index_out_M) if index_out_M is not None else None,
        out_size)

def sparse_scale_infos(index=None, index_out=None, scale=None, out_size=None, in_size=None):
    return (
        sparse_scale_info(index, index_out, scale, out_size),
        sparse_scale_info(index_out, index, scale, in_size)
    )

def sparse_product_info(index1=None, index2=None, index=None, scale=None, out_size=None):

    assert index1 is not None or index2 is not None or index is not None, "At least one of the indices should be not None"
    # assert index is None or index_out is None, "index and index_out cannot be both not None " 

    inter_size = len(index1 or index2 or index)
    index1 = index1 or list(range(inter_size))
    index2 = index2 or list(range(inter_size))
    index = index or list(range(inter_size))
    if scale is not None:
        index_MM1M2, index1_MM1M2, index2_MM1M2, scale_MM1M2 = sort_by_column_key(
            [index, index1, index2, scale])
        batch_M_MM1M2, seg_M_MM1M2, (index_M,) = extract_batch_segments(
            [index_MM1M2]
        )
    else:
        index_MM1M2, index1_MM1M2, index2_MM1M2 = sort_by_column_key(
            [index, index1, index2])
        batch_M_MM1M2, seg_M_MM1M2, (index_M,) = extract_batch_segments(
            [index_MM1M2]
        )
        scale_MM1M2 = None

    if out_size is None:
        out_size = len(index_M)

    if index_M[0] != 0:
        seg_M_MM1M2 = [0] * index_M[0] + seg_M_MM1M2
    if index_M[-1] != out_size:
        seg_M_MM1M2 = seg_M_MM1M2 + [inter_size] * (out_size-index_M[-1]-1)


    if index1_MM1M2 == list(range(inter_size)):
        index1_MM1M2 = None
    if index2_MM1M2 == list(range(inter_size)):
        index2_MM1M2 = None

    seg_new = [0]
    out_count = 0
    for out_M in range(out_size):
        if out_count < len(index_M) and index_M[out_count] == out_M:
            seg_new.append(seg_M_MM1M2[out_count+1])
            out_count+=1
        else:
            seg_new.append(seg_new[-1])

    if seg_new == list(range(out_size+1)):
        seg_new = None
    # if len(seg_M_MM1M2) == len(batch_M_MM1M2)+1:
    #     seg_M_MM1M2 = None
    # if index_M == list(range(out_size)):
    index_M = None
            
    return SparseProductInfo(
        scale=tensor(scale_MM1M2) if scale_MM1M2 is not None else None,
        index1=tensor(index1_MM1M2) if index1_MM1M2 is not None else None,
        index2=tensor(index2_MM1M2) if index2_MM1M2 is not None else None,
        seg_out=tensor(seg_new) if seg_new is not None else None,
        segment_index=tensor(index_MM1M2) if seg_new is not None else None,
        # seg_out=tensor(seg_M_MM1M2) if seg_M_MM1M2 is not None else None,
        index_out=tensor(index_M) if index_M is not None else None,
        out_size=out_size
    )

def sparse_product_infos(index1=None, index2=None, index=None, scale=None, out_size=None, in1_size=None, in2_size=None):
    return (
        sparse_product_info(index1,index2,index,scale,out_size),
        sparse_product_info(index2,index,index1,scale,in1_size),
        sparse_product_info(index,index1,index2,scale,in2_size),
    )

def generate_fully_connected_tp_paths(irreps_out: Irreps, 
                           irreps1: Irreps, irreps2: Irreps):

    paths = []
    for (k,ir_out) in enumerate(irreps_out):
        for (i,ir1) in enumerate(irreps1):
            for (j, ir2) in enumerate(irreps2):
                if has_path(ir_out, ir1, ir2): 
                    paths.append((k,i,j))

    return paths


def prepare_so3(
        irreps_out: Irreps,
        irreps1: Irreps, irreps2: Irreps,
        path: List[Tuple[int, int, int]]=None,
        path_norm: bool = True,
        channel_norm: bool = False,
        channel_scale: float = 1.0):
    
    if not path:
        path = generate_fully_connected_tp_paths(irreps_out, irreps1, irreps2)
    
    seg_out = irrep_segments(irreps_out)
    seg1 = irrep_segments(irreps1)
    seg2 = irrep_segments(irreps2)

    current_k = None
    path_count = {}
    for w_idx, (k, i, j) in enumerate(path):
        if k == current_k:
            path_count[k] += 1
        else:
            path_count[k] = 1
            current_k = k

    cg_vals = []
    Ms = []
    M1s = []
    M2s = []
    k_s = []
    i_s = []
    j_s = []
    w_idcs = []

    for w_idx, (k,i,j) in enumerate(path):
        l, l1, l2 = (
            irreps_out[k].l, irreps1[i].l, irreps2[j].l
        )
        cg = so3_clebsch_gordan(l, l1, l2)
        for km in range(2 * l + 1):
            # Apply path normalization and channel normalization
            norm_scale = (path_count[k] ** (-0.5) if path_norm else 1.0) * \
                         (channel_scale if channel_norm else 1.0)
            for im in range(2 * l1 + 1):
                for jm in range(2 * l2 + 1):
                    cg_val = float(cg[km, im, jm]) * norm_scale
                    if cg_val != 0.0:
                        cg_vals.append(cg_val)
                        Ms.append(seg_out[k] + km)
                        M1s.append(seg1[i]+im)
                        M2s.append(seg2[j]+jm)
                        k_s.append(k)
                        i_s.append(i)
                        j_s.append(j)
                        w_idcs.append(w_idx)
    return cg_vals, Ms, M1s, M2s, k_s, i_s, j_s, w_idcs, len(path)




def create_tp_info(cg_vals, M, M1, M2, k, i, j, w_idx, out_size, in1_size, in2_size):
    k_MM1M2, M_MM1M2, i_MM1M2, j_MM1M2, M1_MM1M2, M2_MM1M2, w_idx_MM1M2, cg_vals = sort_by_column_key(
        [k, M, i, j, M1, M2, w_idx, cg_vals])
    
    # first xy 
    Mij_MM1M2, Mij_seg_MM1M2, (k_Mij, M_Mij, i_Mij, j_Mij, w_idx_Mij)  = extract_batch_segments(
        [k_MM1M2, M_MM1M2, i_MM1M2, j_MM1M2, w_idx_MM1M2]
    )
    infos_inter = sparse_product_infos(M1_MM1M2, M2_MM1M2, Mij_MM1M2, cg_vals, in1_size=in1_size, in2_size=in2_size)
    M_batch_Mij, M_seg_Mij, (M_out,) = extract_batch_segments(
        [M_Mij]
    )
    infos_M = sparse_product_infos(index2=w_idx_Mij, index=M_batch_Mij,out_size=out_size)

    # first W
    kijM1M2_batch_MijM1M2, (k_kijM1M2, i_kijM1M2, j_kijM1M2, M1_kijM1M2, M2_kijM1M2, w_idx_kijM1M2) = extract_scatter_indices(
        [k_MM1M2, i_MM1M2, j_MM1M2, M1_MM1M2, M2_MM1M2, w_idx_MM1M2]
    ) 
    kijM1_batch_kM1M2, (k_kijM1, i_kijM1, M1_kijM1, j_kijM1, w_idx_kijM1) = extract_scatter_indices(
        [k_kijM1M2, i_kijM1M2, M1_kijM1M2, j_kijM1M2, w_idx_kijM1M2]
    )

    M_batch_MM1M2, M_seg_MM1M2, (M_out,) = extract_batch_segments(
        [M_MM1M2]
    )


    infos_kM1j_M1 = sparse_product_infos(index1=M1_kijM1, index2=w_idx_kijM1, in1_size=in1_size)
    infos_kM1M2_kM1j = sparse_product_infos(index1=kijM1_batch_kM1M2, index2=M2_kijM1M2, in2_size=in2_size)
    infos_M_kM1M2 = sparse_scale_infos(index = kijM1M2_batch_MijM1M2, index_out=[M_out[M] for M in M_batch_MM1M2], 
                                       scale=cg_vals, out_size=out_size)

    tp_info = TensorProductInfo(
        *infos_inter,
        *infos_M,
        *infos_kM1j_M1,
        *infos_kM1M2_kM1j,
        *infos_M_kM1M2,
        out_size=out_size
    )
    return tp_info


def tp_info(
        irreps_out: Irreps,
        irreps1: Irreps, irreps2: Irreps,
        path: List[Tuple[int, int, int]]=None,
        path_norm: bool = True,
        channel_norm: bool = False,
        channel_scale: float = 1.0):
    cg_vals, M, M1, M2, k, i, j, w_idx, num_paths = prepare_so3(
        irreps_out, irreps1, irreps2, path, path_norm, channel_norm, channel_scale)

    tp_info = create_tp_info(cg_vals, M, M1, M2, k, i, j, w_idx, irreps_out.dim)
    return tp_info, num_paths


def tp_infos(
        irreps_out: Irreps,
        irreps1: Irreps, irreps2: Irreps,
        path: List[Tuple[int, int, int]]=None,
        path_norm: bool = True,
        channel_norm: bool = False,
        channel_scale: float = 1.0):
    # Note: channel_norm and channel_scale are only applied to the forward pass CG coefficients
    # Backward passes use the original (forward) coefficients
    cg_vals, M, M1, M2, k, i, j, w_idx, num_paths = prepare_so3(
        irreps_out, irreps1, irreps2, path, path_norm, channel_norm, channel_scale)

    tp_forward = create_tp_info(cg_vals, M, M1, M2, k, i, j, w_idx, irreps_out.dim, irreps1.dim, irreps2.dim)
    tp_backward1 = create_tp_info(cg_vals, M1, M2, M, i, j, k, w_idx, irreps1.dim, irreps2.dim, irreps_out.dim)
    tp_backward2 = create_tp_info(cg_vals, M2, M, M1, j, k, i, w_idx, irreps2.dim, irreps_out.dim, irreps1.dim)
    return tp_forward, tp_backward1, tp_backward2, num_paths


