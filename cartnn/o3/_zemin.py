################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
'''
_cartesian_nj is the Cartesian version of _wigner_nj, 
based on _wigner_nj author Ilyes Batatia and e3nn by Mario Geiger.
'''
from pathlib import Path
import collections
from typing import List, Union


import torch

from e3nn.o3 import wigner_3j
from cartnn.util import explicit_default_types
from ._ictd import ICTD
from ._irreps import Irrep, Irreps


CARTNN_CACHE_DIR = Path.home() / ".cache" / "cartnn"


def _cartesian_3j(l1: int, l2: int, l3: int) -> torch.Tensor:     
    with torch.no_grad():
        P1S, D1S, C1S, S1S = ICTD(l1+l2, l3, decomposition=False)
        P2S, D2S, C2S, S2S = ICTD(l3, l3, decomposition=False)
        Z = C1S[-1] @ S2S[0]
        del P1S, D1S, C1S, S1S, P2S, D2S, C2S, S2S
        Z = Z.view(3**l1, 3**l2, 3**l3)
    return Z


def cartesian_3j(l1: int, l2: int, l3: int, dtype=None, device=None) -> torch.Tensor:
    '''
    In practical atomistic machine learning models, very high-order Cartesian tensors 
    are typically not required. However, if one needs to compute cartesian_nj, caching 
    partial results becomes necessary.
    '''
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert isinstance(l1, int) and isinstance(l2, int) and isinstance(l3, int)

    # === cache directory ===
    cache_dir = CARTNN_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{l1}_{l2}_{l3}.pt"
    path = cache_dir / filename

    dtype, device = explicit_default_types(dtype, device)

    # === try to load === 
    Z = None
    if path.exists():
        try:
            Z = torch.load(path, weights_only=False)
        except Exception as e:
            print(f"[cartnn] Warning: Failed to load cache {path}: {e}")
            Z = None

    # === fallback: compute manually ===
    if Z is None:
        Z = _cartesian_3j(l1, l2, l3)
        if not path.exists():
            try:
                torch.save(Z, path)
                file_size = path.stat().st_size / (1024 ** 3)  # GB
                if file_size > 5:
                    print(f"[cartnn] Cache {path} is {file_size:.2f} GB > 5GB, removing.")
                    path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[cartnn] Warning: Failed to save cache {path}: {e}")

    return Z.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)


def _cartesian_nj(
    irrepss: List[Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.cdim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.cdim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.cdim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _cartesian_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = cartesian_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.cdim**0.5
                if normalization == "norm":
                    C *= ir_left.cdim**0.5 * ir.cdim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.cdim, *(irreps.cdim for irreps in irrepss_left), ir.cdim
                )
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.cdim,
                        *(irreps.cdim for irreps in irrepss_left),
                        irreps_right.cdim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.cdim, i + (u + 1) * ir.cdim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.cdim
    return sorted(ret, key=lambda x: x[0])


def _wigner_nj(
    irrepss: List[Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.sdim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.sdim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.sdim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.sdim**0.5
                if normalization == "norm":
                    C *= ir_left.sdim**0.5 * ir.sdim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.sdim, *(irreps.sdim for irreps in irrepss_left), ir.sdim
                )
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.sdim,
                        *(irreps.sdim for irreps in irrepss_left),
                        irreps_right.sdim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.sdim, i + (u + 1) * ir.sdim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.sdim
    return sorted(ret, key=lambda x: x[0])

_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")


def U_matrix_real(
    irreps_in: Union[str, Irreps],
    irreps_out: Union[str, Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irreps_out = Irreps(irreps_out)
    irrepss = [Irreps(irreps_in)] * correlation

    if correlation == 4:
        filter_ir_mid = [(i, 1 if i % 2 == 0 else -1) for i in range(12)]

    wigners = _cartesian_nj(irrepss, normalization, filter_ir_mid, dtype)
    
    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir

    try:
        out += [last_ir, stack]
    except:
        first_dim = irreps_out.dim
        if first_dim != 1:
            size = [first_dim] + [Irreps(irreps_in).dim] * correlation + [1]
        else:
            size = [Irreps(irreps_in).dim] * correlation + [1]
        out = [str(irreps_out)[:-2], torch.zeros(size, dtype=dtype)]
        
    return out

