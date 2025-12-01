################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import List

import torch
from torch import Tensor
import opt_einsum_fx

from ..paths import return_tcs

class MatrixTC(torch.nn.Module):
    def __init__(self, exprs: List[str]) -> None:
        super().__init__()
        self.exprs = exprs
        self.tcs = return_tcs(exprs)
        self.meta = []
        self.ctrs = []
        for tc in self.tcs:
            axes_1 = [a + 2 for a in tc.axes_1]
            axes_2 = [a + 2 for a in tc.axes_2]
            r_1, r_2, r_o, k = tc.r_1, tc.r_2, tc.r_o, tc.k

            t1_feature_dims = list(range(2, r_1 + 2))
            t1_axes_rest = [i for i in t1_feature_dims if i not in axes_1]
            t1_perm = tuple([0, 1] + t1_axes_rest + axes_1)

            t2_feature_dims = list(range(2, r_2 + 2))
            t2_axes_rest = [i for i in t2_feature_dims if i not in axes_2]
            t2_perm = tuple([0, 1] + axes_2 + t2_axes_rest)

            normalizer = 1.0 / sqrt(3**k) 

            M = 3 ** (r_1 - k)
            K = 3 ** k
            N = 3 ** (r_2 - k)

            self.meta.append({
                't1_perm': t1_perm,
                't2_perm': t2_perm,
                'normalizer': normalizer,
                "r_1": r_1,
                'r_2': r_2,
                'r_o': r_o,
                'k': k, 
                'M': M,
                'N': N,
                'K': K,
            })

    def forward(self, T1: Tensor, T2: Tensor) -> List[Tensor]:
        results = []
        for meta in self.meta:
            # [B, K, ...] permute
            T1_perm = T1.permute(meta['t1_perm']).contiguous()
            T2_perm = T2.permute(meta['t2_perm']).contiguous()

            B, C1, C2 = T1_perm.shape[0], T1_perm.shape[1], T2_perm.shape[1]
            M, K, N , r_o= meta['M'], meta['K'], meta['N'], meta['r_o']

            T1_mat = T1_perm.view(B, C1, M, K) 
            T2_mat = T2_perm.view(B, C2, K, N) 
            T = torch.matmul(T1_mat, T2_mat)     

            results.append(T.reshape((B, C1) + (3,) * r_o) * meta['normalizer'])

        return results

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for tc in self.tcs:
            lines.append(f"  {tc}")
        lines.append(")")
        return "\n".join(lines)


_PATH = 4
BATCH = 5
CHANNEL = 6

class TargetMatrixTC(torch.nn.Module):
    def __init__(self, expr: str) -> None:
        super().__init__()
        tc = return_tcs([expr])[0]
        self.axes_1 = [a + 3 for a in tc.axes_1]
        self.axes_2 = [a + 2 for a in tc.axes_2]
        self.r_1 = tc.r_1
        self.r_2 = tc.r_2
        self.r_o = tc.r_o
        self.k = tc.k
        self.n_dim = self.r_1 + self.r_2 - self.k + 2

        ctr = torch.fx.symbolic_trace(
            lambda T1, T2: torch.einsum('pbcij, pbcjk -> pbcik', T1, T2)
        )
        self.ctr = opt_einsum_fx.optimize_einsums_full(
            model=ctr,
            example_inputs=(
                torch.randn(
                    [_PATH]
                    + [BATCH]
                    + [CHANNEL]
                    + [3**(self.r_1-self.k)]
                    + [3**self.k]
                ),
                torch.randn(
                    [_PATH]
                    + [BATCH]
                    + [CHANNEL]
                    + [3**self.k]
                    + [3**(self.r_2-self.k)]
                ),
            ),
        )

    def forward(self, T1: List[Tensor], T2: Tensor) -> Tensor:
        P = len(T1)
        B = T1[0].size(0)
        C = T1[0].size(1)
        REST = (3,) * self.r_o

        T1_mat = T1.reshape(P, B, C, 3**(self.r_1 - self.k), 3**self.k)
        T2_mat = T2.reshape(1, B, C, 3**self.k, 3**(self.r_2 - self.k))
        # T_mat = torch.matmul(T1_mat, T2_mat) 
        T_mat = self.ctr(T1_mat, T2_mat)
        T = T_mat.reshape(P, B, C, *REST)
    
        return T

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_1={self.r_1}, r_2={self.r_2}, r_o={self.r_o}, k={self.k})"
