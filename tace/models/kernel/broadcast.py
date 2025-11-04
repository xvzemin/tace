################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import List

import torch
from torch import Tensor


from ..paths import return_tcs
from ..utils import expand_dims_to


class BroadcastTC(torch.nn.Module):
    def __init__(self, exprs: List[str]) -> None:
        super().__init__()
        self.exprs = exprs
        self.tcs = return_tcs(exprs)
        self.meta = []
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

            n_dim = r_1 + r_2 - k + 2
            sum_axis = tuple(range(r_1 - k + 2, r_1 + 2)) if k > 0 else ()

            normalizer = 1.0 / sqrt(3**k) 

            M = 3 ** (r_1 - k)
            K = 3 ** k
            N = 3 ** (r_2 - k)

            self.meta.append({
                't1_perm': t1_perm,
                't2_perm': t2_perm,
                'n_dim': n_dim,
                'sum_axis': sum_axis,
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
            T1_perm = T1.permute(meta['t1_perm']).contiguous()
            T2_perm = T2.permute(meta['t2_perm']).contiguous()

            T1_exp = expand_dims_to(T1_perm, meta['n_dim'], dim=-1)
            T2_exp = expand_dims_to(T2_perm, meta['n_dim'], dim=2)

            T = T1_exp * T2_exp

            if meta['sum_axis']:
                T = torch.sum(T, dim=meta['sum_axis'])

            results.append(T * meta['normalizer'])

        return results

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for tc in self.tcs:
            lines.append(f"  {tc}")
        lines.append(")")
        return "\n".join(lines)
    
class TargetBroadcastTC(torch.nn.Module):
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

    def forward(self, T1: Tensor, T2: Tensor) -> Tensor:
        P = len(T1)
        B = T1[0].size(0)
        C = T1[0].size(1)
        REST = (3,) * self.r_o

        T = torch.stack(
            [expand_dims_to(t1, self.n_dim, dim=-1) for t1 in T1], dim=0
        ) * expand_dims_to(T2, self.n_dim, dim=2).unsqueeze(0)

        if self.k > 0:
            T = torch.sum(
                T, dim=[i + 1 for i in range(self.r_1 - self.k + 2, self.r_1 + 2)]
            )

        return T.reshape((P, B, C) + REST)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_1={self.r_1}, r_1={self.r_2}, r_1={self.r_o}, k={self.k})"