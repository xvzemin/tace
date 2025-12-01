################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import List

import torch
from torch import nn, Tensor
import opt_einsum_fx


from ..paths import parse_einsum_expr


BATCH = 5
CHANNEL = 6

class EinsumTC(torch.nn.Module):
    def __init__(self, exprs: List[str]) -> None:
        super().__init__()
        self.exprs = exprs
        for expr in exprs: # for a given combination, each expr <-> each path 
            tc = parse_einsum_expr(expr)
            k = tc.k
            r_1 = tc.r_1
            r_2 = tc.r_2
            break
        self.ctrs = nn.ModuleList()
        for expr in exprs:
            ctr = torch.fx.symbolic_trace(lambda T1, T2: torch.einsum(expr, T1, T2))
            self.ctrs.append(
                opt_einsum_fx.optimize_einsums_full(
                    model=ctr,
                    example_inputs=(
                        torch.randn([BATCH] + [CHANNEL] + [3] * r_1),
                        torch.randn([BATCH] + [CHANNEL] + [3] * r_2),
                    ),
                )
            )
        self.normalizer = 1.0 / sqrt(3**k)

    def forward(self, T1: Tensor, T2: Tensor) -> List[Tensor]:
        results = []
        for ctr in self.ctrs:
            results.append(ctr(T1, T2) * self.normalizer)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(paths={self.exprs})"
