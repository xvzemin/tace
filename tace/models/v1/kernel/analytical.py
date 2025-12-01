################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
'''It will be optimized in the future and is only applicable to irreducible Cartesian tensor products'''
'''Rreducible Cartesian tensor products is not correct for this, since not permute'''
from math import sqrt
from typing import List

import torch
from torch import Tensor, nn
from torch.autograd import Function
import opt_einsum_fx

from ..paths import return_tcs
import torch
from torch import nn, Tensor
from torch.autograd import Function
from math import sqrt

_PATH = 4
BATCH = 5
CHANNEL = 6


class AnalyticalTCFunction(Function):
    @staticmethod
    def forward(ctx, T1: Tensor, T2: Tensor, meta, ctr, ctr_back_T1, ctr_back_T2):
        B, C1, C2 = T1.shape[0], T1.shape[1], T2.shape[1]
        M, K, N, r_o = meta['M'], meta['K'], meta['N'], meta['r_o']

        T1_mat = T1.reshape(B, C1, M, K)
        T2_mat = T2.reshape(B, C2, K, N)

        T = ctr(T1_mat, T2_mat)
        out = T.reshape((B, C1) + (3,) * r_o) * meta['normalizer']

        ctx.save_for_backward(T1_mat, T2_mat)
        ctx.meta = meta
        ctx.ctr_back_T1 = ctr_back_T1
        ctx.ctr_back_T2 = ctr_back_T2
        ctx.shape1 = T1.shape
        ctx.shape2 = T2.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        T1_mat, T2_mat = ctx.saved_tensors
        meta = ctx.meta
        normalizer = meta['normalizer']

        grad_output = grad_output * normalizer

        B = grad_output.shape[0]
        C_out = grad_output.shape[1]
        M, N = meta['M'], meta['N']
        grad_mat = grad_output.reshape(B, C_out, M, N)

        grad_T1_mat = ctx.ctr_back_T1(grad_mat, T2_mat)
        grad_T2_mat = ctx.ctr_back_T2(T1_mat, grad_mat)

        grad_T1 = grad_T1_mat.reshape(ctx.shape1)

        # C2_orig = ctx.shape2[1]
        # if C2_orig == grad_T2_mat.shape[1]:
        #     grad_T2 = grad_T2_mat.reshape(ctx.shape2)
        # else:  # C2_orig == 1
        #     grad_T2_reduced = grad_T2_mat.sum(dim=1, keepdim=True)
        #     grad_T2 = grad_T2_reduced.reshape(ctx.shape2)

        grad_T2_reduced = grad_T2_mat.sum(dim=1, keepdim=True)
        grad_T2 = grad_T2_reduced.reshape(ctx.shape2)

        return grad_T1, grad_T2, None, None, None, None


class AnalyticalTC(nn.Module):
    def __init__(self, exprs: List[str]) -> None:
        super().__init__()
        self.exprs = exprs
        self.tcs = return_tcs(exprs)
        self.metas = []
        self.ctrs = []
        self.ctrs_back_T1 = []
        self.ctrs_back_T2 = []

        for tc in self.tcs:
            r_1, r_2, r_o, k = tc.r_1, tc.r_2, tc.r_o, tc.k
            normalizer = 1.0 / sqrt(3**k)
            M = 3 ** (r_1 - k)
            K = 3 ** k
            N = 3 ** (r_2 - k)

            self.metas.append({
                'normalizer': normalizer,
                'r_1': r_1,
                'r_2': r_2,
                'r_o': r_o,
                'k': k,
                'M': M,
                'N': N,
                'K': K,
            })

            # ---- forward einsum ----
            ctr = torch.fx.symbolic_trace(
                lambda T1, T2: torch.einsum('bcik,bckj->bcij', T1, T2)
            )
            ctr_opt = opt_einsum_fx.optimize_einsums_full(
                model=ctr,
                example_inputs=(
                    torch.randn([BATCH, CHANNEL, 3**(r_1 - k), 3**k]),
                    torch.randn([BATCH, CHANNEL, 3**k, 3**(r_2 - k)]),
                ),
            )
            self.ctrs.append(ctr_opt)

            # ---- backward einsum grad_T1 ----
            ctr_b1 = torch.fx.symbolic_trace(
                lambda G, T2: torch.einsum('bcik,bckj->bcij', G, T2.transpose(-2, -1))
            )
            ctr_b1_opt = opt_einsum_fx.optimize_einsums_full(
                model=ctr_b1,
                example_inputs=(
                    torch.randn([BATCH, CHANNEL, 3**(r_1 - k), 3**(r_2 - k)]),
                    torch.randn([BATCH, CHANNEL, 3**k, 3**(r_2 - k)]),
                ),
            )
            self.ctrs_back_T1.append(ctr_b1_opt)

            # ---- backward einsum grad_T2 ----
            ctr_b2 = torch.fx.symbolic_trace(
                lambda T1, G: torch.einsum('bcik,bckj->bcij', T1.transpose(-2, -1), G)
            )
            ctr_b2_opt = opt_einsum_fx.optimize_einsums_full(
                model=ctr_b2,
                example_inputs=(
                    torch.randn([BATCH, CHANNEL, 3**(r_1 - k), 3**k]),
                    torch.randn([BATCH, CHANNEL, 3**(r_1 - k), 3**(r_2 - k)]),
                ),
            )
            self.ctrs_back_T2.append(ctr_b2_opt)

    def forward(self, T1: Tensor, T2: Tensor) -> List[Tensor]:
        results = []
        for meta, ctr, ctr_b1, ctr_b2 in zip(self.metas, self.ctrs, self.ctrs_back_T1, self.ctrs_back_T2):
            results.append(AnalyticalTCFunction.apply(T1, T2, meta, ctr, ctr_b1, ctr_b2))
        return results


class TargetAnalyticalTCFunction(Function):
    @staticmethod
    def forward(ctx, T1, T2, meta, ctr, ctr_b1, ctr_b2):
        M, N, r_o = meta["M"], meta["N"], meta["r_o"]

        # ---- forward contraction ----
        T = ctr(T1, T2)  # [P, B, C, M, N]
        out = T.reshape(T.shape[:3] + (3,) * r_o)

        # ---- save for backward ----
        ctx.save_for_backward(T1, T2)
        ctx.meta = meta
        ctx.ctr_b1 = ctr_b1
        ctx.ctr_b2 = ctr_b2
        ctx.shape1 = T1.shape
        ctx.shape2 = T2.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        T1, T2 = ctx.saved_tensors
        meta = ctx.meta
        M, N = meta["M"], meta["N"]

        P, B, C = grad_output.shape[:3]
        grad_mat = grad_output.reshape(P, B, C, M, N)

        # ---- backprop einsums ----
        grad_T1_mat = ctx.ctr_b1(grad_mat, T2) 
        grad_T2_mat = ctx.ctr_b2(T1, grad_mat) 

        # ---- restore shapes ----
        grad_T1 = grad_T1_mat.reshape(ctx.shape1)

        # P2_orig = ctx.shape2[0]
        # if P2_orig == grad_T2_mat.shape[0]:
        #     grad_T2 = grad_T2_mat.reshape(ctx.shape2)
        # else:  # P2_orig == 1
        #     grad_T2_reduced = grad_T2_mat.sum(dim=0, keepdim=True)
        #     grad_T2 = grad_T2_reduced.reshape(ctx.shape2)

        grad_T2_reduced = grad_T2_mat.sum(dim=0, keepdim=True)
        grad_T2 = grad_T2_reduced.reshape(ctx.shape2)

        return grad_T1, grad_T2, None, None, None, None


class TargetAnalyticalTC(nn.Module):
    def __init__(self, expr: str) -> None:
        super().__init__()
        tc = return_tcs([expr])[0]

        self.r_1 = tc.r_1
        self.r_2 = tc.r_2
        self.r_o = tc.r_o
        self.k = tc.k

        M = 3 ** (self.r_1 - self.k)
        K = 3 ** self.k
        N = 3 ** (self.r_2 - self.k)
        self.meta = {
            "M": M,
            "K": K,
            "N": N,
            "r_o": self.r_o,
        }

        # ===== forward einsum =====
        ctr = torch.fx.symbolic_trace(
            lambda T1, T2: torch.einsum("pbcik,pbckj->pbcij", T1, T2)
        )
        self.ctr = opt_einsum_fx.optimize_einsums_full(
            model=ctr,
            example_inputs=(
                torch.randn([_PATH, BATCH, CHANNEL, M, K]),
                torch.randn([1, BATCH, CHANNEL, K, N]),
            ),
        )

        # ===== backward wrt T1 =====
        ctr_b1 = torch.fx.symbolic_trace(
            lambda G, T2: torch.einsum("pbcij,pbckj->pbcik", G, T2)
        )
        self.ctr_b1 = opt_einsum_fx.optimize_einsums_full(
            model=ctr_b1,
            example_inputs=(
                torch.randn([_PATH, BATCH, CHANNEL, M, N]),
                torch.randn([1, BATCH, CHANNEL, K, N]),
            ),
        )

        # ===== backward wrt T2 =====
        ctr_b2 = torch.fx.symbolic_trace(
            lambda T1, G: torch.einsum("pbcik,pbcij->pbckj", T1, G)
        )
        self.ctr_b2 = opt_einsum_fx.optimize_einsums_full(
            model=ctr_b2,
            example_inputs=(
                torch.randn([_PATH, BATCH, CHANNEL, M, K]),
                torch.randn([_PATH, BATCH, CHANNEL, M, N]),
            ),
        )

    def forward(self, T1: Tensor, T2: Tensor) -> Tensor:
        M, K, N = self.meta["M"], self.meta["K"], self.meta["N"]
        P, B, C = T1.shape[0], T1.shape[1], T1.shape[2]

        T1_mat = T1.reshape(P, B, C, M, K)
        T2_mat = T2.reshape(1, B, C, K, N)

        out = TargetAnalyticalTCFunction.apply(
            T1_mat, T2_mat, self.meta, self.ctr, self.ctr_b1, self.ctr_b2
        )
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(r_1={self.r_1}, "
            f"r_2={self.r_2}, r_o={self.r_o}, k={self.k})"
        )


