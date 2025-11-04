################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import Dict, List, Optional


import torch
from torch import nn, Tensor
import opt_einsum_fx

from .mlp import MLP
from .paths import generate_combinations, TensorContractionUtils, parse_einsum_expr
from .linear import Linear
from .utils import EINSUM_STR


BATCH = 5
CHANNEL = 6

def so2_kernel(T1, T2, r_1, r_2, k):
    B, C = T1.shape[:2]
    shape = [B, C] + [3]*(r_1-k) + [3]*(r_2-k)
    out = torch.zeros(shape, dtype=T1.dtype, device=T1.device)
    T2_values = T2[..., *[0]*r_2] 
    T1_slice = T1[..., *[0]*k] 
    for _ in range(r_1 - k):
        T2_values = T2_values.unsqueeze(-1)
    out[..., *[0]*(r_2-k)] = T1_slice * T2_values
    return out


class SO2TC(torch.nn.Module):
    def __init__(self, exprs: List[str]) -> None:
        super().__init__()
        self.exprs = exprs
        for expr in exprs: # for a given combination, each expr <-> each path 
            tc = parse_einsum_expr(expr)
            self.k = tc.k
            self.r_1 = tc.r_1
            self.r_2 = tc.r_2
            self.r_o = tc.r_o
            break

        if self.r_1 > 0:
            in_1 = 'b' + ''.join(EINSUM_STR[0:2])
            in_2 = 'bc' + ''.join(EINSUM_STR[2:self.r_1+2])
            in_2 = list(in_2)
            in_2[-1] = in_1[-1]
            in_2 = ''.join(in_2)
            out = 'bc' + in_1[1] + in_2[2:-1]
            einsum_str = in_1 + ',' + in_2 + '->' + out
            rotate = torch.fx.symbolic_trace(
                lambda T1, T2: torch.einsum(einsum_str, T1, T2)
            )
            self.rotate = opt_einsum_fx.optimize_einsums_full(
                model=rotate,
                example_inputs=(
                    torch.randn([BATCH,3,3]),
                    torch.randn([BATCH, CHANNEL] +[3]*self.r_1),
                    ),
                )

        if self.r_o > 0:
            in_1 = 'b' + ''.join(EINSUM_STR[0:2])
            in_2 = 'bc' + ''.join(EINSUM_STR[2:self.r_o+2])
            in_2 = list(in_2)
            in_2[-1] = in_1[-1]
            in_2 = ''.join(in_2)
            out = 'bc' + in_1[1] + in_2[2:-1]
            einsum_str = in_1 + ',' + in_2 + '->' + out
            rotate_back = torch.fx.symbolic_trace(
                lambda T1, T2: torch.einsum(einsum_str, T1, T2)
            )
            self.rotate_back = opt_einsum_fx.optimize_einsums_full(
                model=rotate_back,
                example_inputs=(
                    torch.randn([BATCH,3,3]),
                    torch.randn([BATCH, CHANNEL] + [3]*self.r_o),
                    ),
                )
                 
        self.normalizer = 1.0 / sqrt(3**self.k)

    def forward(self, T1: Tensor, T2: Tensor, rot_mat: Tensor, rot_mat_inv) -> List[Tensor]:
        results = []
        for _ in self.exprs:
            for _ in range(self.r_1):
                T1 = self.rotate(rot_mat, T1)
            out = so2_kernel(T1, T2, self.r_1, self.r_2, self.k) * self.normalizer
            for _ in range(self.r_o):
                out = self.rotate_back(rot_mat_inv, out)
            results.append(out)
        return results


    def __repr__(self):
        return f"{self.__class__.__name__}(paths={self.exprs})"

class SO2_Contraction(torch.nn.Module):
    def __init__(
        self,
        num_channel: int = 64,
        num_channel_hidden: int = 64,
        max_r_in: int = 3,
        max_r_out: int = 3,
        num_radial_basis: int = 8,
        radial_mlp: Dict = {},
        inter: Dict = {},
        bias: bool = False,
        layer: int = -1,
        num_layers: int = -1,
        ictd: Dict = {},
    ):
        super().__init__()

        max_paths = inter.get('max_paths', 1)
        allow_nosym = inter.get('allow_nosym', True)
        weight = ictd.get('weight', 'max')
        kernel = inter.get('kernel', 'einsum')
        tensor_product_only = inter.get("tensor_product_only", False)
        combs = generate_combinations(
            max_r_in,
            max_r_out,
            max_r_out,
            restriction=None if layer == 0 else inter["restriction"][layer],
            allow_nosym=allow_nosym,
            tensor_product_only=tensor_product_only,
        )

        # ==== tensor contraction ====
        comb_paths = []  # every comb correspond to a list with multipahts
        comb_path_counts = []
        r_o_path_counts = {r: 0 for r in range(max_r_out + 1)}
        self.tcs = nn.ModuleList()
        for comb in combs:
            (r_1, r_2, r_o) = comb
            paths: List[str] = TensorContractionUtils.generate_paths(
                r_1,
                r_2,
                r_o,
                add_batch_and_channel=True,
                allow_nosym=allow_nosym,
                max_paths=max_paths,
            )
            comb_paths.append(paths)
            r_o_path_counts[r_o] += len(paths)
            comb_path_counts.append(len(paths))

            # SO2
            self.tcs.append(SO2TC(paths))

        if  weight != "all":
            # ==== conv weight ====
            if inter.get('add_source_target_embedding', False):
                radial_in_dim = num_radial_basis + 2 * num_channel
            else:
                radial_in_dim = num_radial_basis
            self.radial_net = MLP(
                radial_in_dim,
                num_channel * sum([len(paths) for paths in comb_paths]),
                radial_mlp["hidden"][layer],
                act=radial_mlp["act"],
                bias=radial_mlp.get('bias', False),
                forward_weight_init=radial_mlp.get("forward_weight_init", True),
                enable_layer_norm=radial_mlp.get('enable_layer_norm', False),
            )
            # === slice ===
            current = 0
            self.weight_starts = []
            for count in comb_path_counts:
                self.weight_starts.append(current)
                current += count * num_channel
            self.enable_lower_weights = False
        else:
            # ==== path weight ====
            self.path_weights = nn.Parameter(
                torch.ones(
                    num_channel * sum([len(paths) for paths in comb_paths]),
                    dtype=torch.get_default_dtype()
                ),
                requires_grad=True,
            )
            # === slice ===
            current = 0
            self.weight_starts = []
            for count in comb_path_counts:
                self.weight_starts.append(current)
                current += count * num_channel
            self.enable_lower_weights = True


        # === linear ===
        self.linear_downs = nn.ModuleList(
            [
                Linear(
                    num_channel * count,
                    num_channel_hidden,
                    bias=False,  # should not be True
                )
                for r_o, count in r_o_path_counts.items()
            ]
        )

        self.combs = combs
        self.comb_path_counts = comb_path_counts
        self.num_channel = num_channel
        self.max_r_in = max_r_in
        self.max_r_out = max_r_out
        self.layer = layer
        self.num_layers = num_layers
        self.kernel = kernel

    def forward(
        self,
        x: Dict[int, Tensor],
        y: Dict[int, Tensor],
        radial: Tensor,
        cutoff: Tensor,
        rot_mat: Optional[Tensor] = None,
        rot_mat_inv: Optional[Tensor] = None,
    ) -> Dict[int, Tensor]:

        if self.enable_lower_weights:
            path_weights = self.path_weights.unsqueeze(0)
            B = path_weights.size(0)

            buffer = torch.jit.annotate(
                Dict[int, List[Tensor]], {r: [] for r in range(self.max_r_out + 1)}
            )

            for i, tc in enumerate(self.tcs):
                num_paths = self.comb_path_counts[i]
                start_idx = self.weight_starts[i]
                r_1, r_2, r_o = self.combs[i]
                outs: List[Tensor] = tc(x[r_1], y[r_2], rot_mat, rot_mat_inv)
                weights = path_weights[
                    :, start_idx : start_idx + num_paths * self.num_channel
                ]
                weights = weights.view(B, num_paths, self.num_channel)

                # [P, B, C, ...]
                stacked_outs = torch.stack(outs, dim=0)

                # [1, B, C, 1, 1, ...] -> [P, B, C, ...]
                expanded_weights = weights.permute(1, 0, 2)  # [P, B, C]
                for _ in range(stacked_outs.ndim - 3):
                    expanded_weights = expanded_weights.unsqueeze(-1)

                # vectorized
                weighted_outs = stacked_outs * expanded_weights
                buffer[r_o].extend(torch.unbind(weighted_outs, dim=0))
        else:
            conv_weights = self.radial_net(radial)
            if cutoff is not None:
                conv_weights = conv_weights * cutoff
            B = conv_weights.size(0)

            buffer = torch.jit.annotate(
                Dict[int, List[Tensor]], {r: [] for r in range(self.max_r_out + 1)}
            )

            for i, tc in enumerate(self.tcs):
                num_paths = self.comb_path_counts[i]
                start_idx = self.weight_starts[i]
                r_1, r_2, r_o = self.combs[i]
                outs: List[Tensor] = tc(x[r_1], y[r_2], rot_mat, rot_mat_inv)
                weights = conv_weights[
                    :, start_idx : start_idx + num_paths * self.num_channel
                ]
                weights = weights.view(B, num_paths, self.num_channel)

                # [P, B, C, ...]
                stacked_outs = torch.stack(outs, dim=0)

                # [1, B, C, 1, 1, ...] -> [P, B, C, ...]
                expanded_weights = weights.permute(1, 0, 2)  # [P, B, C]
                for _ in range(stacked_outs.ndim - 3):
                    expanded_weights = expanded_weights.unsqueeze(-1)

                # vectorized
                weighted_outs = stacked_outs * expanded_weights
                buffer[r_o].extend(torch.unbind(weighted_outs, dim=0))

        OUTS = torch.jit.annotate(Dict[int, Tensor], {})
        for i, linear in enumerate(self.linear_downs):
            t = buffer[i]
            tmp = torch.cat(t, dim=1)
            tmp = linear(tmp)
            OUTS[i] = tmp

        return OUTS
