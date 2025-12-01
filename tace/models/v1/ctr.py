################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import Dict, List


import torch
from torch import nn, Tensor


from .mlp import MLP
from .paths import generate_combinations, TensorContractionUtils
from .linear import Linear
from .kernel.einsum import EinsumTC
from .kernel.broadcast import BroadcastTC
from .kernel.analytical import AnalyticalTC
from .kernel.matrix import MatrixTC


class Contraction(torch.nn.Module):
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
        combs = generate_combinations(
            max_r_in,
            max_r_out,
            max_r_out,
            restriction=None if layer == 0 else inter["restriction"][layer],
            allow_nosym=allow_nosym,
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
            if kernel == 'einsum':
                self.tcs.append(EinsumTC(paths))
            elif kernel == 'broadcast':
                self.tcs.append(BroadcastTC(paths))
            elif kernel == 'matrix':
                self.tcs.append(MatrixTC(paths))  # not correct, should not be used 
            elif kernel == 'analytical':
                self.tcs.append(AnalyticalTC(paths))  # not correct, should not be used 
            else:
                raise ValueError(f"Unsupported kernel type: {kernel}")

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
                outs: List[Tensor] = tc(x[r_1], y[r_2])
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
                outs: List[Tensor] = tc(x[r_1], y[r_2])
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
