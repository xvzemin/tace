################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
import opt_einsum_fx
from cartnn import o3

from .utils import add_to_left
from .linear import Linear, ElementLinear, CWLinear, ElementCWLinear, SelfInteraction
from .paths import satisfy, generate_prod_paths
from .kernel.broadcast import TargetBroadcastTC
from .kernel.analytical import TargetAnalyticalTC
from .kernel.matrix import TargetMatrixTC

_PATH = 4
BATCH = 5
CHANNEL = 6

LINEAR = {
    (False, True): Linear,
    (True, True): ElementLinear,
    (False, False): CWLinear,
    (True, False): ElementCWLinear,
}

class SelfContraction(torch.nn.Module):
    def __init__(
        self,
        num_channel: int = 64,
        num_channel_hidden: int = 64,
        max_r_in: int = 3,
        rank_of_out: List[int] = 2,
        atomic_numbers: List[int] = [],
        prod: Dict = {},
        ictd: Dict = {},
        bias: bool = False,
        layer: int = -1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        # === init ===
        coupled = prod.get("coupled", True)
        element = prod.get("element", True)
        correlation = prod.get("correlation", 3)
        allow_nosym = prod.get("allow_nosym", True)
        restriction = prod.get("restriction", None)
        weight = ictd.get("weight", "max")
        kernel = prod.get("kernel", "einsum")

        if coupled:
            self.cat = True
        else:
            self.cat = False

        if weight == "all" or weight == "max":
            self.decomposition = True
        else:
            self.decomposition = False

        if isinstance(correlation, int):
            self.correlation = correlation
        else:
            self.correlation = correlation[layer]  

        max_left = prod.get("max_left", [[max_r_in] * self.correlation for _ in range(num_layers)])[layer]
        max_right = prod.get("max_right", [[max_r_in] * self.correlation for _ in range(num_layers)])[layer]
        max_hidden = prod.get("max_hidden", [[max_r_in] * self.correlation for _ in range(num_layers)])[layer]

        self.max_r_in = max_r_in
        self.rank_of_out = rank_of_out
        self.layer = layer
        self.kernel = kernel

        # === ICT ===
        for r in range(max_r_in + 1):
            DS = o3.ICTD(r, r)[1]
            self.register_buffer(f"D{r}", DS[0].to(torch.get_default_dtype()))
            del DS

        # === r_o count ===
        r_o_count = [[0] * (max_r_in + 1)]  # v = 1

        for r in range(max_hidden[0]+1):
            r_o_count[0][r]+=1

        for v in range(self.correlation - 1):
            r_o_count.append([0] * (max_r_in + 1))  # v > 1
            for r_1 in range(max_r_in + 1):
                for r_2 in range(max_r_in + 1):
                    for r_o in range(abs(r_2 - r_1), min(max_r_in, r_2 + r_1) + 1, 2):
                        if r_1 <= max_left[v+1] and r_2 <= max_right[v+1] and r_o <= max_hidden[v+1]:
                            if satisfy(r_1, r_2, restriction, r_o):
                                k = (r_1 + r_2 - r_o) // 2
                                if allow_nosym:
                                    r_o_count[v + 1][r_o] += r_o_count[v][r_1]
                                else:
                                    if k == r_1 or k == r_2:
                                        r_o_count[v + 1][r_o] += r_o_count[v][r_1]

                                                               
        # === lienar ===
        self.linears = nn.ModuleDict()
        for v in range(self.correlation):
            inner_dict = nn.ModuleDict()
            for r in rank_of_out:
                if sum([r_o_count[v][r]]) > 0: # TODO for xzm, check if exists BUG
                    linear_layer = LINEAR[(element, coupled)](
                        num_channel_hidden * sum([r_o_count[v][r]]),
                        num_channel_hidden,
                        bias=(r == 0 and bias),
                        atomic_numbers=atomic_numbers,
                    )
                    inner_dict[str(r)] = linear_layer
            self.linears[str(v)] = inner_dict

        self.linear = SelfInteraction(
            in_channel=num_channel_hidden,
            out_channel=num_channel,
            rs=rank_of_out,
            bias=(r == 0 and bias),
        )

        # === prod ===
        self.paths_list_list, self.exprs_list_list = generate_prod_paths(
            max_left, max_right, max_hidden, max_r_in, rank_of_out, self.correlation, allow_nosym, restriction
        )
        self.ctrs = nn.ModuleList()
        self.ctrs = nn.ModuleList()


        for v in range(self.correlation - 1):
            ctrs = nn.ModuleList()
            for (r_1, r_2, r_o), expr in zip(
                self.paths_list_list[v], self.exprs_list_list[v]
            ):
                expr = expr[0]
                if self.kernel == 'einsum':
                    # === expr ===
                    inputs, output = expr.split("->")
                    in1, in2 = [x.strip() for x in inputs.split(",")]
                    in1 = "a" + in1
                    output = "a" + output
                    expr = in1 + "," + in2 + "->" + output

                    ctr = torch.fx.symbolic_trace(
                        lambda T1, T2: torch.einsum(expr, T1, T2)
                    )
                    ctrs.append(
                        opt_einsum_fx.optimize_einsums_full(
                            model=ctr,
                            example_inputs=(
                                torch.randn(
                                    [_PATH]
                                    + [BATCH]
                                    + [CHANNEL]
                                    + [3] * r_1
                                ),
                                torch.randn(
                                    [BATCH]
                                    + [CHANNEL]
                                    + [3] * r_2
                                ),
                            ),
                        )
                    )
                elif kernel == 'broadcast':
                    ctrs.append(TargetBroadcastTC(expr))
                elif kernel == 'matrix':
                    ctrs.append(TargetMatrixTC(expr))
                elif kernel == 'analytical':
                    ctrs.append(TargetAnalyticalTC(expr))
                else:
                    raise ValueError(f"Unsupported kernel type: {kernel}")
            self.ctrs.append(ctrs)

    def D(self, l: int):
        return dict(self.named_buffers())[f"D{l}"]
    
    def forward(
        self,
        node_feats: Dict[int, Tensor],
        node_attrs: Tensor,
        sc: Dict[int, Tensor],
    ) -> Dict[int, Tensor]:
        

        TMP = {
            0: {
                r: [node_feats[r]] for r in node_feats
            }
        }

        for v, ctrs in enumerate(self.ctrs):
            TMP[v + 1] = {r: [] for r in range(self.max_r_in + 1)}
            for idx, ctr in enumerate(ctrs):
                r_1, r_2, r_o = self.paths_list_list[v][idx]
                k = (r_1 + r_2 - r_o) // 2

                tmp = ctr(
                    torch.stack(TMP[v][r_1], dim=0),
                    node_feats[r_2],
                ) / sqrt(3**k)

                if self.decomposition:
                    P = tmp.size(0)
                    B = tmp.size(1)
                    C = tmp.size(2)
                    REST = (3,) * r_o

                    tmp = torch.bmm(
                        tmp.reshape(P, B * C, -1), self.D(r_o).repeat(P, 1, 1)
                    ).reshape((P, B, C) + REST)

                tmp = torch.unbind(tmp, dim=0)
                TMP[v + 1][r_o].extend(tmp)

        outs = {}
        for v_str, linears in self.linears.items():
            v = int(v_str)
            for r_str, linear in linears.items():

                r = int(r_str)

                if self.cat:
                    merged = torch.cat([t for t in TMP[v][r]], dim=1)
                else:
                    merged = torch.stack([t for t in TMP[v][r]], dim=0)

                out = linear(merged, node_attrs)

                if r in outs:
                    outs[r] += out
                else:
                    outs[r] = out



        return add_to_left(self.linear(outs), sc)


