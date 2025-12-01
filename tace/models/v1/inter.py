################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Tuple, Optional, Any

import torch
from torch import Tensor
from cartnn.o3 import ICTD

from .act import ACT
from .mlp import MLP
from .linear import SelfInteraction, ElementLinear
from .ctr import Contraction
from .utils import Graph, LAMMPS_MP, dict2flatten, flatten2dict, expand_dims_to
from cartnn.util.torch_scatter import scatter_sum

class Interaction(torch.nn.Module):

    Qs: List[Tensor]
    num_weights: List[int] 
    
    def __init__(
        self,
        atomic_numbers: int,
        num_channel: int,
        num_channel_hidden: int,
        max_r_in: int,
        r_sc: List[int],
        max_r_out: int,
        avg_num_neighbors,
        num_radial_basis,
        radial_mlp={},
        inter: Dict = {},
        ictd: Dict = {},
        bias: bool = False,
        layer: int = -1,
        num_layers: int = -1,
    ) -> None:
        super().__init__()

        # === arguments ===
        weight = ictd.get('weight', 'max')
        enable_residual = inter.get('residual', False)
        enable_layer_norm = radial_mlp.get('enable_layer_norm', False)
        self.add_source_target_embedding = inter.get('add_source_target_embedding', False)
        normalizer = inter.get('normalizer', {})
        normalizer_act_1 = normalizer.get('act_1', 'silu')
        normalizer_act_2 = normalizer.get('act_2', 'tanh')
        normalizer_hidden_dim = normalizer.get('hidden', [64])
        normalizer_bias = normalizer.get('bias', False)
        self.normalizer_type = normalizer.get('type', 'fixed')
        self.normalizer_scale_shift_trainable = normalizer.get('scale_shift_trainable', False)
        self.register_buffer(
            "avg_num_neighbors",
            torch.tensor(avg_num_neighbors, dtype=torch.get_default_dtype()),
        )
        self.linear_up = SelfInteraction(
            in_channel=num_channel,
            out_channel=num_channel,
            rs=list(range(max_r_in + 1)),
            bias=bias,
        )
        self.tc = Contraction(
            num_channel=num_channel,
            num_channel_hidden=num_channel_hidden,
            max_r_in=max_r_in,
            max_r_out=max_r_out,
            num_radial_basis=num_radial_basis,
            radial_mlp=radial_mlp,
            inter=inter,
            bias=bias,
            layer=layer,
            num_layers=num_layers,
            ictd=ictd,
        )
        self.enable_residual = enable_residual or layer > 0 or num_layers == 1
        if self.enable_residual:
            self.scs = torch.nn.ModuleDict()
            for r in r_sc:
                self.scs[str(r)] = ElementLinear(
                    num_channel_hidden,
                    num_channel,
                    bias=(r == 0 and bias),
                    atomic_numbers=atomic_numbers,
                )

        self.Qs  = []
        self.num_weights = []

        if weight == 'max':
            self.decomposition = True
            self.not_trainable = True
            for r in range(max_r_out + 1):
                PS, DS, CS, SS = ICTD(r, r)
                num_weights = len(DS)
                self.num_weights.append(num_weights)
                self.Qs.append(DS[0].to(dtype=torch.get_default_dtype()))
        elif weight == 'all':
            self.decomposition = True
            self.not_trainable = False
            for r in range(max_r_out + 1):
                PS, DS, CS, SS = ICTD(r)
                num_weights = len(DS)
                self.num_weights.append(num_weights)
                self.Qs.append(
                    torch.stack(DS, dim=0).to(dtype=torch.get_default_dtype())
                )
            self.radial_modulation = MLP(
                num_radial_basis,
                sum([num_channel * self.num_weights[r] for r in range(max_r_out + 1)]),
                radial_mlp["hidden"][layer],
                act=radial_mlp["act"],
                bias=False,
                forward_weight_init=True,
            )
        else:
            self.decomposition = False
            self.not_trainable = True
            for r in range(max_r_out + 1):
                PS, DS, CS, SS = ICTD(r, r)
                num_weights = len(DS)
                self.num_weights.append(num_weights)
                self.Qs.append(DS[0].to(dtype=torch.get_default_dtype()))
        del PS, DS, CS, SS

        if self.add_source_target_embedding:
            self.source_embedding = MLP(
                len(atomic_numbers),
                num_channel,
                hidden_dim=[],
                act=None,
                bias=False,
                forward_weight_init=True,
                enable_layer_norm=False,
            )
            self.target_embedding = MLP(
                len(atomic_numbers),
                num_channel,
                hidden_dim=[],
                act=None,
                bias=False,
                forward_weight_init=True,
                enable_layer_norm=False,
            )
            torch.nn.init.uniform_(self.source_embedding.mlp[0].weight, a=-0.001, b=0.001)
            torch.nn.init.uniform_(self.target_embedding.mlp[0].weight, a=-0.001, b=0.001)

        if self.normalizer_type == 'dynamic':
            # this normalizer_type is based on mace, for UMLIP
            if self.add_source_target_embedding:
                normalizer_in_dim = num_radial_basis + 2 * num_channel
            else:
                normalizer_in_dim = num_radial_basis
            self.density_normalizer = MLP(
                normalizer_in_dim,
                1,
                normalizer_hidden_dim,
                act=normalizer_act_1,
                bias=normalizer_bias,
                forward_weight_init=True,
                enable_layer_norm=enable_layer_norm,
            )
            self.normalizer_act_2 = ACT[normalizer_act_2]()
            if self.normalizer_scale_shift_trainable:
                self.alpha = torch.nn.Parameter(torch.tensor(avg_num_neighbors), requires_grad=True)
                self.beta = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.r_sc = r_sc
        self.max_r_in = max_r_in
        self.max_r_out = max_r_out
        self.layer = layer
        self.num_layers = num_layers
        self.num_channel = num_channel

    def forward(
        self,
        node_feats: Dict[int, Tensor],
        node_attrs: Tensor,
        node_attrs_lmp: Tensor,
        edge_feats: Tensor,
        edge_attrs: Dict[int, Tensor],
        edge_index: Tensor,
        cutoff: Tensor,
        graph: Graph,
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:

        lmp = graph.lmp
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None
        dtype = node_attrs_lmp.dtype
        device = node_attrs_lmp.device
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lmp_data=lmp_data,
            lmp_natoms=lmp_natoms,
            layer=self.layer,
        )

        if self.add_source_target_embedding:
            source_embedding = self.source_embedding(node_attrs)
            target_embedding = self.target_embedding(node_attrs) # TODO BUG check which node_attrs in lmp
            edge_feats = torch.cat(
                [
                    edge_feats,
                    source_embedding[edge_index[0]],
                    target_embedding[edge_index[1]],
                ],
                dim=-1,
            )
        if hasattr(self, 'density_normalizer'):
            edge_density = self.normalizer_act_2(self.density_normalizer(edge_feats) ** 2)
            if cutoff is not None:
                edge_density = edge_density * cutoff
            density = scatter_sum(
                src=edge_density, index=edge_index[1], dim=0, dim_size=node_attrs_lmp.shape[0]
            )
            # if lmp:
            #     density = self.truncate_ghosts(density, nlocal) 
            if self.normalizer_scale_shift_trainable:
                density = density * self.beta + self.alpha
            else:
                density = density + 1
            density = density.masked_fill(density == 0, 1e-9)

        for r_1 in range(self.max_r_in + 1):
            node_feats[r_1] = node_feats[r_1][edge_index[0]]


        m_ji = self.tc(node_feats, edge_attrs, edge_feats, cutoff)
        m_i = torch.jit.annotate(Dict[int, Tensor], {})

        if self.decomposition:
            if self.not_trainable:
                for r in m_ji.keys():
                    T = scatter_sum(
                        src=m_ji[r],
                        index=edge_index[1],
                        dim=0,
                        dim_size=node_attrs_lmp.size(0),
                    )
                    if self.normalizer_type == 'dynamic':
                        normalizer = expand_dims_to(density, T.ndim, dim=-1)
                    else:
                        normalizer = self.avg_num_neighbors
                    T = T / normalizer
                    B = T.size(0)
                    C = T.size(1)
                    REST = (3,) * r
                    m_i[r] = (
                        T.reshape(B, C, -1) @ self.Qs[r].to(device=device, dtype=dtype)
                    ).reshape((B, C) + REST)
            else:
                start_idx = 0
                radial_weights = self.radial_modulation(edge_feats)

                for r in range(self.max_r_out + 1):
                    W = radial_weights[:, start_idx:start_idx+self.num_weights[r]*self.num_channel].reshape(-1, 
                    self.num_weights[r], self.num_channel)
                    start_idx += self.num_weights[r] * self.num_channel
                    W = W.permute(1, 0, 2)
                    for _ in range(r):
                        W = W.unsqueeze(-1)
                    T = m_ji[r]
                    B = T.size(0)
                    C = T.size(1)
                    REST = (3,) * r
                    T = T.reshape(B * C, -1).unsqueeze(0)
                    T = T.repeat(self.num_weights[r], 1, 1)
                    T = torch.bmm(T, self.Qs[r].to(device=device, dtype=dtype)) 
                    T = T.reshape((-1,) + (B, C) + REST)
                    T = T * W
                    T = torch.sum(T, dim=0)
                    T = scatter_sum(
                        src=T,
                        index=edge_index[1],
                        dim=0,
                        dim_size=node_attrs.size(0),
                    )
                    if self.normalizer_type == 'dynamic':
                        normalizer = expand_dims_to(density, T.ndim, dim=-1)
                    else:
                        normalizer = self.avg_num_neighbors
                    T = T / normalizer
                    m_i[r] = T
        else:
            for r in m_ji.keys():
                T = scatter_sum(
                    src=m_ji[r],
                    index=edge_index[1],
                    dim=0,
                    dim_size=node_attrs_lmp.size(0),
                )
                if self.normalizer_type == 'dynamic':
                    normalizer = expand_dims_to(density, T.ndim, dim=-1)
                else:
                    normalizer = self.avg_num_neighbors
                T = T / normalizer
                m_i[r] = T

        residual = torch.jit.annotate(Dict[int, Tensor], {})
        if self.enable_residual:
            for nu, sc in self.scs.items():
                nu = int(nu)
                residual[nu] = sc(m_i[nu], node_attrs_lmp)

        if lmp:
            node_attrs_lmp = self.truncate_ghosts(node_attrs_lmp, nlocal)
            max_r = max(m_i.keys())
            m_i = dict2flatten(max_r, m_i)
            m_i = self.truncate_ghosts(m_i, nlocal)
            m_i = flatten2dict(max_r, m_i, self.num_channel)

            if len(residual) > 0:
                max_r = max(residual.keys())                
                residual = dict2flatten(max_r, residual)
                residual = self.truncate_ghosts(residual, nlocal)
                residual = flatten2dict(max_r, residual, self.num_channel)
        return m_i, residual


    def handle_lammps(
        self,
        node_feats: Dict[int, Tensor],
        lmp_data: Optional[Any],
        lmp_natoms: Tuple[int, int],
        layer: int,
    ) -> Tensor:  
        _, nghosts = lmp_natoms
        first_layer = (layer == 0)
        if lmp_data is None or first_layer or torch.jit.is_scripting():
            return node_feats
        max_r = max(node_feats.keys())
        node_feats = dict2flatten(max_r, node_feats)
        pad = torch.zeros(
            (nghosts, node_feats.shape[1]),
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        node_feats = torch.cat((node_feats, pad), dim=0)
        node_feats = LAMMPS_MP.apply(node_feats, lmp_data)
        return flatten2dict(max_r, node_feats, self.num_channel)
    
    def truncate_ghosts(
        self, t: Tensor, nlocal: Optional[int] = None
    ) -> Tensor:
        return t[:nlocal] if nlocal is not None else t