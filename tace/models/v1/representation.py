################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
from typing import Dict, List, Optional, Union, Any


import torch
from torch import Tensor, nn
from cartnn.math import RadialBasis

from .mlp import MLP
from .angular import LegacyCartesianHarmonics2
from .inter import Interaction
from .prod import SelfContraction
from .embedding import UniversalInvariantEmbedding, UniversalEquivariantEmbedding
from .utils import Graph
from ...dataset.statistics import Statistics


class TACEDescriptor(torch.nn.Module):
    def __init__(
        self,
        statistics: List[Statistics],
        cutoff: float = 6.0,
        num_layers: int = 2,
        num_channel: int = 64,
        num_channel_hidden: int = 64,
        max_r_1: int = 2,
        max_r_2: int = 3,
        bias: bool = False,
        target_property: Dict = {},
        radial_basis: Dict = {},
        angular_basis: Dict = {},
        radial_mlp: Dict = {},
        inter: Dict = {},
        prod: Dict = {},
        ictd: Dict = {},
        universal_embedding: Optional[List[Dict[str, Union[int, str]]]] = None,
        readout_mlp: Dict = {},
        **kwargs,
    ):
        super().__init__()

        # === init ===
        atomic_numbers = sorted(statistics[0]['atomic_numbers'])
        avg_num_neighbors = statistics[0]['avg_num_neighbors']

        if isinstance(max_r_1, int):
            max_r_1 = [max_r_1] * num_layers
        
        if isinstance(max_r_2, int):
            max_r_2 = [max_r_2] * num_layers
            
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype()))
        self.register_buffer("num_layers", torch.tensor(num_layers, dtype=torch.int64))
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
            # === direct Property ===
        target_rank = []
        target_rank.extend([0]) if "energy" in target_property else None
        target_rank.extend([0]) if "magmoms_0" in target_property else None
        target_rank.extend([1]) if "magmoms_1" in target_property else None
        if readout_mlp.get('enable_nonlinearty_for_tensor', False):
            target_rank.extend([0, 1]) if "direct_dipole" in target_property else None
        else:
            target_rank.extend([1]) if "direct_dipole" in target_property else None
        target_rank.extend([0, 2]) if "direct_polarizability" in target_property else None
        if readout_mlp.get('enable_nonlinearty_for_tensor', False):
            target_rank.extend([0, 1]) if "direct_forces" in target_property else None
        else:
            target_rank.extend([1]) if "direct_forces" in target_property else None
        target_rank.extend([0, 2]) if "direct_stress" in target_property or "direct_virials" in target_property else None
        # target_rank.extend([0]) if "nuclear_chemical_shift" in target_property else None
        # target_rank.extend([0, 1, 2]) if "nuclear_shielding" in target_property else None
        # target_rank.extend([0, 2, 4]) if "elasticity_tensor" in target_property else None

        target_rank = sorted(list(set(target_rank)))
        if max(max_r_1) < max(target_rank):
            raise ValueError(
                f"cfg.model.config.max_r_1 {max(max_r_1)} should be greatet than"
                f"the tensor property you want to predict {max(target_property)}."
            )
    
        rank_of_in = []  # in of inter
        rank_of_hidden = []  # out of inter and in of prod
        rank_of_out = []  # out of prod, out of sc

        for layer in range(num_layers):
            if layer == 0:
                rank_of_in.append([0])
                rank_of_hidden.append(list(range(max_r_2[layer] + 1)))
                rank_of_out.append(target_rank if layer == num_layers - 1 else list(range(max_r_1[layer] + 1)))
            else:
                rank_of_in.append(list(range(max_r_1[layer] + 1)))
                rank_of_hidden.append(list(range(max_r_2[layer] + 1)))
                rank_of_out.append(target_rank if layer == num_layers - 1 else list(range(max_r_1[layer] + 1)))

        # === element embedding ===
        self.node_embedding = MLP(
            len(atomic_numbers),
            num_channel,
            hidden_dim=[],
            act=None,
            bias=False,
            forward_weight_init=True,
        )

        # === universal embedding ===
        if universal_embedding is not None:
            self.invariant_embeddings = universal_embedding.get("invariant", None)
            self.equivariant_embeddings = universal_embedding.get("equivariant", None)
            if self.invariant_embeddings is not None:
                self.uie_embedding = UniversalInvariantEmbedding(
                    num_channel,
                    self.invariant_embeddings,
                )
            if self.equivariant_embeddings is not None:
                self.uee_embeddings = nn.ModuleList()
                for _ in range(num_layers):
                    self.uee_embeddings.append(
                        UniversalEquivariantEmbedding(
                            self.equivariant_embeddings,
                            atomic_numbers,
                            num_channel,
                        )
                    )

        # === radial basis ===
        radial_embedding = RadialBasis(
            cutoff=cutoff,
            num_basis=radial_basis.get('num_radial_basis', 8),
            polynomial_cutoff=radial_basis.get('polynomial_cutoff', 5),
            radial_basis=radial_basis.get('radial_basis', 'j0'),
            distance_transform=radial_basis.get('distance_transform', None),
            order=radial_basis.get('order', 0),
            trainable=radial_basis.get('trainable', False),
            apply_cutoff=radial_basis.get("apply_cutoff", True)
        )
        self.radial_embedding = radial_embedding

        # === angular basis ===
        norm = angular_basis.get('norm', True)
        angular_type = angular_basis.get('type', 'ictd')
        self.angular_embedding = LegacyCartesianHarmonics2(max(max_r_2), norm, angular_type != 'gto')

        # === Interaction Layer ===
        self.interactions = nn.ModuleList(
            [
                Interaction(
                    atomic_numbers,
                    num_channel,
                    num_channel_hidden,
                    max(rank_of_in[idx]),
                    rank_of_out[idx],
                    max(rank_of_hidden[idx]),
                    avg_num_neighbors,
                    radial_embedding.out_dim,
                    radial_mlp,
                    inter,
                    ictd,
                    bias,
                    layer=idx,
                    num_layers=num_layers,
                )
                for idx in range(num_layers)
            ]
        )

        # === Product Layer ===
        self.products = nn.ModuleList(
            [
                SelfContraction(
                    num_channel,
                    num_channel_hidden,
                    max(rank_of_hidden[idx]),
                    rank_of_out[idx],
                    atomic_numbers,
                    prod,
                    ictd,
                    bias,
                    idx,
                    num_layers,
                )
                for idx in range(num_layers)
            ]
        )
 
    def forward(self, data: Dict[str, Tensor], graph: Graph) -> Dict[str, Any]:

        lmp = graph.lmp
        nlocal, _ = graph.lmp_natoms
        edge_vector = graph.edge_vector
        edge_length = graph.edge_length

        # === radial and angular ===
        edge_feats, cutoff = self.radial_embedding(
            edge_length,
            data['node_attrs'],
            data['edge_index'],
            self.atomic_numbers,
        )
        edge_attrs = {}
        normed_edge_vector = edge_vector / edge_length
        edge_attrs = self.angular_embedding(normed_edge_vector)

        # === node initialize (element and uie) ===
        node_feats = {0: self.node_embedding(data['node_attrs'])}
        uie_feats = None
        if hasattr(self, "uie_embedding"):
            uie_data = {}
            for k, _ in self.invariant_embeddings.items():
                p = k
                uie_data.update({p: data[p]})
            uie_feats = self.uie_embedding(data["batch"], uie_data)
            node_feats[0] = node_feats[0] + uie_feats

        # === representation Learning ===
        descriptors = []
        for idx, (inter, prod) in enumerate(zip(self.interactions, self.products)):
            node_attrs_lmp = data['node_attrs']
            if lmp and idx > 0:
                node_attrs_lmp = node_attrs_lmp[:nlocal]
            node_feats, sc = inter(
                node_feats,
                data['node_attrs'], 
                node_attrs_lmp, 
                edge_feats, 
                edge_attrs, 
                data['edge_index'],
                cutoff,
                graph,
            )
            if hasattr(self, 'uee_embeddings'):
                node_feats = self.uee_embeddings[idx](node_feats, data)
            if lmp and idx == 0:
                node_attrs_lmp = node_attrs_lmp[:nlocal]
            node_feats = prod(node_feats, node_attrs_lmp, sc)
            descriptors.append(node_feats)

        return {
            "uie_feats": uie_feats,
            "descriptors": descriptors,
        }

