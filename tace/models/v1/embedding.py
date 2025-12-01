################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Dict, Union


import torch
from torch import nn, Tensor


from .act import ACT
from .utils import expand_dims_to


class UniversalInvariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_channel: int,
        invariant_embeddings: List[Dict[str, Union[int, str]]],
    ):
        super().__init__()

        self.invariant_embeddings = invariant_embeddings
        self.num_channel = num_channel
        self.embedding = nn.ModuleDict()

        total_dim = 0
        for k, v in invariant_embeddings.items():
            p = k
            type_ = v["type"]
            in_dim = v["in_dim"]
            out_dim = v["out_dim"]

            if type_ == "discrete":
                self.embedding[p] = nn.Embedding(v["num_classes"], out_dim)
            elif type_ == "continuous":
                act = v["act"]
                bias = v["bias"]
                self.embedding[p] = nn.Sequential(
                    nn.Linear(in_dim, out_dim, bias=bias),
                    ACT[act](),
                    nn.Linear(out_dim, out_dim, bias=bias),
                )
            total_dim += out_dim

        self.project = nn.Sequential(
            nn.Linear(total_dim, num_channel, bias=False),
            nn.SiLU(),
        )

    def forward(
        self,
        batch: Tensor,
        attrs: Dict[str, Tensor],
    ) -> Tensor:
        embeddings = []

        for p, _ in self.embedding.items():
            ie_info = self.invariant_embeddings[p]
            _type = ie_info['type']
            per = ie_info["per"]
            in_dim = ie_info["in_dim"]
            attr = attrs[p]

            if per == "graph":
                attr = attr[batch]

            if _type == 'continuous' and in_dim == 1:
                attr = attr.unsqueeze(-1)
    
            embedding = self.embedding[p](attr)
            embeddings.append(embedding)

        return self.project(torch.cat(embeddings, dim=-1))


def add_rank1_to_left(T: Dict[int, Tensor], rank1: Tensor) -> Dict[int, torch.Tensor]:
    if 1 in T:
        T[1] = T[1] + rank1
    else:
        T[1] = rank1
    return T


def add_rank2_to_left(T: Dict[int, Tensor], rank2: Tensor) -> Dict[int, torch.Tensor]:
    if 2 in T:
        T[2] = T[2] + rank2
    else:
        T[2] = rank2
    return T


def add_rank3_to_left(T: Dict[int, Tensor], rank3: Tensor) -> Dict[int, torch.Tensor]:
    if 3 in T:
        T[3] = T[3] + rank3
    else:
        T[3] = rank3
    return T


ADD_FN = {
    1: add_rank1_to_left,
    2: add_rank2_to_left,
    3: add_rank3_to_left,
}


class EquivariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        p: str,
        rank: int,
        per: str,
        atomic_numbers: List,
        num_channel: int,
        element_trainable: bool = True,
        channel_trainable: bool = True,
    ):
        super().__init__()
        num_elements = len(atomic_numbers)
        if element_trainable:
            self.element_weights = nn.Parameter(
                torch.ones(num_elements, dtype=torch.get_default_dtype())
            )
        else:
            self.register_buffer(
                "element_weights",
                torch.ones(num_elements, dtype=torch.get_default_dtype()),
            )
        if channel_trainable:
            self.channel_weights = nn.Parameter(
                torch.ones(num_channel, dtype=torch.get_default_dtype())
            )
        else:
            self.register_buffer(
                "channel_weights",
                torch.ones(num_channel, dtype=torch.get_default_dtype()),
            )
        self.p = p
        self.add_fn = ADD_FN[rank]
        self.per = per
        self.rank = rank

    def forward(
        self,
        batch: Tensor,
        node_feats: Dict[int, Tensor],
        node_attrs: Tensor,
        data: Dict[str, Tensor],
    ):
        element_idx = torch.argmax(node_attrs, dim=-1)
        label = data[self.p]
        if self.per == "graph":
            label = label[batch].unsqueeze(1)
        else:
            label = label.unsqueeze(1)

        element_weights = expand_dims_to(
            self.element_weights[element_idx], n_dim=self.rank + 2
        )
        shape = (1, -1) + (1,) * self.rank
        channel_weights = self.channel_weights.view(*shape)
        
        embedding = label * element_weights * channel_weights
        return self.add_fn(node_feats, embedding)
  

class UniversalEquivariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        equivariant_embeddings: List[Dict[str, Union[int, str]]],
        atomic_numbers: List,
        num_channel: int,
    ):
        super().__init__()

        self.equivariant_embeddings = equivariant_embeddings
        self.embeddings = nn.ModuleDict()
        for k, v in equivariant_embeddings.items():
            p = k
            per = v["per"]
            rank = v["rank"]
            element_trainable = v["element_trainable"]
            channel_trainable = v["channel_trainable"]

            self.embeddings[p] = EquivariantEmbedding(
                p,
                rank,
                per,
                atomic_numbers,
                num_channel,
                element_trainable,
                channel_trainable,
            )

    def forward(
        self,
        node_feats,
        data: Dict[str, Tensor],
    ) -> Tensor:
        batch = data["batch"]
        node_attrs = data["node_attrs"]
        for p, _ in self.equivariant_embeddings.items():
            node_feats = self.embeddings[p](batch, node_feats, node_attrs, data)
        return node_feats
