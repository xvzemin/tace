################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Dict


import torch
from torch import nn, Tensor


class OneHotToAtomicEnergy(torch.nn.Module):
    def __init__(self, atomic_energies: List[Dict[int, float]]) -> None:
        super().__init__()
        assert atomic_energies is not None
        atomic_energy_list = []
        for atomic_energy in atomic_energies:
            atomic_energy_list.append(
                [float(v) for _, v in atomic_energy.items()]
            )
        self.register_buffer(
            "atomic_energy",
            torch.tensor(
                atomic_energy_list,
                dtype=torch.get_default_dtype(),
            ),
        )

    def forward(self, x: Tensor) -> Tensor: 
        return torch.matmul(x, self.atomic_energy.T)

    def __repr__(self):
        return f"{self.__class__.__name__}(atomic_eneries={[f'{x:.4f}' for x in self.atomic_energy.reshape(-1).tolist()]})"


class ScaleShift(torch.nn.Module):
    def __init__(
        self,
        scale_dicts: List[Dict[int, float]] = [],
        shift_dicts: List[Dict[int, float]] = [],
        scale_trainable: bool = False,
        shift_trainable: bool = False,
    ):
        super().__init__()

        self.has_scale = len(scale_dicts) > 0
        self.has_shift = len(shift_dicts) > 0
        self.num_levels = max(len(scale_dicts), len(shift_dicts))
        atomic_numbers = sorted(
            set().union(*[d.keys() for d in scale_dicts] if scale_dicts else [])
            | set().union(*[d.keys() for d in shift_dicts] if shift_dicts else [])
        )
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )

        if self.has_scale:
            scale_list = []
            for d in scale_dicts:
                scale_list.append([d.get(z, 1.0) for z in atomic_numbers])
            scale_tensor = torch.tensor(scale_list, dtype=torch.get_default_dtype())
            if scale_trainable:
                self.scale = nn.Parameter(scale_tensor)
            else:
                self.register_buffer("scale", scale_tensor)

        if self.has_shift:
            shift_list = []
            for d in shift_dicts:
                shift_list.append([d.get(z, 0.0) for z in atomic_numbers])
            shift_tensor = torch.tensor(shift_list, dtype=torch.get_default_dtype())
            if shift_trainable:
                self.shift = nn.Parameter(shift_tensor)
            else:
                self.register_buffer("shift", shift_tensor)

    def forward(self, node_energy, node_attrs, ptr, edge_index, batch, node_level):
        if not (self.has_scale or self.has_shift):
            return node_energy

        num_graphs = ptr.numel() - 1
        num_nodes = ptr[1:] - ptr[:-1]

        if edge_index.numel() == 0:
            num_edges = torch.zeros(num_graphs, dtype=torch.int64, device=node_energy.device)
        else:
            edge_batch = batch[edge_index[1]]
            num_edges = torch.bincount(edge_batch, minlength=num_graphs)

        isolated_mask = (num_nodes == 1) & (num_edges == 0)

        if self.has_scale:
            node_scale = (node_attrs * self.scale[node_level]).sum(dim=-1)
            if isolated_mask.any():
                isolated_nodes = torch.isin(batch, torch.where(isolated_mask)[0])
                node_scale[isolated_nodes] = 0.0
            node_energy = node_energy * node_scale

        if self.has_shift:
            node_shift = (node_attrs * self.shift[node_level]).sum(dim=-1)
            if isolated_mask.any():
                isolated_nodes = torch.isin(batch, torch.where(isolated_mask)[0])
                node_shift[isolated_nodes] = 0.0
            node_energy = node_energy + node_shift

        return node_energy

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += f"  atomic_numbers = {self.atomic_numbers.tolist()}\n"
        if self.has_scale:
            s += "  scale = {\n"
            for lvl in range(self.scale.shape[0]):
                s += f"    level {lvl}: {self.scale[lvl].detach().cpu().numpy().tolist()}\n"
            s += "  }\n"
        else:
            s += "  scale = None\n"

        if self.has_shift:
            s += "  shift = {\n"
            for lvl in range(self.shift.shape[0]):
                s += f"    level {lvl}: {self.shift[lvl].detach().cpu().numpy().tolist()}\n"
            s += "  }\n"
        else:
            s += "  shift = None\n"

        s += ")"
        return s

    @classmethod
    def build_from_config(cls, statistics, cfg: Dict):
        required_keys = [
            "scale_type",
            "shift_type",
            "scale_trainable",
            "shift_trainable",
            "scale_dict",
            "shift_dict",
        ]
        assert all(
            k in cfg for k in required_keys
        ), f"Missing keys in scale_shift config: {required_keys}"

        scale_key = cfg["scale_type"]
        shift_key = cfg["shift_type"]

        scale_dicts = []
        shift_dicts = []

        for stats in statistics:
            scale_stat = {z: 1.0 for z in stats["atomic_numbers"]}
            shift_stat = {z: 0.0 for z in stats["atomic_numbers"]}

            if scale_key is not None:
                assert hasattr(stats, scale_key), f"{scale_key} not found in statistics"
                scale_stat = getattr(stats, scale_key, scale_stat)

            if shift_key is not None:
                assert hasattr(stats, shift_key), f"{shift_key} not found in statistics"
                shift_stat = getattr(stats, shift_key, shift_stat)

            scale_dict = {int(k): float(v) for k, v in scale_stat.items()}
            shift_dict = {int(k): float(v) for k, v in shift_stat.items()}

            if isinstance(cfg["scale_dict"], list):
                if len(cfg["scale_dict"]) > len(scale_dicts):
                    for k, v in cfg["scale_dict"][len(scale_dicts)].items():
                        k = int(k)
                        assert k in scale_dict, f"Invalid atomic number {k} in scale_dict"
                        scale_dict[k] = float(v)

            if isinstance(cfg["shift_dict"], list):
                if len(cfg["shift_dict"]) > len(shift_dicts):
                    for k, v in cfg["shift_dict"][len(shift_dicts)].items():
                        k = int(k)
                        assert k in shift_dict, f"Invalid atomic number {k} in shift_dict"
                        shift_dict[k] = float(v)

            scale_dicts.append(scale_dict)
            shift_dicts.append(shift_dict)

        return cls(
            scale_dicts=scale_dicts,
            shift_dicts=shift_dicts,
            scale_trainable=cfg["scale_trainable"],
            shift_trainable=cfg["shift_trainable"],
        )

from e3nn import o3


class OneHotEmbedding(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ) -> None:
        """Not use in TACE"""
        super().__init__()
        self.node_attrs = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        self.num_channel = irreps_out.dim

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.node_attrs(x)
