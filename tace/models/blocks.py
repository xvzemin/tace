################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Optional

import torch

from tace.utils.torch_scatter import scatter_sum


def format_list(obj, ndigits=4):
    if isinstance(obj, int):
        return str(obj)
    elif isinstance(obj, float):
        return f"{obj:.{ndigits}f}"
    elif isinstance(obj, (list, tuple)):
        return "[" + ", ".join(format_list(x, ndigits) for x in obj) + "]"
    else:
        return str(obj)


class OneHotToAtomicEnergy(torch.nn.Module):
    def __init__(
        self,
        atomic_energies: List[Dict[int, float]],
        atomic_numbers: List[int],
    ) -> None:
        super().__init__()
        assert atomic_energies is not None
        atomic_energy_list = []
        for atomic_energy in atomic_energies:
            atomic_energy = {int(z): float(value) for z, value in atomic_energy.items()}
            atomic_energy_list.append(
                [atomic_energy.get(int(z), 0.0) for z in atomic_numbers]
            )
        self.register_buffer(
            "atomic_energy",
            torch.tensor(
                atomic_energy_list,
                dtype=torch.get_default_dtype(),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.atomic_energy.T)

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += "  atomic_energies = {\n"

        data = self.atomic_energy.detach().cpu().numpy()

        for i in range(data.shape[0]):
            s += f"    fidelity {i}: {format_list(data[i].tolist(), 4)}\n"

        s += "  }\n"
        s += ")"
        return s


class ScaleShift(torch.nn.Module):
    def __init__(
        self,
        atomic_numbers: List[int],
        scale_dicts: Optional[List[Dict[int, float]]] = None,
        shift_dicts: Optional[List[Dict[int, float]]] = None,
        scale_trainable: bool = False,
        shift_trainable: bool = False,
        all_atoms: bool = False,
    ):
        super().__init__()

        scale_dicts = [] if scale_dicts is None else scale_dicts
        shift_dicts = [] if shift_dicts is None else shift_dicts
        self.all_atoms = all_atoms
        self.has_scale = len(scale_dicts) > 0
        self.has_shift = len(shift_dicts) > 0
        self.num_fidelities = max(len(scale_dicts), len(shift_dicts))
        atomic_numbers = [int(z) for z in atomic_numbers]
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )

        if self.has_scale:
            scale_list = []
            for d in scale_dicts:
                scale_list.append([d.get(z, 1.0) for z in atomic_numbers])
            scale_tensor = torch.tensor(scale_list, dtype=torch.get_default_dtype())
            if scale_trainable:
                self.scale = torch.nn.Parameter(scale_tensor)
            else:
                self.register_buffer("scale", scale_tensor)

        if self.has_shift:
            shift_list = []
            for d in shift_dicts:
                shift_list.append([d.get(z, 0.0) for z in atomic_numbers])
            shift_tensor = torch.tensor(shift_list, dtype=torch.get_default_dtype())
            if shift_trainable:
                self.shift = torch.nn.Parameter(shift_tensor)
            else:
                self.register_buffer("shift", shift_tensor)

    def forward(self, node_energy, node_attrs, ptr, edge_index, batch, node_fidelity):
        if not (self.has_scale or self.has_shift):
            return node_energy

        num_graphs = ptr.numel() - 1
        num_nodes = ptr[1:] - ptr[:-1]

        if self.all_atoms:
            if self.has_scale:
                node_scale = (node_attrs * self.scale[node_fidelity]).sum(dim=-1)
                node_energy = node_energy * node_scale

            if self.has_shift:
                node_shift = (node_attrs * self.shift[node_fidelity]).sum(dim=-1)
                node_energy = node_energy + node_shift
        else:
            if edge_index.numel() == 0:
                num_edges = torch.zeros(
                    num_graphs, dtype=torch.int64, device=node_energy.device
                )
            else:
                edge_batch = batch[edge_index[1]]
                num_edges = scatter_sum(
                    torch.ones_like(edge_batch),
                    edge_batch,
                    dim=0,
                    dim_size=num_graphs,
                )

            isolated_mask = (num_nodes == 1) & (num_edges == 0)

            if self.has_scale:
                node_scale = (node_attrs * self.scale[node_fidelity]).sum(dim=-1)
                isolated_nodes = isolated_mask[batch]
                node_scale = torch.where(
                    isolated_nodes, torch.zeros_like(node_scale), node_scale
                )
                node_energy = node_energy * node_scale

            if self.has_shift:
                node_shift = (node_attrs * self.shift[node_fidelity]).sum(dim=-1)
                isolated_nodes = isolated_mask[batch]
                node_shift = torch.where(
                    isolated_nodes, torch.zeros_like(node_shift), node_shift
                )
                node_energy = node_energy + node_shift

        return node_energy

    def apply_scale(self, node_energy, node_attrs, node_fidelity):
        """Apply the learned-energy scale without an additive shift."""
        if not self.has_scale:
            return node_energy
        node_scale = (node_attrs * self.scale[node_fidelity]).sum(dim=-1)
        return node_energy * node_scale

    def __repr__(self):

        s = f"{self.__class__.__name__}(\n"
        s += f"  atomic_numbers = {self.atomic_numbers.tolist()}\n"

        if self.has_scale:
            s += "  scale = {\n"
            for lvl in range(self.scale.shape[0]):
                data = self.scale[lvl].detach().cpu().numpy().tolist()
                s += f"    fidelity {lvl}: {format_list(data, 4)}\n"
            s += "  }\n"
        else:
            s += "  scale = None\n"

        if self.has_shift:
            s += "  shift = {\n"
            for lvl in range(self.shift.shape[0]):
                data = self.shift[lvl].detach().cpu().numpy().tolist()
                s += f"    fidelity {lvl}: {format_list(data, 4)}\n"
            s += "  }\n"
        else:
            s += "  shift = None\n"

        s += f"all_atoms={self.all_atoms}\n"
        s += ")"
        return s

    @classmethod
    def build_from_config(
        cls,
        statistics,
        cfg: Dict,
        atomic_numbers: List[int],
    ):
        required_keys = [
            "scale_type",
            "shift_type",
            "scale_trainable",
            "shift_trainable",
        ]
        assert all(k in cfg for k in required_keys), (
            f"Missing keys in scale_shift config: {required_keys}"
        )

        scale_key = cfg["scale_type"]
        shift_key = cfg["shift_type"]

        scale_dicts = []
        shift_dicts = []

        for stats in statistics:
            scale_stat = {z: 1.0 for z in stats["atomic_numbers"]}
            shift_stat = {z: 0.0 for z in stats["atomic_numbers"]}

            if stats.get("available", True):
                if scale_key is not None:
                    assert scale_key in stats, f"{scale_key} not found in statistics"
                    scale_stat = stats[scale_key]

                if shift_key is not None:
                    assert shift_key in stats, f"{shift_key} not found in statistics"
                    shift_stat = stats[shift_key]

            scale_dict = {int(k): float(v) for k, v in scale_stat.items()}
            shift_dict = {int(k): float(v) for k, v in shift_stat.items()}
            scale_dicts.append(scale_dict)
            shift_dicts.append(shift_dict)

        return cls(
            scale_dicts=scale_dicts,
            shift_dicts=shift_dicts,
            scale_trainable=cfg["scale_trainable"],
            shift_trainable=cfg["shift_trainable"],
            all_atoms=cfg["all_atoms"],
            atomic_numbers=atomic_numbers,
        )
