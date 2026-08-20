from typing import Dict, Sequence, Union

import torch

from ..adapter import TensorModel
from ..lammps import AOTI_LAMMPS_GHOST_EXCHANGE, Graph
from ..utils import compute_symmetric_displacement
from .compile import compiled_call, trace_and_compile


class CompileTensorModel(TensorModel):
    allowed_properties = frozenset(
        {
            "energy",
            "forces",
            "stress",
            "virials",
            "direct_forces",
            "direct_stress",
            "direct_virials",
            "noncollinear_magnetic_forces",
            "charges",
        }
    )

    def __init__(self, readout_fn: torch.nn.Module) -> None:
        self._validate_compile_properties(readout_fn)
        object.__setattr__(self, "_compiled_cache", {})
        super().__init__(readout_fn)

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, None]]:
        if self._should_compile(data):
            return self._compiled_forward(data)
        return super().forward(data)

    def reset_target_property(self, target_property: list[str]) -> None:
        self._validate_target_property(target_property)
        self._compiled_cache.clear()
        super().reset_target_property(target_property)

    def reset_fidelity_idx(self, fidelity_idx: Union[int, None] = 0) -> None:
        if hasattr(self, "_compiled_cache"):
            self._compiled_cache.clear()
        super().reset_fidelity_idx(fidelity_idx)

    def _apply(self, fn, recurse: bool = True):
        self._compiled_cache.clear()
        return super()._apply(fn, recurse)

    def __getstate__(self):
        state = super().__getstate__()
        state["_compiled_cache"] = {}
        return state

    def _should_compile(self, data: Dict[str, torch.Tensor]) -> bool:
        if self.lmp:
            return False
        valid_graph = data["positions"].shape[0] > 1 and data["edge_index"].shape[1] > 1
        if not valid_graph:
            return False
        return not self.training or data["ptr"].shape[0] > 2

    def _compiled_forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, None]]:
        input_keys = self._input_keys(data)
        output_keys = self._output_keys()
        cache_key = (self.training, input_keys, output_keys)
        cache = self._compiled_cache
        if cache_key not in cache:
            flat_model = _FlatE3nnCompileModel(self, input_keys, output_keys)
            inputs = tuple(data[key] for key in input_keys)
            cache[cache_key] = (
                flat_model,
                *trace_and_compile(
                    flat_model,
                    inputs,
                    backend=getattr(self.readout_fn, "compile_backend", "inductor"),
                ),
            )

        flat_model, compiled, parameter_names, buffer_names = cache[cache_key]
        outputs = compiled_call(
            compiled,
            flat_model,
            (data[key] for key in input_keys),
            parameter_names,
            buffer_names,
        )
        result: Dict[str, Union[torch.Tensor, None]] = {
            "energy": None,
            "node_energy": None,
            "forces": None,
            "virials": None,
            "stress": None,
            "direct_forces": None,
            "direct_virials": None,
            "direct_stress": None,
            "noncollinear_magnetic_forces": None,
            "charges": None,
        }
        result.update(zip(output_keys, outputs))
        return result

    def _input_keys(self, data: Dict[str, torch.Tensor]) -> tuple[str, ...]:
        required = [
            "positions",
            "node_attrs",
            "edge_index",
            "edge_shifts",
            "lattice",
            "batch",
            "ptr",
        ]
        if "fidelity_idx" in data:
            required.append("fidelity_idx")
        if self._requires_noncollinear_magmoms():
            required.append("initial_noncollinear_magmoms")
        if self._requires_total_charge():
            required.append("total_charge")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"missing e3nn compile inputs: {missing}")
        return tuple(required)

    def _output_keys(self) -> tuple[str, ...]:
        target_property = self.get_target_property()
        keys = []
        if "energy" in target_property:
            keys.extend(["energy", "node_energy"])
        keys.extend(
            key
            for key in ("direct_forces", "direct_virials", "direct_stress")
            if key in target_property
        )
        keys.extend(
            key for key in ("forces", "virials", "stress") if key in target_property
        )
        if "noncollinear_magnetic_forces" in target_property:
            keys.append("noncollinear_magnetic_forces")
        if "charges" in target_property:
            keys.append("charges")
        return tuple(keys)

    def _requires_noncollinear_magmoms(self) -> bool:
        return (
            "initial_noncollinear_magmoms" in self.get_embedding_property()
            or "noncollinear_magnetic_forces" in self.get_target_property()
        )

    def _requires_total_charge(self) -> bool:
        return "charges" in self.get_target_property()

    @classmethod
    def _validate_compile_properties(cls, readout_fn: torch.nn.Module) -> None:
        if hasattr(readout_fn, "les"):
            raise ValueError("TACE_USE_COMPILE does not support LES.")
        long_range = getattr(readout_fn, "model_config", {}).get("long_range", {})
        les_cfg = long_range.get("les", {})
        if les_cfg.get("enable", False):
            raise ValueError("TACE_USE_COMPILE does not support LES.")
        cls._validate_target_property(readout_fn.target_property)

    @classmethod
    def _validate_target_property(cls, target_property: list[str]) -> None:
        invalid = set(target_property) - cls.allowed_properties
        if invalid:
            raise ValueError(
                "TACE_USE_COMPILE only supports energy, direct_forces, "
                "direct_stress, direct_virials, forces, stress, virials and "
                "noncollinear_magnetic_forces, and charges; "
                f"got {sorted(invalid)}"
            )
        if {
            "forces",
            "stress",
            "virials",
            "noncollinear_magnetic_forces",
        } & set(target_property):
            if "energy" not in target_property:
                raise ValueError(
                    "TACE_USE_COMPILE requires energy for forces, stress, virials, "
                    "and noncollinear magnetic forces."
                )


class _FlatE3nnCompileModel(torch.nn.Module):
    def __init__(
        self,
        model: CompileTensorModel,
        input_keys: Sequence[str],
        output_keys: Sequence[str],
    ) -> None:
        super().__init__()
        self.readout_fn = model.readout_fn
        self.input_keys = tuple(input_keys)
        self.output_keys = tuple(output_keys)
        self.fidelity_idx = model.fidelity_idx
        self.compute_forces = model.flags.compute_forces
        self.compute_stress = model.flags.compute_stress
        self.compute_virials = model.flags.compute_virials
        self.compute_noncollinear_magnetic_forces = (
            model.flags.compute_noncollinear_magnetic_forces
        )
        self.training_mode = model.training

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        data = {key: value for key, value in zip(self.input_keys, args)}
        graph = self._prepare_graph(data)
        output = self.readout_fn(data, graph)
        if (
            self.compute_forces
            or self.compute_stress
            or self.compute_virials
            or self.compute_noncollinear_magnetic_forces
        ):
            output.update(self._first_derivatives(data, graph, output["energy"]))
        return tuple(output[key] for key in self.output_keys)

    def _prepare_graph(self, data: Dict[str, torch.Tensor]) -> Graph:
        node_fidelity = (
            data["fidelity_idx"][data["batch"]]
            if "fidelity_idx" in data
            else torch.full_like(data["batch"], self.fidelity_idx, dtype=torch.int64)
        )
        num_graphs = data["ptr"].numel() - 1
        data["positions"].requires_grad_(self.compute_forces)
        if self.compute_noncollinear_magnetic_forces:
            data["initial_noncollinear_magmoms"].requires_grad_(True)
        displacement = (
            compute_symmetric_displacement(data, num_graphs)
            if self.compute_stress or self.compute_virials
            else None
        )
        source = data["edge_index"][0]
        target = data["edge_index"][1]
        edge_batch = data["batch"][source]
        edge_vector = (
            data["positions"][target]
            - data["positions"][source]
            + torch.einsum(
                "ni,nij->nj", data["edge_shifts"], data["lattice"][edge_batch]
            )
        )
        edge_length = (edge_vector**2).sum(dim=1, keepdim=True).sqrt() + 1e-9
        num_atoms_arange = torch.arange(
            data["positions"].shape[0],
            device=data["positions"].device,
            dtype=torch.int64,
        )
        return Graph(
            lmp=False,
            lmp_data=None,
            lmp_natoms=(data["positions"].size(0), 0),
            num_graphs=num_graphs,
            displacement=displacement,
            positions=data["positions"],
            edge_vector=edge_vector,
            edge_length=edge_length,
            lattice=data["lattice"],
            node_fidelity=node_fidelity,
            num_atoms_arange=num_atoms_arange,
        )

    def _first_derivatives(
        self,
        data: Dict[str, torch.Tensor],
        graph: Graph,
        energy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        inputs = []
        if self.compute_forces:
            inputs.append(graph.positions)
        if self.compute_stress or self.compute_virials:
            inputs.append(graph.displacement)
        if self.compute_noncollinear_magnetic_forces:
            inputs.append(data["initial_noncollinear_magmoms"])
        if not inputs:
            return {}

        grads = torch.autograd.grad(
            outputs=energy,
            inputs=inputs,
            grad_outputs=torch.ones_like(energy),
            retain_graph=self.training_mode,
            create_graph=self.training_mode,
            allow_unused=True,
        )

        output: Dict[str, torch.Tensor] = {}
        grad_index = 0
        if self.compute_forces:
            grad = grads[grad_index]
            output["forces"] = (
                torch.zeros_like(graph.positions) if grad is None else -grad
            )
            grad_index += 1
        if self.compute_stress or self.compute_virials:
            grad = grads[grad_index]
            virials = torch.zeros_like(data["lattice"]) if grad is None else -grad
            volume = torch.linalg.det(data["lattice"]).abs().unsqueeze(-1)
            stress = -virials / volume.view(-1, 1, 1)
            output["virials"] = virials
            output["stress"] = torch.where(
                torch.abs(stress) < 1e10, stress, torch.zeros_like(stress)
            )
            grad_index += 1
        if self.compute_noncollinear_magnetic_forces:
            grad = grads[grad_index]
            magmoms = data["initial_noncollinear_magmoms"]
            output["noncollinear_magnetic_forces"] = (
                torch.zeros_like(magmoms) if grad is None else -grad
            )
        return output


class _FlatE3nnLammpsCompileModel(torch.nn.Module):
    def __init__(
        self,
        model: CompileTensorModel,
        input_keys: Sequence[str],
    ) -> None:
        super().__init__()
        if "energy" not in model.get_target_property():
            raise ValueError("LAMMPS AOTI export requires an energy model.")
        self.readout_fn = model.readout_fn
        self.input_keys = tuple(input_keys)
        self.fidelity_idx = model.fidelity_idx

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        data = {key: value for key, value in zip(self.input_keys, args)}
        graph = self._prepare_graph(data)
        output = self.readout_fn(data, graph)
        pair_forces = torch.autograd.grad(
            outputs=output["energy"],
            inputs=graph.edge_vector,
            grad_outputs=torch.ones_like(output["energy"]),
            retain_graph=False,
            create_graph=False,
        )[0]
        if pair_forces is None:
            pair_forces = torch.zeros_like(graph.edge_vector)
        return output["energy"][0], output["node_energy"], pair_forces

    def _prepare_graph(self, data: Dict[str, torch.Tensor]) -> Graph:
        node_fidelity = (
            data["fidelity_idx"][data["batch"]]
            if "fidelity_idx" in data
            else torch.full_like(data["batch"], self.fidelity_idx, dtype=torch.int64)
        )
        nlocal = data["batch"].size(0)
        nghosts = data["node_attrs"].size(0) - nlocal
        dtype = data["node_attrs"].dtype
        device = data["node_attrs"].device
        positions = torch.zeros((nlocal, 3), dtype=dtype, device=device)
        edge_vector = data["edge_vector"].requires_grad_(True)
        edge_length = (edge_vector**2).sum(dim=1, keepdim=True).sqrt() + 1e-9
        return Graph(
            lmp=True,
            lmp_data=AOTI_LAMMPS_GHOST_EXCHANGE,
            lmp_natoms=(nlocal, nghosts),
            num_graphs=2,
            displacement=None,
            positions=positions,
            edge_vector=edge_vector,
            edge_length=edge_length,
            lattice=torch.zeros((2, 3, 3), dtype=dtype, device=device),
            node_fidelity=node_fidelity,
            num_atoms_arange=torch.arange(nlocal, device=device, dtype=torch.int64),
        )
