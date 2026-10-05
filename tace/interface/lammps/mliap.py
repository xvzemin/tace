################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import torch
from ase.data import chemical_symbols

from tace.models.lammps import use_lammps_mliap_data

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError as error:
    raise ImportError(
        "The TACE LAMMPS interface requires LAMMPS built with Python ML-IAP support."
    ) from error


class EdgeForcesWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__()
        model.lmp = True
        model.flags.compute_forces = False
        model.flags.compute_stress = False
        model.flags.compute_virials = False
        model.flags.compute_edge_forces = True

        self.model = model
        self.register_buffer("cutoff", model.readout_fn.cutoff)
        self.register_buffer("atomic_numbers", model.readout_fn.atomic_numbers)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        outs = self.model(data)
        node_energy = outs["node_energy"]
        pair_forces = outs["edge_forces"]
        total_energy = outs["energy"][0]

        if pair_forces is None:
            pair_forces = torch.zeros_like(data["edge_vector"])

        return total_energy, node_energy, pair_forces


class TACELammpsCalc(MLIAPUnified):
    """LAMMPS ML-IAP interface for eager TACE models."""

    def __init__(self, model, **kwargs):
        super().__init__()

        self.model = EdgeForcesWrapper(model, **kwargs)
        self.element_types = [
            chemical_symbols[s] for s in model.readout_fn.atomic_numbers
        ]
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5 * float(model.readout_fn.cutoff)
        self.nparams = 1
        self.ndescriptors = 1

        self.dtype = model.readout_fn.cutoff.dtype
        self.device = torch.device("cpu")
        self.initialized = False

    def _initialize_device(self, data):
        device = torch.as_tensor(data.elems).device
        self._validate_lammps_data(data, device)
        self.device = device
        self.model = self.model.to(device)
        logging.info(f"TACE model initialized on device: {device}")
        self.initialized = True

    @staticmethod
    def _validate_lammps_data(data, device):
        exchange_methods = ("forward_exchange", "reverse_exchange")
        if not all(hasattr(data, method) for method in exchange_methods):
            raise RuntimeError(
                "TACE requires the KOKKOS ML-IAP interface for ghost exchange. "
                "Run LAMMPS with '-k on g <Ng> -sf kk'."
            )
        if device.type != "cuda":
            raise RuntimeError(
                "TACE LAMMPS ML-IAP currently requires a CUDA KOKKOS backend. "
                "LAMMPS host-side ML-IAP ghost exchange is not supported."
            )

    def compute_forces(self, data):
        nlocal = data.nlocal
        ntotal = data.ntotal
        nghosts = ntotal - nlocal
        species = torch.as_tensor(data.elems, dtype=torch.int64)

        if not self.initialized:
            self._initialize_device(data)

        batch = self._prepare_batch(data, nlocal, nghosts, species)

        _, node_energy, pair_forces = self.model(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self._update_lammps_data(data, node_energy, pair_forces, nlocal)

    def _prepare_batch(self, data, nlocal, nghosts, species):
        if data.npairs:
            edge_vector = torch.as_tensor(data.rij).to(self.device, self.dtype)
            edge_index = torch.stack(
                [
                    torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
                    torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
                ],
                dim=0,
            )
        else:
            # ML-IAP exposes null edge arrays when there are no neighbors.
            edge_vector = torch.empty(0, 3, dtype=self.dtype, device=self.device)
            edge_index = torch.empty(2, 0, dtype=torch.int64, device=self.device)
        edge_vector.requires_grad_(True)
        return {
            "edge_vector": edge_vector,
            "node_attrs": torch.nn.functional.one_hot(
                species.to(self.device), num_classes=self.num_species
            ).to(self.dtype),
            "edge_index": edge_index,
            "batch": torch.zeros(nlocal, dtype=torch.int64, device=self.device),
            "ptr": torch.tensor([0, nlocal], dtype=torch.int64, device=self.device),
            "lmp_data": data,
            "natoms": (nlocal, nghosts),
        }

    @staticmethod
    def _update_lammps_data(data, node_energy, pair_forces, nlocal):
        pair_forces = pair_forces.to(dtype=torch.float64).contiguous()
        node_energy = node_energy.to(dtype=torch.float64).contiguous()
        eatoms = torch.as_tensor(data.eatoms)
        eatoms.copy_(node_energy[:nlocal])
        data.energy = node_energy[:nlocal].sum().item()
        if pair_forces.numel():
            data.update_pair_forces(pair_forces)

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass


class TACEAOTILammpsCalc(TACELammpsCalc):
    def __init__(
        self,
        package_path: Union[str, Path],
        atomic_numbers: Sequence[int],
        cutoff: float,
        dtype: torch.dtype,
    ) -> None:
        MLIAPUnified.__init__(self)
        package_path = Path(package_path)
        self.package_bytes = package_path.read_bytes()
        self.package_filename = package_path.name
        self.element_types = [chemical_symbols[int(z)] for z in atomic_numbers]
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5 * float(cutoff)
        self.nparams = 1
        self.ndescriptors = 1
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.model = None
        self.initialized = False

    def __getstate__(self):
        state = dict(self.__dict__)
        state["model"] = None
        state["device"] = torch.device("cpu")
        state["initialized"] = False
        return state

    def _initialize_device(self, data):
        from tace.models.compile import load_lammps_aotinductor

        device = torch.as_tensor(data.elems).device
        self._validate_lammps_data(data, device)
        self.device = device
        with tempfile.TemporaryDirectory(prefix="tace-lammps-aoti-") as tmpdir:
            package_path = os.path.join(tmpdir, self.package_filename)
            with open(package_path, "wb") as package_file:
                package_file.write(self.package_bytes)
            self.model = load_lammps_aotinductor(package_path, device)
        logging.info(f"TACE AOTI model initialized on device: {device}")
        self.initialized = True

    def compute_forces(self, data):
        nlocal = data.nlocal
        ntotal = data.ntotal
        nghosts = ntotal - nlocal
        species = torch.as_tensor(data.elems, dtype=torch.int64)

        if not self.initialized:
            self._initialize_device(data)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        batch = self._prepare_batch(data, nlocal, nghosts, species)
        with use_lammps_mliap_data(data):
            _, node_energy, pair_forces = self.model(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self._update_lammps_data(data, node_energy, pair_forces, nlocal)
