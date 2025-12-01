################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
'''
This interface is general. As long as you have successfully compiled LAMMPS 
with any one of the models ``TACE, MACE, NequIP, or Allegro ...``
then all MLIPs that have an interface with LAMMPS-IAP can be used.
'''


import logging
from typing import Dict, Tuple

import torch
from torch import Tensor
from ase.data import chemical_symbols
from e3nn.util.jit import compile_mode


from lammps.mliap.mliap_unified_abc import MLIAPUnified

# TODO check device 
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
        self.register_buffer("num_layers", model.readout_fn.num_layers)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(
        self, data: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:

        outs = self.model(data)
        node_energy = outs["node_energy"]
        pair_forces = outs["edge_forces"]
        total_energy = outs["energy"][0]

        if pair_forces is None:
            pair_forces = torch.zeros_like(data["edge_vector"])

        return total_energy, node_energy, pair_forces


class LAMMPS_MLIAP_TACE(MLIAPUnified):
    '''Not test for cpu running, only cuda devices are tested by author'''
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = EdgeForcesWrapper(model, **kwargs)
        self.element_types = [chemical_symbols[s] for s in model.readout_fn.atomic_numbers]
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5 * float(model.readout_fn.cutoff)
        self.nparams = 1
        self.ndescriptors = 1

        self.dtype = model.readout_fn.cutoff.dtype
        self.device = "cpu"
        self.initialized = False

    def _initialize_device(self, data):
        device = torch.as_tensor(data.elems).device
        self.device = device
        # self.device = (
        #     "cuda" if "kokkos" in data.__class__.__module__.lower() else "cpu"
        # )
        self.model = self.model.to(device)
        logging.info(f"TACE model initialized on device: {device}")
        self.initialized = True

    def compute_forces(self, data):
        nlocal = data.nlocal
        ntotal = data.ntotal
        npairs = data.npairs
        nghosts = ntotal - nlocal
        species = torch.as_tensor(data.elems, dtype=torch.int64)

        if not self.initialized:
            self._initialize_device(data)

        if nlocal == 0 or npairs <= 1:
            return
            
        batch = self._prepare_batch(data, nlocal, nghosts, species)

        _, node_energy, pair_forces = self.model(batch)
        if self.device.type != "cpu":
            torch.cuda.synchronize()

        self._update_lammps_data(data, node_energy, pair_forces, nlocal)

    def _prepare_batch(self, data, nlocal, nghosts, species):
        edge_vector = torch.as_tensor(data.rij).to(self.dtype).to(self.device)
        edge_vector.requires_grad_(True)
        return {
            "edge_vector": edge_vector,
            "node_attrs": torch.nn.functional.one_hot(
                species.to(self.device), num_classes=self.num_species
            ).to(self.dtype),
            "edge_index": torch.stack(
                [
                    torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
                    torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
                ],
                dim=0,
            ),
            "batch": torch.zeros(nlocal, dtype=torch.int64, device=self.device),
            "ptr": torch.tensor([0, nlocal], dtype=torch.int64, device=self.device),
            "lmp_data": data,
            "natoms": (nlocal, nghosts),
        }


    def _update_lammps_data(self, data, node_energy, pair_forces, nlocal):
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
            node_energy = node_energy.double()
        eatoms = torch.as_tensor(data.eatoms)
        eatoms.copy_(node_energy[:nlocal])
        data.energy = torch.sum(node_energy[:nlocal]).detach()
        # print(torch.sum(node_energy[:natoms]))
        # import sys
        # sys.exit()
        data.update_pair_forces_gpu(pair_forces)

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass