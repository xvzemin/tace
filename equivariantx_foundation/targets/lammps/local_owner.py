"""Single-rank Kokkos ML-IAP protocol for frozen local-owner force programs."""

import torch

from .replay import ForceReplay

ENERGY_GLOBAL = 0x01
ENERGY_ATOM = 0x02


class LocalOwnerForceRunner:
    """Mixin for a host loader that initializes ``device`` and ``model``.

    The program consumes local atom types and sender/receiver indices, with
    explicit displacement vectors in original pair order. It returns energy,
    atomic energies, and pair forces. Padded replay requires that a disconnected
    dummy component leave real outputs unchanged. Each adapter owns its valid
    physical domain and its atom/edge replay limits.
    """

    local_owners = True

    def _initialize_device(self, data):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() != 1:
            raise RuntimeError("The local-owner force package requires one MPI rank.")
        self._load_local_package(data)

    def _load_local_package(self, data):
        self.single_threaded = self.graph_max_atoms > 0
        super()._initialize_device(data)
        self.force_graph = None
        self.initialize_local_inputs(self.device, self.model.fidelity_idx)

    def initialize_local_inputs(self, device, fidelity):
        """Initialize single-rank graph metadata for the selected fidelity."""
        self.node_batch = None
        self.exchange_indices = None
        self.fidelity = torch.tensor([fidelity], device=device, dtype=torch.long)
        self.exported_forces = None

    def compute_forces(self, data):
        n, e = data.nlocal, data.npairs
        if not self.initialized:
            self._initialize_device(data)
        torch.cuda.synchronize(self.device)
        batch = self.local_inputs(data, n, e)
        node_energy, forces = self.local_forces(batch, n, e)
        self.write_local_results(data, node_energy, forces, n)

    def local_forces(self, batch, n, e):
        """Evaluate local-owner energies and pair derivatives without committing forces."""
        if n >= 2 and e >= 2 and (n > self.graph_max_atoms or e > self.graph_max_edges):
            self.force_graph = None
            _, node_energy, forces = self.model(batch)
        else:
            if (
                self.force_graph is None
                or self.force_graph.n != n
                or e > self.force_graph.capacity
            ):
                self.force_graph = None
                self.force_graph = ForceReplay(self.model, batch, n, e, self.device)
            node_energy, forces = self.force_graph(batch)
        return node_energy, forces

    def write_local_results(self, data, node_energy, forces, n):
        """Write pair forces and the requested global and per-atom energies."""
        forces = forces.to(torch.float64).contiguous()
        if data.eflag & ENERGY_ATOM:
            torch.as_tensor(data.eatoms).copy_(node_energy[:n])
        if data.eflag & ENERGY_GLOBAL:
            data.energy = node_energy[:n].sum(dtype=torch.float64).item()
        else:
            torch.cuda.current_stream(self.device).synchronize()
        if forces.numel():
            self.exported_forces = forces
            data.update_pair_forces(forces)

    def local_inputs(self, data, n, e):
        """Build local-owner graph tensors from the current Kokkos pair list."""
        # The entry fence completed the previous Kokkos force scatter.
        self.exported_forces = None
        if self.node_batch is None or self.node_batch.numel() != n:
            self.node_batch = torch.zeros(n, device=self.device, dtype=torch.long)
            self.node_ptr = self.node_batch.new_tensor([0, n])
        types = torch.as_tensor(data.elems)[:n].long()
        index = torch.empty((2, e), dtype=torch.long, device=self.device)
        if e:
            vectors = torch.as_tensor(data.rij)
            source = torch.as_tensor(data.pair_j)
            if data.ntotal != n:
                if (
                    self.exchange_indices is None
                    or self.exchange_indices.numel() != data.ntotal
                ):
                    self.exchange_indices = torch.arange(
                        data.ntotal, device=self.device, dtype=torch.float64
                    )
                    self.exchange_owners = torch.empty_like(self.exchange_indices)
                    torch.cuda.current_stream(self.device).synchronize()
                data.forward_exchange(self.exchange_indices, self.exchange_owners, 1)
                torch.cuda.synchronize(self.device)
                torch.index_select(self.exchange_owners.long(), 0, source, out=index[0])
            else:
                index[0].copy_(source)
            index[1].copy_(torch.as_tensor(data.pair_i))
        else:
            vectors = torch.empty((0, 3), device=self.device, dtype=torch.float64)
        return dict(
            edge_vector=vectors,
            edge_index=index,
            node_type=types,
            batch=self.node_batch,
            ptr=self.node_ptr,
            fidelity_idx=self.fidelity,
        )
