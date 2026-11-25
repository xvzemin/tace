"""Single-rank ML-IAP execution of the periodic OAM-L force graph."""

from pathlib import Path

import torch

from tace.interface.lammps.mliap import TACEAOTILammpsCalc

ENERGY_GLOBAL = 0x01
ENERGY_ATOM = 0x02


def replay_edge_capacity(edges):
    """Return the number of edge rows allocated for a force-graph capture."""
    return ((edges + max(8, edges // 50) + 63) // 64) * 64


class OAMLammpsCalc(TACEAOTILammpsCalc):
    """Evaluate a local-owner force package on a single LAMMPS rank."""

    local_owners = True
    graph_max_atoms = 0
    graph_max_edges = 65_536
    compact_package = None

    @classmethod
    def from_packages(cls, streamed, compact, atomic_numbers, cutoff, dtype):
        """Package two frozen force graphs and select one at LAMMPS initialization."""
        calc = cls(streamed, atomic_numbers, cutoff, dtype)
        compact = Path(compact)
        calc.compact_package = (compact.name, compact.read_bytes())
        calc.graph_max_atoms = 1024
        return calc

    def _initialize_device(self, data):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() != 1:
            raise RuntimeError("The OAM-L local-owner package requires one MPI rank.")
        if self.compact_package is not None:
            device = torch.as_tensor(data.elems).device
            free, _ = torch.cuda.mem_get_info(device)
            # Budget includes five live radial states, node states, force VJP and
            # allocator headroom. Only the selected package is loaded on the GPU.
            if data.npairs * 12288 + (2 << 30) > free:
                self.package_filename, self.package_bytes = self.compact_package
            print(f"EQX OAM-L force package: {self.package_filename}", flush=True)
        from . import radial  # noqa: F401  (older benchmark packages omit the registration entry)

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
        if n == 0 or (n == 1 and e == 0):
            # For exactly one edgeless atom the released checkpoint disables
            # the learned scale/shift. Adding a disconnected atom would change
            # that physical energy. The legacy PT2 ABI has no isolated-energy
            # table and cannot evaluate the unpadded shape.
            raise ValueError(
                "OAM-L ML-IAP requires a nonempty system other than a single "
                "isolated atom; use eager TACE for its isolated-atom energy."
            )
        # Both node and edge dimensions of the frozen program require >=2.
        # The disconnected replay row preserves all real energies for N>1 or
        # E>0, including larger sparse systems that normally bypass replay.
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
                self.force_graph = _ForceReplay(self.model, batch, n, e, self.device)
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


class _ForceReplay:
    def __init__(self, model, batch, n, e, device):
        from ....ops.common.cuda.graph import retain_constants

        self.constants = {}
        self.n = n
        self.capacity = replay_edge_capacity(e)
        self.vectors = torch.empty(
            (self.capacity, 3), device=device, dtype=torch.float64
        )
        self.indices = torch.empty((2, self.capacity), device=device, dtype=torch.long)
        self.types = torch.zeros(n + 1, device=device, dtype=torch.long)
        static = dict(
            edge_vector=self.vectors,
            node_type=self.types,
            edge_index=self.indices,
            batch=self.types.new_zeros(n + 1),
            ptr=self.types.new_tensor([0, n + 1]),
            fidelity_idx=self.types.new_tensor([model.fidelity_idx]),
        )
        self.copy(batch)
        self.args = tuple(static[key] for key in model.input_keys)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), retain_constants(self.constants):
            model.compiled_model(*self.args)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with (
            retain_constants(self.constants),
            torch.cuda.graph(
                self.graph, stream=stream, capture_error_mode="thread_local"
            ),
        ):
            self.energy, self.energies, self.forces = model.compiled_model(*self.args)

    def copy(self, batch):
        e = batch["edge_vector"].shape[0]
        self.vectors[:e].copy_(batch["edge_vector"].detach())
        self.indices[:, :e].copy_(batch["edge_index"])
        self.types[: self.n].copy_(batch["node_type"])
        # Keep dummy edges entirely in a disconnected, unobserved component.
        self.vectors[e:].zero_()
        self.vectors[e:, 0].fill_(2.0)
        self.indices[:, e:].fill_(self.n)

    def __call__(self, batch):
        self.copy(batch)
        self.graph.replay()
        return self.energies[: self.n], self.forces[: batch["edge_vector"].shape[0]]


class RadialTangentLammpsCalc(OAMLammpsCalc):
    """Single-rank loader for a frozen radial-tangent OAM-L force package."""

    def _initialize_device(self, data):
        from . import tangent  # noqa: F401

        if data.nlocal > self.graph_max_atoms or data.npairs > self.graph_max_edges:
            # Large systems keep the standard AOTI runner. Capture requires
            # the single-runner loader selected during initialization.
            self.graph_max_atoms = 0
        super()._initialize_device(data)
