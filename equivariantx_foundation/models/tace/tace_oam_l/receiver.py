"""Layerwise receiver tiles for frozen single-rank TACE-OAM-L inference."""

import io

import torch

from equivariantx_foundation.execution.domains import receiver_tiles
from equivariantx_foundation.execution.receiver import is_cuda_allocation_error
from equivariantx_foundation.models.tace.affine_receiver import AffineOAMReceiver
from equivariantx_foundation.models.tace.tace_oam_l.angular import edge_attributes
from equivariantx_foundation.models.tace.tace_oam_l.convolution import MessageTranspose
from equivariantx_foundation.models.tace.tace_oam_l.lammps import OAMLammpsCalc
from equivariantx_foundation.models.tace.tace_oam_l.radial import (
    prepare_radial,
    radial_heads,
)


class ReceiverTiles(AffineOAMReceiver):
    """OAM-L affine receiver with OAM-L radial geometry and domain memory policy."""

    receiver_label = "OAM-L"
    tangent_embedding = None
    domain_tangent_embedding = None
    recompute_domains = False
    domain_tile_limit = 4096
    prepare_radial = staticmethod(prepare_radial)

    def __init__(self, model, tile, retain_messages=False):
        from .model import _OAMLRepresentation

        super().__init__(model, tile, retain_messages)
        self.plan_domains = isinstance(self.rep, _OAMLRepresentation)
        self.coefficient_groups = {}

    def discard_tile_plan(self):
        self.coefficient_groups.clear()

    def plan(self, types, index):
        tiles = []
        self.coefficient_groups = {}
        group_rows = (
            types.is_cuda
            and self.retain_messages
            and torch.version.hip is None
            and self.rep.model_dtype == torch.float32
        )
        for tile in receiver_tiles(types, index, self.tile):
            tiles.append(tile)
            local, sources = tile[0], tile[4]
            if group_rows and local.stop - local.start >= 512 and len(sources):
                from .element_grouped import grouping

                self.coefficient_groups[local.start] = grouping(
                    types[local], self.rep.num_elements
                )
        return tiles

    def edge_vjp(self, i, tile, vec, xs, message_grad, density_grad, metadata):
        weight = self.tangent_embedding
        if self.energy_domains is not None:
            weight = self.domain_tangent_embedding
            if weight is not None and self.used_domain_tile != 4096:
                weight = None
        if weight is None or vec.dtype != torch.float64:
            return super().edge_vjp(
                i, tile, vec, xs, message_grad, density_grad, metadata
            )
        from .tangent import receiver_pullback

        inter, params = self.rep.interactions[i], self.radial[i]
        conv = inter.rejector
        idx, s = tile[3:5]
        types, batch = tile[-2:]
        inputs = inter.linear_up(xs) if i else self.first[types[len(batch) :]]
        x = conv.reshape_in(inputs).contiguous()
        dx, dv = receiver_pullback(
            self.rep,
            weight,
            params,
            metadata,
            x,
            bool(i),
            vec,
            idx,
            s,
            types,
            message_grad.contiguous(),
            density_grad,
        )
        return x, dx, dv

    def edge_operands(self, i, tile, vec, xs):
        local, nodes, edges, idx, s, z, batch = tile
        n = len(batch)
        rep = self.rep
        inter, params = rep.interactions[i], self.radial[i]
        x = inter.linear_up(xs) if i else self.first[z[n:]]
        length = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        basis, cutoff = rep.radial_basis(length, z, idx, rep.atomic_numbers)
        feats = rep.edge_embedding(None, z, basis.to(rep.model_dtype), idx, None)
        attr = edge_attributes(vec, length, cutoff, rep.model_dtype)
        h, density = radial_heads(params, feats, z[idx[0]], z[idx[1]])
        return inter.rejector.reshape_in(x), attr, h, density

    def update(self, i, tile, y, xc):
        z, batch = tile[-2:]
        types = z[: len(batch)]
        groups = self.coefficient_groups.get(tile[0].start)
        inter, product = self.rep.interactions[i], self.rep.products[i]
        sc = inter.resnetBB(xc, types, groups) if i else None
        return product(
            inter.linear_nonlinearity(inter.nonlinearity(y)), types, sc, batch, groups
        )

    def domain_count(self, count, available):
        """Size an initial power-of-two partition from the resident node tensors."""
        peak = self.domain_state_bytes([count] * len(self.readout_widths))
        # Leave room for halo expansion, the convolution workspace and fragmentation.
        budget = max(1, available // 4)
        required = (peak + budget - 1) // budget
        return min(count, 1 << (max(1, required) - 1).bit_length())

    def domain_tape(self, groups):
        """Choose reduced per-node convolution outputs to keep for one domain."""
        from equivariantx_foundation.execution.domains import available_bytes

        tape = [None] * len(groups)
        if self.recompute_domains or not self.plan_domains or not self.first.is_cuda:
            return tape
        stride = self.first.element_size()
        peak = self.domain_state_bytes([len(nodes) for nodes in groups])
        headroom = available_bytes(self.first.device)
        budget = max(0, int(0.75 * headroom) - peak - (2 << 30))
        for i in reversed(range(len(groups))):
            width = self.rep.interactions[i].linear_down.irreps_out.dim
            size = len(groups[i]) * (width + 1) * stride
            if size <= budget:
                # Keep the actual tile outputs, without an extra dense copy.
                tape[i] = []
                budget -= size
        if any(x is not None for x in tape) and not self.message_transposes:
            self.message_transposes = [
                MessageTranspose(i.linear_down, i.rejector.irreps_out)
                for i in self.rep.interactions
            ]
        return tape

    def domain_tile(self, groups, tape, max_degree, available):
        """Select a receiver block width for one domain's forward and reverse passes."""
        sizes = [len(g) for g in groups]
        retained = (
            sum(
                n * (inter.linear_down.irreps_out.dim + 1)
                for n, keep, inter in zip(
                    sizes, tape, self.rep.interactions, strict=True
                )
                if keep is not None
            )
            * self.first.element_size()
        )
        # Reserve live node states, the chosen tape and allocator headroom before
        # budgeting the local edge workspace against the largest receiver degree.
        budget = (
            3 * available // 4 - self.domain_state_bytes(sizes) - retained - (2 << 30)
        )
        limit = min(
            self.domain_tile_limit, max(sizes), budget // (max(1, max_degree) << 14)
        )
        return 1 << (max(1024, limit).bit_length() - 1)


class CapacityOAMLammpsCalc(OAMLammpsCalc):
    """Select the compiled or receiver-tiled force graph on one CUDA device."""

    model_label = "OAM-L"
    receiver_tangent = False
    domain_receiver_tangent = False

    def attach_receiver(self, model):
        """Embed the frozen model used by the receiver-tiled evaluator."""
        from ..foundation import frozen_export_copy
        from .model import optimize_oam_l

        prepared = optimize_oam_l(model, fused_geometry=True, fused_projection=True)
        return self.attach_frozen_receiver(frozen_export_copy(prepared))

    def attach_frozen_receiver(self, frozen):
        """Embed an inference model for the receiver-tiled backend."""
        buffer = io.BytesIO()
        torch.save(frozen, buffer)
        self.receiver_bytes = buffer.getvalue()
        return self

    @staticmethod
    def build_receiver(model, *, tile, retain_messages):
        """Create the layer evaluator for a local-owner tile."""
        return ReceiverTiles(model, tile=tile, retain_messages=retain_messages)

    def needs_tiles(self, data):
        return (
            data.npairs * 6144 + data.nlocal * 160000 + (2 << 30)
            > self.workspace_budget
        )

    def _initialize_device(self, data):
        self.device = torch.as_tensor(data.elems).device
        self.workspace_budget, _ = torch.cuda.mem_get_info(self.device)
        if not self.needs_tiles(data):
            self.receiver = None
            return super()._initialize_device(data)
        self.start_receiver(data)

    def start_receiver(self, data):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() != 1:
            raise RuntimeError("The OAM-L local-owner package requires one MPI rank.")
        self._validate_lammps_data(data, self.device)
        model = torch.load(
            io.BytesIO(self.receiver_bytes),
            map_location=self.device,
            weights_only=False,
        )
        retain = (
            data.nlocal * 110000 + data.npairs * 512 + (4 << 30)
            <= self.workspace_budget
        )
        small_tiles = (
            data.nlocal * 48000 + data.npairs * 224 > self.workspace_budget * 0.8
        )
        self.receiver = self.build_receiver(
            model, tile=1024 if small_tiles else 4096, retain_messages=retain
        )
        normal_tangent = self.receiver_tangent and retain and not small_tiles
        if type(self.receiver) is ReceiverTiles and (
            normal_tangent or self.domain_receiver_tangent
        ):
            from .tangent import radial_embedding

            weight = radial_embedding(self.receiver.rep)
            if normal_tangent:
                self.receiver.tangent_embedding = weight
            if self.domain_receiver_tangent:
                self.receiver.domain_tangent_embedding = weight
        if not self.receiver.retain_messages and data.npairs:
            vectors = torch.as_tensor(data.rij)
            minimum = self.receiver.full_graph_min_bytes(
                data.nlocal, data.npairs, vectors.element_size()
            )
            if minimum > torch.cuda.get_device_properties(self.device).total_memory:
                from equivariantx_foundation.execution.domains import EnergyDomains

                self.receiver.energy_domains = EnergyDomains(data.nlocal)
                print(
                    f"EQX {self.model_label}: exact energy domains (full-graph lower bound "
                    f"{minimum / 2**30:.2f} GiB exceeds device capacity)",
                    flush=True,
                )
        self.initialize_local_inputs(self.device, self.receiver.fidelity)
        self.initialized = True
        print(
            f"EQX {self.model_label} force graph: receiver tiles (retain_messages={self.receiver.retain_messages}, tile={self.receiver.tile})",
            flush=True,
        )

    def compute_forces(self, data):
        if not self.initialized:
            self._initialize_device(data)
        elif self.receiver is None and self.needs_tiles(data):
            torch.cuda.synchronize(self.device)
            self.force_graph = self.model = None
            self.start_receiver(data)
        if self.receiver is None:
            return super().compute_forces(data)
        torch.cuda.synchronize(self.device)
        if self.receiver.energy_domains is not None:
            from equivariantx_foundation.targets.lammps.compact import compute_domains

            self.exported_forces = None
            if compute_domains(self, data):
                return
        batch = self.local_inputs(data, data.nlocal, data.npairs)
        energies, forces = self.receiver(batch)
        self.write_local_results(data, energies, forces, data.nlocal)


class TiledFoundationLammpsCalc(CapacityOAMLammpsCalc):
    """Retry the compiled force calculation in tiles on a CUDA allocation failure."""

    def local_forces(self, batch, n, e):
        try:
            return super().local_forces(batch, n, e)
        except RuntimeError as exc:
            if not is_cuda_allocation_error(exc):
                raise
        # Unwind the entire compiled step before allocating the tiled model.
        # The parent writes to ML-IAP only once, after this function returns.
        self.force_graph = self.model = None
        print(
            f"EQX {self.model_label}: CUDA allocation failed; retrying with receiver tiles",
            flush=True,
        )
        self.receiver = self.build_receiver(
            torch.load(
                io.BytesIO(self.receiver_bytes),
                map_location=self.device,
                weights_only=False,
            ),
            tile=1024,
            retain_messages=False,
        )
        return self.receiver(batch)
