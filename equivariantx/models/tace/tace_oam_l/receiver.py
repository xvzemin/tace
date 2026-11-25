"""Layerwise receiver tiles for frozen single-rank TACE-OAM-L inference."""

import io

import torch

from equivariantx.models.tace.tace_oam_l.angular import edge_attributes
from equivariantx.models.tace.tace_oam_l.convolution import (
    MessageTranspose,
    bias_convolution,
)
from equivariantx.models.tace.tace_oam_l.lammps import OAMLammpsCalc
from equivariantx.models.tace.tace_oam_l.radial import prepare_radial, radial_heads
from tace.models.lammps import Graph


def is_cuda_allocation_error(error):
    """Return whether a CUDA operation failed to allocate memory."""
    return isinstance(error, torch.OutOfMemoryError) or any(
        message in str(error).lower()
        for message in (
            "cuda out of memory", "cuda error: out of memory",
            "cudaerrormemoryallocation", "cublas_status_alloc_failed",
        )
    )


class ReceiverTiles:
    """Evaluate atomic energies and edge-energy derivatives in receiver tiles.

    Parameters
    ----------
    model : torch.nn.Module
        Frozen, specialized FP32 OAM-L model with the affine CUDA convolution.
    tile : int
        Maximum number of receiver atoms in each block.
    retain_messages : bool
        Save normalized node messages for the analytic first derivative.
    """

    energy_domains = None
    tangent_embedding = None
    domain_tangent_embedding = None
    prepare_radial = staticmethod(prepare_radial)
    plan_domains = False
    recompute_domains = False
    domain_tile_limit = 4096

    def __init__(self, model, tile, retain_messages=False):
        from .model import _OAMLRepresentation

        self.model = model.readout_fn
        self.rep = self.model.representation
        self.tile = tile
        self.retain_messages = retain_messages
        self.replay_first_at = None
        self.replay_prefix_at = None
        self.message_transposes = (
            [MessageTranspose(i.linear_down, i.rejector.irreps_out) for i in self.rep.interactions]
            if retain_messages else []
        )
        self.coefficient_groups = {}
        self.plan_domains = isinstance(self.rep, _OAMLRepresentation)
        self.radial = [
            self.prepare_radial(inter, update)
            for inter, update in zip(self.rep.interactions, self.rep.edge_updates)
        ]
        self.first = self.rep.interactions[0].linear_up(
            self.rep.node_embedding.embedding.weight
        )
        self.fidelity = model.get_fidelity_idx()
        self.readout_widths = []
        for i, irreps in enumerate(self.rep.irreps_outs):
            if not self.model.use_alllayer and i < len(self.radial) - 1:
                self.readout_widths.append(0)
                continue
            readout = self.model.energy_readouts[i if self.model.use_alllayer else -1]
            first = (readout.linear2 if readout.last_layer else readout.linear1)[0]
            # Keep every input block consumed by the first readout linear.
            used = {ir for _, ir in first.irreps_out}
            self.readout_widths.append(
                max(
                    (
                        s.stop
                        for (_, ir), s in zip(irreps, irreps.slices())
                        if ir in used
                    ),
                    default=0,
                )
            )

    def full_graph_min_bytes(self, count, edges, vector_itemsize):
        """Lower bound on simultaneously live bytes in the full receiver reverse pass."""
        widths = [ir.dim for ir in self.rep.irreps_outs]
        # At an interior transition the input state and its dense cotangent
        # coexist with the incoming cotangent. Exclude possible readout aliases.
        lanes = max(
            (
                2 * a + b
                for i, (a, b) in enumerate(zip(widths, widths[1:]))
                if a > self.readout_widths[i] and b > self.readout_widths[i + 1]
            ),
            default=0,
        )
        # Global COO (2), tile-local COO (2), inverse source map (1);
        # original pair vectors and their accumulated derivatives.
        edge_bytes = edges * (5 * torch.int64.itemsize + 6 * vector_itemsize)
        return count * lanes * self.first.element_size() + edge_bytes

    def plan(self, types, index):
        source, target = index
        ordered = bool(torch.all(target[1:] >= target[:-1]))
        order = None if ordered else torch.argsort(target, stable=True)
        stops = torch.arange(0, len(types), self.tile, device=types.device)
        sorted_target = target if ordered else target[order]
        offsets = torch.searchsorted(sorted_target, stops).tolist() + [len(target)]
        tiles = []
        self.coefficient_groups = {}
        group_rows = (
            types.is_cuda
            and self.retain_messages
            and torch.version.hip is None
            and self.rep.model_dtype == torch.float32
        )
        for k, begin in enumerate(range(0, len(types), self.tile)):
            end = min(begin + self.tile, len(types))
            rows = slice(offsets[k], offsets[k + 1])
            edges = rows if ordered else order[rows]
            nodes, s = torch.unique(source[edges], return_inverse=True)
            t = target[edges] - begin
            local = slice(begin, end)
            species = torch.cat((types[local], types[nodes]))
            idx = torch.stack((s + end - begin, t))
            batch = torch.zeros_like(types[local])
            tiles.append((local, nodes, edges, idx, s, species, batch))
            if group_rows and end - begin >= 512 and len(s):
                from .element_grouped import grouping

                self.coefficient_groups[begin] = grouping(
                    types[local], self.rep.num_elements
                )
        return tiles

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

    def layer(self, i, tile, vec, xs, xc):
        local, nodes, edges, idx, s, z, batch = tile
        inter, params, n = self.rep.interactions[i], self.radial[i], len(batch)
        conv = inter.rejector
        if len(s):
            x, attr, h, density = self.edge_operands(i, tile, vec, xs)
            message = bias_convolution(
                conv.eqx_tp,
                x,
                attr,
                h,
                params.projection,
                params.projection_bias,
                torch.stack((s, idx[1])),
                n,
            )
            total = density.new_zeros((n, 1)).index_add_(0, idx[1], density)
        else:
            message = self.first.new_zeros(n, conv.eqx_tp.output_dim)
            total = self.first.new_zeros(n, 1)
        norm = inter.scatter_norm
        denom = norm.alpha + norm.beta * total
        y = inter.linear_down(conv.reshape_out(message)) / denom
        return self.update(i, tile, y, xc), y, denom

    def adjoint(self, i, tile, vec, xs, xc, incoming, y, denom):
        from equivariantx.utils.metadata import compact_metadata

        inter, params = self.rep.interactions[i], self.radial[i]
        conv = inter.rejector
        idx, s = tile[3:5]
        z = y.detach().requires_grad_()
        output = self.update(i, tile, z, xc)
        grad = torch.autograd.grad(output, (z, xc) if i else (z,), incoming)
        dz = grad[0]
        if not len(s):
            return (None, None, grad[1]) if i else (None,)
        message_grad = self.message_transposes[i](dz / denom)
        density_grad = -(dz * y).sum(-1, keepdim=True) * inter.scatter_norm.beta / denom
        metadata = compact_metadata(conv.eqx_tp.kernel_metadata)
        weight = self.tangent_embedding
        if self.energy_domains is not None:
            weight = self.domain_tangent_embedding
            if weight is not None and self.used_domain_tile != 4096:
                weight = None
        if weight is not None and vec.dtype == torch.float64:
            from .tangent import receiver_pullback

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
        else:
            x, attrs, h, density = self.edge_operands(i, tile, vec, xs)
            dx, da, dh = torch.ops.eqx_oam_l_conv.affine_vjp(
                x.contiguous(),
                attrs.contiguous(),
                h.contiguous(),
                params.projection.contiguous(),
                params.projection_bias.contiguous(),
                s,
                idx[1],
                message_grad.contiguous(),
                metadata,
                bool(i),
                True,
                True,
            )
            (dv,) = torch.autograd.grad(
                (attrs, h, density), vec, (da, dh, density_grad[idx[1]])
            )
        if i:
            (dsource,) = torch.autograd.grad(x, xs, dx)
            return dv, dsource, grad[1]
        return (dv,)

    def readout(self, tile, vec, descriptors, total_atoms):
        local, nodes, edges, idx, s, z, batch = tile
        n = len(batch)
        graph = Graph(
            True,
            None,
            (n, len(nodes)),
            1,
            None,
            vec.new_zeros(n, 3),
            vec,
            torch.linalg.vector_norm(vec, dim=-1, keepdim=True),
            vec.new_zeros(1, 3, 3),
            torch.full_like(batch, self.fidelity),
            torch.arange(n, device=z.device),
            z,
        )
        data = dict(batch=batch, ptr=batch.new_tensor([0, total_atoms]), edge_index=idx)
        out = self.model.readout_fn(data, graph, {"descriptors": descriptors}, None)
        return out["node_energy"]

    @torch.enable_grad()
    def readout_adjoint(self, tile, vec, states, n):
        xs = []
        for state, ir in zip(states, self.rep.irreps_outs, strict=True):
            x = state[tile[0]]
            if x.shape[1] < ir.dim:
                x = torch.nn.functional.pad(x, (0, ir.dim - x.shape[1]))
            xs.append(x.detach().requires_grad_())
        v = vec.requires_grad_()
        energies = self.readout(tile, v, xs, n)
        gradients = torch.autograd.grad(energies.sum(), (*xs, v), allow_unused=True)
        return energies.detach(), gradients[:-1], gradients[-1]

    @staticmethod
    def seed(state, readout_grad):
        if state.shape == readout_grad.shape:
            return readout_grad
        grad = torch.zeros_like(state)
        grad[:, : readout_grad.shape[1]] = readout_grad
        return grad

    def domain_count(self, count, available):
        """Size an initial power-of-two partition from the resident node tensors."""
        peak = self.domain_state_bytes([count] * len(self.readout_widths))
        # Leave room for halo expansion, the convolution workspace and fragmentation.
        budget = max(1, available // 4)
        required = (peak + budget - 1) // budget
        return min(count, 1 << (max(1, required) - 1).bit_length())

    def domain_state_bytes(self, sizes):
        states = [
            size * ir.dim for size, ir in zip(sizes, self.rep.irreps_outs, strict=True)
        ]
        peak = max(sum(states[:i+1]) + (states[i-1] if i else 0) for i in range(len(states)))
        return (peak + sizes[-1] * sum(self.readout_widths)) * self.first.element_size()

    def domain_tape(self, groups):
        """Choose reduced per-node convolution outputs to keep for one domain."""
        from .domains import available_bytes

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
        retained = sum(
            n * (inter.linear_down.irreps_out.dim + 1)
            for n, keep, inter in zip(sizes, tape, self.rep.interactions, strict=True)
            if keep is not None
        ) * self.first.element_size()
        # Reserve live node states, the chosen tape and allocator headroom before
        # budgeting the local edge workspace against the largest receiver degree.
        budget = (
            3 * available // 4 - self.domain_state_bytes(sizes) - retained - (2 << 30)
        )
        limit = min(
            self.domain_tile_limit, max(sizes), budget // (max(1, max_degree) << 14)
        )
        return 1 << (max(1024, limit).bit_length() - 1)

    def sweep(self, i, tiles, vec, count, previous, save_messages=False):
        state = None
        messages = []
        for tile in tiles:
            local, nodes, edges = tile[:3]
            x, y, denom = self.layer(
                i, tile, vec[edges], previous[nodes] if i else None,
                previous[local] if i else None,
            )
            if state is None:
                state = x.new_empty(count, x.shape[1])
            state[local] = x
            if save_messages:
                messages.append((y, denom))
        return state, messages

    def prepare_elements(self, types):
        """Prepare optional element tables for the current structure."""

    def __call__(self, batch):
        self.prepare_elements(batch["node_type"])
        count = len(batch["node_type"])
        if self.energy_domains is not None:
            if self.energy_domains.count == count:
                return self.energy_domains.evaluate(self, batch)
            self.energy_domains = None
        if self.replay_prefix_at is None or count < self.replay_prefix_at:
            try:
                return self.evaluate(batch, checkpoint=False)
            except torch.OutOfMemoryError:
                if self.retain_messages:
                    raise
        # The failed evaluation's traceback and its tensor locals have been
        # released before starting another complete force evaluation.
        try:
            result = self.evaluate(batch, checkpoint=True)
        except torch.OutOfMemoryError:
            if count < 2:
                raise
        else:
            self.replay_prefix_at = count if self.replay_prefix_at is None else min(
                count, self.replay_prefix_at
            )
            return result
        from .domains import EnergyDomains

        self.coefficient_groups.clear()
        self.energy_domains = EnergyDomains(count)
        print(
            f"EQX OAM-L force graph: exact energy domains after prefix allocation failed "
            f"({count} atoms)",
            flush=True,
        )
        return self.energy_domains.evaluate(self, batch)

    @torch.no_grad()
    def evaluate(self, batch, checkpoint):
        z, vec = batch["node_type"], batch["edge_vector"].detach()
        tiles = self.plan(z, batch["edge_index"])
        states, saved, prefixes = [], [], []
        layers = len(self.rep.interactions)
        for i in range(layers):
            state, blocks = self.sweep(
                i, tiles, vec, len(z), states[-1] if i else None, self.retain_messages
            )
            if checkpoint:
                width = self.readout_widths[i]
                if i == layers - 1 and width == state.shape[1]:
                    prefixes.append(state)
                else:
                    prefixes.append(state[:, :width].clone())
                if 0 < i < layers - 2:
                    states[-1] = None
            states.append(state)
            saved.append(blocks)
        energy = vec.new_empty(len(z))
        forces = torch.zeros_like(vec)
        readout_grads = prefixes if checkpoint else [
            x.new_zeros(len(z), width)
            for x, width in zip(states, self.readout_widths, strict=True)
        ]
        for tile in tiles:
            local, _, edges = tile[:3]
            energies, d, dv = self.readout_adjoint(
                tile, vec[edges], prefixes if checkpoint else states, len(z)
            )
            energy[local] = energies
            for dest, dx in zip(readout_grads, d):
                if dx is not None:
                    dest[local] = dx[:, : dest.shape[1]]
                elif checkpoint:
                    dest[local].zero_()
            if dv is not None:
                forces[edges] = dv
        if not checkpoint and self.replay_first_at is not None and len(z) >= self.replay_first_at:
            states[0] = None
        incoming = self.seed(states[-1], readout_grads.pop())
        for i in reversed(range(layers)):
            states.pop()
            previous = None
            if i:
                if states[-1] is None:
                    prior = None
                    for j, anchor in enumerate(states):
                        if anchor is not None:
                            prior = anchor
                        else:
                            prior, _ = self.sweep(j, tiles, vec, len(z), prior)
                    states[-1] = prior
                    del prior
                readout_grad = readout_grads.pop()
                try:
                    previous = self.seed(states[-1], readout_grad)
                except torch.OutOfMemoryError:
                    if checkpoint or i < 2 or self.retain_messages or states[0] is None:
                        raise
                    states[0] = None
                    self.replay_first_at = len(z)
                if previous is None:
                    previous = self.seed(states[-1], readout_grad)
            for k, tile in enumerate(tiles):
                local, nodes, edges = tile[:3]
                v = vec[edges].requires_grad_()
                xs = states[-1][nodes].detach().requires_grad_() if i else None
                xc = states[-1][local].detach().requires_grad_() if i else None
                with torch.enable_grad():
                    if self.retain_messages:
                        grads = self.adjoint(
                            i, tile, v, xs, xc, incoming[local], *saved[i][k]
                        )
                    else:
                        y = self.layer(i, tile, v, xs, xc)[0]
                        if not y.requires_grad:
                            continue
                        grads = torch.autograd.grad(
                            y,
                            (v, xs, xc) if i else (v,),
                            incoming[local],
                            allow_unused=True,
                        )
                if grads[0] is not None:
                    forces[edges] += grads[0]
                if i:
                    if grads[1] is not None:
                        previous.index_add_(0, nodes, grads[1])
                    if grads[2] is not None:
                        previous[local] += grads[2]
            saved.pop()
            incoming = previous
        return energy, forces


class CapacityOAMLammpsCalc(OAMLammpsCalc):
    """Select the compiled or receiver-tiled force graph on one CUDA device."""

    model_label = "OAM-L"
    receiver_tangent = False
    domain_receiver_tangent = False

    def attach_receiver(self, model):
        """Embed the frozen model used by the receiver-tiled evaluator."""
        from .model import optimize_oam_l

        frozen = optimize_oam_l(
            model.eval().requires_grad_(False),
            fused_geometry=True,
            fused_projection=True,
        )
        return self.attach_frozen_receiver(frozen)

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
                from .domains import EnergyDomains

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
            from .compact import compute_domains

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
        print(f"EQX {self.model_label}: CUDA allocation failed; retrying with receiver tiles", flush=True)
        self.receiver = self.build_receiver(
            torch.load(io.BytesIO(self.receiver_bytes), map_location=self.device, weights_only=False),
            tile=1024, retain_messages=False,
        )
        return self.receiver(batch)
