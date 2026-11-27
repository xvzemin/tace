"""Tiled layer sweeps and reverse accumulation for local message-passing programs."""

import torch

from .domains import EnergyDomains, receiver_tiles


def is_cuda_allocation_error(error):
    """Return whether a CUDA operation failed to allocate memory."""
    return isinstance(error, torch.OutOfMemoryError) or any(
        message in str(error).lower()
        for message in (
            "cuda out of memory",
            "cuda error: out of memory",
            "cudaerrormemoryallocation",
            "cublas_status_alloc_failed",
        )
    )


class LayerwiseReceiver:
    """Accumulate local energies and their derivatives in original COO order.

    Adapters supply per-layer ``state_widths``, consumed leading
    ``readout_widths``, a ``state_template`` for allocations, and the
    ``layer`` and ``readout`` callbacks. A layer reads sender and receiver
    states one hop away; its energy readout must be additive over centers.
    Retaining a layer's two tape values also requires an ``adjoint`` callback.
    """

    energy_domains = None
    receiver_label = "receiver"

    def plan(self, types, index):
        return list(receiver_tiles(types, index, self.tile))

    def discard_tile_plan(self):
        """Release optional adapter workspaces before retrying as exact domains."""

    def full_graph_min_bytes(self, count, edges, vector_itemsize):
        """Lower bound on simultaneously live bytes in the full receiver reverse pass."""
        widths = self.state_widths
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
        return count * lanes * self.state_template.element_size() + edge_bytes

    @torch.enable_grad()
    def readout_adjoint(self, tile, vec, rows, n):
        """Read tile-local descriptors and return cotangents for consumed layers."""
        xs, active = [], []
        zero = None
        for x, width in zip(rows, self.state_widths, strict=True):
            if x is None:
                if zero is None:
                    zero = self.state_template.new_zeros(())
                xs.append(zero.expand(len(tile[-1]), width))
            else:
                if x.shape[1] < width:
                    x = torch.nn.functional.pad(x, (0, width - x.shape[1]))
                x = x.detach().requires_grad_()
                xs.append(x)
                active.append(x)
        v = vec.requires_grad_()
        energies = self.readout(tile, v, xs, n)
        if not energies.requires_grad:
            return energies.detach(), (None,) * len(xs), None
        gradients = torch.autograd.grad(energies.sum(), (*active, v), allow_unused=True)
        ds = iter(gradients[:-1])
        per_layer = tuple(next(ds) if x.requires_grad else None for x in xs)
        return energies.detach(), per_layer, gradients[-1]

    @staticmethod
    def seed(state, readout_grad):
        if state.shape == readout_grad.shape:
            return readout_grad
        grad = torch.zeros_like(state)
        grad[:, : readout_grad.shape[1]] = readout_grad
        return grad

    def domain_state_bytes(self, sizes):
        states = [
            size * width for size, width in zip(sizes, self.state_widths, strict=True)
        ]
        peak = max(
            sum(states[: i + 1]) + (states[i - 1] if i else 0)
            for i in range(len(states))
        )
        return (
            peak + sizes[-1] * sum(self.readout_widths)
        ) * self.state_template.element_size()

    def sweep(self, i, tiles, vec, count, previous, save_messages=False):
        state = None
        messages = []
        for tile in tiles:
            local, nodes, edges = tile[:3]
            x, y, denom = self.layer(
                i,
                tile,
                vec[edges],
                previous[nodes] if i else None,
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
            self.replay_prefix_at = (
                count
                if self.replay_prefix_at is None
                else min(count, self.replay_prefix_at)
            )
            return result
        self.discard_tile_plan()
        self.energy_domains = EnergyDomains(count)
        print(
            f"EQX {self.receiver_label} force graph: exact energy domains after prefix allocation failed "
            f"({count} atoms)",
            flush=True,
        )
        return self.energy_domains.evaluate(self, batch)

    @torch.no_grad()
    def evaluate(self, batch, checkpoint):
        z, vec = batch["node_type"], batch["edge_vector"].detach()
        tiles = self.plan(z, batch["edge_index"])
        states, saved, prefixes = [], [], []
        layers = len(self.state_widths)
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
        readout_grads = (
            prefixes
            if checkpoint
            else [
                x.new_zeros(len(z), width)
                for x, width in zip(states, self.readout_widths, strict=True)
            ]
        )
        for tile in tiles:
            local, _, edges = tile[:3]
            rows = (
                state[local] if width else None
                for state, width in zip(
                    prefixes if checkpoint else states, self.readout_widths, strict=True
                )
            )
            energies, d, dv = self.readout_adjoint(tile, vec[edges], rows, len(z))
            energy[local] = energies
            for dest, dx in zip(readout_grads, d):
                if dx is not None:
                    dest[local] = dx[:, : dest.shape[1]]
                elif checkpoint:
                    dest[local].zero_()
            if dv is not None:
                forces[edges] = dv
        if (
            not checkpoint
            and self.replay_first_at is not None
            and len(z) >= self.replay_first_at
        ):
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
