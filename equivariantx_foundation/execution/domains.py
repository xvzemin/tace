"""Exact center-energy domains with shrinking directed message-passing halos."""

from functools import cached_property

import torch


def available_bytes(device):
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def receiver_tiles(types, index, tile):
    """Yield receiver blocks and their unique source atoms.

    Parameters
    ----------
    types : torch.Tensor
        Element indices of shape ``(nodes,)``.
    index : torch.Tensor
        Source and receiver indices of shape ``(2, edges)``.
    tile : int
        Maximum number of receivers per block.
    """
    source, target = index
    ordered = bool(torch.all(target[1:] >= target[:-1]))
    order = None if ordered else torch.argsort(target, stable=True)
    stops = torch.arange(0, len(types), tile, device=types.device)
    sorted_target = target if ordered else target[order]
    offsets = torch.searchsorted(sorted_target, stops).tolist() + [len(target)]
    for k, begin in enumerate(range(0, len(types), tile)):
        end = min(begin + tile, len(types))
        rows = slice(offsets[k], offsets[k + 1])
        edges = rows if ordered else order[rows]
        nodes, s = torch.unique(source[edges], return_inverse=True)
        local = slice(begin, end)
        species = torch.cat((types[local], types[nodes]))
        idx = torch.stack((s + end - begin, target[edges] - begin))
        yield local, nodes, edges, idx, s, species, torch.zeros_like(types[local])


class IncomingEdges:
    """Rows of the supplied COO graph, indexed by original receiver and edge IDs."""

    def __init__(self, index, count):
        self.source, target = index
        self.order = None
        if not bool(torch.all(target[1:] >= target[:-1])):
            self.order = torch.argsort(target, stable=True)
            target = target[self.order]
        nodes = torch.arange(count + 1, device=index.device)
        self.offsets = torch.searchsorted(target, nodes)

    @cached_property
    def max_degree(self):
        maximum = self.offsets.new_zeros(())
        for start in range(0, len(self.offsets) - 1, 1 << 20):
            part = self.offsets[start : start + (1 << 20) + 1]
            maximum = torch.maximum(maximum, (part[1:] - part[:-1]).max())
        return int(maximum)

    def rows(self, nodes):
        start = self.offsets[nodes]
        count = self.offsets[nodes + 1] - start
        target = torch.repeat_interleave(
            torch.arange(len(nodes), device=nodes.device), count
        )
        end = torch.cumsum(count, 0)
        row = (
            torch.arange(len(target), device=nodes.device)
            + (start + count - end)[target]
        )
        return (row if self.order is None else self.order[row]), target

    def halo(self, core, depth, tile):
        groups = [core]
        for _ in range(depth):
            mask = torch.zeros_like(self.offsets[:-1], dtype=torch.bool)
            mask[groups[-1]] = True
            for target in groups[-1].split(tile):
                e, _ = self.rows(target)
                mask.index_fill_(0, self.source[e].long(), True)
            groups.append(torch.nonzero(mask, as_tuple=True)[0])
        return groups[::-1]

    def tiles(self, nodes, types, tile):
        for begin in range(0, len(nodes), tile):
            local = slice(begin, min(begin + tile, len(nodes)))
            target = nodes[local]
            edges, t = self.rows(target)
            source, s = torch.unique(
                self.source[edges], sorted=True, return_inverse=True
            )
            source = source.long()
            z = torch.cat((types[target], types[source]))
            idx = torch.stack((s + len(target), t))
            yield (local, source, edges, idx, s, z, torch.zeros_like(target)), target


@torch.no_grad()
def chart_partition(csr, index, vectors, parts, tile=8192):
    """Partition by coordinates reconstructed along incoming graph edges.

    The temporary spanning-forest coordinates select centers only. Image
    displacements and owner IDs used by the physical model are unchanged.
    """
    n = len(csr.offsets) - 1
    if not (0 < parts <= n):
        raise ValueError("Expected a positive number of nonempty parts")
    ids = torch.arange(n, device=csr.source.device)
    if not len(vectors):
        return list(torch.tensor_split(ids, parts))
    coords = vectors.new_zeros((n, 3))
    forest = torch.full_like(ids, -1)
    degree = csr.offsets[1:] - csr.offsets[:-1]
    if bool((degree == 0).any()):
        incident = degree > 0
        for neighbors in csr.source.split(1 << 20):
            incident.index_fill_(0, neighbors.long(), True)
        forest[~incident] = 0
    infinity = torch.iinfo(torch.int64).max
    choices = torch.full_like(ids, infinity)
    modulus = len(vectors) + 1
    for component in range(1, 17):
        remaining = torch.nonzero(forest < 0, as_tuple=True)[0]
        if not len(remaining):
            break
        frontier = remaining[:1]
        forest[frontier] = component
        for _ in range(128):
            if not len(frontier):
                break
            for rows in frontier.split(tile):
                e, _ = csr.rows(rows)
                ns = csr.source[e]
                keep = forest[ns] < 0
                ns, e = ns[keep], e[keep]
                proposed = coords[index[1][e]] + vectors[e]
                # Nonnegative float32 bit patterns preserve distance order.
                # Use edge ID to resolve ties between admissible predecessor images.
                length = (
                    proposed.square()
                    .sum(-1)
                    .float()
                    .contiguous()
                    .view(torch.int32)
                    .long()
                )
                key = length * modulus + e
                choices.scatter_reduce_(0, ns.long(), key, "amin")
            frontier = torch.nonzero(
                (choices < infinity) & (forest < 0), as_tuple=True
            )[0]
            e = choices[frontier] % modulus
            coords[frontier] = coords[index[1][e]] + vectors[e]
            forest[frontier] = component
        if len(frontier):
            break
    # Far or disconnected leftovers share a partition label, never a model node.
    targets = [i * (n // parts) + min(i, n % parts) for i in range(parts + 1)]

    def split(rows, lo, hi):
        if hi - lo == 1:
            return [rows.sort().values]
        mid = (lo + hi) // 2
        c = forest[rows]
        if bool(c.min() != c.max()):
            key = c
        else:
            xyz = coords[rows]
            axis = int((xyz.amax(0) - xyz.amin(0)).argmax())
            key = xyz[:, axis]
        seq = rows[torch.argsort(key, stable=True)]
        cut = targets[mid] - targets[lo]
        return split(seq[:cut], lo, mid) + split(seq[cut:], mid, hi)

    return split(ids, 0, parts)


class EnergyDomains:
    """Accumulate exact atomic energies and edge derivatives across disjoint cores.

    Parameters
    ----------
    count : int
        Number of local owner atoms.
    """

    def __init__(self, count):
        self.count = count
        self.cores = None
        self.parts = min(8, count)

    @torch.no_grad()
    def evaluate(self, r, batch, *, csr=None, new_derivative=None):
        z, vec = batch["node_type"], batch["edge_vector"].detach()
        index = batch["edge_index"]
        if csr is None:
            csr = IncomingEdges(index, self.count)
        while True:
            size = 0
            try:
                energy = vec.new_empty(len(z))
                derivative = torch.zeros_like(vec) if new_derivative is None else new_derivative()
                if self.cores is None:
                    if getattr(r, "plan_domains", False) and vec.is_cuda:
                        self.parts = max(
                            self.parts, r.domain_count(self.count, available_bytes(vec.device))
                        )
                    self.cores = chart_partition(csr, index, vec, self.parts)
                for core in self.cores:
                    size = len(core)
                    r.used_domain_tape = False
                    r.used_domain_tile = 0
                    self.core(r, csr, core, z, vec, energy, derivative)
                return energy, derivative
            except torch.OutOfMemoryError:
                retry_without_tape = size and getattr(r, "used_domain_tape", False)
                retry_tile = size and getattr(r, "used_domain_tile", 0) > 1024
                if size <= 1 and not (retry_without_tape or retry_tile):
                    raise
            # Discard every partial contribution before retrying. The traceback
            # with the failed core's GPU tensors is gone outside the except block.
            del energy, derivative
            if retry_tile:
                r.domain_tile_limit = max(1024, r.used_domain_tile // 2)
                continue
            if retry_without_tape:
                r.recompute_domains = True
                continue
            self.parts = min(self.parts * 2, self.count)
            self.cores = None

    @staticmethod
    def core(r, csr, core, z, vec, energy, derivative):
        widths = r.state_widths
        layers = len(widths)
        groups = csr.halo(core, layers - 1, r.tile)
        plan = getattr(r, "domain_tape", None)
        tape = plan(groups) if plan is not None else [None] * layers
        r.used_domain_tape = any(t is not None for t in tape)
        tile_size = r.tile
        if getattr(r, "plan_domains", False) and vec.is_cuda:
            tile_size = r.domain_tile(
                groups, tape, csr.max_degree, available_bytes(vec.device)
            )
            r.used_domain_tile = tile_size
        states = []
        for i, nodes in enumerate(groups):
            states.append(r.state_template.new_empty(len(nodes), widths[i]))
            for tile, ids in csr.tiles(nodes, z, tile_size):
                local, src, edges = tile[:3]
                xs = (
                    states[i - 1][torch.searchsorted(groups[i - 1], src)] if i else None
                )
                xc = (
                    states[i - 1][torch.searchsorted(groups[i - 1], ids)] if i else None
                )
                state, y, denom = r.layer(i, tile, vec[edges], xs, xc)
                states[i][local] = state
                if tape[i] is not None:
                    tape[i].append((y, denom))
                del state, y, denom
        seeds = [r.state_template.new_zeros(len(core), w) for w in r.readout_widths]
        for tile, ids in csr.tiles(core, z, tile_size):
            local, _, edges = tile[:3]
            # The last group is the core in its original row order.
            rows = (
                x[local if i == layers - 1 else torch.searchsorted(g, ids)]
                if w else None
                for i, (g, x, w) in enumerate(
                    zip(groups, states, r.readout_widths, strict=True)
                )
            )
            e, ds, dv = r.readout_adjoint(tile, vec[edges], rows, len(z))
            energy[ids] = e
            for dest, dx in zip(seeds, ds):
                if dx is not None:
                    dest[local] = dx[:, : dest.shape[1]]
            if dv is not None:
                derivative.index_add_(0, edges, dv)
            del ds, dv, e, dest, dx

        def seed(i):
            s = groups[i]
            values = seeds.pop()
            if i == layers - 1 and values.shape[1] == widths[i]:
                return values
            x = r.state_template.new_zeros(len(s), widths[i])
            if values.shape[-1]:
                rows = slice(None) if i == layers - 1 else torch.searchsorted(s, core)
                x[rows, : values.shape[1]] = values
            return x

        adj = seed(layers - 1)
        for i in range(layers - 1, -1, -1):
            states.pop()
            prev = seed(i - 1) if i else None
            blocks = tape.pop()
            for k, (tile, ids) in enumerate(csr.tiles(groups[i], z, tile_size)):
                local, src, edges = tile[:3]
                v = vec[edges].requires_grad_()
                if i:
                    si, ti = (
                        torch.searchsorted(groups[i - 1], src),
                        torch.searchsorted(groups[i - 1], ids),
                    )
                    xs, xc = (
                        states[-1][si].requires_grad_(),
                        states[-1][ti].requires_grad_(),
                    )
                else:
                    xs = xc = None
                with torch.enable_grad():
                    if blocks is not None:
                        grads = r.adjoint(i, tile, v, xs, xc, adj[local], *blocks[k])
                        blocks[k] = None
                    else:
                        y = r.layer(i, tile, v, xs, xc)[0]
                        if not y.requires_grad:
                            continue
                        grads = torch.autograd.grad(
                            y, (v, xs, xc) if i else (v,), adj[local], allow_unused=True
                        )
                        del y
                if grads[0] is not None:
                    derivative.index_add_(0, edges, grads[0])
                if i and grads[1] is not None:
                    prev.index_add_(0, si, grads[1])
                if i and grads[2] is not None:
                    prev[ti] += grads[2]
                del grads
            adj = prev
