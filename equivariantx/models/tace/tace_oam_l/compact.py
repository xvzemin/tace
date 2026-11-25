"""Compact single-rank Kokkos graph and force storage for energy domains."""

import torch

from .domains import IncomingEdges


class CompactRows(IncomingEdges):
    """Stable incoming rows of a pair of 32-bit Kokkos edge columns."""

    def __init__(self, index, count, chunk=1 << 20):
        self.source, target = index
        self.order = None
        nodes = torch.arange(count + 1, device=target.device, dtype=torch.int32)
        ordered = torch.ones((), dtype=torch.bool, device=target.device)
        for start in range(1, len(target), chunk):
            stop = min(len(target), start + chunk)
            ordered &= torch.all(target[start:stop] >= target[start - 1 : stop - 1])
        if bool(ordered):
            self.offsets = torch.searchsorted(target, nodes)
            return

        # Preserve original edge order within each receiver without an E-long
        # int64 argsort or interleaving edges belonging to different images.
        counts = nodes.new_zeros(count)
        one = nodes.new_ones(1)
        for ids in target.split(chunk):
            counts.index_add_(0, ids, one.expand_as(ids))
        self.offsets = torch.cat(
            (nodes.new_zeros(1), counts.cumsum(0, dtype=torch.int32))
        ).long()
        cursors = self.offsets[:-1].int()
        self.order = torch.empty_like(target)
        for start in range(0, len(target), chunk):
            ids = target[start : start + chunk]
            order = torch.argsort(ids, stable=True)
            unique, inverse, counts = torch.unique_consecutive(
                ids[order], return_counts=True, return_inverse=True
            )
            shift = cursors[unique] - (counts.cumsum(0) - counts)
            positions = torch.arange(len(ids), device=ids.device) + shift[inverse]
            self.order[positions] = (order + start).int()
            cursors[unique] += counts.int()

    def rows(self, nodes):
        edges, receivers = super().rows(nodes)
        return edges.long(), receivers


class NodalForce:
    """Accumulate pair derivatives directly into owner and ghost force rows."""

    def __init__(self, target, ghost, count):
        self.target, self.ghost = target, ghost
        self.force = torch.zeros((count, 3), device=target.device, dtype=torch.float64)

    def index_add_(self, dim, edges, values):
        self.force.index_add_(0, self.target[edges], values)
        self.force.index_add_(0, self.ghost[edges], values, alpha=-1)


def graph_inputs(calc, data, target, ghost):
    """Resolve Kokkos ghost indices and preserve the original edge and image order."""
    device = target.device
    n, ntotal = data.nlocal, data.ntotal
    if ntotal == n or not data.npairs:
        source = ghost
    else:
        # The ML-IAP exchange ABI accepts floats and doubles, not integer IDs.
        if calc.exchange_indices is None or len(calc.exchange_indices) != ntotal:
            calc.exchange_indices = torch.arange(
                ntotal, device=device, dtype=torch.float64
            )
            calc.exchange_owners = torch.empty_like(calc.exchange_indices)
            torch.cuda.current_stream(device).synchronize()
        data.forward_exchange(calc.exchange_indices, calc.exchange_owners, 1)
        torch.cuda.synchronize(device)
        source = torch.index_select(calc.exchange_owners.int(), 0, ghost)
    return dict(
        edge_index=(source, target),
        edge_vector=torch.as_tensor(data.rij),
        node_type=torch.as_tensor(data.elems)[:n].long(),
    )


@torch.no_grad()
def compute_domains(calc, data):
    """Evaluate and commit forces using compact graph storage when available."""
    domains = calc.receiver.energy_domains
    if (
        domains.count != data.nlocal
        or not data.npairs
        or max(data.nlocal, data.ntotal, data.npairs) > torch.iinfo(torch.int32).max
        or data.natomneigh > data.nlocal
    ):
        return False
    target, ghost = torch.as_tensor(data.pair_i), torch.as_tensor(data.pair_j)
    if target.dtype != torch.int32 or ghost.dtype != torch.int32:
        return False
    batch = graph_inputs(calc, data, target, ghost)
    calc.receiver.prepare_elements(batch["node_type"])
    rows = CompactRows(batch["edge_index"], data.nlocal)
    # vflag alone cannot distinguish FDOTR from pair_modify nofdotr. Retain
    # the official tally for every virial request, including per-atom stress.
    nodal = not data.vflag and data.f is not None
    energy, force = domains.evaluate(
        calc.receiver,
        batch,
        csr=rows,
        new_derivative=(lambda: NodalForce(target, ghost, data.ntotal))
        if nodal
        else None,
    )
    if nodal:
        torch.as_tensor(data.f).add_(force.force)
        force = batch["edge_vector"][:0]
    calc.write_local_results(data, energy, force, data.nlocal)
    return True
