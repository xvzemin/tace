"""Exact directed receiver sweeps, retries and owner/ghost derivatives without a host model."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("prefix", [False, True])
def test_receiver_replays_evicted_states_after_allocation_failure(prefix):
    from equivariantx_foundation.execution.receiver import LayerwiseReceiver

    class Tiles(LayerwiseReceiver):
        def __init__(self, fail=False):
            self.tile, self.retain_messages = 2, False
            self.replay_first_at = self.replay_prefix_at = None
            n = 5 if prefix else 3
            self.state_widths = (3,) * n
            self.state_template, self.readout_widths = torch.empty(0), [1] * n
            self.calls, self.seeds, self.fail = 0, 0, fail
            self.policies = []

        def evaluate(self, batch, checkpoint):
            if checkpoint and hasattr(self, "failed_state"):
                assert self.failed_state() is None
            self.policies.append(checkpoint)
            return super().evaluate(batch, checkpoint)

        def layer(self, i, tile, v, xs, xc):
            self.calls += int(not i and not torch.is_grad_enabled())
            x = torch.zeros(len(tile[-1]), 3).index_add(0, tile[3][1], v)
            if i:
                x = x.index_add(0, tile[3][1], xs[tile[4]]).tanh() + xc
            return x, None, None

        def readout(self, tile, vec, descriptors, total_atoms):
            return sum(x[:, 0].square() for x in descriptors)

        def seed(self, x, grad):
            self.seeds += 1
            if self.fail and self.seeds in ((2, 3) if prefix else (2,)):
                if self.seeds == 3:
                    import weakref

                    self.failed_state = weakref.ref(x)
                raise torch.OutOfMemoryError("simulated gradient allocation failure")
            return super().seed(x, grad)

    index = torch.tensor([[2, 0, 1, 2], [0, 2, 0, 1]])
    data = {
        "node_type": torch.arange(3),
        "edge_index": index,
        "edge_vector": torch.arange(12, dtype=torch.float32).reshape(4, 3) / 11,
    }
    expected = Tiles()(data)
    recovered = Tiles(True)
    for step in (1, 2):
        actual = recovered(data)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert recovered.replay_first_at == 3
        if prefix:
            assert recovered.policies == [False] + [True] * step
            assert recovered.replay_prefix_at == 3
        else:
            assert recovered.policies == [False] * step
            assert recovered.replay_prefix_at is None and recovered.calls == step * 4


@pytest.mark.parametrize("widths", [(1, 1, 1), (0, 1, 3)])
def test_receiver_domains_retain_only_roots_and_restart_partial_derivatives(
    monkeypatch,
    widths,
):
    from equivariantx_foundation.execution.domains import EnergyDomains
    from equivariantx_foundation.execution.receiver import LayerwiseReceiver
    from equivariantx_foundation.targets.lammps.compact import (
        CompactRows,
        NodalForce,
        compute_domains,
    )

    class Tiles(LayerwiseReceiver):
        def __init__(self, fail):
            self.retain_messages = False
            self.tile, self.replay_first_at, self.replay_prefix_at = 3, None, None
            self.state_template = torch.zeros((), dtype=torch.float64)
            self.readout_widths = widths
            self.state_widths = (3,) * 3
            self.policies, self.fail = [], fail

        def layer(self, i, tile, v, xs, xc):
            target = tile[3][1]
            x = v.new_zeros(len(tile[-1]), 3).index_add(0, target, v)
            if i:
                x = (x.index_add(0, target, xs[tile[4]]) + xc).tanh()
            return x, None, None

        def readout(self, tile, v, ds, total_atoms):
            assert total_atoms == 12
            for x, width in zip(ds, widths, strict=True):
                if not width:
                    assert x.untyped_storage().nbytes() == x.element_size()
                    assert not x.requires_grad
            return sum(
                x[:, :w].square().sum(-1) for x, w in zip(ds, widths) if w
            ).index_add(0, tile[3][1], v.square().sum(-1))

        def evaluate(self, batch, checkpoint):
            self.policies.append(checkpoint)
            if self.fail:
                raise torch.OutOfMemoryError("whole-graph and prefix")
            return super().evaluate(batch, checkpoint)

    n = 12
    s = torch.tensor([0, 1, 0, 2, 2, 6, 7, 7, 6, 4, 5])
    t = torch.tensor([1, 0, 1, 3, 4, 7, 6, 8, 9, 5, 4])
    g = torch.Generator().manual_seed(43)
    v = torch.randn(len(s), 3, dtype=torch.float64, generator=g)
    data = {
        "node_type": torch.arange(n) % 2,
        "edge_vector": v,
        "edge_index": torch.stack((s, t)),
    }
    trial, reference = Tiles(True), Tiles(False)
    core = EnergyDomains.core
    attempted = []

    def limited(r, csr, rows, z, vec, energy, derivative):
        attempted.append(len(rows))
        core(r, csr, rows, z, vec, energy, derivative)
        if len(rows) > 1:
            raise torch.OutOfMemoryError(
                "partial energy and force have already been written"
            )

    monkeypatch.setattr(EnergyDomains, "core", staticmethod(limited))
    for _ in range(2):
        a, b = trial(data), reference(data)
        src, dst = data["edge_index"]
        vectors = v.detach().requires_grad_()
        base = vectors.new_zeros(n, 3).index_add(0, dst, vectors)
        state = base
        dense = base.new_zeros(n).index_add(0, dst, vectors.square().sum(-1))
        for i, width in enumerate(widths):
            if i:
                state = (base.index_add(0, dst, state[src]) + state).tanh()
            dense = dense + state[:, :width].square().sum(-1)
        (force,) = torch.autograd.grad(dense.sum(), vectors)
        for val, ref in zip(a, (dense, force), strict=True):
            torch.testing.assert_close(val, ref, rtol=1e-12, atol=1e-12)
        for val, ref in zip(a, b, strict=True):
            torch.testing.assert_close(val, ref, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(v.T @ a[1], v.T @ b[1], rtol=1e-12, atol=1e-12)
        data["edge_index"] = data["edge_index"].clone()
        data["edge_index"][0, 0] = 10
        v = v.roll(2, 0)
        data["edge_vector"] = v
    assert trial.policies == [False, True]
    assert attempted.count(2) == 1 and attempted.count(1) == 2 * n
    assert len(trial.energy_domains.cores) == n
    empty = {**data, "edge_index": data["edge_index"][:, :0], "edge_vector": v[:0]}
    roots = trial.energy_domains.cores
    for actual in (trial(empty), Tiles(True)(empty)):
        expected = reference(empty)
        for value, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, ref)
    assert roots is trial.energy_domains.cores

    # Multiple periodic images can have the same local model owner but a
    # different LAMMPS ghost; reverse force communication needs the latter.
    index = tuple(row.int() for row in data["edge_index"])
    ghost = torch.arange(n, n + len(s), dtype=torch.int32)
    count = n + len(s)
    compact = EnergyDomains(n)
    energy, nodal = compact.evaluate(
        trial,
        {**data, "edge_index": index},
        csr=CompactRows(index, n, chunk=3),
        new_derivative=lambda: NodalForce(t.int(), ghost, count),
    )
    expected_energy, pair = reference(data)
    expected_force = v.new_zeros((count, 3)).index_add_(0, t, pair)
    expected_force.index_add_(0, ghost, pair, alpha=-1)
    torch.testing.assert_close(energy, expected_energy)
    torch.testing.assert_close(nodal.force, expected_force)
    for large in ("nlocal", "ntotal", "npairs"):
        sizes = {"nlocal": n, "ntotal": count, "npairs": len(s), large: 2**31}
        owner = SimpleNamespace(
            receiver=SimpleNamespace(
                energy_domains=SimpleNamespace(count=sizes["nlocal"])
            )
        )
        assert not compute_domains(owner, SimpleNamespace(**sizes))

    attempts = []

    def oversized_block(r, csr, rows, z, vec, e, d):
        if not attempts or attempts[-1] != (r.domain_tile_limit, r.recompute_domains):
            attempts.append((r.domain_tile_limit, r.recompute_domains))
        core(r, csr, rows, z, vec, e, d)
        if not r.recompute_domains:
            r.used_domain_tile, r.used_domain_tape = r.domain_tile_limit, True
            raise torch.OutOfMemoryError("written energy and forces")

    monkeypatch.setattr(EnergyDomains, "core", staticmethod(oversized_block))
    trial.domain_tile_limit = 4096
    trial.recompute_domains = False
    energy, edges = trial(data)
    assert attempts == [(4096, False), (2048, False), (1024, False), (1024, True)]
    torch.testing.assert_close(energy, expected_energy)
    torch.testing.assert_close(edges, pair)

    def single_root(*args):
        raise torch.OutOfMemoryError("single root")

    monkeypatch.setattr(EnergyDomains, "core", staticmethod(single_root))
    with pytest.raises(torch.OutOfMemoryError, match="single root"):
        trial(data)


@pytest.mark.parametrize("ordered", [True, False])
def test_receiver_tiles_preserve_edge_order_and_empty_receivers(ordered):
    from equivariantx_foundation.execution.domains import IncomingEdges
    from equivariantx_foundation.execution.receiver import LayerwiseReceiver

    types = torch.arange(7)
    source = torch.tensor([1, 5, 6, 6, 0])
    target = torch.tensor([0, 0, 1, 4, 6])
    index = torch.stack((source, target))
    if not ordered:
        index = index[:, [2, 1, 4, 0, 3]]
    assert IncomingEdges(index, len(types)).max_degree == 2
    assert IncomingEdges(index[:, :0], len(types)).max_degree == 0
    tiles = LayerwiseReceiver.plan(SimpleNamespace(tile=2), types, index)
    recovered = torch.full_like(index, -1)
    for local, nodes, edges, ij, s, species, batch in tiles:
        assert isinstance(edges, slice) == ordered
        torch.testing.assert_close(species[ij[0]], types[nodes[s]])
        recovered[0, edges] = nodes[s]
        recovered[1, edges] = ij[1] + local.start
    torch.testing.assert_close(recovered, index)
    assert not tiles[1][4].numel()
    state = torch.randn(7, 9)
    scalars = torch.randn(7, 2)
    seed = LayerwiseReceiver.seed(state, scalars)
    torch.testing.assert_close(seed[:, :2], scalars)
    assert not seed[:, 2:].count_nonzero()
