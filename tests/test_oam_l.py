"""Checkpoint contracts, force derivatives and training buckets for TACE-OAM-L."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from e3nn import o3

from equivariantx.models.tace.tace_oam_l import optimize_oam_l
from equivariantx.models.tace.tace_oam_l.angular import edge_attributes
from equivariantx.models.tace.tace_oam_l.convolution import bias_convolution
from equivariantx.models.tace.tace_oam_l.element import OAMLElementLinear
from equivariantx.models.tace.tace_oam_l.radial import (
    affine_heads,
    layer_radial,
    prepare_radial,
    project,
)
from equivariantx.models.tace.training import _efs_loss, _pack
from equivariantx.o3.cgtp.convolution import O3TensorProductConv
from tace.models.linear import e3nnElementLinear
from tace.models.mlp import MLPV1


def test_rejects_unrelated_models():
    with pytest.raises(ValueError, match="TACE-OAM-L"):
        optimize_oam_l(torch.nn.Linear(64, 64))
    with pytest.raises(ValueError, match="memory_mode"):
        optimize_oam_l(torch.nn.Linear(64, 64), memory_mode="cached")


def test_shared_radial_projection_keeps_independent_heads_and_mixed_derivatives():
    from equivariantx.models.tace.tace_oam_7m.model import project_radial_heads

    rng = torch.Generator().manual_seed(17)

    def tensor(*shape):
        return torch.randn(
            *shape, generator=rng, dtype=torch.float64, requires_grad=True
        )

    radial = tensor(5, 3)
    heads = [(tensor(3, k), tensor(4, k), tensor(4, k), tensor(k)) for k in (2, 4, 3)]
    source, target = torch.tensor([0, 2, 0, 2, 3]), torch.tensor([2, 0, 1, 1, 3])
    ref = tuple(
        torch.addmm(b, radial, w) + t[target] + s[source] for w, s, t, b in heads
    )
    packed = project_radial_heads(radial, heads, source, target)
    torch.testing.assert_close(packed, ref)
    operands = [radial, *(v for head in heads for v in head)]

    def derivatives(values):
        loss = sum(v.sin().sum() for v in values)
        first = torch.autograd.grad(loss, operands, create_graph=True)
        mixed = torch.autograd.grad(sum(x.square().sum() for x in first), operands)
        return first, mixed

    torch.testing.assert_close(derivatives(packed), derivatives(ref))


def test_stream_message_captures_its_force_adjoint():
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.models.tace.tace_oam_l.tangent import stream_message

    def force_graph(x, y, r, idx):
        dummy = x.detach()
        z, d = stream_message(
            x,
            y,
            r,
            dummy,
            dummy,
            idx,
            idx,
            idx,
            idx,
            dummy,
            [dummy] * 10,
            1e-5,
            1e-5,
            "",
        )
        return torch.autograd.grad(z.square().sum() + d.square().sum(), (x, y, r))

    x, y, r = (torch.randn(7, width, requires_grad=True) for width in (64, 4, 1))
    inputs = (x, y, r, torch.arange(7))
    traced = make_fx(force_graph, tracing_mode="fake")(*inputs)
    exported = torch.export.export(traced, inputs)
    ops = {n.target for n in exported.graph.nodes if n.op == "call_function"}
    assert torch.ops.eqx_oam_l_tangent.stream_message.default in ops
    assert torch.ops.eqx_oam_l_tangent.vjp.default in ops


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA frozen message")
def test_streamed_heads_cross_tiles_and_match_released_message(monkeypatch):
    from equivariantx.models.tace.tace_oam_l import tangent
    from equivariantx.utils.metadata import compact_metadata

    torch.manual_seed(13)
    spec = o3.TensorProduct(
        "64x0e",
        "0e",
        "64x0e",
        [(0, 0, 0, "uvu", True)],
        shared_weights=False,
        internal_weights=False,
    )
    conv = O3TensorProductConv(spec, backend="cuda")
    idx = torch.randint(0, 11, (2, 73), device="cuda")

    def rand(*shape):
        return torch.randn(*shape, device="cuda")

    x, y, r = (
        t.requires_grad_() for t in (rand(11, 64), rand(73, 1), rand(73, 1).double())
    )
    p = [
        rand(64, 192),
        rand(11, 192),
        rand(11, 192),
        rand(192),
        rand(192),
        rand(192),
        rand(64, 1),
        rand(1),
        rand(128, 64) / 12,
        rand(64),
    ]
    args = (
        x,
        y,
        r,
        rand(73, 64),
        rand(73, 64),
        *idx,
        *idx,
        x.detach(),
        p,
        1e-5,
        1e-5,
        compact_metadata(conv.kernel_metadata),
    )
    cot = rand(11, 64), rand(73, 1)

    def evaluate(op):
        z, d = op(*args)
        return z, d, *torch.autograd.grad((z, d), (x, y, r), cot)

    expected = evaluate(tangent.message)
    for tile in (4, 65_536):
        monkeypatch.setattr(tangent, "HEAD_TILE", tile)
        for got, ref in zip(evaluate(tangent.stream_message), expected, strict=True):
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=5e-5)


def test_foundation_capacity_retries_only_allocation_failures(monkeypatch):
    import io

    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from equivariantx.models.tace.tace_oam_l.lammps import OAMLammpsCalc
    from equivariantx.models.tace.tace_oam_l.receiver import TiledFoundationLammpsCalc

    buf = io.BytesIO()
    torch.save(torch.nn.Identity(), buf)
    calc = object.__new__(TiledFoundationLammpsCalc)
    calc.receiver_bytes = buf.getvalue()
    calc.device = torch.device("cpu")
    calc.force_graph = calc.model = object()
    batch = {"x": torch.randn(2)}
    cases = (
        ValueError("invalid graph"),
        RuntimeError("invalid kernel"),
        torch.OutOfMemoryError("test"),
    )
    for error in cases:

        def fail(self, batch, n, e):
            raise error

        monkeypatch.setattr(OAMLammpsCalc, "local_forces", fail)
        if not isinstance(error, torch.OutOfMemoryError):
            with pytest.raises(type(error), match="invalid"):
                calc.local_forces(batch, 2, 2)
            assert calc.model is not None
        else:
            monkeypatch.setattr(calc, "build_receiver", lambda model, **opts: model)
            assert calc.local_forces(batch, 2, 2) is batch
            assert calc.force_graph is calc.model is None

    from equivariantx.models.tace.tace_oam_7m.receiver import SevenReceiverTiles
    from equivariantx.models.tace.tace_oam_l.receiver import ReceiverTiles

    seven = object.__new__(SevenReceiverTiles)
    seven.tile = seven.capacity_limit = 4096
    widths = []

    for failure in (
        torch.OutOfMemoryError("tile"), RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
    ):
        seven.tile = seven.capacity_limit = 4096
        widths.clear()

        def too_wide(self, batch, checkpoint):
            widths.append(self.capacity_limit)
            if self.capacity_limit > 1024:
                raise failure
            return batch

        monkeypatch.setattr(ReceiverTiles, "evaluate", too_wide)
        assert seven.evaluate(batch, False) is batch and widths == [4096, 1024]
    seven.tile = seven.capacity_limit = 4096

    def invalid(self, batch, checkpoint):
        raise RuntimeError("invalid receiver")

    monkeypatch.setattr(ReceiverTiles, "evaluate", invalid)
    with pytest.raises(RuntimeError, match="invalid receiver"):
        seven.evaluate(batch, False)
    assert seven.capacity_limit == 4096

    def full(self, batch, checkpoint):
        raise RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")

    monkeypatch.setattr(ReceiverTiles, "evaluate", full)
    with pytest.raises(torch.OutOfMemoryError, match="CUBLAS_STATUS_ALLOC_FAILED"):
        seven.evaluate(batch, False)
    assert seven.tile == seven.capacity_limit == 1024


def test_foundation_dispatch_rejects_missing_representation():
    from equivariantx.models.tace.foundation import (
        foundation_architecture,
        representation_architecture,
    )

    assert representation_architecture(None) is None
    with pytest.raises(ValueError, match="No specialized"):
        foundation_architecture(SimpleNamespace(readout_fn=torch.nn.Identity()))
    from tace.models.spherical.inter import O2CgtpInteraction
    from tace.models.spherical.prod import CgtpACE

    other = SimpleNamespace(
        num_channel=64,
        interactions=[object.__new__(O2CgtpInteraction)] * 5,
        products=[object.__new__(CgtpACE)] * 5,
    )
    assert representation_architecture(other) is None


def test_rra_throughput_selects_one_graph_for_available_workspace(monkeypatch):
    from contextlib import nullcontext

    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from equivariantx.models.tace.tece_oam_rra.lammps import (
        StreamedRRALammpsCalc,
        ThroughputRRALammpsCalc,
    )

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (70 << 30, 72 << 30))
    monkeypatch.setattr(
        StreamedRRALammpsCalc, "_initialize_device", lambda self, data: None
    )
    for n, selected, tiled in (
        (625, "native", False),
        (5000, "streamed", False),
        (13720, "streamed", False),
        (40000, "streamed", True),
    ):
        calc = object.__new__(ThroughputRRALammpsCalc)
        calc.package_filename, calc.package_bytes = "native", b"0"
        calc.streamed = "streamed", b"1"
        data = SimpleNamespace(nlocal=n, npairs=74 * n, elems=torch.empty(0))
        calc._initialize_device(data)
        assert calc.package_filename == selected
        assert calc.needs_tiles(data) is tiled

    calc = object.__new__(ThroughputRRALammpsCalc)
    calc.package_filename, calc.package_bytes = "native", b"0"
    calc.saved, calc.streamed = ("saved", b"1"), ("streamed", b"2")
    calc.device, calc.initialized, calc.receiver = torch.device("cpu"), True, None
    data = SimpleNamespace(nlocal=40, npairs=74 * 40, elems=torch.empty(0))
    calc._initialize_device(data)
    calls, unloads = [], []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: unloads.append(calc.graph_mode)
    )
    monkeypatch.setattr(
        StreamedRRALammpsCalc,
        "compute_forces",
        lambda self, _: calls.append(self.graph_mode),
    )
    for n in (40, 625, 40, 13720, 5000):
        data.nlocal, data.npairs = n, 74 * n
        calc.compute_forces(data)
    assert calls == ["indexed", "saved", "saved", "streamed", "streamed"]
    assert unloads == ["indexed", "saved"]


@pytest.mark.parametrize("matrix", [False, True])
def test_message_transpose_preserves_individual_path_layout_and_normalization(matrix):
    from equivariantx.models.tace.tace_oam_l.convolution import MessageTranspose
    from tace.models.layout import LayoutTransform
    from tace.models.linear import e3nnLinear

    raw = o3.Irreps("3x0e + 2x1o + 2x1o + 2x1o + 4x2e + 4x2e")
    linear = (
        e3nnLinear(
            raw.simplify(),
            o3.Irreps("5x0e + 3x1o + 2x2e"),
            bias=True,
            use_matrix_weight="1" if matrix else "0",
        )
        .double()
        .requires_grad_(False)
    )
    layout = LayoutTransform(
        raw, layout_in="flatten_ir_mul", layout_out="flatten_mul_ir"
    )
    rng = torch.Generator().manual_seed(42)
    primal = torch.randn(
        7, raw.dim, generator=rng, dtype=torch.float64, requires_grad=True
    )
    cotangent = torch.randn(
        7, linear.irreps_out.dim, generator=rng, dtype=torch.float64
    )
    (expected,) = torch.autograd.grad(linear(layout(primal)), primal, cotangent)
    torch.testing.assert_close(
        MessageTranspose(linear, raw)(cotangent), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA receiver grouping")
def test_receiver_groups_follow_current_species_and_policy():
    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from equivariantx.models.tace.tace_oam_l.receiver import ReceiverTiles

    engine = SimpleNamespace(
        tile=1024,
        retain_messages=True,
        rep=SimpleNamespace(model_dtype=torch.float32, num_elements=3),
    )
    target = torch.arange(1031, device="cuda")
    edges = torch.stack((target.roll(1), target))
    for types in (target.remainder(3), target.remainder(5).clamp_max(2)):
        ReceiverTiles.plan(engine, types, edges)
        assert engine.coefficient_groups.keys() == {0}
        rows, owners = engine.coefficient_groups[0]
        valid = rows < 1024
        torch.testing.assert_close(rows[valid].sort().values, target[:1024])
        torch.testing.assert_close(
            types[rows[valid]], owners.repeat_interleave(16)[valid]
        )
    engine.retain_messages = False
    ReceiverTiles.plan(engine, types, edges)
    assert not engine.coefficient_groups


@pytest.mark.parametrize("prefix", [False, True])
def test_receiver_replays_evicted_states_after_allocation_failure(prefix):
    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from equivariantx.models.tace.tace_oam_l.receiver import ReceiverTiles

    class Tiles(ReceiverTiles):
        def __init__(self, fail=False):
            self.tile, self.retain_messages = 2, False
            self.replay_first_at = self.replay_prefix_at = None
            n = 5 if prefix else 3
            self.rep = SimpleNamespace(
                interactions=range(n), irreps_outs=[o3.Irreps("3x0e")] * n
            )
            self.first, self.readout_widths = torch.empty(0), [1] * n
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


def test_receiver_full_graph_storage_bound(monkeypatch):
    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from equivariantx.models.tace.tace_oam_l.receiver import ReceiverTiles

    state = torch.empty(3, 7)
    vectors = torch.empty(11, 3, dtype=torch.float64)
    edge_index = torch.empty(2, 11, dtype=torch.int64)
    inverse = torch.empty(11, dtype=torch.int64)
    engine = SimpleNamespace(
        first=state,
        readout_widths=[1, 2],
        rep=SimpleNamespace(irreps_outs=[o3.Irreps("7x0e"), o3.Irreps("9x0e")]),
    )
    # Only count actual buffers: node state, its cotangent, output cotangent,
    # both COO matrices, unique's inverse indices, vectors and their cotangents.
    edges = 2 * edge_index.nbytes + inverse.nbytes + 2 * vectors.nbytes
    assert ReceiverTiles.full_graph_min_bytes(engine, 3, 11, 8) == (
        2 * state.nbytes + state.new_empty(3, 9).nbytes + edges
    )
    engine.readout_widths[0] = 7
    assert ReceiverTiles.full_graph_min_bytes(engine, 3, 11, 8) == edges

    receiver = object.__new__(ReceiverTiles)
    receiver.first = state
    receiver.readout_widths = [64] * 5
    receiver.rep = SimpleNamespace(
        irreps_outs=[SimpleNamespace(dim=d) for d in (2304, 2304, 2304, 2304, 64)]
    )
    # Budget left after allocating the graph and the force accumulator.
    assert receiver.domain_count(2_282_665, 55 << 30) == 8
    assert receiver.domain_count(4_705_960, 42 << 30) == 32
    assert receiver.domain_count(3, 1) == 3
    width = SimpleNamespace(
        linear_down=SimpleNamespace(irreps_out=SimpleNamespace(dim=2304))
    )
    receiver.rep.interactions = [width] * 5
    groups = [range(10000)] * 5
    assert receiver.domain_tile(groups, [[]] * 5, 80, 40 << 30) == 4096
    assert receiver.domain_tile(groups, [[]] * 5, 80, 4 << 30) == 1024
    assert receiver.domain_tile(groups, [[]] * 5, 1000, 40 << 30) == 1024
    receiver.domain_tile_limit = 2048
    assert receiver.domain_tile(groups, [None] * 5, 80, 40 << 30) == 2048

    from equivariantx.models.tace.tace_oam_7m.receiver import (
        ReceiverWorkspace,
        SevenReceiverTiles,
    )

    layout = ReceiverWorkspace(41984, 2048, 7552, 1920)
    resident = layout.resident(1080000, 79920000)

    def fits(rows, degree, halo, available=63 << 30):
        active = layout.active(rows, rows * degree, halo)
        return layout.fits(resident + active, available, 72 << 30)

    assert fits(4096, 74, 10000)
    assert not fits(4096, 74, 10000, available=47 << 30)
    assert not fits(4096, 560, 10000)
    assert not fits(4096, 74, 150000)
    assert fits(2048, 200, 18000)

    seven = object.__new__(SevenReceiverTiles)
    seven.workspace = layout
    types = torch.empty(1080000, 0)
    index = torch.empty(1).expand(2, 79920000)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    for free, degree, sources, wide in (
        (63, 74, 10000, True), (47, 74, 10000, False),
        (63, 560, 10000, False), (63, 74, 150000, False),
    ):
        seven.capacity_limit, seven.last_capacity_tile = 4096, None
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda device: (free << 30, 72 << 30)
        )

        def plan(self, types, index):
            halo = range(sources * self.tile // 4096)
            edges = range(degree * self.tile)
            return [(None, halo, None, None, edges, None, range(self.tile))]

        monkeypatch.setattr(ReceiverTiles, "plan", plan)
        assert seven.plan(types, index)
        assert (seven.tile == 4096) == wide


def test_receiver_domains_retain_only_roots_and_restart_partial_derivatives(
    monkeypatch,
):
    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from equivariantx.models.tace.tace_oam_l.compact import (
        CompactRows,
        NodalForce,
        compute_domains,
    )
    from equivariantx.models.tace.tace_oam_l.domains import EnergyDomains
    from equivariantx.models.tace.tace_oam_l.receiver import ReceiverTiles

    class Tiles(ReceiverTiles):
        def __init__(self, fail):
            self.retain_messages = False
            self.tile, self.replay_first_at, self.replay_prefix_at = 3, None, None
            self.coefficient_groups = {}
            self.first = torch.zeros((), dtype=torch.float64)
            self.readout_widths = [1] * 3
            self.rep = SimpleNamespace(
                interactions=range(3), irreps_outs=[o3.Irreps("3x0e")] * 3
            )
            self.policies, self.fail = [], fail

        def layer(self, i, tile, v, xs, xc):
            target = tile[3][1]
            x = v.new_zeros(len(tile[-1]), 3).index_add(0, target, v)
            if i:
                x = (x.index_add(0, target, xs[tile[4]]) + xc).tanh()
            return x, None, None

        def readout(self, tile, v, ds, total_atoms):
            assert total_atoms == 12
            return sum(x[:, 0].square() for x in ds).index_add(
                0, tile[3][1], v.square().sum(-1)
            )

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
    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from equivariantx.models.tace.tace_oam_l.domains import IncomingEdges
    from equivariantx.models.tace.tace_oam_l.receiver import ReceiverTiles

    types = torch.arange(7)
    source = torch.tensor([1, 5, 6, 6, 0])
    target = torch.tensor([0, 0, 1, 4, 6])
    index = torch.stack((source, target))
    if not ordered:
        index = index[:, [2, 1, 4, 0, 3]]
    assert IncomingEdges(index, len(types)).max_degree == 2
    assert IncomingEdges(index[:, :0], len(types)).max_degree == 0
    tiles = ReceiverTiles.plan(SimpleNamespace(tile=2), types, index)
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
    seed = ReceiverTiles.seed(state, scalars)
    torch.testing.assert_close(seed[:, :2], scalars)
    assert not seed[:, 2:].count_nonzero()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution contract")
@pytest.mark.parametrize("out", ["5x0e + 3x1o", "5x0e"])
def test_frozen_indexed_adjoint(out):
    layer = (
        e3nnElementLinear(
            "8x0e + 6x1o", out, bias=True, num_elements=7, use_matrix_weight="0"
        )
        .double()
        .cuda()
        .requires_grad_(False)
    )
    inputs = torch.randn(
        9, layer.irreps_in.dim, device="cuda", dtype=torch.float64, requires_grad=True
    )
    types = torch.tensor([4, 2, 0, 2, 5, 3, 0, 4, 3], device="cuda")
    cot = torch.randn(
        9, layer.irreps_out.dim, device="cuda", dtype=torch.float64, requires_grad=True
    )

    def evaluate(function):
        y = function(inputs, types)
        (dx,) = torch.autograd.grad(y, inputs, cot, create_graph=True)
        (dy,) = torch.autograd.grad(dx.sum(), cot)
        return y, dx, dy

    expected = evaluate(layer)
    layer.__class__ = OAMLElementLinear
    layer.prepare_plan()
    for a, b in zip(evaluate(layer), expected, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution contract")
def test_grouped_frozen_map_with_inactive_reverse_columns():
    from equivariantx.models.tace.tace_oam_l.element_grouped import grouping

    # N and input width exceed the grouped dispatch threshold. Only one irrep
    # maps to the output: its transpose must zero all other feature columns.
    torch.manual_seed(42)
    layer = (
        e3nnElementLinear(
            "384x0e + 448x2e",
            "64x2e",
            bias=False,
            num_elements=7,
            use_matrix_weight="0",
        )
        .cuda()
        .requires_grad_(False)
    )
    layer.__class__ = OAMLElementLinear
    layer.prepare_plan()
    z = torch.randint(7, (529,), device="cuda")
    x = torch.randn(529, layer.irreps_in.dim, device="cuda", requires_grad=True)
    cot = torch.randn(529, layer.irreps_out.dim, device="cuda")

    def evaluate(groups):
        y = layer(x, z, groups)
        return y, torch.autograd.grad(y, x, cot)[0]

    baseline = evaluate(None)
    for a, b in zip(evaluate(grouping(z, 7)), baseline, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_cached_checkpoint_registration_and_flags():
    path = Path.home() / ".cache" / "tace" / "TACE-OAM-L.pt"
    if not path.is_file():
        pytest.skip("no locally cached foundation checkpoint")
    from tace.lightning import load_tace

    ref = load_tace(
        path, device="cpu", dtype="float32", target_property=["energy", "forces"]
    ).eval()
    opt = optimize_oam_l(ref, memory_mode="recompute")
    assert not opt.training and opt.get_target_property() == ref.get_target_property()
    assert opt.state_dict().keys() == ref.state_dict().keys()
    assert all(a is b for a, b in zip(opt.parameters(), ref.parameters(), strict=True))
    assert opt.readout_fn is not ref.readout_fn
    assert opt.readout_fn.representation.memory_mode == "recompute"
    assert optimize_oam_l(opt, memory_mode="recompute") is opt
    fast = optimize_oam_l(
        opt, fused_geometry=True, fused_projection=True, grouped_coefficients=True
    )
    assert fast.readout_fn.representation.grouped_coefficients
    assert not opt.readout_fn.representation.grouped_coefficients
    with pytest.raises(ValueError, match="fused_projection"):
        optimize_oam_l(ref, memory_mode="recompute", fused_projection=True)


@pytest.mark.parametrize("frozen", [True, False])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_conditioned_radial(device, frozen):
    torch.manual_seed(127)
    dtype = torch.float32 if device == "cuda" else torch.float64
    layers = (
        torch.nn.ModuleList(
            [
                MLPV1([192, 128, 384], layer_norm=True, bias=True),
                MLPV1([192, 64, 1], layer_norm=True, bias=True),
                torch.nn.Embedding(89, 64),
                torch.nn.Embedding(89, 64),
            ]
        )
        .to(device=device, dtype=dtype)
        .requires_grad_(not frozen)
    )
    info, dens, target, source = layers
    inter = SimpleNamespace(
        edge_info=info, scatter_norm=SimpleNamespace(edge_density=dens)
    )
    update = SimpleNamespace(target_embedding=target, source_embedding=source)
    E, N = 83, 14
    edge = torch.randint(0, N, (2, E), device=device)
    species = torch.randint(0, 89, (N,), device=device)
    x = torch.randn(E, 64, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn(E, 384, dtype=dtype, device=device)
    dd = torch.randn(E, 1, dtype=dtype, device=device)
    dx = torch.randn_like(x)
    args = [x] if frozen else [x, *layers.parameters()]
    tol = (
        dict(atol=6e-5, rtol=8e-5) if device == "cuda" else dict(atol=1e-10, rtol=1e-10)
    )

    def actual():
        h, w, b, d = layer_radial(inter, update, x, species, edge)
        return project(h, w, b), d

    def aligned():
        p = prepare_radial(inter, update)
        h, w, d = affine_heads(p, x, species[edge[0]], species[edge[1]])
        assert h.shape == (E, 136)
        return h @ w, d

    def reference():
        features = ((x, None), (target(species), edge[1]), (source(species), edge[0]))
        return info(features), dens(features).square().tanh()

    def first(order, f):
        y = f()
        g = torch.autograd.grad(y, args, (dy, dd), create_graph=order == 2)
        return y, g

    if device == "cuda" and frozen:
        from equivariantx.models.tace.tace_oam_l.radial import _torch_heads
        from equivariantx.models.tace.tace_oam_l.tangent_kernels import heads

        params = prepare_radial(inter, update)
        st, tt = species[edge[0]], species[edge[1]]
        values, tangents = torch.func.jvp(
            lambda features: _torch_heads(params, features, st, tt), (x,), (dx,)
        )
        result = heads(params, x.detach(), dx, st, tt)
        for a, b in zip(result, (*values, *tangents), strict=True):
            torch.testing.assert_close(a, b, **tol)

    for order in (1, 2):
        ref, grad = first(order, reference)
        for fn in (actual, aligned):
            val, pred = first(order, fn)
            for a, b in zip((*val, *pred), (*ref, *grad), strict=True):
                torch.testing.assert_close(a, b, **tol)
        if order == 2:
            # Also test a force loss: mixed parameter/geometry derivatives.
            g2 = torch.autograd.grad((grad[0] * dx).sum(), args, materialize_grads=True)
            p2 = torch.autograd.grad((pred[0] * dx).sum(), args, materialize_grads=True)
            for a, b in zip(p2, g2, strict=True):
                if a is None or b is None:
                    assert a is b
                else:
                    torch.testing.assert_close(a, b, **tol)

    # Simulate a parameter update and a new geometry without rebuilding either MLP.
    with torch.no_grad():
        target.weight.mul_(0.8)
        x.add_(0.07)
    for a, b in zip(actual(), reference(), strict=True):
        torch.testing.assert_close(a, b, **tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA geometry kernel")
@pytest.mark.parametrize("cutoff_dtype", [torch.float32, torch.float64])
def test_y5_geometry_derivatives(cutoff_dtype):
    torch.manual_seed(287)
    device = "cuda"
    v = torch.randn((257, 3), device=device, dtype=torch.float64)
    v[:3] = torch.eye(3, device=device, dtype=torch.float64)
    v[3] = 0
    v[4] = torch.tensor([1e-14, -1e-14, 0.0], device=device)
    v.requires_grad_()
    # Use independent lengths for the singular points as well as physical,
    # position-dependent lengths and cutoff for the other edges.
    length = torch.cat((v[:5, :1] * 0 + 1e-14, v[5:].norm(dim=-1, keepdim=True)), 0)
    length = torch.cat((v[:3].norm(dim=-1, keepdim=True), length[3:]), 0)
    cutoff = (0.5 + torch.sigmoid(length)).to(cutoff_dtype)
    g = torch.randn((257, 36), device=device, requires_grad=True)
    tangent = torch.randn_like(v)

    def evaluate(native, create_graph):
        d = v / length.clamp_min(1e-12)
        y = (
            edge_attributes(v, length, cutoff)
            if native
            else o3.spherical_harmonics(list(range(6)), d, False, "component").float()
            * cutoff.float()
        )
        (grad,) = torch.autograd.grad(
            y, v, g, create_graph=create_graph, retain_graph=True
        )
        if create_graph:
            hv, dg = torch.autograd.grad(grad, (v, g), tangent, retain_graph=True)
            return y, grad, hv, dg
        return y, grad

    for higher in (False, True):
        for a, b in zip(evaluate(True, higher), evaluate(False, higher)):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        edge_attributes(v, length, None),
        o3.spherical_harmonics(
            list(range(6)), v / length.clamp_min(1e-12), False, "component"
        ).float(),
    )
    # Go beyond the fused mixed derivative on a regular geometry.
    u = v.detach()[5:10].requires_grad_()
    r = u.norm(dim=-1, keepdim=True)
    c = r.sigmoid().to(cutoff_dtype)
    reference = (
        o3.spherical_harmonics(list(range(6)), u / r, False, "component").float()
        * c.float()
    )
    thirds = []
    for y in (reference, edge_attributes(u, r, c)):
        dy = torch.autograd.grad(y, u, g[5:10], create_graph=True)[0]
        ddy = torch.autograd.grad(dy, u, tangent[5:10], create_graph=True)[0]
        thirds.append(torch.autograd.grad(ddy, u, tangent[5:10], retain_graph=True)[0])
    torch.testing.assert_close(*thirds, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("node_grad", [True, False])
@pytest.mark.parametrize("channels,hidden,lmax", [(64, 128, 5), (128, 64, 3)])
def test_released_convolution_widths(node_grad, channels, hidden, lmax):
    torch.manual_seed(983)
    sh = o3.Irreps.spherical_harmonics(lmax)
    spec = o3.TensorProduct(
        [(channels, o3.Irrep("0e"))],
        sh,
        [(channels, ir) for _, ir in sh],
        [(0, j, j, "uvu", True) for j in range(len(sh))],
        shared_weights=False,
        internal_weights=False,
    )
    conv = O3TensorProductConv(spec, backend="cuda").cuda()
    x = torch.randn(12, channels, device="cuda", requires_grad=node_grad)
    idx = torch.randint(0, 12, (2, 37), device="cuda")
    h = torch.randn(37, hidden, device="cuda", requires_grad=True)
    a = torch.randn(37, sh.dim, device="cuda", requires_grad=True)
    w = torch.randn(hidden, spec.weight_numel, device="cuda") / hidden**0.5
    b = torch.randn(spec.weight_numel, device="cuda")
    cot = torch.randn(12, conv.output_dim, device="cuda")
    tangent = torch.randn_like(h)
    args = (a, h, x) if node_grad else (a, h)
    if node_grad:
        from equivariantx.models.tace.tace_oam_l.tangent_kernels import (
            coordinate_adjoint,
        )
        from equivariantx.utils.metadata import compact_metadata

        rows = torch.addmm(b, h, w).detach().requires_grad_()
        path_tangent = tangent @ w
        z = conv(x, a, rows, rows[:0], idx, num_nodes=12)
        gx, gy, gw = torch.autograd.grad(z, (x, a, rows), cot)
        dx, dy, dr = torch.zeros_like(x), torch.zeros_like(a), h.new_zeros(len(a), 1)
        coordinate_adjoint(
            compact_metadata(conv.kernel_metadata),
            *idx,
            x,
            a,
            rows,
            cot,
            dx,
            dy,
            dr,
            path_tangent,
        )
        for actual, expected in zip(
            (dx, dy, dr),
            (gx, gy, (gw * path_tangent).sum(-1, keepdim=True)),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=6e-5, atol=5e-5)
    for higher in (False, True):
        vals = []
        for fused in (False, True):
            if fused:
                z = bias_convolution(conv, x, a, h, w, b, idx, 12)
            else:
                z = conv(
                    x,
                    a,
                    torch.addmm(b, h, w),
                    w.new_empty(0, spec.weight_numel),
                    idx,
                    num_nodes=12,
                )
            dg = torch.autograd.grad(z, args, cot, create_graph=higher)
            hv = (torch.autograd.grad(dg[1], a, tangent)[0],) if higher else ()
            vals.append((z, *dg, *hv))
        for actual, expected in zip(vals[1], vals[0], strict=True):
            torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-5)


def _batch():
    return dict(
        positions=torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=torch.float64),
        ptr=torch.tensor([0, 2]),
        batch=torch.zeros(2, dtype=torch.long),
        atomic_numbers=torch.tensor([32, 52]),
        fidelity_idx=torch.tensor([0]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_shifts=torch.zeros(2, 3, dtype=torch.long),
        lattice=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 4,
        pbc=torch.ones(1, 3, dtype=torch.bool),
        energy=torch.tensor([2.1], dtype=torch.float64),
        forces=torch.arange(6, dtype=torch.float64).reshape(2, 3),
        stress=torch.eye(3, dtype=torch.float64).unsqueeze(0),
    )


@pytest.mark.parametrize("stress", (True, False))
def test_oam_l_bucket_is_disconnected_and_ignores_padding_in_loss(stress):
    data = _batch()
    packed = _pack(data, 4, 9, 1, stress)
    assert packed["ptr"].tolist() == [0, 2, 5]
    torch.testing.assert_close(packed["edge_index"][:, :2], data["edge_index"])
    src, dst = packed["edge_index"][:, 2:]
    assert src.min() >= 2 and src.max() < 5
    torch.testing.assert_close(src, dst)
    r = packed["edge_shifts"][2:].double() @ packed["lattice"][-1]
    assert (r.norm(dim=1) > 6.0).all()
    theta = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    pred = dict(
        energy=packed["energy"] * theta + 0.1, forces=packed["positions"] * theta
    )
    if stress:
        pred["stress"] = packed["lattice"] * theta
    val = _efs_loss(pred, packed)
    ref = ((pred["energy"][:1] - data["energy"]) / 2).square().mean()
    ref = ref + (pred["forces"][:2] - data["forces"]).square().mean()
    if stress:
        ref = ref + (pred["stress"][:1] - data["stress"]).square().mean()
    from tace.utils.loss.normal import NormalLoss

    targets = ["energy", "forces"] + (["stress"] if stress else [])
    names = ["mse_energy_per_atom", "mse_forces"] + (["mse_stress"] if stress else [])
    normal = NormalLoss(targets, names, [1.0] * len(names))
    normalized = normal(pred, _pack(data, 4, 9, 1, stress, joint=True))
    torch.testing.assert_close(normalized, ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(val, ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(val, theta, retain_graph=True)[0],
        torch.autograd.grad(ref, theta)[0],
        atol=1e-12,
        rtol=1e-12,
    )


def test_oam_l_bucket_rejects_missing_labels_and_overflow():
    data = _batch()
    with pytest.raises(ValueError, match="bucket"):
        _pack(data, 1, 9, 1, True)
    data["forces_weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unit forces_weight"):
        _pack(data, 4, 9, 1, True)


def test_oam_l_multibucket_reuses_graphs_and_optimizer(monkeypatch):
    from equivariantx.models.tace import training

    weight = torch.nn.Parameter(torch.tensor(1.0))
    reference = torch.nn.Parameter(weight.detach().clone())
    optimizer = torch.optim.AdamW([weight])
    baseline = torch.optim.AdamW([reference])
    captures = []

    def update(p, opt, batch):
        opt.zero_grad(set_to_none=True)
        (p * batch["positions"].sum() - 0.3).square().backward()
        opt.step()

    def capture(p, opt, batch, *, max_atoms, max_edges, joint):
        captures.append((max_atoms, max_edges))
        return lambda b: update(p, opt, b)

    monkeypatch.setattr(training, "BucketedEFSTrainingStep", capture)
    step = training.MultiBucketEFSTrainingStep(weight, optimizer, [(3, 4), (2, 2)])
    with pytest.raises(ValueError, match="FP32 CUDA OAM-L"):
        training.MultiBucketEFSTrainingStep(
            weight, optimizer, [(2, 2)], native_product=True
        )
    a = _batch()
    b = {
        **a,
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
        "positions": torch.ones(3, 3),
        "ptr": torch.tensor([0, 3]),
    }
    assert step.prepare_all([a, a]) is step
    for batch in (a, b, a):
        step.prepare(batch)
        step(batch)
        update(reference, baseline, batch)
        torch.testing.assert_close(weight, reference)
    assert captures == [(2, 2), (3, 4)]
    assert optimizer.state[weight]["step"].item() == 3
    with pytest.raises(ValueError, match="exceeds"):
        step({**b, "positions": torch.ones(4, 3)})
    with pytest.raises(ValueError, match="structures"):
        step({**a, "ptr": torch.tensor([0, 1, 2])})

    with pytest.raises(ValueError, match="non-joint fused AdamW"):
        training.MultiBucketEFSTrainingStep(
            weight, optimizer, [(2, 2)], share_workspace=True
        )


def test_oam_l_prepare_all_phases_and_restores_failed_buckets(monkeypatch):
    from types import SimpleNamespace

    from equivariantx.models.tace import training

    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([p])
    caller_grad = p.grad = torch.ones_like(p)
    native, layer = object(), SimpleNamespace(native_square=None)
    events, failure = [], set()

    def build(model, optimizer, batch, *, max_atoms, _recordings=None, **options):
        def record():
            assert layer.native_square is native
            events.append(("record", max_atoms))
            p.grad = torch.zeros_like(p)
            if max_atoms in failure:
                raise RuntimeError("cannot record")

        assert layer.native_square is native
        events.append(("warm", max_atoms))
        p.grad = torch.zeros_like(p)
        if _recordings is None:
            record()
        else:
            _recordings.append(record)
        return lambda b: max_atoms

    monkeypatch.setattr(training, "BucketedEFSTrainingStep", build)
    step = training.MultiBucketEFSTrainingStep(p, optimizer, [(2, 2), (3, 3), (4, 4)])
    step.workspace = SimpleNamespace(
        params=(p,), check=lambda: None, publish=lambda x: x
    )
    step._native_products = ((layer, native),)
    a = _batch()
    b = {**a, "positions": torch.ones(3, 3)}
    c = {**a, "positions": torch.ones(4, 3)}
    failure.add(3)
    with pytest.raises(RuntimeError, match="cannot record"):
        step.prepare_all([a, b])
    assert step.graphs is None and not step._steps
    assert p.grad is caller_grad and layer.native_square is None
    failure.clear()
    events.clear()
    assert step.prepare_all(iter([a, a, b])) is step
    assert events == [("warm", 2), ("warm", 3), ("record", 2), ("record", 3)]
    assert p.grad is caller_grad and layer.native_square is None and not optimizer.state
    events.clear()
    assert step(a) == 2 and step(b) == 3
    assert not events
    assert step(c) == 4 and events == [("warm", 4), ("record", 4)]
    with pytest.raises(ValueError, match="exceeds"):
        step.prepare_all([a, {**b, "positions": torch.ones(7, 3)}])
    assert step(b) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph workspace")
def test_oam_l_serial_workspace_owns_outputs_and_rejects_other_streams():
    from concurrent.futures import ThreadPoolExecutor
    from threading import get_ident

    from equivariantx.models.tace.training import _SerialTrainingWorkspace

    p = torch.nn.Parameter(torch.ones(3, device="cuda"))
    w = object.__new__(_SerialTrainingWorkspace)
    w.params, w.gradients = (p,), [torch.empty_like(p)]
    w.thread, w.caller_stream = get_ident(), torch.cuda.current_stream()
    w.check()
    with (
        torch.cuda.stream(torch.cuda.Stream()),
        pytest.raises(RuntimeError, match="stream"),
    ):
        w.check()
    with ThreadPoolExecutor(1) as worker, pytest.raises(RuntimeError, match="thread"):
        worker.submit(w.check).result()
    private = torch.arange(3.0, device="cuda")
    p.grad = private
    old = w.publish({"forces": private})["forces"]
    private.zero_()
    torch.testing.assert_close(old, torch.arange(3.0, device="cuda"))
    torch.testing.assert_close(p.grad, old)
    p.grad = private
    new = w.publish({"forces": private})["forces"]
    torch.testing.assert_close(p.grad, new)
    torch.testing.assert_close(old, torch.arange(3.0, device="cuda"))


@pytest.mark.parametrize("joint", [False, True])
def test_oam_l_microbatch_uses_full_efs_denominators_and_one_update(joint):
    from equivariantx.models.tace.tace_oam_l.training import BucketedEFSTrainingStep
    from tace.utils.loss.normal import NormalLoss

    a, b = _batch(), _batch()
    for k in ("positions", "forces", "atomic_numbers", "batch"):
        b[k] = b[k][:1]
    b["ptr"] = torch.tensor([0, 1])
    b["edge_index"], b["edge_shifts"] = b["edge_index"][:, :0], b["edge_shifts"][:0]
    b["forces"] = b["forces"] + 3
    theta = torch.nn.Parameter(torch.tensor(1.3, dtype=torch.float64))
    reference = torch.nn.Parameter(theta.detach().clone())
    opt = torch.optim.AdamW([theta], lr=1e-3)
    baseline = torch.optim.AdamW([reference], lr=1e-3)

    def predictions(p, d):
        return dict(
            energy=p * d["energy"] + 0.2,
            forces=p * d["positions"],
            stress=p * d["lattice"],
        )

    predicted = [predictions(reference, d) for d in (a, b)]
    expected = sum(
        ((v["energy"] - d["energy"]) / len(d["positions"])).square().mean() / 2
        + (v["forces"] - d["forces"]).square().sum() / 9
        + (v["stress"] - d["stress"]).square().mean() / 2
        for d, v in zip((a, b), predicted, strict=True)
    )
    expected.backward()
    baseline.step()
    grad = torch.zeros_like(theta)
    normal = NormalLoss(
        ["energy", "forces", "stress"],
        ["mse_energy_per_atom", "mse_forces", "mse_stress"],
        [1.0] * 3,
    )

    def replay(d):
        with torch.enable_grad():
            out = predictions(theta, d)
            loss = normal(out, d) if joint else _efs_loss(out, d, microbatches=2)
            grad.copy_(torch.autograd.grad(loss, theta)[0])
        return {k: v.detach() for k, v in out.items()} | {"loss": loss.detach()}

    step = object.__new__(BucketedEFSTrainingStep)
    step.cap, step.microbatches = (4, 9, 1, True, joint), 2
    step.step = SimpleNamespace(replay=replay, optimizer=opt)
    step.active, step.accumulated = [(theta, grad)], [torch.empty_like(grad)]
    with pytest.raises(ValueError, match="2 structure microbatches"):
        step([a])
    out = step([a, b])
    for k in predicted[0]:
        torch.testing.assert_close(out[k], torch.cat([v[k] for v in predicted]))
    torch.testing.assert_close(out["loss"], expected)
    torch.testing.assert_close(theta.grad, reference.grad)
    torch.testing.assert_close(theta, reference)
    assert opt.state[theta]["step"].item() == 1


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native tiled quadratic kernels"
)
def test_tiled_quadratic_value_and_adjoint(monkeypatch):
    """Exercise the same-operand transpose and a ragged final node tile."""
    from equivariantx.models.tace.tace_oam_l import product as tiled
    from equivariantx.models.tace.tace_oam_l.model import _QuadraticACE

    torch.manual_seed(303)
    inp = "3x0e + 3x1o"
    out = "3x0e + 3x1o + 3x2e"
    tp = o3.TensorProduct(
        inp,
        inp,
        out,
        [
            (0, 0, 0, "uuu", False),
            (0, 1, 1, "uuu", False),
            (1, 1, 0, "uuu", False),
            (1, 1, 2, "uuu", False),
        ],
    )
    coef = OAMLElementLinear(
        out, "2x0e + 2x1o + 1x2e", num_elements=4, bias=True, use_matrix_weight="0"
    )
    head = torch.nn.Module()
    head.tp = tp
    product = torch.nn.Module()
    product.aces = torch.nn.ModuleList([head])
    product.coefs = torch.nn.ModuleList([torch.nn.Identity(), coef])
    product.double().cuda().requires_grad_(False)
    coef.prepare_plan()
    _QuadraticACE.prepare_paths(product)
    monkeypatch.setattr(tiled, "TILE", 3)
    x = torch.randn(
        8, tp.irreps_in1.dim, dtype=torch.float64, device="cuda", requires_grad=True
    )
    z = torch.tensor([0, 1, 2, 3, 0, 3, 2, 1], device="cuda")
    cot = torch.randn(8, coef.irreps_out.dim, dtype=x.dtype, device=x.device)
    expected = e3nnElementLinear.forward(coef, tp(x, x), z)
    actual = tiled.frozen_quadratic(product, x, z)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, cot)[0],
        torch.autograd.grad(expected, x, cot)[0],
        rtol=2e-6,
        atol=2e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native OAM-L square")
def test_native_layout_quadratic_value_vjp_and_hvp():
    from equivariantx.models.tace.tace_oam_l.model import _QuadraticACE
    from equivariantx.models.tace.tace_oam_l.product_kernels import (
        adjoint,
        mixed,
        square,
        square_metadata,
    )

    torch.manual_seed(404)
    inp = "64x0e + 64x1o"
    tp = o3.TensorProduct(
        inp,
        inp,
        "64x0e + 64x0e + 64x1o + 64x2e",
        [
            (0, 0, 0, "uuu", False),
            (1, 1, 1, "uuu", False),
            (0, 1, 2, "uuu", False),
            (1, 1, 3, "uuu", False),
        ],
    ).cuda()
    product = torch.nn.Module()
    head = torch.nn.Module()
    head.tp = tp
    product.aces = torch.nn.ModuleList([head])
    product.register_parameter(
        "dummy", torch.nn.Parameter(torch.empty(0, device="cuda"))
    )
    _QuadraticACE.prepare_paths(product)
    x = torch.randn(11, tp.irreps_in1.dim, device="cuda", requires_grad=True)
    meta = square_metadata(product)
    q = square(meta, x)
    ref = tp(x, x)
    dq = torch.randn_like(ref)
    torch.testing.assert_close(q, ref, rtol=4e-6, atol=3e-6)
    direct = torch.autograd.grad(q, x, dq, retain_graph=True)[0]
    vjps = [torch.autograd.grad(t, x, dq, create_graph=True)[0] for t in (q, ref)]
    for v in [direct, vjps[0]]:
        torch.testing.assert_close(v, vjps[1], rtol=4e-6, atol=3e-6)
    vector = torch.randn_like(x)
    hvps = [torch.autograd.grad(v, x, vector)[0] for v in vjps]
    torch.testing.assert_close(*hvps, rtol=2e-6, atol=2e-6)
    expected = tp(x, vector) + tp(vector, x)
    jvp_native = mixed(meta, x, vector)
    _, jvp_func = torch.func.jvp(lambda t: square(meta, t), (x,), (vector,))
    for actual in (jvp_native, jvp_func):
        torch.testing.assert_close(actual, expected, rtol=5e-6, atol=4e-6)
    _, hessian_func = torch.func.jvp(lambda t: adjoint(meta, t, dq), (x,), (vector,))
    torch.testing.assert_close(hessian_func, hvps[1], rtol=5e-6, atol=4e-6)
    _, mixed_func = torch.func.jvp(lambda t: mixed(meta, t, vector), (x,), (vector,))
    torch.testing.assert_close(mixed_func, 2 * tp(vector, vector), rtol=5e-6, atol=4e-6)
    from torch.fx.experimental.proxy_tensor import make_fx

    def force_graph(t, g):
        y = square(meta, t)
        return y, torch.autograd.grad(y, t, g)[0]

    traced = make_fx(force_graph)(x, dq)
    exported = torch.export.export(traced, (x, dq), strict=False)
    for graph in (
        traced.graph,
        exported.graph_module.graph,
        exported.run_decompositions().graph_module.graph,
    ):
        calls = {node.target for node in graph.nodes}
        assert torch.ops.eqx_oam_l_product.square.default in calls
        assert torch.ops.eqx_oam_l_product.square_vjp.default in calls


def test_rotary_force_export_passes_saved_primals_to_its_pullback():
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.models.tace.tece_oam_rra.force import (
        capped_program,
        saved_interaction,
        saved_nodes,
    )
    from equivariantx.utils.program import Program
    from tace.models.compile.aot import custom_ops_libs_from_model

    p = Program()
    specs = (
        (2, "source"),
        (2, "edge"),
        (2, "shared"),
        (2, "shared"),
        (1, "edge"),
        (1, "edge"),
        (1, "edge"),
    )
    x, c, *_ = [p.input(i, kind, width) for i, (width, kind) in enumerate(specs)]
    h = p.unary("silu", x)
    score = p.slice(c, 0, 1)
    gate = p.scale(p.unary("silu", c), 1.67)
    value = p.product((h, h, x, gate, x, p.constant(1, 0)), (1, (2,) * 6, ()), 5)
    base = tuple(p.nodes), score, value, specs, 1, 2, 1e-12
    assert saved_nodes(base) == (score, c, h)
    metadata = repr((repr((0, 0, 2, 1)), base))

    def force(source, target, values):
        y = saved_interaction(metadata, source, target, values)[0]
        return y, torch.autograd.grad(y.sum(), (values[0], values[1], values[4]))

    source = torch.empty(7, dtype=torch.long, device="meta")
    target = torch.empty_like(source)
    shapes = ((3, 1, 2), (7, 2), (2,), (2,), (7, 1), (7, 1), (7, 1))
    values = [
        torch.empty(s, device="meta", requires_grad=i in (0, 1, 4))
        for i, s in enumerate(shapes)
    ]
    graph_module = make_fx(force)(source, target, values)
    libs = custom_ops_libs_from_model(graph_module)
    assert {"equivariantx", "equivariantx.models.tace.tece_oam_rra.force"} <= libs
    for graph in (
        graph_module.graph,
        torch.export.export(graph_module, (source, target, values), strict=False)
        .run_decompositions()
        .graph_module.graph,
    ):
        calls = {n.target: n for n in graph.nodes if n.op == "call_function"}
        primal = calls[torch.ops.eqx_tece_rra.saved_interaction.default]
        pullback = calls[torch.ops.eqx.tece_contraction.default]
        assert [n.args for n in pullback.args[3][-3:]] == [
            (primal, i) for i in (3, 4, 5)
        ]

    metadata = capped_program(metadata, 2)
    graph = make_fx(force)(source, target, values)
    calls = {n.target: n for n in graph.graph.nodes if n.op == "call_function"}
    primal = calls[torch.ops.eqx_tece_rra.saved_interaction.default]
    assert [v.shape[0] for v in primal.meta["val"][3:]] == [7, 2, 2]
    assert torch.ops.eqx_tece_rra.expansion_pullback.default in calls


def test_rotary_edge_prefix_preserves_shared_receiver_adjoints():
    from equivariantx.models.tace.tece_oam_rra.expressions import first_adjoint
    from equivariantx.models.tace.tece_oam_rra.force import (
        capped_program,
        edge_slots,
        saved_adjoint,
        saved_interaction,
    )
    from equivariantx.ops.common.torch import tece
    from equivariantx.utils.program import Program

    p = Program()
    specs = (
        (2, "source"),
        (2, "edge"),
        (1, "shared"),
        (1, "shared"),
        (1, "edge"),
        (1, "edge"),
        (1, "edge"),
    )
    x, c, *_ = [p.input(i, kind, w) for i, (w, kind) in enumerate(specs)]
    h, s = p.unary("silu", x), p.slice(c, 0, 1)
    w = p.unary("silu", c)
    spec = (1, (2,) * 6, (((0,) * 6, 1.0), ((1,) * 6, -0.5)))
    v = p.product((h, h, x, w, x, p.constant(1, 0)), spec, 5)
    base = tuple(p.nodes), s, v, specs, 1, 2, 1e-12
    meta = repr((repr((0, 0, 2, 1)), base))
    source = torch.tensor([0, 1, 2, 1, 0, 1, 2])
    target = torch.tensor([1, 1, 1, 2, 2, 0, 0])
    rng = torch.Generator().manual_seed(19)

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=rng)

    values = [
        rand(3, 1, 2),
        rand(7, 2),
        rand(1),
        rand(1),
        rand(7, 1).sigmoid(),
        rand(7, 1),
        rand(7, 1),
    ]
    values[4][0] = 0
    y, den, mx = tece.forward(meta, source, target, values, reference=True)
    args = [*values, y, den, mx, rand(*y.shape), rand(*den.shape)]
    active = (0, 1, 4)
    complete = tece.contraction_reference(
        first_adjoint(base, active), source, target, args
    )
    first = saved_adjoint(base, active, len(args))
    tail = saved_adjoint(base, active, len(args), score_only=True)
    for kept in (0, 2, 7):
        outputs = [torch.zeros_like(t) for t in complete]
        operands = [
            *args,
            values[1][:, :1],
            values[1][:kept],
            torch.nn.functional.silu(values[0].flatten(1))[source[:kept]],
        ]
        for b, e, plan in ((0, kept, first), (kept, 7, tail)):
            if e == b:
                continue
            slots, flags = edge_slots(plan)
            partial = tece.contraction_reference(
                plan,
                source[b:e],
                target[b:e],
                [v[b:e] if i in slots else v for i, v in enumerate(operands)],
            )
            for out, edge, grad in zip(outputs, flags, partial, strict=True):
                (out[b:e] if edge else out).add_(grad)
        for got, want in zip(outputs, complete, strict=True):
            torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)
        if torch.cuda.is_available():
            cuda_values = [
                t.cuda().requires_grad_(i in active) for i, t in enumerate(values)
            ]
            result = saved_interaction(
                capped_program(meta, kept), source.cuda(), target.cuda(), cuda_values
            )
            torch.testing.assert_close(result[0].cpu(), y)
            assert [t.shape[0] for t in result[3:]] == [len(source), kept, kept]
            gradients = torch.autograd.grad(
                result[:2],
                [cuda_values[i] for i in active],
                [t.cuda() for t in args[-2:]],
            )
            for got, want in zip(gradients, complete, strict=True):
                torch.testing.assert_close(got.cpu(), want, rtol=1e-10, atol=1e-12)


def test_frozen_rotary_expansion_pairs_only_matching_cotangents():
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.models.tace.tece_oam_rra.expansion import (
        lower_expansion,
        source,
        tile_factory,
    )
    from equivariantx.models.tace.tece_oam_rra.force import (
        indexed_expansion,
        saved_interaction,
        saved_nodes,
    )
    from equivariantx.utils.program import Program

    p = Program()
    h, c = p.input(0, "edge", 3520), p.input(1, "edge", 3040)

    def scale(x, a):
        return p.scale(p.unary("silu", x), a)

    g = scale(p.slice(h, 0, 640), 0.75)
    s = scale(p.slice(h, 640, 160), 1.25)
    w = scale(c, 1.5)
    meta = (32, (110, 110, 20, 95, 5, 45), (((12, 7, 3, 55, 2, 9), -0.25),))
    with pytest.raises(ValueError, match="path index"):
        source((*meta[:2], (((0, 0, 0, 95, 0, 0), 1.0),)), (1.0, 1.0, 1.0), False)
    y = p.product((h, h, g, w, s, p.constant(1, 0)), meta, 5)
    base = tuple(p.nodes), w, y, (), 4, 64, 1e-12
    lowered = lower_expansion(base, saved_nodes(base))
    desc = lowered[0][y][3][0]
    assert desc[2][-1] == (0.75, 1.5, 1.25)
    rows = [
        ("product", 3520, (0, 1, 6), (desc, 0, 4)),
        ("product", 3040, (0, 1, 8), (desc, 1, 4)),
        ("product", 3040, (0, 1, 6), (desc, 1, 4)),
    ]
    assert tile_factory(tuple(rows)).keywords["roles"] == {
        0: (0, 2),
        1: (1, None),
        2: (1, 0),
    }
    assert tile_factory(tuple(rows[:1])).keywords["roles"] == {0: (0, None)}
    bad = list(p.nodes)
    bad[g] = ("tanh", 640, (h,), ())
    assert lower_expansion((tuple(bad), *base[1:]), saved_nodes(base)) is None

    def indexed_force(h, c):
        y = indexed_expansion(repr(meta), (0.75, 1.5, 1.25), h, c)
        return y, torch.autograd.grad(y.sum(), (h, c))

    inputs = tuple(
        torch.empty((3, n), device="meta", requires_grad=True) for n in (3520, 3040)
    )
    indexed = make_fx(indexed_force)(*inputs)
    graph = torch.export.export(indexed, inputs, strict=False).run_decompositions()
    calls = {n.target for n in graph.graph.nodes}
    from tace.models.compile.aot import custom_ops_libs_from_model

    assert "equivariantx.models.tace.tece_oam_rra.force" in custom_ops_libs_from_model(
        indexed
    )
    assert torch.ops.eqx_tece_rra.indexed_expansion.default in calls
    assert torch.ops.eqx_tece_rra.indexed_expansion_vjp.default in calls

    def indexed_training(h, c, dy, vh, vc):
        y = indexed_expansion(repr(meta), (0.75, 1.5, 1.25), h, c)
        first = torch.autograd.grad(y, (h, c), dy, create_graph=True)
        return torch.autograd.grad(first, (h, c, dy), (vh, vc))

    second_inputs = (
        *inputs,
        inputs[0].new_empty((3, 1440)).requires_grad_(),
        torch.empty_like(inputs[0]),
        torch.empty_like(inputs[1]),
    )
    trace = make_fx(indexed_training)(*second_inputs)
    training_graph = torch.export.export(
        trace, second_inputs, strict=False
    ).run_decompositions()
    assert torch.ops.eqx_tece_rra.indexed_expansion_dual.default in {
        node.target for node in training_graph.graph.nodes
    }
    transposed = tuple(torch.empty((n, 3), device="meta").T for n in (3520, 3040, 1440))
    assert all(
        t.is_contiguous()
        for t in torch.ops.eqx_tece_rra.indexed_expansion_vjp(
            repr(meta), [0.75, 1.5, 1.25], *transposed
        )
    )

    from equivariantx.models.tace.tece_oam_rra.model import (
        RRAInteraction,
        enable_frozen_expansion,
    )

    layer = RRAInteraction.__new__(RRAInteraction)
    torch.nn.Module.__init__(layer)
    layer.rejector = SimpleNamespace(
        _eqx_update_metadata=repr(meta),
        projection_slices=((0, 640), (640, 160)),
        scalar_act=torch.nn.SiLU(),
        tensor_act=torch.nn.SiLU(),
    )
    frozen = torch.nn.Linear(1, 1).eval()
    frozen.readout_fn = SimpleNamespace(
        representation=SimpleNamespace(interactions=[None, layer])
    )
    with pytest.raises(ValueError, match="frozen evaluation"):
        enable_frozen_expansion(frozen)
    frozen.requires_grad_(False)
    layer.rejector.scalar_act = torch.nn.Tanh()
    enable_frozen_expansion(frozen)
    assert layer.frozen_expansion_scales is None
    layer.rejector.scalar_act = torch.nn.SiLU()
    enable_frozen_expansion(frozen)
    assert layer.frozen_expansion_scales == (1, 1, 1)
    layer.rejector.projection_slices = ((0, 32), (32, 160))
    enable_frozen_expansion(frozen)
    assert layer.frozen_expansion_scales is None

    score, value = p.slice(c, 0, 1), p.slice(y, 0, 2)
    specs = (
        (3520, "edge"),
        (3040, "edge"),
        (1, "shared"),
        (1, "shared"),
        (1, "edge"),
        (1, "edge"),
        (1, "edge"),
    )
    base = tuple(p.nodes), score, value, specs, 1, 2, 1e-12
    metadata = repr((repr((0, 0, 2, 1)), base))

    def force(indices, values):
        out = saved_interaction(metadata, indices, indices, values)[0]
        return torch.autograd.grad(out.sum(), (values[0], values[1]))

    indices = torch.empty(3, dtype=torch.long, device="meta")
    for dtype, expected in (
        (torch.float32, torch.ops.eqx_tece_rra.expansion_pullback.default),
        (torch.float64, torch.ops.eqx.tece_contraction.default),
    ):
        values = [
            torch.empty(
                (3 if k == "edge" else 1, w),
                device="meta",
                dtype=dtype,
                requires_grad=i in (0, 1),
            )
            for i, (w, k) in enumerate(specs)
        ]
        model = make_fx(force)(indices, values)
        graph = torch.export.export(
            model, (indices, values), strict=False
        ).run_decompositions()
        assert expected in {node.target for node in graph.graph.nodes}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA expansion")
def test_rotary_expansion_joint_input_adjoint():
    from torch.nn.functional import silu

    from equivariantx.models.tace.tece_oam_rra.expansion import expansion, tile_factory
    from equivariantx.models.tace.tece_oam_rra.force import indexed_expansion
    from equivariantx.ops.common.cuda.execution import execution_plan
    from equivariantx.utils.program import Program

    meta = (
        32,
        (110, 110, 20, 95, 5, 45),
        tuple(
            (indices, weight)
            for k in range(45)
            for indices, weight in (
                ((k, k % 11, k % 20, k, k % 5, k), -0.25),
                ((k, -1, -1, -1, -1, k), 1.0),
            )
        ),
    )
    scales = (0.75, 1.5, 1.25)
    for n in (0, 13):
        h, c = (
            torch.randn((n, width * 2), device="cuda")[:, ::2].requires_grad_()
            for width in (3520, 3040)
        )
        h3, w3 = h.reshape(n, 110, 32), c.reshape(n, 95, 32)
        k = torch.arange(45, device=h.device)
        expected = h3[:, k] - 0.25 * h3[:, k] * h3[:, k % 11] * (
            silu(h3[:, k % 20]) * scales[0]
        ) * (silu(w3[:, k]) * scales[1]) * (silu(h3[:, 20 + k % 5]) * scales[2])
        dy = torch.randn_like(expected).reshape(n, 1440)
        (actual,) = expansion(meta, scales, h, c)
        torch.testing.assert_close(
            actual, expected.reshape(n, 1440), rtol=2e-6, atol=1e-6
        )
        if n:
            from equivariantx.models.tace.tece_oam_rra.expansion import (
                reference_expansion,
            )

            hh, cc, dy0, vh, vc = [
                (0.25 * torch.randn_like(x)).requires_grad_() for x in (h, c, dy, h, c)
            ]
            with torch.no_grad():
                hh[0, ::2] = 0
                cc[0, ::2] = 0

            def derivatives(fn, third):
                pairs = (hh, cc)
                first = torch.autograd.grad(fn(*pairs), pairs, dy0, create_graph=True)
                second = torch.autograd.grad(
                    first, (*pairs, dy0), (vh, vc), create_graph=third
                )
                if not third:
                    return second
                return torch.autograd.grad(
                    second, (*pairs, dy0, vh, vc), [torch.ones_like(x) for x in second]
                )

            for third in (False, True):
                refs = derivatives(
                    lambda x, w: reference_expansion(meta, scales, x, w), third
                )
                actuals = derivatives(
                    lambda x, w: indexed_expansion(repr(meta), list(scales), x, w),
                    third,
                )
                for got, want in zip(actuals, refs, strict=True):
                    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
        gradients = torch.autograd.grad(expected, (h, c), dy.reshape(n, 45, 32))
        indexed = indexed_expansion(repr(meta), list(scales), h, c)
        torch.testing.assert_close(indexed, actual, rtol=0, atol=0)
        for result, ref in zip(
            torch.autograd.grad(indexed, (h, c), dy), gradients, strict=True
        ):
            torch.testing.assert_close(result, ref, rtol=3e-6, atol=1e-6)
        for out, ref in zip(expansion(meta, scales, h, c, dy), gradients, strict=True):
            torch.testing.assert_close(out, ref, rtol=3e-6, atol=1e-6)
        if n:
            for role in (0, 1):
                p = Program()
                args = tuple(
                    p.input(i, "edge", size)
                    for i, size in enumerate((3520, 3040, 1440))
                )
                description = (
                    32,
                    (110, 95, 45),
                    ("frozen_expansion", meta, *args[:2], scales),
                )
                root = p.product(args, description, role, 4)
                plan = execution_plan(repr((tuple(p.nodes), ((root, 0, "edge"),))))
                idx = torch.arange(n, device="cuda")
                tile = tile_factory(plan[0])(plan, [h, c, dy], idx, idx, 2, 9, {})
                torch.testing.assert_close(
                    tile.value(root), gradients[role][2:9], rtol=3e-6, atol=1e-6
                )


def test_frozen_rotary_endpoint_layout_and_radial_roles():
    from equivariantx.models.tace.tece_oam_rra.expressions import (
        packed_layout,
        paired_features,
    )
    from equivariantx.models.tace.tece_oam_rra.layout import paired_layout, paired_tile
    from equivariantx.ops.common.cuda.execution import execution_plan
    from equivariantx.utils.frame_program import rotate
    from equivariantx.utils.program import Program

    p = Program()
    wigner = p.input(5, "edge", 165)
    source, target = (p.input(0, side, 1600) for side in ("source", "target"))
    alt = paired_features(p, target, source, 25, 64)
    rotate(p, packed_layout(5, 4, 128), alt, wigner, 25, 25, packed=True)
    assert paired_layout(tuple(p.nodes)) is None
    x = rotate(
        p,
        packed_layout(5, 4, 128),
        paired_features(p, source, target, 25, 64),
        wigner,
        25,
        25,
        packed=True,
    )
    weights, grad = p.input(1, "edge", 1920), p.input(2, "edge", 3200)
    radial = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        9,
        10,
        11,
        12,
        13,
        12,
        13,
        14,
        14,
    )
    layout = (128, (25, 15, 25), tuple(((i, k, i), 1.0) for i, k in enumerate(radial)))
    products = [p.product((x, weights, grad), layout, role) for role in (2, 0, 1)]
    nodes = tuple(p.nodes)
    assert paired_layout(nodes)[:2] == (x, dict(zip(products, (2, 0, 1))))
    if not torch.cuda.is_available():
        return
    s = torch.tensor([1, 1, 0, 3, 1, 2], device="cuda")
    t = torch.tensor([0, 3, 1, 1, 2, 1], device="cuda")
    graph = execution_plan(
        repr((nodes, tuple((i, j, "edge") for j, i in enumerate((*products, x)))))
    )
    feats = torch.randn(4, 25, 64, device=s.device)
    w = torch.randn(len(s), 15, 128, device=s.device)
    d = torch.randn(len(s), 165, device=s.device)
    g = torch.randn(len(s), 25, 128, device=s.device)
    z = torch.cat((feats[s], feats[t]), dim=-1).double()
    blocks = []
    for l in range(5):
        dim = 2 * l + 1
        offset = l * (4 * l * l - 1) // 3
        blocks.append(
            d[:, offset : offset + dim * dim].double().reshape(-1, dim, dim)
            @ z[:, l * l : (l + 1) ** 2]
        )
    order = [
        blocks[l][:, l + k]
        for m in range(5)
        for k in ((m, -m) if m else (0,))
        for l in range(m, 5)
    ]
    local = torch.stack(order, 1).requires_grad_()
    radial_weights = w.double().requires_grad_()
    value = local * radial_weights[:, radial]
    dx, dw = torch.autograd.grad(value, (local, radial_weights), g.double())
    tile = paired_tile(nodes)(graph, [feats, w, g, None, None, d], s, t, 1, 5, {})
    for i, ref in zip((*products, x), (value, dx, dw, local), strict=True):
        torch.testing.assert_close(
            tile.value(i).double(), ref[1:5].flatten(1), rtol=3e-6, atol=5e-6
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_frozen_rotary_degree_consumers():
    from equivariantx.models.tace.tece_oam_rra.expressions import (
        packed_layout,
        paired_features,
    )
    from equivariantx.models.tace.tece_oam_rra.layout import paired_layout, paired_tile
    from equivariantx.ops.common.cuda.execution import TiledProgram, execution_plan
    from equivariantx.utils.frame_program import rotate
    from equivariantx.utils.program import Program

    p = Program()
    w = p.input(5, "edge", 165)
    x, y = (p.input(0, side, 1600) for side in ("source", "target"))
    rotate(
        p,
        packed_layout(5, 4, 128),
        paired_features(p, x, y, 25, 64),
        w,
        25,
        25,
        packed=True,
    )
    degrees = paired_layout(tuple(p.nodes))[2]
    combined = p.concatenate(
        tuple(p.slice(j, k + 2, 11) for k, j in enumerate(degrees))
    )
    nodes = tuple(p.nodes)
    graph = execution_plan(
        repr((nodes, tuple((i, k, "edge") for k, i in enumerate((combined, *degrees)))))
    )
    s = torch.tensor([1, 1, 0, 2, 0, 0], device="cuda")
    t = torch.tensor([0, 2, 2, 1, 1, 1], device="cuda")
    rng = torch.Generator(device="cuda").manual_seed(42)
    h = torch.randn(3, 1600, device="cuda", generator=rng)
    frame = torch.randn(6, 165, device="cuda", generator=rng)

    def tiles(a, b, begin, end):
        args = (graph, [a, None, None, None, None, b], s, t, begin, end, {})
        return paired_tile(nodes)(*args), TiledProgram(*args)

    for begin, end in ((1, 5), (3, 4)):
        tile, ref = tiles(h, frame, begin, end)
        assert tile.degrees == degrees
        out = h.new_empty(end - begin, 88)[:, ::2]
        assert tile.value(combined, out) is out
        torch.testing.assert_close(out, ref.value(combined), atol=0, rtol=0)
        assert not tile.degrees
        for i in reversed(degrees):
            out = torch.empty_like(tile.values[i])
            assert tile.value(i, out) is out
            torch.testing.assert_close(out, ref.value(i), atol=0, rtol=0)

    tile, _ = tiles(h, frame, 2, 2)
    _, outputs = tile.expression_inputs(combined)
    assert not tile.degrees and all(x.size(0) == 0 for x in outputs)

    tile, _ = tiles(h, frame, 0, 1)
    tile.value(degrees[2])
    assert not set(tile.values).intersection(set(degrees) - {degrees[2]})
    saved = tile.values[degrees[2]]
    tile.value(combined)
    assert tile.values[degrees[2]] is saved
    for a, b in ((h.double(), frame.double()), (h, frame.T.contiguous().T)):
        tile, ref = tiles(a, b, 1, 5)
        assert not tile.degrees
        torch.testing.assert_close(
            tile.value(combined), ref.value(combined), atol=0, rtol=0
        )


def test_frozen_rotary_score_matches_complete_expressions_only():
    from equivariantx.models.tace.tece_oam_rra.score import (
        score_nodes,
        score_roles,
        template,
    )

    for reverse in (False, True):
        pattern, roots = template(reverse)
        nodes = list(pattern)
        nodes[15] = ("constant", 4, (), 0.125)
        nodes = tuple(nodes)
        assert len(score_roles(score_nodes(nodes, roots, reverse))) == len(roots)
        if reverse:
            assert score_nodes(nodes, roots[:-1], reverse) is nodes
        i = next(i for i, n in enumerate(nodes) if n[0] == "sin")
        permuted = (*nodes[:i], ("cos", *nodes[i][1:]), *nodes[i + 1 :])
        assert score_nodes(permuted, roots, reverse) is permuted
        edge_bias = (*nodes[:10], ("input", 320, (), (10, "edge")), *nodes[11:])
        assert score_nodes(edge_bias, roots, reverse) is edge_bias


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_frozen_rotary_score_and_all_input_derivatives():
    from equivariantx.models.tace.tece_oam_rra.score import WIDTHS, score_outputs

    rng = torch.Generator(device="cuda").manual_seed(42)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=rng)

    n, scale = 19, 0.0625
    q, k = ([rand(n, w) for w in WIDTHS] for _ in range(2))
    qb, kb, rad, rb, tau, seed = (
        rand(320),
        rand(320),
        rand(n, 8),
        rand(8),
        rand(4),
        rand(n, 4),
    )
    rad[0, 4:] = torch.tensor([-40.0, 0.0, 1e-7, 40.0], device="cuda") - rb[4:]
    inputs = (*q, *k, qb, kb, rad, rb, tau)
    actual = score_outputs(inputs, scale, False) + score_outputs(
        (*inputs, seed), scale, True
    )
    *blocks, radial = [x.double().requires_grad_() for x in (*q, *k, rad)]
    phase = (radial + rb.double())[:, 4:].tanh() * torch.pi
    parts = []
    for m in range(5):
        qr, kr = blocks[m], blocks[m + 5]
        if not m:
            part = (qr + qb.double()) * (kr + kb.double())
        else:
            qr, qi = qr.chunk(2, dim=1)
            kr, ki = kr.chunk(2, dim=1)
            phase_channels = (m * phase).repeat_interleave(16, dim=1).repeat(1, 5 - m)
            part = (qr * kr + qi * ki) * phase_channels.cos() + (
                qi * kr - qr * ki
            ) * phase_channels.sin()
        parts.append(part.reshape(n, -1, 4, 16).sum((1, 3)))
    y = sum(parts) * tau.double() * scale + (radial + rb.double())[:, :4]
    expected = (y, *torch.autograd.grad(y, (*blocks, radial), seed.double()))
    for x, v in zip(actual, expected, strict=True):
        torch.testing.assert_close(x.double(), v, rtol=2e-5, atol=5e-6)
    empty = (*[x[:0] for x in (*q, *k)], qb, kb, rad[:0], rb, tau, seed[:0])
    assert all(x.size(0) == 0 for x in score_outputs(empty, scale, True))


def test_rotary_output_frame_and_both_adjoints():
    from equivariantx.models.tace.tece_oam_rra.expressions import packed_layout
    from equivariantx.models.tace.tece_oam_rra.frame import (
        OUTPUT_FRAME,
        lower_output,
        output_frame,
    )
    from equivariantx.utils.frame_program import rotate
    from equivariantx.utils.program import Program

    program = Program()
    widths = (320, 512, 384, 256, 128)
    local = program.concatenate(
        [program.input(j, "edge", w) for j, w in enumerate(widths)]
    )
    matrix = program.input(6, "edge", 165)
    layout = packed_layout(5, 4, 64)
    value = rotate(program, layout, local, matrix, 25, 25, inverse=True, packed=True)
    different = rotate(
        program,
        layout,
        local,
        program.input(7, "edge", 165),
        25,
        25,
        inverse=True,
        packed=True,
    )
    nodes = tuple(program.nodes)
    assert lower_output(nodes, different) is nodes
    assert lower_output(nodes, value)[value] == (
        "product",
        1600,
        (local, matrix, local),
        (OUTPUT_FRAME, 2, 0),
    )
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(7, 25, 64, device=device, generator=rng).double().requires_grad_()
    d = torch.randn(7, 165, device=device, generator=rng).double().requires_grad_()
    g = torch.randn(7, 25, 64, device=device, generator=rng).double()
    order = [
        (l, l + k)
        for m in range(5)
        for k in ((m, -m) if m else (0,))
        for l in range(m, 5)
    ]
    parts = []
    for l, block in enumerate(d.split([j * j for j in (1, 3, 5, 7, 9)], 1)):
        width = 2 * l + 1
        parts.append(
            block.reshape(-1, width, width).mT
            @ x[:, [order.index((l, j)) for j in range(width)]]
        )
    y = torch.cat(parts, 1)
    dx, dd = torch.autograd.grad(y, (x, d), g)
    for role, (a, b, ref) in enumerate(((g, d, dx), (x, g, dd), (x, d, y))):
        got = output_frame(a.flatten(1).float(), b.flatten(1).float(), role)
        torch.testing.assert_close(got.double(), ref.flatten(1), rtol=3e-6, atol=5e-6)
    dest = x.new_empty((1600, 7)).T
    with pytest.raises(ValueError, match="destination"):
        output_frame(x.flatten(1).float(), d.float(), 2, dest)


def test_rotary_saved_primals_respect_device_budget(monkeypatch):
    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from equivariantx.models.tace.tece_oam_rra.lammps import (
        StreamedRRALammpsCalc,
        ThroughputRRALammpsCalc,
    )

    monkeypatch.setattr(StreamedRRALammpsCalc, "_initialize_device", lambda *_: None)
    data = SimpleNamespace(nlocal=5000, npairs=375000, elems=torch.empty(0))
    for free, expected in ((70 << 30, "saved"), (48 << 30, "streamed")):
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda _, free=free: (free, 72 << 30)
        )
        calc = object.__new__(ThroughputRRALammpsCalc)
        calc.package_filename, calc.package_bytes = "native", b"0"
        calc.streamed, calc.saved = ("streamed", b"1"), ("saved", b"2")
        calc._initialize_device(data)
        assert calc.package_filename == expected
        assert not calc.needs_tiles(data)
    data.nlocal, data.npairs = 6655, 6655 * 74
    calc.workspace_budget = 71 << 30
    calc.graph_mode = "indexed"
    assert calc.graph_for(data) == "streamed"
    calc.primal_edge_limit = 393216
    assert calc.graph_for(data) == "saved"
    calc.workspace_budget = 65 << 30
    assert calc.graph_for(data) == "streamed"
    calc.workspace_budget = 70 << 30
    data.npairs = 540000
    assert calc.graph_for(data) == "streamed"
    data.npairs = 6655 * 74
    calc.graph_mode = "streamed"
    assert calc.graph_for(data) == "streamed"
    update = SimpleNamespace(
        _eqx_metadata="valid",
        use_asymmetric_contraction=True,
        num_head=4,
        linear_up=SimpleNamespace(irreps_out=SimpleNamespace(dim=3520)),
        linear_coefs=SimpleNamespace(irreps_out=SimpleNamespace(dim=3040)),
        scalar_act=torch.nn.SiLU(),
    )
    assert ThroughputRRALammpsCalc.supports_saved_primals((update,))
    update.scalar_act = torch.nn.Tanh()
    assert not ThroughputRRALammpsCalc.supports_saved_primals((update,))
