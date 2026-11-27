"""Local-owner ML-IAP protocol; the shared runner does not depend on a release."""

import sys
from types import SimpleNamespace

import pytest
import torch

from equivariantx_foundation.targets.lammps.local_owner import LocalOwnerForceRunner


def test_local_runner_checks_rank_before_opening_a_package(monkeypatch):
    class Host:
        def _initialize_device(self, data):
            self.device = data.elems.device
            self.model = SimpleNamespace(fidelity_idx=4)

    class Runner(LocalOwnerForceRunner, Host):
        graph_max_atoms = 0

    mpi = SimpleNamespace(COMM_WORLD=SimpleNamespace(Get_size=lambda: 2))
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=mpi))
    runner = Runner()
    data = SimpleNamespace(elems=torch.tensor([0]))
    with pytest.raises(RuntimeError, match="one MPI rank"):
        runner._initialize_device(data)
    assert not hasattr(runner, "model")
    mpi.COMM_WORLD.Get_size = lambda: 1
    for atoms in (0, 3):
        runner.graph_max_atoms = atoms
        runner._initialize_device(data)
        assert runner.single_threaded == (atoms > 0) and runner.force_graph is None
        assert runner.fidelity.tolist() == [4]


def test_release_adapters_keep_their_own_shape_policy():
    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from equivariantx_foundation.models.tace.tace_oam_7m.lammps import (
        SevenLammpsCalc,
        SevenThroughputLammpsCalc,
    )
    from equivariantx_foundation.models.tace.tace_oam_l.lammps import OAMLammpsCalc
    from equivariantx_foundation.models.tace.tece_oam_rra.lammps import (
        ThroughputRRALammpsCalc,
    )

    assert not issubclass(SevenLammpsCalc, OAMLammpsCalc)
    assert not hasattr(SevenThroughputLammpsCalc, "compact_package")
    for cls in (SevenLammpsCalc, SevenThroughputLammpsCalc, OAMLammpsCalc):
        assert issubclass(cls, LocalOwnerForceRunner)
        calc = object.__new__(cls)
        for n in (0, 1):
            with pytest.raises(ValueError, match="isolated atom"):
                calc.compute_forces(SimpleNamespace(nlocal=n, npairs=0))
    assert issubclass(ThroughputRRALammpsCalc, LocalOwnerForceRunner)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Kokkos/CUDA replay")
@pytest.mark.parametrize("adapter", ["standalone", "7m", "rra"])
def test_owner_exchange_pair_order_direct_replay_and_writeback(adapter):
    class Program:
        fidelity_idx = 3
        input_keys = ("edge_vector", "edge_index", "node_type", "fidelity_idx")

        def compiled_model(self, vector, index, types, fidelity):
            src, dst = index
            k = 1 + types[src] + 0.5 * types[dst]
            node = (
                (types + fidelity)
                .double()
                .index_add(0, dst, k * vector.square().sum(-1))
            )
            return node.sum(), node, (2 * k[:, None] * vector).float()

        def __call__(self, batch):
            return self.compiled_model(*(batch[k] for k in self.input_keys))

    if adapter == "standalone":
        runner = LocalOwnerForceRunner()
    else:
        pytest.importorskip("lammps.mliap.mliap_unified_abc")
        if adapter == "7m":
            from equivariantx_foundation.models.tace.tace_oam_7m.lammps import (
                SevenThroughputLammpsCalc,
            )

            runner = object.__new__(SevenThroughputLammpsCalc)
            runner.projection_budget, runner.projection_bytes_per_edge = 1 << 30, 1024
            runner.retaining = None
        else:
            from equivariantx_foundation.models.tace.tece_oam_rra.lammps import (
                ThroughputRRALammpsCalc,
            )

            runner = object.__new__(ThroughputRRALammpsCalc)
            runner.graph_mode, runner.receiver, runner.workspace_budget = (
                "streamed",
                None,
                1 << 40,
            )
    runner.model, runner.device, runner.initialized = (
        Program(),
        torch.device("cuda"),
        True,
    )
    runner.initialize_local_inputs(runner.device, 3)
    runner.force_graph = None
    runner.graph_max_atoms, runner.graph_max_edges = 3, 2

    def ids(a):
        return torch.tensor(a, dtype=torch.long, device="cuda")

    def call(sources, eflag):
        n, e = 3, len(sources)
        owners, types = ids([0, 1, 2, 1, 0]), ids([2, 5, 7])
        src, dst = ids(sources), torch.arange(e, device="cuda") % n
        vec = torch.arange(e * 3, dtype=torch.float64, device="cuda").reshape(e, 3) / 4
        updates = []
        data = SimpleNamespace(
            nlocal=n,
            npairs=e,
            ntotal=len(owners),
            elems=types[owners],
            eflag=eflag,
            eatoms=torch.full((n,), -1.0, device="cuda", dtype=torch.float64),
            energy=-1.0,
            update_pair_forces=updates.append,
        )
        if e:  # Real ML-IAP exposes null edge arrays when there are no pairs.
            data.rij, data.pair_i, data.pair_j = vec, dst, src

            def exchange(indices, result, count):
                assert count == 1
                result.copy_(indices[owners])

            data.forward_exchange = exchange
        runner.compute_forces(data)
        energy, node, forces = runner.model.compiled_model(
            vec, torch.stack((owners[src], dst)), types, ids([3])
        )
        torch.testing.assert_close(
            data.eatoms, node if eflag & 2 else torch.full_like(node, -1)
        )
        assert data.energy == (energy.item() if eflag & 1 else -1.0)
        if e:
            assert len(updates) == 1 and updates[0] is runner.exported_forces
            torch.testing.assert_close(updates[0], forces.double())
        else:
            assert not updates and runner.exported_forces is None

    call([4, 3, 0], 3)  # Above the edge limit: run directly, map periodic images.
    assert runner.force_graph is None
    call([3, 4], 1)
    captured = runner.force_graph
    assert captured is not None
    call([4], 2)
    call([], 0)
    assert runner.force_graph is captured
    call([3, 4, 1], 3)
    assert runner.force_graph is None
