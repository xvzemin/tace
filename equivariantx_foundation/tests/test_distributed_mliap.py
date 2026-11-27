"""Optional real-MPI verification; source the Open MPI/Kokkos environment first."""

import os
import re
import shutil
import subprocess
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("lammps.mliap.mliap_unified_abc")
from equivariantx_foundation.targets.lammps.distributed import (
    RegionalDistributedLammpsCalc,
    export_distributed,
    lower_rank_local,
)
from equivariantx_foundation.tests.test_tace_plan import make_model
from tace.interface.lammps.mliap import TACELammpsCalc
from tace.models.cartesian.tace import CartesianTACE
from tace.models.lammps import GhostExchange
from tace.models.spherical.tace import SphericalTACE


def potential(basis):
    def build(**cfg):
        cfg.update(
            num_layers=3,
            statistics=[
                dict(
                    atomic_numbers=[26], avg_num_neighbors=12.0, atomic_energy={26: 0.0}
                )
            ],
        )
        if basis == "mixed_o2":
            cfg["num_channel"] = 4
            cfg["atomic_basis"]["type"] = ["uu_o2", "uv_o2", "uu_o2"]
            cfg["radial_basis"]["apply_cutoff"] = False
        elif basis == "rotary":
            cfg.update(parity=False, num_channel=4, Lmax=1, lmax=1, mmax=1)
            cfg["atomic_basis"].update(
                type=["cgtp", "so2", "cgtp"], num_head=2, edge_ace_hidden=3
            )
            cfg["edge_update"]["type"] = "element2"
        return (CartesianTACE if basis == "cartesian" else SphericalTACE)(**cfg)

    return make_model(build).float()


@pytest.mark.parametrize("basis", ["spherical", "cartesian"])
def test_portable_export_does_not_change_source(tmp_path, monkeypatch, basis):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    src = potential(basis).train()
    before = tuple(src.parameters())
    path = export_distributed(src, tmp_path / "regional.pt")
    calc = torch.load(path, weights_only=False, map_location="cpu")
    assert not src.lmp and src.training and any(p.requires_grad for p in before)
    assert not calc.initialized and calc.model.model.lmp
    assert [type(x) for x in src.readout_fn.representation.interactions] == [
        type(x) for x in calc.model.model.readout_fn.representation.interactions
    ]
    assert lower_rank_local(calc.model.model, backend="torch") == tuple(
        f"{kind}.{i}" for kind in ("interactions", "products") for i in range(3)
    )
    with pytest.raises(ValueError, match="unoptimized eager"):
        export_distributed(calc.model.model, tmp_path / "reexport.pt")


def test_rank_local_keeps_custom_siblings_and_only_lowers_known_nodes(monkeypatch):
    from equivariantx_foundation.models.tace import plan
    from equivariantx_foundation.tests.test_tace_plan import CustomProduct
    from tace.models.spherical.inter import O3CgtpInteraction

    monkeypatch.setenv("TACE_USE_EQX", "0")
    model = potential("spherical").eval()
    rep = model.readout_fn.representation

    class CustomInteraction(O3CgtpInteraction):
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs) * 0.5

    rep.interactions[0].__class__ = CustomInteraction
    # On the *ordinary* interaction type, the contraction being replaced
    # has a non-equivalent user-installed forward. It too must stay intact.
    tp = rep.interactions[1].rejector.tp
    native_forward = tp.forward

    def scaled(*args, **kwargs):
        return native_forward(*args, **kwargs) * 0.75

    tp.forward = scaled
    rep.products[2].__class__ = CustomProduct
    protected = ("interactions.0", "interactions.1", "products.2")
    untouched = {key: rep.get_submodule(key) for key in protected}
    rejected = rep.interactions[1].rejector
    reported = plan(model, task="inference", backend="torch", regional_only=True)
    regions = {r.path: r for r in reported.regions}
    assert all(regions[p].lower is None for p in protected)
    assert lower_rank_local(model, "torch") == (
        "interactions.2",
        "products.0",
        "products.1",
    )
    assert all(
        rep.get_submodule(key) is original for key, original in untouched.items()
    )
    assert type(rep.interactions[0]) is CustomInteraction
    assert type(rep.products[2]) is CustomProduct
    assert rep.interactions[1].rejector is rejected and rejected.tp.forward is scaled
    assert rep.interactions[2].rejector.eqx_tp.backend == "torch"


@pytest.mark.parametrize("customize", ["driver", "radial_hook"])
def test_rank_local_preserves_custom_call_boundaries(monkeypatch, customize):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    model = potential("spherical").eval()
    rep = model.readout_fn.representation
    if customize == "driver":
        original = rep.forward

        def custom_driver(*args, **kwargs):
            # Example host extension whose execution depends on the provider.
            assert not hasattr(rep.products[0], "scheduled_contraction")
            return original(*args, **kwargs)

        monkeypatch.setattr(rep, "forward", custom_driver)
        assert lower_rank_local(model, "torch") == ()
        assert rep.forward is custom_driver and not hasattr(
            rep.products[0], "scheduled_contraction"
        )
    else:
        layer = rep.interactions[1]
        with layer.edge_info.register_forward_hook(lambda m, args, x: x * 0.5):
            assert lower_rank_local(model, "torch") == (
                "interactions.0",
                "interactions.2",
                "products.0",
                "products.1",
                "products.2",
            )
            assert rep.interactions[1] is layer


@pytest.mark.parametrize("basis", ["uv_magnetic", "wigner6j"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rank_local_keeps_magnetic_interactions_even_with_eager_provider(
    basis, device, monkeypatch
):
    from equivariantx_foundation.models.tace import plan
    from equivariantx_foundation.tests.test_interaction_schedule import _wigner6j_model
    from equivariantx_foundation.tests.test_tace_plan import physics_model
    from tace.models.spherical.mtace import (
        O2MagneticInteraction,
        O3MagneticWigner6jInteraction,
    )

    if device == "cuda" and (not torch.cuda.is_available() or torch.version.hip):
        pytest.skip("eager magnetic regional kernels require NVIDIA CUDA")
    monkeypatch.setenv("TACE_USE_EQX", "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    if basis == "wigner6j":
        model, expected = _wigner6j_model(), O3MagneticWigner6jInteraction
    else:
        model, expected = (
            physics_model("noncollinear_magnetic_forces"),
            O2MagneticInteraction,
        )
    model = model.float().to(device).eval()
    rep = model.readout_fn.representation
    backend = "torch" if device == "cpu" else "cuda"
    report = plan(model, task="inference", backend=backend, regional_only=True)
    interactions = [
        r
        for r in report.regions
        if r.path.startswith("interactions.") and r.path.count(".") == 1
    ]
    assert len(interactions) == len(rep.interactions) and all(
        type(i) is expected for i in rep.interactions
    )
    if device == "cuda":
        # This contrast is important: these are real full-graph providers,
        # not just classes incapable of lowering on either target.
        assert any(r.lower is not None for r in interactions)
    before = [
        (layer, type(layer), layer.rejector, type(layer.rejector))
        for layer in rep.interactions
    ]
    selected = lower_rank_local(model, backend)
    assert not any(p.startswith("interactions.") for p in selected)
    assert all(
        now is old
        and type(now) is cls
        and now.rejector is conv
        and type(now.rejector) is conv_cls
        for now, (old, cls, conv, conv_cls) in zip(
            rep.interactions, before, strict=True
        )
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip,
    reason="CUDA rank-local Wigner frames",
)
def test_rank_local_preserves_host_frame_driver_on_mixed_o2(monkeypatch):
    from equivariantx_foundation.models.tace import plan
    from tace.models.spherical.inter import UuO2Interaction, UvO2Interaction

    monkeypatch.setenv("TACE_USE_EQX", "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    model = potential("mixed_o2").cuda().eval()
    rep = model.readout_fn.representation
    assert tuple(type(i) for i in rep.interactions) == (
        UuO2Interaction,
        UvO2Interaction,
        UuO2Interaction,
    )
    graph_regions = plan(
        model, task="inference", backend="cuda", regional_only=True
    ).regions
    assert any(
        r.path.startswith("interactions.") and r.lower is not None
        for r in graph_regions
    )
    wigner, old_backend = rep.o2_angular_basis, rep.o2_angular_basis.backend
    classes = [(type(i), type(i.rejector)) for i in rep.interactions]
    assert lower_rank_local(model, "cuda") == (
        "interactions.1",
        "interactions.2",
        "products.0",
        "products.1",
        "products.2",
    )
    assert rep.o2_angular_basis is wigner and wigner.backend == old_backend
    assert not getattr(rep, "_foundation_wigner_backend", None)
    # Only the two supported local maps change. Do not replace interaction
    # drivers with whole-graph variants that consume canonical direction data.
    for i, (layer, (host, conv)) in enumerate(
        zip(rep.interactions, classes, strict=True)
    ):
        assert type(layer) is host
        assert (type(layer.rejector) is conv) == (i == 0)


@pytest.mark.parametrize(
    "backend,memory,hook,wide",
    [
        ("torch", True, False, False),
        ("cuda", False, False, False),
        ("cuda", True, False, False),
        ("cuda", True, True, False),
        ("cuda", False, False, True),
    ],
)
def test_rank_local_radial_policy(
    backend, memory, hook, wide, double_precision, monkeypatch
):
    from equivariantx_foundation.models.tace.interactions import (
        _HostFrameRadialInteraction,
    )
    from tace.interface.lammps.mliap import EdgeForcesWrapper
    from tace.utils.env import ACCELERATION_ENV

    if backend == "cuda" and (not torch.cuda.is_available() or torch.version.hip):
        pytest.skip("NVIDIA CUDA rank-local schedule")
    for key in ACCELERATION_ENV.values():
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", str(int(memory)))
    device = "cuda" if backend == "cuda" else "cpu"

    def build(**cfg):
        cfg["edge_update"]["type"] = "element2"
        cfg["target_property"] = ["energy", "forces"]
        return SphericalTACE(**cfg)

    native = make_model(build).to(device).eval()
    if wide:
        native.float()
        # Exercise the wide projection on a small CI graph.
        monkeypatch.setattr(_HostFrameRadialInteraction, "LAST_PROJECTION_BYTES", 1)
        monkeypatch.setattr(_HostFrameRadialInteraction, "LAST_PROJECTION_EXPANSION", 0)
    regional = deepcopy(native)
    assert "interactions.1" in lower_rank_local(regional, backend=backend)
    owner = torch.tensor([0, 1, 2, 0, 2, 1], device=device)
    index = torch.tensor([[4, 5, 3, 1, 4, 2, 3], [1, 0, 2, 2, 1, 0, 0]], device=device)

    class Exchange:
        def forward_exchange(self, x, out, width):
            out.copy_(x[owner])

        def reverse_exchange(self, grad, out, width):
            out.zero_().index_add_(0, owner, grad)

    data = dict(
        edge_vector=torch.randn(index.size(1), 3, device=device),
        edge_index=index,
        node_type=(owner % 2),
        batch=owner[:3] * 0,
        ptr=owner.new_tensor([0, 3]),
        natoms=(3, 3),
        lmp_data=Exchange(),
    )
    streamed = backend == "cuda" and memory and not hook
    calls, hooked = [], []
    with ExitStack() as stack:
        if hook:
            # Install after lowering. Bypassing a customized edge MLP would
            # change the potential as well as the call graph.
            def adjust(layer, inputs, weights):
                hooked.append(layer)
                return weights + 0.25

            for model in (native, regional):
                for layer in model.readout_fn.representation.interactions:
                    stack.enter_context(layer.edge_info.register_forward_hook(adjust))
        for layer in regional.readout_fn.representation.interactions:

            def observe(conv, args, kwargs, output, *, layer=layer):
                x, attrs, radial, projection, edges = args[:5]
                assert x.size(0) == len(owner) and output.size(0) == 3
                assert attrs is not None and kwargs.get("vectors") is None
                net = kwargs.get("radial_network")
                calls.append((net is not None, projection.numel() > 0))
                if net is not None:
                    assert list(net) == list(layer.edge_info.mlp)[:-1]
                    assert [(v.size(0), kind) for v, kind in radial] == [
                        (edges.size(1), "edge"),
                        (len(owner), "target"),
                        (len(owner), "source"),
                    ]
                    assert projection.numel()
                elif wide:
                    assert radial.shape == (edges.size(1), projection.size(0))
                else:
                    assert radial.shape == (edges.size(1), conv.weight_numel)
                    assert projection.numel() == 0

            stack.enter_context(
                layer.rejector.eqx_tp.register_forward_hook(observe, with_kwargs=True)
            )
        reference = EdgeForcesWrapper(native)(data)
        actual = EdgeForcesWrapper(regional)(data)
    assert calls == [(streamed, streamed or wide)] * 2
    assert len(hooked) == (4 if hook else 0)
    torch.testing.assert_close(
        actual, reference, atol=2e-6 if wide else 2e-10, rtol=1e-5 if wide else 2e-9
    )
    assert actual[2][index[0] >= 3].abs().max() > 1e-7


class _AuditLoader(RegionalDistributedLammpsCalc):
    """Test-only probe of real rank-local CUDA calls and both MPI exchanges."""

    def compute_forces(self, data):
        if not self.initialized:
            self._initialize_device(data)
        counts = Counter()

        def count(name, operation):
            def measured(*args, **kwargs):
                counts[name] += 1
                return operation(*args, **kwargs)

            return measured

        def count_convolution(_, inputs, output):
            assert inputs[0].size(0) == data.ntotal
            assert output.size(0) == data.nlocal
            counts["conv"] += 1

        with ExitStack() as stack:
            for name in ("forward", "backward"):
                op = getattr(GhostExchange, name)
                stack.enter_context(
                    patch.object(GhostExchange, name, staticmethod(count(name, op)))
                )
            rep = self.model.model.readout_fn.representation
            from equivariantx_foundation.models.tace.local import _UuFrames, _UvFrames
            from equivariantx_foundation.models.tace.tece_oam_rra.regional import (
                StreamedRotaryInteraction,
            )
            from tace.models.spherical.inter import UuO2Interaction
            from tace.models.spherical.tece_oam_rra import UvSO2Interaction

            mixed = type(rep.interactions[0]) is UuO2Interaction
            if mixed:
                assert type(rep.interactions[1].rejector) is _UvFrames
                assert type(rep.interactions[2].rejector) is _UuFrames
            rotary = isinstance(rep.interactions[1], UvSO2Interaction)
            if rotary:
                assert type(rep.interactions[1]) is StreamedRotaryInteraction
                stream = import_module(
                    "equivariantx_foundation.models.tace.tece_oam_rra.interaction"
                )
                run = stream.interaction

                def rotary_kernel(metadata, source, target, inputs):
                    outputs = run(metadata, source, target, inputs)
                    assert inputs[0].size(0) == data.ntotal
                    assert all(out.size(0) == data.nlocal for out in outputs)
                    counts["conv"] += 1
                    return outputs

                stack.enter_context(patch.object(stream, "interaction", rotary_kernel))
            for i, layer in enumerate(rep.interactions):
                conv = layer.rejector
                if rotary and i == 1:
                    continue
                if mixed and i == 2:
                    frame = conv.local_frame_out
                    stack.enter_context(
                        patch.object(
                            frame, "scatter", count("uu_scatter", frame.scatter)
                        )
                    )
                if mixed and i != 1:
                    op = conv
                else:
                    op = getattr(conv, "scheduled_convolution", conv.eqx_tp)
                stack.enter_context(op.register_forward_hook(count_convolution))
            for product in rep.products:
                stack.enter_context(
                    product.scheduled_contraction.register_forward_hook(
                        lambda *_: counts.update(["product"])
                    )
                )
            super().compute_forces(data)
        assert data.ntotal > data.nlocal > 0
        expected = {"conv": 3, "product": 3, "forward": 2, "backward": 2}
        if mixed:
            expected["uu_scatter"] = 1
        assert counts == expected, counts
        assert self.device == next(self.model.parameters()).device
        print(
            f"TACE_MPI_AUDIT device={self.device.index} local={data.nlocal} ghost={data.ntotal - data.nlocal} {dict(counts)}",
            flush=True,
        )


@pytest.mark.skipif(
    os.environ.get("TACE_RUN_MPI_MLIAP_TEST") != "1",
    reason="requires a CUDA-aware Open MPI + GPU Kokkos LAMMPS installation",
)
@pytest.mark.parametrize("basis", ["spherical", "cartesian", "mixed_o2", "rotary"])
def test_real_two_rank_owner_energy_and_ghost_forces(tmp_path, monkeypatch, basis):
    """Compare owner energies and all site forces across rank decompositions."""
    import numpy as np

    for name in ("TACE_USE_EQX", "TACE_USE_OEQ", "TACE_USE_CUE", "TACE_USE_LUNA"):
        monkeypatch.setenv(name, "0")
    torch.set_default_dtype(torch.float32)
    model = potential(basis)
    export_distributed(model, tmp_path / "regional.pt")
    calc = torch.load(tmp_path / "regional.pt", weights_only=False)
    mod = import_module("equivariantx_foundation.tests.test_distributed_mliap")
    calc.__class__ = mod._AuditLoader
    torch.save(calc, tmp_path / "regional.pt")
    torch.save(TACELammpsCalc(model), tmp_path / "native.pt")
    inp = Path(__file__).with_name("in.distributed_mliap")
    mpi, lmp = shutil.which("mpirun"), shutil.which("lmp")
    assert mpi and lmp, "source the CUDA-aware Open MPI/LAMMPS installation environment"
    # The serialized audit loader lives in this checkout, not in the wheel.
    monkeypatch.setenv(
        "PYTHONPATH", str(Path(__file__).resolve().parents[2]), prepend=os.pathsep
    )
    expected = None
    for name, n in (("native", 1), ("native", 2), ("regional", 2)):
        tag = f"{name}{n}"
        command = [
            mpi,
            "-np",
            str(n),
            "--bind-to",
            "none",
            lmp,
            "-k",
            "on",
            "g",
            str(min(n, torch.cuda.device_count())),
            "-sf",
            "kk",
            "-pk",
            "kokkos",
            "newton",
            "on",
            "neigh",
            "half",
            "gpu/aware",
            "on",
            "-var",
            "nranks",
            str(n),
            "-var",
            "model",
            f"{name}.pt",
            "-var",
            "forces",
            f"{tag}.dump",
            "-log",
            f"{tag}.log",
            "-in",
            str(inp),
        ]
        p = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True, timeout=180
        )
        assert p.returncode == 0, p.stdout + p.stderr
        if name == "regional":
            devices = re.findall(r"TACE_MPI_AUDIT device=(\d+)", p.stdout)
            assert len(devices) == n and set(map(int, devices)) == set(
                range(min(n, torch.cuda.device_count()))
            ), p.stdout
        log = (tmp_path / f"{tag}.log").read_text()
        energy = float(re.search(r"^\s*0\s+72\s+([-+.eE\d]+)", log, re.M)[1])
        dump = np.loadtxt(tmp_path / f"{tag}.dump", skiprows=9)
        # Atom IDs may change across decompositions; compare physical sites.
        xyz = dump[:, 2:5].round(8)
        order = np.lexsort(xyz.T[::-1])
        result = energy, xyz[order], dump[order, 5:]
        if expected is None:
            expected = result
        np.testing.assert_allclose(result[0], expected[0], rtol=0, atol=1e-5)
        np.testing.assert_array_equal(result[1], expected[1])
        np.testing.assert_allclose(result[2], expected[2], rtol=0, atol=2e-5)
