"""End-to-end E/F/stress contracts for snapshots with no released schedule."""

from copy import deepcopy

import pytest
import torch

from tace.models.adapter import TensorModel
from tace.models.spherical.default import DEFAULT_MODEL_CONFIG
from tace.models.spherical.prod import CgtpACE
from tace.models.spherical.tace import SphericalTACE
from tace.utils.env import ACCELERATION_ENV


def lower_model(model, **options):
    """Exercise the planner's portable lowering independently of CUDA deployment."""
    from equivariantx_foundation.models.tace.planner import _optimize_with_report

    return _optimize_with_report(model, **options)[1]


def make_model(model_type=SphericalTACE):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=4.0,
        num_layers=2,
        num_channel=5,
        Lmax=2,
        lmax=2,
        mmax=2,
        parity=True,
        statistics=[
            dict(
                atomic_numbers=[1, 8],
                avg_num_neighbors=3.0,
                atomic_energy={1: 0.0, 8: 0.0},
            )
        ],
        target_property=["energy", "forces", "stress"],
    )
    config["radial_basis"].update(hidden=[8, 9], num_radial_basis=4)
    config["readout_emlp"].update(hidden=[4], use_one_body_magmoms=False)
    config["scale_shift"]["enable"] = False
    return TensorModel(model_type(**config))


def test_whole_model_schedule_is_selected_only_for_declared_tasks(monkeypatch):
    from equivariantx_foundation.models.tace import planner, tace_oam_l

    # A new recipe can register its architecture and builder without changing
    # any planner branch. It does not inherit training or export support.
    model = make_model().eval()
    calls = []
    monkeypatch.setattr(tace_oam_l, "TASKS", {"inference"})
    monkeypatch.setattr(tace_oam_l, "matches", lambda source, *, task: source is model)

    def build(source, *, tile):
        calls.append(tile)
        return source

    monkeypatch.setattr(tace_oam_l, "optimize", build)
    whole = planner.plan(model)
    assert whole.schedule == "tace_oam_l"
    assert whole.regions and all(r.scope == "readout_fn" for r in whole.regions)
    assert any(r.path.startswith("energy_readouts.") for r in whole.regions)
    regional = planner.plan(model, task="inference", regional_only=True)
    assert regional.schedule == "regional" and regional.regions
    for task in ("joint_training", "lammps"):
        with pytest.raises(ValueError, match="whole-model schedule"):
            planner.plan(model, task=task, regional_only=True)
    assert lower_model(model, tile=64) is model
    assert calls == [64]
    assert planner.plan(model, task="training").schedule == "regional"
    assert planner.release_schedule(model, task="lammps")[0] is None
    assert planner.release_schedule(model, task="training_buckets")[0] is None


def graph(device):
    # Two independent structures, with a periodic image on one reciprocal bond.
    pos = torch.tensor(
        [
            [0.2, 0.1, 0.3],
            [3.6, 0.5, 0.8],
            [1.2, 1.4, 0.5],
            [0.3, 0.8, 0.4],
            [1.9, 1.1, 0.3],
            [0.8, 0.5, 1.7],
        ],
        dtype=torch.float64,
        device=device,
    )
    edges = torch.tensor(
        [[0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5], [1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4]],
        device=device,
    )
    shifts = torch.zeros(edges.size(1), 3, dtype=torch.long, device=device)
    shifts[0, 0], shifts[1, 0] = -1, 1
    return dict(
        positions=pos,
        edge_index=edges,
        edge_shifts=shifts,
        node_type=torch.tensor([0, 1, 1, 1, 0, 1], device=device),
        lattice=torch.diag(pos.new_tensor([4.1, 4.4, 4.6])).expand(2, -1, -1).clone(),
        pbc=torch.ones((2, 3), dtype=torch.bool, device=device),
        batch=torch.tensor([0, 0, 0, 1, 1, 1], device=device),
        ptr=torch.tensor([0, 3, 6], device=device),
        fidelity_idx=torch.zeros(2, dtype=torch.long, device=device),
    )


def test_planner_preserves_generated_code_with_user_hooks():
    from equivariantx_foundation.models.tace import plan
    from equivariantx_foundation.models.tace.foundation import frozen_export_copy

    model = make_model().eval()
    batch = graph("cpu")
    tp = model.readout_fn.representation.interactions[0].rejector.tp
    program = tp._compiled_main_left_right
    calls = []

    def scale(module, arguments, output):
        calls.append(True)
        return 0.5 * output

    handle = program.register_forward_hook(scale)
    try:
        report = plan(model, backend="torch")
        assert "e3nn generated module" in report.reason
        assert lower_model(model, backend="torch") is model
        assert torch.isfinite(model(batch)["forces"]).all() and calls
        with pytest.raises(ValueError, match="Cannot copy an e3nn generated module"):
            frozen_export_copy(model)
    finally:
        handle.remove()
    assert plan(model, backend="torch").regions


def test_training_install_matches_the_live_source_once_per_attempt(monkeypatch):
    from equivariantx_foundation.models.tace import planner
    from tace.lightning.foundation import FoundationTrainingCallback

    model = make_model().train()
    wrapper = torch.nn.Module()
    wrapper.model = model
    callback = FoundationTrainingCallback(backend="torch")
    real_plan = planner.plan
    sources = []

    def counted(source, **options):
        sources.append(source)
        return real_plan(source, **options)

    monkeypatch.setattr(planner, "plan", counted)
    assert callback._install(wrapper, model)
    installed = wrapper.model
    assert installed is not model and sources == [model]
    assert next(installed.parameters()) is next(model.parameters())

    # Rebuilding after native checkpoint load must see current model state;
    # it cannot reuse a stale report or a previously lowered model.
    wrapper.model = model
    program = model.readout_fn.representation.interactions[0].rejector.tp
    with program._compiled_main_left_right.register_forward_hook(lambda *args: None):
        with pytest.raises(ValueError, match="No Foundation training plan applies"):
            callback._install(wrapper, model)
        assert wrapper.model is model and sources == [model, model]
    assert callback._install(wrapper, model)
    assert wrapper.model is not installed and sources == [model] * 3


EFS = ("energy", "forces", "stress")


def efs_loss(output):
    # A nonzero target on each component keeps the mixed force/strain derivatives
    # useful even with small random energies.
    return sum(
        (
            scale * output[key]
            - torch.linspace(
                0.1,
                0.8,
                output[key].numel(),
                device=output[key].device,
                dtype=output[key].dtype,
            ).reshape_as(output[key])
        )
        .square()
        .mean()
        for key, scale in zip(EFS, (0.2, 1.5, 25.0), strict=True)
    )


@pytest.fixture(autouse=True)
def native_source(monkeypatch):
    for variable in ACCELERATION_ENV.values():
        monkeypatch.setenv(variable, "0")


DEVICES = [
    ("cpu", "torch"),
    pytest.param(
        "cuda",
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
]


@pytest.mark.parametrize("device,backend", DEVICES)
def test_retained_host_cgtp_dispatch(device, backend, monkeypatch):
    from equivariantx_foundation.models.tace import plan

    # EQX is available when the model is constructed; that alone doesn't
    # mean the host will call it. The convolution switch is read at runtime.
    source = make_model().to(device).train()
    conv = source.readout_fn.representation.interactions[0].rejector

    def region():
        report = plan(source, backend=backend, regional_only=True)
        return next(r for r in report.regions if r.path == "interactions.0")

    assert region().provenance == "regional"
    monkeypatch.setenv("TACE_USE_EQX", "1")
    selected = region()
    assert (selected.provenance, selected.provider, selected.lower) == (
        "host_selected",
        f"eqx_{backend}",
        None,
    )
    calls = []
    with conv.eqx_tp.register_forward_hook(lambda *_: calls.append(True)):
        out = source(graph(device))
        assert calls and out["forces"].isfinite().all()
        assert region().provider is None
    assert region().provider == selected.provider


@pytest.mark.skipif(not torch.cuda.is_available(), reason="external CUDA convolutions")
@pytest.mark.parametrize(
    "switch,package,provider",
    [
        ("TACE_USE_CUE", "cuequivariance_torch", "cueq"),
        ("TACE_USE_OEQ", "openequivariance", "oeq"),
    ],
)
def test_host_external_cgtp_provider(monkeypatch, switch, package, provider):
    from equivariantx_foundation.models.tace import plan

    pytest.importorskip(package)
    monkeypatch.setenv(switch, "1")
    source = make_model().cuda().eval()
    # This choice was made on construction and is retained at execution.
    monkeypatch.setenv(switch, "0")
    chosen = plan(source, backend="cuda", regional_only=True)
    regions = [
        r for r in chosen.regions if r.path in ("interactions.0", "interactions.1")
    ]
    assert len(regions) == 2
    assert all(
        (r.provenance, r.provider, r.lower) == ("host_selected", provider, None)
        for r in regions
    )
    called = []
    op = source.readout_fn.representation.interactions[0].rejector.fused_tp
    with op.register_forward_hook(lambda *_: called.append(True)):
        assert torch.isfinite(source(graph("cuda"))["forces"]).all()
    assert called


@pytest.mark.parametrize("device,backend", DEVICES)
@pytest.mark.parametrize("task", ["inference", "training"])
def test_unnamed_model_plan_and_live_optimizer(device, backend, task):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace.planner import plan

    torch.set_default_dtype(torch.float64)
    training = task == "training"
    reference = make_model().to(device).train(training)
    source = make_model().to(device).train(training)
    source.load_state_dict(reference.state_dict())
    original_parameters = dict(source.named_parameters())
    # Real optimizers are often created before adapting the model.
    optimizer = torch.optim.SGD(source.parameters(), lr=0.01)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    report = plan(source, task=task, backend=backend)
    assert report.schedule == "regional"
    choices = {r.path: r.implementation for r in report.regions}
    for i in (0, 1):
        for family in ("interactions", "products"):
            assert choices[f"{family}.{i}"] != "native", choices

    if device == "cpu":
        with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA device"):
            optimize(source, family="tace", task=task, backend=backend)
        scheduled = lower_model(source, task=task, backend=backend)
    else:
        scheduled = optimize(source, family="tace", task=task, backend=backend)
    got_params = dict(scheduled.named_parameters())
    assert got_params.keys() == original_parameters.keys()
    assert all(p is got_params[k] for k, p in original_parameters.items())
    # Published state remains strictly loadable in both directions.
    source.load_state_dict(scheduled.state_dict(), strict=True)
    scheduled.load_state_dict(source.state_dict(), strict=True)
    rep = scheduled.readout_fn.representation
    kernels = [inter.rejector.eqx_tp for inter in rep.interactions]
    kernels.extend(product.scheduled_contraction for product in rep.products)
    counters = []
    for kernel in kernels:
        calls = []
        kernel.register_forward_hook(
            lambda _m, _a, _out, calls=calls: calls.append(True)
        )
        counters.append(calls)

    data = graph(device)
    ref_out = reference(data)
    out = scheduled(data)
    assert all(counters)
    for key in EFS:
        assert ref_out[key].abs().max() > 0
        torch.testing.assert_close(out[key], ref_out[key], rtol=1e-7, atol=1e-9)
    if not training:
        return

    efs_loss(ref_out).backward()
    efs_loss(out).backward()
    reference_params = dict(reference.named_parameters())
    active = set()
    for name, p in got_params.items():
        expected = reference_params[name].grad
        assert (p.grad is None) == (expected is None), name
        if expected is not None:
            torch.testing.assert_close(p.grad, expected, rtol=2e-7, atol=1e-9, msg=name)
            if expected.abs().max() > 1e-10:
                active.add(name)
    assert any(".interactions." in name for name in active)
    assert any(".products." in name for name in active)
    optimizer.step()
    ref_optimizer.step()
    after = scheduled(data)
    ref_after = reference(data)
    assert not torch.equal(after["energy"], out["energy"])
    for key in EFS:
        torch.testing.assert_close(after[key], ref_after[key], rtol=2e-7, atol=1e-9)
    for name, p in original_parameters.items():
        torch.testing.assert_close(p, reference_params[name], rtol=2e-7, atol=1e-9)


class CustomProduct(CgtpACE):
    """A future user subclass whose forward is not algebraically equivalent."""

    def forward(self, *args, **kwargs):
        return 0.75 * super().forward(*args, **kwargs)


def test_cartesian_storage_does_not_select_spherical_kernels(monkeypatch):
    from equivariantx.co3.conv import CartesianTensorProductConv
    from equivariantx_foundation.models.tace import plan, planner, tace_oam_l
    from tace.models.cartesian.tace import CartesianTACE

    model = make_model(CartesianTACE).eval()
    report = plan(model)
    assert report.schedule == "regional"
    optimized = lower_model(model, backend="torch")
    for interaction in optimized.readout_fn.representation.interactions:
        assert isinstance(interaction.rejector.eqx_tp, CartesianTensorProductConv)
    data = graph("cpu")
    result = optimized(data)
    for key, value in model(data).items():
        if key in EFS:
            torch.testing.assert_close(result[key], value, rtol=2e-4, atol=2e-5)
    # A future recipe may explicitly match this host. The registry itself
    # must not exclude Cartesian storage before consulting the recipe.
    monkeypatch.setattr(
        tace_oam_l, "matches", lambda m, *, task: type(m.readout_fn) is CartesianTACE
    )
    assert planner.release_schedule(model, task="inference")[0] == "tace_oam_l"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_streamed_geometry_isolated_atoms_and_custom_harmonics():
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace.interactions import StreamedInteraction

    torch.set_default_dtype(torch.float64)
    source = make_model().to("cuda").eval()
    for customized in (False, True):
        if customized:
            basis = source.readout_fn.representation.o3_angular_basis
            basis.register_forward_hook(lambda module, args, result: result * 0.5)
        scheduled = optimize(source, family="tace", backend="cuda")
        streamed = isinstance(
            scheduled.readout_fn.representation.interactions[0], StreamedInteraction
        )
        assert streamed == (not customized)
        data = graph("cuda")
        for empty in (False, True):
            if empty:
                data["edge_index"] = data["edge_index"][:, :0]
                data["edge_shifts"] = data["edge_shifts"][:0]
            expected, actual = source(data), scheduled(data)
            for key in EFS:
                torch.testing.assert_close(
                    actual[key], expected[key], rtol=1e-7, atol=1e-9
                )


def test_non_equivalent_subclass_stays_native_but_other_layers_are_planned():
    from equivariantx_foundation.models.tace.planner import plan

    torch.set_default_dtype(torch.float64)
    source = make_model().eval()
    source.readout_fn.representation.products[0].__class__ = CustomProduct
    report = plan(source, task="inference", backend="torch")
    choices = {r.path: r for r in report.regions}
    assert choices["products.0"].implementation == "native"
    assert choices["products.0"].provenance == "host_fallback"
    assert choices["products.0"].provider is None
    assert choices["products.0"].reason
    assert choices["products.1"].implementation != "native"
    assert choices["products.1"].provenance == "regional"
    assert choices["products.1"].provider == "eqx_torch"
    optimized = lower_model(source, task="inference", backend="torch")
    repeated = {r.path: r for r in plan(optimized, backend="torch").regions}
    assert repeated["products.1"].provenance == "already_regional"
    assert repeated["products.1"].provider == "eqx_torch"
    data = graph("cpu")
    ref = source(data)
    out = optimized(data)
    for key in EFS:
        torch.testing.assert_close(out[key], ref[key], rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("customize", ["helper", "convolution", "hook"])
def test_opaque_message_boundary_preserves_callbacks_and_sibling_regions(customize):
    from types import MethodType

    from equivariantx_foundation.models.tace import plan
    from equivariantx_foundation.models.tace.matching import require_native_messages

    torch.set_default_dtype(torch.float64)
    model = make_model().eval()
    rep = model.readout_fn.representation
    require_native_messages(rep)
    layer = rep.interactions[1]
    calls = []
    if customize == "helper":
        original = type(layer)._compute_messages

        def scaled(self, *args, **kwargs):
            calls.append(True)
            return 1.5 * original(self, *args, **kwargs)

        layer._compute_messages = MethodType(scaled, layer)
    elif customize == "convolution":

        class ScaledConv(type(layer.rejector)):
            def forward(self, *args, **kwargs):
                calls.append(True)
                return 1.5 * super().forward(*args, **kwargs)

        layer.rejector.__class__ = ScaledConv
    else:
        # The hook can run even on a lowered layer; consuming the altered
        # live tables, not just invoking the callback, is part of its contract.
        def change_couplings(module, args):
            calls.append(True)
            for name, tensor in module.rejector.tp.named_buffers():
                if "w3j" in name:
                    tensor.zero_()

        layer.register_forward_pre_hook(change_couplings)

    with pytest.raises(ValueError, match=r"interactions\.1"):
        require_native_messages(rep)
    report = plan(model)
    choices = {r.path: r for r in report.regions}
    fast = lower_model(model)
    if customize == "hook":
        # Merely copying an e3nn ScriptModule can change what its program
        # reads after an enclosing callback mutates its buffers.
        assert fast is model
        assert "e3nn generated module" in report.reason
    else:
        assert choices["interactions.0"].provenance == "regional"
        assert choices["interactions.1"].provenance == "host_fallback"
        assert not any(path.startswith("interactions.1.") for path in choices)
    assert not calls
    expected = model(graph("cpu"))
    assert len(calls) == 1
    actual = fast(graph("cpu"))
    assert len(calls) == 2
    for key in EFS:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-8, atol=1e-10)


def physics_model(property_name):
    """HIP's pair readout or the shipped spin-orbit mTACE topology at test width."""

    def host(**cfg):
        cfg["target_property"] = [*EFS, property_name]
        cfg["radial_basis"]["apply_cutoff"] = False
        if property_name == "direct_hessian":
            cfg["special"]["direct_hessian"].update(cutoff=3.0, interaction_type="cgtp")
        else:
            cfg["statistics"] = [
                dict(
                    atomic_numbers=[26], atomic_energy={26: 0.0}, avg_num_neighbors=2.0
                )
            ]
            cfg["fidelity"]["magnetic_scale"] = {26: 5.0}
            cfg["atomic_basis"]["type"] = "uv_o2_mag"
            cfg["node_update"]["magnetic_type"] = "element"
            cfg["angular_basis"].update(magnetic_Lmax=2, use_spin_orbit_coupling=True)
            # Let the magnetic interactions, rather than a one-body head, generate
            # the magnetic force in this regression.
            cfg["readout_emlp"]["use_one_body_magmoms"] = False
        return SphericalTACE(**cfg)

    return make_model(host)


@pytest.mark.parametrize("device,backend", DEVICES)
def test_periodic_les_reciprocal_grid(device, backend, monkeypatch, double_precision):
    les = pytest.importorskip("les")
    if getattr(les, "__version__", None) != "0.2.0":
        pytest.skip("requires LES 0.2.0")
    from equivariantx_foundation.models.tace.planner import optimize_model, plan

    def periodic_model(**cfg):
        options = cfg["long_range"]["les"]
        options["enable"] = True
        options["les_arguments"].update(is_periodic=True, N_max=3)
        return SphericalTACE(**cfg)

    monkeypatch.setenv("TACE_USE_EQX", "0")
    model = make_model(periodic_model).to(device).double()
    path = "les.les.ewald"
    grid = model.readout_fn.les.les.ewald

    def decisions(host, backend):
        report = plan(host, backend=backend, regional_only=True)
        return [r for r in report.regions if r.path == path]

    assert not decisions(model, None)
    with grid.register_forward_hook(lambda *_: None):
        assert not decisions(model, backend)
    with monkeypatch.context() as mp:
        mp.setattr(grid, "is_periodic", False)
        assert not decisions(model, backend)
    with monkeypatch.context() as mp:
        mp.setattr(model.readout_fn.les.les, "compute_bec", True, raising=False)
        assert not decisions(model, backend)
    # A matching shape is not evidence of a static, canonical table. Keep even
    # edits to the discarded side on their host; do not detach a live factor.
    for name, value in (
        ("nvec_all", grid.nvec_all.roll(1, 0)),
        ("hemisphere_mask", ~grid.hemisphere_mask),
        (
            "factors",
            grid.factors.clone().index_fill_(
                0, torch.tensor([0], device=device), float("nan")
            ),
        ),
        ("factors", grid.factors.clone().requires_grad_()),
    ):
        with monkeypatch.context() as mp:
            mp.setitem(grid._buffers, name, value)
            assert not decisions(model, backend)
    adapted = optimize_model(model, backend=backend, regional_only=True)
    compact = adapted.readout_fn.les.les.ewald
    (region,) = decisions(adapted, None)
    assert region.lower is None and region.provider == "les_torch"
    assert compact is not grid and grid.hemisphere_mask.numel() == 343
    assert compact.hemisphere_mask.all() and compact.hemisphere_mask.numel() == 172
    assert compact.nvec_all.equal(grid.nvec_all[grid.hemisphere_mask])
    assert model.state_dict().keys() == adapted.state_dict().keys()

    data = graph(device)
    data["lattice"][:, 1, 0] = 0.8
    reference = model(
        {**data, "atomic_numbers": model.readout_fn.atomic_numbers[data["node_type"]]}
    )
    with_indices = adapted(dict(data))
    for key in ("energy", "forces", "stress", "les_latent_charges"):
        torch.testing.assert_close(
            with_indices[key], reference[key], atol=1e-9, rtol=1e-8
        )
    r = data["positions"].requires_grad_()
    cell = data["lattice"].requires_grad_()
    q = torch.randn(r.shape[0], device=device, dtype=r.dtype, requires_grad=True)

    def derivatives(module):
        energy = module(r=r, q=q, cell=cell, batch=data["batch"])[0]
        first = torch.autograd.grad(energy.sum(), (q, r, cell), create_graph=True)
        mixed = torch.autograd.grad(sum(x.square().sum() for x in first), (q, r, cell))
        return energy, first, mixed

    torch.testing.assert_close(
        derivatives(compact), derivatives(grid), rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("device,backend", DEVICES)
@pytest.mark.parametrize("cartesian", [False, True])
def test_periodic_les_automatic_memory_contract(
    device, backend, cartesian, monkeypatch
):
    if getattr(pytest.importorskip("les"), "__version__", None) != "0.2.0":
        pytest.skip("requires LES 0.2.0")
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace import plan
    from tace.models.cartesian.tace import CartesianTACE

    def charge_model(**cfg):
        cfg.update(
            num_channel=24 if cartesian else 48,
            num_layers=2 if cartesian else 3,
            lmax=3,
        )
        cfg["long_range"]["les"]["enable"] = True
        cfg["long_range"]["les"]["les_arguments"].update(is_periodic=True, N_max=10)
        return (CartesianTACE if cartesian else SphericalTACE)(**cfg)

    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    model = make_model(charge_model).to(device=device, dtype=torch.float32)
    grid = model.readout_fn.les.les.ewald

    def decisions(host, **kwargs):
        return [r for r in plan(host, **kwargs).regions if r.path == "les.les.ewald"]

    for training in (False, True):
        model.train(training)
        if device == "cpu":
            assert not decisions(model)
            assert decisions(model, backend=backend)
            continue
        (r,) = decisions(model)
        assert r.provider == "les_torch" and r.provenance == "regional"
        with grid.register_forward_pre_hook(lambda *_: None):
            assert not decisions(model)
        with monkeypatch.context() as mp:
            mp.setenv("TACE_MEMORY_FIRST", "0")
            assert not decisions(model)
            assert decisions(model, backend=backend)
        with torch.autocast("cuda"):
            assert not decisions(model)
        with monkeypatch.context() as mp:
            # Nonstandard retained vector, with untouched mask and weights.
            changed = grid.nvec_all.clone()
            changed[-1, 1] += 1
            mp.setitem(grid._buffers, "nvec_all", changed)
            assert not decisions(model)
        with monkeypatch.context() as mp:
            mp.setattr(model.readout_fn, "readout_fn", lambda *a, **kw: {})
            assert not decisions(model)
        with monkeypatch.context() as mp:
            mp.setattr(grid, "k_sq_max", grid.k_sq_max / 2)
            assert not decisions(model)
            assert decisions(model, backend=backend)
        adapted = optimize(model, family="tace")
        compact = adapted.readout_fn.les.les.ewald
        assert grid.nvec_all.size(0) == 9261 and compact.nvec_all.size(0) == 4631
        (again,) = decisions(adapted)
        assert again.lower is None and again.provenance == "already_regional"


def test_configured_property_heads_match_called_modules_and_retargeting(
    double_precision,
):
    from contextlib import ExitStack

    from equivariantx_foundation.models.tace.planner import plan

    def multihead(**cfg):
        cfg["target_property"] = [
            "energy",
            "direct_forces",
            "direct_stress",
            "direct_polarizability",
        ]
        return SphericalTACE(**cfg)

    source = make_model(multihead).eval()
    adapted = lower_model(source, backend="torch", regional_only=True)
    for model in (source, adapted):
        report = plan(model, backend="torch", regional_only=True)
        assert all(
            isinstance(model.get_submodule(r.module_path), torch.nn.Module)
            for r in report.regions
        )
        heads = [r for r in report.regions if r.scope == "readout_fn"]
        assert {r.path.split(".")[0] for r in heads} == {
            "energy_readouts",
            "direct_forces_readouts",
            "direct_virials_readout0s",
            "direct_virials_readout2s",
            "direct_polarizability_readout0s",
            "direct_polarizability_readout2s",
        }
        calls = set()
        with ExitStack() as hooks:
            for r in heads:
                assert r.lower is None and r.implementation == "native"
                module = model.get_submodule(r.module_path)
                hooks.enter_context(
                    module.register_forward_hook(lambda *_, key=r.path: calls.add(key))
                )
            assert model(graph("cpu"))["direct_forces"].isfinite().all()
        assert calls == {r.path for r in heads}
    adapted.reset_target_property(["energy"])
    energy_only = plan(adapted, backend="torch")
    assert all(
        r.path.startswith("energy_readouts.")
        for r in energy_only.regions
        if r.scope == "readout_fn"
    )
    assert lower_model(adapted, backend="torch") is adapted


@pytest.mark.parametrize("device,backend", DEVICES)
def test_hip_pair_provider_is_host_owned_and_device_guarded(
    device, backend, monkeypatch, double_precision
):
    from contextlib import ExitStack

    from equivariantx_foundation.models.tace import plan

    monkeypatch.setenv("TACE_USE_EQX", "1")
    source = physics_model("direct_hessian").to(device).eval()
    # Row-wise eligibility is selected on construction, not the planner switch.
    monkeypatch.setenv("TACE_USE_EQX", "0")
    adapted = lower_model(source, backend=backend, regional_only=True)
    for model in (source, adapted):
        regions = {
            r.path: r for r in plan(model, backend=backend, regional_only=True).regions
        }
        pair_path = "pair_representation.interactions.0"
        pair = regions[pair_path]
        assert pair.provenance == "host_selected" and pair.lower is None
        assert pair.provider == ("eqx_cuda" if device == "cuda" else "e3nn")
        assert "autocast off" in pair.reason
        calls = set()
        with ExitStack() as hooks:
            for path in ("pair_representation", pair_path, "direct_hessian_readout"):
                region = regions[path]
                assert (
                    region.scope == "readout_fn" and region.implementation == "native"
                )
                hooks.enter_context(
                    model.get_submodule(region.module_path).register_forward_hook(
                        lambda *_, key=path: calls.add(key)
                    )
                )
            product = model.get_submodule(pair.module_path).rejector
            for key in ("contraction", "tp"):
                hooks.enter_context(
                    getattr(product, key).register_forward_hook(
                        lambda *_, key=key: calls.add(key)
                    )
                )
            assert model(graph(device))["direct_hessian"].isfinite().all()
            assert pair.path in calls
            assert ("contraction" in calls) == (device == "cuda")
            assert ("tp" in calls) == (device == "cpu")
            assert {"direct_hessian_readout", "pair_representation"} <= calls
        with product.register_forward_hook(lambda *_: None):
            assert (
                next(
                    r
                    for r in plan(model, backend=backend, regional_only=True).regions
                    if r.path == pair_path
                ).provider
                is None
            )
        assert not hasattr(product, "_foundation_regional_provider")
        cg = next(v for n, v in product.tp.named_buffers() if "_w3j_" in n)
        cg.requires_grad_()
        guarded = next(
            r
            for r in plan(model, backend=backend, regional_only=True).regions
            if r.path == pair_path
        )
        assert guarded.provider == "e3nn" and guarded.lower is None
        cg.requires_grad_(False)
    adapted.reset_target_property(["energy"])
    assert not any(
        "hessian" in r.path or "pair_representation" in r.path
        for r in plan(adapted, backend=backend).regions
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA magnetic contractions")
@pytest.mark.parametrize("basis", ["uv_o2_mag", "w6j_mag"])
def test_magnetic_zero_spin_and_periodic_self_images(
    basis, monkeypatch, double_precision
):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.tests.test_interaction_schedule import _wigner6j_model

    monkeypatch.setenv("TACE_USE_EQX", "0")
    prop = "noncollinear_magnetic_forces"
    model = (
        (_wigner6j_model() if basis == "w6j_mag" else physics_model(prop))
        .cuda()
        .train()
    )
    prepared = optimize(model, family="tace", regional_only=True, backend="cuda")
    data = graph("cuda")
    data["node_type"].zero_()
    data["initial_noncollinear_magmoms"] = torch.zeros_like(data["positions"])
    data["lattice"] *= 0.72
    for images in (0, 2):
        data["edge_index"] = data["ptr"].new_ones(2, images)
        data["edge_shifts"] = data["ptr"].new_tensor([[1, 0, 0], [-1, 0, 0]])[:images]

        def evaluate(m):
            out = m(dict(data))
            terms = tuple(out[k] for k in (*EFS, prop))
            grads = torch.autograd.grad(
                sum((y - 0.2).square().mean() for y in terms),
                tuple(m.parameters()),
                allow_unused=True,
            )
            return terms, grads

        torch.testing.assert_close(
            evaluate(prepared), evaluate(model), rtol=3e-7, atol=3e-9
        )


@pytest.mark.parametrize("device,backend", DEVICES)
@pytest.mark.parametrize(
    "property_name", ["direct_hessian", "noncollinear_magnetic_forces"]
)
def test_regional_physics_heads_update(device, backend, property_name, monkeypatch):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace import plan

    for var in ACCELERATION_ENV.values():
        monkeypatch.setenv(var, "0")
    torch.set_default_dtype(torch.float64)
    ref = physics_model(property_name).to(device).train()
    source = physics_model(property_name).to(device).train()
    source.load_state_dict(ref.state_dict())
    optimizer = torch.optim.SGD(source.parameters(), lr=2e-3)
    ref_optimizer = torch.optim.SGD(ref.parameters(), lr=2e-3)
    report = plan(source, task="training", backend=backend)
    assert report.schedule == "regional"
    enabled = [r.path for r in report.regions if r.implementation != "native"]
    if property_name == "direct_hessian":
        assert "interactions.0" in enabled and "interactions.1" in enabled
    if not enabled and device != "cpu":
        with pytest.raises(NotImplementedError, match="No Foundation schedule applies"):
            optimize(source, family="tace", task="training", backend=backend)
        return
    fast = (
        lower_model(source, task="training", backend=backend)
        if device == "cpu"
        else optimize(source, family="tace", task="training", backend=backend)
    )
    if not enabled:
        assert fast is source
    assert fast.state_dict().keys() == source.state_dict().keys()
    assert all(
        a is b for a, b in zip(source.parameters(), fast.parameters(), strict=True)
    )
    data = graph(device)
    if property_name == "noncollinear_magnetic_forces":
        data["node_type"].zero_()
        data["initial_noncollinear_magmoms"] = data["positions"].new_tensor(
            [
                [0.21, 0.37, -0.13],
                [0.47, -0.29, 0.54],
                [-0.39, -0.57, 0.44],
                [0.64, 0.33, 0.27],
                [-0.58, 0.49, -0.11],
                [0.36, -0.18, -0.52],
            ]
        )
    # Check the plan is actually used by the property graph. The magnetic
    # interaction itself can be ineligible even when its product is schedulable.
    calls = set()
    rep = fast.readout_fn.representation
    for path in enabled:
        child = rep.get_submodule(path)
        if path.count(".") > 1:
            kernel = child.eqx_linear
        elif path.startswith("interactions."):
            kernel = child.rejector.eqx_tp
        else:
            kernel = child.scheduled_contraction
        kernel.register_forward_hook(lambda _m, _a, _out, key=path: calls.add(key))

    def evaluate(model):
        return model({k: v.detach().clone() for k, v in data.items()})

    def compare(actual, expected):
        for key in (*EFS, property_name):
            assert expected[key].abs().max() > 1e-10, key
            torch.testing.assert_close(
                actual[key], expected[key], rtol=3e-7, atol=2e-9, msg=key
            )

    out, original = evaluate(fast), evaluate(ref)
    assert calls == set(enabled)
    compare(out, original)
    value = out[property_name]
    if property_name == "direct_hessian":
        # Two independent 3-atom structures, all nine ordered pairs in each.
        assert value.shape == (18, 3, 3)
        blocks = value.reshape(2, 3, 3, 3, 3)
        torch.testing.assert_close(blocks, blocks.permute(0, 2, 1, 4, 3))
    else:
        assert value.shape == (6, 3)
        # Independent physical check of the magnetic-force sign and axis order.
        spin = data["initial_noncollinear_magmoms"]
        direction = torch.linspace(-0.4, 0.65, spin.numel(), device=device).reshape_as(
            spin
        )
        energies = [
            fast(
                {**data, "initial_noncollinear_magmoms": spin + sign * 1e-5 * direction}
            )["energy"].sum()
            for sign in (1, -1)
        ]
        torch.testing.assert_close(
            (energies[0] - energies[1]) / 2e-5,
            -(value * direction).sum(),
            atol=2e-7,
            rtol=2e-5,
        )
    for prediction in (out, original):
        observable = prediction[property_name]
        target = torch.linspace(
            -0.3, 0.4, observable.numel(), device=device
        ).reshape_as(observable)
        (efs_loss(prediction) + (observable - target).square().mean()).backward()
    active = set()
    expected_params = dict(ref.named_parameters())
    for name, param in fast.named_parameters():
        grad = expected_params[name].grad
        assert (param.grad is None) == (grad is None), name
        if grad is not None:
            torch.testing.assert_close(param.grad, grad, rtol=4e-7, atol=2e-9, msg=name)
            if grad.abs().max() > 1e-10:
                active.add(name)
    assert any(".interactions." in name for name in active)
    if property_name == "direct_hessian":
        for prefix in ("direct_hessian_readout.", "pair_representation."):
            assert any(prefix in name for name in active)
    optimizer.step()
    ref_optimizer.step()
    after, ref_after = evaluate(fast), evaluate(ref)
    assert not torch.equal(after[property_name], value)
    compare(after, ref_after)
    for name, p in fast.named_parameters():
        torch.testing.assert_close(
            p, expected_params[name], rtol=4e-7, atol=2e-9, msg=name
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="magnetic Uv CUDA path")
def test_magnetic_uv_keeps_independent_inverse_and_magnetic_channel_derivatives(
    monkeypatch,
):
    from equivariantx.o2 import WignerD
    from equivariantx_foundation.models.tace.foundation import shared_parameter_copy
    from equivariantx_foundation.models.tace.magnetic import magnetic_rule

    torch.set_default_dtype(torch.float64)
    model = physics_model("noncollinear_magnetic_forces").cuda().train()
    layer = model.readout_fn.representation.interactions[1]
    conv = layer.rejector
    lower, reason = magnetic_rule(layer, backend="cuda", task="training")
    if conv.use_time_reversal:
        assert lower is None and "time-odd" in reason
        return
    assert lower is not None, reason
    fast_layer = lower(shared_parameter_copy(layer))
    fast = fast_layer.rejector
    restore, _ = magnetic_rule(fast_layer, backend="torch", task="training")
    assert type(restore(shared_parameter_copy(fast_layer)).rejector) is type(conv)
    with monkeypatch.context() as patch:
        patch.setenv("TACE_USE_EQX", "1")
        assert magnetic_rule(layer, backend="cuda", task="inference")[0] is None
    with conv.linear_up.register_forward_hook(lambda *_: None):
        assert magnetic_rule(layer, backend="cuda", task="training")[0] is None
    with monkeypatch.context() as patch:
        patch.setattr(conv.eqx_tp, "gates", ())
        assert magnetic_rule(layer, backend="cuda", task="training")[0] is None

    def rand(*shape):
        return torch.randn(*shape, device="cuda", requires_grad=True)

    idx = torch.randint(0, 9, (2, 23), device="cuda")
    vector, other_vector = rand(23, 3), rand(23, 3)
    wigner = WignerD(mmax=2, lmax=2).cuda()
    frame, _ = wigner(vector)
    _, inv = wigner(other_vector)
    nodes, spin = rand(9, conv.irreps_in.dim), rand(23, conv.magnetic_edge_irreps.dim)
    weights, cutoff = rand(23, conv.weight_numel), rand(23, 1)
    operands = (nodes, spin, weights, vector, other_vector, cutoff, *conv.parameters())

    def derivatives(op):
        out = op(nodes, spin, weights, idx, frame, inv, edge_cutoff=cutoff)
        first = torch.autograd.grad(
            out.sin().sum(), operands, create_graph=True, retain_graph=True
        )
        mixed = torch.autograd.grad(
            first[1].square().sum() + first[4].square().sum(),
            operands,
            retain_graph=True,
        )
        return out, first, mixed

    expected = derivatives(conv)
    called = []
    with fast.eqx_tp.register_forward_hook(lambda *_: called.append(True)):
        actual = derivatives(fast)
    assert called and expected[1][1].abs().max() > 0 and expected[1][4].abs().max() > 0
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed noSOC CUDA producer")
def test_packed_scalar_spin_frames_are_eager_and_reversible(monkeypatch):
    from unittest.mock import patch

    from equivariantx_foundation.models.tace import magnetic
    from equivariantx_foundation.models.tace.planner import optimize_model
    from equivariantx_foundation.tests.test_local_schedule import _scalar_spin_model
    from tace.models.lammps import Graph
    from tace.models.spherical.representation import NodeRepresentation

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    native = _scalar_spin_model(48).cuda().float()
    model = optimize_model(native, task="training", backend="cuda", regional_only=True)
    rep = model.readout_fn.representation
    assert type(rep) is magnetic._ScalarSpinRepresentation
    assert all(
        a is b for a, b in zip(native.parameters(), model.parameters(), strict=True)
    )
    assert native.state_dict().keys() == model.state_dict().keys()
    data = graph("cuda")
    data["initial_noncollinear_magmoms"] = torch.randn_like(data["positions"])
    expected = native(dict(data))
    with patch.object(magnetic, "uv_convolution", wraps=magnetic.uv_convolution) as uv:
        with patch.object(magnetic, "wigner_D", wraps=magnetic.wigner_D) as packed:
            actual = model(dict(data))
    assert packed.call_count == 1 and packed.call_args.kwargs == {"backend": "cuda"}
    assert uv.call_count == 3
    frames = [call.args[5] for call in uv.call_args_list]
    assert all(call.args[6] is frames[0] for call in uv.call_args_list)
    assert all(frame is frames[0] for frame in frames)
    assert frames[0].shape == (data["edge_index"].shape[1], 35)
    for key in (*EFS, "noncollinear_magnetic_forces"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-4, atol=3e-5)

    # Check precisely the frame producer independently of later consumers.
    v = torch.randn(11, 3, device="cuda")
    idx = torch.zeros(2, 11, dtype=torch.long, device="cuda")
    g = Graph(False, None, (0, 0), 1, None, v, v, v, v, idx, idx)

    def dense(data, graph):
        out = rep._spatial_frames(data, graph)
        assert out[0].ndim == 3 and out[1] is not out[0]
        torch.testing.assert_close(out, rep.o2_angular_basis(graph.edge_vector))
        return out

    with rep.interactions[0].rejector.register_forward_hook(lambda *_: None):
        dense({"edge_index": idx}, g)
    dense({"edge_index": idx, "edge_pair_rows": idx[0]}, g)
    dense({"edge_index": idx}, g._replace(lmp=True))
    dense({"edge_index": idx[:, :0]}, g._replace(edge_vector=v[:0]))
    with monkeypatch.context() as mp:
        mp.setenv("TACE_MEMORY_FIRST", "0")
        dense({"edge_index": idx}, g)

    class Frames(torch.nn.Module):
        def __init__(self, representation):
            super().__init__()
            self.rep = representation

        def forward(self, v, idx):
            local = Graph(False, None, (0, 0), 1, None, v, v, v, v, idx, idx)
            return self.rep._spatial_frames({"edge_index": idx}, local)

    # Strict compiler capture intentionally retains the host's two dense outputs.
    captured = torch.export.export(Frames(rep), (v, idx), strict=True).module()
    torch.testing.assert_close(captured(v, idx), rep.o2_angular_basis(v))
    assert (
        type(
            optimize_model(
                model, backend="torch", regional_only=True
            ).readout_fn.representation
        )
        is NodeRepresentation
    )

    scale = rep.o2_angular_basis.cg_2.requires_grad_()
    inverse = dense({"edge_index": idx}, g)[1]
    assert torch.autograd.grad(inverse.square().sum(), scale)[0].abs().max() > 0
    with torch.no_grad():
        scale.requires_grad_(False).mul_(2)
    dense({"edge_index": idx}, g)
    model.double()
    dense({"edge_index": idx}, g._replace(edge_vector=v.double()))
    model.cpu()
    dense({"edge_index": idx.cpu()}, g._replace(edge_vector=v.cpu().double()))
