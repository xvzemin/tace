import json
import os
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Union
from unittest.mock import Mock

import pytest
import torch

import tace.models.compile.wrapper as compile_wrapper
from tace.dataset.quantity import (
    SUPPORT_EMBEDDING_PROPERTY,
    get_embedding_property,
    get_need_property,
)
from tace.lightning.torch_model import (
    _prune_removed_keys,
    _should_warn_without_aoti,
    create_model,
)
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models.compile.aot import (
    TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY,
    AOTICompiledTensorModel,
    _custom_ops_libs_from_model,
    _embed_custom_ops_libs,
    _ensure_sample_inputs,
    _export_metadata,
    _graph_aoti_input_keys,
    _graph_dynamic_shapes,
    _import_custom_ops_libs,
    _size_oblivious_export,
    _synthetic_graph_sample,
)
from tace.models.compile.compile import trace_to_fx
from tace.models.compile.wrapper import CompileTensorModel, _FlatE3nnCompileModel


@pytest.mark.parametrize("use_eqx", [False, True])
def test_aoti_dependencies(tmp_path, monkeypatch, use_eqx):
    model = torch.nn.Linear(2, 2)
    if use_eqx:
        import eqx  # noqa: F401

        graph = torch.fx.Graph()
        operator = torch.ops.eqx.gather_sum.default
        inputs = tuple(
            graph.placeholder(arg.name) for arg in operator._schema.arguments
        )
        graph.output(graph.call_function(operator, inputs))
        model = torch.fx.GraphModule(torch.nn.Module(), graph)
    libs = _custom_ops_libs_from_model(model)
    assert libs == ({"eqx"} if use_eqx else set())

    path = tmp_path / "model.pt2"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/archive_format", "pt2")
    _embed_custom_ops_libs(path, libs)
    if libs:
        with zipfile.ZipFile(path) as archive:
            assert (
                set(
                    archive.read(f"model/{TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY}")
                    .decode()
                    .split()
                )
                == libs
            )

    importer = Mock()
    monkeypatch.setattr("tace.models.compile.aot.importlib.import_module", importer)
    _import_custom_ops_libs(path)
    assert [call.args[0] for call in importer.call_args_list] == sorted(libs)


@pytest.mark.parametrize("use_luna", [False, True], ids=["native", "luna"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_training_and_export(tmp_path, dtype, use_luna):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if use_luna:
        if sys.platform != "linux" or torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("LunaEquivariance requires Linux and compute capability >= 8.0")
        pytest.importorskip("LunaEquivariance")
    from tace.models.compile.aot import export_aotinductor, load_aotinductor
    from tace.utils.env import enable_acceleration

    torch.set_default_dtype(dtype)
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2, num_channel=2, Lmax=1, lmax=1, cutoff=4.0, max_neighbors=None
    )
    config["atomic_basis"]["type"] = "cgtp"
    config["radial_basis"].update(hidden=[4], apply_cutoff=False)
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    targets = ["energy", "forces", "stress"]
    config["target_property"] = targets
    statistics = [
        dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)
    ]
    enable_acceleration(force=True)
    reference = create_model(config, statistics, targets, []).cuda()
    enable_acceleration(enable_luna=use_luna, enable_compile=True, force=True)
    model = create_model(config, statistics, targets, []).cuda()
    model.load_state_dict(reference.state_dict())
    sample = _synthetic_graph_sample(model)
    tolerance = 2e-4 if dtype == torch.float32 else 2e-9
    for step in range(2):
        data = {key: value.clone() for key, value in sample.items()}
        data["positions"] *= 1 + 0.05 * step
        if step:
            data["edge_index"] = data["edge_index"].flip(1)[:, :3]
            data["edge_shifts"] = data["edge_shifts"].flip(0)[:3]
        expected = reference(dict(data))
        actual = model(dict(data))
        for key in targets:
            assert actual[key] is not None and expected[key] is not None
            torch.testing.assert_close(
                actual[key], expected[key], atol=tolerance, rtol=tolerance
            )
        gradients = [
            torch.autograd.grad(
                sum(out[key].square().sum() for key in targets),
                tuple(module.parameters()),
                allow_unused=True,
            )
            for out, module in ((actual, model), (expected, reference))
        ]
        for got, wanted in zip(*gradients):
            assert (got is None) == (wanted is None)
            if got is not None:
                torch.testing.assert_close(got, wanted, atol=tolerance, rtol=tolerance)
        with torch.no_grad():
            for module, grads in zip((model, reference), gradients):
                for parameter, grad in zip(module.parameters(), grads):
                    if grad is not None:
                        parameter.add_(grad, alpha=-1e-4)
    assert model._compiled_cache

    path = export_aotinductor(model, tmp_path / "model.pt2", sample)
    loaded = load_aotinductor(path, "cuda")
    sample = data  # Use a different edge count from the export example.
    expected = reference.eval()(dict(sample))
    actual = loaded(dict(sample))
    for key in targets:
        torch.testing.assert_close(
            actual[key], expected[key], atol=tolerance, rtol=tolerance
        )
    # Loading must restore registrations without constructing the original model.
    case = tmp_path / "case.pt"
    torch.save(
        {
            "sample": sample,
            "expected": {key: expected[key].detach() for key in targets},
        },
        case,
    )
    script = """
import sys
import torch
from tace.models.compile.aot import load_aotinductor
torch.set_num_threads(4)
case = torch.load(sys.argv[2], weights_only=True)
model = load_aotinductor(sys.argv[1], "cuda")
actual = model(case["sample"])
for key, expected in case["expected"].items():
    torch.testing.assert_close(actual[key], expected,
                               atol=float(sys.argv[3]), rtol=float(sys.argv[3]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(case), str(tolerance)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "TACE_USE_LUNA": "0", "TACE_USE_COMPILE": "0"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_constant_construction_precision():
    from e3nn import o3

    from tace.models._e3nn.basis_change import DirectVirials
    from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm
    from tace.models._e3nn.magnetic import MagneticBasis
    from tace.models._e3nn.symmetric_contraction import Contraction
    from tace.models.angular import CartesianHarmonics
    from tace.models.blocks import OneHotToAtomicEnergy, ScaleShift
    from tace.models.linear import e3nnLinear
    from tace.models.radial import (
        AgnesiTransform,
        C2PolynomialCutoff,
        C3PolynomialCutoff,
        CosineCutoff,
        GaussianBasis,
        MollifierCutoff,
        SoftTransform,
        ZBLBasis,
        j0SphericalBesselBasis,
        jnSphericalBesselBasis,
    )

    constants = []
    for dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        modules = torch.nn.ModuleList(
            [
                j0SphericalBesselBasis(cutoff=4.123456789),
                jnSphericalBesselBasis(cutoff=4.123456789, order=2),
                GaussianBasis(cutoff=4.123456789),
                CosineCutoff(4.123456789),
                MollifierCutoff(4.123456789),
                C2PolynomialCutoff(4.123456789),
                C3PolynomialCutoff(4.123456789),
                AgnesiTransform(),
                SoftTransform(),
                ZBLBasis("cosine"),
                OneHotToAtomicEnergy([{1: -0.123456789}], [1]),
                ScaleShift([1], [{1: 0.123456789}], [{1: -0.123456789}]),
                MagneticBasis([3.123456789], 4, 1, [1]),
                CartesianHarmonics(3),
                DirectVirials(),
                EquivariantMergeLayerNorm([0, 1, 2], 2),
                e3nnLinear("2x0e + 2x1o", "2x0e + 2x1o"),
                Contraction(
                    o3.Irreps("2x0e + 2x1o"), o3.Irreps("0e"), 2, num_elements=1
                ),
            ]
        )
        buffers = {
            f"{name}.{key}": value
            for name, module in modules.named_modules()
            if type(module).__module__.startswith("tace.")
            for key, value in module.named_buffers(recurse=False)
            if value.is_floating_point()
        }
        assert buffers
        assert all(value.dtype == torch.float64 for value in buffers.values())
        assert all(parameter.dtype == dtype for parameter in modules.parameters())
        constants.append(buffers)

    assert constants[0].keys() == constants[1].keys()
    for name in constants[0]:
        torch.testing.assert_close(
            constants[0][name], constants[1][name], atol=0, rtol=0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_fidelities", [1, 2])
def test_atomic_energy_lookup(dtype, num_fidelities):
    from tace.models.blocks import OneHotToAtomicEnergy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    elements = list(range(1, 103))
    energies = [
        {z: -100_000.123456789 * z - fidelity for z in elements}
        for fidelity in range(num_fidelities)
    ]
    module = OneHotToAtomicEnergy(energies, elements).to(device=device, dtype=dtype)
    node_type = torch.arange(8192, device=device) % len(elements)
    features = torch.nn.functional.one_hot(node_type, len(elements)).to(dtype)
    expected = module.atomic_energy.T[node_type]
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for n in (0, 1, 128, 3072, 8192):
            for indices in (None, node_type[:n]):
                torch.testing.assert_close(
                    module(features[:n], node_type=indices),
                    expected[:n],
                    atol=0,
                    rtol=0,
                )
        compiled = torch.compile(module, fullgraph=True)
        torch.testing.assert_close(
            compiled(features, node_type), expected, atol=0, rtol=0
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("interaction", ["cgtp", "o2"])
def test_lightning_model_training_precision(dtype, interaction):
    import lightning as L
    from torch.utils.data import DataLoader

    from tace.lightning import create_model

    torch.set_default_dtype(dtype)
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
    )
    config["atomic_basis"]["type"] = interaction
    config["atomic_basis"]["num_head"] = 1
    config["radial_basis"]["hidden"] = [4]
    config["radial_basis"]["apply_cutoff"] = False
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    statistics = [
        {"atomic_numbers": [1], "atomic_energy": {1: 0.0}, "avg_num_neighbors": 2.0}
    ]
    model = create_model(config, statistics, ["energy", "forces"], [])
    assert all(p.dtype == dtype for p in model.parameters())
    assert all(b.dtype == dtype for b in model.buffers() if b.is_floating_point())
    assert model.readout_fn.atomic_numbers.dtype == torch.int64

    class TrainingModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = model

        def training_step(self, batch, batch_idx):
            output = self.model(batch)
            assert output["energy"].dtype == dtype
            assert output["forces"].dtype == dtype
            loss = output["energy"].square().mean() + output["forces"].square().mean()
            assert torch.isfinite(loss)
            return loss

        def on_after_backward(self):
            gradients = [p.grad for p in self.parameters() if p.grad is not None]
            assert gradients
            assert all(g.dtype == dtype and torch.isfinite(g).all() for g in gradients)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-3)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32-true" if dtype == torch.float32 else "64-true",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    data = DataLoader([_magnetic_embedding_sample() for _ in range(2)], batch_size=None)
    trainer.fit(TrainingModel(), train_dataloaders=data)
    assert trainer.global_step == 2


@pytest.mark.parametrize("path_class", [PurePosixPath, PureWindowsPath])
def test_foundation_aoti_output_keeps_version(monkeypatch, path_class):
    from tace.scripts import export_eval

    monkeypatch.setattr(export_eval, "Path", path_class)
    assert (
        export_eval._default_aoti_output_path("TECE-OAM-RRA-1.0")
        == "TECE-OAM-RRA-1.0.pt2"
    )
    assert path_class(
        export_eval._default_aoti_output_path("models/TACE-OAM-7M.pt")
    ) == path_class("models/TACE-OAM-7M.pt2")


@pytest.mark.parametrize("value", [None, "0", "1"])
@pytest.mark.parametrize("training", [False, True])
def test_tf32_environment(monkeypatch, value, training):
    from tace.utils.env import set_tf32
    from tace.utils.utils import set_precision

    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    if value is None:
        monkeypatch.delenv("TACE_USE_TF32", raising=False)
    else:
        monkeypatch.setenv("TACE_USE_TF32", value)
    expected = training if value is None else value == "1"
    configurations = [lambda: set_tf32(training=training)]
    if training:
        configurations.append(lambda: set_precision({"trainer": {"precision": 32}}))
    try:
        for configure in configurations:
            torch.backends.cuda.matmul.allow_tf32 = not expected
            torch.backends.cudnn.allow_tf32 = not expected
            configure()
            assert torch.backends.cuda.matmul.allow_tf32 is expected
            assert torch.backends.cudnn.allow_tf32 is expected
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
            previous
        )


@pytest.mark.parametrize("mask", range(16))
@pytest.mark.parametrize(
    "supported",
    [("cue", "eqt", "oeq", "eqx"), ("cue", "oeq", "eqx"), ("eqt", "eqx"), ("eqt",)],
)
def test_acceleration_priority(monkeypatch, mask, supported):
    from tace.utils.env import ACCELERATION_ENV, select_acceleration

    priority = ("eqx", "oeq", "eqt", "cue")
    enabled = [name for i, name in enumerate(priority) if mask & (1 << i)]
    for name in priority:
        monkeypatch.setenv(ACCELERATION_ENV[name], "1" if name in enabled else "0")
    monkeypatch.setenv("TACE_USE_COMPILE", "1")
    expected = next((name for name in enabled if name in supported), None)
    assert select_acceleration(*supported, kernel="conv") == expected


def test_acceleration_priority_respects_eqx_kernel_switch(monkeypatch):
    from tace.utils.env import EQX_KERNELS, select_acceleration

    monkeypatch.setenv("TACE_USE_EQX", "1")
    monkeypatch.setenv("TACE_USE_OEQ", "1")
    monkeypatch.setitem(EQX_KERNELS, "conv", False)
    assert select_acceleration("eqx", "oeq", kernel="conv") == "oeq"
    assert select_acceleration("eqx", "eqt", kernel="product") == "eqx"


def test_model_input_properties_include_required_embeddings():
    assert {
        "charges",
        "total_charge",
        "initial_collinear_magmoms",
        "initial_noncollinear_magmoms",
    }.issubset(SUPPORT_EMBEDDING_PROPERTY)
    cfg = {
        "loss": {
            "loss_property": [
                "charges",
                "collinear_magnetic_forces",
                "noncollinear_magnetic_forces",
            ]
        },
        "model": {
            "config": {
                "universal_embedding": {
                    "electric_field": {"enable": True},
                    "magnetic_field": {"enable": False},
                },
                "atomic_basis": {"type": "cgtp"},
            }
        },
    }

    assert get_embedding_property(cfg) == [
        "electric_field",
        "total_charge",
        "initial_collinear_magmoms",
        "initial_noncollinear_magmoms",
    ]


def test_model_loading_prunes_removed_architecture_keys():
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["atomic_basis"]["removed_atomic_option"] = True
    config["product_basis"]["removed_product_option"] = True
    config["radial_basis"]["unrelated_option"] = True

    cleaned = _prune_removed_keys(config)

    assert "removed_atomic_option" not in cleaned["atomic_basis"]
    assert "removed_product_option" not in cleaned["product_basis"]
    assert cleaned["radial_basis"]["unrelated_option"] is True
    assert "removed_atomic_option" in config["atomic_basis"]
    assert "removed_product_option" in config["product_basis"]


@pytest.mark.parametrize(
    ("target_property", "expected"),
    [
        (["energy", "forces"], True),
        (["noncollinear_magnetic_forces"], True),
        (["charges"], True),
        (["direct_hessian"], True),
        (["energy", "dipole"], False),
        (["dipole"], False),
        ([], False),
    ],
)
def test_should_warn_without_aoti_requires_supported_target_subset(
    target_property,
    expected,
):
    assert _should_warn_without_aoti(target_property) is expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("interaction_type", ["cgtp", "o2"])
def test_compile_direct_hessian(dtype, interaction_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(dtype)
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    targets = ["energy", "forces", "direct_hessian"]
    config["special"]["direct_hessian"]["interaction_type"] = interaction_type
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=2,
        lmax=2,
        cutoff=4.0,
        max_neighbors=None,
        fidelity=[{"name": "DFT"}],
        target_property=targets,
    )
    config["radial_basis"].update(hidden=[4], apply_cutoff=False)
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    statistics = [
        dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)
    ]
    eager = create_model(config, statistics, targets, []).to(device)
    model = CompileTensorModel(deepcopy(eager.readout_fn)).to(device)
    model.readout_fn.compile_backend = "aot_eager"
    sample = _synthetic_graph_sample(model)
    sample["pbc"].zero_()
    expected = eager(dict(sample))
    actual = model(dict(sample))
    assert "direct_hessian_pair_index" not in sample
    assert "direct_hessian_edge_pairs" not in sample
    assert "direct_hessian_edge_shifts" not in sample
    for key in targets:
        torch.testing.assert_close(actual[key], expected[key])
    expected_grads = torch.autograd.grad(
        sum(expected[key].square().sum() for key in targets),
        tuple(eager.parameters()),
        allow_unused=True,
    )
    actual_grads = torch.autograd.grad(
        sum(actual[key].square().sum() for key in targets),
        tuple(model.parameters()),
        allow_unused=True,
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        if expected_grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, expected_grad)

    model.eval()
    eager.eval()
    keys = _graph_aoti_input_keys(model)
    outputs = model._output_keys()
    _ensure_sample_inputs(sample, keys, model)
    flat_model = _FlatE3nnCompileModel(model, keys, outputs)
    inputs = tuple(sample[key] for key in keys)
    with _size_oblivious_export():
        traced = trace_to_fx(flat_model, inputs)
        exported = torch.export.export(
            traced,
            inputs,
            dynamic_shapes=_graph_dynamic_shapes(keys, num_graphs=2),
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
    metadata = _export_metadata(model, keys, outputs)
    loaded = AOTICompiledTensorModel(exported.module(), metadata, device)
    sample.pop("direct_hessian_pair_index")
    sample.pop("direct_hessian_edge_pairs")
    sample.pop("direct_hessian_edge_shifts")
    original_positions = sample["positions"].clone()
    for distance in (1.0, model.readout_fn.pair_representation.cutoff + 1):
        sample["positions"] = original_positions * distance
        expected = eager(dict(sample))
        actual = loaded(dict(sample))
        for key in targets:
            torch.testing.assert_close(actual[key], expected[key])
        assert "direct_hessian_pair_index" not in sample
        assert "direct_hessian_edge_pairs" not in sample
        assert actual["direct_hessian"].shape == (8, 3, 3)
        if distance > 1:
            assert not actual["direct_hessian"][[1, 2, 5, 6]].any()

    isolated = {key: value[:2] for key, value in sample.items()}
    isolated["ptr"] = sample["ptr"].new_tensor([0, 1, 2])
    isolated["batch"] = sample["batch"].new_tensor([0, 1])
    isolated["edge_index"] = sample["edge_index"][:, :0]
    isolated["edge_shifts"] = sample["edge_shifts"][:0]
    expected = eager(dict(isolated))
    actual = loaded(dict(isolated))
    for key in targets:
        torch.testing.assert_close(actual[key], expected[key])
    assert actual["direct_hessian"].shape == (2, 3, 3)
    isolated = {key: value[:1] for key, value in isolated.items()}
    isolated["ptr"] = sample["ptr"].new_tensor([0, 1])
    isolated["edge_index"] = sample["edge_index"][:, :0]
    with pytest.raises(ValueError, match="at least two atoms"):
        loaded(isolated)

    sample["positions"] = original_positions
    sample["lattice"][0] = sample["lattice"].new_tensor(
        [[12, 0, 0], [2, 14, 0], [1, 2, 16]]
    )
    sample["pbc"][0, 0] = True
    expected = eager(dict(sample))
    for module in (model, loaded):
        actual = module(dict(sample))
        for key in targets:
            torch.testing.assert_close(actual[key], expected[key])
    # Moving one atom by a lattice vector must leave all predictions unchanged.
    sample["positions"] = original_positions.clone()
    sample["positions"][1] += sample["lattice"][0, 0]
    sample["edge_shifts"][:2, 0] = sample["edge_shifts"].new_tensor([-1, 1])
    for module in (eager, model, loaded):
        actual = module(dict(sample))
        for key in targets:
            torch.testing.assert_close(actual[key], expected[key])

    model.reset_target_property(["direct_hessian"])
    assert model._output_keys() == ("direct_hessian",)


class _MagneticEmbeddingReadout(torch.nn.Module):
    def __init__(
        self,
        embedding_property: list[str],
        atomic_basis_type: Union[str, None],
    ) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.register_buffer("cutoff", torch.tensor(4.0))
        self.register_buffer("atomic_numbers", torch.tensor([1], dtype=torch.int64))
        self.target_property = ["energy", "forces"]
        self.embedding_property = embedding_property
        self.model_config = (
            {"atomic_basis": {"type": atomic_basis_type}}
            if atomic_basis_type is not None
            else {}
        )
        self.max_neighbors = None

    def forward(self, data, graph):
        node_energy = self.scale * (
            graph.positions.square().sum(dim=-1)
            + data["initial_noncollinear_magmoms"].square().sum(dim=-1)
        )
        energy = torch.zeros(
            graph.num_graphs,
            dtype=node_energy.dtype,
            device=node_energy.device,
        ).index_add(0, data["batch"], node_energy)
        return {"energy": energy, "node_energy": node_energy}


class _ChargeReadout(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.register_buffer("cutoff", torch.tensor(4.0))
        self.register_buffer("atomic_numbers", torch.tensor([1], dtype=torch.int64))
        self.target_property = ["charges"]
        self.embedding_property = []
        self.model_config = {}
        self.max_neighbors = None

    def forward(self, data, graph):
        raw_charges = self.scale * graph.positions[:, 0]
        graph_charges = torch.zeros_like(data["total_charge"]).index_add(
            0,
            data["batch"],
            raw_charges,
        )
        num_atoms = data["ptr"][1:] - data["ptr"][:-1]
        correction = (data["total_charge"] - graph_charges) / num_atoms
        return {"charges": raw_charges + correction[data["batch"]]}


def _magnetic_embedding_sample() -> dict[str, torch.Tensor]:
    return {
        "pbc": torch.ones(1, 3, dtype=torch.bool),
        "positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "node_attrs": torch.ones(3, 1),
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_shifts": torch.zeros(4, 3),
        "lattice": torch.eye(3).unsqueeze(0) * 10.0,
        "batch": torch.zeros(3, dtype=torch.int64),
        "ptr": torch.tensor([0, 3], dtype=torch.int64),
        "fidelity_idx": torch.zeros(1, dtype=torch.int64),
        "initial_noncollinear_magmoms": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        ),
    }


def test_aoti_charges_requires_total_charge_input():
    model = CompileTensorModel(_ChargeReadout()).eval()
    sample = _magnetic_embedding_sample()
    sample.pop("initial_noncollinear_magmoms")

    with pytest.raises(KeyError, match="total_charge"):
        model._input_keys(sample)

    input_keys = _graph_aoti_input_keys(model)
    assert input_keys[-1] == "total_charge"
    with pytest.raises(KeyError, match="total_charge"):
        _ensure_sample_inputs(dict(sample), input_keys, model)

    sample["total_charge"] = torch.tensor([1.5])
    assert model._input_keys(sample)[-1] == "total_charge"
    output_keys = model._output_keys()
    assert output_keys == ("charges",)
    flat_model = _FlatE3nnCompileModel(model, input_keys, output_keys).eval()
    inputs = tuple(sample[key] for key in input_keys)
    (charges,) = flat_model(*inputs)
    torch.testing.assert_close(charges.sum(), sample["total_charge"].sum())
    traced_model = trace_to_fx(flat_model, inputs)
    (traced_charges,) = traced_model(*inputs)
    torch.testing.assert_close(traced_charges, charges)

    synthetic = _synthetic_graph_sample(model)
    assert synthetic["total_charge"].shape == (2,)
    metadata = _export_metadata(model, input_keys, output_keys)
    assert "total_charge" in json.loads(metadata["tace_input_keys"])
    assert "total_charge" in json.loads(metadata["tace_embedding_property"])
    assert json.loads(metadata["tace_output_keys"]) == ["charges"]


@pytest.mark.parametrize(
    ("training", "grad_enabled"),
    [(False, False), (True, True)],
)
def test_compiled_call_uses_grad_only_during_training(
    monkeypatch,
    training,
    grad_enabled,
):
    model = CompileTensorModel(_ChargeReadout()).train(training)
    sample = _magnetic_embedding_sample()
    sample.pop("initial_noncollinear_magmoms")
    sample["total_charge"] = torch.tensor([1.5])
    observed = []

    monkeypatch.setattr(
        compile_wrapper,
        "trace_and_compile",
        lambda *args, **kwargs: (object(), (), ()),
    )

    def fake_compiled_call(*args, **kwargs):
        observed.append(torch.is_grad_enabled())
        return (torch.zeros(3),)

    monkeypatch.setattr(compile_wrapper, "compiled_call", fake_compiled_call)

    with torch.enable_grad():
        model._compiled_forward(sample)

    assert observed == [grad_enabled]


def _slice_update(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    output = x.clone()
    output[:, 1:2, :] = value
    return output


def test_trace_to_fx_removes_full_dynamic_slice_scatter():
    x = torch.randn(4, 3, 5)
    value = torch.randn(4, 1, 5)
    traced = trace_to_fx(_slice_update, (x, value), functionalize=True)

    for node in traced.graph.nodes:
        if node.op != "call_function":
            continue
        assert node.target != torch.ops.aten.copy_.default
        if node.target == torch.ops.aten.slice_scatter.default:
            start = node.args[3] if len(node.args) > 3 else None
            end = node.args[4] if len(node.args) > 4 else None
            assert (start, end) != (0, sys.maxsize)

    torch.testing.assert_close(traced(x, value), _slice_update(x, value))


@pytest.mark.parametrize(
    ("embedding_property", "atomic_basis_type"),
    [
        pytest.param([], "o2_mag", id="o2-magnetic"),
        pytest.param(
            ["initial_noncollinear_magmoms"],
            "cgtp",
            id="universal-embedding",
        ),
    ],
)
def test_aoti_keeps_magnetic_embedding_without_magnetic_force_target(
    embedding_property,
    atomic_basis_type,
):
    model = CompileTensorModel(
        _MagneticEmbeddingReadout(embedding_property, atomic_basis_type)
    ).eval()
    sample = _magnetic_embedding_sample()

    assert "initial_noncollinear_magmoms" in get_need_property(
        model.get_target_property(),
        model.get_embedding_property(),
        training=True,
    )
    assert "noncollinear_magnetic_forces" not in get_need_property(
        model.get_target_property(),
        model.get_embedding_property(),
        training=True,
    )

    input_keys = _graph_aoti_input_keys(model)
    output_keys = model._output_keys()
    assert "initial_noncollinear_magmoms" in input_keys
    assert "noncollinear_magnetic_forces" not in output_keys

    flat_model = _FlatE3nnCompileModel(model, input_keys, output_keys).eval()
    output = flat_model(*(sample[key] for key in input_keys))
    zero_magmoms = dict(sample)
    zero_magmoms["initial_noncollinear_magmoms"] = torch.zeros_like(
        sample["initial_noncollinear_magmoms"]
    )
    output_without_magmoms = flat_model(*(zero_magmoms[key] for key in input_keys))
    assert output_keys == ("energy", "node_energy", "forces")
    assert not torch.equal(output[0], output_without_magmoms[0])
    torch.testing.assert_close(output[2], output_without_magmoms[2])
    torch.testing.assert_close(output[2], -2.0 * sample["positions"])

    metadata = _export_metadata(model, input_keys, output_keys)
    assert "initial_noncollinear_magmoms" in json.loads(metadata["tace_input_keys"])
    assert "initial_noncollinear_magmoms" in json.loads(
        metadata["tace_embedding_property"]
    )
    assert "noncollinear_magnetic_forces" not in json.loads(
        metadata["tace_output_keys"]
    )


def test_import_without_cpu_affinity_or_cuda(tmp_path):
    script = """
import os
import sys

if hasattr(os, "sched_getaffinity"):
    del os.sched_getaffinity
sys.modules["torch.utils.cpp_extension"] = None

from tace.interface.ase import TACEAseCalc
from tace.lightning import load_tace
from tace.scripts.train import main
from eqx.kernels import cuda

assert not hasattr(os, "sched_getaffinity")
assert not cuda._POOL._threads
assert not cuda.runtime.cache_info().currsize
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "MPLCONFIGDIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
