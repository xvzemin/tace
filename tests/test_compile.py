import importlib
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
from tace.foundations import resolve_model_path, tace_foundations
from tace.lightning.torch_model import (
    _prune_removed_keys,
    _should_warn_without_aoti,
)
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models.compile.aot import (
    EQX_CUSTOM_OPS_MODULES,
    TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY,
    _custom_ops_libs_from_model,
    _embed_custom_ops_libs,
    _ensure_sample_inputs,
    _export_metadata,
    _graph_aoti_input_keys,
    _import_custom_ops_libs,
    _synthetic_graph_sample,
)
from tace.models.compile.compile import trace_to_fx
from tace.models.compile.wrapper import CompileTensorModel, _FlatE3nnCompileModel


def test_aoti_eqx_function_dependencies():
    importlib.import_module("eqx.kernels.layout")

    graph = torch.fx.Graph()
    features = graph.placeholder("features")
    indices = graph.placeholder("indices")
    output = graph.call_function(
        torch.ops.eqx.gather_sum.default,
        ([features, features], [None, indices]),
    )
    graph.output(graph.call_function(torch.ops.aten.neg.default, (output,)))
    model = torch.fx.GraphModule(torch.nn.Module(), graph)
    assert _custom_ops_libs_from_model(model) == {"eqx.kernels.layout"}
    assert _custom_ops_libs_from_model(torch.nn.Linear(2, 2)) == set()


@pytest.mark.parametrize("libs", [set(), {"openequivariance"}, {"eqx.kernels.layout"}])
def test_aoti_import_eqx_dependencies(tmp_path, monkeypatch, libs):
    path = tmp_path / "model.pt2"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/archive_format", "pt2")
        archive.writestr(
            "model/data/aotinductor/model/test.wrapper.json",
            json.dumps(
                {
                    "nodes": [
                        {"node": {"target": target}}
                        for target in (
                            "eqx::gather_sum",
                            "eqx::gather_sum",
                            "eqx::element_linear",
                            "aten::sin",
                        )
                    ]
                }
            ),
        )
    _embed_custom_ops_libs(path, libs)
    if libs:
        with zipfile.ZipFile(path) as archive:
            assert set(
                archive.read(f"model/{TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY}").decode().split()
            ) == libs

    importer = Mock()
    monkeypatch.setattr("tace.models.compile.aot.importlib.import_module", importer)
    _import_custom_ops_libs(path)
    assert [call.args[0] for call in importer.call_args_list] == sorted(
        libs | {"eqx.kernels.layout", "eqx.o3.contraction"}
    )


def test_aoti_without_custom_ops(tmp_path, monkeypatch):
    path = tmp_path / "model.pt2"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/archive_format", "pt2")
    importer = Mock()
    monkeypatch.setattr("tace.models.compile.aot.importlib.import_module", importer)
    _import_custom_ops_libs(path)
    importer.assert_not_called()


def test_aoti_eqx_registration_modules():
    for lib in sorted(set(EQX_CUSTOM_OPS_MODULES.values())):
        importlib.import_module(lib)
    for name in EQX_CUSTOM_OPS_MODULES:
        assert torch._C._dispatch_find_schema_or_throw(name, "").schema().name == name


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


@pytest.fixture
def foundation_model(tmp_path, monkeypatch, double_precision):
    import tace.foundations.download_link as downloads
    from tace.lightning import create_model, export_tace

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=1, num_channel=2, Lmax=1, lmax=1, cutoff=4.0, max_neighbors=None
    )
    config["radial_basis"]["hidden"] = [4]
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    statistics = [
        {
            "atomic_numbers": [1],
            "atomic_energy": {1: 0.0},
            "avg_num_neighbors": 2.0,
        }
    ]
    model = create_model(config, statistics, ["energy", "forces"], [])
    path = tmp_path / "download.pt"
    export_tace(model, str(path))
    cache = tmp_path / "cache"
    cache.mkdir()
    download = Mock(return_value=str(path))
    monkeypatch.setattr(downloads, "CACHE_DIR", cache)
    monkeypatch.setattr(downloads, "hf_hub_download", download)
    return path, download


@pytest.mark.parametrize("name", ["TACE-OAM-7M", "TECE-OAM-RRA-1.0"])
def test_foundation_name_resolution(foundation_model, name):
    source, download = foundation_model
    assert name in tace_foundations
    assert "unknown" not in tace_foundations
    download.assert_not_called()
    path = resolve_model_path(name)
    assert path.name == name + ".pt"
    assert path.read_bytes() == source.read_bytes()
    assert resolve_model_path(Path(name)) == path
    assert resolve_model_path(path) == path
    download.assert_called_once_with(
        repo_id="xvzemin/tace-foundations",
        filename=name + ".pt",
        revision="main",
        cache_dir=path.parent,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_load_foundation_matches_file(foundation_model, dtype):
    from tace.lightning import load_tace

    source, download = foundation_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = load_tace(source, device=device, dtype=dtype).eval()
    module = load_tace("TACE-OAM-7M", device=device, dtype=dtype).eval()
    assert module.get_model_dtype() == dtype
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(module.state_dict()[name], value, atol=0, rtol=0)
    data = {
        key: value.to(device) for key, value in _magnetic_embedding_sample().items()
    }
    results = [
        model({key: value.clone() for key, value in data.items()})
        for model in (reference, module)
    ]
    for key in ("energy", "forces"):
        torch.testing.assert_close(results[0][key], results[1][key])
    assert load_tace(module, device=device, target_property=["energy"]) is module
    assert module.get_target_property() == ["energy"]
    download.assert_called_once()


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".PT", ".ckpt", ".pt2"])
def test_local_model_dispatch(foundation_model, tmp_path, monkeypatch, suffix):
    import tace.models.compile as compile_model
    from tace.lightning import load_tace
    from tace.lightning.lit_model import LightningWrapperModel

    source, download = foundation_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = load_tace(source, device=device)
    path = tmp_path / ("TACE-OAM-7M" + suffix)
    checkpoint = Mock(return_value=reference)
    aoti = Mock(return_value=reference)
    monkeypatch.setattr(LightningWrapperModel, "load_from_checkpoint", checkpoint)
    monkeypatch.setattr(compile_model, "load_aotinductor", aoti)
    if suffix.lower() in (".pt", ".pth"):
        path.write_bytes(source.read_bytes())
    module = load_tace(path, device=device, use_ema=False)
    assert module.get_model_dtype() == reference.get_model_dtype()
    assert checkpoint.call_count == (suffix == ".ckpt")
    assert aoti.call_count == (suffix == ".pt2")
    if suffix == ".ckpt":
        checkpoint.assert_called_once_with(
            path, map_location=device, strict=True, use_ema=False, dtype=None
        )
    elif suffix == ".pt2":
        aoti.assert_called_once_with(path, device)
    download.assert_not_called()


def test_unknown_model_does_not_download(foundation_model, tmp_path):
    from tace.lightning import load_tace

    _, download = foundation_model
    for name in ("unknown", "model.bin", str(tmp_path / "TACE-OAM-7M")):
        with pytest.raises(ValueError, match="registered foundation model"):
            load_tace(name)
    with pytest.raises(FileNotFoundError):
        load_tace(tmp_path / "TACE-OAM-7M.pt")
    download.assert_not_called()


@pytest.mark.parametrize(
    "script,output",
    [
        ("export_train", "-state.pt"),
        ("export_eval", "-state_dict.pt"),
        ("convert_cgtp", "-converted.pt"),
    ],
)
def test_foundation_cli_exports(
    foundation_model, monkeypatch, tmp_path, script, output
):
    import importlib

    name = "TECE-OAM-RRA-1.0"
    module = importlib.import_module("tace.scripts." + script)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [script, "-m", name, "--device", device])
    module.main()
    assert (tmp_path / (name + output)).is_file()
    foundation_model[1].assert_called_once()


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


def test_foundation_ase_and_finetune(foundation_model):
    from ase import Atoms

    from tace.interface.ase import TACEAseCalc
    from tace.lightning.lit_model import finetune

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for model in (foundation_model[0], "TACE-OAM-7M"):
        atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0.2, 0]], cell=[6, 6, 6])
        atoms.calc = TACEAseCalc(
            model, device=device, dtype="float64", neighborlist_backend="ase"
        )
        results.append((atoms.get_potential_energy(), atoms.get_forces()))
    for actual, expected in zip(*results):
        torch.testing.assert_close(
            torch.as_tensor(actual), torch.as_tensor(expected), atol=1e-12, rtol=1e-12
        )
    model = finetune(
        {"finetune_from_model": "TACE-OAM-7M", "trainer": {"precision": 64}}
    )
    assert model.training
    assert model.get_model_dtype() == torch.float64
    foundation_model[1].assert_called_once()


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
