"""The two training commands share Hydra and data preparation, not execution policy."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def xl_training_recipe():
    return OmegaConf.create(
        {
            "model": {
                "config": {
                    "_target_": "tace.models.SphericalTACE",
                    "num_layers": 10,
                    "num_channel": 128,
                    "Lmax": 4,
                    "lmax": 4,
                    "mlp_type": "mlp_v2",
                    "radial_basis": {"radial_basis": "j0"},
                    "resnet": {"linear_type": "identity"},
                    "edge_embedding": {"type": "element"},
                    "atomic_basis": {"type": "cgtp", "scatter_norm": "cutoff"},
                    "product_basis": {"type": "nonlinear_cgtp", "correlation": 2},
                }
            },
            "optimizer": {"_target_": "tace.utils.optimizer.HybridMuonOptimizer"},
            "loss": {
                "_target_": "tace.utils.loss.NormalLoss",
                "loss_property": ["energy", "forces", "stress"],
                "loss_function_name": [
                    "mae_energy_per_atom",
                    "l2mae_forces",
                    "mae_voigt_stress",
                ],
            },
            "misc": {"env": {}},
            "trainer": {
                "strategy": {"_target_": "tace.utils.strategy.SimpleDDPStrategy"},
                "inference_mode": False,
            },
        }
    )


def test_fast_train_automatic_contract(xl_training_recipe, monkeypatch):
    from tace.scripts import train_fast, train_oamv2

    for name in train_oamv2.EXECUTION:
        monkeypatch.delenv(name, raising=False)
    cfg = xl_training_recipe
    original = OmegaConf.to_container(cfg)
    assert train_fast.select_schedule(cfg) == "oam_v2"
    assert OmegaConf.to_container(cfg) == original
    for key, value in (
        ("model.config.product_basis.correlation", 3),
        ("model.config.num_channel", 64),
        ("model.config.long_range.les.enable", True),
        ("model.config.dropout.stochastic_depth", 0.1),
        ("optimizer._target_", "torch.optim.AdamW"),
        ("loss.loss_function_name.1", "mse_forces"),
        ("loss.loss_property.2", "direct_hessian"),
        ("trainer.devices", [0]),
        ("trainer.accumulate_grad_batches", 2),
        ("trainer.precision", "bf16-mixed"),
        ("dtype.matmul", "ieee"),
        (
            "callbacks.custom._target_",
            "tace.utils.callbacks.FoundationTrainingCallback",
        ),
        ("resume_from_model", "another.ckpt"),
        ("misc.env.TACE_USE_EQX", "0"),
        ("fast_schedule", "foundation"),
    ):
        changed = OmegaConf.create(original)
        OmegaConf.update(changed, key, value)
        assert train_fast.select_schedule(changed) == "foundation", key
    monkeypatch.setenv("TACE_USE_MATRIX_WEIGHT", "0")
    assert train_fast.select_schedule(cfg) == "foundation"


def test_fast_train_resume_uses_checkpoint_metadata(
    xl_training_recipe, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import torch

    from tace.lightning.oamv2 import OAMLightningModule
    from tace.scripts import train_fast, train_oamv2
    from tace.utils.optimizer import HybridMuonOptimizer

    for key in train_oamv2.EXECUTION:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = xl_training_recipe
    saved = OmegaConf.to_container(cfg, resolve=True)
    saved["dtype"] = {"model": "float32", "matmul": "tf32", "autocast": None}
    table = torch.nn.Parameter(torch.eye(2))
    bias = torch.nn.Parameter(torch.ones(2))
    optimizer = HybridMuonOptimizer(
        [{"params": [table], "use_muon": True}, {"params": [bias], "use_muon": False}]
    )
    (table.sum() + bias.sum()).backward()
    optimizer.step()
    state = {}
    for i in range(10):
        key = f"model.readout_fn.representation.interactions.{i}.linear_up"
        state[f"{key}.linear.output_mask"] = torch.ones(2)
        state[f"{key}.weight.0"] = table.detach().clone()
    checkpoint = {
        "state_dict": state,
        "global_step": 1,
        "loops": {"fit_loop": {}},
        "hyper_parameters": {
            "cfg": saved,
            "target_property": ["energy", "forces", "stress"],
            "embedding_property": [],
        },
        "optimizer_states": [optimizer.state_dict()],
    }
    OAMLightningModule.on_save_checkpoint(
        SimpleNamespace(canonical_model=None, capture_limit=256), checkpoint
    )
    path = tmp_path / "resume.ckpt"
    cfg.resume_from_model = path.name
    load = torch.load

    def metadata(file, **kwargs):
        assert kwargs == dict(mmap=True, map_location="meta", weights_only=True)
        return load(file, **kwargs)

    monkeypatch.setattr(torch, "load", metadata)

    def schedule():
        torch.save(checkpoint, path)
        return train_fast.select_schedule(cfg)

    assert schedule() == "oam_v2"
    for container, key, value in [
        (checkpoint, "oam_execution", {}),
        (checkpoint, "optimizer_states", []),
        (saved["model"]["config"], "lmax", 6),
        (saved["model"]["config"]["radial_basis"], "radial_basis", "gaussian"),
        (saved["model"]["config"], "cutoff", 7.0),
        (saved["loss"], "loss_function_name", ["mse_energy"]),
        (saved["optimizer"], "_target_", "torch.optim.AdamW"),
        (saved["dtype"], "model", "float64"),
        (
            state,
            "model.readout_fn.representation.interactions.0.linear_up.weight",
            table.flatten(),
        ),
        (
            state,
            "model.readout_fn.representation.interactions.2.linear_up.weight.0",
            table.flatten(),
        ),
        (checkpoint["optimizer_states"][0], "state", {}),
    ]:
        with monkeypatch.context() as patch:
            patch.setitem(container, key, value)
            assert schedule() == "foundation", key
    path.write_bytes(b"not a checkpoint")
    assert train_fast.select_schedule(cfg) == "foundation"
    with monkeypatch.context() as patch:
        patch.setattr(
            torch,
            "load",
            lambda *a, **kw: pytest.fail("explicit schedules must not probe the file"),
        )
        for selected in ("foundation", "oam_v2"):
            cfg.fast_schedule = selected
            assert train_fast.select_schedule(cfg) == selected


def test_training_entry_policies(monkeypatch):
    from tace.scripts import train, train_fast, train_oamv2
    from tace.utils.env import acceleration_enabled

    cfg = OmegaConf.create({"trainer": {"accumulate_grad_batches": 2}})
    monkeypatch.delenv("TACE_USE_EQX", raising=False)
    building, fitting = [], []

    def build(cfg):
        building.append(acceleration_enabled("eqx", kernel="product"))
        return cfg, [], [], [], object(), None

    monkeypatch.setattr(train, "build", build)
    fit = Mock(
        side_effect=lambda **kw: fitting.append(
            acceleration_enabled("eqx", kernel="product")
        )
    )
    monkeypatch.setattr(train, "train", fit)
    monkeypatch.setattr("equivariantx_foundation.registry.require_cuda", Mock())
    monkeypatch.setattr(
        train_oamv2, "configure_environment", Mock(side_effect=AssertionError)
    )
    train.main.__wrapped__(cfg)
    train_fast.launch(cfg)
    assert [c.kwargs["fast"] for c in fit.call_args_list] == [False, True]
    assert building == [True, False] and fitting == [True, True]
    assert all(
        c.kwargs["module_type"] is train.LightningWrapperModel
        for c in fit.call_args_list
    )
    assert cfg.trainer.accumulate_grad_batches == 2
    cfg.fast_schedule = "oam_v2"
    with pytest.raises(ValueError, match="tace-fast-train"):
        train.main.__wrapped__(cfg)
    cfg.fast_schedule = "typo"
    with pytest.raises(ValueError, match="fast_schedule"):
        train_fast.launch(cfg)


def test_fast_train_without_cuda(monkeypatch, tmp_path, capsys):
    from tace.scripts import train_fast, train_oamv2

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        train_oamv2, "configure_environment", Mock(side_effect=AssertionError)
    )
    for schedule in ("foundation", "oam_v2"):
        with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA device"):
            train_fast.launch(OmegaConf.create({"fast_schedule": schedule}))
    # Inspecting config outside the source tree never checks for the backend.
    monkeypatch.setattr(
        "equivariantx_foundation.registry.require_cuda",
        Mock(side_effect=AssertionError),
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tiny.yaml").write_text("example: 42\n")
    monkeypatch.setattr(sys, "argv", ["tace-fast-train", "-cn", "tiny", "--cfg", "job"])
    train_fast.main()
    assert "example: 42" in capsys.readouterr().out


def test_fast_train_sweep_does_not_mix_process_policies(monkeypatch):
    from tace.scripts import train_fast

    jobs = [
        OmegaConf.create({"fast_schedule": s})
        for s in ("oam_v2", "oam_v2", "foundation")
    ]
    monkeypatch.setattr(
        train_fast, "training_cli", lambda task: lambda: [task(cfg) for cfg in jobs]
    )
    launch = Mock()
    monkeypatch.setattr(train_fast, "launch", launch)
    with pytest.raises(ValueError, match="separate processes"):
        train_fast.main()
    assert [call.args[0] for call in launch.call_args_list] == jobs[:2]


@pytest.mark.parametrize("schedule", ["auto", "oam_v2"])
def test_fast_train_hydra_specialization(tmp_path, schedule, xl_training_recipe):
    # Start cold: importing model or Inductor config before selecting the OAM
    # environment would freeze matrix layout / compile settings too early.
    xl_training_recipe.width = 4
    OmegaConf.save(xl_training_recipe, tmp_path / "base.yaml")
    (tmp_path / "recipe.yaml").write_text(
        "defaults: [base, _self_]\nchannels: ${mul:${width},2}\n"
    )
    script = """
import os, sys
from types import ModuleType
from unittest.mock import patch
from tace.scripts import train_fast
import torch
assert not torch.cuda.is_initialized()
assert not {'equivariantx','equivariantx_foundation','e3nn','lightning','torch._inductor.config'} & sys.modules.keys()
def fit(cfg, module_type, *, fast):
    import torch._inductor.config as inductor
    assert fast and module_type.__name__ == 'OAMLightningModule'
    assert cfg.channels == 12 and cfg.trainer.accelerator == 'cuda' and cfg.trainer.devices == 'auto'
    assert os.environ['TACE_USE_MATRIX_WEIGHT'] == '1' and os.environ['TACE_MATMUL_PRECISION'] == 'tf32'
    assert inductor.compile_threads == 12
    print('captured OAM-v2')
stub = ModuleType('tace.scripts.train')
stub.train_from_config = fit
sys.modules[stub.__name__] = stub
with patch('equivariantx_foundation.registry.require_cuda'), patch('torch.cuda.set_device') as select:
    train_fast.main()
    select.assert_called_once_with(0)
"""
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        CUDA_VISIBLE_DEVICES="",
        LOCAL_RANK="0",
    )
    from tace.scripts.train_oamv2 import EXECUTION

    for name in EXECUTION:
        env.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            "-cn",
            "recipe.yaml",
            "width=6",
            *(["+fast_schedule=oam_v2"] if schedule == "oam_v2" else []),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "captured OAM-v2" in result.stdout
