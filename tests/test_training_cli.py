"""The two training commands share Hydra and data preparation, not execution policy."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf


def test_training_entry_policies(monkeypatch):
    from tace.scripts import train, train_fast, train_oamv2

    cfg = OmegaConf.create({"trainer": {"accumulate_grad_batches": 2}})
    monkeypatch.setattr(train, "build", lambda cfg: (cfg, [], [], [], object(), None))
    fit = Mock()
    monkeypatch.setattr(train, "train", fit)
    monkeypatch.setattr("equivariantx_foundation.registry.require_cuda", Mock())
    monkeypatch.setattr(
        train_oamv2, "configure_environment", Mock(side_effect=AssertionError)
    )
    train.main.__wrapped__(cfg)
    train_fast.launch(cfg)
    assert [c.kwargs["fast"] for c in fit.call_args_list] == [False, True]
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


def test_replay_leaves_implicit_products_to_foundation(monkeypatch):
    from tace.scripts import train, train_fast
    from tace.utils.env import acceleration_enabled, set_env

    monkeypatch.delenv("TACE_USE_EQT", raising=False)
    monkeypatch.setattr("equivariantx_foundation.registry.require_cuda", Mock())

    def selected(cfg, *, fast):
        set_env(cfg)
        return acceleration_enabled("eqt")

    monkeypatch.setattr(train, "train_from_config", selected)
    cfg = OmegaConf.create({"misc": {"env": {}}, "callbacks": {}})
    assert train_fast.launch(cfg) is None
    cfg.callbacks.plan = {"replay_bucket": [10, 20]}
    assert train_fast.launch(cfg) is False
    assert "TACE_USE_EQT" not in os.environ
    monkeypatch.setenv("TACE_USE_EQT", "1")
    assert train_fast.launch(cfg) is True
    cfg.misc.env.TACE_USE_EQT = "0"
    assert train_fast.launch(cfg) is False


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


def test_fast_train_hydra_specialization(tmp_path):
    # Start cold: importing model or Inductor config before selecting the OAM
    # environment would freeze matrix layout / compile settings too early.
    (tmp_path / "base.yaml").write_text("width: 4\n")
    (tmp_path / "recipe.yaml").write_text(
        "defaults: [base, _self_]\nchannels: ${mul:${width},2}\n"
        "misc: {env: {}}\ntrainer: {strategy: {_target_: tace.utils.strategy.SimpleDDPStrategy}}\n"
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
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            "-cn",
            "recipe.yaml",
            "width=6",
            "+fast_schedule=oam_v2",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "captured OAM-v2" in result.stdout
