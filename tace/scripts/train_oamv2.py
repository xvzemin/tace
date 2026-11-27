################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Train TACE-OAM-XL-v2 on all visible CUDA devices."""

import os
import pickle
import sys
from collections.abc import Mapping
from pathlib import Path

EXECUTION = {
    "EQX_XL_CG_REGISTER_LIMIT": "255",
    "EQX_XL_CG_ROW_SIZE": "16",
    "EQX_XL_COTANGENT_RESIDENT": "1",
    "EQX_XL_EDGE_ANGULAR_BLOCK": "1",
    "EQX_XL_LAUNCH_CONTRACT": "1",
    "EQX_XL_LAYOUT_ADJOINTS": "1",
    "EQX_XL_LT_ALGORITHM": "0",
    "EQX_XL_LT_RADIAL": "1",
    "EQX_XL_LT_SM_TARGET": "56",
    "EQX_XL_MIXED_OWNER": "source",
    "EQX_XL_MIXED_REUSE_ANGULAR": "1",
    "EQX_XL_MIXED_ROW_LIMIT": "16",
    "EQX_XL_NODE_SWEEP": "1",
    "EQX_XL_NODE_TANGENT": "1",
    "EQX_XL_ORDERED_REDUCTIONS": "1",
    "EQX_XL_PACKED_EDGES": "1",
    "EQX_XL_PACKED_FRAGMENTS": "1",
    "EQX_XL_PARALLEL_MATRICES": "1",
    "EQX_XL_POLYNOMIAL_PRODUCTS": "auto",
    "EQX_XL_RECEIVER_PIPELINE": "auto",
    "EQX_XL_SKIP_EDGE_ZERO": "1",
    "EQX_XL_SOURCE_ROW_LIMIT": "16",
    "EQX_XL_SOURCE_VJP": "1",
    "EQX_XL_SOURCE_VJP_REGISTER_LIMIT": "192",
    "EQX_XL_STREAM_ANGULAR_OUTPUTS": "auto",
    "OMP_NUM_THREADS": "4",
    "PYTORCH_CUDA_ALLOC_CONF": "roundup_power2_divisions:2",
    "TACE_USE_COMPILE": "1",
    "TACE_USE_CUE": "0",
    "TACE_USE_DENS": "0",
    "TACE_USE_EQX": "1",
    "TACE_USE_LUNA": "0",
    "TACE_USE_MATRIX_WEIGHT": "1",
    "TACE_USE_OEQ": "0",
    "TORCHINDUCTOR_COMPILE_THREADS": "12",
    "TACE_MATMUL_PRECISION": "tf32",
    "TACE_USE_TF32": "1",
}


def matches_config(cfg):
    """Match the benchmarked XL-v2 joint E/F/S training configuration."""
    model = cfg.get("model", {}).get("config", {})
    if (
        model.get("_target_") != "tace.models.SphericalTACE"
        or (model.get("num_layers"), model.get("num_channel")) != (10, 128)
        or (model.get("Lmax"), model.get("lmax")) != (4, 4)
        or cfg.get("finetune_from_model")
    ):
        return False

    inter = model.get("atomic_basis", {})
    prod = model.get("product_basis", {})
    trainer = cfg.get("trainer", {})
    strategy = trainer.get("strategy")
    loss = cfg.get("loss", {})
    misc = cfg.get("misc", {})
    if not (
        model.get("wrapper", {}).get("_target_", "tace.models.TensorModel")
        == "tace.models.TensorModel"
        and inter.get("type") == "cgtp"
        and inter.get("scatter_norm") == "cutoff"
        and prod.get("type") == "nonlinear_cgtp"
        and prod.get("correlation") == 2
        and model.get("resnet", {}).get("linear_type") == "identity"
        and model.get("mlp_type") == "mlp_v2"
        and model.get("node_embedding", {}).get("type", "linear") == "linear"
        and model.get("edge_update", {}).get("type", "identity") == "identity"
        and model.get("edge_embedding", {}).get("type") == "element"
        and model.get("radial_basis", {}).get("radial_basis") == "j0"
        and not model.get("dropout", {}).get("stochastic_depth")
        and not model.get("long_range", {}).get("les", {}).get("enable")
        and not any(
            v.get("enable") for v in model.get("universal_embedding", {}).values()
        )
        and loss.get("_target_") == "tace.utils.loss.NormalLoss"
        and tuple(loss.get("loss_property", ())) == ("energy", "forces", "stress")
        and tuple(loss.get("loss_function_name", ()))
        == ("mae_energy_per_atom", "l2mae_forces", "mae_voigt_stress")
        and cfg.get("optimizer", {}).get("_target_")
        == "tace.utils.optimizer.HybridMuonOptimizer"
        and misc.get("device", "cuda") == "cuda"
        and not misc.get("LossSkipController", {}).get("enable")
        and trainer.get("accelerator", "auto") in ("auto", "gpu", "cuda")
        and trainer.get("devices", "auto") == "auto"
        and trainer.get("accumulate_grad_batches", 1) == 1
        and not trainer.get("inference_mode", True)
        and str(trainer.get("precision", "32-true")) in ("32", "32-true")
        and isinstance(strategy, Mapping)
        and strategy.get("_target_") == "tace.utils.strategy.SimpleDDPStrategy"
    ):
        return False
    for callback in (cfg.get("callbacks") or {}).values():
        if isinstance(callback, Mapping) and (
            str(callback.get("_target_", "")).endswith(".FoundationTrainingCallback")
            or "training_package" in callback
            or "replay_bucket" in callback
        ):
            return False
    configured = misc.get("env") or {}
    for name, expected in EXECUTION.items():
        if name in configured and str(configured[name]) != expected:
            return False
        if (
            name.startswith(("TACE_", "EQX_"))
            and os.environ.get(name, expected) != expected
        ):
            return False
    from tace.utils.precision import resolve_precision

    policy = resolve_precision({"dtype": dict(cfg.get("dtype") or {})}, training=True)
    if (policy.model, policy.autocast, policy.matmul) != ("float32", None, "tf32"):
        return False
    resume = cfg.get("resume_from_model")
    return not resume or matches_checkpoint(cfg, resume)


def matches_checkpoint(cfg, path):
    """Match a canonical OAM training checkpoint without materializing tensor storage."""
    import torch
    from hydra.utils import to_absolute_path

    try:
        path = Path(to_absolute_path(str(Path(path).expanduser())))
        if path.suffix != ".ckpt" or not path.is_file():
            return False
        checkpoint = torch.load(path, weights_only=True, map_location="meta", mmap=True)
        marker = checkpoint["oam_execution"]
        hparams = checkpoint["hyper_parameters"]
        saved = hparams["cfg"]
        if not (
            marker["format"] == 1
            and marker["canonical_tace_state"] is True
            and marker["model_name"] == "TACE-OAM-XL-v2"
            and marker["model_precision"] == "fp32/fp64/tf32"
            and tuple(hparams["target_property"]) == ("energy", "forces", "stress")
            and not hparams["embedding_property"]
            and saved["loss"] == cfg["loss"]
            and saved["optimizer"] == cfg["optimizer"]
            and saved["dtype"]["model"] == "float32"
            and saved["dtype"].get("autocast") is None
            and saved["dtype"]["matmul"] == "tf32"
            and "fit_loop" in checkpoint["loops"]
            and checkpoint["global_step"] > 0
        ):
            return False
        # create_model fills these fields from graph statistics and loss properties.
        derived = {"statistics", "target_property", "embedding_property"}
        model = cfg["model"]["config"]
        if model.get("atomic_numbers") is None:
            derived.add("atomic_numbers")
        actual = saved["model"]["config"]
        if {k: v for k, v in model.items() if k not in derived} != {
            k: v for k, v in actual.items() if k not in derived
        }:
            return False
        state = checkpoint["state_dict"]
        suffix = ".linear.output_mask"
        wrappers = {k.removesuffix(suffix) for k in state if k.endswith(suffix)}
        stages = {
            f"model.readout_fn.representation.interactions.{i}.linear_up"
            for i in range(model["num_layers"])
        }
        blocks = {}
        for name, weight in state.items():
            prefix, separator, index = name.rpartition(".weight.")
            if separator and index.isdecimal():
                if weight.ndim != 2 or weight.dtype != torch.float32:
                    return False
                blocks.setdefault(prefix, set()).add(int(index))
        if not stages <= wrappers or not all(
            f"{key}.weight" not in state
            and key in blocks
            and blocks[key] == set(range(len(blocks[key])))
            for key in wrappers
        ):
            return False
        (optimizer,) = checkpoint["optimizer_states"]
        groups = optimizer["param_groups"]
        if {g["use_muon"] for g in groups if g["params"]} != {False, True}:
            return False
        for group in groups:
            required = (
                {"momentum_buffer"} if group["use_muon"] else {"exp_avg", "exp_avg_sq"}
            )
            if any(
                not required <= optimizer["state"][i].keys() for i in group["params"]
            ):
                return False
        return True
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        RuntimeError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
    ):
        return False


def configure_environment(*, preserve_existing=False):
    """Select the FP32/TF32 schedule before importing EQX."""
    if preserve_existing:
        for name, value in EXECUTION.items():
            if (
                name != "PYTORCH_CUDA_ALLOC_CONF"
                or "PYTORCH_ALLOC_CONF" not in os.environ
            ):
                os.environ.setdefault(name, value)
    else:
        os.environ.update(EXECUTION)
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    os.environ.setdefault("WANDB_MODE", "offline")


def launch(cfg):
    """Train with the composed Hydra configuration."""
    import torch
    from omegaconf import open_dict

    from equivariantx_foundation.models.tace.tace_oam_xl_v2.precision import (
        validate_environment,
    )
    from tace.lightning.oamv2 import OAMLightningModule
    from tace.scripts.train import train_from_config
    from tace.utils.precision import resolve_precision

    if cfg.trainer.get("accumulate_grad_batches", 1) != 1:
        raise ValueError("Captured training requires accumulate_grad_batches=1")
    strategy = cfg.trainer.get("strategy")
    if (
        isinstance(strategy, str)
        or not strategy
        or strategy.get("_target_") != "tace.utils.strategy.SimpleDDPStrategy"
    ):
        raise ValueError("Captured training requires SimpleDDPStrategy")
    if cfg.trainer.get("inference_mode", False):
        raise ValueError("Force validation requires inference_mode=false")
    with open_dict(cfg):
        policy = resolve_precision(cfg, training=True)
        cfg.trainer.accelerator = "cuda"
        cfg.trainer.devices = "auto"
    if (
        policy.model != "float32"
        or policy.autocast is not None
        or policy.matmul != "tf32"
    ):
        raise ValueError("This training schedule uses FP32 model tensors and TF32 GEMM")
    configured = cfg.misc.get("env") or {}
    validate_environment({**os.environ, **configured})
    for name, value in configured.items():
        if name in EXECUTION and str(value) != EXECUTION[name]:
            raise ValueError(
                f"Config overrides the selected execution schedule: {name}"
            )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    train_from_config(cfg, OAMLightningModule, fast=True)
