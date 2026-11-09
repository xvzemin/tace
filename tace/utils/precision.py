################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Precision policies for data, geometry, and learned features."""

import contextlib
import logging
import os
from dataclasses import asdict, dataclass

import torch

DATA_DTYPE = torch.float64


@contextlib.contextmanager
def torch_default_dtype(dtype):
    """Use a default tensor dtype within a context, restoring it on exit.

    Parameters
    ----------
    dtype : torch.dtype
        Default floating-point dtype within the context.

    Examples
    --------
    >>> with torch_default_dtype(torch.float64):
    ...     torch.ones(1).dtype
    torch.float64
    """
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(default_dtype)


def set_precision(cfg: dict) -> None:
    """Apply and log the training precision policy."""
    policy = resolve_precision(cfg, training=True)
    policy.apply()
    log_precision(policy)


@dataclass(frozen=True)
class PrecisionPolicy:
    """Select network storage, autocast, and matrix multiplication precision.

    Parameters
    ----------
    model : {"float32", "float64"}
        Learned feature and parameter dtype.
    autocast : {None, "bfloat16", "float16"}
        Autocast dtype for supported network operations.
    matmul : {"ieee", "tf32"}
        FP32 matrix multiplication mode.

    Examples
    --------
    >>> PrecisionPolicy(model="float32").torch_dtype
    torch.float32
    """

    model: str = "float32"
    autocast: str | None = None
    matmul: str = "ieee"

    def __post_init__(self):
        if self.model not in ("float32", "float64"):
            raise ValueError("dtype.model must be float32 or float64.")
        if self.autocast not in (None, "bfloat16", "float16"):
            raise ValueError("dtype.autocast must be null, bfloat16, or float16.")
        if self.matmul not in ("ieee", "tf32"):
            raise ValueError(
                "dtype.matmul must be ieee or tf32; emulation is not implemented."
            )
        if self.model == "float64" and self.autocast is not None:
            raise ValueError("Autocast requires dtype.model=float32.")
        if self.autocast is not None and self.matmul != "ieee":
            raise ValueError("Select autocast or TF32, not both.")

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.model)

    def apply(self) -> None:
        enabled = self.matmul == "tf32" and self.model == "float32"
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled


def resolve_precision(cfg: dict, *, training: bool = False) -> PrecisionPolicy:
    """Resolve configuration and environment settings without changing data dtype."""
    options = dict(cfg.get("dtype") or {})
    unknown = options.keys() - {"model", "autocast", "matmul"}
    if unknown:
        raise ValueError(f"Unknown dtype options: {sorted(unknown)}")
    trainer_precision = cfg.get("trainer", {}).get("precision", "32-true")
    legacy = str(trainer_precision)
    options.setdefault("model", "float64" if legacy in ("64", "64-true") else "float32")
    options.setdefault(
        "autocast", {"bf16-mixed": "bfloat16", "16-mixed": "float16"}.get(legacy)
    )
    env_mode = os.environ.get("TACE_MATMUL_PRECISION")
    env_tf32 = os.environ.get("TACE_USE_TF32")
    if env_tf32 is not None:
        if env_tf32 not in ("0", "1"):
            raise ValueError("TACE_USE_TF32 must be 0 or 1.")
        tf32_mode = "tf32" if env_tf32 == "1" else "ieee"
        if env_mode is not None and env_mode != tf32_mode:
            raise ValueError("TACE_MATMUL_PRECISION conflicts with TACE_USE_TF32.")
        env_mode = tf32_mode
    if env_mode is not None:
        if options.get("matmul") is not None and options["matmul"] != env_mode:
            raise ValueError(
                "dtype.matmul conflicts with the precision environment setting."
            )
        options["matmul"] = env_mode
    if options.get("matmul") is None:
        options["matmul"] = (
            "tf32" if training and options["autocast"] is None else "ieee"
        )
    policy = PrecisionPolicy(**options)
    cfg["dtype"] = asdict(policy)
    return policy


def apply_model_precision(model, fn, recurse=True):
    """Apply a tensor conversion without rounding protected FP64 tensors."""
    parameter = next(model.parameters(), None)
    current_dtype = getattr(model, "model_dtype", torch.get_default_dtype())
    probe = fn(
        torch.empty(
            0,
            dtype=current_dtype,
            device=parameter.device if parameter is not None else None,
        )
    )
    if probe.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "Model storage supports float32 and float64; use autocast for lower precision."
        )
    if probe.device.type == "mps":
        raise ValueError(
            "TACE requires FP64 geometry; use CPU on macOS because MPS does not support float64."
        )
    preserved = {}
    parameters = {}
    parameter_values = {}
    for module in model.modules():
        geometry = getattr(module, "geometry_precision", False)
        protected = module.modules() if geometry else (module,)
        for child in protected:
            names = (
                child._buffers if geometry else getattr(child, "float64_buffers", ())
            )
            for name in names:
                value = child._buffers.get(name)
                if value is not None and value.is_floating_point():
                    preserved[(child, name)] = value
            if geometry:
                for name, value in child.named_parameters(recurse=False):
                    parameters[(child, name)] = id(value)
                    parameter_values.setdefault(
                        id(value),
                        (
                            value.detach(),
                            None if value.grad is None else value.grad.detach(),
                        ),
                    )
    torch.nn.Module._apply(model, fn, recurse=recurse)
    converted = {}
    for (module, name), value in preserved.items():
        device = module._buffers[name].device
        key = (id(value), device)
        if key not in converted:
            converted[key] = value.to(device=device, dtype=DATA_DTYPE)
        module._buffers[name] = converted[key]
    for (module, name), key in parameters.items():
        parameter = module._parameters[name]
        value, gradient = parameter_values[key]
        parameter.data = value.to(device=parameter.device, dtype=DATA_DTYPE)
        if gradient is not None:
            parameter.grad = gradient.to(device=parameter.device, dtype=DATA_DTYPE)
    for module in model.modules():
        if hasattr(module, "model_dtype"):
            module.model_dtype = probe.dtype
    model.model_dtype = probe.dtype
    if isinstance(getattr(model, "readout_fn", None), torch.nn.Module):
        model.readout_fn.model_dtype = probe.dtype
    return model


def configure_model_precision(model, dtype: torch.dtype, device=None):
    """Convert learned tensors while preserving FP64 geometry and physical constants."""
    from tace.models.adapter import TensorModel

    if dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "Model storage supports float32 and float64; use autocast for lower precision."
        )
    if isinstance(model, TensorModel):
        return model.to(device=device, dtype=dtype)
    return apply_model_precision(
        model,
        lambda value: value.to(
            device=device,
            dtype=dtype if value.is_floating_point() or value.is_complex() else None,
        ),
    )


def log_precision(policy: PrecisionPolicy) -> None:
    logging.info(
        "Precision: data/geometry/physical reductions=float64, model=%s, autocast=%s, matmul=%s",
        policy.model,
        policy.autocast,
        policy.matmul,
    )
