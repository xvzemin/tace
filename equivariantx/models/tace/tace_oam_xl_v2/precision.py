"""Model precision contract; optimizer implementations are outside this boundary."""

import os

import torch
from torch.utils._pytree import tree_leaves


def validate_environment(environment=None):
    """Reject enabled legacy storage/conversion options before compiling.

    Zero/off values remain accepted for compatibility with old launch files.
    No model-side reduced storage implementation is installed in this package.
    """
    environment = os.environ if environment is None else environment
    numeric = (
        "EQX_XL_RADIAL_HALF_SCALE", "EQX_XL_RADIAL_HALF_FIRST_SCALE",
        "EQX_XL_NODE_HALF_SCALE", "EQX_XL_MATRIX_FIXED_SCALE",
        "EQX_XL_MATRIX_EPILOGUE",
    )
    for name in numeric:
        if float(environment.get(name, "0")) != 0:
            raise ValueError(f"{name} is disabled: model arithmetic requires FP32/FP64/TF32")
    if environment.get("EQX_XL_MATRIX_PRECISION", "off").lower() not in ("off", "0", "none"):
        raise ValueError("EQX_XL_MATRIX_PRECISION is disabled by the model precision policy")
    if environment.get("EQX_XL_TABLE_PRECISION", "tf32x3") not in ("ieee", "tf32", "tf32x3"):
        raise ValueError("Unsupported reduced table precision")


def require_tensors(values):
    """Allow FP32/FP64 floating tensors and ordinary integer graph metadata."""
    for value in tree_leaves(values):
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError(f"Model tensor {value.dtype} is forbidden; use FP32 or FP64")


def require_unscaled(value, name):
    """Keep zero-valued legacy arguments without retaining scaled storage."""
    if value != 0:
        raise ValueError(f"{name} must be zero; reduced model storage was removed")


def validate_call(data):
    """Check the model boundary without inspecting values or synchronizing CUDA."""
    if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
        raise ValueError("Model autocast is disabled; use FP32/FP64 with optional TF32 GEMM")
    require_tensors(dict(data.items()))


def validate_graph(graph):
    """Reject reduced intermediates in the explicit prediction/gradient graph."""
    for node in graph.nodes:
        try:
            require_tensors(node.meta.get("val", node.meta.get("example_value")))
            for value in tree_leaves((node.args, node.kwargs)):
                if isinstance(value, torch.dtype) and value.is_floating_point:
                    if value not in (torch.float32, torch.float64):
                        raise TypeError(f"Forbidden cast/allocation to {value}")
        except TypeError as error:
            raise TypeError(f"Precision violation at {node.name} ({node.target}): {error}") from error
