"""Operand precision shared by Torch and native kernels."""

import torch

SUPPORTED_DTYPES = (torch.float32, torch.float64)


def validate_module_precision(module, values):
    """Check operands and direct module state without wrapping ``forward``."""
    validate_precision((values, module._parameters, module._buffers))


def floating_tensors(values):
    """Yield floating operands from nested argument containers."""
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() or value.is_complex():
                yield value
        elif isinstance(value, dict):
            yield from floating_tensors(value.values())
        elif isinstance(value, (tuple, list)):
            yield from floating_tensors(value)


def validate_precision(values):
    """Require FP32 or FP64 storage without lower-precision autocast.

    Parameters
    ----------
    values : iterable
        Tensor operands, possibly nested. Integer indices are ignored.

    Notes
    -----
    TF32 is an FP32 matrix arithmetic policy, not a storage dtype. EQX
    respects the active PyTorch matrix policy without changing global state.

    Examples
    --------
    >>> validate_precision([torch.ones(2), torch.arange(2)])
    >>> validate_precision([torch.ones(2, dtype=torch.float64)])
    """
    for value in floating_tensors(values):
        if value.dtype not in SUPPORTED_DTYPES:
            raise TypeError("EQX operands require float32 or float64 storage.")
        device = value.device.type
        if (
            torch.is_autocast_enabled(device)
            and torch.get_autocast_dtype(device) not in SUPPORTED_DTYPES
        ):
            raise TypeError("EQX does not support lower-precision autocast.")


def validate_uniform(values):
    """Check a native kernel's uniform floating-point operand ABI."""
    tensors = tuple(floating_tensors(values))
    validate_precision(tensors)
    if tensors and any(
        value.dtype != tensors[0].dtype or value.device != tensors[0].device
        for value in tensors
    ):
        raise ValueError("EQX kernel floating operands must share a dtype and device.")
