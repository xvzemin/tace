"""Operand precision shared by Torch and native kernels."""

import torch

SUPPORTED_DTYPES = (torch.float32, torch.float64)
CPU_MATMUL_ERROR = "EQX FP32 operands do not support MKLDNN BF16 matrix arithmetic."


def cpu_matmul_allowed():
    """Read the effective CPU policy without changing PyTorch state."""
    getter = getattr(torch._C, "_get_fp32_precision_getter", None)
    return (
        getter is None
        or not torch.backends.mkldnn.enabled
        or getter("mkldnn", "matmul") != "bf16"
    )


def guard_cpu_matmul(context):
    """Guard mutable CPU matrix policy during Dynamo capture."""
    from torch._dynamo.exc import Unsupported
    from torch._dynamo.guards import GuardBuilder, install_guard
    from torch._dynamo.source import CallFunctionNoArgsSource

    value = context.get_local("policy")
    variable = value._i_will_not_complain_if_bc_breaks_VariableTracker()
    source = CallFunctionNoArgsSource(variable.source)
    install_guard(source.make_guard(GuardBuilder.EQUALS_MATCH))
    if not value.as_python_constant()():
        raise Unsupported(CPU_MATMUL_ERROR)


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
    """Require FP32 or FP64 storage and supported matrix arithmetic.

    Parameters
    ----------
    values : iterable
        Tensor operands, possibly nested. Integer indices are ignored.

    Notes
    -----
    TF32 is an FP32 matrix arithmetic policy, not a storage dtype. EQX
    rejects CPU FP32 operands when enabled MKLDNN uses BF16 matrix arithmetic,
    without changing global state. ``torch.compile`` guards policy changes.
    Standalone AOTI callers must preserve an allowed policy after capture;
    its packages do not retain Python policy guards.

    Examples
    --------
    >>> validate_precision([torch.ones(2), torch.arange(2)])
    >>> validate_precision([torch.ones(2, dtype=torch.float64)])
    """
    cpu_float32 = False
    for value in floating_tensors(values):
        if value.dtype not in SUPPORTED_DTYPES:
            raise TypeError("EQX operands require float32 or float64 storage.")
        device = value.device.type
        if (
            torch.is_autocast_enabled(device)
            and torch.get_autocast_dtype(device) not in SUPPORTED_DTYPES
        ):
            raise TypeError("EQX does not support lower-precision autocast.")
        cpu_float32 = cpu_float32 or (device == "cpu" and value.dtype == torch.float32)
    if cpu_float32 and hasattr(torch._C, "_get_fp32_precision_getter"):
        if torch.compiler.is_compiling():
            from torch._dynamo.comptime import comptime

            policy = cpu_matmul_allowed
            comptime(guard_cpu_matmul)
        elif not cpu_matmul_allowed():
            raise TypeError(CPU_MATMUL_ERROR)


def validate_uniform(values):
    """Check a native kernel's uniform floating-point operand ABI."""
    tensors = tuple(floating_tensors(values))
    validate_precision(tensors)
    if tensors and any(
        value.dtype != tensors[0].dtype or value.device != tensors[0].device
        for value in tensors
    ):
        raise ValueError("EQX kernel floating operands must share a dtype and device.")
