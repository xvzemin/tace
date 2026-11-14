"""Local IEEE matrix arithmetic for functional transformations."""

import torch

from ...precision import validate_uniform


def reference(left, right):
    """Evaluate the ordinary Torch matrix product under the caller's policy."""
    validate_uniform((left, right))
    return left @ right


def matmul(left, right):
    """Multiply without changing global TF32 policy or reducing operand precision."""
    validate_uniform((left, right))
    if left.ndim < 2 or right.ndim < 2 or left.shape[-1] != right.shape[-2]:
        raise ValueError("Matrix dimensions do not match.")
    if left.dtype == torch.float64:
        return left @ right
    if left.shape[-1] == 0:
        return left.sum(-1, keepdim=True) * right.sum(-2, keepdim=True)
    result = None
    for start in range(0, left.shape[-1], 16):
        value = (
            left[..., start : start + 16, None]
            * right[..., None, start : start + 16, :]
        ).sum(-2)
        result = value if result is None else result + value
    return result
