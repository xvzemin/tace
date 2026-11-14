"""Native Cartesian frame maps sharing the rectangular tensor-axis kernel."""

from ...common.cuda.tensor_axis import evaluate, inline, transform
from ...common.torch.tensor_axis import primal
from ...torch.co2 import local_frame as torch_kernel


def to_local(module, features, rotation):
    """Contract every retained planar order with native tensor axes."""
    return torch_kernel.to_local(
        module, features, rotation, primal if inline() else evaluate
    )


def to_global(module, features, rotation):
    """Reconstruct global Cartesian features with native tensor axes."""
    return torch_kernel.to_global(
        module, features, rotation, primal if inline() else evaluate
    )


def rotate(module, features, rotation):
    """Rotate every tensor index with the shared native rectangular map."""
    return torch_kernel.rotate(
        module, features, rotation, primal if inline() else evaluate
    )


def transform_tensor(tensor, matrix, rank):
    """Apply native rectangular maps to a flattened Cartesian tensor."""
    return transform(tensor, matrix, rank)
