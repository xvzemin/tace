"""Native angular/channel products and differentiable CUDA adjoints."""


def evaluate(module, x, y, weight, supplied_weight):
    """Execute native product kernels for the validated instruction layout."""
    if not all((value.is_cuda for value in (x, y, weight))):
        raise ValueError("backend='cuda' requires CUDA inputs and weights.")
    return module.contraction(x, y, weight)
