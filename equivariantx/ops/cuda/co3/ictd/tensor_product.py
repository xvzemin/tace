"""Native angular/channel products and differentiable CUDA adjoints."""

from ....torch.co3.ictd import tensor_product as torch_kernel


def evaluate(module, x, y, weight):
    """Execute native product kernels for the validated instruction layout."""
    if module.kernel_metadata is not None:
        from equivariantx.ops.common.channel_product import tensor_product

        return module.projection(
            tensor_product(
                module.kernel_metadata, x, y, weight, module.output_dim, backend="cuda"
            )
        )
    return torch_kernel.evaluate(module, x, y, weight)
