"""Native angular/channel products and differentiable CUDA adjoints."""


def evaluate(module, x, y, weight):
    """Execute native product kernels for the validated instruction layout."""
    from equivariantx.ops.common.channel_product import tensor_product as cuda_product

    output = (
        cuda_product(
            module.kernel_metadata, x, y, weight, module.irreps_out.dim, backend="cuda"
        )
        if module.kernel_metadata is not None
        else module.contraction(x, y, weight, module.path_scales.to(x))
    )
    return module.projection(output)
