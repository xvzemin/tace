"""Native angular/channel products and differentiable CUDA adjoints."""


def evaluate(module, input1, input2, weight):
    """Execute native product kernels for the validated instruction layout."""
    from equivariantx.ops.common.channel_product import tensor_product as cuda_product

    if module.kernel_metadata is None:
        return module.forward_paths(input1, input2, weight, native=True)
    return cuda_product(
        module.kernel_metadata,
        input1,
        input2,
        weight,
        module.irreps_out.dim,
        backend="cuda",
    )
