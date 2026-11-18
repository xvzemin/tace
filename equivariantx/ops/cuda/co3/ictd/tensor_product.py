"""Native angular/channel products and differentiable CUDA adjoints."""

from ....torch.co3.ictd import tensor_product as torch_kernel


def symmetric_product(module, x, y, l1, l2, basis):
    from .....utils.autograd import forward_ad_active
    from ....common.cuda.cartesian import pack_product

    if not x.is_cuda or forward_ad_active():
        return basis.pack(module.couple(x, y, l1, l2, basis.rank))
    return pack_product(x, y, l1, l2, basis)


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
