"""Convert Prophet spatial tensor-product convolutions."""

from collections import OrderedDict

import torch
from e3nn import o3

from ...utils.module import default_dtype
from ..convolution import (
    Convolution,
    RadialFeatures,
    convert_interactions,
    spatial_backend,
)


class TensorProductConvolution(Convolution):
    """Use Prophet's convolution argument order."""

    def forward(self, features, sh, radial, receivers, senders):
        return super().forward(features, sh, radial, (senders, receivers))


def convert_prophet_to_eqx(
    model, *, implementation="o3", inplace=False, backend=None, stream_radial=False
):
    """Replace Prophet spatial convolutions, retaining cutoff branches.

    Parameters
    ----------
    model : torch.nn.Module
        Uncompiled spatial Prophet model loaded with use_kernel=False.
    implementation : {"o3", "o2"}, optional
        Direct or aligned tensor product. Defaults to "o3".
    inplace : bool, optional
        Modify the model instead of returning a copy.
    backend : {None, "cuda", "torch"}, optional
        Execution backend. Defaults to CUDA when available, otherwise PyTorch.
    stream_radial : bool, optional
        Fuse the complete radial MLP if True, or its final projection if False.

    Returns
    -------
    torch.nn.Module
        Converted spatial model with the original forward interface.

    Notes
    -----
    Prophet-Spin is not supported. KairosCalculator produces float32 graph
    inputs, so its model must also use float32.

    Examples
    --------
    .. code-block:: python

        from equivariantx.models.prophet import convert_prophet_to_eqx

        model = convert_prophet_to_eqx(model, implementation="o3")
    """
    backend = spatial_backend(implementation, backend)

    def factory(module):
        if (
            type(module).__name__ != "TPConvolution"
            or type(module).__module__ != "prophet.model"
        ):
            return None

        if isinstance(getattr(module, "tp_conv", None), TensorProductConvolution):
            module.tp_conv.check_options(
                module.radial_mlp, implementation, backend, stream_radial
            )
            return module
        if module.kernel or not isinstance(module.tp, o3.TensorProduct):
            raise NotImplementedError(
                "Load Prophet with use_kernel=False before conversion."
            )
        net = module.radial_mlp
        projection = net.layers[-1]
        layers = OrderedDict()
        for index, layer in enumerate(net.layers[:-1]):
            layers[f"linear_{index}"] = layer
            layers[f"activation_{index}"] = net.activation
        radial = RadialFeatures(
            layers, projection.bias is not None, stream_radial=stream_radial
        ).train(net.training)
        parameter = next(module.parameters())
        with default_dtype(parameter.dtype):
            convolution = (
                TensorProductConvolution(
                    module.tp,
                    projection,
                    implementation=implementation,
                    backend=backend,
                    radial_mode="full" if stream_radial else "last",
                )
                .to(parameter.device)
                .train(module.tp.training)
            )
        # Combine path sorting with the node output-layout permutation.
        index = convolution.output_index
        if not index.numel():
            index = torch.arange(module.tp.irreps_out.dim, device=parameter.device)
        convolution.set_permutation(
            "output_index", torch.cat([index[s] for s in module.sort.slices_sorted])
        )
        convolution.irreps_out = module.sort.irreps_sorted
        module.tp_conv = convolution
        module.radial_mlp = radial
        module.sort = torch.nn.Identity()
        del module.tp
        module.kernel = True
        return module

    return convert_interactions(
        model, factory, inplace=inplace, name="spatial Prophet convolutions"
    )
