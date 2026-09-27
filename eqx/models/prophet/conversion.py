"""Convert Prophet spatial tensor-product convolutions."""

from collections import OrderedDict

import torch
from e3nn import o3

from eqx.models.convolution import (
    Convolution,
    RadialFeatures,
    convert_modules,
    default_dtype,
)


class TensorProductConvolution(Convolution):
    """Use Prophet's convolution argument order."""

    def forward(self, features, sh, radial, receivers, senders):
        return super().forward(features, sh, radial, torch.stack((senders, receivers)))


def convert_prophet_to_eqx(
    model, *, implementation="o3", inplace=False, backend="cuda"
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
    backend : {"cuda", "torch"}, optional
        Convolution backend. Defaults to CUDA, with PyTorch on CPU.

    Returns
    -------
    torch.nn.Module
        Model with the original forward interface. Convert before constructing
        the optimizer, or assign the returned model to KairosCalculator.model.
        Prophet-Spin is not included in this conversion.
    """
    if implementation not in ("o3", "o2") or backend not in ("cuda", "torch"):
        raise ValueError(
            "Expected implementation 'o3'/'o2' and backend 'cuda'/'torch'."
        )
    count = 0

    def factory(module):
        nonlocal count
        if (
            type(module).__name__ != "TPConvolution"
            or type(module).__module__ != "prophet.model"
        ):
            return None
        count += 1
        if isinstance(getattr(module, "tp_conv", None), TensorProductConvolution):
            if (
                module.tp_conv.implementation != implementation
                or module.tp_conv.backend != backend
            ):
                raise ValueError(
                    "Convert the original model to select another implementation."
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
        radial = RadialFeatures(layers, projection.bias is not None).train(net.training)
        parameter = next(module.parameters())
        with default_dtype(parameter.dtype):
            convolution = (
                TensorProductConvolution(
                    module.tp,
                    projection,
                    implementation=implementation,
                    backend=backend,
                )
                .to(parameter.device)
                .train(module.tp.training)
            )
        module.tp_conv = convolution
        module.radial_mlp = radial
        del module.tp
        module.kernel = True
        return module

    converted = convert_modules(model, factory, inplace=inplace)
    if not count:
        raise ValueError("No supported spatial Prophet convolutions were found.")
    return converted
