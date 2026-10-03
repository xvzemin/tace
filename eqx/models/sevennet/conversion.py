"""Convert SevenNet graph convolutions."""

from collections import OrderedDict

import torch
from e3nn import o3

from eqx.models.convolution import Convolution, RadialFeatures
from eqx.utils import convert_modules, default_dtype


class IrrepsConvolution(torch.nn.Module):
    """Preserve SevenNet graph fields and neighbor normalization."""

    def __init__(self, module, implementation, backend, normalization, normalize):
        super().__init__()
        if module.is_parallel:
            raise NotImplementedError("Parallel SevenNet partitions are not supported.")
        if not isinstance(module.convolution, o3.TensorProduct):
            raise NotImplementedError(
                "Load an instantiated SevenNet model without other kernels first."
            )
        for name in (
            "key_x",
            "key_filter",
            "key_weight_input",
            "key_edge_idx",
            "_comm_size",
        ):
            setattr(self, name, getattr(module, name))
        self.is_parallel = False
        self.denominator = module.denominator
        net = module.weight_nn
        if not isinstance(net, torch.nn.Sequential) or not len(net):
            raise NotImplementedError("Expected a sequential SevenNet radial MLP.")
        projection = net[-1]
        self.weight_nn = RadialFeatures(hs=net.hs)
        self.convolution = Convolution(
            module.convolution,
            projection,
            implementation=implementation,
            backend=backend,
            normalization=normalization,
            normalize=normalize,
            radial_network=torch.nn.Sequential(
                OrderedDict(list(net.named_children())[:-1])
            ),
        )

    def forward(self, data):
        radial = self.weight_nn(data[self.key_weight_input])
        edge_index = data[self.key_edge_idx].flip(0)
        data[self.key_x] = (
            self.convolution(
                data[self.key_x],
                data[self.key_filter],
                radial,
                edge_index,
            )
            / self.denominator
        )
        return data


def convert_sevennet_to_eqx(
    model, *, implementation="o3", inplace=False, backend="cuda"
):
    """Replace SevenNet convolutions with the same paths and normalization.

    Parameters
    ----------
    model : torch.nn.Module
        Instantiated, uncompiled serial SevenNet model. Modalities are retained.
    implementation : {"o3", "o2"}, optional
        Direct or aligned tensor product. Defaults to "o3".
    inplace : bool, optional
        Modify the model instead of returning a copy.
    backend : {"cuda", "torch"}, optional
        Convolution backend. Defaults to CUDA, with PyTorch on CPU.

    Returns
    -------
    torch.nn.Module
        Model for training or SevenNetCalculator with file_type="model_instance".
        Load checkpoints before conversion and construct the optimizer afterwards.

    Notes
    -----
    Select the fidelity through the calculator's ``modal`` argument. SevenNet's
    embedding produces float32 features even if the model is cast to float64.
    """
    if implementation not in ("o3", "o2") or backend not in ("cuda", "torch"):
        raise ValueError(
            "Expected implementation 'o3'/'o2' and backend 'cuda'/'torch'."
        )
    converted_modules = {
        child
        for module in model.modules()
        if isinstance(module, Convolution)
        for child in module.modules()
    }
    harmonics = [
        m
        for m in model.modules()
        if m not in converted_modules and isinstance(m, o3.SphericalHarmonics)
    ]
    options = {(m.normalization, m.normalize) for m in harmonics}
    if implementation == "o2" and len(options) > 1:
        raise NotImplementedError(
            "Multiple edge harmonic conventions require separate conversion."
        )
    normalization, normalize = next(iter(options), ("component", True))
    count = 0

    def factory(module):
        nonlocal count
        if isinstance(module, IrrepsConvolution):
            count += 1
            if (
                module.convolution.implementation != implementation
                or module.convolution.backend != backend
            ):
                raise ValueError(
                    "Convert the original model to select another implementation."
                )
            return module
        if type(module).__name__ != "IrrepsConvolution" or not type(
            module
        ).__module__.startswith("sevenn."):
            return None
        count += 1
        parameter = next(module.parameters())
        with default_dtype(parameter.dtype):
            return (
                IrrepsConvolution(
                    module, implementation, backend, normalization, normalize
                )
                .to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                .train(module.training)
            )

    converted = convert_modules(model, factory, inplace=inplace)
    if not count:
        raise ValueError("No supported SevenNet convolutions were found.")
    return converted
