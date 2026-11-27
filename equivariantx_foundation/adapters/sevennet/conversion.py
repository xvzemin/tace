"""Convert SevenNet graph convolutions."""

from collections import OrderedDict

import torch
from e3nn import o3

from equivariantx.utils.module import default_dtype
from equivariantx_foundation.registry import require_cuda

from ..convolution import (
    Convolution,
    RadialFeatures,
    convert_interactions,
    harmonic_convention,
    spatial_backend,
)


class IrrepsConvolution(torch.nn.Module):
    """Preserve SevenNet graph fields and neighbor normalization."""

    def __init__(
        self, module, implementation, backend, normalization, normalize, stream_radial
    ):
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
        self.weight_nn = RadialFeatures(
            OrderedDict(list(net.named_children())[:-1]),
            hs=net.hs,
            stream_radial=stream_radial,
        )
        self.convolution = Convolution(
            module.convolution,
            projection,
            implementation=implementation,
            backend=backend,
            normalization=normalization,
            normalize=normalize,
            radial_mode="full" if stream_radial else "last",
        )

    def forward(self, data):
        radial = self.weight_nn(data[self.key_weight_input])
        target, source = data[self.key_edge_idx]
        data[self.key_x] = (
            self.convolution(
                data[self.key_x],
                data[self.key_filter],
                radial,
                (source, target),
            )
            / self.denominator
        )
        return data


def convert_sevennet_to_eqx(
    model, *, implementation="o3", inplace=False, backend=None, stream_radial=False
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
    backend : {None, "cuda", "torch"}, optional
        CUDA or PyTorch operators on the model's CUDA device.
    stream_radial : bool, optional
        Fuse the complete radial MLP if True, or its final projection if False.

    Returns
    -------
    torch.nn.Module
        Converted model, accepted by ``SevenNetCalculator(file_type="model_instance")``.

    Notes
    -----
    Select the fidelity through the calculator's ``modal`` argument. SevenNet's
    embedding produces float32 features even if the model is cast to float64.

    Examples
    --------
    .. code-block:: python

        from equivariantx_foundation.adapters.sevennet import convert_sevennet_to_eqx

        model = convert_sevennet_to_eqx(model, implementation="o3")
    """
    require_cuda(model)
    backend = spatial_backend(implementation, backend)
    normalization, normalize = (
        harmonic_convention(model) if implementation == "o2" else ("component", True)
    )

    def factory(module):
        if isinstance(module, IrrepsConvolution):
            module.convolution.check_options(
                module.weight_nn, implementation, backend, stream_radial
            )
            return module
        if type(module).__name__ != "IrrepsConvolution" or not type(
            module
        ).__module__.startswith("sevenn."):
            return None

        parameter = next(module.parameters())
        with default_dtype(parameter.dtype):
            return (
                IrrepsConvolution(
                    module,
                    implementation,
                    backend,
                    normalization,
                    normalize,
                    stream_radial,
                )
                .to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                .train(module.training)
            )

    return convert_interactions(
        model, factory, inplace=inplace, name="SevenNet convolutions"
    )
