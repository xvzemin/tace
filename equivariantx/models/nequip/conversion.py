"""Convert NequIP interactions without changing their graph interface."""

from collections import OrderedDict

import torch
from e3nn import o3

from ...utils.module import default_dtype
from ..convolution import (
    Convolution,
    RadialFeatures,
    convert_interactions,
    harmonic_convention,
    spatial_backend,
)


class TensorProductScatter(Convolution):
    """Apply an EQX convolution with NequIP's source and destination arguments."""

    _nequip_custom_ops_libs = ("equivariantx.conv",)

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        return super().forward(x, edge_attr, edge_weight, (edge_src, edge_dst))


def convert_nequip_to_eqx(
    model, *, implementation="o3", inplace=False, backend=None, stream_radial=False
):
    """Replace NequIP interactions with streamed radial convolutions.

    Parameters
    ----------
    model : torch.nn.Module
        Uncompiled NequIP model with TensorProductScatter interactions.
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
        Converted model with the original graph interface.

    Examples
    --------
    .. code-block:: python

        from equivariantx.models.nequip import convert_nequip_to_eqx

        model = convert_nequip_to_eqx(model, implementation="o3")
    """
    backend = spatial_backend(implementation, backend)
    normalization, normalize = (
        harmonic_convention(model) if implementation == "o2" else ("component", True)
    )

    def factory(module):
        if type(module).__name__ != "InteractionBlock" or not type(
            module
        ).__module__.split(">.")[-1].startswith("nequip."):
            return None

        if isinstance(module.tp_scatter, TensorProductScatter):
            module.tp_scatter.check_options(
                module.edge_mlp, implementation, backend, stream_radial
            )
            return module
        tp = getattr(module.tp_scatter, "tp", None)
        if (
            tp is None
            or type(tp).__name__ != "TensorProduct"
            or not type(tp).__module__.split(">.")[-1].startswith("e3nn.")
        ):
            raise NotImplementedError(
                "Load NequIP without another tensor-product backend first."
            )
        net = module.edge_mlp.mlp
        if not isinstance(net, torch.nn.Sequential) or not len(net):
            raise NotImplementedError("Expected a sequential NequIP radial MLP.")
        projection = net[-1]
        if not hasattr(projection, "alpha") or not hasattr(projection, "weight"):
            raise NotImplementedError("Expected a ScalarLinearLayer radial projection.")
        parameter = next(module.parameters())
        with default_dtype(parameter.dtype):
            # Packaged models carry their own e3nn class identities.
            if not isinstance(tp, o3.TensorProduct):
                tp = o3.TensorProduct(
                    str(tp.irreps_in1),
                    str(tp.irreps_in2),
                    str(tp.irreps_out),
                    [
                        (
                            i.i_in1,
                            i.i_in2,
                            i.i_out,
                            i.connection_mode,
                            i.has_weight,
                            i.path_weight**2,
                        )
                        for i in tp.instructions
                    ],
                    irrep_normalization="none",
                    path_normalization="none",
                    internal_weights=False,
                    shared_weights=False,
                    compile_left_right=False,
                )
            convolution = (
                TensorProductScatter(
                    tp,
                    projection,
                    implementation=implementation,
                    backend=backend,
                    normalization=normalization,
                    normalize=normalize,
                    radial_mode="full" if stream_radial else "last",
                )
                .to(parameter.device)
                .train(module.tp_scatter.training)
            )
        radial = RadialFeatures(
            OrderedDict(list(net.named_children())[:-1]),
            projection.bias is not None,
            stream_radial=stream_radial,
        ).train(module.edge_mlp.training)
        module.tp_scatter = convolution
        module.edge_mlp = radial
        return module

    return convert_interactions(
        model, factory, inplace=inplace, name="NequIP interactions"
    )
