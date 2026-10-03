"""Convert NequIP interactions without changing their graph interface."""

from collections import OrderedDict

import torch
from e3nn import o3

from eqx.models.convolution import Convolution, RadialFeatures
from eqx.utils import convert_modules, default_dtype


class TensorProductScatter(Convolution):
    """Apply an EQX convolution with NequIP's source and destination arguments."""

    _nequip_custom_ops_libs = ("eqx.conv",)

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        return super().forward(
            x, edge_attr, edge_weight, torch.stack((edge_src, edge_dst))
        )


def convert_nequip_to_eqx(model, *, implementation="o3", inplace=False, backend="cuda"):
    """Fuse NequIP interaction contractions and radial networks.

    Parameters
    ----------
    model : torch.nn.Module
        Uncompiled NequIP model with TensorProductScatter interactions.
    implementation : {"o3", "o2"}, optional
        Direct or aligned tensor product. Defaults to "o3".
    inplace : bool, optional
        Modify the model instead of returning a copy.
    backend : {"cuda", "torch"}, optional
        Convolution backend. Defaults to CUDA, with PyTorch on CPU.

    Returns
    -------
    torch.nn.Module
        Model with the same graph interface and differentiable parameters.
        Convert before constructing the optimizer. Load original checkpoints
        before conversion; converted state dictionaries use adapted names.
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
        if m not in converted_modules
        and type(m).__name__ == "SphericalHarmonics"
        and type(m).__module__.split(">.")[-1].startswith("e3nn.")
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
        if type(module).__name__ != "InteractionBlock" or not type(
            module
        ).__module__.split(">.")[-1].startswith("nequip."):
            return None
        count += 1
        if isinstance(module.tp_scatter, TensorProductScatter):
            if (
                module.tp_scatter.implementation != implementation
                or module.tp_scatter.backend != backend
            ):
                raise ValueError(
                    "Convert the original model to select another implementation."
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
                    radial_network=torch.nn.Sequential(
                        OrderedDict(list(net.named_children())[:-1])
                    ),
                )
                .to(parameter.device)
                .train(module.tp_scatter.training)
            )
        radial = RadialFeatures().train(module.edge_mlp.training)
        module.tp_scatter = convolution
        module.edge_mlp = radial
        return module

    converted = convert_modules(model, factory, inplace=inplace)
    if not count:
        raise ValueError("No supported NequIP interactions were found.")
    return converted
