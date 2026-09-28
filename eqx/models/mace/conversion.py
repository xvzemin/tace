"""Convert MACE interactions to streamed tensor-product convolutions."""

from collections import OrderedDict

import torch
from e3nn import o3

from eqx.kernels import wigner_D
from eqx.models.convolution import Convolution, RadialFeatures, copy_model
from eqx.o2 import WignerD

__all__ = ["convert_mace_to_eqx"]


class FrameAttributes(torch.nn.Module):
    """Keep the original harmonics and append one shared packed Wigner matrix.

    The harmonic prefix remains available to other model components. Only
    the replaced convolution reads the appended entries.
    """

    def __init__(self, harmonics, lmax, backend):
        super().__init__()
        self.harmonics = harmonics
        self.backend = backend
        self.irreps_out = harmonics.irreps_out
        self.normalization = harmonics.normalization
        self.normalize = harmonics.normalize
        self._lmax = harmonics._lmax
        self.wigner = WignerD(lmax, lmax)
        self.packed_dim = sum((2 * ell + 1) ** 2 for ell in range(lmax + 1))
        self.register_buffer(
            "degrees",
            torch.tensor([ir.l for _, ir in self.irreps_out]),
            persistent=False,
        )

    def forward(self, vectors):
        values = [
            self.harmonics(vectors),
            wigner_D(self.wigner, vectors, backend=self.backend),
        ]
        if not self.normalize:
            values.append(vectors.norm(dim=-1, keepdim=True).pow(self.degrees))
        return torch.cat(values, dim=-1)


def convert_mace_to_eqx(
    model, *, implementation="o3", enable_cueq=False, inplace=False, backend="cuda"
):
    """Replace MACE spatial convolutions with streamed EQX operations.

    Parameters
    ----------
    model : torch.nn.Module
        MACE model with supported RealAgnostic interactions. Set its device
        and floating-point precision before conversion.
    implementation : {"o3", "o2"}, optional
        Direct or aligned convolution. Defaults to "o3".
    enable_cueq : bool, optional
        Also convert the remaining operators using MACE's cuEquivariance
        converter. Defaults to False. Existing cuEquivariance operators
        are retained without requiring another conversion. If enabled, apply
        selective parameter freezing after MACE's conversion rebuilds the model.
    inplace : bool, optional
        Modify the supplied model. Defaults to False, returning an independent
        copy. Enabling cuEquivariance may construct a new model even when
        this is True.
    backend : {"cuda", "torch"}, optional
        Convolution backend. Defaults to CUDA kernels on CUDA tensors and
        PyTorch operations on CPU. Select "torch" for a reference on either
        device.

    Returns
    -------
    torch.nn.Module
        Model with the original forward interface, supporting energy, force
        and stress training and ``MACECalculator(models=...)``. Learned
        parameters and all tensor-product paths are retained.

    Notes
    -----
    Supports RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock,
    RealAgnosticDensityInteractionBlock, RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticAttResidualInteractionBlock and
    RealAgnosticResidualNonLinearInteractionBlock. Edge harmonics must have
    natural spatial parity for the aligned implementation. Magnetic interactions
    are not supported.

    Load weights before conversion, and construct the optimizer afterwards.
    Force training requires ``training=True`` in the model's forward call.
    Converted state dictionaries require an identically converted architecture.
    Use a separate model instance for ASE, which disables parameter gradients.
    """
    from mace.modules.irreps_tools import tp_out_irreps_with_instructions
    from mace.modules.wrapper_ops import get_layout
    from mace.tools.torch_tools import default_dtype

    if implementation not in ("o3", "o2"):
        raise ValueError("implementation must be 'o3' or 'o2'.")
    if backend not in ("cuda", "torch"):
        raise ValueError("backend must be 'cuda' or 'torch'.")
    if not hasattr(model, "spherical_harmonics") or not hasattr(model, "interactions"):
        raise TypeError("Expected a MACE model with spatial interactions.")
    converted = [isinstance(layer.conv_tp, Convolution) for layer in model.interactions]
    if any(converted):
        if not all(converted) or any(
            layer.conv_tp.implementation != implementation
            or layer.conv_tp.backend != backend
            for layer in model.interactions
        ):
            raise ValueError(
                "Convert the original model to select another implementation."
            )
        return model if inplace else copy_model(model)
    supported = {
        "RealAgnosticInteractionBlock",
        "RealAgnosticResidualInteractionBlock",
        "RealAgnosticDensityInteractionBlock",
        "RealAgnosticDensityResidualInteractionBlock",
        "RealAgnosticAttResidualInteractionBlock",
        "RealAgnosticResidualNonLinearInteractionBlock",
    }
    if not len(model.interactions):
        raise ValueError("The model has no interactions to convert.")
    for layer in model.interactions:
        if type(layer).__name__ not in supported:
            raise NotImplementedError(
                f"Unsupported interaction: {type(layer).__name__}"
            )
    parameter = next(model.parameters())
    if parameter.dtype not in (torch.float32, torch.float64):
        raise TypeError("EQX convolutions require float32 or float64 model weights.")
    with default_dtype(parameter.dtype):
        config = getattr(model.interactions[0], "cueq_config", None)
        if enable_cueq and not (config is not None and config.enabled):
            from mace.cli.convert_e3nn_cueq import run

            training = model.training
            model = run(model, device=str(parameter.device), return_model=True)
            model.train(training)
        elif not inplace:
            model = copy_model(model)

        products, radials = [], []
        for layer in model.interactions:
            if isinstance(layer.conv_tp, o3.TensorProduct):
                tp = layer.conv_tp
            else:
                irreps_out, instructions = tp_out_irreps_with_instructions(
                    layer.edge_irreps, layer.edge_attrs_irreps, layer.target_irreps
                )
                tp = o3.TensorProduct(
                    layer.edge_irreps,
                    layer.edge_attrs_irreps,
                    irreps_out,
                    instructions,
                    internal_weights=False,
                    shared_weights=False,
                    compile_left_right=False,
                )
            if tp.weight_numel != layer.conv_tp.weight_numel:
                raise ValueError(
                    "MACE and EQX tensor-product path counts do not match."
                )
            radial = layer.conv_tp_weights
            net = radial.net if hasattr(radial, "net") else radial
            if not isinstance(net, torch.nn.Sequential) or not len(net):
                raise NotImplementedError(
                    "Expected a nonempty sequential MACE radial MLP."
                )
            projection = net[-1]
            affine = isinstance(projection, torch.nn.Linear)
            if not affine and (
                not hasattr(projection, "h_in") or projection.act is not None
            ):
                raise NotImplementedError("The final radial projection must be linear.")
            products.append(tp)
            radials.append((radial, net, projection, affine))

        attributes = model.spherical_harmonics
        if implementation == "o2":
            lmax = max(max(tp.irreps_in1.lmax, tp.irreps_out.lmax) for tp in products)
            attributes = FrameAttributes(attributes, lmax, backend).train(
                attributes.training
            )
        convolutions, radial_features = [], []
        for layer, tp, (radial, net, projection, affine) in zip(
            model.interactions, products, radials
        ):
            convolutions.append(
                Convolution(
                    tp,
                    projection,
                    layout=get_layout(getattr(layer, "cueq_config", None)),
                    implementation=implementation,
                    backend=backend,
                    normalization=attributes.normalization,
                    normalize=attributes.normalize,
                    packed_dim=getattr(attributes, "packed_dim", 0),
                ).train(layer.conv_tp.training)
            )
            radial_features.append(
                RadialFeatures(
                    OrderedDict(list(net.named_children())[:-1]),
                    affine and projection.bias is not None,
                    radial.hs,
                ).train(radial.training)
            )
        for layer, convolution, radial in zip(
            model.interactions, convolutions, radial_features
        ):
            layer.conv_tp = convolution
            layer.conv_tp_weights = radial
            layer.conv_fusion = True
        model.spherical_harmonics = attributes
        return model.to(device=parameter.device, dtype=parameter.dtype)
