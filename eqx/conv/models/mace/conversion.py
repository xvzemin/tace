"""Convert MACE interactions to streamed tensor-product convolutions."""

from collections import OrderedDict
from copy import deepcopy

import torch

from eqx.conv import O2O3TensorProductConv
from eqx.kernels import wigner_D
from eqx.o2 import O3TensorProduct, WignerD

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


class RadialFeatures(torch.nn.Sequential):
    """Evaluate the unchanged radial MLP except for its final projection."""

    def __init__(self, layers, bias, hs):
        super().__init__(layers)
        self.bias = bias
        # Preserve the full MLP widths for MACE's configuration extraction.
        self.hs = list(hs)

    def forward(self, inputs):
        features = super().forward(inputs)
        if self.bias:
            features = torch.cat(
                (features, features.new_ones((features.size(0), 1))), dim=-1
            )
        return features


class Convolution(torch.nn.Module):
    """Evaluate MACE paths with fused radial projection and target reduction."""

    def __init__(self, tensor_product, projection, layout, attributes, backend):
        super().__init__()
        self.eqx_tp = O2O3TensorProductConv(tensor_product, backend=backend)
        self.projection = projection
        self.irreps_in1 = tensor_product.irreps_in1
        self.irreps_in2 = tensor_product.irreps_in2
        self.irreps_out = tensor_product.irreps_out
        self.weight_numel = tensor_product.weight_numel
        self.harmonic_dim = attributes.irreps_out.dim
        self.packed_dim = attributes.packed_dim
        self.normalize = attributes.normalize
        self.affine = isinstance(projection, torch.nn.Linear)
        self.projection_scale = (
            1.0
            if self.affine
            else (projection.h_in * projection.var_in / projection.var_out) ** -0.5
        )

        input_index, output_index = [], []
        if layout == "mul_ir":
            for (mul, ir), section in zip(self.irreps_in1, self.irreps_in1.slices()):
                input_index.extend(
                    section.start + channel * ir.dim + m
                    for m in range(ir.dim)
                    for channel in range(mul)
                )
            for (mul, ir), section in zip(self.irreps_out, self.irreps_out.slices()):
                output_index.extend(
                    section.start + m * mul + channel
                    for channel in range(mul)
                    for m in range(ir.dim)
                )
        elif layout != "ir_mul":
            raise ValueError(f"Unsupported MACE layout: {layout}")
        self.register_buffer(
            "input_index", torch.tensor(input_index, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "output_index",
            torch.tensor(output_index, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "amplitudes", torch.ones(1, len(self.irreps_in2)), persistent=False
        )

    def forward(self, node_feats, edge_attrs, radial, edge_index):
        if self.input_index.numel():
            node_feats = node_feats.index_select(-1, self.input_index)
        if self.affine:
            projection = self.projection.weight.T
            if self.projection.bias is not None:
                projection = torch.cat((projection, self.projection.bias[None]), dim=0)
        else:
            projection = self.projection.weight * self.projection_scale
        end = self.harmonic_dim + self.packed_dim
        amplitudes = (
            self.amplitudes if self.normalize else edge_attrs[:, end:].contiguous()
        )
        message = self.eqx_tp(
            node_feats.contiguous(),
            radial.contiguous(),
            projection,
            edge_attrs[:, self.harmonic_dim : end].contiguous(),
            amplitudes,
            edge_index,
            node_feats.size(0),
        )
        if self.output_index.numel():
            message = message.index_select(-1, self.output_index)
        return message


def convert_mace_to_eqx(model, *, enable_cueq=False, inplace=False, backend="cuda"):
    """Replace MACE spatial convolutions with streamed EQX operations.

    Parameters
    ----------
    model : torch.nn.Module
        MACE model with supported RealAgnostic interactions. Set its device
        and floating-point precision before conversion.
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
    natural spatial parity. Magnetic interactions are not supported.

    The interaction classes, node linear maps, product bases and readouts
    remain unchanged. Only the convolution, radial MLP and harmonic attributes
    are adapted. The final radial projection is moved into the convolution,
    changing its parameter names but not its values or trainability.

    Convert before constructing the optimizer or distributed wrapper.
    Force training requires the usual MACE ``training=True`` forward argument.
    Do not enable another backend conversion in the ASE calculator afterwards;
    it disables parameter gradients for inference, so use a separate model
    copy if training is also needed.

    Save the converted module directly, or load its state dict into a model
    converted in the same way. Original MACE state dicts must be loaded before
    conversion. Importing this interface does not require MACE; calling it does.
    """
    from mace.modules.irreps_tools import tp_out_irreps_with_instructions
    from mace.modules.wrapper_ops import get_layout
    from mace.tools.torch_tools import default_dtype

    if backend not in ("cuda", "torch"):
        raise ValueError("backend must be 'cuda' or 'torch'.")
    if not hasattr(model, "spherical_harmonics") or not hasattr(model, "interactions"):
        raise TypeError("Expected a MACE model with spatial interactions.")
    if isinstance(model.spherical_harmonics, FrameAttributes):
        return model if inplace else deepcopy(model)
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
            model = deepcopy(model)

        products, radials = [], []
        for layer in model.interactions:
            irreps_out, instructions = tp_out_irreps_with_instructions(
                layer.edge_irreps, layer.edge_attrs_irreps, layer.target_irreps
            )
            tp = O3TensorProduct(
                layer.edge_irreps,
                layer.edge_attrs_irreps,
                irreps_out,
                instructions,
                internal_weights=False,
                shared_weights=False,
                normalization=model.spherical_harmonics.normalization,
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

        attributes = FrameAttributes(
            model.spherical_harmonics, max(tp.lmax for tp in products), backend
        ).train(model.spherical_harmonics.training)
        convolutions, radial_features = [], []
        for layer, tp, (radial, net, projection, affine) in zip(
            model.interactions, products, radials
        ):
            convolutions.append(
                Convolution(
                    tp,
                    projection,
                    get_layout(getattr(layer, "cueq_config", None)),
                    attributes,
                    backend,
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
