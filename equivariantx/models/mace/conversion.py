"""Convert MACE convolutions and symmetric contractions."""

from collections import OrderedDict

import torch
from e3nn import o3

from ...o2.wigner import WignerD
from ...ops import resolve_backend
from ...ops.common.wigner import wigner_D
from ...utils.module import copy_model, default_dtype
from ..convolution import Convolution, FrameInput, RadialFeatures
from .symmetric_contraction import UuuSymmetricContraction

__all__ = ["convert_mace_to_eqx"]


class FrameAttributes(torch.nn.Module):
    """Prepare shared rotations, retaining harmonics for other consumers if needed."""

    def __init__(self, harmonics, lmax, backend, *, convolution_only=False):
        super().__init__()
        self.harmonics = harmonics
        self.backend = backend
        self.irreps_out = harmonics.irreps_out
        self.normalization = harmonics.normalization
        self.normalize = harmonics.normalize
        self.convolution_only = convolution_only
        self._lmax = harmonics._lmax
        self.wigner = WignerD(lmax, lmax, backend=backend)
        self.packed_dim = sum((2 * ell + 1) ** 2 for ell in range(lmax + 1))
        self.register_buffer(
            "degrees",
            torch.tensor([ir.l for _, ir in self.irreps_out]),
            persistent=False,
        )

    def forward(self, vectors):
        packed = wigner_D(self.wigner, vectors, backend=self.backend)
        amplitudes = (
            None
            if self.normalize
            else vectors.norm(dim=-1, keepdim=True).pow(self.degrees)
        )
        if self.convolution_only:
            return FrameInput(packed, amplitudes)
        values = [self.harmonics(vectors), packed]
        if amplitudes is not None:
            values.append(amplitudes)
        return torch.cat(values, dim=-1)


def convert_mace_to_eqx(
    model,
    *,
    implementation="o3",
    enable_cueq=False,
    inplace=False,
    backend=None,
    stream_radial=False,
):
    """Replace MACE convolutions and symmetric contractions with EQX operations.

    Parameters
    ----------
    model : torch.nn.Module
        MACE model with supported spatial or magnetic interactions. Set its device
        and floating-point precision before conversion.
    implementation : {"o3", "o2", "wigner6j"}, optional
        Direct or aligned convolution, or magnetic Wigner-6j recoupling.
        The aligned form requires natural-parity edge harmonics. The magnetic form
        retains the original radial networks, products and readouts.
    enable_cueq : bool, optional
        Use MACE's CuEquivariance converter for the remaining operators.
        Not supported with ``"wigner6j"``.
    inplace : bool, optional
        Modify the supplied model instead of copying it. CuEquivariance may return a new model.
    backend : {None, "cuda", "torch"}, optional
        Execution backend. Defaults to CUDA when available, otherwise PyTorch.
    stream_radial : bool, optional
        Fuse the full radial MLP or only its final projection. Must be False
        with ``"wigner6j"``.

    Returns
    -------
    torch.nn.Module
        Converted model with the original forward interface and learned parameters.

    Notes
    -----
    Supports RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock,
    RealAgnosticDensityInteractionBlock, RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticAttResidualInteractionBlock and
    RealAgnosticResidualNonLinearInteractionBlock. Magnetic conversion supports
    MACE v0.3.17 density and residual density interactions. Chemical node
    attributes must be one-hot. Pass ``training=True`` to the MACE forward
    call for force training.

    Examples
    --------
    Given a loaded MACE model:

    .. code-block:: python

        from equivariantx.models.mace import convert_mace_to_eqx
        from mace.calculators import MACECalculator

        model = convert_mace_to_eqx(model.cuda())
        calculator = MACECalculator(models=model, device="cuda")

    For an e3nn magnetic MACE model:

    .. code-block:: python

        from mace.calculators import MagneticMACECalculator

        model = convert_mace_to_eqx(model.cuda(), implementation="wigner6j")
        calculator = MagneticMACECalculator(models=model, device="cuda")
    """
    backend = resolve_backend(backend)
    if implementation == "wigner6j":
        if enable_cueq or stream_radial:
            raise ValueError(
                "Wigner-6j conversion retains the original radial networks and operators."
            )
        from .magnetic import convert_magnetic_mace

        return convert_magnetic_mace(model, backend=backend, inplace=inplace)
    if implementation not in ("o3", "o2"):
        raise ValueError("implementation must be 'o3', 'o2', or 'wigner6j'.")
    if not hasattr(model, "spherical_harmonics") or not hasattr(model, "interactions"):
        raise TypeError("Expected a MACE model with spatial interactions.")

    from mace.modules.irreps_tools import tp_out_irreps_with_instructions
    from mace.modules.wrapper_ops import get_layout

    converted = [isinstance(layer.conv_tp, Convolution) for layer in model.interactions]
    if any(converted):
        if not all(converted):
            raise ValueError(
                "Convert the original model to select another implementation."
            )
        for layer in model.interactions:
            layer.conv_tp.check_options(
                layer.conv_tp_weights, implementation, backend, stream_radial
            )
        if any(
            not isinstance(block.symmetric_contractions, UuuSymmetricContraction)
            for block in model.products
        ):
            raise ValueError("The model contains unconverted symmetric contractions.")
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
            # These forwards pass edge_attrs only to the converted interactions.
            # A different forward can still consume the original harmonic prefix.
            from mace.modules.models import MACE, ScaleShiftMACE

            convolution_only = type(model) in (MACE, ScaleShiftMACE) and all(
                type(layer).__module__ == "mace.modules.blocks"
                for layer in model.interactions
            )
            attributes = FrameAttributes(
                attributes, lmax, backend, convolution_only=convolution_only
            ).train(attributes.training)
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
                    radial_mode="full" if stream_radial else "last",
                ).train(layer.conv_tp.training)
            )
            radial_features.append(
                RadialFeatures(
                    OrderedDict(list(net.named_children())[:-1]),
                    affine and projection.bias is not None,
                    radial.hs,
                    stream_radial=stream_radial,
                ).train(radial.training)
            )
        for layer, convolution, radial in zip(
            model.interactions, convolutions, radial_features
        ):
            layer.conv_tp = convolution
            layer.conv_tp_weights = radial
            layer.conv_fusion = True
        model.spherical_harmonics = attributes
        for block in model.products:
            block.symmetric_contractions = UuuSymmetricContraction(
                block.symmetric_contractions,
                backend,
                layout=get_layout(getattr(block, "cueq_config", None)),
            )
        return model.to(device=parameter.device, dtype=parameter.dtype)
