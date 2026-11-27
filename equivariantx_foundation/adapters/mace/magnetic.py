"""Wigner-6j adapters for MACE magnetic interactions."""

import torch
from e3nn import o3

from equivariantx.o3.wigner6j import Wigner6jTensorProductConv
from equivariantx.ops.common.torch import channel_linear
from equivariantx.ops.common.torch.linear import linear_layout
from equivariantx.utils.module import copy_model, default_dtype


class Wigner6jInteraction(torch.nn.Module):
    """Replace two magnetic tensor products by a recoupled convolution.

    Parameters
    ----------
    interaction : torch.nn.Module
        MACE magnetic density interaction with two external-weight ``uvu``
        tensor products. Other layers and their parameters are retained.
    backend : {"torch", "cuda"}
        Convolution implementation.
    """

    def __init__(self, interaction, backend):
        super().__init__()
        first, second = interaction.conv_tp, interaction.magmom_conv_tp
        if not all(isinstance(tp, o3.TensorProduct) for tp in (first, second)):
            raise TypeError("Convert magnetic MACE from its e3nn implementation.")
        if first.irreps_out != second.irreps_in1:
            raise ValueError(
                "Magnetic tensor products must have matching intermediates."
            )
        if any(tp.internal_weights or tp.shared_weights for tp in (first, second)):
            raise ValueError("Magnetic interactions require external edge weights.")

        instructions = [
            [
                (
                    ins.i_in1,
                    ins.i_in2,
                    ins.i_out,
                    ins.connection_mode,
                    ins.has_weight,
                    ins.path_weight**2,
                )
                for ins in tp.instructions
            ]
            for tp in (first, second)
        ]
        parameter = next(interaction.parameters())
        with default_dtype(parameter.dtype):
            self.conv_tp = Wigner6jTensorProductConv(
                first.irreps_in1,
                first.irreps_in2,
                second.irreps_in2,
                first.irreps_out,
                second.irreps_out,
                *instructions,
                irrep_normalization="none",
                path_normalization="none",
                internal_weights=False,
                shared_weights=False,
                backend=backend,
                recouple=True,
            ).to(parameter)
        if (self.conv_tp.weight_numel1, self.conv_tp.weight_numel2) != (
            first.weight_numel,
            second.weight_numel,
        ):
            raise ValueError("The original tensor-product weight layouts must match.")

        for name in (
            "linear_up",
            "conv_tp_weights",
            "conv_tp_weights_magmom",
            "density_fn",
            "magmom_linear",
            "reshape",
        ):
            setattr(self, name, getattr(interaction, name))
        self.skip_tp = getattr(interaction, "skip_tp", None)
        self.magmom_skip_tp = getattr(interaction, "magmom_skip_tp", None)
        self.linear_metadata = repr(
            (
                "mul_ir",
                linear_layout(
                    tuple((ir.dim, mul) for mul, ir in self.magmom_linear.irreps_in),
                    tuple((ir.dim, mul) for mul, ir in self.magmom_linear.irreps_out),
                    self.magmom_linear.instructions,
                ),
            )
        )
        self.irreps_out = interaction.irreps_out
        self.train(interaction.training)

    def forward(
        self,
        node_attrs,
        node_feats,
        edge_attrs,
        edge_feats,
        edge_index,
        magmom_node_inv_feats,
        magmom_node_attrs,
    ):
        """Return aggregated magnetic features and the original skip connection."""
        source, target = edge_index[0], edge_index[1]
        skip = (
            self.skip_tp(node_feats, node_attrs) if self.skip_tp is not None else None
        )
        node_feats = self.linear_up(node_feats)
        radial_input = torch.cat((edge_feats, magmom_node_inv_feats[source]), dim=-1)
        weight1 = self.conv_tp_weights(radial_input)
        weight2 = self.conv_tp_weights_magmom(radial_input)
        edge_density = torch.tanh(self.density_fn(edge_feats).square())
        density = edge_density.new_zeros((node_feats.shape[0], 1)).index_add(
            0, target, edge_density
        )
        message = self.conv_tp(
            node_feats,
            edge_attrs,
            magmom_node_attrs,
            edge_index,
            weight1,
            weight2,
        )
        message = channel_linear.primal(
            self.linear_metadata,
            message,
            self.magmom_linear.weight,
            self.magmom_linear.bias,
        ) / (density + 1)
        if self.magmom_skip_tp is not None:
            message = self.magmom_skip_tp(message, node_attrs)
        return self.reshape(message), skip


def convert_magnetic_mace(model, *, backend, inplace=False):
    """Replace magnetic interactions, leaving products and readouts unchanged."""
    supported = {
        "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
        "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
    }
    if not hasattr(model, "interactions") or not len(model.interactions):
        raise TypeError("Expected a magnetic MACE model with interactions.")
    for layer in model.interactions:
        if isinstance(layer, Wigner6jInteraction):
            if layer.conv_tp.backend != backend:
                raise ValueError("Convert the original model to change its backend.")
        elif type(layer).__name__ not in supported:
            raise NotImplementedError(
                f"Unsupported magnetic interaction: {type(layer).__name__}"
            )
        config = getattr(layer, "cueq_config", None)
        if config is not None and config.enabled:
            raise ValueError("Convert magnetic MACE before enabling CuEquivariance.")
    if not inplace:
        model = copy_model(model)
    replacements = [
        layer
        if isinstance(layer, Wigner6jInteraction)
        else Wigner6jInteraction(layer, backend)
        for layer in model.interactions
    ]
    model.interactions = torch.nn.ModuleList(replacements)
    return model
