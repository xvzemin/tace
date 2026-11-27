"""Trainable rotary-attention convolution on the host's existing local frames."""

import torch

from equivariantx import o2
from tace.models.linear import has_lora
from tace.models.spherical.tece_oam_rra import Convolution, UvSO2Interaction
from tace.utils.env import acceleration_enabled, memory_first

from ..matching import (
    custom_forward,
    directed_radial,
    mapped_radial_inputs,
    streamable_activation,
    streamable_radial_mlp,
    tracing_rows,
)


class StreamedRotaryInteraction(UvSO2Interaction):
    supports_receiver_count = True

    def _rotary_radial(self, features, edge_features, edge_index):
        if (
            not features.is_cuda
            or features.dtype not in (torch.float32, torch.float64)
            or torch.is_autocast_enabled("cuda")
            or self.rejector._eqx_metadata is None
        ):
            return None
        if not memory_first() or directed_radial(edge_features, edge_index):
            return self._convolution_radial(edge_features)
        if (
            isinstance(edge_features, tuple)
            and edge_features
            and (
                tracing_rows(edge_index)
                or (len(edge_features) == 3 and edge_features[0][0].ndim == 2)
            )
        ):
            # The indexed expression carries both maps as live int64 operands.
            # Do not specialize an exported graph to the example's endpoints.
            layers = self.edge_info.mlp
            radial, network = mapped_radial_inputs(
                edge_features, layers[:-1], edge_index, features
            )
            return radial, layers[-1], network
        return None


def _lower(layer):
    # A previously prepared host may have changed activations or path buffers.
    # Rebuild on the copy, using the current model rather than its cached strings.
    layer.rejector._eqx_weight_metadata = None
    layer.rejector.prepare_eqx()
    if layer.rejector._eqx_metadata is None:
        raise ValueError("Unsupported rotary convolution expression")
    layer.__class__ = StreamedRotaryInteraction
    return layer


def _restore(layer):
    layer.__class__ = UvSO2Interaction
    return layer


def rotary_rule(layer, *, backend, task, automatic=False):
    """Select an eager CUDA interaction without changing the angular basis."""
    if type(layer) is StreamedRotaryInteraction:
        if backend == "torch" and not acceleration_enabled("eqx", kernel="conv"):
            return _restore, "restored the host rotary convolution"
        return None, "retained the existing rotary convolution provider"
    if type(layer) is not UvSO2Interaction:
        return None, "requires a native rotary interaction"
    if acceleration_enabled("eqx", kernel="conv"):
        return None, "preserved the host rotary convolution provider"
    if (
        automatic
        or backend != "cuda"
        or task not in ("inference", "training")
        or torch.version.hip
    ):
        return None, "rotary streaming requires explicit NVIDIA CUDA"
    if not memory_first():
        return None, "kept the host rotary interaction in the speed-first plan"
    conv, radial = layer.rejector, layer.edge_info
    if (
        type(conv) is not Convolution
        or not (conv.use_radial_rotary_attention and layer.use_radial_rotary_attention)
        or not conv.use_asymmetric_contraction
        or conv.gate_m0
        or conv.mmax != conv.lmax
        or not streamable_radial_mlp(radial)
        or not all(streamable_activation(m) for m in (conv.scalar_act, conv.tensor_act))
        or type(conv.nonlinearity) is not o2.Gate
        or type(conv.ece) is not o2.TensorProduct
        or type(conv.radial_product) is not o2.TensorProduct
        or any(
            a is not None
            for gate in (conv.nonlinearity.act_gates, conv.nonlinearity.act_scalars)
            for a in gate.acts
        )
        or any(
            ins.connection_mode != "uuu" or not ins.has_weight
            for ins in conv.ece.instructions
        )
        or len(conv.radial_product.instructions) != conv.mmax + 1
        or not all(
            (
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                ins.connection_mode,
                ins.has_weight,
                ins.path_weight,
            )
            == (i, i, i, "uuu", False, 1.0)
            for i, ins in enumerate(conv.radial_product.instructions)
        )
        or not all(
            type(getattr(conv, name)) is o2.Linear
            for name in (
                "linear_up",
                "linear_down",
                "linear_coefs",
                "key_proj",
                "query_proj",
                "radial_proj",
            )
        )
    ):
        return None, "unsupported rotary expression or radial network"
    if (
        custom_forward(layer, recursive=True)
        or any(has_lora(m) for m in layer.modules())
        or any(
            k in vars(layer)
            for k in ("_compute_messages", "_rotary_radial", "_convolution_radial")
        )
        or any(
            k in vars(conv) for k in ("align", "unalign", "rotary_score", "prepare_eqx")
        )
    ):
        return None, "preserved a customized rotary interaction"
    weight = conv.temperature_logit
    if not weight.is_cuda or weight.dtype not in (torch.float32, torch.float64):
        return None, "rotary streaming requires CUDA FP32 or FP64"
    return (
        _lower,
        "streamed radial network, edge product and rotary attention on the host frames",
    )
