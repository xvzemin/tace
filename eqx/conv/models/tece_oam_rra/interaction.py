"""Streaming rotary-attention interaction for TECE-OAM-RRA."""

from weakref import WeakKeyDictionary, ref

import torch
from torch.func import functional_call

from ....kernels.recompute import replay
from ...attention import StreamingGraphAttention

_programs = WeakKeyDictionary()


@torch._dynamo.assume_constant_result
def program(module, tile_size):
    """Construct a reusable execution plan without retaining the model."""
    if module not in _programs:
        _programs[module] = {}
    if tile_size not in _programs[module]:
        reference = ref(module)
        names = tuple(name for name, _ in module.named_parameters())
        heads, channels = module.num_head, module.num_channel

        def score_value(values):
            (
                x,
                radial,
                matrix,
                bias,
                index,
                cutoff,
                rotation,
                inverse,
                basis,
                valid,
                *weights,
            ) = values
            return functional_call(
                reference(),
                dict(zip(names, weights)),
                (x, radial @ matrix + bias, index, None, rotation, inverse, basis),
                {"stage": "score_value", "fused": True},
            )

        def fake(values):
            x = values[0]
            return (
                x.new_empty((x.shape[0], values[7].shape[1], channels)),
                x.new_empty((x.shape[0], heads)),
                x.new_empty((x.shape[0], heads)),
            )

        _programs[module][tile_size] = StreamingGraphAttention(
            score_value,
            fake,
            {1: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
            target_slot=4,
            normalizer_slot=5,
            value_weight_slot=5,
            valid_slot=9,
            tile_size=tile_size,
            eps=module.attention_eps,
        )
        _programs[module][tile_size].resource = reference
    return _programs[module][tile_size].key


def pad_edges(x, size):
    if x.shape[0] < size:
        return torch.cat((x, x.new_zeros((size - x.shape[0], *x.shape[1:]))))
    return x


def stream(
    module,
    features,
    radial,
    projection,
    bias,
    edge_index,
    cutoff,
    wigner,
    wigner_inv,
    radial_basis,
    tile_size=2048,
):
    """Stream radial projection, local updates and graph attention together.

    Only node-level attention statistics are retained. Convolution weights,
    scores, queries, keys and messages are produced in bounded edge tiles.
    """
    if not module.use_radial_rotary_attention:
        raise ValueError(
            "Streaming rotary attention requires an attention-enabled convolution."
        )
    edges = edge_index.shape[1]
    if cutoff is None:
        cutoff = features.new_ones((edges, 1))
    if bias is None:
        bias = projection.new_zeros(projection.shape[1])
    padded = ((edges + tile_size - 1) // tile_size) * tile_size
    valid = torch.arange(padded, device=features.device) < edges
    output = replay(
        [
            features,
            pad_edges(radial, padded),
            projection,
            bias,
            pad_edges(edge_index.T, padded).T,
            pad_edges(cutoff, padded),
            pad_edges(wigner, padded),
            pad_edges(wigner_inv, padded),
            pad_edges(radial_basis, padded),
            valid,
            *module.parameters(),
        ],
        program(module, tile_size),
    )[0]
    return module.reshape_out.inverse(output)
