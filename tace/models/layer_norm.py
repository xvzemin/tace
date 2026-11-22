################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from equivariantx.nn import MergeLayerNorm, MergeRmsNorm, SoftCapRmsNorm
from tace.utils.env import acceleration_enabled


def load_norm_state(module, state, prefix, *args):
    """Translate saved TACE normalization entries before loading EQX parameters."""
    state.pop(prefix + "expand_index", None)
    weights = state.pop(prefix + "block_weights", None)
    legacy = state.pop(prefix + "balance_degree_weight", None)
    if legacy is not None:
        if legacy.shape != (1, sum(module.components)):
            raise ValueError(f"Invalid legacy normalization shape at {prefix}.")
        weights = torch.stack(
            [part[0, 0] for part in legacy.split(module.components, -1)]
        )
    if weights is not None and weights.shape != (len(module.channels),):
        raise ValueError(f"Invalid normalization block weights at {prefix}.")
    module.block_weights = (
        tuple(weights.tolist()) if weights is not None else module.default_block_weights
    )
    for old, new in (("affine_weight", "weight"), ("affine_bias", "bias")):
        if prefix + old in state:
            value = state.pop(prefix + old)
            if new == "weight" and (
                not module.uniform
                or value.shape != (len(module.channels), module.channels[0])
            ):
                raise ValueError(f"Invalid legacy affine weight shape at {prefix}.")
            state[prefix + new] = value.flatten() if new == "weight" else value


def save_norm_state(module, state, prefix, metadata):
    """Save block weights only when they differ from the analytic defaults."""
    if module.block_weights != module.default_block_weights:
        state[prefix + "block_weights"] = torch.tensor(
            module.block_weights, dtype=torch.float64, device="cpu"
        )


def residual_update_count(
    num_layers: int,
    resnet_type: str | None,
    use_first_resnet: bool,
    use_first_pre_norm: bool,
) -> int:
    per_layer = {"BB": 1, "AB": 1, "BAB": 2}.get(resnet_type, 0)
    if not (use_first_resnet and use_first_pre_norm):
        num_layers -= 1
    return per_layer * max(0, num_layers)


def residual_update_scale(num_updates: int) -> float:
    if num_updates < 0:
        raise ValueError("num_updates must be non-negative")
    return max(1, num_updates) ** -0.5


def get_normalization_layer(norm_type: str, irreps, **kwargs) -> torch.nn.Module:
    """Build the selected feature normalization."""
    kwargs.setdefault("backend", None if acceleration_enabled("eqx") else "torch")
    if norm_type == "soft_cap_rms_norm":
        return SoftCapRmsNorm(irreps, **kwargs)
    if norm_type == "merge_layer_norm":
        module = MergeLayerNorm(irreps, **kwargs)
    elif norm_type == "merge_rms_norm":
        module = MergeRmsNorm(irreps, **kwargs)
    else:
        raise ValueError(f"Unknown normalization: {norm_type!r}.")
    module.register_load_state_dict_pre_hook(load_norm_state)
    module.register_state_dict_post_hook(save_norm_state)
    return module
