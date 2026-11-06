################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from tace.models.blocks import ScaleShift
from tace.models.radial import AgnesiTransform, RadialBasis, SoftTransform, ZBLBasis
from tace.models.spherical.layer_norm import EquivariantMergeLayerNorm
from tace.models.spherical.mtace import MagneticBasis
from tace.models.spherical.scatter_norm import ScatterNorm
from tace.utils.loss.uncertainty import UncertaintyLoss

# Exclude all parameters recursively, including those of child modules.
NO_WEIGHT_DECAY_MODULES = (
    torch.nn.LayerNorm,
    torch.nn.RMSNorm,
    torch.nn.GroupNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.Embedding,
    torch.nn.EmbeddingBag,
    torch.nn.PReLU,
    EquivariantMergeLayerNorm,
    ScatterNorm,
    ScaleShift,
    AgnesiTransform,
    SoftTransform,
    RadialBasis,
    MagneticBasis,
    ZBLBasis,
    UncertaintyLoss,
)

# Glob patterns match the full parameter name, including module names.
NO_WEIGHT_DECAY_NAMES = (
    "*bias",
    "*bias.*",
    "biases",
    "biases.*",
    "*.biases",
    "*.biases.*",
    "temperature_logit",
    "*.temperature_logit",
    "*_readout*",
    "*.node_embedding.*",
    "*.source_embedding.*",
    "*.target_embedding.*",
    "*.uie.*",
    "*.uee.*",
    "*.embedding.*",
    "*.router.*",
)
