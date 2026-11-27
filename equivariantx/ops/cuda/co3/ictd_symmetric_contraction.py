"""Cartesian projection and basis conversion for CUDA contractions."""

from functools import lru_cache

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ...common.torch.basis import project_features
from .basis import project as project_blocks
from .basis import project_mixed_layout


@lru_cache(maxsize=64)
def _mixed_sections(irreps, groups):
    if tuple(degree for degree, _ in groups) != (2, 3):
        return ()
    slices = irreps.slices()
    bounds = sorted(
        (slices[i].start, slices[i].stop, l)
        for l, entries in groups
        for i, _ in entries
    )
    sections = []
    for start, stop, degree in bounds:
        if sections and sections[-1] == degree and sections[-2] == start:
            sections[-2] = stop
        else:
            sections.extend((start, stop, degree))
    return tuple(sections) if len(sections) <= 24 else ()


def project(module, features):
    """Project flattened Cartesian features with CUDA operations."""
    from ....co3 import Projector

    # Wide mixed Cartesian intermediates otherwise concatenate inputs by rank
    # before two GEMMs, then concatenate the outputs again (and both adjoints).
    # Keep unmeasured precisions, angular layouts and compiled paths unchanged.
    if (
        not torch.compiler.is_compiling()
        and get_proxy_mode() is None
        and not torch.jit.is_tracing()
        and type(module) is Projector
        and module.irreps_in.dim >= 4096
        and features.is_cuda
        and features.dtype == torch.float32
        and features.ndim == 2
        and features.is_contiguous()
        and not forward_ad_active()
        and not torch.is_autocast_enabled("cuda")
        and torch.get_float32_matmul_precision() == "highest"
        and (sections := _mixed_sections(module.irreps_in, module.groups))
        and not module.basis_3.requires_grad
        and module.basis_3.shape == (27, 7)
        and module.basis_3.dtype == features.dtype
    ):
        # Use the current basis, including edits/load_state_dict/.to(); no cached
        # geometry or implicit assumption that its seven columns are canonical.
        basis = module.basis_3
        return project_mixed_layout(features, basis @ basis.T, list(sections))
    return project_features(module, features, project_blocks)
