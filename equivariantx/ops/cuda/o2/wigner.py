"""Native packed Wigner geometry and recursive CUDA derivatives."""

import torch

from ...common.wigner import alignment_cuda, packed_wigner, rotation
from equivariantx.ops.common.quaternion import quaternion_wigner


def packed(module, vectors, method):
    """Construct native quaternion or recursive degree matrices."""
    if method in ("auto", "quaternion"):
        return quaternion_wigner(vectors, module.lmax, backend="cuda")
    if not vectors.requires_grad or not torch.is_grad_enabled():
        return packed_wigner(
            vectors,
            module.lmax,
            [getattr(module, f"cg_{l}").to(vectors) for l in range(2, module.lmax + 1)],
        )
    aligned = alignment_cuda("None", [vectors])[0]
    matrices = [aligned.new_ones((vectors.size(0), 1))]
    if module.lmax:
        matrices.append(aligned)
    for degree in range(2, module.lmax + 1):
        placeholder = aligned.new_empty(1).expand(
            vectors.size(0), (2 * degree + 1) ** 2
        )
        matrices.append(
            rotation(
                getattr(module, f"cg_{degree}").to(aligned),
                2,
                [aligned, matrices[-1], placeholder],
            )
        )
    return torch.cat(matrices, dim=1)
