"""Wigner block construction, dense storage, and ordinary recurrence."""

import torch


def dense(module, vectors):
    """Construct degree matrices or their dense block storage."""
    packed = module.forward_packed(vectors)
    dim = (module.lmax + 1) ** 2
    if module.mmax < module.lmax:
        packed = packed.index_select(1, module._packed_indices)
    wigner = vectors.new_zeros((vectors.size(0), module.local_indices.numel() * dim))
    wigner = wigner.index_copy(1, module._dense_indices, packed).view(
        vectors.size(0), module.local_indices.numel(), dim
    )
    wigner_inv = wigner.transpose(1, 2).contiguous()
    if module.mmax < module.lmax:
        wigner_inv = wigner_inv * module.inverse_scale.to(wigner_inv)
    return (wigner, wigner_inv)


def from_matrix(module, rotation):
    """Construct degree matrices or their dense block storage."""
    batch = rotation.shape[0]
    matrices = [rotation.new_ones((batch, 1, 1))]
    if module.lmax >= 1:
        matrices.append(rotation)
    for l in range(2, module.lmax + 1):
        cg = getattr(module, f"cg_{l}").to(rotation)
        matrix = torch.einsum("abm,eac->ebmc", cg, rotation)
        matrix = torch.einsum("ebmc,ebd->emcd", matrix, matrices[-1])
        matrix = torch.einsum("emcd,cdn->emn", matrix, cg)
        matrices.append(matrix * (2 * l + 1))
    return matrices


def packed(module, vectors, method):
    """Choose exact quaternion polynomials or the ordinary Wigner recurrence."""
    if method == "quaternion":
        from equivariantx.ops.common.quaternion import quaternion_wigner

        return quaternion_wigner(vectors, module.lmax, backend="torch")
    from equivariantx.o2.rotation_matrix import rotation_matrix_to_y_axis

    return torch.cat(
        [
            value.flatten(1)
            for value in module.from_matrix(rotation_matrix_to_y_axis(vectors))
        ],
        dim=1,
    )
