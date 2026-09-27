"""Order restriction in spherical storage without a transverse frame."""

import math

import torch
from e3nn import o3

from ...co2 import SphericalCoupling


def layout(irreps, transverse=False):
    """Return angular widths, multiplicities and offsets for local entries."""
    entries, offset = [], 0
    for ir, mul in irreps:
        dim = 2 * ir.m + 1 if transverse else ir.dim
        entries.append((dim, mul, offset))
        offset += dim * mul
    return tuple(entries)


class TransverseFrame(torch.nn.Module):
    """Restrict spherical features to transverse order subspaces.

    Parameters
    ----------
    frame : LocalFrame
        Representations, local order cutoff and basis convention.
    backend : {"torch", "cuda"}, optional
        CUDA fuses the angular contractions and indexed aggregation.
    """

    def __init__(self, frame, backend="cuda"):
        super().__init__()
        if not frame.basis_change:
            raise ValueError("Transverse evaluation requires the canonical O(2) basis.")
        self.backend = backend
        self.irreps_in = frame.irreps_in
        self.input_dim = frame.input_dim
        self.irreps_out = frame.irreps_out
        self.layout = layout(frame.irreps_out, True)
        self.output_dim = sum(dim * mul for dim, mul, _ in self.layout)
        self.couplings = torch.nn.ModuleDict()
        paths = [[], []]
        permutation = [0] * self.output_dim
        offset = 0
        for entry in frame._entries:
            l, mul = entry.degree, entry.mul
            for m, (index, section) in enumerate(
                zip(entry.local_indices, entry.local_slices)
            ):
                pole = torch.zeros(
                    2 * l + 1, 2 * m + 1, dtype=torch.float64, device="cpu"
                )
                if m == 0:
                    pole[l, 0] = 1
                elif entry.odd:
                    pole[l - m, 2 * m] = -1
                    pole[l + m, 0] = 1
                else:
                    pole[l + m, 2 * m] = 1
                    pole[l - m, 0] = 1
                scale = math.sqrt((2 * l + 1) / (2 * min(l, frame.mmax) + 1))
                for inverse, matrix in enumerate((pole, pole.T * scale)):
                    l1, l3 = (m, l) if inverse else (l, m)
                    start, end = (
                        (offset, entry.global_slice.start)
                        if inverse
                        else (entry.global_slice.start, offset)
                    )
                    reconstructed = torch.zeros_like(matrix)
                    for l2 in range(abs(l1 - l3), l1 + l3 + 1):
                        cg = o3.wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu")
                        value = (matrix * cg[:, l2, :]).sum() / cg[
                            :, l2, :
                        ].square().sum()
                        if abs(value) < 1e-14:
                            continue
                        reconstructed += value * cg[:, l2, :]
                        name = f"{l1}_{l2}_{l3}"
                        if name not in self.couplings:
                            self.couplings[name] = SphericalCoupling(l1, l2, l3, "norm")
                        paths[inverse].append(
                            (
                                start,
                                0,
                                end,
                                mul,
                                1,
                                2 * l1 + 1,
                                2 * l2 + 1,
                                2 * l3 + 1,
                                -1,
                                float(value),
                                tuple(
                                    (a, b, c, float(cg[a, b, c]))
                                    for a, b, c in cg.nonzero().tolist()
                                ),
                            )
                        )
                    if not torch.allclose(
                        matrix, reconstructed, atol=2e-12, rtol=2e-12
                    ):
                        raise ValueError("Cannot resolve the transverse restriction.")
                dim, width, begin = self.layout[index]
                for a in range(dim):
                    for c in range(mul):
                        permutation[begin + a * width + section.start + c] = (
                            offset + a * mul + c
                        )
                offset += dim * mul
        self.paths = tuple(tuple(values) for values in paths)
        self.metadata = tuple(
            repr((values, 0, True, ("transverse", "norm"))) for values in self.paths
        )
        permutation = torch.tensor(permutation, dtype=torch.long)
        self.register_buffer("permutation", permutation, persistent=False)
        self.register_buffer(
            "inverse_permutation", permutation.argsort(), persistent=False
        )

    def __repr__(self):
        return f"{type(self).__name__}({self.irreps_in} -> {self.irreps_out}, backend={self.backend!r})"

    def forward(self, features, direction, source, target, num_nodes, *, inverse=False):
        """Restrict or lift features using unit directions and indexed aggregation."""
        if inverse:
            features = features.index_select(1, self.inverse_permutation)
        width = self.input_dim if inverse else self.output_dim
        if self.backend == "cuda" and features.is_cuda:
            from ..o3.convolution import contraction

            values = [
                features,
                features.new_empty((1, 0)),
                features.new_empty((0, 0)),
                features.new_ones((1, 1)),
                features.new_empty(1).expand(num_nodes, width),
                direction,
            ]
            result = contraction(
                self.metadata[inverse],
                repr(((tuple(range(6)), False, ((4, 0),)),)),
                source,
                target,
                values,
            )[0]
        else:
            result = (
                features.new_zeros((num_nodes, width))
                + (features.sum() + direction.sum()) * 0
            )
            for (
                start,
                _,
                end,
                mul,
                _,
                dim,
                harmonic_dim,
                dim_out,
                _,
                scale,
                _,
            ) in self.paths[inverse]:
                x = features[source, start : start + mul * dim].reshape(
                    source.numel(), dim, mul
                )
                name = (
                    f"{(dim - 1) // 2}_{(harmonic_dim - 1) // 2}_{(dim_out - 1) // 2}"
                )
                value = self.couplings[name](x.transpose(1, 2), direction[:, None, :])
                value = value.transpose(1, 2).flatten(1) * scale
                result[:, end : end + mul * dim_out] += value.new_zeros(
                    (num_nodes, mul * dim_out)
                ).index_add(0, target, value)
        return result if inverse else result.index_select(1, self.permutation)
