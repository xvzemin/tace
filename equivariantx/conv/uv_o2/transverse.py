"""Order restriction in spherical storage without a transverse frame."""

import math

import torch
from e3nn import o3

from ...co2.spherical import SphericalCoupling
from ...o2.layout import irrep_layout
from ...ops.torch.conv.uv_o2 import transverse as torch_kernel
from ...ops.precision import validate_module_precision
from ...utils.harmonics import harmonic_program


class TransverseFrame(torch.nn.Module):
    """Restrict spherical features to transverse order subspaces.

    Parameters
    ----------
    frame : LocalFrame
        Representations, local order cutoff and basis convention.
    backend : {"torch", "cuda"}, optional
        CUDA fuses the angular contractions and indexed aggregation.
    """

    def __init__(self, frame, backend="torch"):
        super().__init__()
        if not frame.basis_change:
            raise ValueError("Transverse evaluation requires the canonical O(2) basis.")
        self.backend = backend
        self.irreps_in = frame.irreps_in
        self.input_dim = frame.input_dim
        self.irreps_out = frame.irreps_out
        self.layout = irrep_layout(frame.irreps_out, True)
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
        self.permutation_indices = tuple(permutation)
        self.inverse_indices = tuple(
            sorted(range(len(permutation)), key=permutation.__getitem__)
        )
        permutation = torch.tensor(permutation, dtype=torch.long)
        self.register_buffer("permutation", permutation, persistent=False)
        self.register_buffer(
            "inverse_permutation", permutation.argsort(), persistent=False
        )

    def __repr__(self):
        return f"{type(self).__name__}({self.irreps_in} -> {self.irreps_out}, backend={self.backend!r})"

    def program(self, program, features, direction, *, inverse=False, native=True):
        """Encode restriction or lifting within a complete convolution tile."""
        if inverse:
            features = program.gather(features, self.inverse_indices)
        width = self.input_dim if inverse else self.output_dim
        if not native:
            from ...ops.common.convolution import tensor_product_program

            harmonics, offsets, position = [], {}, 0
            for degree in sorted({(path[6] - 1) // 2 for path in self.paths[inverse]}):
                offsets[degree] = position
                harmonics.append(harmonic_program(program, direction, degree))
                position += 2 * degree + 1
            paths = []
            for path in self.paths[inverse]:
                path = list(path)
                path[1] = offsets[(path[6] - 1) // 2]
                paths.append(tuple(path))
            output = tensor_product_program(
                program,
                repr((tuple(paths), 0, True)),
                features,
                program.concatenate(harmonics),
                program.constant(0, 0),
                width,
            )
            return (
                output if inverse else program.gather(output, self.permutation_indices)
            )
        empty = program.constant(0, 0)
        output = program.add(
            "convolution",
            width,
            (
                features,
                empty,
                empty,
                program.constant(1, 1),
                program.constant(width, 0),
                direction,
            ),
            ("o3", self.metadata[inverse], 4, False, (0, 0), 0),
        )
        return output if inverse else program.gather(output, self.permutation_indices)

    def forward(self, features, direction, source, target, num_nodes, *, inverse=False):
        """Restrict or lift features using unit directions and indexed aggregation."""
        validate_module_precision(
            self, (features, direction, source, target, num_nodes, inverse)
        )
        if self.backend == "cuda" and features.is_cuda:
            from ...ops.cuda.conv.uv_o2 import transverse as cuda_kernel

            return cuda_kernel.evaluate(
                self, features, direction, source, target, num_nodes, inverse
            )
        return torch_kernel.evaluate(
            self, features, direction, source, target, num_nodes, inverse
        )
