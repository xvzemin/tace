"""Permutation-symmetric many-body contractions."""

import math
from collections import Counter
from itertools import product

import torch

from .asymmetric_contraction import AsymmetricContraction, _Path


class SymmetricContraction(AsymmetricContraction):
    """Contract powers of one O(2) feature tensor.

    Parameters
    ----------
    irreps_in, irreps_out : Irreps, str, or sequence
        Input and requested output representations with a common multiplicity.
    correlation : int
        Maximum polynomial degree.
    algorithm : {"recursive", "dense"}, optional
        Evaluate shared coupling paths or weighted generalized tensors.
    path_mode : {"sum", "expand"}, optional
        Sum paths into each output type or retain their multiplicities.

    Notes
    -----
    Permutation-equivalent and linearly dependent paths are removed. Both
    algorithms use the same basis and externally supplied path weights.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        correlation,
        *,
        algorithm="recursive",
        path_mode="sum",
    ):
        self.path_normalizations = {}
        super().__init__(
            irreps_in,
            irreps_out,
            correlation,
            algorithm="recursive",
            path_mode=path_mode,
        )
        self._path_scales = tuple(
            tuple(
                scale * self.path_normalizations[path.leaves, path.intermediates]
                for path, scale in zip(paths, scales)
            )
            for paths, scales in zip(self._paths_by_order, self._path_scales)
        )
        self.set_algorithm(algorithm)

    def _enumerate_states(self):
        offsets = []
        offset = 0
        for ir in self._input_irreps:
            offsets.append(offset)
            offset += ir.dim
        result = []
        for states in super()._enumerate_states():
            selected, bases = [], {}
            for leaves, intermediates in states:
                if tuple(sorted(leaves)) != leaves:
                    continue
                ir = intermediates[-1]
                if ir not in self.irreps_out_types:
                    continue
                coordinates = tuple(
                    product(
                        *(
                            range(offsets[i], offsets[i] + self._input_irreps[i].dim)
                            for i in leaves
                        )
                    )
                )
                monomials = sorted({tuple(sorted(x)) for x in coordinates})
                locations = {x: i for i, x in enumerate(monomials)}
                coefficient = self._coupling_tensor(_Path(leaves, intermediates, 0))
                symmetric = coefficient.new_zeros(len(monomials), ir.dim)
                for indices, value in zip(coordinates, coefficient.reshape(-1, ir.dim)):
                    symmetric[locations[tuple(sorted(indices))]] += value
                # Frobenius metric of the symmetrized coefficient tensor.
                multiplicities = coefficient.new_tensor(
                    [
                        math.factorial(len(leaves))
                        / math.prod(math.factorial(n) for n in Counter(x).values())
                        for x in monomials
                    ]
                )
                vector = (symmetric / multiplicities.sqrt()[:, None]).flatten()
                norm = vector.norm()
                if norm < 1e-12:
                    continue
                orthogonal = vector / norm
                basis = bases.setdefault((leaves, ir), [])
                for _ in range(2):
                    for previous in basis:
                        orthogonal = (
                            orthogonal - torch.dot(previous, orthogonal) * previous
                        )
                length = orthogonal.norm()
                if length < 1e-10:
                    continue
                basis.append(orthogonal / length)
                selected.append((leaves, intermediates))
                self.path_normalizations[leaves, intermediates] = math.sqrt(
                    ir.dim
                ) / float(norm)
            result.append(tuple(selected))
        return tuple(result)

    def forward(self, features: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Evaluate symmetric powers with external path weights.

        Parameters
        ----------
        features : torch.Tensor
            Features of shape ``(..., irreps_in.dim)`` in ``ir_mul`` layout.
        weight : torch.Tensor
            Weights of shape ``(..., weight_numel)``. Leading dimensions
            broadcast with the features.

        Returns
        -------
        torch.Tensor
            Features of shape ``(..., irreps_out.dim)`` in ``ir_mul`` layout.
        """
        return super().forward((features,) * self.correlation, weight)
