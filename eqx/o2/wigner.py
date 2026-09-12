################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math

import opt_einsum_fx
import torch
from e3nn import o3

from .rotation_matrix import rotation_matrix_to_y_axis


class WignerD(torch.nn.Module):
    """Construct global-to-local and local-to-global rotation matrices.

    Parameters
    ----------
    mmax : int
        Largest local O(2) order retained in the local matrix axis.
    lmax : int
        Largest global O(3) degree represented by the matrices.
    use_opt_einsum_fx : bool, optional
        If ``True``, pre-optimize the recursive contractions for degrees two
        and above.

    Notes
    -----
    Each input vector defines a local frame whose second axis is aligned with
    the vector. The matrices use degree-major global storage and truncated
    order-major local storage compatible with :class:`LocalFrame`.
    """

    def __init__(
        self,
        mmax: int,
        lmax: int,
        use_opt_einsum_fx: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(lmax, int):
            raise TypeError("lmax must be an integer.")
        if not isinstance(mmax, int):
            raise TypeError("mmax must be an integer.")
        if lmax < 0:
            raise ValueError("lmax must be non-negative.")
        if not 0 <= mmax <= lmax:
            raise ValueError("mmax must satisfy 0 <= mmax <= lmax.")

        self.mmax = mmax
        self.lmax = lmax
        self.use_opt_einsum_fx = use_opt_einsum_fx

        for l in range(2, self.lmax + 1):
            self.register_buffer(f"cg_{l}", o3.wigner_3j(1, l - 1, l), persistent=False)
            if self.use_opt_einsum_fx:
                self._register_fx(l)

        local_indices = []
        inverse_scale = []
        for m in range(mmax + 1):
            for signed_m in (0,) if m == 0 else (m, -m):
                for l in range(m, lmax + 1):
                    local_indices.append(l**2 + l + signed_m)
                    inverse_scale.append(
                        math.sqrt((2 * l + 1) / (2 * min(l, mmax) + 1))
                    )
        self.register_buffer(
            "local_indices",
            torch.tensor(local_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "inverse_scale",
            torch.tensor(inverse_scale).view(1, 1, -1),
            persistent=False,
        )

    def forward(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build both rotation directions for a batch of vectors.

        Parameters
        ----------
        vectors : torch.Tensor
            Three-dimensional vectors with shape ``(batch, 3)``. Their
            magnitudes do not affect the resulting frame.

        Returns
        -------
        wigner : torch.Tensor
            Global-to-local matrix with shape
            ``(batch, local_dim, (lmax + 1)**2)``.
        wigner_inv : torch.Tensor
            Local-to-global matrix with shape
            ``(batch, (lmax + 1)**2, local_dim)``. Truncated degrees include
            the variance-preserving inverse scale.
        """
        if vectors.ndim != 2 or vectors.shape[-1] != 3:
            raise ValueError("vectors must have shape (batch, 3).")
        rotation = rotation_matrix_to_y_axis(vectors)
        batch = vectors.shape[0]
        matrices = [rotation.new_ones((batch, 1, 1))]
        if self.lmax >= 1:
            matrices.append(rotation)
        for l in range(2, self.lmax + 1):
            cg = getattr(self, f"cg_{l}")
            if self.use_opt_einsum_fx:
                matrix = getattr(self, f"fx_{l}")(rotation, matrices[-1], cg)
            else:
                matrix = torch.einsum("abm,eac->ebmc", cg, rotation)
                matrix = torch.einsum("ebmc,ebd->emcd", matrix, matrices[-1])
                matrix = torch.einsum("emcd,cdn->emn", matrix, cg)
            matrices.append(matrix * (2 * l + 1))

        dim = (self.lmax + 1) ** 2
        wigner = rotation.new_zeros((batch, dim, dim))
        for l, matrix in enumerate(matrices):
            wigner[:, l**2 : (l + 1) ** 2, l**2 : (l + 1) ** 2] = matrix
        wigner = wigner.index_select(1, self.local_indices)
        wigner_inv = wigner.transpose(1, 2).contiguous() * self.inverse_scale
        return wigner, wigner_inv

    def _register_fx(self, degree: int) -> None:
        equation = "abm,eac,ebd,cdn->emn"
        contraction = torch.fx.symbolic_trace(
            lambda d1, d_prev, cg: torch.einsum(equation, cg, d1, d_prev, cg)
        )
        contraction = opt_einsum_fx.optimize_einsums_full(
            model=contraction,
            example_inputs=(
                torch.randn(4, 3, 3),
                torch.randn(4, 2 * degree - 1, 2 * degree - 1),
                torch.randn(3, 2 * degree - 1, 2 * degree + 1),
            ),
        )
        self.add_module(f"fx_{degree}", contraction)

    def extra_repr(self) -> str:
        return f"mmax={self.mmax}, lmax={self.lmax}"
