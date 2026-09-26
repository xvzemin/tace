################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math

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
    method : {"auto", "quaternion", "recursive"}, optional
        ``"auto"`` uses direct quaternion polynomials for CUDA float32 and
        float64 inputs and recursive PyTorch contractions otherwise.
        ``"quaternion"`` requires CUDA and its toolkit. ``"recursive"``
        selects PyTorch contractions on either device.

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
        *,
        method: str = "auto",
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
        if method not in ("auto", "quaternion", "recursive"):
            raise ValueError("method must be auto, quaternion or recursive.")

        self.mmax = mmax
        self.lmax = lmax
        self.method = method

        for l in range(2, self.lmax + 1):
            self.register_buffer(f"cg_{l}", o3.wigner_3j(1, l - 1, l), persistent=False)

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
        matrices = self.matrix_blocks(vectors)
        dim = (self.lmax + 1) ** 2
        wigner = vectors.new_zeros((vectors.size(0), dim, dim))
        for l, matrix in enumerate(matrices):
            wigner[:, l**2 : (l + 1) ** 2, l**2 : (l + 1) ** 2] = matrix
        wigner = wigner.index_select(1, self.local_indices)
        wigner_inv = wigner.transpose(1, 2).contiguous() * self.inverse_scale
        return wigner, wigner_inv

    def forward_packed(self, vectors: torch.Tensor, *, method=None) -> torch.Tensor:
        """Return full degree blocks concatenated as ``(batch, sum((2*l+1)**2))``.

        No zero padding, order regrouping or inverse copy is stored. The
        convolution reads transposed blocks for the inverse rotation.
        This layout retains every order, independently of ``mmax``.
        ``method`` optionally overrides the construction method for this call.
        """
        method = getattr(self, "method", "auto") if method is None else method
        if method not in ("auto", "quaternion", "recursive"):
            raise ValueError("method must be auto, quaternion or recursive.")
        if method == "quaternion" or (
            method == "auto"
            and vectors.is_cuda
            and vectors.dtype in (torch.float32, torch.float64)
        ):
            from eqx.kernels import wigner_D

            return wigner_D(self, vectors, method=method)
        return torch.cat(
            [
                matrix.flatten(1)
                for matrix in self.matrix_blocks(vectors, method="recursive")
            ],
            dim=1,
        )

    def matrix_blocks(
        self, vectors: torch.Tensor, *, method=None
    ) -> list[torch.Tensor]:
        """Return degree matrices, optionally overriding the construction method."""
        if vectors.ndim != 2 or vectors.shape[-1] != 3:
            raise ValueError("vectors must have shape (batch, 3).")
        method = getattr(self, "method", "auto") if method is None else method
        if method != "recursive" and (vectors.is_cuda or method == "quaternion"):
            packed = self.forward_packed(vectors, method=method)
            return [
                block.view(vectors.size(0), 2 * l + 1, 2 * l + 1)
                for l, block in enumerate(
                    packed.split(
                        [(2 * l + 1) ** 2 for l in range(self.lmax + 1)], dim=1
                    )
                )
            ]
        rotation = rotation_matrix_to_y_axis(vectors)
        batch = vectors.shape[0]
        matrices = [rotation.new_ones((batch, 1, 1))]
        if self.lmax >= 1:
            matrices.append(rotation)
        for l in range(2, self.lmax + 1):
            cg = getattr(self, f"cg_{l}")
            matrix = torch.einsum("abm,eac->ebmc", cg, rotation)
            matrix = torch.einsum("ebmc,ebd->emcd", matrix, matrices[-1])
            matrix = torch.einsum("emcd,cdn->emn", matrix, cg)
            matrices.append(matrix * (2 * l + 1))

        return matrices

    def extra_repr(self) -> str:
        return f"mmax={self.mmax}, lmax={self.lmax}"
