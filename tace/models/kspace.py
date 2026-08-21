################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Sequence, Tuple, Union

import torch

from tace.utils.torch_scatter import scatter_sum


GridSize = Union[int, Sequence[int]]


def _canonical_grid_size(grid_size: GridSize) -> Tuple[int, int, int]:
    if isinstance(grid_size, int) and not isinstance(grid_size, bool):
        grid_size = (grid_size, grid_size, grid_size)
    else:
        grid_size = tuple(grid_size)
    if len(grid_size) != 3:
        raise ValueError("grid_size must contain exactly three dimensions.")
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1
        for size in grid_size
    ):
        raise ValueError("Every reciprocal grid dimension must be positive.")
    return grid_size


class ReciprocalModes(torch.nn.Module):

    def __init__(self, grid_size: GridSize) -> None:
        super().__init__()
        self.grid_size = _canonical_grid_size(grid_size)

        integer_axes = [
            torch.fft.fftfreq(size, d=1.0 / size).to(torch.int64)
            for size in self.grid_size
        ]
        mode_indices = torch.stack(
            torch.meshgrid(*integer_axes, indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)

        grid_axes = [torch.arange(size) for size in self.grid_size]
        grid_indices = torch.stack(
            torch.meshgrid(*grid_axes, indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)
        negative_grid_indices = torch.stack(
            [
                (-grid_indices[:, axis]) % self.grid_size[axis]
                for axis in range(3)
            ],
            dim=-1,
        )
        negative_mode_index = (
            (negative_grid_indices[:, 0] * self.grid_size[1])
            + negative_grid_indices[:, 1]
        ) * self.grid_size[2] + negative_grid_indices[:, 2]

        self.register_buffer("mode_indices", mode_indices, persistent=False)
        self.register_buffer(
            "negative_mode_index",
            negative_mode_index,
            persistent=False,
        )

    @property
    def num_modes(self) -> int:
        return self.mode_indices.size(0)

    def reciprocal_vectors(self, lattice: torch.Tensor) -> torch.Tensor:
        """Return Cartesian reciprocal vectors with shape ``(B, K, 3)``."""
        if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
            raise ValueError("lattice must have shape (num_graphs, 3, 3).")
        inverse_lattice = torch.linalg.inv(lattice)
        return self._reciprocal_vectors_from_inverse(inverse_lattice)

    def _reciprocal_vectors_from_inverse(
        self,
        inverse_lattice: torch.Tensor,
    ) -> torch.Tensor:
        inverse_transpose = inverse_lattice.transpose(-1, -2)
        modes = self.mode_indices.to(dtype=inverse_lattice.dtype)
        return 2.0 * math.pi * torch.einsum(
            "kd,bdc->bkc",
            modes,
            inverse_transpose,
        )

    def phases(
        self,
        positions: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return dimensionless phases ``k . r`` with shape ``(N, K)``."""
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape (num_atoms, 3).")
        if batch.ndim != 1 or batch.shape[0] != positions.shape[0]:
            raise ValueError("batch must assign one graph to every atom.")
        if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
            raise ValueError("lattice must have shape (num_graphs, 3, 3).")
        inverse_lattice = torch.linalg.inv(lattice)
        return self._phases_from_inverse(positions, inverse_lattice, batch)

    def _phases_from_inverse(
        self,
        positions: torch.Tensor,
        inverse_lattice: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        atom_inverse_lattice = inverse_lattice.index_select(0, batch)
        fractional = torch.einsum(
            "ni,nij->nj",
            positions,
            atom_inverse_lattice,
        )
        modes = self.mode_indices.to(dtype=positions.dtype)
        return 2.0 * math.pi * torch.einsum("nd,kd->nk", fractional, modes)

    def geometry(
        self,
        positions: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return phases and reciprocal vectors from one batched cell inverse."""
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape (num_atoms, 3).")
        if batch.ndim != 1 or batch.shape[0] != positions.shape[0]:
            raise ValueError("batch must assign one graph to every atom.")
        if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
            raise ValueError("lattice must have shape (num_graphs, 3, 3).")

        inverse_lattice = torch.linalg.inv(lattice)
        phases = self._phases_from_inverse(positions, inverse_lattice, batch)
        k_vectors = self._reciprocal_vectors_from_inverse(inverse_lattice)
        return phases, k_vectors

    def deposit(
        self,
        node_features: torch.Tensor,
        phases: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Sum real node features into complex reciprocal modes."""
        if node_features.is_complex():
            raise TypeError("node_features must use a real dtype.")
        if node_features.ndim != 2:
            raise ValueError("node_features must have shape (num_atoms, dim).")
        if phases.shape != (node_features.size(0), self.num_modes):
            raise ValueError("phases have an incompatible shape.")

        phase = torch.complex(torch.cos(phases), -torch.sin(phases))
        features = torch.complex(node_features, torch.zeros_like(node_features))
        contributions = phase.unsqueeze(-1) * features.unsqueeze(1)
        output = scatter_sum(
            src=contributions,
            index=batch,
            dim=0,
            dim_size=num_graphs,
        )
        counts = torch.bincount(batch, minlength=num_graphs).clamp_min(1)
        normalization = counts.to(node_features.dtype).rsqrt().reshape(-1, 1, 1)
        return output * normalization

    def gather(
        self,
        spectrum: torch.Tensor,
        phases: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate a Hermitian spectrum at every atomic position."""
        if not spectrum.is_complex():
            raise TypeError("spectrum must use a complex dtype.")
        if spectrum.ndim != 3 or spectrum.size(1) != self.num_modes:
            raise ValueError("spectrum must have shape (num_graphs, K, dim).")
        if phases.shape != (batch.size(0), self.num_modes):
            raise ValueError("phases have an incompatible shape.")

        phase = torch.complex(torch.cos(phases), torch.sin(phases))
        atom_spectrum = spectrum.index_select(0, batch)
        output = torch.einsum("nkd,nk->nd", atom_spectrum, phase)
        return output.real / math.sqrt(self.num_modes)

    def hermitianize(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Project a spectrum onto the real-field Hermitian subspace."""
        partner = spectrum.index_select(1, self.negative_mode_index)
        return 0.5 * (spectrum + partner.conj())
