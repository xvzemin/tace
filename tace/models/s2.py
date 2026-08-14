###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import copy
import logging
import math
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from e3nn import o3

from .o2 import CoefficientMappingModule


class FibonacciLattice:
    @staticmethod
    def generate(num_points: int) -> torch.Tensor:
        assert isinstance(num_points, int) and num_points > 0, (
            "num_points musst should be int and > 0"
        )
        GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))
        i = torch.arange(num_points, dtype=torch.float32)
        z = 1 - 2 * i / (num_points - 1) if num_points > 1 else torch.zeros(1)
        radius = torch.sqrt(1 - z * z)
        theta = GOLDEN_ANGLE * i
        x = torch.cos(theta) * radius
        y = torch.sin(theta) * radius
        points = torch.stack([x, y, z], dim=1)
        return points

    @staticmethod
    def visualize(
        num_points: int,
        show: bool = True,
        save_path: Union[str, None] = None,
        point_size: int = 10,
        dpi: int = 300,
    ) -> None:

        points = FibonacciLattice.generate(num_points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
        u, v = np.meshgrid(u, v)
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            xs,
            ys,
            zs,
            rstride=5,
            cstride=5,
            color="lightgray",
            alpha=0.2,
            edgecolor="none",
        )

        ax.scatter(x, y, z, s=point_size, c=z, cmap="viridis", alpha=0.8)

        ax.set_title(f"Fibonacci Lattice on Sphere (num_points={num_points})")
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi)
            logging.info(f"Image save to : {save_path}")

        if show:
            plt.show()

        if not show and save_path is None:
            plt.close(fig)


class HealpixLattice:
    @staticmethod
    def generate(num_points: int) -> torch.Tensor:
        try:
            import healpy as hp
        except ImportError as e:
            raise ImportError(
                "Failed to import healpy. Please make sure it is installed correctly. You can install it via 'conda install -c conda-forge healpy'."
            ) from e

        assert isinstance(num_points, int) and num_points > 0, (
            "num_points musst should be int and > 0"
        )
        nside = int((num_points / 12) ** 0.5)

        if 12 * nside**2 != num_points:
            closest = 12 * nside**2
            raise ValueError(
                f"num_points must be 12 times N², e.g., 12, 48, 192, etc. The closest valid value is {closest}."
            )

        npix = 12 * nside**2
        theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)

        x = torch.from_numpy(np.sin(theta) * np.cos(phi)).float()
        y = torch.from_numpy(np.sin(theta) * np.sin(phi)).float()
        z = torch.from_numpy(np.cos(theta)).float()

        return torch.stack([x, y, z], dim=1)

    @staticmethod
    def visualize(
        num_points: int,
        show: bool = True,
        save_path: Union[str, None] = None,
        point_size: int = 10,
    ):
        points = HealpixLattice.generate(num_points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Spherical reference grid
        u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
        u, v = np.meshgrid(u, v)
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            xs,
            ys,
            zs,
            rstride=5,
            cstride=5,
            color="lightgray",
            alpha=0.2,
            edgecolor="none",
        )

        ax.scatter(x, y, z, s=point_size, c=z, cmap="coolwarm", alpha=0.8)

        ax.set_title(f"Healpix Lattice on Sphere (num_points={num_points})")
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            logging.info(f"Image save to: {save_path}")

        if show:
            plt.show()

        if not show and save_path is None:
            plt.close(fig)


class PlatonicSolidLattice:
    @staticmethod
    def generate(num_points: int) -> torch.Tensor:
        # The initial vertex of a regular icosahedron
        phi = (1 + math.sqrt(5)) / 2
        vertices = torch.tensor(
            [
                [0, 1, phi],
                [0, -1, phi],
                [0, 1, -phi],
                [0, -1, -phi],
                [1, phi, 0],
                [-1, phi, 0],
                [1, -phi, 0],
                [-1, -phi, 0],
                [phi, 0, 1],
                [-phi, 0, 1],
                [phi, 0, -1],
                [-phi, 0, -1],
            ],
            dtype=torch.float32,
        )

        # Unitization: Projection onto the unit sphere
        vertices /= torch.norm(vertices, dim=1, keepdim=True)

        # Calculate the number of subdivisions k
        k = 0
        while 2 + 10 * (4**k) < num_points:
            k += 1
        if 2 + 10 * (4**k) != num_points:
            closest = 2 + 10 * (4**k)
            raise ValueError(
                f"num_points must be in the form of 2 + 10 times 4^k, e.g., 12, 42, 162, etc. The closest valid value is {closest}."
            )

        # Triangular face: 12 vertices correspond to 20 triangles
        faces = torch.tensor(
            [
                [0, 11, 5],
                [0, 5, 1],
                [0, 1, 7],
                [0, 7, 10],
                [0, 10, 11],
                [1, 5, 9],
                [5, 11, 4],
                [11, 10, 2],
                [10, 7, 6],
                [7, 1, 8],
                [3, 9, 4],
                [3, 4, 2],
                [3, 2, 6],
                [3, 6, 8],
                [3, 8, 9],
                [4, 9, 5],
                [2, 4, 11],
                [6, 2, 10],
                [8, 6, 7],
                [9, 8, 1],
            ],
            dtype=torch.int64,
        )

        # Recursive Subdivision
        for _ in range(k):
            faces, vertices = PlatonicSolidLattice.subdivide(faces, vertices)

        return vertices

    @staticmethod
    def subdivide(faces: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
        # Vertices before subdivision
        new_faces = []
        new_vertices = vertices.clone().detach()  # Initialize new_vertices

        # Subdivision step: take midpoints of each edge and normalize
        edge_map = {}

        def get_midpoint(v1, v2, new_vertices):
            # Calculate the midpoint of the edge and normalize
            edge = (vertices[v1] + vertices[v2]) / 2
            norm = torch.norm(edge)
            midpoint = edge / norm
            # Return a unique index to avoid duplication
            if (v1, v2) in edge_map or (v2, v1) in edge_map:
                return edge_map.get((v1, v2)) or edge_map.get((v2, v1))
            new_vertex_index = len(new_vertices)
            edge_map[(v1, v2)] = new_vertex_index
            edge_map[(v2, v1)] = new_vertex_index
            new_vertices = torch.cat(
                [new_vertices, midpoint.view(1, -1)], dim=0
            )  # Use torch.cat() to add new vertex
            return new_vertex_index, new_vertices

        # Subdivide all faces
        for face in faces:
            v0, v1, v2 = face
            # Midpoint indices
            m0, new_vertices = get_midpoint(v0, v1, new_vertices)
            m1, new_vertices = get_midpoint(v1, v2, new_vertices)
            m2, new_vertices = get_midpoint(v2, v0, new_vertices)
            # New subdivided faces
            new_faces.append([v0, m0, m2])
            new_faces.append([v1, m1, m0])
            new_faces.append([v2, m2, m1])
            new_faces.append([m0, m1, m2])

        return torch.tensor(new_faces, dtype=torch.int64), new_vertices

    @staticmethod
    def visualize(
        num_points: int,
        show: bool = True,
        save_path: Union[str, None] = None,
        point_size: int = 10,
    ):
        points = PlatonicSolidLattice.generate(num_points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
        u, v = np.meshgrid(u, v)
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            xs,
            ys,
            zs,
            rstride=5,
            cstride=5,
            color="lightgray",
            alpha=0.2,
            edgecolor="none",
        )

        ax.scatter(x, y, z, s=point_size, c=z, cmap="coolwarm", alpha=0.8)
        ax.set_title(f"Platonic Solid Lattice on Sphere (num_points={num_points})")
        ax.set_box_aspect([1, 1, 1])

        ax.set_axis_off()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            logging.info(f"Image saved to: {save_path}")

        if show:
            plt.show()

        if not show and save_path is None:
            plt.close(fig)


class RandomSampling:
    @staticmethod
    def generate(num_points: int) -> torch.Tensor:
        phi = torch.rand(num_points) * 2 * math.pi
        theta = torch.acos(2 * torch.rand(num_points) - 1)

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        points = torch.stack([x, y, z], dim=1)
        return points

    @staticmethod
    def visualize(num_points: int, show: bool = True, save_path: str = None):
        points = RandomSampling.generate(num_points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        u, v = torch.meshgrid(
            torch.linspace(0, 2 * math.pi, 100), torch.linspace(0, math.pi, 100)
        )
        xs = torch.cos(u) * torch.sin(v)
        ys = torch.sin(u) * torch.sin(v)
        zs = torch.cos(v)

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            xs.numpy(),
            ys.numpy(),
            zs.numpy(),
            rstride=5,
            cstride=5,
            color="lightgray",
            alpha=0.2,
            edgecolor="none",
        )

        ax.scatter(
            x.numpy(),
            y.numpy(),
            z.numpy(),
            s=10,
            c=z.numpy(),
            cmap="viridis",
            alpha=0.8,
        )

        ax.set_title(f"Random Sampling on Sphere (n={num_points})")
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Image saved to: {save_path}")

        if show:
            plt.show()


class SO3Grid(torch.nn.Module):
    """
    Helper functions for grid representation of the irreps

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
        normalization (str):    Default: 'component'
                                How grid samples are normalized.
        resolution_list (list:int):
                                Default: None
                                List of grid resolutions corresponding to `lat_resolution` and `long_resolution`.
                                Set to `None` to use default resolutions.
        use_m_primary (bool):   Default: False
                                Whether to change the layout of m components.
                                If `False`, the layout of m is (0), (-1, 0, +1), (-2, -1, 0, +1, +2), ...
                                If `True`, the layout of m is (0, 0, ...), (1, 1, ...), ...
                                The second one is used in SO(2) linear operations to avoid redundant
                                matrix multiplications.
    """

    def __init__(
        self,
        lmax,
        mmax,
        normalization="component",
        resolution_list=None,
        use_m_primary=False,
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_m_primary = use_m_primary
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution_list is not None:
            assert isinstance(resolution_list, list)
            resolution_list = copy.deepcopy(resolution_list)
            self.lat_resolution = resolution_list[0]
            self.long_resolution = resolution_list[1]

        mapping = CoefficientMappingModule(
            lmax=self.lmax, mmax=self.lmax, use_rotate_inv_rescale=False
        )

        to_grid = o3.ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,  # normalization="integral",
            device="cpu",
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l**2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    to_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
                )
        to_grid_mat = to_grid_mat[:, :, mapping.coefficient_idx(self.lmax, self.mmax)]

        from_grid = o3.FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,  # normalization="integral",
            device="cpu",
        )
        from_grid_mat = torch.einsum(
            "am, mbi -> bai", from_grid.sha, from_grid.shb
        ).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l**2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    from_grid_mat[:, :, start_idx : (start_idx + length)]
                    * rescale_factor
                )
        from_grid_mat = from_grid_mat[
            :, :, mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        # flatten and permute
        to_grid_mat = to_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.permute(1, 0)

        # change the layout of m components
        if self.use_m_primary:
            temp = CoefficientMappingModule(self.lmax, self.mmax, False)
            to_grid_mat = torch.einsum("ai, ji -> aj", to_grid_mat, temp.to_m)
            from_grid_mat = torch.einsum("ia, ji -> ja", from_grid_mat, temp.to_m)

        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)

    def get_to_grid_mat(self):
        return self.to_grid_mat

    def get_from_grid_mat(self):
        return self.from_grid_mat

    def to_grid(self, embedding):
        grid = torch.einsum("aj, njc -> nac", self.to_grid_mat, embedding)
        return grid

    def from_grid(self, grid):
        embedding = torch.einsum("ja, nac -> njc", self.from_grid_mat, grid)
        return embedding

    def extra_repr(self):
        return "lmax={}, mmax={}, lat_resolution={}, long_resolution={}, use_m_primary={}".format(
            self.lmax,
            self.mmax,
            self.lat_resolution,
            self.long_resolution,
            self.use_m_primary,
        )


class SO3VstpGrid(torch.nn.Module):
    """
    Using the same grid convention as e3nn ToS2Grid / FromS2Grid:
        beta_i  = pi * (i + 0.5) / res_beta
        alpha_j = 2pi * j / res_alpha
    """

    def __init__(
        self,
        lmax,
        mmax=None,
        normalization="component",
        resolution_list=None,
        use_m_primary=False,
        eps=1e-8,
        normalize_products=True,
    ):
        super().__init__()

        self.lmax = lmax
        self.mmax = lmax if mmax is None else mmax
        self.normalization = normalization
        self.use_m_primary = use_m_primary
        self.eps = eps
        self.normalize_products = normalize_products

        self.lat_resolution = 2 * (self.lmax + 1)

        if self.lmax == self.mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * self.mmax + 1

        if resolution_list is not None:
            assert isinstance(resolution_list, list)
            resolution_list = copy.deepcopy(resolution_list)
            self.lat_resolution = resolution_list[0]
            self.long_resolution = resolution_list[1]

        mapping = CoefficientMappingModule(
            lmax=self.lmax,
            mmax=self.lmax,
            use_rotate_inv_rescale=False,
        )

        to_grid = o3.ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,
        )

        from_grid = o3.FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,
        )

        to_grid_mat = torch.einsum(
            "mbi,am->bai",
            to_grid.shb,
            to_grid.sha,
        ).detach()

        from_grid_mat = torch.einsum(
            "am,mbi->bai",
            from_grid.sha,
            from_grid.shb,
        ).detach()

        d_beta_mat, d_alpha_mat, sin_beta = self._build_derivative_matrices(to_grid)

        if self.lmax != self.mmax:
            for l in range(self.lmax + 1):
                if l <= self.mmax:
                    continue

                start_idx = l**2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * self.mmax + 1))

                to_grid_mat[:, :, start_idx : start_idx + length] *= rescale_factor
                from_grid_mat[:, :, start_idx : start_idx + length] *= rescale_factor
                d_beta_mat[:, :, start_idx : start_idx + length] *= rescale_factor
                d_alpha_mat[:, :, start_idx : start_idx + length] *= rescale_factor

        coeff_idx = mapping.coefficient_idx(
            self.lmax,
            self.mmax,
        )
        coeff_l_index = mapping.l_harmonic[coeff_idx].clone()

        to_grid_mat = to_grid_mat[:, :, coeff_idx]
        from_grid_mat = from_grid_mat[:, :, coeff_idx]
        d_beta_mat = d_beta_mat[:, :, coeff_idx]
        d_alpha_mat = d_alpha_mat[:, :, coeff_idx]

        to_grid_mat = to_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.flatten(0, 1).permute(1, 0)

        d_beta_mat = d_beta_mat.flatten(0, 1)
        d_alpha_mat = d_alpha_mat.flatten(0, 1)

        sin_beta = sin_beta[:, None].expand(
            self.lat_resolution,
            self.long_resolution,
        )
        sin_beta = sin_beta.flatten()

        if self.use_m_primary:
            temp = CoefficientMappingModule(
                self.lmax,
                self.mmax,
                False,
            )
            source_idx = temp.to_m.argmax(dim=1)
            coeff_l_index = coeff_l_index[source_idx]

            to_grid_mat = torch.einsum(
                "ai,ji->aj",
                to_grid_mat,
                temp.to_m,
            )

            from_grid_mat = torch.einsum(
                "ia,ji->ja",
                from_grid_mat,
                temp.to_m,
            )

            d_beta_mat = torch.einsum(
                "ai,ji->aj",
                d_beta_mat,
                temp.to_m,
            )

            d_alpha_mat = torch.einsum(
                "ai,ji->aj",
                d_alpha_mat,
                temp.to_m,
            )

        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)

        self.register_buffer("d_beta_to_grid_mat", d_beta_mat)
        self.register_buffer("d_alpha_to_grid_mat", d_alpha_mat)

        self.register_buffer(
            "inv_sin_beta",
            1.0 / sin_beta.clamp_min(self.eps),
        )
        self.register_buffer("coeff_l_index", coeff_l_index.long())

        if self.normalize_products:
            sym_norm, anti_norm = self._build_product_normalizers(
                to_grid_mat,
                from_grid_mat,
                d_beta_mat,
                d_alpha_mat,
                self.inv_sin_beta,
                self.coeff_l_index,
            )
        else:
            shape = (1, from_grid_mat.size(0), 1)
            sym_norm = torch.ones(shape, dtype=from_grid_mat.dtype)
            anti_norm = torch.ones(shape, dtype=from_grid_mat.dtype)

        self.register_buffer("symmetric_output_norm", sym_norm)
        self.register_buffer("antisymmetric_output_norm", anti_norm)
        self.register_buffer(
            "d_beta_grid_mat",
            torch.einsum("ai,ib->ab", d_beta_mat, from_grid_mat),
        )
        self.register_buffer(
            "d_alpha_grid_mat",
            torch.einsum("ai,ib->ab", d_alpha_mat, from_grid_mat),
        )
        self.register_buffer(
            "antisymmetric_grid_norm",
            self._build_grid_antisymmetric_normalizer(
                self.to_grid_mat,
                self.d_beta_grid_mat,
                self.d_alpha_grid_mat,
                self.inv_sin_beta,
            ),
        )

    def _build_derivative_matrices(
        self,
        to_grid,
    ):
        """
        Build analytic spectral derivative matrices:
            d_beta_mat[b, a, i]  = ∂beta Y_i(beta_b, alpha_a)
            d_alpha_mat[b, a, i] = ∂alpha Y_i(beta_b, alpha_a)
        This uses autograd on the analytic spherical harmonics once during
        module construction.
        """

        betas = to_grid.betas.detach().clone()
        alphas = to_grid.alphas.detach().clone()

        beta_grid, alpha_grid = torch.meshgrid(
            betas,
            alphas,
            indexing="ij",
        )

        beta_flat = beta_grid.reshape(-1).detach().clone()
        alpha_flat = alpha_grid.reshape(-1).detach().clone()

        beta_flat.requires_grad_(True)
        alpha_flat.requires_grad_(True)

        xyz = o3.angles_to_xyz(
            alpha_flat,
            beta_flat,
        )

        Y = o3.spherical_harmonics(
            list(range(self.lmax + 1)),
            xyz,
            normalize=True,
            normalization=self.normalization,
        )

        num_grid, num_coeffs = Y.shape

        d_beta_cols = []
        d_alpha_cols = []

        for i in range(num_coeffs):
            grad_beta, grad_alpha = torch.autograd.grad(
                Y[:, i].sum(),
                [beta_flat, alpha_flat],
                retain_graph=True,
                create_graph=False,
            )

            d_beta_cols.append(grad_beta)
            d_alpha_cols.append(grad_alpha)

        d_beta = torch.stack(
            d_beta_cols,
            dim=1,
        )

        d_alpha = torch.stack(
            d_alpha_cols,
            dim=1,
        )

        d_beta = d_beta.reshape(
            self.lat_resolution,
            self.long_resolution,
            num_coeffs,
        )

        d_alpha = d_alpha.reshape(
            self.lat_resolution,
            self.long_resolution,
            num_coeffs,
        )

        sin_beta = torch.sin(betas)

        return (
            d_beta.detach(),
            d_alpha.detach(),
            sin_beta.detach(),
        )

    def _degree_average_norm(
        self,
        variance,
        coeff_l_index,
    ):
        norm = torch.zeros_like(variance)
        for l in range(self.lmax + 1):
            mask = coeff_l_index == l
            if not bool(mask.any()):
                continue
            degree_var = variance[mask].mean().clamp_min(0.0)
            if degree_var.item() > self.eps:
                norm[mask] = torch.rsqrt(degree_var)
        return norm.view(1, -1, 1)

    def _build_product_normalizers(
        self,
        to_grid_mat,
        from_grid_mat,
        d_beta_mat,
        d_alpha_mat,
        inv_sin_beta,
        coeff_l_index,
    ):
        """
        Estimate per-degree output normalizers for random unit-variance inputs.

        This keeps the integral GTP and VSTP products close to CGTP scale at
        initialization without introducing dense path-wise factors that would
        destroy the factorized grid implementation.
        """
        to_grid_mat = to_grid_mat.double()
        from_grid_mat = from_grid_mat.double()
        d_beta_mat = d_beta_mat.double()
        d_alpha_mat = d_alpha_mat.double()
        inv_sin_beta = inv_sin_beta.double()

        signal_cov = to_grid_mat @ to_grid_mat.T
        sym_grid_cov = signal_cov.square()
        sym_var = torch.einsum(
            "ia,ib,ab->i",
            from_grid_mat,
            from_grid_mat,
            sym_grid_cov,
        ).clamp_min(0.0)

        beta_beta = d_beta_mat @ d_beta_mat.T
        alpha_alpha = d_alpha_mat @ d_alpha_mat.T
        beta_alpha = d_beta_mat @ d_alpha_mat.T
        alpha_beta = d_alpha_mat @ d_beta_mat.T

        anti_grid_cov = (
            beta_beta * alpha_alpha
            + alpha_alpha * beta_beta
            - beta_alpha * alpha_beta
            - alpha_beta * beta_alpha
        )
        anti_grid_cov = anti_grid_cov * inv_sin_beta[:, None] * inv_sin_beta[None, :]
        anti_var = torch.einsum(
            "ia,ib,ab->i",
            from_grid_mat,
            from_grid_mat,
            anti_grid_cov,
        ).clamp_min(0.0)

        sym_norm = self._degree_average_norm(sym_var, coeff_l_index)
        anti_norm = self._degree_average_norm(anti_var, coeff_l_index)
        return sym_norm.float(), anti_norm.float()

    def _build_grid_antisymmetric_normalizer(
        self,
        to_grid_mat,
        d_beta_grid_mat,
        d_alpha_grid_mat,
        inv_sin_beta,
    ):
        to_grid_mat = to_grid_mat.double()
        d_beta_grid_mat = d_beta_grid_mat.double()
        d_alpha_grid_mat = d_alpha_grid_mat.double()
        inv_sin_beta = inv_sin_beta.double()

        signal_cov = to_grid_mat @ to_grid_mat.T
        sym_variance = signal_cov.diagonal().square().mean().clamp_min(0.0)

        beta_cov = d_beta_grid_mat @ signal_cov @ d_beta_grid_mat.T
        alpha_cov = d_alpha_grid_mat @ signal_cov @ d_alpha_grid_mat.T
        beta_alpha_cov = d_beta_grid_mat @ signal_cov @ d_alpha_grid_mat.T

        anti_variance = 2.0 * (
            beta_cov.diagonal() * alpha_cov.diagonal()
            - beta_alpha_cov.diagonal().square()
        )
        anti_variance = anti_variance * inv_sin_beta.square()
        anti_variance = anti_variance.mean().clamp_min(0.0)

        if anti_variance.item() <= self.eps:
            return torch.tensor(0.0, dtype=torch.get_default_dtype())
        return torch.sqrt(sym_variance / anti_variance).to(
            dtype=torch.get_default_dtype()
        )

    def get_to_grid_mat(self):
        return self.to_grid_mat

    def get_from_grid_mat(self):
        return self.from_grid_mat

    def to_grid(self, embedding):
        """
        embedding:
            [num_nodes, num_coefficients, channels]

        return:
            [num_nodes, num_grid_points, channels]
        """
        return torch.matmul(self.to_grid_mat.to(dtype=embedding.dtype), embedding)

    def from_grid(self, grid):
        """
        grid:
            [num_nodes, num_grid_points, channels]

        return:
            [num_nodes, num_coefficients, channels]
        """
        return torch.matmul(self.from_grid_mat.to(dtype=grid.dtype), grid)

    def d_beta_to_grid(self, embedding):
        """
        Compute ∂beta f
        """
        return torch.matmul(
            self.d_beta_to_grid_mat.to(dtype=embedding.dtype), embedding
        )

    def d_alpha_to_grid(self, embedding):
        """
        Compute ∂alpha f
        """
        return torch.matmul(
            self.d_alpha_to_grid_mat.to(dtype=embedding.dtype), embedding
        )

    def d_beta_grid(self, grid):
        return torch.matmul(self.d_beta_grid_mat.to(dtype=grid.dtype), grid)

    def d_alpha_grid(self, grid):
        return torch.matmul(self.d_alpha_grid_mat.to(dtype=grid.dtype), grid)

    def symmetric_grid_product(self, x_grid, y_grid):
        return x_grid * y_grid

    def antisymmetric_grid_product(self, x_grid, y_grid):
        x_beta = self.d_beta_grid(x_grid)
        x_alpha = self.d_alpha_grid(x_grid)
        y_beta = self.d_beta_grid(y_grid)
        y_alpha = self.d_alpha_grid(y_grid)
        inv_sin_beta = self.inv_sin_beta.to(x_grid.dtype).view(1, -1, 1)
        bracket_grid = (x_beta * y_alpha - x_alpha * y_beta) * inv_sin_beta
        return bracket_grid * self.antisymmetric_grid_norm.to(dtype=bracket_grid.dtype)

    def full_grid_product(
        self,
        x_grid,
        y_grid,
        symmetric_scale=1.0,
        antisym_scale=1.0,
    ):
        sym = self.symmetric_grid_product(x_grid, y_grid)
        anti = self.antisymmetric_grid_product(x_grid, y_grid)
        return sym + anti

    def symmetric_product(self, x, y):
        """
        symmetric GTP
        """
        x_grid = self.to_grid(x)
        y_grid = self.to_grid(y)

        product_grid = x_grid * y_grid

        out = self.from_grid(product_grid)
        return out * self.symmetric_output_norm.to(dtype=out.dtype)

    def antisymmetric_product(self, x, y):
        """
        Antisymmetric VSTP
        """
        x_beta = self.d_beta_to_grid(x)
        x_alpha = self.d_alpha_to_grid(x)

        y_beta = self.d_beta_to_grid(y)
        y_alpha = self.d_alpha_to_grid(y)

        inv_sin_beta = self.inv_sin_beta.to(x.dtype).view(1, -1, 1)

        bracket_grid = (x_beta * y_alpha - x_alpha * y_beta) * inv_sin_beta

        out = self.from_grid(bracket_grid)
        return out * self.antisymmetric_output_norm.to(dtype=out.dtype)

    def full_vstp_product(
        self,
        x,
        y,
        symmetric_scale=1.0,
        antisym_scale=1.0,
    ):
        """
        Theorem 2: GTP + VSTP
        """
        sym = self.symmetric_product(x, y)
        anti = self.antisymmetric_product(x, y)

        return sym + anti  # TODO

        # return symmetric_scale * sym + antisym_scale * anti

    def extra_repr(self):
        return (
            f"lmax={self.lmax}, "
            f"mmax={self.mmax}, "
            f"lat_resolution={self.lat_resolution}, "
            f"long_resolution={self.long_resolution}, "
            f"use_m_primary={self.use_m_primary}"
        )
