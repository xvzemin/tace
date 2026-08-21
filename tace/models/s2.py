###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
import math
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch


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
