################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
If you use wignerD in this file, please cite our paper
"""

import math

import opt_einsum_fx
import torch
from e3nn import o3

from .rotation_matrix import init_edge_rot_mat_quaternion

_BATCH = 10000


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
        and above. This may reduce repeated eager execution cost at the expense
        of additional module setup.

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
    ):
        super().__init__()

        if isinstance(lmax, bool) or not isinstance(lmax, int):
            raise TypeError("lmax must be an integer.")
        if isinstance(mmax, bool) or not isinstance(mmax, int):
            raise TypeError("mmax must be an integer.")
        if lmax < 0:
            raise ValueError("lmax must be non-negative.")
        if not 0 <= mmax <= lmax:
            raise ValueError("mmax must satisfy 0 <= mmax <= lmax.")

        self.mmax = mmax
        self.lmax = lmax
        self.use_opt_einsum_fx = use_opt_einsum_fx

        for l in range(2, self.lmax + 1):
            self.register_buffer(f"CG_{l}", o3.wigner_3j(1, l - 1, l), persistent=False)
            if self.use_opt_einsum_fx:
                self._register_fx(l)

        wigner_index_to_m_array, wigner_inv_rescale = self._build_o2_layout(
            self.lmax, self.mmax
        )

        self.register_buffer("wigner_index_to_m_array", wigner_index_to_m_array)
        self.register_buffer("wigner_inv_rescale", wigner_inv_rescale)  # [1, 16, 14]

    @staticmethod
    def _build_o2_layout(lmax: int, mmax: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the truncated, order-major O(2) layout and inverse scaling."""
        global_indices = []
        local_degrees = []
        for order in range(mmax + 1):
            signed_orders = (0,) if order == 0 else (order, -order)
            for signed_order in signed_orders:
                for degree in range(order, lmax + 1):
                    global_indices.append(degree**2 + degree + signed_order)
                    local_degrees.append(degree)

        full_dim = (lmax + 1) ** 2
        local_dim = len(global_indices)
        to_m = torch.zeros(local_dim, full_dim)
        rows = torch.arange(local_dim)
        columns = torch.tensor(global_indices, dtype=torch.int64)
        to_m[rows, columns] = 1.0

        inverse_scale = torch.ones(1, full_dim, local_dim)
        for degree in range(mmax + 1, lmax + 1):
            scale = math.sqrt((2 * degree + 1) / (2 * mmax + 1))
            local_indices = [
                index
                for index, local_degree in enumerate(local_degrees)
                if local_degree == degree
            ]
            inverse_scale[
                :, degree**2 : (degree + 1) ** 2, local_indices
            ] = scale
        return to_m, inverse_scale

    def get_wigner(self, edge_vector) -> tuple[torch.Tensor]:
        """Build both rotation directions for a batch of vectors.

        Parameters
        ----------
        edge_vector : torch.Tensor
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
        rot_mat3x3 = init_edge_rot_mat_quaternion(edge_vector)
        wigner = self._rotation_to_wigner_matrix_recursive(
            rot_mat3x3,
            0,
            self.lmax,
        )
        wigner = torch.einsum(
            "mi, nij -> nmj", self.wigner_index_to_m_array, wigner
        )  # [14, 16] @ [16, 16]
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        wigner_inv = wigner_inv * self.wigner_inv_rescale
        return wigner, wigner_inv

    def _register_fx(self, degree: int) -> None:
        expr = "abm,eac,ebd,cdn->emn"
        ctr = torch.fx.symbolic_trace(
            lambda d1, d_prev, cg: torch.einsum(expr, cg, d1, d_prev, cg)
        )
        ctr = opt_einsum_fx.optimize_einsums_full(
            model=ctr,
            example_inputs=(
                torch.randn(_BATCH, 3, 3),
                torch.randn(
                    _BATCH,
                    2 * degree - 1,
                    2 * degree - 1,
                ),
                torch.randn(3, 2 * degree - 1, 2 * degree + 1),
            ),
        )
        self.add_module(f"fx_{degree}", ctr)

    def _compute_one_wigner(
        self,
        degree: int,
        d1: torch.Tensor,
        d_prev: torch.Tensor,
        cg: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_opt_einsum_fx:
            ctr = getattr(self, f"fx_{degree}")
            return ctr(d1, d_prev, cg)
        left = torch.einsum("abm,eac->ebmc", cg, d1)
        left = torch.einsum("ebmc,ebd->emcd", left, d_prev)
        return torch.einsum("emcd,cdn->emn", left, cg)

    def _rotation_to_wigner_matrix_recursive(
        self,
        edge_rot_mat: torch.Tensor,
        start_lmax: int,
        end_lmax: int,
    ) -> torch.Tensor:
        batch = edge_rot_mat.shape[0]
        all_blocks = [edge_rot_mat.new_ones(batch, 1, 1)]
        if end_lmax >= 1:
            all_blocks.append(edge_rot_mat)
        for degree in range(2, end_lmax + 1):
            cg = getattr(self, f"CG_{degree}")
            block = self._compute_one_wigner(
                degree,
                all_blocks[1],
                all_blocks[degree - 1],
                cg,
            )
            all_blocks.append(block * (2 * degree + 1))
        blocks = all_blocks[start_lmax : end_lmax + 1]
        size = int((end_lmax + 1) ** 2) - int(start_lmax**2)
        wigner = edge_rot_mat.new_zeros(batch, size, size)
        offset = 0
        for block in blocks:
            width = block.shape[-1]
            wigner[:, offset : offset + width, offset : offset + width] = block
            offset += width
        return wigner

    def extra_repr(self):
        return "mmax={}, lmax={}".format(self.mmax, self.lmax)
