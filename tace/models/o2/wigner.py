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


class CoefficientMappingModule(torch.nn.Module):
    """
    Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

    Args:
        lmax (int):             Maximum degree of the spherical harmonics
        mmax (int):             Maximum order of the spherical harmonics
        use_rotate_inv_rescale (bool):
                                Whether to pre-compute inverse rotation rescale matrices
    """

    def __init__(self, lmax, mmax, use_rotate_inv_rescale=False):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.use_rotate_inv_rescale = use_rotate_inv_rescale

        m_complex = []  # this m belongs to which SO(3) m
        l_harmonic = []  # this m belongs to which SO(3) l
        m_harmonic = []  # this m belongs to which SO(2) m

        for l in range(0, self.lmax + 1):
            mmax = min(self.mmax, l)
            m = torch.arange(-mmax, mmax + 1).long()
            m_complex.append(m)
            m_harmonic.append(torch.abs(m).long())
            l_harmonic.append(torch.fill(m, l))
        m_complex = torch.cat(
            m_complex, dim=0
        )  # tensor([0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3])
        m_harmonic = torch.cat(
            m_harmonic, dim=0
        )  # tensor([0, 1, 0, 1, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3])
        l_harmonic = torch.cat(
            l_harmonic, dim=0
        )  # tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])

        num_components = len(l_harmonic)
        to_m = torch.zeros([num_components, num_components])

        offset = 0
        for m in range(self.mmax + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)
            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        to_m = to_m.detach()

        self.register_buffer("l_harmonic", l_harmonic)
        self.register_buffer("m_harmonic", m_harmonic)
        self.register_buffer("m_complex", m_complex)
        self.register_buffer("to_m", to_m)

        # for `torch.compile()` compatibility
        self.pre_compute_coefficient_idx()
        if self.use_rotate_inv_rescale:
            self.pre_compute_rotate_inv_rescale()

    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        if lmax == -1:
            lmax = self.lmax

        indices = torch.arange(len(l_harmonic))
        mask_r = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([]).long()
        if m != 0:
            mask_i = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    def pre_compute_coefficient_idx(self):
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask = torch.bitwise_and(self.l_harmonic.le(l), self.m_harmonic.le(m))
                indices = torch.arange(len(mask))
                mask_indices = torch.masked_select(indices, mask)
                self.register_buffer(
                    "coefficient_idx_l{}_m{}".format(l, m), mask_indices
                )
        return

    def prepare_coefficient_idx(self) -> list[list[torch.Tensor]]:
        # idx = lmax, mmax
        coefficient_idx_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(
                    getattr(self, "coefficient_idx_l{}_m{}".format(l, m), None)
                )
            coefficient_idx_list.append(l_list)
        return coefficient_idx_list

    def coefficient_idx(self, lmax, mmax):
        if lmax > self.lmax or mmax > self.lmax:
            mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
            indices = torch.arange(len(mask), device=mask.device)
            mask_indices = torch.masked_select(indices, mask)
            return mask_indices
        else:
            temp = self.prepare_coefficient_idx()
            return temp[lmax][mmax]

    def pre_compute_rotate_inv_rescale(self):
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask_indices = self.coefficient_idx(l, m)
                rotate_inv_rescale = torch.ones(
                    (1, int((l + 1) ** 2), int((l + 1) ** 2))
                )
                for l_sub in range(l + 1):
                    if l_sub <= m:
                        continue
                    start_idx = l_sub**2
                    length = 2 * l_sub + 1
                    rescale_factor = math.sqrt(length / (2 * m + 1))
                    rotate_inv_rescale[
                        :,
                        start_idx : (start_idx + length),
                        start_idx : (start_idx + length),
                    ] = rescale_factor
                rotate_inv_rescale = rotate_inv_rescale[:, :, mask_indices]
                self.register_buffer(
                    "rotate_inv_rescale_l{}_m{}".format(l, m), rotate_inv_rescale
                )
        return

    def prepare_rotate_inv_rescale(self):
        rotate_inv_rescale_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(
                    getattr(self, "rotate_inv_rescale_l{}_m{}".format(l, m), None)
                )
            rotate_inv_rescale_list.append(l_list)
        return rotate_inv_rescale_list

    def get_rotate_inv_rescale(self, lmax, mmax):
        temp = self.prepare_rotate_inv_rescale()
        return temp[lmax][mmax]

    def __repr__(self):
        return f"{self.__class__.__name__}(mmax={self.mmax}, lmax={self.lmax})"


class WignerD(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        use_opt_einsum_fx: bool = False,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.use_opt_einsum_fx = use_opt_einsum_fx

        for l in range(2, self.lmax + 1):
            self.register_buffer(f"CG_{l}", o3.wigner_3j(1, l - 1, l), persistent=False)
            if self.use_opt_einsum_fx:
                self._register_fx(l)

        mapping = CoefficientMappingModule(
            lmax=self.lmax, mmax=self.lmax, use_rotate_inv_rescale=True
        )
        wigner_index_mask = mapping.coefficient_idx(self.lmax, self.mmax)
        wigner_inv_rescale = mapping.get_rotate_inv_rescale(self.lmax, self.mmax)
        # Merge converting m and l layout
        mapping = CoefficientMappingModule(
            lmax=self.lmax, mmax=self.mmax, use_rotate_inv_rescale=False
        )
        to_m = mapping.to_m
        wigner_inv_rescale = torch.einsum("nia, ba -> nib", wigner_inv_rescale, to_m)
        wigner_index_to_m_array = torch.zeros(to_m.shape[0], ((self.lmax + 1) ** 2))
        # to_m [14, 14]
        # wigner_index_mask [14], tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14])
        wigner_index_to_m_array[:, wigner_index_mask] = to_m

        self.register_buffer("wigner_index_to_m_array", wigner_index_to_m_array)
        self.register_buffer("wigner_inv_rescale", wigner_inv_rescale)  # [1, 16, 14]

    def get_wigner(self, edge_vector) -> tuple[torch.Tensor]:
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
