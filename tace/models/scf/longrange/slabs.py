# specifically for electrocatalysis, certain charge compensation schemes and dipole corrections are needed
import torch
from scipy.constants import pi

from tace.utils.torch_scatter import scatter_mean, scatter_sum

from .utils import CUBIC_MADELUNG, FIELD_CONSTANT


def get_nonperiodic_charge_dipole(
    source_feats,  # [n_node, (max_l_s+1)**2]
    node_positions,
    batch,
):
    total_charge = scatter_sum(src=source_feats[:, 0], index=batch, dim=-1)

    q_r = node_positions * source_feats[:, 0].unsqueeze(-1)  # [N_atoms,3]
    total_dipole_q = scatter_sum(src=q_r, index=batch, dim=0)
    if source_feats.shape[-1] > 1:
        total_dipole_p = scatter_sum(src=source_feats[..., 1:4], index=batch, dim=-2)
        total_dipole = total_dipole_q + total_dipole_p[..., [2, 0, 1]]
    else:
        total_dipole = total_dipole_q  # [n_graph, 3]

    return total_charge, total_dipole


def _is_batch1(batch: torch.Tensor) -> bool:
    if batch.numel() == 0:
        return True
    return int(batch.max()) == 0


def _get_total_dipole_z(
    source_feats: torch.Tensor, node_positions: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    charges = source_feats[:, 0]
    if _is_batch1(batch):
        total_dipole_z = (node_positions[:, 2] * charges).sum().unsqueeze(0)
        if source_feats.shape[-1] > 1:
            local_dipoles = source_feats[:, 1:4]
            total_dipole_z = total_dipole_z + local_dipoles.sum(dim=0)[1].unsqueeze(0)
        return total_dipole_z

    total_dipole_z = scatter_sum(src=node_positions[:, 2] * charges, index=batch, dim=0)
    if source_feats.shape[-1] > 1:
        local_dipoles = source_feats[:, 1:4]
        total_dipole_p = scatter_sum(src=local_dipoles, index=batch, dim=0)
        total_dipole_z = total_dipole_z + total_dipole_p[:, 1]
    return total_dipole_z


def slab_dipole_correction_energy(
    source_feats,
    node_positions,
    volumes,
    batch,
):
    total_dipole_z = _get_total_dipole_z(source_feats, node_positions, batch)
    A = FIELD_CONSTANT / (4 * pi)
    dipole_norms_squared = total_dipole_z**2
    delta_E = A * 2 * pi * dipole_norms_squared / volumes
    return delta_E


def slab_dipole_correction_total_field(total_dipole, volumes):
    A = FIELD_CONSTANT / (4 * pi)
    total_field_z = A * 4 * pi * total_dipole[:, 2] / volumes
    total_field = torch.zeros_like(total_dipole)
    total_field[:, 2] = total_field_z
    return total_field


def slab_dipole_correction_node_fields(
    source_feats: torch.Tensor,
    node_positions: torch.Tensor,
    volumes: torch.Tensor,
    batch: torch.Tensor,
):
    total_dipole_z = _get_total_dipole_z(source_feats, node_positions, batch)
    A = FIELD_CONSTANT / (4 * pi)
    total_field_z = A * 4 * pi * total_dipole_z / volumes
    spread_total_field_z = torch.index_select(total_field_z, 0, batch)

    delta_V_nodes = spread_total_field_z * node_positions[:, 2]
    node_fields = torch.zeros(
        (node_positions.shape[0], 4),
        dtype=node_positions.dtype,
        device=node_positions.device,
    )
    node_fields[:, 0] = delta_V_nodes
    node_fields[:, 3] = spread_total_field_z
    return node_fields


class CorrectivePotentialBlock(torch.nn.Module):
    """Implements the point charge corrective potential from
    https://journals.aps.org/prb/pdf/10.1103/PhysRevB.77.115139"""

    def __init__(
        self,
        density_max_l,
        quadrupole_feature_corrections=False,
    ):
        super().__init__()
        self.const = FIELD_CONSTANT / (4 * pi)
        self.density_max_l = density_max_l
        self.include_quadrupole_corrections = quadrupole_feature_corrections

    def forward(self, charge_coefficients, positions, volumes, batch):
        # get charge, dipole, quadrupole
        total_charge = scatter_sum(src=charge_coefficients[:, 0], index=batch, dim=-1)
        q_r = positions * charge_coefficients[:, 0].unsqueeze(-1)  # [N_atoms,3]
        total_dipole = scatter_sum(src=q_r, index=batch, dim=0)
        r_squared = torch.sum(torch.square(positions), dim=-1)
        q_rr = r_squared * charge_coefficients[:, 0]
        quadrupole = scatter_sum(src=q_rr, index=batch, dim=0)

        # extra if L>0
        if self.density_max_l > 0:
            local_dipoles_cartesian = charge_coefficients[..., [3, 1, 2]]
            total_dipole += scatter_sum(
                src=local_dipoles_cartesian, index=batch, dim=-2
            )
            positions_normed = positions / (
                torch.norm(positions, dim=-1, keepdim=True) + 1e-3
            )
            p_dot_r = torch.einsum("bi,bi->b", positions, local_dipoles_cartesian)
            quadrupole += 2 * scatter_sum(src=p_dot_r, index=batch, dim=0)

        spread_dipoles = torch.index_select(total_dipole, 0, batch)
        spread_total_charge = torch.index_select(total_charge, 0, batch)
        spread_volumes = torch.index_select(volumes, 0, batch)
        spread_total_quadrupole = torch.index_select(quadrupole, 0, batch)

        node_fields = torch.zeros(
            (positions.shape[0], 4),
            dtype=torch.get_default_dtype(),
            device=positions.device,
        )

        # L=0 piece has several terms
        Ls = torch.pow(volumes, 0.333333)
        delta_V_0 = CUBIC_MADELUNG * self.const * total_charge / Ls
        node_delta_V = torch.index_select(delta_V_0, 0, batch)
        node_delta_V += (
            -self.const
            * 2
            * pi
            * spread_total_charge
            * r_squared
            / (3 * spread_volumes)
        )
        node_delta_V += (
            self.const
            * 4
            * pi
            * torch.einsum("bi,bi->b", spread_dipoles, positions)
            / (3 * spread_volumes)
        )
        node_delta_V += (
            -self.const * 2 * pi * spread_total_quadrupole / (3 * spread_volumes)
        )
        node_fields[:, 0] = node_delta_V

        # L=1 piece
        quantity_a = spread_dipoles - spread_total_charge.unsqueeze(-1) * positions
        node_fields[:, 1:] = (
            4 * pi * self.const * quantity_a / (3 * spread_volumes.unsqueeze(-1))
        )

        return node_fields


class MonopoleDipoleCorrectionBlock(torch.nn.Module):
    r"""
    Implements corrections for evaluating clusters large boxes. See
    - https://journals.aps.org/prb/pdf/10.1103/PhysRevB.51.4014 (Makov and Payne)
    - https://journals.aps.org/prb/pdf/10.1103/PhysRevB.77.115139 (Dabo et al)

    NOTEs:
    - assumes a cubic box
    - the formulas are comparable to those by Makov and Payne, except that they generally take the
      origin to make the dipole unit 0.

    This implements corrections to the energy of an isolated system which was computed in PBC.
    The corrections are: $E = E_{pbc} + \Delta E$
        $$\Delta E = \frac{1}{2}C\alpha q^2/L + \frac{2C\pi}{3L^3} \left( |p|^2 - q\sum_i q_i r_i^2\right)$$
    where $\alpha$ is the cubic Madelung constant and $C$ is for units.
    The three terms are charge, dipole and isotropic quadrupole.
    """

    def __init__(self, density_max_l):
        super().__init__()
        self.const = FIELD_CONSTANT / (4 * pi)
        self.density_max_l = density_max_l

    def forward(
        self,
        charge_coefficients,
        positions,
        volumes,
        batch,
    ):
        # get charge
        total_charge = scatter_sum(src=charge_coefficients[:, 0], index=batch, dim=-1)
        charge_norms_squared = torch.square(total_charge)

        # get dipole
        q_r = positions * charge_coefficients[:, 0].unsqueeze(-1)  # [N_atoms,3]
        total_dipole = scatter_sum(src=q_r, index=batch, dim=0)

        # get isotropic quadrupole
        r_squared = torch.sum(torch.square(positions), dim=-1)
        q_rr = r_squared * charge_coefficients[:, 0]
        quadrupole = scatter_sum(src=q_rr, index=batch, dim=0)

        # extra if L>0
        if self.density_max_l > 0:
            local_dipoles_cartesian = charge_coefficients[..., [3, 1, 2]]
            total_dipole += scatter_sum(
                src=local_dipoles_cartesian, index=batch, dim=-2
            )
            positions_normed = positions / (
                torch.norm(positions, dim=-1, keepdim=True) + 1e-3
            )
            p_dot_r = torch.einsum("bi,bi->b", positions, local_dipoles_cartesian)
            quadrupole += 2 * scatter_sum(src=p_dot_r, index=batch, dim=0)

        # charge correction
        Ls = torch.pow(volumes, 0.3333)
        delta_E = 0.5 * CUBIC_MADELUNG * self.const * charge_norms_squared / Ls

        # dipole correction
        delta_E += (
            2
            * self.const
            * pi
            * torch.sum(torch.square(total_dipole), dim=-1)
            / (3 * volumes)
        )

        # quadrupole correction
        delta_E += -2 * self.const * pi * total_charge * quadrupole / (3 * volumes)

        return delta_E
