"""Equivariant readout of full Cartesian Hessians."""

from math import sqrt

import torch
from e3nn import o3

from tace.dataset.hessian import hessian_pair_indices

from ..linear import e3nnLinear
from ..radial import RadialBasis
from .inter import O3CgtpEdgeInteraction


class DirectHessianReadout(torch.nn.Module):
    """Predict diagonal blocks from nodes and off-diagonal blocks from edges.

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps
        Node feature representations.
    num_channel : int
        Hidden multiplicity in the edge interaction.
    num_fidelities : int
        Number of output heads.
    atomic_numbers : sequence of int
        Atomic species in node-attribute order.
    cutoff : float
        Hessian interaction cutoff in angstroms.
    radial_basis : dict
        Radial basis and MLP configuration.
    gate_m0 : bool, optional
        Gate scalars instead of applying scalar activations directly.
    scalar_act, tensor_act : str or None, optional
        Scalar and tensor-gate activations.
    bias : bool, optional
        Include a scalar bias in the diagonal readout.

    Notes
    -----
    Returns ``(sum_g N_g**2, 3, 3)`` in row-major atom-pair order.
    Enforces ``H_ij = H_ji.T`` but not energy-derivative consistency or
    the acoustic sum rule. The readout is for nonperiodic structures.
    """

    def __init__(
        self,
        irreps_in,
        num_channel,
        num_fidelities,
        atomic_numbers,
        cutoff,
        radial_basis,
        *,
        gate_m0=False,
        scalar_act=None,
        tensor_act=None,
        bias=False,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.num_fidelities = num_fidelities
        if cutoff <= 0:
            raise ValueError("Hessian cutoff must be positive.")
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.irreps_out = o3.Irreps(
            f"{num_fidelities}x0e+{num_fidelities}x1e+{num_fidelities}x2e"
        )
        self.node_readout = e3nnLinear(self.irreps_in, self.irreps_out, bias=bias)
        self.angular_basis = o3.SphericalHarmonics(
            o3.Irreps.spherical_harmonics(2), normalize=True, normalization="component"
        )
        self.radial_basis = RadialBasis(
            cutoff=cutoff,
            num_basis=radial_basis["num_radial_basis"],
            cutoff_fn=radial_basis["cutoff_fn"],
            polynomial_cutoff=radial_basis["polynomial_cutoff"],
            radial_basis=radial_basis["radial_basis"],
            distance_transform=radial_basis["distance_transform"],
            trainable=radial_basis["trainable"],
            apply_cutoff=False,
            gaussian_width=radial_basis["gaussian_width"],
        )
        self.interaction = O3CgtpEdgeInteraction(
            self.irreps_in,
            self.angular_basis.irreps_out,
            self.irreps_out,
            num_channel,
            len(atomic_numbers),
            radial_basis,
            gate_m0=gate_m0,
            scalar_act=scalar_act,
            tensor_act=tensor_act,
        )
        self.register_buffer(
            "basis",
            torch.cat(
                [
                    o3.wigner_3j(1, 1, ell, dtype=torch.float64) * sqrt(2 * ell + 1)
                    for ell in range(3)
                ],
                dim=-1,
            ),
        )

    def to_cartesian(self, coefficients, fidelity):
        """Select fidelity heads and convert irreps to Cartesian blocks."""
        parts = coefficients.split(
            [self.num_fidelities, 3 * self.num_fidelities, 5 * self.num_fidelities],
            dim=-1,
        )
        coefficients = torch.cat(
            [
                part.reshape(coefficients.size(0), self.num_fidelities, dim)
                for part, dim in zip(parts, (1, 3, 5))
            ],
            dim=-1,
        )
        row = torch.arange(coefficients.size(0), device=coefficients.device)
        return torch.einsum("pk,abk->pab", coefficients[row, fidelity], self.basis)

    def forward(
        self,
        features,
        positions,
        ptr,
        batch,
        node_attrs,
        node_fidelity,
        pair_index=None,
    ):
        if pair_index is None:
            pair_index = hessian_pair_indices(ptr)
        source, target = pair_index.unbind(0)
        counts = ptr[1:] - ptr[:-1]
        offsets = counts.square().cumsum(0) - counts.square()
        node = torch.arange(features.size(0), device=features.device)
        diagonal = offsets[batch] + (node - ptr[batch]) * (counts[batch] + 1)
        blocks = features.new_zeros((source.numel(), 3, 3)).index_copy(
            0, diagonal, self.to_cartesian(self.node_readout(features), node_fidelity)
        )

        # Self pairs are handled by the node readout, without a direction or norm.
        pair = torch.arange(source.numel(), device=source.device)[source != target]
        edge_index = pair_index[:, pair]
        vectors = positions[edge_index[1]] - positions[edge_index[0]]
        lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        within_cutoff = lengths[:, 0] < self.cutoff
        pair = pair[within_cutoff]
        edge_index = edge_index[:, within_cutoff]
        if edge_index.size(1) > 0:
            edge_feats, edge_cutoff = self.radial_basis(
                lengths[within_cutoff], node_attrs, edge_index, self.atomic_numbers
            )
            edge_attrs = self.angular_basis(vectors[within_cutoff])
            coefficients = self.interaction(
                features, node_attrs, edge_feats, edge_attrs, edge_index, edge_cutoff
            )
            blocks = blocks.index_copy(
                0, pair, self.to_cartesian(coefficients, node_fidelity[edge_index[0]])
            )

        graph = batch[source]
        reverse = (
            offsets[graph] + (target - ptr[graph]) * counts[graph] + source - ptr[graph]
        )
        return (blocks + blocks[reverse].transpose(-1, -2)) * 0.5
