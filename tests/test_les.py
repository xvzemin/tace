import pytest
import torch
from e3nn import o3

from tace.models._e3nn.les import (
    LinearLesReadoutBlock,
    NonLinearLesReadoutBlock,
    TACELes,
    required_les_irreps,
)


@pytest.mark.parametrize(
    "block_type", [LinearLesReadoutBlock, NonLinearLesReadoutBlock]
)
def test_les_tensor_readout_is_equivariant_for_every_fidelity(block_type):
    torch.manual_seed(0)
    irreps = o3.Irreps("4x0e+3x1o+2x1e")
    block = block_type(irreps, num_fidelities=2).double()
    features = torch.randn(5, irreps.dim, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)

    output = block(features)
    rotated_output = block(features @ irreps.D_from_matrix(rotation).T)
    expected = torch.einsum("ij,nfjk,lk->nfil", rotation, output, rotation)

    assert output.shape == (5, 2, 3, 3)
    assert torch.allclose(output, output.transpose(-1, -2), atol=1e-10)
    assert torch.allclose(rotated_output, expected, atol=1e-6)


def test_required_les_irreps_cover_enabled_sources():
    assert required_les_irreps({}) == ["0e"]
    assert set(
        required_les_irreps(
            {
                "use_dipole": True,
                "use_quad": True,
                "use_induced_dipole": True,
                "use_anisotropic_polarizability": True,
                "alpha_irreps": "0e+1e+2e",
            }
        )
    ) == {"0e", "1o", "1e", "2e"}


def test_complete_les_readout_with_upstream_backend():
    les = pytest.importorskip("les")
    arguments = {
        "use_atomwise": False,
        "compute_bec": True,
        "use_dipole": True,
        "use_quad": True,
        "use_induced_charge": True,
        "use_induced_dipole": True,
        "use_anisotropic_polarizability": True,
        "alpha_irreps": "0e+1o+2e",
        "make_alpha_positive": True,
        "make_kappa_positive": True,
        "output_scale": 0.01,
        "alpha_scale": 0.001,
        "kappa_scale": 0.001,
    }
    irreps = o3.Irreps("4x0e+3x1o+2x2e")
    model = TACELes(
        backend=les.Les(arguments),
        les_arguments=arguments,
        num_layers=1,
        hidden_channel=[8],
        bias=True,
        num_elements=2,
        num_fidelities=2,
        use_alllayer=False,
        parity=True,
        irreps_in=[irreps],
    )

    features = torch.randn(4, irreps.dim, requires_grad=True)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
        requires_grad=True,
    )
    output = model(
        descriptors=[features],
        node_fidelity=torch.tensor([0, 1, 0, 1]),
        atom_indices=torch.arange(4),
        positions=positions,
        cell=torch.zeros(2, 3, 3),
        batch=torch.tensor([0, 0, 0, 1]),
        atomic_numbers=torch.tensor([1, 8, 1, 8]),
        external_field=torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]),
        pbc=torch.zeros(2, 3, dtype=torch.bool),
    )

    assert output["les_energy"].shape == (2,)
    assert output["les_latent_charges"].shape == (4,)
    assert output["les_latent_dipoles"].shape == (4, 3)
    assert output["les_latent_quadrupoles"].shape == (4, 3, 3)
    assert output["les_latent_polarizabilities"].shape == (4, 3, 3)
    assert output["les_latent_kappas"].shape == (4,)
    assert output["les_born_effective_charges"].shape == (4, 2, 3, 3)
    assert torch.allclose(
        output["les_latent_quadrupoles"].diagonal(dim1=-1, dim2=-2).sum(-1),
        torch.zeros(4),
        atol=1e-6,
    )

    feature_grad, position_grad = torch.autograd.grad(
        output["les_energy"].sum(), (features, positions)
    )
    assert torch.isfinite(feature_grad).all()
    assert torch.isfinite(position_grad).all()
