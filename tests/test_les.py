import pytest
import torch
from e3nn import o3

from tace.models._e3nn.les import TACELES, required_les_irreps
from tace.models._e3nn.readout import TensorReadOut


def test_required_les_irreps_cover_direct_readouts():
    assert required_les_irreps({}) == ["0e"]
    assert set(
        required_les_irreps(
            {
                "use_dipole": True,
                "use_quad": True,
                "use_induced_dipole": True,
                "use_anisotropic_polarizability": True,
            }
        )
    ) == {"0e", "1o", "2e"}


def test_les_uses_the_standard_tace_irrep_readout():
    pytest.importorskip("les")
    irreps = o3.Irreps("4x0e+3x1o+2x2e")
    model = TACELES(
        {"use_quad": True},
        num_layers=1,
        hidden_channel=[8],
        bias=True,
        num_elements=2,
        num_fidelities=2,
        use_alllayer=False,
        parity=True,
        irreps_in=[irreps],
    )
    assert isinstance(model.quad_readouts[0], TensorReadOut)

    descriptors = [torch.randn(4, irreps.dim)]
    output = model._sum_readouts(
        model.quad_readouts,
        descriptors,
        torch.arange(4),
        torch.tensor([0, 1, 0, 1]),
        5,
    )
    assert output.shape == (4, 5)


def test_tace_les_reads_out_sources_and_passes_them_to_upstream():
    pytest.importorskip("les")
    model = TACELES(
        {
            "output_scaling_factor": 0.01,
            "use_dipole": True,
            "use_quad": True,
            "use_induced_charge": True,
            "use_induced_dipole": True,
            "use_anisotropic_polarizability": True,
            "make_alpha_positive": True,
            "make_kappa_positive": True,
        },
        num_layers=1,
        hidden_channel=[8],
        bias=True,
        num_elements=2,
        num_fidelities=2,
        use_alllayer=False,
        parity=True,
        irreps_in=[o3.Irreps("4x0e+3x1o+2x2e")],
    )
    model.les.compute_bec = True
    model.les.bec_output_index = None

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
        requires_grad=True,
    )
    descriptors = [torch.randn(4, 23, requires_grad=True)]
    node_fidelity = torch.tensor([0, 1, 0, 1])

    output = model(
        descriptors=descriptors,
        node_fidelity=node_fidelity,
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

    gradients = torch.autograd.grad(
        output["les_energy"].sum(),
        (positions, descriptors[0]),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
