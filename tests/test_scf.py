from types import SimpleNamespace

import pytest
import torch
from e3nn import o3

from tace.models.scf import (
    EnergyFunctionalSCF,
    FixedPointSCF,
    FixedPointSCFOptions,
    GraphLongRangeElectrostatics,
    LocalSplitCharges,
    PairwiseGaussianElectrostatics,
    QEq,
)


@pytest.fixture
def small_system():
    dtype = torch.float64
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        dtype=dtype,
        requires_grad=True,
    )
    data = {
        "positions": positions,
        "batch": torch.tensor([0, 0]),
        "ptr": torch.tensor([0, 2]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "node_attrs": torch.eye(2, dtype=dtype),
        "total_charge": torch.tensor([0.25], dtype=dtype),
        "electric_field": torch.zeros((1, 3), dtype=dtype),
        "pbc": torch.zeros((1, 3), dtype=torch.bool),
        "lattice": torch.zeros((1, 3, 3), dtype=dtype),
    }
    graph = SimpleNamespace(
        positions=positions,
        edge_length=torch.full((2, 1), 1.2, dtype=dtype),
    )
    descriptors = [torch.randn((2, 2), dtype=dtype, requires_grad=True)]
    base_results = {
        "descriptors": descriptors,
        "scalar_descriptor": descriptors[0],
        "energy": torch.zeros(1, dtype=dtype, requires_grad=True),
    }
    return data, graph, base_results


def test_pairwise_gaussian_matches_analytic_monopole_energy():
    dtype = torch.float64
    distance = torch.tensor(1.7, dtype=dtype)
    sigma = 0.8
    positions = torch.stack(
        [torch.zeros(3, dtype=dtype), torch.tensor([distance, 0.0, 0.0])]
    )
    density = torch.tensor([[0.7], [-0.4]], dtype=dtype)
    data = {
        "positions": positions,
        "batch": torch.tensor([0, 0]),
        "ptr": torch.tensor([0, 2]),
    }
    operator = PairwiseGaussianElectrostatics(sigma)
    actual = operator.energy(density, data)[0]
    expected = (
        operator.coulomb_constant
        * density[0, 0]
        * density[1, 0]
        * torch.erf(distance / (2.0 * sigma))
        / distance
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("max_l", [0, 1])
def test_vendored_longrange_supports_multipoles_and_gradients(max_l):
    dtype = torch.float64
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]],
        dtype=dtype,
        requires_grad=True,
    )
    density = torch.randn((2, (max_l + 1) ** 2), dtype=dtype, requires_grad=True)
    data = {
        "positions": positions,
        "batch": torch.tensor([0, 0]),
        "ptr": torch.tensor([0, 2]),
        "pbc": torch.zeros((1, 3), dtype=torch.bool),
        "lattice": torch.zeros((1, 3, 3), dtype=dtype),
    }
    operator = GraphLongRangeElectrostatics(
        density_max_l=max_l,
        feature_max_l=max_l,
        feature_smearing_widths=[1.0],
    )
    energy = operator.energy(density, data)
    features = operator.features(density, data)
    assert energy.shape == (1,)
    assert features.shape == density.shape
    gradients = torch.autograd.grad(energy.sum(), (density, positions))
    assert all(torch.all(torch.isfinite(gradient)) for gradient in gradients)


def test_local_split_charge_is_exactly_conserving(small_system):
    data, graph, base_results = small_system
    model = LocalSplitCharges(
        irreps_in=[o3.Irreps("2x0e")],
        scalar_descriptor_dim=2,
        num_elements=2,
        density_max_l=0,
        formal_charges_from_data=False,
        atomic_formal_charges=[1.0, -1.0],
        electrostatics={"backend": "pairwise"},
    ).to(torch.float64)
    output = model(data, graph, base_results)
    torch.testing.assert_close(
        output["charges"].sum(), torch.zeros((), dtype=torch.float64)
    )
    output["energy"].sum().backward()
    assert base_results["descriptors"][0].grad is not None


def test_qeq_enforces_total_charge_and_is_differentiable(small_system):
    data, graph, base_results = small_system
    model = QEq(
        irreps_in=[o3.Irreps("2x0e")],
        electrostatics={"backend": "pairwise"},
    ).to(torch.float64)
    output = model(data, graph, base_results)
    torch.testing.assert_close(output["charges"].sum(), data["total_charge"][0])
    gradients = torch.autograd.grad(
        output["energy"].sum(),
        (base_results["descriptors"][0], data["positions"]),
    )
    assert all(torch.all(torch.isfinite(gradient)) for gradient in gradients)


def test_fixed_point_scf_preserves_charge_and_backpropagates(small_system):
    data, graph, base_results = small_system
    model = FixedPointSCF(
        irreps_in=[o3.Irreps("2x0e")],
        density_max_l=0,
        feature_max_l=0,
        feature_smearing_widths=[1.0],
        electrostatics={"backend": "pairwise"},
        options={"num_scf_steps": 4, "mixing_parameter": 0.5},
    ).to(torch.float64)
    output = model(data, graph, base_results)
    torch.testing.assert_close(output["charges"].sum(), data["total_charge"][0])
    assert output["density_history"].shape[-1] >= 2
    gradient = torch.autograd.grad(
        output["energy"].sum(), base_results["descriptors"][0]
    )[0]
    assert torch.all(torch.isfinite(gradient))


def test_energy_functional_scf_preserves_charge(small_system):
    data, graph, base_results = small_system
    model = EnergyFunctionalSCF(
        irreps_in=[o3.Irreps("2x0e")],
        density_max_l=0,
        electrostatics={"backend": "pairwise"},
        options={"num_scf_steps": 4, "learning_rate": 0.02},
        density_energy_mode="quadratic",
    ).to(torch.float64)
    output = model(data, graph, base_results)
    torch.testing.assert_close(output["charges"].sum(), data["total_charge"][0])
    assert torch.all(torch.isfinite(output["energy"]))


def test_fixed_point_options_validate_values():
    with pytest.raises(ValueError, match="mixing_parameter"):
        FixedPointSCFOptions(mixing_parameter=0.0)
