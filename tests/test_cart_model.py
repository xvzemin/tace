from copy import deepcopy

import pytest
import torch

from tace.models._cart.default import DEFAULT_MODEL_CONFIG
from tace.models._cart.tace import cartTACE
from tace.models._e3nn.tace import e3nnTACE
from tace.models.lammps import Graph


def _model_config(*, num_fidelities: int = 2) -> dict:
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    statistics = {
        "atomic_numbers": [1, 8],
        "avg_num_neighbors": 2.0,
        "atomic_energy": {1: 0.0, 8: 0.0},
    }
    config.update(
        cutoff=5.0,
        max_neighbors=None,
        statistics=[statistics] * num_fidelities,
        target_property=[
            "energy",
            "direct_forces",
            "direct_polarizability",
            "direct_stress",
        ],
        num_layers=2,
        num_channel=4,
        Lmax=2,
        lmax=2,
    )
    config["fidelity"] = [
        {"name": f"level-{index}", "atomic_energy": None}
        for index in range(num_fidelities)
    ]
    config["radial_basis"]["hidden"] = [8]
    config["radial_basis"]["apply_cutoff"] = False
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    return config


def _model(*, num_fidelities: int = 2) -> cartTACE:
    return cartTACE(**_model_config(num_fidelities=num_fidelities))


def _inputs(
    positions: torch.Tensor,
    *,
    multi_fidelity: bool = True,
) -> tuple[dict[str, torch.Tensor], Graph]:
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
        device=positions.device,
    )
    edge_vector = positions[edge_index[1]] - positions[edge_index[0]]
    lattice = torch.eye(3, dtype=positions.dtype, device=positions.device).unsqueeze(0)
    data = {
        "node_attrs": positions.new_tensor([[1, 0], [0, 1], [1, 0]]),
        "edge_index": edge_index,
        "batch": torch.zeros(3, dtype=torch.int64, device=positions.device),
        "ptr": torch.tensor([0, 3], device=positions.device),
        "positions": positions,
        "lattice": lattice,
        "atomic_numbers": torch.tensor([1, 8, 1], device=positions.device),
    }
    node_fidelity = torch.tensor(
        [0, int(multi_fidelity), 0],
        device=positions.device,
    )
    graph = Graph(
        lmp=False,
        lmp_data=None,
        lmp_natoms=(3, 0),
        num_graphs=1,
        displacement=None,
        positions=positions,
        edge_vector=edge_vector,
        edge_length=edge_vector.norm(dim=-1, keepdim=True) + 1.0e-9,
        lattice=lattice,
        node_fidelity=node_fidelity,
        num_atoms_arange=torch.arange(3, device=positions.device),
    )
    return data, graph


def test_cart_model_forward_shapes_and_finite_outputs() -> None:
    model = _model()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.0, 0.3]])
    data, graph = _inputs(positions)
    output = model(data, graph)

    assert output["energy"].shape == (1,)
    assert output["direct_forces"].shape == (3, 3)
    assert output["direct_polarizability"].shape == (1, 3, 3)
    assert output["direct_stress"].shape == (1, 3, 3)
    assert all(
        torch.isfinite(output[name]).all()
        for name in (
            "energy",
            "direct_forces",
            "direct_polarizability",
            "direct_stress",
        )
    )


def test_cart_model_rotation_equivariance() -> None:
    torch.manual_seed(7)
    model = _model(num_fidelities=1).eval()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.0, 0.3]])
    matrix, _ = torch.linalg.qr(torch.randn(3, 3))
    matrix = matrix * torch.linalg.det(matrix)

    data, graph = _inputs(positions, multi_fidelity=False)
    rotated_data, rotated_graph = _inputs(
        positions @ matrix.T,
        multi_fidelity=False,
    )
    output = model(data, graph)
    rotated = model(rotated_data, rotated_graph)

    torch.testing.assert_close(
        rotated["energy"], output["energy"], atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        rotated["direct_forces"],
        output["direct_forces"] @ matrix.T,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        rotated["direct_polarizability"],
        matrix @ output["direct_polarizability"] @ matrix.T,
        atol=2e-5,
        rtol=2e-5,
    )


def test_cart_model_rejects_local_frame_interaction() -> None:
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=5.0,
        max_neighbors=None,
        statistics=[
            {
                "atomic_numbers": [1],
                "avg_num_neighbors": 1.0,
                "atomic_energy": {1: 0.0},
            }
        ],
    )
    config["fidelity"] = [{"name": "level-0", "atomic_energy": None}]
    config["atomic_basis"]["type"] = "o2"
    with pytest.raises(ValueError, match="atomic_basis.type does not support"):
        cartTACE(**config)


@pytest.mark.parametrize("gate_m0", [False, True])
def test_cart_model_parameter_count_matches_spherical_model(gate_m0: bool) -> None:
    config = _model_config(num_fidelities=1)
    config["atomic_basis"]["gate_m0"] = gate_m0
    config["product_basis"]["correlation"] = [3, 3]
    config["readout_emlp"]["use_alllayer"] = True
    cartesian = cartTACE(**deepcopy(config))
    spherical_config = deepcopy(config)
    spherical_config["atomic_basis"]["type"] = "cgtp"
    spherical_config["product_basis"]["type"] = "cgtp"
    spherical = e3nnTACE(**spherical_config)
    assert sum(parameter.numel() for parameter in cartesian.parameters()) == sum(
        parameter.numel() for parameter in spherical.parameters()
    )


def test_cart_path_features_are_simplified_before_linear_maps() -> None:
    config = _model_config(num_fidelities=1)
    config["product_basis"]["correlation"] = [3, 3]
    model = cartTACE(**config)

    for interaction in model.representation.interactions:
        path_irreps = interaction.rejector.tp.path_irreps_out
        assert interaction.rejector.irreps_out == path_irreps.simplify()
        assert interaction.linear_down.irreps_in == path_irreps.simplify()

    for product in model.representation.products:
        correlation_irreps = [
            product.irreps_hidden,
            *(ace.irreps_out for ace in product.aces),
        ]
        for coefficient, irreps in zip(product.coefs, correlation_irreps):
            assert coefficient.irreps_in == irreps.simplify()


def test_cart_model_rejects_magnetic_interaction() -> None:
    config = _model_config(num_fidelities=1)
    config["atomic_basis"]["type"] = ["w6j_mag"] * config["num_layers"]
    with pytest.raises(ValueError, match="does not support"):
        cartTACE(**config)
