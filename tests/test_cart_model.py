from copy import deepcopy
from typing import Optional

import pytest
import torch

from tace.models._cart.default import DEFAULT_MODEL_CONFIG
from tace.models._cart.tace import cartTACE
from tace.models.lammps import Graph


def _model(*, num_fidelities: int = 2, magnetic: bool = False) -> cartTACE:
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    statistics = {
        "atomic_numbers": [1, 8],
        "avg_num_neighbors": 2.0,
        "atomic_energy": {1: 0.0, 8: 0.0},
        "magmoms_norm_by_element": {1: 1.0, 8: 2.0},
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
    config["readout_emlp"]["use_one_body_magmoms"] = False
    config["scale_shift"]["enable"] = False
    if magnetic:
        config["parity"] = True
        config["atomic_basis"]["type"] = "o3_w6j_mag"
        config["embedding_property"] = ["initial_noncollinear_magmoms"]
        config["readout_emlp"]["use_one_body_magmoms"] = True
    return cartTACE(**config)


def _inputs(
    positions: torch.Tensor,
    *,
    multi_fidelity: bool = True,
    magnetic_moments: Optional[torch.Tensor] = None,
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
    if magnetic_moments is not None:
        data["initial_noncollinear_magmoms"] = magnetic_moments
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
    with pytest.raises(ValueError, match="Cartesian atomic_basis.type"):
        cartTACE(**config)


@pytest.mark.parametrize("improper", [False, True])
def test_cart_magnetic_interaction_rotation_invariance(improper: bool) -> None:
    torch.manual_seed(11)
    model = _model(num_fidelities=1, magnetic=True).eval()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.0, 0.3]])
    magnetic_moments = torch.randn(3, 3)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3))
    if bool(torch.linalg.det(matrix) < 0) != improper:
        matrix[:, 0] *= -1
    data, graph = _inputs(
        positions,
        multi_fidelity=False,
        magnetic_moments=magnetic_moments,
    )
    rotated_data, rotated_graph = _inputs(
        positions @ matrix.T,
        multi_fidelity=False,
        magnetic_moments=magnetic_moments @ (torch.linalg.det(matrix) * matrix).T,
    )

    output = model(data, graph)
    rotated = model(rotated_data, rotated_graph)
    torch.testing.assert_close(
        rotated["energy"], output["energy"], atol=2e-5, rtol=2e-5
    )
