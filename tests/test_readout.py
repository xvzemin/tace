from copy import deepcopy
from types import SimpleNamespace

import torch
from e3nn import o3

from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.readout import ElementScalarReadOut, ElementTensorReadOut
from tace.models._e3nn.tace import e3nnTACE
from tace.models.mag import MagmomsNormalizer
from tace.models.radial import MagneticChebyshevBasis


def _one_body_magmoms_model(num_mag_radial_basis=10):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        {
            "cutoff": 4.0,
            "max_neighbors": None,
            "statistics": [
                {
                    "atomic_numbers": [26],
                    "avg_num_neighbors": 4.0,
                    "magmoms_norm_by_element": {26: 4.0},
                    "atomic_energy": {26: 0.0},
                }
            ],
            "fidelity": [{"name": "PBE", "atomic_energy": None}],
            "target_property": ["energy"],
            "embedding_property": ["initial_noncollinear_magmoms"],
            "Lmax": 1,
            "lmax": 1,
            "mag_Lmax": 1,
            "num_layers": 1,
            "num_channel": 4,
        }
    )
    config["atomic_basis"]["type"] = "cgtp"
    config["atomic_basis"]["nonlinear"] = None
    config["atomic_basis"]["edge_nonlinear"] = None
    config["product_basis"]["correlation"] = 1
    config["radial_basis"]["num_mag_radial_basis"] = num_mag_radial_basis
    config["scale_shift"]["enable"] = False
    return e3nnTACE(**config)


def test_shared_magnetic_basis_matches_separate_bases():
    x = torch.linspace(-1.0, 1.0, 11).unsqueeze(-1)
    shared = MagneticChebyshevBasis(num_basis=10, include_constant=True)(x)
    interaction = MagneticChebyshevBasis(num_basis=9, include_constant=False)(x)
    one_body = MagneticChebyshevBasis(num_basis=10, include_constant=True)(x)

    torch.testing.assert_close(shared[..., 1:], interaction)
    torch.testing.assert_close(shared, one_body)


def test_element_scalar_readout_selects_element_weights():
    readout = ElementScalarReadOut(
        layer=0,
        num_layers=1,
        hidden_channel=[],
        bias=False,
        num_fidelities=1,
        parity=True,
        irreps_in=o3.Irreps("2x0e"),
        irreps_out=o3.Irreps("0e"),
        num_elements=2,
    )
    with torch.no_grad():
        readout.linear2[0].weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        )

    x = torch.tensor([[3.0, 5.0], [3.0, 5.0]])
    node_attrs = torch.eye(2)
    output = readout(x, node_attrs=node_attrs)

    expected = torch.tensor([[3.0], [10.0]]) / 2.0**0.5
    torch.testing.assert_close(output, expected)


def test_element_tensor_readout_is_equivariant_and_differentiable():
    irreps_in = o3.Irreps("2x0e+2x1o")
    irreps_out = o3.Irreps("1o")
    readout = ElementTensorReadOut(
        layer=0,
        num_layers=1,
        hidden_channel=[3],
        bias=False,
        num_fidelities=1,
        parity=True,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        num_elements=2,
    ).double()
    node_attrs = torch.eye(2, dtype=torch.float64)
    x = torch.randn(2, irreps_in.dim, dtype=torch.float64, requires_grad=True)
    rotation = o3.rand_matrix(dtype=torch.float64)

    output = readout(x, node_attrs=node_attrs)
    rotated_output = readout(
        x @ irreps_in.D_from_matrix(rotation).T,
        node_attrs=node_attrs,
    )
    expected = output @ irreps_out.D_from_matrix(rotation).T

    torch.testing.assert_close(rotated_output, expected, atol=1e-10, rtol=1e-10)
    torch.autograd.grad(output.square().sum(), x)


def test_one_body_magmoms_readout_has_radial_magnetic_force():
    magmoms = torch.tensor(
        [[1.0, 2.0, -1.0], [0.5, -1.0, 2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    node_attrs = torch.eye(2, dtype=torch.float64)
    normalized = MagmomsNormalizer([4.0, 5.0], num_elements=2).double()(
        magmoms, node_attrs
    )
    basis = MagneticChebyshevBasis(
        num_basis=4,
        include_constant=True,
    )(normalized)
    readout = ElementScalarReadOut(
        layer=0,
        num_layers=1,
        hidden_channel=[],
        bias=False,
        num_fidelities=1,
        parity=True,
        irreps_in=o3.Irreps("4x0e"),
        irreps_out=o3.Irreps("0e"),
        num_elements=2,
    ).double()
    energy = readout(basis, node_attrs=node_attrs).sum()
    magnetic_force = -torch.autograd.grad(energy, magmoms)[0]

    cross = torch.linalg.cross(magnetic_force, magmoms, dim=-1)
    torch.testing.assert_close(cross, torch.zeros_like(cross), atol=1e-10, rtol=0.0)


def test_one_body_magmoms_is_enabled_for_magnetic_input():
    num_basis = 6
    model = _one_body_magmoms_model(num_basis)

    assert model.use_one_body_magmoms
    assert "initial_noncollinear_magmoms" in model.embedding_property
    assert model.representation.magnetic_radial_basis.include_constant
    assert model.representation.magnetic_radial_basis.num_basis == num_basis
    assert model.one_body_magmoms_readout.irreps_in.dim == num_basis


def test_one_body_magmoms_energy_bypasses_scale_shift():
    class DoubleScale(torch.nn.Module):
        def forward(self, node_energy, *args):
            return 2.0 * node_energy

    model = _one_body_magmoms_model()
    model.scale_shift = DoubleScale()
    with torch.no_grad():
        for parameter in model.energy_readouts.parameters():
            parameter.zero_()

    num_nodes = 2
    node_attrs = torch.ones(num_nodes, 1)
    node_fidelity = torch.zeros(num_nodes, dtype=torch.int64)
    one_body_magmoms_basis = torch.randn(num_nodes, 10)
    expected = model.one_body_magmoms_readout(
        one_body_magmoms_basis,
        node_fidelity=node_fidelity,
        node_attrs=node_attrs,
    )[:, 0]
    data = {
        "node_attrs": node_attrs,
        "batch": torch.zeros(num_nodes, dtype=torch.int64),
        "ptr": torch.tensor([0, num_nodes]),
        "edge_index": torch.empty(2, 0, dtype=torch.int64),
    }
    graph = SimpleNamespace(
        lmp_natoms=(num_nodes, 0),
        num_graphs=1,
        node_fidelity=node_fidelity,
        num_atoms_arange=torch.arange(num_nodes),
    )
    representation = {
        "descriptors": [
            torch.zeros(num_nodes, model.representation.products[-1].irreps_out.dim)
        ],
        "one_body_magmoms_basis": one_body_magmoms_basis,
    }

    output = model.readout_fn(data, graph, representation)

    torch.testing.assert_close(output["node_energy"], expected)
