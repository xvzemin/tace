"""Data handling, equivariance and training for target properties."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.db import connect
from ase.io import write
from e3nn import o3
from torch_geometric.data import Batch

from tace.dataset import quantity
from tace.dataset.element import build_element_lookup
from tace.dataset.graph import from_atoms, read_properties
from tace.dataset.quantity import DIRECT_PROPERTIES, PROPERTIES, KeySpecification
from tace.dataset.read import ase_db_connect, ase_io_read
from tace.dataset.utils import hessian_pair_indices, shape_fn_for_hessian
from tace.lightning import create_model, export_tace, load_tace
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.les import TACELES, required_les_irreps
from tace.models._e3nn.readout import DirectHessianReadout
from tace.models._e3nn.representation import PairRepresentation
from tace.models.utils import get_target_irreps, prepare_direct_hessian_inputs
from tace.utils.loss.common import hessian_loss
from tace.utils.loss.normal import NormalLoss
from tace.utils.loss.uncertainty import UncertaintyLoss
from tace.utils.mask_metrics import MaskMAE, supports_weight_filter
from tace.utils.utils import torch_default_dtype


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_property_definitions():
    for name, value in vars(quantity).items():
        if name.endswith("_PROPERTIES"):
            assert isinstance(value, tuple), name
            assert all(property_name in PROPERTIES for property_name in value), name
    fields = (
        "ase_name",
        "type",
        "scope",
        "rank",
        "irreps",
        "time_reversal",
        "abbreviation",
        "shape",
        "default_value_fn",
        "requires_periodic_cell",
        "must_be_with",
        "enable_prediction",
        "enable_embedding",
        "first_derivative",
        "second_derivative",
        "requires_grad_with",
    )
    periodic = {"stress", "direct_stress", "atomic_stresses", "polarization"}
    for name, config in PROPERTIES.items():
        assert tuple(config) == fields, name
        assert (name in DIRECT_PROPERTIES) == (
            not (config["first_derivative"] or config["second_derivative"])
        ), name
        assert tuple(config["shape"]) == ("in_data", "shape_fn"), name
        assert config["requires_periodic_cell"] is (name in periodic), name
        for field in (
            "enable_prediction",
            "enable_embedding",
            "first_derivative",
            "second_derivative",
        ):
            assert isinstance(config[field], bool), (name, field)


@pytest.mark.parametrize("parity", [None, False, True])
@pytest.mark.parametrize("hessian", ["hessian", "direct_hessian"])
def test_target_irreps(parity, hessian):
    actual = {
        o3.Irrep(ir)
        for ir in get_target_irreps(["energy", "forces", hessian], parity=parity)
    }
    expected = {o3.Irrep("0e")}
    if hessian == "direct_hessian":
        expected.update((o3.Irrep("1o"), o3.Irrep("2e")))
        if parity is None:
            expected.add(o3.Irrep("1e"))
    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_graph_properties(dtype):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    keys = KeySpecification.from_defaults()
    keys.info_keys["energy"] = "reference_energy"
    keys.arrays_keys["forces"] = "reference_forces"
    atoms.info["reference_energy"] = 2.0
    atoms.info["forces_weight"] = 3.0
    atoms.arrays["reference_forces"] = np.ones((2, 3))
    properties = ["energy", "forces", "fidelity_idx"]
    actual = read_properties(atoms, keys, properties, dtype, periodic_cell=False)
    assert actual["energy"].item() == 2.0
    assert actual["forces"].shape == (2, 3)
    assert actual["forces_weight"].item() == 3.0
    assert actual["energy"].dtype == actual["forces"].dtype == dtype
    assert actual["fidelity_idx"].dtype == torch.int64
    for invalid in (np.nan, np.inf, -np.inf):
        atoms.arrays["reference_forces"][0, 0] = invalid
        with pytest.raises(RuntimeError, match="forces.*NaN or infinite"):
            read_properties(atoms, keys, properties, dtype, periodic_cell=False)
    del atoms.arrays["reference_forces"]
    actual = read_properties(atoms, keys, properties, dtype, periodic_cell=False)
    assert actual["forces_weight"].item() == 0
    assert not actual["forces"].any()
    assert not any(name.endswith("_mask") for name in actual)

    for key, value in (
        ("stress", np.ones(6)),
        ("direct_stress", np.ones(6)),
        ("atomic_stresses", np.ones((2, 3, 3))),
        ("polarization", np.ones(3)),
    ):
        if PROPERTIES[key]["scope"] == "per-atom":
            atoms.arrays[keys.arrays_keys[key]] = value
        else:
            atoms.info[keys.info_keys[key]] = value
        for periodic, cell in ((False, None), (False, [5, 5, 5]), (True, [5, 5, 0])):
            atoms.pbc = [periodic, periodic, False]
            atoms.set_cell(np.zeros((3, 3)) if cell is None else cell)
            with pytest.raises(RuntimeError, match=key):
                from_atoms(
                    build_element_lookup([1]),
                    atoms,
                    4.0,
                    None,
                    keys,
                    [key],
                    [],
                    dtype=dtype,
                    neighborlist_backend="ase",
                )
        atoms.pbc = [True, False, False]
        atoms.set_cell([5, 5, 5])
        graph = from_atoms(
            build_element_lookup([1]),
            atoms,
            4.0,
            None,
            keys,
            [key],
            [],
            dtype=dtype,
            neighborlist_backend="ase",
        )
        assert not any(name.endswith("_mask") for name in graph.keys())
        assert graph[f"{key}_weight"].item() == 1
        atoms.pbc = False
        atoms.info[f"{key}_weight"] = 0.0
        graph = from_atoms(
            build_element_lookup([1]),
            atoms,
            4.0,
            None,
            keys,
            [key],
            [],
            dtype=dtype,
            neighborlist_backend="ase",
        )
        assert not graph[key].any()
        assert graph[f"{key}_weight"].item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_property_scopes(dtype, device):
    keys = KeySpecification.from_defaults()
    targets = ["edge_forces", "hessian", "direct_hessian"]
    assert PROPERTIES["edge_forces"]["scope"] == "per-edge"
    assert all(PROPERTIES[key]["scope"] == "per-pair" for key in targets[1:])
    assert keys.info_keys["edge_forces"] == "edge_forces"
    structures = [
        Atoms("H3", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 3]]),
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]),
    ]
    graphs = []
    for atoms, num_edges in zip(structures, (4, 0, 2)):

        def graph():
            return from_atoms(
                build_element_lookup([1]),
                atoms,
                1.25,
                None,
                keys,
                targets,
                [],
                dtype=dtype,
                neighborlist_backend="ase",
            )

        missing = graph()
        assert missing.edge_index.shape[1] == num_edges
        assert missing.edge_forces.shape == (num_edges, 3)
        assert (
            missing.hessian.shape
            == missing.direct_hessian.shape
            == (len(atoms) ** 2, 3, 3)
        )
        assert all(missing[f"{key}_weight"].item() == 0 for key in targets)
        atoms.info["edge_forces"] = np.arange(3 * num_edges).reshape(num_edges, 3)
        for key in targets[1:]:
            atoms.info[key] = np.eye(3 * len(atoms))
        graphs.append(graph())
        torch.testing.assert_close(
            graphs[-1].edge_forces, torch.tensor(atoms.info["edge_forces"], dtype=dtype)
        )
        atoms.info["edge_forces"] = np.zeros((len(atoms) ** 2, 3))
        with pytest.raises(RuntimeError, match="per-edge property requires"):
            graph()

    batch = Batch.from_data_list(graphs).to(device)
    assert batch.edge_forces.shape == (6, 3)
    assert batch.hessian.shape == batch.direct_hessian.shape == (17, 3, 3)
    # Edge order need not be grouped by structure.
    order = torch.tensor([4, 0, 5, 1, 2, 3], device=device)
    batch.edge_index = batch.edge_index[:, order]
    batch.edge_shifts = batch.edge_shifts[order]
    batch.edge_forces = batch.edge_forces[order]
    edge_graph = batch.batch[batch.edge_index[0]]
    counts = batch.ptr.diff().square()
    pair_graph = torch.repeat_interleave(torch.arange(3, device=device), counts)
    for key in targets:
        assert supports_weight_filter(key)
        batch[f"{key}_weight"][-1] = 0
        graph_index = edge_graph if key == "edge_forces" else pair_graph
        error = batch[key].new_tensor([2, 3, 100])[graph_index]
        prediction = batch[key] + error.reshape((-1,) + (1,) * (batch[key].ndim - 1))
        metric = MaskMAE(key, 1.0).to(device=device, dtype=dtype)
        metric.update(prediction, batch[key], batch.to_dict())
        expected = 2.0 if key == "edge_forces" else (9 * 2 + 4 * 3) / 13
        torch.testing.assert_close(
            metric.compute(), metric.sum_abs_error.new_tensor(expected)
        )
        assert metric.count.item() == (12 if key == "edge_forces" else 117)

    # Periodic images are separate edges, not additional per-pair entries.
    atoms = Atoms("H", cell=[1, 1, 1], pbc=True)
    graph = from_atoms(
        build_element_lookup([1]),
        atoms,
        1.1,
        None,
        keys,
        targets,
        [],
        dtype=dtype,
        neighborlist_backend="ase",
    )
    assert graph.edge_forces.shape == (6, 3)
    assert graph.hessian.shape == graph.direct_hessian.shape == (1, 3, 3)


def check_hessian_layouts(n, dtype, device):
    matrix = torch.arange(9 * n * n, dtype=dtype, device=device).reshape(3 * n, 3 * n).T
    blocks = matrix.reshape(n, 3, n, 3).permute(0, 2, 1, 3).contiguous()
    for value in (matrix, matrix.flatten(), blocks, blocks.reshape(n * n, 3, 3)):
        result = shape_fn_for_hessian(value, n)
        torch.testing.assert_close(result, blocks.reshape(-1, 3, 3))
        assert result.is_contiguous()
        assert result.dtype == dtype and result.device == matrix.device


def check_hessian_invalid_shape(shape):
    with pytest.raises(ValueError, match="Invalid Hessian shape"):
        shape_fn_for_hessian(torch.zeros(shape), 2)


def check_hessian_data_roundtrip(tmp_path, key):
    matrix = np.arange(81, dtype=float).reshape(9, 9)
    blocks = matrix.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)
    structures = []
    with connect(tmp_path / f"{key}.db") as database:
        for value in (matrix, matrix.ravel(), blocks, blocks.reshape(9, 3, 3)):
            atoms = Atoms("H3", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
            atoms.info[key] = value
            database.write(atoms, data=atoms.info)
            structures.append(atoms)
    write(tmp_path / f"{key}.xyz", structures)
    expected = torch.from_numpy(blocks.reshape(9, 3, 3))
    for reader, filename in (
        (ase_db_connect, f"{key}.db"),
        (ase_io_read, f"{key}.xyz"),
    ):
        restored = reader(str(tmp_path / filename))
        assert len(restored) == 4
        for atoms in restored:
            graph = from_atoms(
                build_element_lookup([1]),
                atoms,
                4.0,
                None,
                KeySpecification.from_defaults(),
                [key],
                [],
                neighborlist_backend="ase",
                dtype=torch.float64,
            )
            torch.testing.assert_close(graph[key], expected)
            assert graph[key].is_contiguous()


def make_graph(n, labeled=True, dtype=torch.float64, key="direct_hessian"):
    atoms = Atoms(numbers=[1] * n, positions=np.arange(n * 3).reshape(n, 3) * 0.2)
    if labeled:
        atoms.info["reference_hessian"] = np.eye(3 * n)
    keys = KeySpecification.from_defaults()
    keys.info_keys[key] = "reference_hessian"
    return from_atoms(
        build_element_lookup([1]),
        atoms,
        4.0,
        None,
        keys,
        [key],
        [],
        neighborlist_backend="ase",
        dtype=dtype,
    )


def check_hessian_batch(key):
    graphs = [
        make_graph(2, key=key),
        make_graph(4, key=key),
        make_graph(1, labeled=False, key=key),
    ]
    batch = Batch.from_data_list(graphs)
    assert batch[key].shape == (21, 3, 3)
    assert "direct_hessian_pair_index" not in batch
    assert batch[f"{key}_weight"].tolist() == [1, 1, 0]
    assert not batch[key][-1].any()
    metric = MaskMAE(key, 1.0)
    prediction = batch[key] + 1
    prediction[-1] = 100
    metric.update(prediction, batch[key], batch.to_dict())
    torch.testing.assert_close(metric.compute(), torch.tensor(1.0))


def check_direct_hessian_missing_uncertainty(device):
    label = (
        Batch.from_data_list(
            [make_graph(2, labeled=False), make_graph(4, labeled=False)]
        )
        .to(device)
        .to_dict()
    )
    prediction = torch.ones_like(label["direct_hessian"], requires_grad=True)
    loss = UncertaintyLoss(["direct_hessian"], ["mse_direct_hessian"], [1.0]).to(device)
    value = loss({"direct_hessian": prediction}, label)
    value.backward()
    assert value == 0
    assert not prediction.grad.any()
    assert loss.log_sigmas["direct_hessian"].grad == 0


def check_hessian_invalid_labels(key):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.info["reference"] = np.eye(6)
    keys = KeySpecification.from_defaults()
    keys.info_keys[key] = "reference"
    for invalid in (np.nan, np.inf, -np.inf):
        atoms.info["reference"][0, 1] = invalid
        with pytest.raises(RuntimeError, match=f"{key}.*NaN or infinite"):
            from_atoms(
                build_element_lookup([1]),
                atoms,
                4.0,
                None,
                keys,
                [key],
                [],
                neighborlist_backend="ase",
            )


def check_direct_hessian_loss(device, loss, reduction):
    label = Batch.from_data_list([make_graph(2), make_graph(4)]).to(device).to_dict()
    target = label["direct_hessian"]
    prediction = (
        target + torch.cat((torch.ones(4, 3, 3), 2 * torch.ones(16, 3, 3))).to(target)
    ).requires_grad_()
    fn = NormalLoss(
        ["direct_hessian"],
        [f"{loss}_direct_hessian"],
        [1.0],
        [{"reduction": reduction}],
    )
    value = fn({"direct_hessian": prediction}, label)
    a, b = {
        "mse": (1, 4),
        "mae": (1, 2),
        "l2mae": (3, 6),
        "huber": (0.00995, 0.01995),
    }[loss]
    expected = (a + b) / 2 if reduction == "structure" else (4 * a + 16 * b) / 20
    torch.testing.assert_close(value, value.new_tensor(expected))
    value.backward()
    assert torch.isfinite(prediction.grad).all()
    label["direct_hessian_weight"].zero_()
    prediction.grad = None
    value = fn({"direct_hessian": prediction}, label)
    value.backward()
    assert value == 0 and not prediction.grad.any()


def check_direct_hessian_l2mae_weights(device, reduction):
    label = Batch.from_data_list([make_graph(2), make_graph(3)]).to(device).to_dict()
    target = label["direct_hessian"]
    label["entropy"][:] = torch.tensor([2, 3], device=device)
    prediction = target.clone()
    prediction[0, 0, :2] += torch.tensor([3, 4], device=device)
    prediction[4, 1, 0] += 12
    prediction.requires_grad_()
    value = hessian_loss(
        {"direct_hessian": prediction}, label, "direct_hessian", "l2mae", reduction
    )
    expected = (
        (2 * 5 / 4 + 3 * 12 / 9) / 2
        if reduction == "structure"
        else (2 * 5 + 3 * 12) / 13
    )
    torch.testing.assert_close(value, value.new_tensor(expected))
    value.backward()
    assert torch.isfinite(prediction.grad).all()
    assert not prediction.grad[2].any()


def check_direct_hessian_huber_threshold(device, huber_delta):
    label = Batch.from_data_list([make_graph(1)]).to(device).to_dict()
    error = (
        label["direct_hessian"]
        .new_tensor([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
        .reshape(1, 3, 3)
        * huber_delta
    )
    label["direct_hessian_weight"][:] = 3
    prediction = (label["direct_hessian"] + error).requires_grad_()
    loss = NormalLoss(
        ["direct_hessian"],
        ["huber_direct_hessian"],
        [1.0],
        [{"huber_delta": huber_delta}],
    )
    value = loss({"direct_hessian": prediction}, label)
    expected = (
        3
        * torch.where(
            error.abs() <= huber_delta,
            error.square() / 2,
            huber_delta * (error.abs() - huber_delta / 2),
        ).mean()
    )
    torch.testing.assert_close(value, expected)
    value.backward()
    assert torch.isfinite(prediction.grad).all()


def direct_hessian_output(
    head,
    representation,
    features,
    positions,
    ptr,
    batch,
    node_attrs,
    fidelity,
    lattice=None,
    pbc=None,
):
    data = dict(positions=positions, ptr=ptr)
    data["lattice"] = (
        positions.new_zeros((ptr.numel() - 1, 3, 3)) if lattice is None else lattice
    )
    if pbc is not None:
        data["pbc"] = pbc
    data = prepare_direct_hessian_inputs(data, representation.cutoff)
    edge_index = data["direct_hessian_pair_index"][:, data["direct_hessian_edge_pairs"]]
    edge_feats = representation(
        features,
        node_attrs,
        positions,
        edge_index,
        data["direct_hessian_edge_shifts"],
        data["lattice"],
        batch,
    )
    return head(
        features,
        edge_feats,
        ptr,
        batch,
        fidelity,
        data["direct_hessian_pair_index"],
        data["direct_hessian_edge_pairs"],
    )


def check_direct_hessian_readout_symmetries(device, interaction_type):
    radial = deepcopy(DEFAULT_MODEL_CONFIG["radial_basis"])
    radial.update(hidden=[4], mlp_type="mlp_v2")
    with torch_default_dtype(torch.float64):
        representation = PairRepresentation(
            "2x0e+2x1o+2x2e",
            o3.Irreps.spherical_harmonics(2),
            "2x0e+2x1e+2x2e",
            2,
            [1, 8],
            16.0,
            radial,
            interaction_type=interaction_type,
        ).to(device)
        head = DirectHessianReadout(
            representation.irreps_in, representation.irreps_out, 2, 2
        ).to(device)
    x = torch.randn(
        4, head.irreps_in.dim, dtype=torch.float64, device=device, requires_grad=True
    )
    positions = torch.randn(
        4, 3, dtype=torch.float64, device=device, requires_grad=True
    )
    ptr = torch.tensor([0, 3, 4], device=device)
    batch = torch.tensor([0, 0, 0, 1], device=device)
    node_attrs = torch.eye(2, dtype=torch.float64, device=device)[[0, 1, 0, 1]]
    fidelity = torch.tensor([0, 0, 0, 1], device=device)
    expected = direct_hessian_output(
        head, representation, x, positions, ptr, batch, node_attrs, fidelity
    )
    blocks = expected[:9].reshape(3, 3, 3, 3)
    torch.testing.assert_close(blocks, blocks.permute(1, 0, 3, 2))
    assert (blocks[0, 1] - blocks[0, 1].T).abs().max() > 1e-8
    for sign in (1, -1):
        rotation = sign * o3.rand_matrix(dtype=torch.float64, device=device)
        with torch_default_dtype(torch.float64):
            d = head.irreps_in.D_from_matrix(rotation.cpu()).to(device)
        actual = direct_hessian_output(
            head,
            representation,
            x @ d.T,
            positions @ rotation.T + 1.7,
            ptr,
            batch,
            node_attrs,
            fidelity,
        )
        torch.testing.assert_close(
            actual, rotation @ expected @ rotation.T, atol=1e-8, rtol=1e-8
        )
    permutation = torch.tensor([2, 0, 1, 3], device=device)
    actual = direct_hessian_output(
        head,
        representation,
        x[permutation],
        positions[permutation],
        ptr,
        batch,
        node_attrs[permutation],
        fidelity,
    )
    torch.testing.assert_close(
        actual[:9].reshape(3, 3, 3, 3), blocks[permutation[:3]][:, permutation[:3]]
    )
    first = torch.autograd.grad(
        expected.square().sum(), (x, positions), create_graph=True
    )
    second = torch.autograd.grad(sum(t.square().sum() for t in first), (x, positions))
    assert all(torch.isfinite(t).all() for t in first + second)


def check_direct_hessian_node_edge_readouts(device, interaction_type):
    radial = deepcopy(DEFAULT_MODEL_CONFIG["radial_basis"])
    radial.update(hidden=[4], bias=True)
    with torch_default_dtype(torch.float64):
        representation = PairRepresentation(
            "2x0e+2x1o+2x2e",
            o3.Irreps.spherical_harmonics(2),
            "2x0e+2x1e+2x2e",
            2,
            [1, 8],
            4.0,
            radial,
            interaction_type=interaction_type,
        ).to(device)
        head = DirectHessianReadout(
            representation.irreps_in,
            representation.irreps_out,
            2,
            2,
            bias=True,
        ).to(device)
    x = torch.randn(3, head.irreps_in.dim, dtype=torch.float64, device=device)
    positions = x.new_tensor([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
    ptr = torch.tensor([0, 2, 3], device=device)
    batch = torch.tensor([0, 0, 1], device=device)
    node_attrs = x.new_tensor([[1, 0], [0, 1], [1, 0]])
    fidelity = torch.tensor([0, 0, 1], device=device)
    output = direct_hessian_output(
        head, representation, x, positions, ptr, batch, node_attrs, fidelity
    )
    node_blocks = head.to_cartesian(head.node_readout(x), fidelity)
    torch.testing.assert_close(
        output[[0, 3, 4]], (node_blocks + node_blocks.transpose(-1, -2)) / 2
    )

    edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
    edge_feats = representation(
        x,
        node_attrs,
        positions,
        edge_index,
        x.new_zeros((2, 3)),
        x.new_zeros((2, 3, 3)),
        batch,
    )
    edge_coefficients = head.edge_readout(edge_feats)
    assert edge_coefficients.shape == (2, 18)
    edge_blocks = head.to_cartesian(edge_coefficients, fidelity[edge_index[0]])
    torch.testing.assert_close(
        output[[1, 2]], (edge_blocks + edge_blocks.flip(0).transpose(-1, -2)) / 2
    )

    # No self edge or cross-structure edge reaches the interaction.
    with patch.object(
        representation,
        "forward",
        wraps=representation.forward,
    ) as call:
        direct_hessian_output(
            head, representation, x, positions, ptr, batch, node_attrs, fidelity
        )
        torch.testing.assert_close(call.call_args.args[3], edge_index)
    positions[1, 0] = 4.1
    outside = direct_hessian_output(
        head, representation, x, positions, ptr, batch, node_attrs, fidelity
    )
    assert not outside[[1, 2]].any()
    torch.testing.assert_close(outside[[0, 3, 4]], output[[0, 3, 4]])

    # Biases in the radial MLP must not reintroduce messages at the cutoff.
    for interaction in representation.interactions:
        for parameter in interaction.edge_info.parameters():
            if parameter.ndim == 1:
                with torch.no_grad():
                    parameter.fill_(0.2)
    positions[1, 0] = 4.0 - 1e-3
    positions.requires_grad_()
    near = direct_hessian_output(
        head, representation, x, positions, ptr, batch, node_attrs, fidelity
    )[1]
    first = torch.autograd.grad(near.sum(), positions, create_graph=True)[0]
    second = torch.autograd.grad(first[1, 0], positions)[0]
    assert near.abs().max() < 1e-7
    assert first.abs().max() < 1e-4
    assert second.abs().max() < 0.1


def small_model(
    dtype,
    device,
    joint=False,
    hessian_cutoff=16.0,
    interaction_type="cgtp",
    Lmax=2,
    lmax=2,
    parity=None,
):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    targets = ["energy", "forces", "direct_hessian"] if joint else ["direct_hessian"]
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=Lmax,
        lmax=lmax,
        parity=parity,
        cutoff=4.0,
        max_neighbors=None,
        fidelity=[{"name": "DFT"}],
        target_property=targets,
    )
    config["radial_basis"].update(hidden=[4], apply_cutoff=False)
    config["special"]["direct_hessian"]["cutoff"] = hessian_cutoff
    config["special"]["direct_hessian"]["interaction_type"] = interaction_type
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    statistics = [
        dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)
    ]
    with torch_default_dtype(dtype):
        model = create_model(config, statistics, targets, [])
    return model.to(device=device, dtype=dtype)


def check_direct_hessian_training(device, dtype, joint, loss_name, interaction_type):
    model = small_model(dtype, device, joint, interaction_type=interaction_type)
    data = (
        Batch.from_data_list([make_graph(2, dtype=dtype), make_graph(4, dtype=dtype)])
        .to(device)
        .to_dict()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    values = []
    for _ in range(12):
        optimizer.zero_grad()
        if joint:
            output = model(data)
        else:
            with patch(
                "torch.autograd.grad",
                side_effect=AssertionError("Hessian forward called autograd"),
            ):
                output = model(data)
        loss = hessian_loss(output, data, "direct_hessian", loss_name)
        if joint:
            loss = loss + 0.01 * (
                output["energy"].square().mean() + output["forces"].square().mean()
            )
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        optimizer.step()
        values.append(loss.item())
    assert values[-1] < values[0]
    assert output["direct_hessian"].shape == (20, 3, 3)
    if not joint:
        model.eval()
        with torch.inference_mode():
            result = model(data)["direct_hessian"]
        assert torch.isfinite(result).all()


def check_direct_hessian_saved_model(device, tmp_path, interaction_type, parity):
    model = small_model(
        torch.float64,
        device,
        interaction_type=interaction_type,
        Lmax=3,
        lmax=1,
        parity=parity,
    ).eval()
    data = Batch.from_data_list([make_graph(2), make_graph(4)]).to(device).to_dict()
    data["positions"][2:] = data["positions"].new_tensor(
        [[0, 0, 0], [1.2, 0.1, 0.2], [0.2, 1.1, 0.1], [0.3, 0.2, 1.4]]
    )
    prepared = model.prepare_property_data(data)
    assert prepared is not data
    torch.testing.assert_close(
        prepared["direct_hessian_pair_index"], hessian_pair_indices(data["ptr"])
    )
    with patch.object(model, "get_target_property", return_value=["energy"]):
        assert model.prepare_property_data(data) is data
    with patch.object(model, "lmp", True):
        assert model.prepare_property_data(data) is data
    core = model.readout_fn
    assert isinstance(core.pair_representation, PairRepresentation)
    assert core.pair_representation.irreps_out.lmax == 3
    assert core.direct_hessian_readout.edge_readout.irreps_hidden.lmax == 3
    assert core.direct_hessian_readout.irreps_out == o3.Irreps("0e + 1e + 2e")
    node_irreps = core.representation.irreps_outs[-1]
    assert {ir for _, ir in node_irreps} == {ir for _, ir in core.target_irreps}
    expected = {
        ir_out
        for _, ir_in in node_irreps
        for _, ir_sh in o3.Irreps.spherical_harmonics(1)
        for ir_out in ir_in * ir_sh
        if ir_out.l <= 3
        and (
            parity is True
            or ir_out.p == (-1) ** ir_out.l
            or (parity is None and ir_out == o3.Irrep("1e"))
        )
    }
    assert {ir for _, ir in core.pair_representation.irreps_out} == expected
    assert {
        ir for _, ir in core.direct_hessian_readout.edge_readout.irreps_hidden
    } == expected
    assert o3.Irrep("0o") not in expected
    assert o3.Irrep("3e") not in expected
    for interaction in core.pair_representation.interactions:
        assert interaction.irreps_sh == o3.Irreps.spherical_harmonics(1)
    assert (core.pair_representation.irreps_out.count(o3.Irrep("1e")) > 0) == (
        parity is not False
    )
    if parity is None:
        assert o3.Irrep("1e") in {ir for _, ir in node_irreps}
        assert {ir for ir in expected if ir.p != (-1) ** ir.l} == {o3.Irrep("1e")}
    assert not any(
        isinstance(module, PairRepresentation)
        for module in core.direct_hessian_readout.modules()
    )
    calls = []
    hooks = [
        module.register_forward_hook(
            lambda module, inputs, output, name=name: calls.append(name)
        )
        for name, module in (
            ("node", core.representation),
            ("edge", core.pair_representation),
            ("readout", core.direct_hessian_readout),
        )
    ]
    with torch.inference_mode():
        reference = model(data)["direct_hessian"]
    for hook in hooks:
        hook.remove()
    assert calls == ["node", "edge", "readout"]
    antisymmetric = reference - reference.transpose(-1, -2)
    if parity is False:
        torch.testing.assert_close(antisymmetric, torch.zeros_like(antisymmetric))
    else:
        assert antisymmetric.abs().max() > 1e-10
    if parity is None:
        prediction = model(data)["direct_hessian"]
        prediction.square().sum().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().max() > 0 for gradient in gradients)
    assert "direct_hessian_pair_index" not in data
    assert "direct_hessian_edge_pairs" not in data
    path = tmp_path / "direct_hessian.pt"
    export_tace(model, str(path))
    for dtype in (torch.float32, torch.float64):
        loaded = load_tace(path, device=device, dtype=dtype).eval()
        values = {
            key: value.to(dtype) if value.is_floating_point() else value
            for key, value in data.items()
        }
        with torch.inference_mode():
            actual = loaded(values)["direct_hessian"]
        assert actual.dtype == dtype
        torch.testing.assert_close(actual.double(), reference, atol=2e-5, rtol=2e-5)
    with torch.inference_mode():
        for sign in (1, -1):
            rotation = sign * o3.rand_matrix(dtype=torch.float64, device=device)
            rotated = dict(data, positions=data["positions"] @ rotation.T + 2.1)
            torch.testing.assert_close(
                model(rotated)["direct_hessian"],
                rotation @ reference @ rotation.T,
                atol=1e-8,
                rtol=1e-8,
            )


def check_direct_hessian_periodic(device, dtype, interaction_type):
    model = small_model(
        dtype,
        device,
        joint=True,
        hessian_cutoff=3.2,
        interaction_type=interaction_type,
    )
    head = model.readout_fn.direct_hessian_readout
    representation = model.readout_fn.pair_representation
    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.2, 0.3], [1.4, 0.35, 0.2]],
        cell=[[2.5, 0, 0], [0.6, 2.8, 0], [0.3, 0.4, 3.0]],
        pbc=True,
    )

    def make_batch(structure):
        return (
            Batch.from_data_list(
                [
                    from_atoms(
                        build_element_lookup([1]),
                        item,
                        4.0,
                        None,
                        KeySpecification.from_defaults(),
                        ["energy", "forces", "direct_hessian"],
                        [],
                        dtype=dtype,
                    )
                    for item in (structure, Atoms("H"))
                ]
            )
            .to(device)
            .to_dict()
        )

    data = make_batch(atoms)
    prepared = prepare_direct_hessian_inputs(data, representation.cutoff)
    pairs = prepared["direct_hessian_edge_pairs"]
    assert pairs.numel() > 2
    assert set(pairs.tolist()) == {1, 2}
    features = torch.randn(3, head.irreps_in.dim, dtype=dtype, device=device)
    fidelity = data["batch"].new_zeros(3)
    actual = direct_hessian_output(
        head,
        representation,
        features,
        data["positions"],
        data["ptr"],
        data["batch"],
        data["node_attrs"],
        fidelity,
        data["lattice"],
        data["pbc"],
    )
    expected = features.new_zeros((5, 3, 3))
    expected[[0, 3, 4]] = head.to_cartesian(head.node_readout(features), fidelity)
    # Independently enumerate images, including multiple images of the same pair.
    shifts = torch.cartesian_prod(*[torch.arange(-3, 4, device=device)] * 3).to(dtype)
    for source, target in ((0, 1), (1, 0)):
        vectors = (
            data["positions"][target]
            - data["positions"][source]
            + shifts @ data["lattice"][0]
        )
        selected = torch.linalg.vector_norm(vectors, dim=-1) < representation.cutoff
        edge_shifts = shifts[selected]
        edge_index = (
            data["ptr"].new_tensor([[source], [target]]).expand(2, edge_shifts.size(0))
        )
        edge_feats = representation(
            features,
            data["node_attrs"],
            data["positions"],
            edge_index,
            edge_shifts,
            data["lattice"],
            data["batch"],
        )
        expected[source * 2 + target] = head.to_cartesian(
            head.edge_readout(edge_feats), fidelity[edge_index[0]]
        ).sum(0)
    expected = (expected + expected[[0, 2, 1, 3, 4]].transpose(-1, -2)) / 2
    torch.testing.assert_close(actual, expected)

    output = model(dict(data))
    assert output["direct_hessian"].shape == (5, 3, 3)
    assert all(
        key not in data
        for key in (
            "direct_hessian_pair_index",
            "direct_hessian_edge_pairs",
            "direct_hessian_edge_shifts",
        )
    )
    loss = sum(output[key].square().sum() for key in model.get_target_property())
    loss.backward()
    assert any(parameter.grad is not None for parameter in head.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    model.eval()
    expected = model(dict(data))
    wrapped = atoms.copy()
    wrapped.positions[1] += atoms.cell[0] - 2 * atoms.cell[1]
    actual = model(make_batch(wrapped))
    for key in model.get_target_property():
        torch.testing.assert_close(actual[key], expected[key], atol=2e-5, rtol=2e-5)
    for sign in (1, -1):
        rotation = sign * o3.rand_matrix(dtype=dtype, device=device)
        rotated = atoms.copy()
        rotated.positions = atoms.positions @ rotation.cpu().numpy().T
        rotated.set_cell(atoms.cell.array @ rotation.cpu().numpy().T)
        actual = model(make_batch(rotated))
        torch.testing.assert_close(
            actual["direct_hessian"],
            rotation @ expected["direct_hessian"] @ rotation.T,
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eval_property_scopes(device, dtype, tmp_path, monkeypatch):
    from tace.scripts import eval as eval_script

    targets = ["energy", "edge_forces", "direct_hessian"]
    model = small_model(dtype, device, joint=True).eval()
    model.reset_target_property(targets)
    structures = [
        Atoms("H3", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 8]]),
        Atoms("H", positions=[[0, 0, 0]]),
        Atoms("H2", positions=[[0, 0, 0], [0, 1, 0]]),
    ]
    filename = tmp_path / "structures.xyz"
    write(filename, structures)
    saved = []
    monkeypatch.setattr(eval_script, "load_tace", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        eval_script.ase.io, "write", lambda path, atoms, **kwargs: saved.extend(atoms)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tace-eval",
            "-i",
            str(filename),
            "-m",
            "unused.pt",
            "-b",
            "3",
            "--device",
            device.type,
            "--dtype",
            str(dtype).split(".")[-1],
            "--nl_backend",
            "ase",
        ],
    )
    eval_script.main()
    assert len(saved) == len(structures)
    for atoms, result in zip(structures, saved):
        graph = from_atoms(
            model.get_torch_element(),
            atoms,
            model.get_cutoff(),
            None,
            KeySpecification.from_defaults(),
            targets,
            [],
            training=False,
            dtype=dtype,
            neighborlist_backend="ase",
        )
        prediction = model(Batch.from_data_list([graph]).to(device))
        for key in targets:
            actual = np.asarray(result.info[f"TACE_{key}"])
            expected = prediction[key].detach().cpu().numpy()
            assert actual.size == expected.size
            np.testing.assert_allclose(
                actual.ravel(), expected.ravel(), atol=2e-5, rtol=2e-5
            )
        assert result.info["TACE_direct_hessian"].shape == (len(atoms) ** 2, 3, 3)


def test_hessian(device, tmp_path):
    assert PROPERTIES["hessian"]["shape"] == PROPERTIES["direct_hessian"]["shape"]
    assert (
        PROPERTIES["hessian"]["default_value_fn"]
        is PROPERTIES["direct_hessian"]["default_value_fn"]
    )
    for dtype in (torch.float32, torch.float64):
        for n in (1, 2, 3, 4):
            check_hessian_layouts(n, dtype, device)
    for shape in ((2, 3, 2, 3), (4, 9), (2, 2, 9), (2, 3)):
        check_hessian_invalid_shape(shape)
    check_hessian_data_roundtrip(tmp_path, "hessian")
    check_hessian_batch("hessian")
    check_hessian_invalid_labels("hessian")

    for dtype in (torch.float32, torch.float64):
        data = (
            Batch.from_data_list(
                [make_graph(n, dtype=dtype, key="hessian") for n in (2, 3)]
            )
            .to(device)
            .to_dict()
        )
        model = small_model(dtype, device, joint=True).eval()
        model.reset_target_property(["energy", "forces", "hessian"])
        actual = model(dict(data))["hessian"]
        assert actual.shape == data["hessian"].shape == (13, 3, 3)
        assert actual.dtype == dtype and actual.device == data["positions"].device

        model.reset_target_property(["energy"])
        reference = torch.autograd.functional.hessian(
            lambda positions: model(dict(data, positions=positions))["energy"].sum(),
            data["positions"],
        ).permute(0, 2, 1, 3)
        source, target = hessian_pair_indices(data["ptr"])
        torch.testing.assert_close(actual, reference[source, target])
        assert not reference[:2, 2:].any()

        for name in ("mse", "mae", "l2mae", "huber"):
            for reduction in ("structure", "component"):
                labels = dict(
                    data,
                    direct_hessian=data["hessian"],
                    direct_hessian_weight=data["hessian_weight"],
                )
                prediction = {"hessian": actual, "direct_hessian": actual}
                torch.testing.assert_close(
                    hessian_loss(prediction, labels, "hessian", name, reduction),
                    hessian_loss(prediction, labels, "direct_hessian", name, reduction),
                )


@pytest.mark.parametrize("interaction_type", ["cgtp", "o2"])
def test_direct_hessian(device, tmp_path, interaction_type):
    check_hessian_data_roundtrip(tmp_path, "direct_hessian")
    check_hessian_batch("direct_hessian")
    check_hessian_invalid_labels("direct_hessian")
    check_direct_hessian_missing_uncertainty(device)
    for reduction in ("structure", "component"):
        for loss in ("mse", "mae", "l2mae", "huber"):
            check_direct_hessian_loss(device, loss, reduction)
        check_direct_hessian_l2mae_weights(device, reduction)
    for huber_delta in (0.01, 0.5):
        check_direct_hessian_huber_threshold(device, huber_delta)
    check_direct_hessian_readout_symmetries(device, interaction_type)
    check_direct_hessian_node_edge_readouts(device, interaction_type)
    for dtype in (torch.float32, torch.float64):
        check_direct_hessian_periodic(device, dtype, interaction_type)
        for joint in (False, True):
            for loss in ("mse", "l2mae", "huber"):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(0)
                    check_direct_hessian_training(
                        device, dtype, joint, loss, interaction_type
                    )
    for parity in (False, True, None):
        check_direct_hessian_saved_model(device, tmp_path, interaction_type, parity)


@pytest.fixture
def small_system(double_precision):
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=3,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
        statistics=[
            dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)
        ],
    )
    config["radial_basis"]["hidden"] = [4]
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = dict(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]]),
        lattice=4.0 * torch.eye(3).unsqueeze(0),
        node_attrs=torch.ones(3, 1),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
        edge_shifts=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        batch=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3]),
        fidelity_idx=torch.zeros(1, dtype=torch.long),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    return config, {key: value.to(device) for key, value in data.items()}


def test_embedding_parity_paths(small_system):
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    config, data = small_system
    assert DEFAULT_MODEL_CONFIG["parity"] is None
    config["parity"] = None
    config["target_property"] = ["energy"]
    config["universal_embedding"]["magnetic_field"]["enable"] = True
    data["magnetic_field"] = data["positions"].new_tensor([[0.2, 0.1, -0.1]])
    model = TensorModel(e3nnTACE(**config)).to(data["positions"].device)
    core = model.readout_fn
    assert {(ir.l, ir.p) for _, ir in core.target_irreps} == {(0, 1)}
    assert {(ir.l, ir.p) for _, ir in core.representation.irreps_outs[-1]} == {(0, 1)}
    intermediate = core.representation.products[0].irreps_out
    assert (1, 1) in {(ir.l, ir.p) for _, ir in intermediate}
    assert {(ir.l, ir.p) for _, ir in intermediate if ir.p != (-1) ** ir.l} == {(1, 1)}
    output = model(data)
    assert torch.isfinite(output["energy"]).all()
    output["energy"].sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    "denoising,all_layers", [(False, False), (False, True), (True, False)]
)
def test_vector_readouts(small_system, monkeypatch, denoising, all_layers):
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    monkeypatch.setenv("TACE_USE_DENS", str(int(denoising)))
    config, data = small_system
    config["target_property"] = ["energy", "direct_forces"]
    config["readout_emlp"]["use_alllayer"] = all_layers
    data["forces"] = torch.randn_like(data["positions"])
    model = TensorModel(e3nnTACE(**config)).to(data["positions"].device).train()
    keys = ["direct_forces", "noise_vec"] if denoising else ["direct_forces"]
    output = model(data)
    for key in keys:
        assert output[key].shape == data["positions"].shape
        assert torch.isfinite(output[key]).all()
    sum(output[key].square().sum() for key in keys).backward()
    heads = [model.readout_fn.direct_forces_readouts]
    if denoising:
        heads.append(model.readout_fn.dens_noise_readouts)
    for head in heads:
        gradients = [p.grad for p in head.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
    model.eval()
    with torch.no_grad():
        output = model(data)
    assert all(torch.isfinite(output[key]).all() for key in keys)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("empty_edges", [False, True])
def test_edge_forces_match_position_derivatives(small_system, training, empty_edges):
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel
    from tace.models.utils import compute_forces_stress_from_edge_forces

    config, data = small_system
    if empty_edges:
        data["edge_index"] = data["edge_index"][:, :0]
        data["edge_shifts"] = data["edge_shifts"][:0]
    config["target_property"] = ["energy", "forces", "stress"]
    model = TensorModel(e3nnTACE(**config)).to(data["positions"].device).train(training)
    reference = model({key: value.clone() for key, value in data.items()})
    model.reset_target_property(["energy", "edge_forces"])
    edge_data = {key: value.clone() for key, value in data.items()}
    graph = model.prepare_graph(edge_data)
    assert graph.edge_vector.requires_grad
    assert not edge_data["positions"].requires_grad
    output = model(edge_data)
    forces, stress = compute_forces_stress_from_edge_forces(
        output["edge_forces"],
        graph.edge_vector,
        data["edge_index"],
        data["batch"],
        data["lattice"],
        data["pbc"],
    )
    torch.testing.assert_close(output["energy"], reference["energy"])
    torch.testing.assert_close(forces, reference["forces"], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(stress, reference["stress"], atol=1e-10, rtol=1e-10)
    if training and not empty_edges:
        parameters = tuple(model.parameters())
        actual = torch.autograd.grad(
            forces.square().sum() + stress.square().sum(), parameters, allow_unused=True
        )
        expected = torch.autograd.grad(
            reference["forces"].square().sum() + reference["stress"].square().sum(),
            parameters,
            allow_unused=True,
        )
        for value, ref in zip(actual, expected):
            assert (value is None) == (ref is None)
            if value is not None:
                torch.testing.assert_close(value, ref, atol=1e-9, rtol=1e-9)


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


def test_tace_les_reads_out_latent_quantities_and_passes_them_to_upstream():
    pytest.importorskip("les")
    model = TACELES(
        {
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
    ).to(dtype=torch.get_default_dtype())
    assert model.batch_external_field
    assert model.scale == {
        "latent_charges": 0.1,
        "latent_dipoles": 0.1,
        "latent_quads": 0.1,
        "latent_kappas": 0.01,
        "latent_alphas": 0.01,
    }
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
        num_atoms_arange=torch.arange(4),
        positions=positions,
        lattice=torch.zeros(2, 3, 3),
        batch=torch.tensor([0, 0, 0, 1]),
        atomic_numbers=torch.tensor([1, 8, 1, 8]),
        external_field=torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]),
        pbc=torch.zeros(2, 3, dtype=torch.bool),
    )

    assert output["les_energy"].shape == (2,)
    assert output["les_latent_charges"].shape == (4,)
    assert output["les_latent_dipoles"].shape == (4, 3)
    assert output["les_latent_quads"].shape == (4, 3, 3)
    assert output["les_latent_alphas"].shape == (4, 3, 3)
    assert output["les_latent_kappas"].shape == (4,)
    assert output["les_born_effective_charges"].shape == (4, 2, 3, 3)

    gradients = torch.autograd.grad(
        output["les_energy"].sum(),
        (positions, descriptors[0]),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_nonbatch_external_field_uses_first_graph_without_value_checks(monkeypatch):
    pytest.importorskip("les")
    model = TACELES(
        {},
        num_layers=1,
        hidden_channel=[8],
        bias=True,
        num_elements=1,
        num_fidelities=1,
        use_alllayer=False,
        parity=False,
        irreps_in=[o3.Irreps("4x0e")],
    )
    model.batch_external_field = False
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(model, "_solve", solve)
    external_field = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    model._run_les(
        atomic_numbers=torch.tensor([1, 1]),
        latent_charges=torch.zeros(2),
        latent_dipoles=None,
        latent_quads=None,
        latent_alphas=None,
        latent_kappas=None,
        positions=torch.zeros(2, 3),
        lattice=torch.zeros(2, 3, 3),
        batch=torch.tensor([0, 1]),
        external_field=external_field,
    )

    assert len(calls) == 1
    assert torch.equal(calls[0]["e_ext"], external_field[0])
