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

from tace.dataset.element import build_element_lookup
from tace.dataset.graph import from_atoms
from tace.dataset.hessian import hessian_blocks, hessian_pair_indices
from tace.dataset.quantity import KeySpecification
from tace.dataset.read import ase_db_connect, ase_io_read
from tace.lightning import create_model, export_tace, load_tace
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.hessian import DirectHessianReadout
from tace.utils.loss.hessian import direct_hessian_loss
from tace.utils.loss.normal import NormalLoss
from tace.utils.loss.uncertainty import UncertaintyLoss
from tace.utils.mask_metrics import MaskMAE
from tace.utils.utils import torch_default_dtype


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_hessian_layouts(n, dtype, device):
    matrix = torch.arange(9 * n * n, dtype=dtype, device=device).reshape(3 * n, 3 * n).T
    blocks = matrix.reshape(n, 3, n, 3).permute(0, 2, 1, 3).contiguous()
    for value in (matrix, matrix.flatten(), blocks, blocks.reshape(n * n, 3, 3)):
        result = hessian_blocks(value, n)
        torch.testing.assert_close(result, blocks.reshape(-1, 3, 3))
        assert result.is_contiguous()
        assert result.dtype == dtype and result.device == matrix.device


def check_hessian_invalid_shape(shape):
    with pytest.raises(ValueError, match="Invalid Hessian shape"):
        hessian_blocks(torch.zeros(shape), 2)


def check_hessian_data_roundtrip(tmp_path):
    matrix = np.arange(81, dtype=float).reshape(9, 9)
    blocks = matrix.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)
    structures = []
    with connect(tmp_path / "hessian.db") as database:
        for value in (matrix, matrix.ravel(), blocks, blocks.reshape(9, 3, 3)):
            atoms = Atoms("H3", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
            atoms.info["direct_hessian"] = value
            database.write(atoms, data=atoms.info)
            structures.append(atoms)
    write(tmp_path / "hessian.xyz", structures)
    expected = torch.from_numpy(blocks.reshape(9, 3, 3))
    for reader, filename in (
        (ase_db_connect, "hessian.db"),
        (ase_io_read, "hessian.xyz"),
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
                ["direct_hessian"],
                [],
                neighborlist_backend="ase",
                dtype=torch.float64,
            )
            torch.testing.assert_close(graph.direct_hessian, expected)
            assert graph.direct_hessian.is_contiguous()


def make_graph(n, labeled=True, dtype=torch.float64):
    atoms = Atoms(numbers=[1] * n, positions=np.arange(n * 3).reshape(n, 3) * 0.2)
    if labeled:
        atoms.info["reference_hessian"] = np.eye(3 * n)
    keys = KeySpecification.from_defaults()
    keys.info_keys["direct_hessian"] = "reference_hessian"
    return from_atoms(
        build_element_lookup([1]),
        atoms,
        4.0,
        None,
        keys,
        ["direct_hessian"],
        [],
        neighborlist_backend="ase",
        dtype=dtype,
    )


def check_hessian_batch():
    graphs = [make_graph(2), make_graph(4), make_graph(1, labeled=False)]
    batch = Batch.from_data_list(graphs)
    assert batch.direct_hessian.shape == (21, 3, 3)
    torch.testing.assert_close(
        batch.hessian_pair_index, hessian_pair_indices(batch.ptr)
    )
    assert batch.direct_hessian_weight.tolist() == [1, 1, 0]
    assert not batch.direct_hessian_mask[-1].any()
    metric = MaskMAE("direct_hessian", 1.0)
    prediction = batch.direct_hessian + 1
    prediction[-1] = 100
    metric.update(prediction, batch.direct_hessian, batch.to_dict())
    torch.testing.assert_close(metric.compute(), torch.tensor(1.0))


def check_hessian_missing_uncertainty(device):
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


def check_hessian_partial_labels():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.info["reference"] = np.eye(6)
    atoms.info["reference"][0, 1] = float("nan")
    keys = KeySpecification.from_defaults()
    keys.info_keys["direct_hessian"] = "reference"
    graph = from_atoms(
        build_element_lookup([1]),
        atoms,
        4.0,
        None,
        keys,
        ["direct_hessian"],
        [],
        neighborlist_backend="ase",
    )
    assert graph.direct_hessian_mask.sum() == 35
    assert torch.isfinite(graph.direct_hessian).all()
    assert graph.direct_hessian_weight == 1


def check_hessian_loss(device, loss, reduction):
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
    target[:] = float("nan")
    prediction.grad = None
    value = fn({"direct_hessian": prediction}, label)
    value.backward()
    assert value == 0 and not prediction.grad.any()


def check_hessian_l2mae_masks(device, reduction):
    label = Batch.from_data_list([make_graph(2), make_graph(3)]).to(device).to_dict()
    target = label["direct_hessian"]
    target[:] = float("nan")
    target[0, 0, :2] = 0
    target[2] = 0
    target[4, 1, 0] = 0
    label["direct_hessian_mask"] = torch.isfinite(target)
    label["entropy"][:] = torch.tensor([2, 3], device=device)
    prediction = target.clone()
    prediction[0, 0, :2] = torch.tensor([3, 4], device=device)
    prediction[4, 1, 0] = 12
    prediction.requires_grad_()
    value = direct_hessian_loss(
        {"direct_hessian": prediction}, label, "l2mae", reduction
    )
    expected = (
        (2 * 5 / 2 + 3 * 12) / 2 if reduction == "structure" else (2 * 5 + 3 * 12) / 3
    )
    torch.testing.assert_close(value, value.new_tensor(expected))
    value.backward()
    assert torch.isfinite(prediction.grad).all()
    assert not prediction.grad[~label["direct_hessian_mask"]].any()
    assert not prediction.grad[2].any()


def check_hessian_huber_threshold(device, huber_delta):
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


def check_hessian_readout_symmetries(device):
    radial = deepcopy(DEFAULT_MODEL_CONFIG["radial_basis"])
    radial.update(hidden=[4], mlp_type="mlp_v2")
    with torch_default_dtype(torch.float64):
        head = DirectHessianReadout("2x0e+2x1o+2x2e", 2, 2, [1, 8], 16.0, radial).to(
            device
        )
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
    expected = head(x, positions, ptr, batch, node_attrs, fidelity)
    blocks = expected[:9].reshape(3, 3, 3, 3)
    torch.testing.assert_close(blocks, blocks.permute(1, 0, 3, 2))
    assert (blocks[0, 1] - blocks[0, 1].T).abs().max() > 1e-8
    for sign in (1, -1):
        rotation = sign * o3.rand_matrix(dtype=torch.float64, device=device)
        with torch_default_dtype(torch.float64):
            d = head.irreps_in.D_from_matrix(rotation.cpu()).to(device)
        actual = head(
            x @ d.T, positions @ rotation.T + 1.7, ptr, batch, node_attrs, fidelity
        )
        torch.testing.assert_close(
            actual, rotation @ expected @ rotation.T, atol=1e-8, rtol=1e-8
        )
    permutation = torch.tensor([2, 0, 1, 3], device=device)
    actual = head(
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


def check_hessian_node_edge_readouts(device, gate_m0):
    radial = deepcopy(DEFAULT_MODEL_CONFIG["radial_basis"])
    radial.update(hidden=[4], bias=True)
    with torch_default_dtype(torch.float64):
        head = DirectHessianReadout(
            "2x0e+2x1o+2x2e",
            2,
            2,
            [1, 8],
            4.0,
            radial,
            gate_m0=gate_m0,
            bias=True,
        ).to(device)
    x = torch.randn(3, head.irreps_in.dim, dtype=torch.float64, device=device)
    positions = x.new_tensor([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
    ptr = torch.tensor([0, 2, 3], device=device)
    batch = torch.tensor([0, 0, 1], device=device)
    node_attrs = x.new_tensor([[1, 0], [0, 1], [1, 0]])
    fidelity = torch.tensor([0, 0, 1], device=device)
    output = head(x, positions, ptr, batch, node_attrs, fidelity)
    node_blocks = head.to_cartesian(head.node_readout(x), fidelity)
    torch.testing.assert_close(
        output[[0, 3, 4]], (node_blocks + node_blocks.transpose(-1, -2)) / 2
    )

    edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
    edge_feats, cutoff = head.radial_basis(
        x.new_ones(2, 1), node_attrs, edge_index, head.atomic_numbers
    )
    edge_attrs = head.angular_basis(positions[edge_index[1]] - positions[edge_index[0]])
    edge_coefficients = head.interaction(
        x, node_attrs, edge_feats, edge_attrs, edge_index, cutoff
    )
    assert edge_coefficients.shape == (2, 18)
    edge_blocks = head.to_cartesian(edge_coefficients, fidelity[edge_index[0]])
    torch.testing.assert_close(
        output[[1, 2]], (edge_blocks + edge_blocks.flip(0).transpose(-1, -2)) / 2
    )

    # No self edge or cross-structure edge reaches the interaction.
    with patch.object(
        head.interaction, "forward", wraps=head.interaction.forward
    ) as call:
        head(x, positions, ptr, batch, node_attrs, fidelity)
        torch.testing.assert_close(call.call_args.args[4], edge_index)
    positions[1, 0] = 4.1
    with patch.object(head.interaction, "forward", side_effect=AssertionError):
        outside = head(x, positions, ptr, batch, node_attrs, fidelity)
    assert not outside[[1, 2]].any()
    torch.testing.assert_close(outside[[0, 3, 4]], output[[0, 3, 4]])

    # Biases in the radial MLP must not reintroduce messages at the cutoff.
    for parameter in head.interaction.edge_info.parameters():
        if parameter.ndim == 1:
            with torch.no_grad():
                parameter.fill_(0.2)
    positions[1, 0] = 4.0 - 1e-3
    positions.requires_grad_()
    near = head(x, positions, ptr, batch, node_attrs, fidelity)[1]
    first = torch.autograd.grad(near.sum(), positions, create_graph=True)[0]
    second = torch.autograd.grad(first[1, 0], positions)[0]
    assert near.abs().max() < 1e-7
    assert first.abs().max() < 1e-4
    assert second.abs().max() < 0.1


def small_model(dtype, device, joint=False):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    targets = ["energy", "forces", "direct_hessian"] if joint else ["direct_hessian"]
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=2,
        lmax=2,
        cutoff=4.0,
        max_neighbors=None,
        fidelity=[{"name": "DFT"}],
        target_property=targets,
    )
    config["radial_basis"].update(hidden=[4], apply_cutoff=False)
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    statistics = [
        dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)
    ]
    with torch_default_dtype(dtype):
        model = create_model(config, statistics, targets, [])
    return model.to(device=device, dtype=dtype)


def check_hessian_training(device, dtype, joint, loss_name):
    model = small_model(dtype, device, joint)
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
        loss = direct_hessian_loss(output, data, loss_name)
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


def check_hessian_saved_model(device, tmp_path):
    model = small_model(torch.float64, device).eval()
    data = Batch.from_data_list([make_graph(2), make_graph(4)]).to(device).to_dict()
    with torch.inference_mode():
        reference = model(data)["direct_hessian"]
        # The complete pair readout is independent of neighbor-list pair counts.
        matrix = data.copy()
        matrix.pop("hessian_pair_index")
        torch.testing.assert_close(model(matrix)["direct_hessian"], reference)
    path = tmp_path / "hessian.pt"
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


def test_direct_hessian(device, tmp_path):
    for dtype in (torch.float32, torch.float64):
        for n in (1, 2, 3, 4):
            check_hessian_layouts(n, dtype, device)
    for shape in ((2, 3, 2, 3), (4, 9), (2, 2, 9), (2, 3)):
        check_hessian_invalid_shape(shape)
    check_hessian_data_roundtrip(tmp_path)
    check_hessian_batch()
    check_hessian_partial_labels()
    check_hessian_missing_uncertainty(device)
    for reduction in ("structure", "component"):
        for loss in ("mse", "mae", "l2mae", "huber"):
            check_hessian_loss(device, loss, reduction)
        check_hessian_l2mae_masks(device, reduction)
    for huber_delta in (0.01, 0.5):
        check_hessian_huber_threshold(device, huber_delta)
    check_hessian_readout_symmetries(device)
    for gate_m0 in (False, True):
        check_hessian_node_edge_readouts(device, gate_m0)
    for dtype in (torch.float32, torch.float64):
        for joint in (False, True):
            for loss in ("mse", "l2mae", "huber"):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(0)
                    check_hessian_training(device, dtype, joint, loss)
    check_hessian_saved_model(device, tmp_path)
