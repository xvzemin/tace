from copy import deepcopy

import pytest
import torch
from e3nn import o3

from tace.models._e3nn.base import _to_possible_tp_irreps
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.fused import uvuTensorProduct
from tace.models._e3nn.tace import e3nnTACE
from tace.models._e3nn.wigner6j import (
    O3Wigner6jScatterTensorProduct,
    sympy_wigner_6j,
    wigner_6j,
)
from tace.models.adapter import TensorModel
from tace.models.time_reversal import spherical_harmonics_irreps, supports_time_reversal

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_standard_wigner_6j_symbol():
    assert sympy_wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1.0 / 6.0)
    assert wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1.0 / 2.0)


@pytest.mark.parametrize("shared_weights", [False, True])
def test_uvu_tensor_product_oeq_matches_e3nn(shared_weights, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("OEQ requires CUDA")
    pytest.importorskip("openequivariance")
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    irreps_in1 = o3.Irreps("2x0e + 2x1o")
    irreps_in2 = o3.Irreps("1x1e")
    irreps_out = o3.Irreps("2x1e + 2x0o + 2x1o + 2x2o")
    instructions = [
        (0, 0, 0, "uvu", True, 1.0),
        (1, 0, 1, "uvu", True, 1.0),
        (1, 0, 2, "uvu", True, 1.0),
        (1, 0, 3, "uvu", True, 1.0),
    ]

    monkeypatch.setenv("TACE_USE_OEQ", "0")
    reference = uvuTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights=shared_weights,
    ).to(DEVICE)
    monkeypatch.setenv("TACE_USE_OEQ", "1")
    actual = uvuTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights=shared_weights,
    ).to(DEVICE)
    assert actual.use_oeq
    assert hasattr(actual, "fused_tp")
    assert not hasattr(reference, "fused_tp")

    num_nodes = 7
    x = torch.randn(num_nodes, irreps_in1.dim, device=DEVICE, requires_grad=True)
    y = torch.randn(num_nodes, irreps_in2.dim, device=DEVICE, requires_grad=True)
    weight_shape = (
        (reference.weight_numel,)
        if shared_weights
        else (num_nodes, reference.weight_numel)
    )
    weights = torch.randn(*weight_shape, device=DEVICE, requires_grad=True)

    expected = reference(x, y, weights)
    observed = actual(x, y, weights)
    torch.testing.assert_close(observed, expected, atol=2.0e-12, rtol=2.0e-12)

    grad_output = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(
        (expected * grad_output).sum(), (x, y, weights), retain_graph=True
    )
    observed_grads = torch.autograd.grad(
        (observed * grad_output).sum(), (x, y, weights)
    )
    for observed_grad, expected_grad in zip(observed_grads, expected_grads):
        torch.testing.assert_close(
            observed_grad,
            expected_grad,
            atol=3.0e-12,
            rtol=3.0e-12,
        )


def _build_tensor_product(*, weight_level="edge"):
    irreps_node_feats = o3.Irreps("2x0e + 2x1o + 2x1e")
    irreps_edge_attrs = o3.Irreps.spherical_harmonics(2, p=-1)
    irreps_out = o3.Irreps("2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o")
    module = O3Wigner6jScatterTensorProduct(
        irreps_node_feats,
        irreps_edge_attrs,
        irreps_out,
        extra_irreps_node_attrs=o3.Irreps("0o + 0e + 1o + 1e + 2o + 2e"),
        weight_level=weight_level,
        register_reference=True,
    )
    return module.to(DEVICE)


def _random_inputs(module, *, requires_grad=False):
    num_nodes = 5
    num_edges = 11
    edge_index = torch.stack(
        [
            torch.randint(num_nodes, (num_edges,), device=DEVICE),
            torch.randint(num_nodes, (num_edges,), device=DEVICE),
        ]
    )
    node_feats = torch.randn(
        num_nodes,
        module.irreps_node_feats.dim,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    edge_attrs = torch.randn(
        num_edges,
        module.irreps_edge_attrs.dim,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    extra_node_attrs = torch.randn(
        num_nodes,
        module.extra_irreps_node_attrs.dim,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    edge_weights = torch.randn(
        num_edges,
        module.edge_weight_numel,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    num_extra_weights = num_edges if module.weight_level == "edge" else num_nodes
    extra_weights = torch.randn(
        num_extra_weights,
        module.extra_weight_numel,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    return (
        node_feats,
        edge_attrs,
        extra_node_attrs,
        edge_weights,
        extra_weights,
        edge_index,
    )


@pytest.mark.parametrize(
    ("weight_level", "improper"),
    [("edge", False), ("node", True)],
)
def test_wigner6j_matches_reference_gradients_and_o3(weight_level, improper):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    module = _build_tensor_product(weight_level=weight_level)
    inputs = _random_inputs(module, requires_grad=True)

    recoupled = module(*inputs)
    reference = module.forward_reference(*inputs)
    torch.testing.assert_close(recoupled, reference, atol=2.0e-12, rtol=2.0e-12)

    grad_output = torch.randn_like(recoupled)
    differentiable_inputs = inputs[:-1]
    recoupled_grads = torch.autograd.grad(
        (recoupled * grad_output).sum(),
        differentiable_inputs,
        retain_graph=True,
    )
    reference_grads = torch.autograd.grad(
        (reference * grad_output).sum(),
        differentiable_inputs,
    )
    for recoupled_grad, reference_grad in zip(recoupled_grads, reference_grads):
        torch.testing.assert_close(
            recoupled_grad,
            reference_grad,
            atol=3.0e-12,
            rtol=3.0e-12,
        )

    node_feats, edge_attrs, extra_node_attrs, edge_weights, extra_weights, _ = inputs
    rotation = o3.rand_matrix(dtype=torch.float64)
    if improper:
        rotation = -rotation
    node_rotation = module.irreps_node_feats.D_from_matrix(rotation).to(DEVICE)
    edge_rotation = module.irreps_edge_attrs.D_from_matrix(rotation).to(DEVICE)
    extra_rotation = module.extra_irreps_node_attrs.D_from_matrix(rotation).to(DEVICE)
    output_rotation = module.irreps_out.D_from_matrix(rotation).to(DEVICE)
    rotated_inputs = (
        node_feats @ node_rotation.T,
        edge_attrs @ edge_rotation.T,
        extra_node_attrs @ extra_rotation.T,
        edge_weights,
        extra_weights,
        inputs[-1],
    )

    rotated_output = module(*rotated_inputs)
    expected = recoupled @ output_rotation.T
    torch.testing.assert_close(
        rotated_output,
        expected,
        atol=3.0e-11,
        rtol=3.0e-11,
    )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_wigner6j_matches_reference_and_time_reversal():
    node_irreps = o3.Irreps("2x0ee + 2x1eo")
    edge_irreps = spherical_harmonics_irreps(1, p=-1)
    magnetic_irreps = spherical_harmonics_irreps(
        1,
        p=1,
        time_reversal=-1,
    )
    intermediate_irreps = _to_possible_tp_irreps(
        node_irreps,
        edge_irreps,
        parity=True,
        lmax=2,
    )
    output_irreps = (
        _to_possible_tp_irreps(
            intermediate_irreps,
            magnetic_irreps,
            parity=True,
            lmax=1,
        )
        * 2
    ).regroup()
    tensor_product = O3Wigner6jScatterTensorProduct(
        node_irreps,
        edge_irreps,
        output_irreps,
        magnetic_irreps,
        weight_level="edge",
        register_reference=True,
    )
    assert (
        tensor_product.recoupled_node_edge_tp.irreps_out
        == tensor_product.reference_edge_edge_tp.irreps_out
        == tensor_product.irreps_out
    )

    num_nodes = 4
    num_edges = 7
    edge_index = torch.randint(num_nodes, (2, num_edges))
    node_features = torch.randn(num_nodes, node_irreps.dim)
    edge_features = torch.randn(num_edges, edge_irreps.dim)
    magnetic_features = torch.randn(num_nodes, magnetic_irreps.dim)
    edge_weights = torch.randn(num_edges, tensor_product.edge_weight_numel)
    magnetic_weights = torch.randn(num_edges, tensor_product.extra_weight_numel)

    def reverse(features, irreps):
        matrix = irreps.D_from_matrix(
            -torch.eye(3),
            parity=False,
            time_reversal=True,
        )
        return features @ matrix.T

    output = tensor_product(
        node_features,
        edge_features,
        magnetic_features,
        edge_weights,
        magnetic_weights,
        edge_index,
    )
    reference = tensor_product.forward_reference(
        node_features,
        edge_features,
        magnetic_features,
        edge_weights,
        magnetic_weights,
        edge_index,
    )
    torch.testing.assert_close(output, reference)
    observed = tensor_product(
        reverse(node_features, node_irreps),
        reverse(edge_features, edge_irreps),
        reverse(magnetic_features, magnetic_irreps),
        edge_weights,
        magnetic_weights,
        edge_index,
    )
    torch.testing.assert_close(
        observed,
        reverse(output, tensor_product.irreps_out),
    )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
@pytest.mark.parametrize(
    ("kernel", "environment"),
    [("CUE", "TACE_USE_CUE"), ("OEQ", "TACE_USE_OEQ")],
)
def test_time_reversal_wigner6j_rejects_scatter_acceleration(
    monkeypatch,
    kernel,
    environment,
):
    for variable in ("TACE_USE_CUE", "TACE_USE_OEQ"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(environment, "1")
    with pytest.raises(ValueError, match=f"{kernel} does not support"):
        O3Wigner6jScatterTensorProduct(
            o3.Irreps("2x0ee + 2x1eo"),
            spherical_harmonics_irreps(1, p=-1),
            o3.Irreps("2x0ee + 2x1eo"),
            spherical_harmonics_irreps(1, p=1, time_reversal=-1),
            weight_level="edge",
        )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_w6j_mag_model_is_time_reversal_invariant(monkeypatch):
    for environment in ("TACE_USE_CUE", "TACE_USE_OEQ"):
        monkeypatch.delenv(environment, raising=False)

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=4.0,
        max_neighbors=None,
        statistics=[
            {
                "atomic_numbers": [26],
                "avg_num_neighbors": 2.0,
                "atomic_energy": {26: 0.0},
            }
        ],
        num_layers=1,
        num_channel=2,
        Lmax=1,
        lmax=1,
        parity=True,
        target_property=["energy"],
    )
    config["fidelity"] = [{"name": "PBE", "atomic_energy": None, "magnetic_scale": 2.0}]
    config["atomic_basis"]["type"] = "w6j_mag"
    config["radial_basis"]["hidden"] = [4]
    config["angular_basis"]["magnetic_Lmax"] = 1
    config["readout_emlp"]["hidden"] = [2]
    config["readout_emlp"]["use_one_body_magmoms"] = False
    config["scale_shift"]["enable"] = False

    model = TensorModel(e3nnTACE(**config)).double().eval()
    representation = model.readout_fn.representation
    interaction = representation.interactions[0]
    assert representation.use_time_reversal
    assert representation.node_updates is None
    assert representation.magnetic_edge_irreps_out is None
    assert {str(ir) for _, ir in interaction.irrreps_tp_out} == {
        "0ee",
        "0oo",
        "1eo",
        "1oe",
        "1oo",
    }
    assert interaction.edge_info.dims == [
        representation.edge_updates[0].out_dim
        + config["radial_basis"]["num_mag_radial_basis"],
        *config["radial_basis"]["hidden"],
        interaction.rejector.edge_weight_numel,
    ]
    assert interaction.magnetic_info.dims == [
        interaction.edge_info.dims[0],
        interaction.rejector.extra_weight_numel,
    ]

    moments = torch.tensor(
        [[1.0, 0.2, 0.3], [-0.4, 0.5, 0.6]],
        dtype=torch.float64,
    )
    data = {
        "positions": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        "node_attrs": torch.ones(2, 1, dtype=torch.float64),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_shifts": torch.zeros(2, 3, dtype=torch.float64),
        "lattice": torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
        "batch": torch.zeros(2, dtype=torch.int64),
        "ptr": torch.tensor([0, 2]),
        "initial_noncollinear_magmoms": moments,
    }
    energy = model(data)["energy"]
    data["initial_noncollinear_magmoms"] = -moments
    reversed_energy = model(data)["energy"]
    torch.testing.assert_close(reversed_energy, energy)
