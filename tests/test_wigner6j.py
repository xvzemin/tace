import inspect

import pytest
import torch
from e3nn import o3

from tace.models._e3nn.fused import O3ScatterTensorProduct, uvuTensorProduct
from tace.models._e3nn.inter import (
    O3GeneralizedWigner6jInteraction,
    O3Wigner6jMagneticInteraction,
)
from tace.models._e3nn.wigner6j import (
    O3Wigner6jScatterTensorProduct,
    sympy_wigner_6j,
    wigner_6j,
)
from tace.models.mag import MagneticBasis

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


def _build_tensor_product(*, weight_level="edge", register_reference=True):
    irreps_node_feats = o3.Irreps("2x0e + 2x1o + 2x1e")
    irreps_edge_attrs = o3.Irreps.spherical_harmonics(2, p=-1)
    irreps_out = o3.Irreps("2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o")
    module = O3Wigner6jScatterTensorProduct(
        irreps_node_feats,
        irreps_edge_attrs,
        irreps_out,
        extra_irreps_node_attrs=o3.Irreps("0o + 0e + 1o + 1e + 2o + 2e"),
        weight_level=weight_level,
        register_reference=register_reference,
    )
    assert isinstance(module.recoupled_node_node_tp, uvuTensorProduct)
    assert module.recoupled_node_node_tp.shared_weights == (weight_level == "edge")
    assert isinstance(module.recoupled_node_edge_tp, O3ScatterTensorProduct)
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


@pytest.mark.parametrize("weight_level", ["edge", "node"])
def test_wigner6j_recoupling_matches_reference_and_gradients(
    weight_level,
):
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


@pytest.mark.parametrize("weight_level", ["edge", "node"])
@pytest.mark.parametrize("improper", [False, True])
def test_wigner6j_tensor_product_is_o3_equivariant(
    improper,
    weight_level,
):
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    module = _build_tensor_product(weight_level=weight_level)
    inputs = _random_inputs(module)
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

    output = module(*inputs)
    rotated_output = module(*rotated_inputs)
    expected = output @ output_rotation.T
    torch.testing.assert_close(
        rotated_output,
        expected,
        atol=3.0e-11,
        rtol=3.0e-11,
    )


def test_wigner6j_requires_extra_irreps_node_attrs():
    with pytest.raises(TypeError, match="extra_irreps_node_attrs"):
        O3Wigner6jScatterTensorProduct(
            o3.Irreps("1x0e"),
            o3.Irreps("1x0e"),
            o3.Irreps("1x0e"),
            weight_level="edge",
        )


def test_wigner6j_requires_weight_level():
    with pytest.raises(TypeError, match="weight_level"):
        O3Wigner6jScatterTensorProduct(
            o3.Irreps("1x0e"),
            o3.Irreps("1x0e"),
            o3.Irreps("1x0e"),
            o3.Irreps("1x0e"),
        )


def test_wigner6j_does_not_register_reference_by_default():
    module = _build_tensor_product(register_reference=False)
    assert not hasattr(module, "reference_node_edge_tp")
    assert not hasattr(module, "reference_edge_edge_tp")
    assert not any(name.startswith("reference_") for name in module.state_dict())
    with pytest.raises(RuntimeError, match="register_reference=True"):
        module.forward_reference(*_random_inputs(module))


def test_wigner6j_rejects_unknown_weight_level():
    with pytest.raises(ValueError, match="weight_level"):
        _build_tensor_product(weight_level="graph")


def test_generalized_wigner6j_interaction_is_abstract():
    assert inspect.isabstract(O3GeneralizedWigner6jInteraction)
    assert issubclass(
        O3Wigner6jMagneticInteraction,
        O3GeneralizedWigner6jInteraction,
    )


def _build_interaction():
    module = O3Wigner6jMagneticInteraction(
        layer=0,
        num_layers=1,
        num_elements=2,
        avg_num_neighbors=4.0,
        mmax=2,
        Lmax=2,
        lmax=2,
        correlation=[1],
        num_channel=2,
        edge_feats_channel=4,
        target_irreps=o3.Irreps("0e"),
        num_radial_basis=4,
        num_mag_radial_basis=3,
        magnetic_irreps=o3.Irreps("1e"),
        radial_mlp=[8],
        radial_bias=True,
        irreps_in=o3.Irreps("2x0e + 2x1o"),
        scalar_act=None,
        tensor_act=None,
        edge_ace_hidden=None,
        parity=True,
        nonlinear=None,
    )
    return module.to(DEVICE)


@pytest.mark.parametrize("weight_level", ["edge", "node"])
def test_wigner6j_interaction_weight_levels(weight_level, monkeypatch):
    torch.manual_seed(2)
    monkeypatch.setattr(O3Wigner6jMagneticInteraction, "weight_level", weight_level)
    module = _build_interaction()
    assert module.extra_irreps_node_attrs == module.magnetic_irreps
    assert not hasattr(module, "magnetic_angular_basis")
    expected_edge_input = module.edge_feats_channel + module.num_mag_radial_basis
    assert module.edge_info.dims[0] == expected_edge_input
    num_nodes = 5
    num_edges = 9
    edge_index = torch.stack(
        [
            torch.randint(num_nodes, (num_edges,), device=DEVICE),
            torch.randint(num_nodes, (num_edges,), device=DEVICE),
        ]
    )
    node_feats = torch.randn(
        num_nodes,
        module.irreps_in.dim,
        device=DEVICE,
        requires_grad=True,
    )
    node_attrs = torch.nn.functional.one_hot(
        torch.randint(2, (num_nodes,), device=DEVICE),
        2,
    ).to(node_feats)
    edge_feats = torch.randn(num_edges, 4, device=DEVICE, requires_grad=True)
    edge_attrs = torch.randn(
        num_edges,
        module.irreps_sh.dim,
        device=DEVICE,
        requires_grad=True,
    )
    initial_noncollinear_magmoms = torch.randn(
        num_nodes,
        3,
        device=DEVICE,
        requires_grad=True,
    )
    magnetic_basis = MagneticBasis(
        [4.0, 4.0],
        num_basis=4,
        magnetic_irreps=module.magnetic_irreps,
        num_elements=2,
    ).to(DEVICE)
    magnetic_radial_basis, magnetic_node_attrs = magnetic_basis(
        initial_noncollinear_magmoms,
        node_attrs,
    )
    magnetic_radial_basis = magnetic_radial_basis[..., 1:]

    output = module._compute_messages(
        node_feats,
        node_attrs,
        None,
        edge_feats,
        edge_attrs,
        edge_index,
        torch.rand(num_edges, 1, device=DEVICE),
        magnetic_radial_basis=magnetic_radial_basis,
        magnetic_node_attrs=magnetic_node_attrs,
    )
    assert output.shape == (num_nodes, module.rejector.irreps_out.dim)
    output.square().sum().backward()
