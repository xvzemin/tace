"""TACE-OAM-XL-v2 execution, gradients and optional TACE integration."""

import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx.conv import O3TensorProductConv
from equivariantx.models.tace.tace_oam_xl_v2 import TACEOAMXLV2
from equivariantx.models.tace.tace_oam_xl_v2.angular import (
    aligned_convolution,
    packed_frame,
)
from equivariantx.models.tace.tace_oam_xl_v2.edge import edge_contract
from equivariantx.models.tace.tace_oam_xl_v2.gated import (
    gated_product,
    gated_product_metadata,
)
from equivariantx.models.tace.tace_oam_xl_v2.graph import (
    graph_plan_scope,
    prepare_graph,
)
from equivariantx.models.tace.tace_oam_xl_v2.layout import (
    channel_contiguous_metadata,
    ir_mul_order,
)
from equivariantx.models.tace.tace_oam_xl_v2.linear import linear_contract
from equivariantx.models.tace.tace_oam_xl_v2.model import grouped_networks
from equivariantx.models.tace.tace_oam_xl_v2.precision import (
    require_tensors,
    require_unscaled,
    validate_call,
    validate_environment,
    validate_graph,
)
from equivariantx.models.tace.tace_oam_xl_v2.product import product_contract
from equivariantx.models.tace.tace_oam_xl_v2.program import (
    MixedGradientPass,
    fuse_gradient_sums,
)
from equivariantx.models.tace.tace_oam_xl_v2.training import GradientWorkspace
from equivariantx.ops.channel_product import channel_product


@pytest.mark.parametrize("radial_mode", ["stream", "materialize"])
def test_conversion_checkpoint_and_mixed_derivatives(monkeypatch, radial_mode):
    pytest.importorskip("tace")
    from tace.lightning.torch_model import create_model
    from tace.models.spherical.default import DEFAULT_MODEL_CONFIG

    for name, value in {"EQX": "1", "MATRIX_WEIGHT": "1", "COMPILE": "0"}.items():
        monkeypatch.setenv(f"TACE_USE_{name}", value)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        config = deepcopy(DEFAULT_MODEL_CONFIG)
        config.update(
            num_layers=2,
            num_channel=4,
            Lmax=1,
            lmax=1,
            mmax=1,
            cutoff=4.0,
            max_neighbors=None,
            mlp_type="mlp_v2",
        )
        config["edge_update"]["type"] = "element2"
        config["edge_embedding"]["type"] = "linear"
        config["radial_basis"].update(hidden=[4, 4], bias=True, apply_cutoff=False)
        config["atomic_basis"].update(type="cgtp", scatter_norm="density")
        config["resnet"].update(type="BAB", linear_type="identity")
        config["layer_norm"].update(
            pre_norm_type="merge_rms_norm", final_norm_type="merge_rms_norm"
        )
        config["product_basis"].update(
            type="nonlinear_cgtp", correlation=2, num_expert=1, num_channel_per_expert=8
        )
        config["readout_emlp"]["hidden"] = [8]
        config["scale_shift"]["enable"] = False
        targets = ["energy", "forces", "stress"]
        config["target_property"] = targets
        stats = [
            dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)
        ]
        source = create_model(config, stats, targets, []).double().train()
        candidate = TACEOAMXLV2.from_tace(source, radial_mode=radial_mode)
        original = dict(source.named_parameters())
        converted = dict(candidate.named_parameters())
        assert original.keys() == converted.keys()
        assert all(parameter is converted[name] for name, parameter in original.items())
        assert source.state_dict().keys() == candidate.state_dict().keys()
        for name, value in source.state_dict().items():
            torch.testing.assert_close(
                value, candidate.state_dict()[name], rtol=0, atol=0
            )
        candidate.load_state_dict(source.state_dict(), strict=True)
        assert candidate.get_fidelity_idx() == source.get_fidelity_idx()
        candidate.eval()
        assert not candidate.adapter.training
        candidate.train()
        data = {
            "positions": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.31, -0.18],
                    [0.0, 0.0, 0.0],
                    [2.0, -0.22, 0.41],
                ]
            ),
            "node_type": torch.zeros(4, dtype=torch.int64),
            "edge_index": torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            "edge_shifts": torch.zeros(4, 3, dtype=torch.int64),
            "lattice": torch.eye(3).repeat(2, 1, 1) * 16.0,
            "pbc": torch.ones(2, 3, dtype=torch.bool),
            "batch": torch.tensor([0, 0, 1, 1]),
            "ptr": torch.tensor([0, 2, 4]),
            "fidelity_idx": torch.zeros(2, dtype=torch.int64),
        }
        results = []
        for model in (source, candidate):
            batch = model.prepare_property_data(
                {name: value.clone() for name, value in data.items()}
            )
            output = model(batch)
            loss = sum(output[name].square().sum() for name in targets)
            gradients = torch.autograd.grad(
                loss, tuple(original.values()), allow_unused=True
            )
            results.append((output, gradients))
        for name in targets:
            torch.testing.assert_close(
                results[0][0][name], results[1][0][name], rtol=1e-10, atol=1e-11
            )
        for reference, actual in zip(results[0][1], results[1][1]):
            if reference is None:
                assert actual is None
            else:
                torch.testing.assert_close(reference, actual, rtol=1e-9, atol=1e-10)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution")
def test_aligned_force_parameter_derivatives():
    torch.manual_seed(31)
    saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        irreps = o3.Irreps("4x0e + 4x1o + 4x2e")
        harmonics = o3.Irreps.spherical_harmonics(2)
        instructions = [
            (i, j, k, "uvu", True)
            for i, (_, a) in enumerate(irreps)
            for j, (_, b) in enumerate(harmonics)
            for k, (_, c) in enumerate(irreps)
            if c in a * b
        ]
        tp = o3.TensorProduct(
            irreps,
            harmonics,
            irreps,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        original = O3TensorProductConv(tp, normalize=False).cuda()
        aligned = aligned_convolution(tp).cuda()
        node = torch.randn(9, irreps.dim, device="cuda", requires_grad=True)
        position = torch.randn(9, 3, device="cuda", requires_grad=True)
        radial_weight = torch.randn(
            4, tp.weight_numel, device="cuda", requires_grad=True
        )
        edge_index = torch.randint(0, 9, (2, 37), device="cuda")
        edge_index[1] = (edge_index[0] + torch.randint(1, 9, (37,), device="cuda")) % 9
        comparison = []
        for transformed in (False, True):
            vector = position[edge_index[1]] - position[edge_index[0]]
            lengths = vector.norm(dim=-1, keepdim=True)
            direction = vector / lengths
            radial = torch.cat(
                (lengths.sin(), lengths.cos(), lengths.exp().reciprocal(), lengths**2),
                dim=-1,
            )
            if transformed:
                frame = packed_frame(aligned.frame, direction)
                output = aligned(
                    node,
                    radial,
                    radial_weight,
                    frame,
                    lengths.sin().expand(-1, len(harmonics)),
                    edge_index,
                    9,
                )
            else:
                output = original(
                    node,
                    None,
                    radial,
                    radial_weight,
                    edge_index,
                    vectors=direction,
                    amplitudes=lengths.sin(),
                )
            energy = output.sin().sum()
            force = -torch.autograd.grad(energy, position, create_graph=True)[0]
            loss = force.square().mean() + energy.square() * 0.001
            gradients = torch.autograd.grad(loss, (node, radial_weight))
            comparison.append((output, force, *gradients))
        for expected, actual in zip(*comparison):
            torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-10)
    finally:
        torch.set_default_dtype(saved_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution")
def test_gated_product_double_backward():
    """Compare the full parameter gradient of a force loss to two operations."""
    torch.manual_seed(15)
    width = 3
    metadata = repr(
        (
            0,
            (6, 6, 6),
            (
                ((0, 0, 0), 0.7, width, (2, 2, 2)),
                ((1, 1, 0), -0.3, width, (2, 2, 2)),
                ((0, 1, 1), 0.2, width, (2, 2, 2)),
                ((1, 0, 1), 0.9, width, (2, 2, 2)),
            ),
        )
    )
    mapping = [0, 0, 1, 1, 2, 2]
    fused = gated_product_metadata(metadata, mapping, 3)
    position = torch.randn(5, 6, device="cuda", dtype=torch.float64, requires_grad=True)
    weight = torch.randn(6, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    outputs = []
    for combined in (False, True):
        left, right = position.sin(), position.cos()
        gate = torch.nn.functional.silu(position @ weight)
        if combined:
            value = gated_product(fused, left, right, gate, 6)
        else:
            value = (
                channel_product(
                    metadata,
                    repr((((0, 1, 2), 2, 0, 0),)),
                    [left, right, left.new_empty(0, 6)],
                )[0]
                * gate[:, mapping]
            )
        energy = value.sum()
        forces = -torch.autograd.grad(energy, position, create_graph=True)[0]
        loss = (forces - 0.2).square().sum() + energy.square()
        gradient = torch.autograd.grad(loss, weight)[0]
        outputs.append((value, forces, gradient))
    for reference, candidate in zip(*outputs):
        torch.testing.assert_close(reference, candidate, rtol=1e-11, atol=1e-11)


def test_gated_product_rejects_nonaffine_mapping():
    metadata = repr((0, (3, 3, 3), (((0, 0, 0), 1.0, 3, (1, 1, 1)),)))
    with pytest.raises(ValueError, match="not affine"):
        gated_product_metadata(metadata, [0, 2, 1], 3)


@pytest.mark.parametrize(
    "reuse,alpha,expected", [(False, 1, 1), (True, 1, 0), (False, 0.5, 0)]
)
def test_delay_weight_gradient(reuse, alpha, expected):
    metadata = repr(("mul_ir", "mul_ir", 7, 10, ((0, 7, 0, 10, 1, 0.7),)))

    def compute(x, w, y):
        early = linear_contract(
            metadata, repr(((0, (0, 1, 2), 0, 0), (0, (0, 1, 2), 1, 1))), [x, w, y]
        )
        late = linear_contract(
            metadata, repr(((0, (0, 1, 2), 1, 0),)), [x + early[0], w, y.sin()]
        )
        total = torch.add(early[1], late[0], alpha=alpha)
        return (total, early[1]) if reuse else (total, early[0])

    torch.manual_seed(55)
    inputs = tuple(
        torch.randn(shape, dtype=torch.float64)
        for shape in ((13, 7), (7, 10), (13, 10))
    )
    module = make_fx(compute)(*inputs)
    reference = module(*inputs)
    assert fuse_gradient_sums(module.graph) == expected
    module.recompile()
    candidate = module(*inputs)
    for a, b in zip(reference, candidate):
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)


def test_delay_radial_mixed_gradient():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from e3nn import o3

    from equivariantx.o3.cgtp.convolution import O3TensorProductConv
    from equivariantx.ops.o3 import contraction

    product = o3.TensorProduct(
        "8x0e+8x1o",
        "0e+1o",
        "8x0e+8x1o",
        [
            (0, 0, 0, "uvu", True),
            (1, 1, 0, "uvu", True),
            (0, 1, 1, "uvu", True),
            (1, 0, 1, "uvu", True),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    metadata = O3TensorProductConv(product).kernel_metadata
    mixed = repr(
        (
            ((6, 1, 2, 3, 5), False, ((1, 0), (4, 1))),
            ((0, 7, 2, 3, 5), False, ((0, 2), (4, 1))),
            ((0, 1, 2, 8, 5), False, ((0, 2), (1, 0), (4, 1))),
        )
    )
    reverse = repr((((0, 1, 2, 3, 5), False, ((0, 0), (1, 1))),))

    def compute(source, target, *values):
        early = contraction(
            metadata, mixed, source, target, list(values), batch_paths=True
        )
        late = contraction(
            metadata,
            reverse,
            source,
            target,
            [*values[:5], values[5] + early[1]],
            batch_paths=True,
        )
        return early[0] + late[1], early[2] + late[0], early[1]

    torch.manual_seed(71)
    indices = torch.randint(7, (2, 73), device="cuda")
    inputs = tuple(
        torch.randn(shape, dtype=torch.float64, device="cuda") * 0.1
        for shape in (
            (7, 32),
            (73, 32),
            (0, 32),
            (73, 4),
            (7, 32),
            (7, 32),
            (7, 32),
            (73, 32),
            (73, 4),
        )
    )
    module = make_fx(compute)(*indices, *inputs)
    reference = module(*indices, *inputs)
    assert fuse_gradient_sums(module.graph) == 2
    module.recompile()
    for a, b in zip(reference, module(*indices, *inputs)):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
    from torch._inductor import config

    with config.patch(post_grad_custom_pre_pass=MixedGradientPass()):
        compiled = torch.compile(compute, fullgraph=True)(*indices, *inputs)
    for a, b in zip(reference, compiled):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)


def test_compiled_argument_growth():
    from torch._inductor import config

    metadata = repr(("mul_ir", "mul_ir", 7, 10, ((0, 7, 0, 10, 1, 0.7),)))
    program = repr(((0, (0, 1, 2), 1, 0),))

    def compute(x, w, y):
        first = linear_contract(metadata, program, [x, w, y])[0]
        second = linear_contract(metadata, program, [x.cos(), w, y.sin()])[0]
        return first + second

    inputs = tuple(
        torch.randn(shape, dtype=torch.float64)
        for shape in ((13, 7), (7, 10), (13, 10))
    )
    reference = compute(*inputs)
    with config.patch(post_grad_custom_pre_pass=MixedGradientPass()):
        actual = torch.compile(compute, fullgraph=True)(*inputs)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)


def test_preserve_gradient_dtype_promotion():
    metadata = repr(("mul_ir", "mul_ir", 7, 10, ((0, 7, 0, 10, 1, 0.7),)))
    program = repr(((0, (0, 1, 2), 1, 0),))

    def compute(x, w, y):
        low = linear_contract(metadata, program, [x, w, y])[0]
        high = linear_contract(metadata, program, [x.double(), w.double(), y.double()])[
            0
        ]
        return low + high

    inputs = tuple(torch.randn(shape) for shape in ((13, 7), (7, 10), (13, 10)))
    module = make_fx(compute)(*inputs)
    reference = module(*inputs)
    assert fuse_gradient_sums(module.graph) == 0
    module.recompile()
    torch.testing.assert_close(module(*inputs), reference, rtol=0, atol=0)


def test_scoped_sort_reuse_and_in_place_invalidation(monkeypatch):
    source = torch.tensor([2, 0, 1, 2, 0])
    target = torch.tensor([1, 2, 0, 0, 2])
    argsort = torch.argsort
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return argsort(*args, **kwargs)

    monkeypatch.setattr(torch, "argsort", counted)
    with graph_plan_scope():
        order = prepare_graph(source, target, 0)
        first = prepare_graph(source, target, 0, 4)
        assert first[0] is order
        with graph_plan_scope():
            assert prepare_graph(source, target, 0, 4) is first
        assert len(calls) == 1
        source.copy_(torch.tensor([0, 2, 1, 0, 2]))
        changed = prepare_graph(source, target, 0, 4)
        assert changed is not first
        assert len(calls) == 2
        torch.testing.assert_close(changed[0], torch.tensor([0, 3, 2, 1, 4]))
        torch.testing.assert_close(changed[1], torch.tensor([0, 2, 3, 5, 5]))

    with graph_plan_scope():
        source.fill_(1)
        order, pointers = prepare_graph(source, target, 0, 4)
        torch.testing.assert_close(order, torch.arange(5))
        torch.testing.assert_close(pointers, torch.tensor([0, 0, 5, 5, 5]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution")
def test_contiguous_product_mixed_derivatives():
    metadata = repr(
        (
            0,
            (9, 9, 9),
            (
                ((0, 1, 2), 0.3, 3, (3, 3, 3)),
                ((1, 2, 0), -0.6, 3, (3, 3, 3)),
                ((2, 0, 1), 0.7, 3, (3, 3, 3)),
                ((1, 1, 2), 0.2, 3, (3, 3, 3)),
            ),
        )
    )
    order = ir_mul_order(o3.Irreps("3x1o"))
    converted = channel_contiguous_metadata(metadata, (order, order, order), 3)
    order = order.cuda()
    torch.manual_seed(20)
    coordinates = torch.randn(
        7, 9, device="cuda", dtype=torch.float64, requires_grad=True
    )
    weight = torch.randn(9, 9, device="cuda", dtype=torch.float64, requires_grad=True)
    results = []
    for contiguous in (False, True):
        left = coordinates.sin()
        right = (coordinates @ weight).cos()
        if contiguous:
            left, right = left[:, order], right[:, order]
        output = channel_product(
            converted if contiguous else metadata,
            repr((((0, 1, 2), 2, 0, 0),)),
            [left, right, left.new_empty(0, 9)],
        )[0]
        if contiguous:
            output = output[:, order.argsort()]
        energy = output.sin().sum()
        force = -torch.autograd.grad(energy, coordinates, create_graph=True)[0]
        gradient = torch.autograd.grad(force.square().sum(), weight)[0]
        results.append((output, force, gradient))
    for reference, candidate in zip(*results):
        torch.testing.assert_close(reference, candidate, rtol=1e-11, atol=1e-11)


def test_layout_contract_rejects_partial_channels():
    metadata = repr((0, (9, 9, 9), (((0, 0, 0), 1.0, 2, (3, 3, 3)),)))
    order = ir_mul_order(o3.Irreps("3x1o"))
    with pytest.raises(ValueError, match="uniform channel count"):
        channel_contiguous_metadata(metadata, (order, order, order), 3)


@pytest.mark.parametrize("edges", [0, 1057])
def test_scheduled_edges_mixed_derivatives(edges):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(155)
    product = o3.TensorProduct(
        "8x0e+8x1o",
        "0e+1o",
        "8x0e+8x1o",
        [
            (0, 0, 0, "uvu", True),
            (1, 1, 0, "uvu", True),
            (0, 1, 1, "uvu", True),
            (1, 0, 1, "uvu", True),
        ],
        internal_weights=False,
        shared_weights=False,
    )
    conv = O3TensorProductConv(product, batch_paths=True).cuda().double()
    # Node 6 is isolated; source/target order has repeated indices.
    indices = torch.randint(6, (2, edges), device="cuda")
    tensors = [
        torch.randn(shape, device="cuda", dtype=torch.float64, requires_grad=True)
        for shape in ((7, 32), (edges, 4), (edges, product.weight_numel))
    ]
    x, attrs, w = tensors
    empty = w.new_empty((0, product.weight_numel))
    paired = []
    for native in (False, True):
        if native:
            y = edge_contract(
                conv.kernel_metadata,
                repr((((0, 1, 2, 3, 4), False, ((4, 0),)),)),
                indices[0],
                indices[1],
                [x, w, empty, attrs, x.new_empty(1).expand(7, 32)],
            )[0]
        else:
            y = conv(x, attrs, w, empty, indices)
        # Request both angular and radial coordinate adjoints, just as the
        # production coordinate backward needs both geometry dependencies.
        dy = torch.autograd.grad(y.sin().sum(), (attrs, w), create_graph=True)
        loss = y.square().sum() + sum(value.square().sum() for value in dy)
        gradients = torch.autograd.grad(loss, tensors)
        paired.append((y, *dy, *gradients))
    for a, b in zip(*paired):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-9)
    torch.testing.assert_close(
        paired[1][0][6], torch.zeros_like(paired[1][0][6]), rtol=0, atol=0
    )


def function(x, a, b, layout):
    metadata = repr(
        (layout, "mul_ir", 28, 40, ((0, 7, 0, 10, 1, 0.7), (7, 7, 10, 10, 3, 0.3)))
    )
    return linear_contract(
        metadata,
        repr(((0, (0, 1, 3), 2, 0), (1, (0, 2, 3), 2, 0))),
        [x, a, b, x.new_empty(1).expand(x.size(0), 40)],
    )[0]


def reference(x, a, b, layout):
    scalar = (x[:, :7] @ a) * 0.7
    value = (
        x[:, 7:].reshape(-1, 3, 7)
        if layout == "ir_mul"
        else x[:, 7:].reshape(-1, 7, 3).transpose(1, 2)
    )
    vector = (value @ b).transpose(1, 2).reshape(-1, 30) * 0.3
    return torch.cat((scalar, vector), -1)


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize(
    "device,dtype",
    [("cpu", torch.float64), ("cuda", torch.float32), ("cuda", torch.float64)],
)
def test_linear_mixed_adjoint(layout, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(92)
    tensors = [
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        for shape in ((13, 28), (7, 10), (7, 10))
    ]
    paired = []
    seed = torch.randn_like(tensors[0])
    for evaluate in (reference, function):
        value = evaluate(*tensors, layout)
        # The nonlinear readout gives both the bilinear and loss-curvature
        # terms of coordinate/parameter mixed derivatives.
        forces = torch.autograd.grad(value.sin().sum(), tensors[0], create_graph=True)[
            0
        ]
        loss = value.square().mean() + (forces * seed).sum()
        gradients = torch.autograd.grad(loss, tensors)
        paired.append((value, forces, *gradients))
    for a, b in zip(*paired):
        torch.testing.assert_close(
            a,
            b,
            rtol=1e-10 if dtype == torch.float64 else 4e-4,
            atol=1e-10 if dtype == torch.float64 else 4e-5,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_partial_weight_seed(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(
        4,
        28,
        device=device,
        dtype=torch.float64 if device == "cpu" else torch.float32,
        requires_grad=True,
    )
    a = torch.randn(7, 10, device=device, dtype=x.dtype, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    grad = torch.autograd.grad(function(x, a, b, "mul_ir").sum(), a, create_graph=True)[
        0
    ]
    mixed = torch.autograd.grad(grad.sum(), x)[0]
    torch.testing.assert_close(
        mixed[:, 7:], torch.zeros_like(mixed[:, 7:]), rtol=0, atol=0
    )


def test_tf32_scaled_cotangents():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.manual_seed(181)
        x = torch.randn(193, 128, device="cuda", requires_grad=True)
        weight = torch.randn(128, 128, device="cuda", requires_grad=True)
        seed = torch.randn(193, 128, device="cuda")
        scale = 0.173
        reference = (x @ weight) * scale
        expected = torch.autograd.grad(reference, (x, weight), seed)
        metadata = repr(("mul_ir", "mul_ir", 128, 128, ((0, 128, 0, 128, 1, scale),)))
        actual = linear_contract(
            metadata,
            repr(((0, (0, 1, 2), 2, 0),)),
            [x, weight, x.new_empty(1).expand(x.size(0), 128)],
        )[0]
        gradients = torch.autograd.grad(actual, (x, weight), seed)
        for a, b in zip(expected, gradients):
            relative = (a.double() - b.double()).norm() / a.double().norm()
            assert relative.item() < 2e-5
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize(
    "nodes,shared", [(0, False), (1, False), (17, False), (17, True)]
)
def test_product_mixed_gradients(nodes, shared):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Include an absent operand to check required-role masks after differentiating.
    metadata = repr(
        (
            8,
            (2, 3, 3),
            (
                ((0, 0, 0), 0.5),
                ((1, 1, 0), -0.7),
                ((0, 1, 1), 0.2),
                ((1, 2, 2), 0.8),
                ((0, -1, 2), 0.4),
            ),
        )
    )
    torch.manual_seed(51)
    x = torch.randn(nodes, 16, dtype=torch.float64, device="cuda", requires_grad=True)
    y = torch.randn(
        1 if shared else nodes,
        24,
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    seed = torch.randn_like(x)

    def compute(operation):
        output = operation(
            metadata, repr((((0, 1, 2), 2, 0, 0),)), [x.sin(), y, x.new_empty((0, 24))]
        )[0]
        energy = output.square().sum()
        force = torch.autograd.grad(energy, x, create_graph=True)[0]
        gradients = torch.autograd.grad(energy + (force * seed).sum(), (x, y))
        return (output, force, *gradients)

    reference, candidate = compute(channel_product), compute(product_contract)
    for a, b in zip(reference, candidate):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reject_reduced_inputs_and_intermediates(dtype):
    value = torch.ones(3, dtype=dtype)
    with pytest.raises(TypeError, match="forbidden"):
        require_tensors({"positions": value})
    graph = torch.fx.Graph()
    node = graph.placeholder("hidden")
    node.meta["val"] = value
    graph.output(node)
    with pytest.raises(TypeError, match="Precision violation at hidden"):
        validate_graph(graph)
    node.meta.clear()
    with graph.inserting_before(next(n for n in graph.nodes if n.op == "output")):
        graph.call_function(torch.ops.aten._to_copy.default, (node,), {"dtype": dtype})
    with pytest.raises(TypeError, match="Forbidden cast"):
        validate_graph(graph)


def test_allow_data_precision_but_reject_autocast():
    data = {
        "positions": torch.ones(4, 3, dtype=torch.float64),
        "features": torch.ones(4, 5),
        "batch": torch.zeros(4, dtype=torch.int64),
    }
    validate_call(data)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            validate_call(data)


@pytest.mark.parametrize(
    "key,value",
    [
        ("EQX_XL_RADIAL_HALF_SCALE", "1048576"),
        ("EQX_XL_RADIAL_HALF_FIRST_SCALE", "1024"),
        ("EQX_XL_NODE_HALF_SCALE", "1024"),
        ("EQX_XL_MATRIX_PRECISION", "half"),
        ("EQX_XL_TABLE_PRECISION", "bf16x3"),
    ],
)
def test_old_reduced_options_fail(key, value):
    with pytest.raises(ValueError):
        validate_environment({key: value})


def test_zero_legacy_arguments_and_extension_cache(monkeypatch):
    validate_environment(
        {"EQX_XL_RADIAL_HALF_SCALE": "0", "EQX_XL_MATRIX_PRECISION": "off"}
    )
    require_unscaled(0, "radial_scale")
    with pytest.raises(ValueError, match="removed"):
        require_unscaled(1, "radial_scale")
    from equivariantx.models.tace.tace_oam_xl_v2 import extensions

    monkeypatch.setattr(extensions, "GRAPH_EXTENSIONS", {})
    monkeypatch.setattr(extensions, "LOCKED", False)
    empty = extensions.extension_identity()
    monkeypatch.setattr(extensions, "LOCKED", False)
    seen = []
    extensions.register_graph_extension("identity", "v1", lambda g: seen.append(g))
    assert extensions.extension_identity() != empty
    graph = torch.fx.Graph()
    extensions.apply_graph_extensions(graph)
    assert seen == [graph]
    with pytest.raises(RuntimeError, match="before"):
        extensions.register_graph_extension("late", "v1", lambda g: None)


def test_artifact_operator_registration():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
import torch

class WithoutTACE(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tace' or fullname.startswith('tace.'):
            raise ModuleNotFoundError('This check runs without TACE')

sys.meta_path.insert(0, WithoutTACE())
from equivariantx.models.tace.tace_oam_xl_v2 import TACEOAMXLV2, register_runtime_ops
assert TACEOAMXLV2.name == 'TACE-OAM-XL-v2'
assert TACEOAMXLV2.__module__ == 'equivariantx.models.tace.tace_oam_xl_v2.model'
register_runtime_ops()
register_runtime_ops()
assert hasattr(torch.ops.eqx_oam_xl, 'edge')
assert hasattr(torch.ops.eqx_oam_xl, 'linear')
assert 'tace' not in sys.modules
""",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


class mlpLinear(torch.nn.Module):
    """Minimal input-major affine map matching the conversion protocol."""

    def __init__(self, inputs, outputs):
        super().__init__()
        self.in_dim, self.out_dim = inputs, outputs
        self.weight = torch.nn.Parameter(
            torch.randn(inputs, outputs, dtype=torch.float64) / inputs**0.5
        )
        self.bias = torch.nn.Parameter(torch.randn(outputs, dtype=torch.float64))

    def get_weight(self):
        return self.weight

    def forward(self, x):
        return x @ self.weight + self.bias


@pytest.mark.parametrize("normalization", [False, True])
@pytest.mark.parametrize("hidden", [False, True])
def test_grouped_force_parameter_gradients(normalization, hidden):
    """Compare to materialized per-edge networks through force-loss gradients."""
    torch.manual_seed(190)
    zsource = torch.tensor([0, 1, 0, 2, 1, 2, 2])
    ztarget = torch.tensor([2, 0, 1, 1, 0, 2, 1])
    positions = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    networks, updates, parameters = [], [], []
    for index in range(3):
        width = 4 if index < 2 else 6
        layers = [mlpLinear(7, width)]
        if normalization:
            layers.append(torch.nn.LayerNorm(width, dtype=torch.float64))
        layers.extend([torch.nn.SiLU(), mlpLinear(width, 2)])
        network = SimpleNamespace(mlp=torch.nn.Sequential(*layers))
        update = SimpleNamespace(
            source_embedding=torch.nn.Embedding(3, 2, dtype=torch.float64),
            target_embedding=torch.nn.Embedding(3, 2, dtype=torch.float64),
        )
        networks.append(network)
        updates.append(update)
        parameters.extend(network.mlp.parameters())
        parameters.extend(update.source_embedding.parameters())
        parameters.extend(update.target_embedding.parameters())
    results = []
    for grouped in (False, True):
        radial = positions.sin()
        if grouped:
            outputs = grouped_networks(
                networks, updates, radial, zsource, ztarget, hidden=hidden
            )
        else:
            outputs = []
            for network, update in zip(networks, updates):
                values = torch.cat(
                    (
                        radial,
                        update.target_embedding(ztarget),
                        update.source_embedding(zsource),
                    ),
                    -1,
                )
                layers = network.mlp[:-1] if hidden else network.mlp
                outputs.append(layers(values))
        energy = sum(output.sin().sum() for output in outputs)
        forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
        loss = (forces - 0.4).square().sum() + energy.square() / 17
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        results.append((energy, forces, gradients))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-11, atol=1e-11)
    for reference, candidate in zip(results[0][2], results[1][2]):
        if reference is None:
            assert candidate is None
        else:
            torch.testing.assert_close(reference, candidate, rtol=1e-10, atol=1e-10)


def test_shared_gradient_outputs_and_unused_parameters():
    original = [
        torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float64)) for _ in range(3)
    ]
    candidate = [torch.nn.Parameter(p.detach().clone()) for p in original]
    a = torch.optim.AdamW(original, lr=0.01)
    b = torch.optim.AdamW(candidate, lr=0.01)
    workspace = GradientWorkspace(candidate)
    for used in ((0, 1), (0, 2)):
        a.zero_grad(set_to_none=True)
        shared = torch.randn(3, 4, dtype=torch.float64)
        gradients = [shared if i in used else None for i in range(3)]
        for p, g in zip(original, gradients):
            p.grad = None if g is None else g.clone()
        workspace.attach(gradients)
        expected_norm = torch.nn.utils.clip_grad_norm_(original, 0.4)
        actual_norm = workspace.clip(0.4)
        torch.testing.assert_close(expected_norm, actual_norm, rtol=1e-14, atol=1e-14)
        for p, q in zip(original, candidate):
            if p.grad is None:
                assert q.grad is None
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-14, atol=1e-14)
        a.step()
        b.step()
        for p, q in zip(original, candidate):
            torch.testing.assert_close(p, q, rtol=1e-14, atol=1e-14)


def test_refuse_ordinary_autograd_loss():
    import pytest

    p = torch.nn.Parameter(torch.randn(3))
    with pytest.raises(ValueError, match="unscaled explicit joint"):
        GradientWorkspace([p]).attach_loss(p.square().sum())


def test_explicit_loss_requires_matching_parameter_order():
    import pytest

    class ParameterLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, gradients, *parameters):
            ctx.save_for_backward(*gradients)
            return loss.clone()

        @staticmethod
        def backward(ctx, scale):
            return None, None, *(g * scale for g in ctx.saved_tensors)

    a = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))
    b = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))
    gradients = (a.detach() * 2, b.detach() * 3)
    loss = ParameterLoss.apply(torch.tensor(1.0), gradients, a, b)
    correct = GradientWorkspace([a, b])
    correct.attach_loss(loss)
    for p, g in zip((a, b), gradients):
        torch.testing.assert_close(p.grad, g)
    with pytest.raises(ValueError, match="parameter order"):
        GradientWorkspace([b, a]).attach_loss(loss)
