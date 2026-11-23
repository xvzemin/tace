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
from equivariantx.ops.common.channel_product import channel_product


@pytest.fixture
def tf32_precision():
    previous = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rows", [0, 5])
def test_projected_radial_heads_mixed_gradients(device, rows):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from equivariantx.models.tace.tace_oam_xl_v2.radial_projection import (
        projected_heads,
    )

    torch.manual_seed(52)
    widths = (6, 7)
    inputs = [
        torch.randn(2, rows, 4, device=device, dtype=torch.float64, requires_grad=True)
        for _ in widths
    ]
    weights = [
        torch.randn(2, 4, width, device=device, dtype=torch.float64, requires_grad=True)
        for width in widths
    ]
    columns = torch.stack([torch.randperm(13, device=device) for _ in range(2)])
    columns = list(columns.split(widths, dim=1))
    actual = projected_heads(inputs, weights, columns, 13)
    expected = actual.new_zeros(actual.shape)
    for left, right, index in zip(inputs, weights, columns):
        expected = expected.scatter(2, index[:, None].expand(-1, rows, -1), left @ right)
    torch.testing.assert_close(actual, expected)
    parameters = (*inputs, *weights)
    gradients = [
        torch.autograd.grad(value.square().sum(), parameters, create_graph=True)
        for value in (actual, expected)
    ]
    seeds = [torch.randn_like(value) for value in parameters]
    mixed = [
        torch.autograd.grad(sum((g * seed).sum() for g, seed in zip(values, seeds)), parameters)
        for values in gradients
    ]
    for result in (gradients, mixed):
        for a, b in zip(*result):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels", [64, 128, 1024])
def test_projected_radial_heads_round_tf32(tf32_precision, channels):
    from equivariantx.models.tace.tace_oam_xl_v2.radial_projection import (
        projected_heads,
    )

    torch.manual_seed(53)
    inputs = [torch.randn(2, 65, channels, device="cuda") for _ in range(2)]
    weights = [torch.randn(2, channels, width, device="cuda") for width in (128, 256)]
    permutation = torch.stack([torch.randperm(384, device="cuda") for _ in range(2)])
    columns = list(permutation.split((128, 256), dim=1))
    actual = projected_heads(inputs, weights, columns, 384)
    for left, right, index in zip(inputs, weights, columns):
        expected = left @ right
        selected = actual.gather(2, index[:, None].expand(-1, 65, -1))
        assert (selected - expected).norm() / expected.norm() < 2e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batched", [False, True])
def test_lt_matrix_plans_read_updated_inputs(tf32_precision, batched):
    from equivariantx.models.tace.tace_oam_xl_v2.lt_matrix import MatrixPlan

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        prefix = (3,) if batched else ()
        left = torch.randn(*prefix, 128, 512, device="cuda").transpose(-1, -2)
        right = torch.randn(*prefix, 128, 256, device="cuda")
        plan = MatrixPlan(left, right, workspace_bytes=32 * 1024**2, candidates=8)
        expected = left @ right
        choices = []
        for index in range(plan.count):
            if plan.algorithms[index].status:
                continue
            try:
                actual = plan.run(left, right, index)
                error = (actual - expected).norm() / expected.norm()
                if error < 0.001:
                    choices.append(index)
            except RuntimeError:
                continue
        assert choices
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = plan.run(left, right, choices[0])
        for _ in range(3):
            left.add_(0.01)
            right.mul_(0.99)
            graph.replay()
            expected = left @ right
            assert (result - expected).norm() / expected.norm() < 0.001
        stream.synchronize()
        del graph
        plan.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_matrix_plan_lifetime_and_precision(tf32_precision):
    from equivariantx.models.tace.tace_oam_xl_v2.lt_matrix import lt_radial_bmm
    from equivariantx.ops.common.cuda.graph import retain_constants

    constants = {}
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), retain_constants(constants):
        left = torch.randn(3, 8192, 128, device="cuda")
        right = torch.randn(3, 128, 256, device="cuda")
        result = lt_radial_bmm(left, right)
        assert constants
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            result = lt_radial_bmm(left, right)
        left.add_(0.01)
        graph.replay()
        expected = torch.bmm(left, right)
        assert (result - expected).norm() / expected.norm() < 0.001
        count = len(constants)
        a, b = left.double(), right.double()
        torch.testing.assert_close(lt_radial_bmm(a, b), torch.bmm(a, b), rtol=0, atol=0)
        assert len(constants) == count
        stream.synchronize()
        del graph
        constants.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_matrix_plan_lifetime(monkeypatch, tf32_precision):
    import weakref

    from equivariantx.models.tace.tace_oam_xl_v2.lt_matrix import PLANS, selected_mm
    from equivariantx.ops.common.cuda.graph import retain_constants

    monkeypatch.setenv("EQX_XL_LT_RADIAL", "1")
    monkeypatch.setenv("EQX_XL_LT_ALGORITHM", "-1")
    constants, previous = {}, set(PLANS)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), retain_constants(constants):
        left = torch.randn(8192, 128, device="cuda").t()
        right = torch.randn(8192, 5376, device="cuda")
        actual = selected_mm(left, right)
        expected = left @ right
        assert (actual - expected).norm() / expected.norm() < 0.001
        assert set(PLANS) == previous
        (plan,) = constants.values()
        reference = weakref.ref(plan)
        del plan
        stream.synchronize()
    constants.clear()
    assert reference() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("edges", [0, 4])
@pytest.mark.parametrize("reset_pool", [False, True])
def test_capture_reuses_warmup_and_tracks_live_inputs(monkeypatch, edges, reset_pool):
    joint = pytest.importorskip("tace.models.compile.joint")
    from equivariantx.models.tace.tace_oam_xl_v2.capture import CapturedTraining

    weight = torch.nn.Parameter(torch.tensor([0.2, -0.3, 0.4], device="cuda"))
    unused = torch.nn.Parameter(torch.ones(1, device="cuda"))
    adapter = SimpleNamespace(prepare_property_data=lambda data: data)
    workspace = GradientWorkspace([weight, unused])
    executions = torch.zeros((), device="cuda", dtype=torch.int64)
    failures = []

    class ExplicitProgram:
        def __init__(self, model, loss_fn, batch):
            assert model is adapter

        def __call__(self, batch):
            if reset_pool and batch["positions"].shape[0] == 4 and not failures:
                failures.append(True)
                raise torch.OutOfMemoryError("Exercise graph-pool recovery")
            executions.add_(1)
            positions = batch["positions"]
            values = positions @ weight
            energy = values.square().sum().reshape(1)
            residual = energy - batch["energy"]
            gradient = 4 * residual * (positions.T @ values)
            loss = joint.ParameterLoss.apply(
                residual.square().sum().detach(),
                (gradient.detach(), None),
                weight,
                unused,
            )
            return {"energy": energy}, loss

    monkeypatch.setattr(joint, "JointTraining", ExplicitProgram)
    capture = CapturedTraining(
        adapter,
        SimpleNamespace(loss_property=["energy"]),
        workspace,
        shared_graph_plan=False,
    )
    for index, nodes in enumerate((3, 3, 4, 3)):
        if index == 2 and reset_pool:
            capture.constants["previous_pool"] = torch.ones(1, device="cuda")
        with torch.no_grad():
            weight.add_(0.1)
        batch = {
            "positions": torch.arange(nodes * 3, device="cuda")
            .float()
            .reshape(nodes, 3)
            / (index + 2),
            "energy": torch.tensor([index + 0.5], device="cuda"),
            "ptr": torch.tensor([0, nodes], device="cuda"),
            "edge_index": torch.zeros((2, edges), device="cuda", dtype=torch.int64),
        }
        expected = (batch["positions"] @ weight).square().sum().reshape(1)
        expected_loss = (expected - batch["energy"]).square().sum()
        (expected_gradient,) = torch.autograd.grad(expected_loss, weight)
        result, loss = capture(batch)
        torch.testing.assert_close(result["energy"], expected)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(weight.grad, expected_gradient)
        assert unused.grad is None
        assert executions.item() == index + 1
    assert capture.captures == capture.warmups == 2 + int(reset_pool)
    assert len(capture.entries) == 2
    assert capture.pool_resets == int(reset_pool)
    assert "previous_pool" not in capture.constants


@pytest.mark.parametrize(
    "device,radial_mode,scatter_norm,conditioning,projection,topology",
    [
        ("cpu", "stream", "density", "linear", "individual", "directed"),
        ("cpu", "stream", "cutoff", "element", "individual", "directed"),
        ("cpu", "stream", "cutoff", "element2", "individual", "pairs"),
        ("cpu", "stream", "density", "linear", "individual", "empty"),
        ("cpu", "materialize", "density", "linear", "composed", "pairs"),
        ("cpu", "materialize", "cutoff", "element2", "batched", "pairs"),
        ("cpu", "materialize", "density", "linear", "split", "directed"),
        ("cpu", "materialize", "density", "linear", "jet", "directed"),
        ("cpu", "materialize", "cutoff", "element", "jet", "directed"),
        ("cuda", "materialize", "density", "linear", "individual", "directed"),
        ("cuda", "materialize", "cutoff", "element", "individual", "directed"),
        ("cuda", "materialize", "density", "element2", "individual", "pairs"),
        ("cuda", "materialize", "cutoff", "element2", "individual", "pairs"),
        ("cuda", "materialize", "density", "linear", "individual", "empty"),
    ],
)
def test_conversion_checkpoint_and_mixed_derivatives(
    monkeypatch,
    radial_mode,
    scatter_norm,
    device,
    conditioning,
    projection,
    topology,
):
    check_conversion(
        monkeypatch,
        radial_mode,
        scatter_norm,
        device,
        conditioning,
        projection,
        topology,
        "cgtp",
    )


@pytest.mark.parametrize("interaction", ["uv_o2", "uu_o2"])
@pytest.mark.parametrize(
    "device,embedding,degree,radial_mode",
    [
        ("cpu", "linear", 1, "materialize"),
        ("cpu", "element", 1, "stream"),
        ("cuda", "element2", 1, "materialize"),
        ("cuda", "element2", 4, "materialize"),
        ("cuda", "element2", 1, "stream"),
    ],
)
def test_local_conversion(
    monkeypatch, interaction, device, embedding, degree, radial_mode
):
    check_conversion(
        monkeypatch,
        radial_mode,
        "cutoff",
        device,
        embedding,
        "individual",
        "pairs",
        interaction,
        degree,
        norm_type="soft_cap_rms_norm"
        if degree == 1 and embedding != "element"
        else None,
    )


@pytest.mark.parametrize("modes", [["cgtp", "uv_o2"], ["uu_o2", "cgtp"]])
def test_mixed_local_conversion(monkeypatch, modes):
    check_conversion(
        monkeypatch,
        "materialize",
        "density",
        "cpu",
        "element2",
        "batched",
        "pairs",
        modes,
    )


@pytest.mark.parametrize(
    "basis,device,radial_mode,projection,embedding,interpolate",
    [
        ("spb", "cpu", "stream", "individual", "element2", False),
        ("spb", "cpu", "materialize", "composed", "linear", True),
        ("spb", "cuda", "materialize", "individual", "linear", False),
        ("spb", "cuda", "materialize", "individual", "element2", True),
        ("spb", "cuda", "stream", "individual", "element", False),
        ("ejn", "cpu", "stream", "individual", "element2", False),
        ("ejn", "cuda", "materialize", "individual", "element2", False),
        ("ejn", "cuda", "stream", "individual", "element", False),
        ("j0", "cuda", "materialize", "individual", "element2", False),
        ("jn", "cpu", "materialize", "composed", "element2", False),
        ("jn", "cuda", "materialize", "individual", "element2", False),
        ("jn", "cuda", "stream", "individual", "element", False),
    ],
)
def test_spectral_radial_conversion(
    monkeypatch, basis, device, radial_mode, projection, embedding, interpolate
):
    if basis == "spb":
        pytest.importorskip("physical_basis")
    monkeypatch.setenv("TACE_MEMORY_FIRST", str(int(radial_mode == "stream")))
    check_conversion(
        monkeypatch,
        radial_mode,
        "density" if embedding == "linear" else "cutoff",
        device,
        embedding,
        projection,
        "pairs",
        "cgtp",
        radial_basis=dict(
            radial_basis=basis,
            max_eigenvalue=None if basis in ("spb", "ejn") else 6.0,
            use_interpolation=interpolate,
        ),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("embedding", ["element", "element2"])
@pytest.mark.parametrize("topology", ["directed", "shared"])
@pytest.mark.parametrize("radial_jets", [False, True])
def test_reciprocal_spectral_conversion(
    monkeypatch, device, embedding, topology, radial_jets
):
    check_conversion(
        monkeypatch,
        "materialize",
        "cutoff",
        device,
        embedding,
        "individual",
        topology,
        "cgtp",
        degree=2,
        norm_type="soft_cap_rms_norm",
        radial_basis=dict(radial_basis="jn", max_eigenvalue=8.0),
        bias=False,
        layers=3,
        radial_jets=radial_jets,
    )


def check_conversion(
    monkeypatch,
    radial_mode,
    scatter_norm,
    device,
    conditioning,
    projection,
    topology,
    interaction,
    degree=1,
    norm_type=None,
    radial_basis=None,
    bias=None,
    layers=2,
    radial_jets=False,
):
    pytest.importorskip("tace")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA execution")
    from tace.lightning.torch_model import create_model
    from tace.models.spherical.default import DEFAULT_MODEL_CONFIG

    for name, value in {"EQX": "1", "MATRIX_WEIGHT": "1", "COMPILE": "0"}.items():
        monkeypatch.setenv(f"TACE_USE_{name}", value)
    edge_mode = (
        "scheduled"
        if device == "cuda" and (interaction != "cgtp" or radial_mode != "stream")
        else "original"
    )
    associate_norm = scatter_norm == "cutoff" and conditioning == "linear"
    pre_norm = norm_type or (
        "const_norm" if conditioning == "element" else "merge_rms_norm"
    )
    monkeypatch.setenv("EQX_XL_ASSOCIATE_NORM", str(int(associate_norm)))
    monkeypatch.setenv(
        "EQX_XL_COMPRESSED_RADIAL", str(int(topology in ("pairs", "shared")))
    )
    monkeypatch.setenv("EQX_XL_RADIAL_JET", str(int(projection == "jet")))
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        config = deepcopy(DEFAULT_MODEL_CONFIG)
        config.update(
            num_layers=layers,
            num_channel=4,
            Lmax=degree,
            lmax=degree,
            mmax=degree,
            cutoff=4.0,
            max_neighbors=None,
            mlp_type="mlp_v2",
        )
        config["edge_update"]["type"] = (
            "element2" if conditioning == "linear" else "identity"
        )
        config["edge_embedding"]["type"] = conditioning
        config["radial_basis"].update(hidden=[4, 4], bias=True, apply_cutoff=False)
        if radial_basis is not None:
            config["radial_basis"].update(radial_basis)
        config["atomic_basis"].update(type=interaction, scatter_norm=scatter_norm)
        if interaction == "uv_o2" or (
            isinstance(interaction, list) and "uv_o2" in interaction
        ):
            config["atomic_basis"].update(
                edge_nonlinear="gate", scalar_act=["silu", "tanh"]
            )
        config["resnet"].update(type="BAB", linear_type="identity")
        config["layer_norm"].update(
            pre_norm_type=pre_norm, final_norm_type=norm_type or "merge_rms_norm"
        )
        config["product_basis"].update(
            type="nonlinear_cgtp", correlation=2, num_expert=1, num_channel_per_expert=8
        )
        if bias is not None:
            for name in ("radial_basis", "atomic_basis", "product_basis"):
                config[name]["bias"] = bias
            config["readout_emlp"]["bias"] = True
        config["readout_emlp"]["hidden"] = [8]
        config["scale_shift"]["enable"] = False
        targets = ["energy", "forces", "stress"]
        config["target_property"] = targets
        stats = [
            dict(
                atomic_numbers=[1, 8],
                atomic_energy={1: 0.0, 8: 0.0},
                avg_num_neighbors=2.0,
            )
        ]
        source = create_model(config, stats, targets, []).double().to(device).train()
        embedding = source.readout_fn.representation.edge_embedding
        for name in getattr(embedding, "table_names", ()):
            with torch.no_grad():
                getattr(embedding, name).weight.normal_(std=0.1)
        production = (
            radial_mode == "materialize"
            and not associate_norm
            and edge_mode == "scheduled"
        )
        if production:
            from tace.models.linear import switch_e3nn_weight_layout

            switch_e3nn_weight_layout(source, "matrix")
        schedule = (
            dict(
                product_mode="packed",
                compose_products=True,
                affine_radial=True,
                gradient_schedule="fused",
                padded_batches=True,
            )
            if production
            else {}
        )
        candidate = TACEOAMXLV2.from_tace(
            source,
            radial_mode=radial_mode,
            edge_mode=edge_mode,
            radial_projection="individual" if projection == "jet" else projection,
            radial_jets=radial_jets,
            **schedule,
        )
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
                    [0.3, 2.7, -0.12],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "node_type": torch.tensor([0, 1, 0, 1]),
            "atomic_numbers": torch.tensor([1, 8, 1, 8]),
            "edge_index": torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
            "edge_shifts": torch.zeros(6, 3, dtype=torch.int64),
            "lattice": torch.eye(3).repeat(2, 1, 1) * 16.0,
            "pbc": torch.ones(2, 3, dtype=torch.bool),
            "batch": torch.tensor([0, 0, 0, 1]),
            "ptr": torch.tensor([0, 3, 4]),
            "fidelity_idx": torch.zeros(2, dtype=torch.int64),
        }
        if topology == "empty":
            data["edge_index"] = data["edge_index"][:, :0]
            data["edge_shifts"] = data["edge_shifts"][:0]
        data = {name: value.to(device) for name, value in data.items()}
        results = []
        for model in (source, candidate):
            batch = {name: value.clone() for name, value in data.items()}
            if production and model is candidate:
                from equivariantx.models.tace.tace_oam_xl_v2.padding import pad_batch

                batch = pad_batch(batch, 8, 16)
            if topology == "pairs" and model is candidate:
                from equivariantx.models.tace.tace_oam_xl_v2.reciprocal import (
                    reciprocal_plan,
                )

                batch = reciprocal_plan(batch)
            batch = model.prepare_property_data(batch)
            if production and model is candidate and "edge_pair_rows" in batch:
                from equivariantx.models.tace.tace_oam_xl_v2.adapter import (
                    OAMCompileAdapter,
                )

                batch = OAMCompileAdapter(candidate.readout_fn).prepare_property_data(
                    batch
                )
                rows = batch["edge_pair_rows"]
                if rows.numel():
                    assert rows.numel() >= batch["edge_index"].shape[1] // 2 + 128
                    assert batch["edge_pair_inverse"].max() < rows.numel()
            output = model(batch)
            loss = sum(output[name].square().sum() for name in targets)
            gradients = torch.autograd.grad(
                loss, tuple(original.values()), allow_unused=True
            )
            if production and model is candidate:
                dummy_forces = output["forces"][data["positions"].shape[0] :]
                torch.testing.assert_close(
                    dummy_forces, torch.zeros_like(dummy_forces), rtol=0, atol=0
                )
                output["forces"] = output["forces"][: data["positions"].shape[0]]
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


def test_angular_packed_product_reconstructs_all_csr_roles(monkeypatch):
    from equivariantx.models.tace.nonlinear_ace import build_product
    from equivariantx.models.tace.tace_oam_xl_v2 import packed_product as packed

    inputs = o3.Irreps("2x1o+2x1o+2x2e")
    outputs = o3.Irreps("2x0e+2x0e+2x1o+2x1o+2x2e")
    paths = [(0, 0, 0), (1, 1, 1), (0, 2, 2), (1, 2, 3), (0, 1, 4)]
    tp = o3.TensorProduct(inputs, inputs, outputs, [(*p, "uuu", False) for p in paths])
    orders = tuple(ir_mul_order(x.simplify()) for x in (inputs, inputs, outputs))
    metadata = channel_contiguous_metadata(build_product(tp), orders, 2)
    gates = torch.tensor(
        [
            2 * i + c
            for i, (_, ir) in enumerate(outputs)
            for c in range(2)
            for _ in range(ir.dim)
        ]
    )[orders[2]].tolist()
    blocks = [mul * ir.dim // 2 for mul, ir in outputs.simplify()]
    expected = packed.packed_metadata(metadata, gates, 2 * len(outputs), blocks)
    monkeypatch.setattr(packed, "SPARSE_ROWS", 1)
    actual = packed.angular_metadata(tp, gates, 2 * len(outputs), blocks, 2)
    assert packed.parse_metadata(actual)[-1][0] == "wigner"
    for role in range(4):
        reference = packed.sparse_product_table(expected, role, torch.float64, "cpu")
        tables = packed.sparse_product_table(actual, role, torch.float64, "cpu")
        for a, b in zip(reference, tables):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution")
def test_sparse_packed_product_program(monkeypatch):
    from equivariantx.models.tace.tace_oam_xl_v2 import packed_product as product

    torch.manual_seed(12)
    rows = tuple(
        ((a, b, (a + b) % 2, (a + b) % 3), (a - b + 0.4) / 5)
        for a in range(3)
        for b in range(3)
    )
    metadata = repr((4, (3, 3, 2, 3), (1, 2), rows))
    operands = [
        torch.randn(5, 2 * cols, dtype=torch.float64, device="cuda")[:, ::2]
        for cols in (12, 12, 8)
    ]
    packed = torch.randn(5, 12, dtype=torch.float64, device="cuda")
    operands += [packed]
    terms = (
        ((0, 0, 2, 3), 0, 0, 0),
        ((0, 0, 2, 3), 1, 0, 0),
        ((0, 1, 2, 3), 2, 1, 0),
        ((0, 1, 2, 3), 3, 2, 0),
    )
    monkeypatch.setattr(product, "use_sparse_product", lambda _: False)
    expected = product.run_packed(metadata, repr(terms), operands)
    monkeypatch.setattr(product, "use_sparse_product", lambda _: True)
    actual = product.run_packed(metadata, repr(terms), operands)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    pieces = packed.reshape(-1).split((20, 40))
    actual = product.run_packed(
        metadata,
        repr(terms[:-1]),
        [*operands[:3], packed.new_empty(0, 0), *(x.view(1, -1) for x in pieces)],
        fragments=((3, (4, 5)),),
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


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
    from equivariantx.ops.common.o3 import contraction

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA angular preparation")
def test_prepared_angular_cache():
    from equivariantx.models.tace.tace_oam_xl_v2.prepared_angular import (
        geometry_scope,
        prepare_angular,
    )

    path = (0, 0, 0, 1, 1, 1, 2, 1, 0, 1.0, ((0, 0, 0, 2.0), (0, 1, 0, -1.0)))
    metadata = repr(((path,), 1, False))
    with geometry_scope():
        value = torch.randn(4, 2, device="cuda")
        out = prepare_angular(metadata, value)
        assert prepare_angular(metadata, value) is out
        value.add_(1)
        torch.testing.assert_close(
            prepare_angular(metadata, value)[:, 0], 2 * value[:, 0] - value[:, 1]
        )
        with torch.inference_mode():
            data = value.clone()
            first = prepare_angular(metadata, data)
            data.add_(1)
            torch.testing.assert_close(prepare_angular(metadata, data), first + 1)


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_parameter_force_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from equivariantx.models.tace.tace_oam_xl_v2.parallel import split_parameter_batches

    torch.manual_seed(157)
    values = [
        (0.1 * torch.randn(shape, device=device, dtype=torch.float64)).requires_grad_()
        for shape in ((3, 97, 5), (3, 5, 7), (3, 7))
    ]

    def compute(position, weight, output):
        hidden = torch.bmm(position, weight)
        energy = (hidden.sin() * output[:, None]).sum()
        force = -torch.autograd.grad(energy, position, create_graph=True)[0]
        loss = energy.square() * 0.01 + (force - 0.05).square().sum()
        gradients = torch.autograd.grad(loss, (weight, output))
        return energy, force, *gradients

    graph = make_fx(compute)(*values)
    with torch.no_grad():
        expected = graph(*values)
        assert split_parameter_batches(graph.graph, minimum_k=32) > 0
        assert split_parameter_batches(graph.graph, minimum_k=32) == 0
        graph.recompile()
        for actual, reference in zip(graph(*values), expected):
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        compiled = torch.compile(graph, fullgraph=True)
        for actual, reference in zip(compiled(*values), expected):
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("compressed", [False, True])
def test_parallel_edge_adjoints_cuda(compressed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from equivariantx.models.tace.tace_oam_xl_v2.compressed import compressed_edge
    from equivariantx.models.tace.tace_oam_xl_v2.parallel import (
        overlap_parameter_matrices,
    )

    torch.manual_seed(155)
    product = o3.TensorProduct(
        "4x0e",
        "0e",
        "4x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    conv = O3TensorProductConv(product, batch_paths=True).cuda().double()
    edges = 1060
    indices = torch.randint(7, (2, edges), device="cuda")
    radial_map = torch.arange(edges, device="cuda") // 2
    shapes = (
        (7, 4),
        (edges // 2 if compressed else edges, 4),
        (edges, 1),
        (7, 4),
        (3, 16),
        (16, 5),
    )
    inputs = tuple(
        torch.randn(shape, device="cuda", dtype=torch.float64) for shape in shapes
    )
    program = repr((((0, 1, 2, 3, 4), False, ((0, 0), (1, 1), (3, 2))),))

    def evaluate(x, weights, attrs, cotangent, left, right):
        matrix = left @ right
        operands = [x, weights, weights.new_empty((0, 4)), attrs, cotangent]
        operation = compressed_edge if compressed else edge_contract
        outputs = operation(
            conv.kernel_metadata,
            program,
            indices[0],
            indices[1],
            operands,
            *((radial_map,) if compressed else ()),
        )
        return (*outputs, matrix)

    graph = make_fx(evaluate)(*inputs)
    reference = graph(*inputs)
    count, _ = overlap_parameter_matrices(graph.graph, minimum_k=4)
    assert count == 1
    graph.recompile()
    actual = graph(*inputs)
    for a, b in zip(reference, actual):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-10)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        actual = graph(*inputs)
    capture.replay()
    for a, b in zip(reference, actual):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-10)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("roles", [(0, 1), (0, 1, 3)])
def test_shared_source_adjoints_cuda(monkeypatch, paired, packed, roles):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from equivariantx.models.tace.tace_oam_xl_v2.compressed import (
        compressed_edge,
        paired_edge,
    )
    from equivariantx.models.tace.tace_oam_xl_v2.edge import scheduled_edges
    from equivariantx.models.tace.tace_oam_xl_v2.packed_edge import pack_edge_metadata

    torch.manual_seed(155)
    irreps = o3.Irreps("128x0e+128x1o+128x2e")
    harmonics = o3.Irreps("0e+1o+2e")
    instructions, outputs = [], []
    for i, (_, left) in enumerate(irreps):
        for j, (_, right) in enumerate(harmonics):
            for output in left * right:
                if output in [ir for _, ir in irreps]:
                    instructions.append((i, j, len(outputs), "uvu", True))
                    outputs.append((128, output))
    product = o3.TensorProduct(
        irreps,
        harmonics,
        o3.Irreps(outputs),
        instructions,
        internal_weights=False,
        shared_weights=False,
    )
    conv = O3TensorProductConv(product, batch_paths=True).cuda().double()
    metadata = (
        pack_edge_metadata(conv.kernel_metadata) if packed else conv.kernel_metadata
    )
    edges = 1060
    indices = torch.randint(7, (2, edges), device="cuda")
    radial_map = torch.arange(edges, device="cuda") // 2
    shapes = (
        (7, irreps.dim),
        (edges if paired else edges // 2, product.weight_numel),
        (0, product.weight_numel),
        (edges, harmonics.dim),
        (7, conv.output_dim),
    )
    operands = [
        torch.randn(shape, device="cuda", dtype=torch.float64) for shape in shapes
    ]
    program = repr(
        (
            (
                (0, 1, 2, 3, 4),
                False,
                tuple((role, slot) for slot, role in enumerate(roles)),
            ),
        )
    )
    results = []
    if paired:
        reference = scheduled_edges(metadata, program, indices[0], indices[1], operands)
        slot = roles.index(1)
        value = reference[slot]
        reference[slot] = value.new_zeros((edges // 2, value.shape[1])).index_add_(
            0, radial_map, value
        )
    monkeypatch.setenv("EQX_XL_SOURCE_ROW_LIMIT", "16")
    monkeypatch.setenv("EQX_XL_MIXED_ROW_LIMIT", "16")
    monkeypatch.setenv("EQX_XL_MIXED_REUSE_ANGULAR", "1")
    for enabled in ("0", "1"):
        monkeypatch.setenv("EQX_XL_SOURCE_VJP", enabled)
        monkeypatch.setenv("EQX_XL_MIXED_OWNER", "source" if enabled == "1" else "off")
        results.append(
            (paired_edge if paired else compressed_edge)(
                metadata,
                program,
                indices[0],
                indices[1],
                operands,
                radial_map,
                *((edges // 2,) if paired else ()),
            )
        )
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-10)
    if paired:
        for actual, expected in zip(results[-1], reference):
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "parallel,extra_use,edges",
    [
        (False, False, 1060),
        (True, False, 1060),
        (True, True, 1060),
        (False, False, 0),
    ],
)
def test_paired_radial_gradient_graph(parallel, extra_use, edges):
    from equivariantx.models.tace.tace_oam_xl_v2.compressed import pair_radial_gradients
    from equivariantx.models.tace.tace_oam_xl_v2.parallel import (
        overlap_parameter_matrices,
    )

    product = o3.TensorProduct(
        "4x0e",
        "0e",
        "4x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    conv = O3TensorProductConv(product, batch_paths=True).cuda().double()
    indices = torch.randint(7, (2, edges), device="cuda")
    shapes = ((7, 4), (edges, 4), (edges, 1), (7, 4), (3, 16), (16, 5))
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.float64) for shape in shapes
    ]
    inputs += [
        torch.arange(edges // 2, device="cuda"),
        torch.arange(edges, device="cuda") // 2,
    ]
    program = repr((((0, 1, 2, 3, 4), False, ((0, 0), (1, 1), (3, 2))),))

    def evaluate(x, weights, attrs, cotangent, left, right, representatives, inverse):
        matrix = left @ right
        values = edge_contract(
            conv.kernel_metadata,
            program,
            indices[0],
            indices[1],
            [x, weights, weights.new_empty((0, 4)), attrs, cotangent],
        )
        radial = (
            values[1]
            .new_zeros((representatives.shape[0], 4))
            .index_put((inverse,), values[1], accumulate=True)
        )
        return (
            values[0],
            radial,
            values[2],
            matrix,
            *((values[1].sum(),) if extra_use else ()),
        )

    graph = make_fx(evaluate)(*inputs)
    if parallel:
        assert overlap_parameter_matrices(graph.graph, minimum_k=4)[0] == 1
    assert pair_radial_gradients(graph.graph) == int(not extra_use)
    graph.recompile()
    graph(*inputs)
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        actual = graph(*inputs)
    for _ in range(2):
        inputs[0].add_(0.01)
        inputs[-1].copy_(inputs[-1].flip(0))
        captured.replay()
        expected = evaluate(*inputs)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-10)


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

    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    workspace = GradientWorkspace([parameter])
    workspace.attach([torch.tensor([3e21, 4e21], dtype=parameter.dtype)])
    norm = workspace.clip(10.0)
    torch.testing.assert_close(norm, norm.new_tensor(5e21), rtol=1e-7, atol=0)
    torch.testing.assert_close(parameter.grad, parameter.new_tensor([6.0, 8.0]))


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_element_index_force_loss(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA execution")
    from equivariantx.models.tace.tace_oam_xl_v2.ordered_reductions import (
        replace_index_put,
    )

    torch.manual_seed(21)
    torch.set_num_threads(4)
    layers, groups, width, edges = 3, 11, 16, 271
    source = torch.randint(groups, (edges,), device="cuda")
    target = torch.randint(groups, (edges,), device="cuda")
    shapes = [
        (edges, 3),
        (layers, groups, width),
        (layers, groups, width),
        (layers, 1, width),
        (layers, width),
    ]
    values = [
        (0.1 * torch.randn(s, device="cuda", dtype=dtype)).requires_grad_()
        for s in shapes
    ]

    def compute(position, left, right, radial, output):
        length2 = position.square().sum(-1, keepdim=True)
        h = length2[None] * radial + left[:, source] + right[:, target]
        energy = (torch.nn.functional.silu(h) * output[:, None]).sum()
        force = -torch.autograd.grad(energy, position, create_graph=True)[0]
        loss = energy.square() * 0.01 + (force - 0.05).square().sum()
        gradients = torch.autograd.grad(loss, (left, right, radial, output))
        return energy, force, *gradients

    with torch.enable_grad():
        graph = make_fx(compute)(*values)
    with torch.no_grad():
        expected = graph(*values)
        assert replace_index_put(graph.graph) == 2
        graph.recompile()
        result = graph(*values)
        tolerance = 3e-6 if dtype == torch.float32 else 2e-13
        for actual, reference in zip(result, expected):
            torch.testing.assert_close(
                actual, reference, rtol=tolerance, atol=tolerance
            )
        compiled = torch.compile(graph, fullgraph=True)
        result = compiled(*values)
        for actual, reference in zip(result, expected):
            torch.testing.assert_close(
                actual, reference, rtol=tolerance, atol=tolerance
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_element_index_replay_and_empty(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA execution")
    from equivariantx.models.tace.tace_oam_xl_v2.ordered_reductions import grouped_sum

    torch.manual_seed(22)
    base = torch.randn((2, 7, 13), device="cuda", dtype=dtype)
    index = torch.randint(7, (37,), device="cuda")
    value = torch.randn((2, 37, 13), device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        grouped_sum(base, index, value)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            result = grouped_sum(base, index, value)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        index.random_(0, 7)
        value.normal_()
        capture.replay()
        expected = torch.ops.aten.index_put.default(base, [None, index], value, True)
        torch.testing.assert_close(result, expected)
    negative = index - base.shape[1]
    torch.testing.assert_close(
        grouped_sum(base, negative, value),
        torch.ops.aten.index_put.default(base, [None, negative], value, True),
    )
    empty = grouped_sum(base, index[:0], value[:, :0])
    torch.testing.assert_close(empty, base, rtol=0, atol=0)
