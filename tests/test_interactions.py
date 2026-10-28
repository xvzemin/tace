"""TACE interactions, equivariance and accelerated execution."""

import sys
import types
from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
from e3nn import o3

from eqx import conv as eqx_conv
from eqx import o2
from tace.models._e3nn.base import NodeUpdate
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.fused import UuO2TensorProductConv
from tace.models._e3nn.inter import O2MagneticInteraction
from tace.models._e3nn.node import NODE_UPDATE
from tace.models._e3nn.o2 import (
    MagneticUvO2TensorProductConv,
    UvO2TensorProductConv,
)
from tace.models._e3nn.representation import NodeRepresentation
from tace.models.layout import LayoutTransform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


@pytest.mark.parametrize("method", ["baseline", "generator", "cg", "wigner", "auto"])
def test_o2_cgtp_stream_methods(method, monkeypatch, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.fused import O2TensorProductConv, O3TensorProductConv

    for backend in ("EQX", "OEQ", "EQT", "CUEQ"):
        monkeypatch.setenv(f"TACE_USE_{backend}", "0")
    irreps_in, irreps_sh, irreps_out = "2x1o", "0e+1o", "2x0e+2x1o+2x1e+2x2e"
    module = O2TensorProductConv(irreps_in, irreps_sh, irreps_out).cuda()
    module.eqx_tp.method = method
    reference = O3TensorProductConv(irreps_in, irreps_sh, irreps_out).cuda()
    x = torch.randn(3, 6, device="cuda", requires_grad=True)
    vectors = torch.randn(8, 3, device="cuda", requires_grad=True)
    radial = torch.randn(8, 4, device="cuda", requires_grad=True)
    projection = torch.randn(4, module.weight_numel, device="cuda", requires_grad=True)
    cutoff = torch.rand(8, 1, device="cuda", requires_grad=True)
    edges = torch.randint(3, (2, 8), device="cuda")
    graph = types.SimpleNamespace(
        edge_vector=vectors, edge_length=vectors.norm(dim=-1, keepdim=True)
    )
    actual = module.forward_stream(x, radial, projection, edges, None, cutoff, graph)
    harmonics = o3.spherical_harmonics(
        irreps_sh, vectors / graph.edge_length, False, "component"
    )
    expected = reference(x, harmonics, (radial @ projection) * cutoff, edges)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    inputs = x, vectors, radial, projection, cutoff
    gradients = [
        torch.autograd.grad(y.square().sum(), inputs, create_graph=True)
        for y in (actual, expected)
    ]
    for value, target in zip(*gradients):
        torch.testing.assert_close(value, target, atol=2e-9, rtol=2e-9)
    gradients = [
        torch.autograd.grad(g[1].square().sum(), inputs, retain_graph=True)
        for g in gradients
    ]
    for value, target in zip(*gradients):
        torch.testing.assert_close(value, target, atol=2e-7, rtol=2e-8)


@pytest.mark.parametrize("magnetic", [False, True])
@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("mmax", [0, 2])
@pytest.mark.parametrize("packed", [False, True])
def test_uv_o2_cuda_convolution(monkeypatch, magnetic, attention, mmax, packed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.o2 import (
        MagneticUvO2TensorProductConv,
        UvO2TensorProductConv,
    )

    torch.manual_seed(401)
    torch.set_default_dtype(torch.float64)
    time = hasattr(o3.Irrep("0e"), "t")
    irreps = (
        "2x0ee+2x0oo+2x1oe+2x1eo+2x2ee+2x2oo"
        if time
        else "2x0e+2x0o+2x1o+2x1e+2x2e+2x2o"
    )
    cls = MagneticUvO2TensorProductConv if magnetic else UvO2TensorProductConv
    args = (irreps, irreps, irreps) if magnetic else (irreps, irreps)
    module = (
        cls(
            *args,
            num_channel=2,
            mmax=mmax,
            even_scalar_act=torch.nn.SiLU(),
            odd_scalar_act=torch.nn.Tanh(),
            tensor_act=torch.nn.Sigmoid(),
            num_head=2,
            num_radial_basis=3,
            use_radial_rotary_attention=attention,
        )
        .cuda()
        .double()
    )
    frame = o2.WignerD(2, 2).cuda().double()
    for edges in (9, 0):

        def rand(*shape):
            return (
                torch.randn(*shape, device="cuda", dtype=torch.float64) * 0.2
            ).requires_grad_()

        x, mag = rand(4, o3.Irreps(irreps).dim), rand(edges, o3.Irreps(irreps).dim)
        vectors = rand(edges, 3)
        weights, radial, cutoff = (
            rand(edges, module.weight_numel),
            rand(edges, 3),
            rand(edges, 1).sigmoid(),
        )
        index = torch.randint(4, (2, edges), device="cuda")
        inputs = (x, vectors, weights, radial, cutoff) + ((mag,) if magnetic else ())
        inputs += tuple(module.parameters())

        def run(enabled):
            monkeypatch.setenv("TACE_USE_EQX", str(int(enabled)))
            w, wi = (
                (frame.forward_packed(vectors), None)
                if enabled and packed
                else frame(vectors)
            )
            args = (
                (x, mag, weights, index, w, wi)
                if magnetic
                else (x, weights, index, w, wi)
            )
            value = module(*args, edge_radial_basis=radial, edge_cutoff=cutoff)
            result = [value]
            if edges:
                for _ in range(3 if magnetic and attention and mmax == 2 else 2):
                    grads = torch.autograd.grad(
                        value.sin().sum(), inputs, create_graph=True, allow_unused=True
                    )
                    result.extend(g for g in grads if g is not None)
                    value = (
                        torch.cat([g.flatten() for g in grads if g is not None]) / 10
                    )
            return result

        expected, actual = run(False), run(True)
        assert len(actual) == len(expected)
        for i, (a, b) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(
                a, b, atol=2e-8, rtol=2e-8, msg=lambda message: f"Result {i}: {message}"
            )


@pytest.mark.parametrize("packed", [False, True])
def test_uv_o2_compile_and_force_training(monkeypatch, double_precision, packed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.o2 import MagneticUvO2TensorProductConv
    from tace.models.compile.compile import trace_to_fx

    monkeypatch.setenv("TACE_USE_EQX", "1")
    irreps = "2x0ee+2x1eo+2x1oe" if hasattr(o3.Irrep("0e"), "t") else "2x0e+2x1e+2x1o"
    module = MagneticUvO2TensorProductConv(
        irreps,
        irreps,
        irreps,
        num_channel=2,
        mmax=1,
        even_scalar_act=torch.nn.SiLU(),
        odd_scalar_act=torch.nn.Tanh(),
        tensor_act=torch.nn.Sigmoid(),
        num_head=2,
        num_radial_basis=3,
        use_radial_rotary_attention=True,
    ).cuda()
    frame = o2.WignerD(1, 1).cuda()

    def evaluate(x, mag, weights, vectors, radial, cutoff, index):
        w, wi = (frame.forward_packed(vectors), None) if packed else frame(vectors)
        return module(x, mag, weights, index, w, wi, radial, cutoff)

    compiled = torch.compile(
        evaluate, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for nodes, edges in ((3, 5), (3, 0), (0, 0)):
        index = torch.randint(max(nodes, 1), (2, edges), device="cuda")
        inputs = [
            torch.randn(shape, device="cuda", requires_grad=True) * 0.1
            for shape in (
                (nodes, module.irreps_in.dim),
                (edges, module.magnetic_edge_irreps.dim),
                (edges, module.weight_numel),
                (edges, 3),
                (edges, 3),
                (edges, 1),
            )
        ]
        inputs[-1] = inputs[-1].sigmoid()
        actual, expected = compiled(*inputs, index), evaluate(*inputs, index)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        if edges:

            def energy_forces(*args):
                energy = evaluate(*args).square().sum()
                forces = -torch.autograd.grad(energy, args[3], create_graph=True)[0]
                return energy, forces

            traced = trace_to_fx(energy_forces, (*inputs, index))
            compiled_forces = torch.compile(traced, backend="aot_eager", fullgraph=True)
            a, b = compiled_forces(*inputs, index), energy_forces(*inputs, index)
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, atol=1e-9, rtol=1e-9)
            parameters = tuple(module.parameters())
            ga, gb = [
                torch.autograd.grad(e + f.square().sum(), parameters) for e, f in (a, b)
            ]
            for x, y in zip(ga, gb):
                torch.testing.assert_close(x, y, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("mask", range(32))
@pytest.mark.parametrize("compile_enabled", [False, True])
def test_scatter_backend_priority(monkeypatch, mask, compile_enabled):
    from tace.models._e3nn.fused import O3TensorProductConv
    from tace.utils.env import ACCELERATION_ENV

    enabled = [
        name
        for i, name in enumerate(("eqx", "luna", "oeq", "eqt", "cue"))
        if mask & (1 << i)
    ]
    for name in ("eqx", "luna", "oeq", "eqt", "cue"):
        monkeypatch.setenv(ACCELERATION_ENV[name], "1" if name in enabled else "0")
    monkeypatch.setenv("TACE_USE_COMPILE", "1" if compile_enabled else "0")
    expected = next((name for name in enabled if name != "eqt"), None)
    if expected == "cue" and compile_enabled:
        expected = None
    constructed = []

    def luna(**kwargs):
        constructed.append("luna")
        return torch.nn.Identity()

    def oeq(**kwargs):
        constructed.append("oeq")
        return torch.nn.Identity()

    def cue(**kwargs):
        constructed.append("cue")
        return torch.nn.Identity()

    monkeypatch.setitem(
        sys.modules,
        "tace.models.luna",
        types.SimpleNamespace(LunaTensorProductConv=luna),
    )
    monkeypatch.setitem(
        sys.modules,
        "tace.models.oeq",
        types.SimpleNamespace(OeqTensorProductConv=oeq),
    )
    monkeypatch.setitem(
        sys.modules,
        "tace.models.cue",
        types.SimpleNamespace(CueTensorProductConv=cue),
    )
    module = O3TensorProductConv("2x0e", "0e", "2x0e")
    assert module.use_eqx == (expected == "eqx")
    assert module.use_luna == (expected == "luna")
    assert module.use_oeq == (expected == "oeq")
    assert module.use_cue == (expected == "cue")
    assert constructed == ([expected] if expected in ("luna", "oeq", "cue") else [])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_luna_scatter(dtype):
    if sys.platform != "linux" or not torch.cuda.is_available():
        pytest.skip("LunaEquivariance requires Linux and CUDA")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("LunaEquivariance requires compute capability >= 8.0")
    pytest.importorskip("LunaEquivariance")
    from tace.models._e3nn.fused import O3TensorProductConv
    from tace.utils.env import enable_acceleration

    torch.set_default_dtype(dtype)
    enable_acceleration(force=True)
    irreps = "2x0e+2x1o", "0e+1o", "2x0e+2x1o+2x1e+2x2e"
    reference = O3TensorProductConv(*irreps).to(device="cuda", dtype=dtype)
    enable_acceleration(enable_luna=True, force=True)
    actual = O3TensorProductConv(*irreps).to(device="cuda", dtype=dtype)
    assert actual.use_luna
    assert actual.state_dict().keys() == reference.state_dict().keys()
    actual.load_state_dict(reference.state_dict(), strict=True)
    reference.load_state_dict(actual.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        actual(
            torch.zeros(4, actual.irreps_in1.dim),
            torch.zeros(0, actual.irreps_in2.dim),
            torch.zeros(0, actual.weight_numel),
            torch.empty(2, 0, dtype=torch.long),
        )

    for edges in (
        [[1, 0, 2, 1, 0], [2, 0, 1, 0, 2]],
        [[0, 2, 1, 0, 2], [1, 0, 2, 2, 0]],
        [[], []],
    ):
        edge_index = torch.tensor(edges, dtype=torch.long, device="cuda")
        inputs = tuple(
            torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            for shape in (
                (4, actual.irreps_in1.dim),
                (edge_index.size(1), actual.irreps_in2.dim),
                (edge_index.size(1), actual.weight_numel),
            )
        )
        outputs = [module(*inputs, edge_index) for module in (actual, reference)]
        atol, rtol = (2e-5, 2e-5) if dtype == torch.float32 else (2e-10, 2e-10)
        torch.testing.assert_close(*outputs, atol=atol, rtol=rtol)
        for output in outputs:
            assert not output[3].any()  # Isolated node.
        gradients = [
            torch.autograd.grad(output.square().sum(), inputs, create_graph=True)
            for output in outputs
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=atol * 10, rtol=rtol * 10)
        second = [
            torch.autograd.grad(sum(g.square().sum() for g in grad), inputs)
            for grad in gradients
        ]
        for a, b in zip(*second):
            torch.testing.assert_close(a, b, atol=atol * 100, rtol=rtol * 100)

    dtype = torch.float64 if dtype == torch.float32 else torch.float32
    torch.set_default_dtype(dtype)
    actual.to(dtype=dtype)
    rebuilt = O3TensorProductConv(*irreps).cuda()
    assert actual.fused_tp.aoti_config["dtype"] == str(dtype).removeprefix("torch.")
    edge_index = torch.tensor([[0, 1, 2], [2, 0, 1]], device="cuda")
    inputs = (
        torch.randn(4, actual.irreps_in1.dim, device="cuda"),
        torch.randn(3, actual.irreps_in2.dim, device="cuda"),
        torch.randn(3, actual.weight_numel, device="cuda"),
    )
    torch.testing.assert_close(
        actual(*inputs, edge_index), rebuilt(*inputs, edge_index)
    )


@pytest.mark.parametrize("mmax", [0, 1, 2])
@pytest.mark.parametrize("edges", [0, 7])
def test_tece_packed_frames(monkeypatch, mmax, edges, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from eqx.models.tace.tece_oam_rra.interaction import stream
    from tace.models._e3nn.tece_oam_rra import Convolution

    monkeypatch.setenv("TACE_USE_EQX", "1")
    irreps = o3.Irreps("4x0e+4x1o+4x2e")
    module = Convolution(
        mmax,
        2,
        4,
        2,
        2,
        3,
        False,
        True,
        True,
        LayoutTransform(irreps),
        LayoutTransform(irreps),
        torch.nn.SiLU(),
        torch.nn.Sigmoid(),
    ).cuda()
    frame = o2.WignerD(mmax, 2).cuda()
    inputs = [
        (torch.randn(shape, device="cuda") * 0.1).requires_grad_()
        for shape in (
            (4, irreps.dim),
            (edges, 3),
            (3, module.weight_numel),
            (edges, 35),
            (edges, 3),
        )
    ]
    x, radial, weight, packed, basis = inputs
    cutoff = torch.rand(edges, 1, device="cuda", requires_grad=True)
    index = torch.randint(4, (2, edges), device="cuda")
    local_dim = frame.local_indices.numel()
    selected = packed.index_select(1, frame._packed_indices)
    dense = (
        packed.new_zeros((edges, local_dim * 9))
        .index_copy(1, frame._dense_indices, selected)
        .view(edges, local_dim, 9)
    )
    inverse = dense.transpose(1, 2) * frame.inverse_scale
    actual = stream(
        module, x, radial, weight, None, index, cutoff, packed, packed, basis
    )
    expected = module(x, radial @ weight, index, cutoff, dense, inverse, basis)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    if edges == 7 and mmax == 2:
        compiled = torch.compile(
            lambda *args: stream(module, *args), backend="aot_eager", fullgraph=True
        )
        output = compiled(x, radial, weight, None, index, cutoff, packed, packed, basis)
        torch.testing.assert_close(output, expected, atol=1e-11, rtol=1e-10)
        gradients = [
            torch.autograd.grad(result.sum(), packed, retain_graph=True)[0]
            for result in (output, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-11, rtol=1e-10)
    inputs = (*inputs, cutoff, *module.parameters())
    losses = [value.square().sum() for value in (actual, expected)]
    for _ in range(2):
        grads = [
            torch.autograd.grad(loss, inputs, create_graph=True, allow_unused=True)
            for loss in losses
        ]
        for value, a, b in zip(inputs, *grads):
            a = torch.zeros_like(value) if a is None else a
            b = torch.zeros_like(value) if b is None else b
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)
        if not edges:
            break
        losses = [
            sum(g.square().sum() for g in values if g is not None) for values in grads
        ]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mmax", [0, 1, 2])
@pytest.mark.parametrize("ece", [False, True])
@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("gate_m0", [False, True])
def test_tece_state_dict(device, mmax, ece, attention, gate_m0, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from tace.models._e3nn.tece_oam_rra import Convolution
    from tace.models.mlp import get_scaled_activation

    torch.manual_seed(18)
    irreps = o3.Irreps("4x0e+4x1o+4x2e")
    kwargs = dict(
        mmax=mmax,
        lmax=2,
        num_channel=4,
        num_head=2,
        edge_ace_hidden=3,
        num_radial_basis=5,
        gate_m0=gate_m0,
        use_asymmetric_contraction=ece,
        use_radial_rotary_attention=attention,
        reshape_in=LayoutTransform(irreps),
        reshape_out=LayoutTransform(irreps),
        scalar_act=get_scaled_activation("silu"),
        tensor_act=get_scaled_activation("silu" if ece else "sigmoid"),
    )
    reference = Convolution(**deepcopy(kwargs)).to(device=device, dtype=torch.float64)
    if attention:
        with torch.no_grad():
            reference.radial_proj.weight.normal_(std=0.2)
            reference.temperature_logit.uniform_(-1, 1)
    model = Convolution(**kwargs).to(device=device, dtype=torch.float64)
    model.load_state_dict(reference.state_dict(), strict=True)
    assert isinstance(model.linear_up, o2.Linear)
    assert isinstance(model.linear_down, o2.Linear)
    assert isinstance(model.nonlinearity, o2.Gate)
    if ece:
        assert isinstance(model.ece, o2.TensorProduct)
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )
    restored = deepcopy(model)
    restored.load_state_dict(model.state_dict(), strict=True)
    nodes, edges = 4, 7
    degrees = torch.tensor(
        [
            ell
            for m in range(mmax + 1)
            for _ in range(1 if m == 0 else 2)
            for ell in range(m, 3)
        ],
        device=device,
    )
    columns = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2], device=device)
    mask = degrees[:, None] == columns[None, :]
    inputs = [
        torch.randn(*shape, device=device, dtype=torch.float64)
        .mul_(0.25)
        .requires_grad_()
        for shape in (
            (nodes, irreps.dim),
            (edges, model.weight_numel),
            (edges, len(degrees), 9),
            (edges, 9, len(degrees)),
            (edges, 5),
        )
    ]
    x, weight, rotation, inverse, radial = inputs
    cutoff = torch.rand(edges, 1, device=device, dtype=torch.float64).requires_grad_()
    with torch.no_grad():
        cutoff[0] = 0
    index = torch.tensor([[0, 1, 2, 3, 1, 2, 3], [1, 2, 3, 0, 0, 0, 0]], device=device)
    args = (x, weight, index, cutoff, rotation * mask, inverse * mask.T, radial)
    expected = reference(*args)
    actual = model(*args, fused=device == "cuda")
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=1e-10)
    torch.testing.assert_close(restored(*args), expected, atol=2e-12, rtol=1e-10)
    if mmax == 2 and ece and attention and not gate_m0 and device == "cpu":
        from tace.models.compile.compile import trace_to_fx

        def evaluate(*values):
            output = model(*values)
            derivative = torch.autograd.grad(
                output.sum(), values[0], create_graph=True
            )[0]
            return output, derivative

        traced = trace_to_fx(evaluate, args)
        assert all(
            "eqx." not in str(node.target)
            for node in traced.graph.nodes
            if node.op == "call_function"
        )
        exported = torch.export.export(traced, args, strict=False)
        compiled = torch.compile(exported.module(), backend="aot_eager", fullgraph=True)
        for a, b in zip(compiled(*args), evaluate(*args)):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    losses = [out.square().sum() for out in (actual, expected)]
    actual_grads = torch.autograd.grad(
        losses[0], tuple(model.parameters()), retain_graph=True
    )
    reference_grads = torch.autograd.grad(
        losses[1], tuple(reference.parameters()), retain_graph=True
    )
    for actual_grad, expected_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=1e-8)
    for _ in range(3):
        gradients = [
            torch.autograd.grad(
                loss, (*inputs, cutoff), create_graph=True, allow_unused=True
            )
            for loss in losses
        ]
        for a, b in zip(*gradients):
            if a is None or b is None:
                assert a is None and b is None
            else:
                torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)
        losses = [
            sum(g.square().sum() for g in gs if g is not None) for gs in gradients
        ]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("edges", [0, 7, 65, 257, 1025])
def test_tece_streaming_derivatives(monkeypatch, device, edges):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from eqx.models.tace.tece_oam_rra.interaction import stream
    from tace.models._e3nn.tece_oam_rra import Convolution

    if edges == 1025:
        from functools import partial

        import eqx.backends.cuda.execution as convolution_execution
        import eqx.backends.cuda.models.tace.attention as execution

        monkeypatch.setattr(execution, "BACKWARD_TILE_SIZE", 256)
        monkeypatch.setattr(
            execution, "forward", partial(execution.forward, tile_size=256)
        )
        monkeypatch.setattr(
            convolution_execution,
            "launch",
            partial(convolution_execution.launch, tile_size=256),
        )

    monkeypatch.setenv("TACE_USE_EQX", "1")
    irreps = o3.Irreps("4x0e+4x1o+4x2e")
    module = Convolution(
        2,
        2,
        4,
        2,
        2,
        3,
        False,
        True,
        True,
        LayoutTransform(irreps),
        LayoutTransform(irreps),
        torch.nn.SiLU(),
        torch.nn.Sigmoid(),
    ).to(device=device, dtype=torch.float64)
    with torch.no_grad():
        module.radial_proj.weight.normal_(std=0.1)
    nodes = 256 if edges == 65 else 4
    values = [
        torch.randn(*shape, device=device, dtype=torch.float64)
        .mul_(0.15)
        .requires_grad_()
        for shape in [
            (nodes, irreps.dim),
            (edges, 3),
            (3, module.weight_numel),
            (module.weight_numel,),
            (edges, 9, 9),
            (edges, 9, 9),
            (edges, 3),
        ]
    ]
    x, radial, projection, bias, rotation, inverse, basis = values
    # Frames rotate each O(3) degree independently; rows are ordered by m.
    degrees = torch.tensor([0, 1, 2, 1, 2, 1, 2, 2, 2], device=device)
    columns = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2], device=device)
    mask = degrees[:, None] == columns[None, :]
    rotation, inverse = rotation * mask, inverse * mask.T
    cutoff = torch.rand(edges, 1, device=device, dtype=torch.float64).requires_grad_()
    index = torch.stack(
        (
            torch.arange(edges, device=device) % nodes,
            (torch.arange(edges, device=device) + 1) % nodes,
        )
    )
    if edges >= 257:
        # Exercise split neighborhoods, empty receivers and zero edge weights.
        index[1] = (torch.arange(edges, device=device) % 5 == 0).long()
        with torch.no_grad():
            cutoff[::7] = 0
    actual = stream(
        module,
        x,
        radial,
        projection,
        bias,
        index,
        cutoff,
        rotation,
        inverse,
        basis,
    )
    expected = module(
        x, radial @ projection + bias, index, cutoff, rotation, inverse, basis
    )
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    inputs = (*values, cutoff, *module.parameters())
    losses = [v.square().sum() for v in (actual, expected)]
    for _ in range(3):
        grads = [
            torch.autograd.grad(loss, inputs, create_graph=True, allow_unused=True)
            for loss in losses
        ]
        for i, (a, b) in enumerate(zip(*grads)):
            a = torch.zeros_like(inputs[i]) if a is None else a
            b = torch.zeros_like(inputs[i]) if b is None else b
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)
        losses = [
            sum(g.square().sum() for g in values if g is not None) for values in grads
        ]
    if device == "cuda" and edges == 7:
        compiled = torch.compile(
            lambda *args: stream(module, *args), backend="aot_eager", fullgraph=True
        )
        output = compiled(
            x, radial, projection, bias, index, cutoff, rotation, inverse, basis
        )
        torch.testing.assert_close(output, expected, atol=1e-11, rtol=1e-10)
        gradients = [
            torch.autograd.grad(
                value.sum(), inputs, allow_unused=True, retain_graph=True
            )
            for value in (output, expected)
        ]
        for a, b in zip(*gradients):
            if a is not None and b is not None:
                torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)
    if device == "cuda" and edges == 65:
        with torch.no_grad():
            module.temperature_logit.add_(0.2)
            projection.add_(0.01)
        actual = stream(
            module,
            x,
            radial,
            projection,
            bias,
            index.flip(0),
            cutoff,
            rotation,
            inverse,
            basis,
        )
        expected = module(
            x,
            radial @ projection + bias,
            index.flip(0),
            cutoff,
            rotation,
            inverse,
            basis,
        )
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("ece", [False, True])
@pytest.mark.parametrize("gate_m0", [False, True])
@pytest.mark.parametrize("mmax", [0, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tece_native_variants(monkeypatch, ece, gate_m0, mmax, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import eqx.ops.recompute as recompute
    from eqx.models.tace.tece_oam_rra.interaction import stream
    from tace.models._e3nn.tece_oam_rra import Convolution
    from tace.models.mlp import get_scaled_activation

    def no_replay(*args, **kwargs):
        raise AssertionError("The native interaction must not use Graph replay.")

    monkeypatch.setattr(recompute, "replay", no_replay)
    torch.manual_seed(20)
    channels = 16 if ece and mmax == 2 and dtype == torch.float64 else 2
    irreps = o3.Irreps([(channels, (l, (-1) ** l)) for l in range(3)])
    module = Convolution(
        mmax,
        2,
        channels,
        1,
        2,
        3,
        gate_m0,
        ece,
        True,
        LayoutTransform(irreps),
        LayoutTransform(irreps),
        get_scaled_activation("tanh"),
        get_scaled_activation("silu"),
    ).to(device="cuda", dtype=dtype)
    angular = sum((3 - m) * (1 if m == 0 else 2) for m in range(mmax + 1))
    inputs = [
        (torch.randn(*shape, device="cuda", dtype=dtype) * 0.1).requires_grad_()
        for shape in (
            (4, irreps.dim),
            (5, 3),
            (3, module.weight_numel),
            (5, angular, 9),
            (5, 9, angular),
            (5, 3),
        )
    ]
    x, radial, projection, rotation, inverse, basis = inputs
    cutoff = torch.tensor(
        [[0.0], [0.0], [0.7], [1.0], [0.2]],
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    index = torch.tensor([[0, 1, 2, 3, 0], [1, 1, 2, 2, 2]], device="cuda")
    with torch.no_grad():
        module.radial_proj.weight.normal_(std=0.2)
    network = (
        torch.nn.Sequential(
            torch.nn.Linear(3, 3), torch.nn.LayerNorm(3), torch.nn.SiLU()
        ).to(device="cuda", dtype=dtype)
        if gate_m0
        else None
    )
    actual = stream(
        module,
        x,
        radial,
        projection,
        None,
        index,
        cutoff,
        rotation,
        inverse,
        basis,
        radial_network=network,
    )
    expected = module(
        x,
        (network(radial) if network is not None else radial) @ projection,
        index,
        cutoff,
        rotation,
        inverse,
        basis,
    )
    atol, rtol = (2e-6, 1e-4) if dtype == torch.float32 else (2e-11, 1e-10)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    variables = (*inputs, cutoff, *module.parameters())
    if network is not None:
        variables += tuple(network.parameters())
    grads = [
        torch.autograd.grad(value.square().sum(), variables, create_graph=True)
        for value in (actual, expected)
    ]
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=10 * atol, rtol=10 * rtol)


def test_uu_o2_compile_and_empty_graph(double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    frame_in = o2.LocalFrame("3x0e+3x1e").cuda()
    frame_out = o2.LocalFrame("3x0o+3x1o").cuda()
    linear = o2.UuLinear(frame_in.irreps_out, frame_out.irreps_out, 3)
    module = eqx_conv.UuO2TensorProductConv(frame_in, linear, frame_out).cuda()
    frame = o2.WignerD(1, 1).cuda()

    def evaluate(x, vectors, radial, projection, cutoff, index):
        return module(
            x,
            radial,
            projection,
            frame.forward_packed(vectors.detach()),
            cutoff,
            index,
            x.size(0),
            vectors=vectors,
        )

    compiled = torch.compile(evaluate, fullgraph=True, dynamic=True)
    for nodes, edges in ((3, 7), (3, 1), (3, 0), (0, 0)):
        index = torch.randint(max(nodes, 1), (2, edges), device="cuda")
        inputs = [
            torch.randn(shape, device="cuda", requires_grad=True)
            for shape in (
                (nodes, frame_in.input_dim),
                (edges, 3),
                (edges, 2),
                (2, linear.weight_numel),
                (edges, 1),
            )
        ]
        actual, expected = compiled(*inputs, index), evaluate(*inputs, index)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        gradients = [
            torch.autograd.grad(value.square().sum(), inputs)
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-9)
        if edges == 7:
            from tace.models.compile.compile import trace_to_fx

            def energy_forces(*values):
                energy = evaluate(*values).square().sum()
                forces = -torch.autograd.grad(energy, values[1], create_graph=True)[0]
                return energy, forces

            traced = trace_to_fx(energy_forces, (*inputs, index))
            compiled_forces = torch.compile(traced, fullgraph=True)
            actual = compiled_forces(*inputs, index)
            expected = energy_forces(*inputs, index)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-9)
            gradients = [
                torch.autograd.grad(energy + forces.square().sum(), inputs)
                for energy, forces in (actual, expected)
            ]
            for a, b in zip(*gradients):
                torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "interaction,bias",
    [("cgtp", False), ("cgtp", True), ("o2_cgtp", True), (["cgtp", "o2"], False)],
)
def test_streaming_force_training(monkeypatch, device, interaction, bias):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.lightning import convert_cgtp, load_tace
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    for name in ("TACE_USE_EQX", "TACE_USE_EQT", "TACE_USE_OEQ", "TACE_USE_CUEQ"):
        monkeypatch.setenv(name, "0")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        config = deepcopy(DEFAULT_MODEL_CONFIG)
        config.update(
            cutoff=4.0,
            max_neighbors=None,
            num_layers=2,
            num_channel=3,
            Lmax=2,
            lmax=2,
            mmax=2,
            parity=True,
            statistics=[
                dict(atomic_numbers=[1], avg_num_neighbors=2.0, atomic_energy={1: 0.0})
            ],
            target_property=["energy", "forces", "stress", "virials"],
        )
        config["node_embedding"]["type"] = "linear"
        config["atomic_basis"]["type"] = interaction
        config["readout_emlp"]["use_one_body_magmoms"] = False
        config["radial_basis"]["hidden"] = [128 if bias else 4]
        config["radial_basis"]["bias"] = bias
        if isinstance(interaction, list):
            config["radial_basis"]["apply_cutoff"] = False
        config["readout_emlp"]["hidden"] = [2]
        config["scale_shift"]["enable"] = False
        reference = TensorModel(e3nnTACE(**deepcopy(config))).train().to(device)
        monkeypatch.setenv("TACE_USE_EQX", "1")
        model = convert_cgtp(reference, "o2")
        representation = model.readout_fn.representation
        mixed = isinstance(interaction, list)
        assert representation.use_packed_wigner == mixed
        assert representation.use_local_frame == mixed
        assert representation.use_o3_angular_basis == mixed

        def no_edge_message(*args, **kwargs):
            raise AssertionError("The streamed convolution must not call an edge TP")

        for layer in representation.interactions:
            if getattr(layer, "use_eqx", False) and hasattr(layer.rejector, "tp"):
                monkeypatch.setattr(layer.rejector.tp, "forward", no_edge_message)
                monkeypatch.setattr(layer.edge_info, "forward", no_edge_message)

        data = dict(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]], device=device
            ),
            node_attrs=torch.ones(3, 1, device=device),
            edge_index=torch.tensor(
                [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], device=device
            ),
            edge_shifts=torch.zeros(6, 3, device=device),
            lattice=torch.eye(3, device=device).unsqueeze(0) * 8,
            batch=torch.zeros(3, dtype=torch.long, device=device),
            ptr=torch.tensor([0, 3], device=device),
            fidelity_idx=torch.zeros(1, dtype=torch.long, device=device),
        )
        outputs = []
        for network, enabled in ((reference, "0"), (model, "1")):
            monkeypatch.setenv("TACE_USE_EQX", enabled)
            output = network({key: value.clone() for key, value in data.items()})
            outputs.append(output)
            sum(
                output[key].square().sum() for key in ("energy", "forces", "stress")
            ).backward()
        for key in ("energy", "forces", "stress", "virials"):
            torch.testing.assert_close(
                outputs[0][key], outputs[1][key], atol=2e-9, rtol=2e-8
            )
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            expected = reference_parameters[name].grad
            if expected is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(
                    parameter.grad, expected, atol=2e-8, rtol=2e-7
                )
        if device == "cpu" and interaction == "cgtp" and not bias:
            next(reference.parameters()).requires_grad_(False)
            reference.retain_graph = True
            loaded = load_tace(reference, device="cpu")
            assert loaded is reference
            assert not loaded.readout_fn.representation.use_o2
            converted = convert_cgtp(reference)
            assert not converted.readout_fn.representation.use_packed_wigner
            for enabled in ("0", "1", "0"):
                monkeypatch.setenv("TACE_USE_EQX", enabled)
                assert not converted.readout_fn.representation.use_packed_wigner
                output = converted({key: value.clone() for key, value in data.items()})
                for key in ("energy", "forces", "stress", "virials"):
                    torch.testing.assert_close(
                        output[key], outputs[0][key], atol=2e-9, rtol=2e-8
                    )
            monkeypatch.setenv("TACE_USE_EQX", "0")
            restored = convert_cgtp(converted)
            assert not next(restored.parameters()).requires_grad
            assert restored.retain_graph
            assert not restored.readout_fn.representation.use_packed_wigner
            assert not any(
                getattr(layer, "use_eqx", False)
                for layer in restored.readout_fn.representation.interactions
            )
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(
                    restored.state_dict()[name], value, atol=0, rtol=0
                )
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mlp_type", ["mlp_v1", "mlp_v2"])
def test_tace_radial_bias_cutoff_and_force_training(
    monkeypatch, device, mlp_type, double_precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.fused import O3TensorProductConv
    from tace.models.mlp import MLP

    monkeypatch.setenv("TACE_USE_OEQ", "0")
    monkeypatch.setenv("TACE_USE_CUEQ", "0")
    monkeypatch.setenv("TACE_USE_EQX", "0")
    module = O3TensorProductConv("3x0e+3x1o", "0e+1o", "3x0e+3x1o+3x2e").to(device)
    mlp = MLP[mlp_type]([4, 5, module.weight_numel], bias=True).to(device)
    positions = torch.randn(5, 3, device=device, requires_grad=True)
    x = torch.randn(5, module.irreps_in1.dim, device=device, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 0]], device=device)
    vectors = positions[edges[1]] - positions[edges[0]]
    lengths = vectors.square().sum(-1, keepdim=True)
    radial = torch.cat([lengths**n for n in range(4)], -1)
    cutoff = torch.exp(-lengths)
    attrs = o3.spherical_harmonics(module.irreps_in2, vectors, normalize=True)
    expected = module(x, attrs, mlp(radial) * cutoff, edges)
    hidden = mlp.mlp[:-1](radial)
    last = mlp.mlp[-1]
    hidden = torch.cat((hidden, torch.ones_like(hidden[:, :1])), -1)
    weight = torch.cat((last.get_weight(), last.bias.unsqueeze(0)), 0)
    actual = module.forward_stream(x, attrs, hidden, weight, edges, cutoff)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    monkeypatch.setenv("TACE_USE_EQX", "1")
    torch.testing.assert_close(module(x, attrs, mlp(radial) * cutoff, edges), expected)
    forces = [
        torch.autograd.grad(value.square().sum(), positions, create_graph=True)[0]
        for value in (actual, expected)
    ]
    torch.testing.assert_close(*forces, atol=2e-9, rtol=2e-10)
    parameters = tuple(mlp.parameters()) + (x,)
    gradients = [
        torch.autograd.grad(force.square().sum(), parameters, retain_graph=True)
        for force in forces
    ]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-9)


@pytest.mark.parametrize(
    "interaction", ["cgtp", "o2_cgtp", "uu_o2", "o2", ["uu_o2", "o2"]]
)
@pytest.mark.parametrize(
    "mlp_type,mlp_norm",
    [
        ("mlp_v1", None),
        ("mlp_v2", None),
        ("mlp_v2", "layer_norm"),
        ("mlp_v2", "rms_norm"),
    ],
)
def test_tace_model_force_training(
    monkeypatch, double_precision, interaction, mlp_type, mlp_norm
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
    from tace.models._e3nn.fused import O3TensorProductConv, UuO2TensorProductConv
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    for name in ("EQX", "OEQ", "CUE", "EQT"):
        monkeypatch.setenv(f"TACE_USE_{name}", "0")
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=4.0,
        max_neighbors=None,
        num_layers=2,
        num_channel=3,
        Lmax=2,
        lmax=2,
        mmax=1,
        statistics=[
            dict(
                atomic_numbers=[1, 8],
                avg_num_neighbors=2.0,
                atomic_energy={1: 0.0, 8: 0.0},
            )
        ],
        target_property=["energy", "forces", "stress", "virials"],
    )
    config["atomic_basis"]["type"] = interaction
    config["atomic_basis"]["use_radial_rotary_attention"] = False
    config["atomic_basis"]["num_head"] = 1
    config["node_embedding"]["type"] = "linear"
    config["radial_basis"]["hidden"] = [4]
    config["radial_basis"]["bias"] = True
    config["radial_basis"]["apply_cutoff"] = False
    config["mlp_type"] = mlp_type
    config["radial_basis"]["mlp_norm"] = mlp_norm
    if mlp_type == "mlp_v2":
        config["edge_update"]["type"] = "element2_v2"
    config["readout_emlp"]["hidden"] = [3]
    config["readout_emlp"]["use_one_body_magmoms"] = False
    config["scale_shift"]["enable"] = False
    reference_model = TensorModel(e3nnTACE(**config)).cuda().train()
    monkeypatch.setenv("TACE_USE_EQX", "1")
    model = TensorModel(e3nnTACE(**config)).cuda().train()
    model.load_state_dict(reference_model.state_dict(), strict=True)
    assert model.state_dict().keys() == reference_model.state_dict().keys()
    data = dict(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]], device="cuda"
        ),
        node_attrs=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device="cuda"),
        edge_index=torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], device="cuda"
        ),
        edge_shifts=torch.zeros(6, 3, device="cuda"),
        lattice=torch.eye(3, device="cuda").unsqueeze(0) * 8,
        batch=torch.zeros(3, dtype=torch.long, device="cuda"),
        ptr=torch.tensor([0, 3], device="cuda"),
        fidelity_idx=torch.zeros(1, dtype=torch.long, device="cuda"),
    )

    def no_edge_message(*args):
        raise AssertionError(
            "The fused interaction must not materialize edge messages or weights"
        )

    if interaction == "uu_o2":
        monkeypatch.setattr(
            model.readout_fn.representation.o2_angular_basis,
            "forward",
            no_edge_message,
        )
        monkeypatch.setattr(
            model.readout_fn.representation.o2_angular_basis,
            "forward_packed",
            no_edge_message,
        )

    for layer in model.readout_fn.representation.interactions:
        if hasattr(layer.rejector, "tp"):
            monkeypatch.setattr(layer.rejector.tp, "forward", no_edge_message)
        elif isinstance(layer.rejector, UuO2TensorProductConv):
            monkeypatch.setattr(layer.rejector.linear, "forward", no_edge_message)
            monkeypatch.setattr(
                layer.rejector.local_frame_in, "to_local", no_edge_message
            )
            monkeypatch.setattr(
                layer.rejector.local_frame_out, "to_global", no_edge_message
            )
        if isinstance(layer.rejector, (O3TensorProductConv, UuO2TensorProductConv)):
            monkeypatch.setattr(layer.edge_info, "forward", no_edge_message)
    results = []
    for network, enabled in ((reference_model, "0"), (model, "1")):
        monkeypatch.setenv("TACE_USE_EQX", enabled)
        output = network({key: value.clone() for key, value in data.items()})
        results.append(output)
        sum(
            output[key].square().sum() for key in ("energy", "forces", "stress")
        ).backward()
    for key in ("energy", "forces", "stress", "virials"):
        torch.testing.assert_close(
            results[0][key], results[1][key], atol=2e-9, rtol=2e-8
        )
    reference_parameters = dict(reference_model.named_parameters())
    for name, parameter in model.named_parameters():
        expected = reference_parameters[name].grad
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, atol=2e-8, rtol=2e-7)


def time_reverse_features(features: torch.Tensor, irreps) -> torch.Tensor:
    output = features.clone()
    for ir_mul, ir_slice in zip(irreps, irreps.slices()):
        output[..., ir_slice] *= ir_mul.ir.t
    return output


@pytest.mark.parametrize("interaction", ["o2", "uu_o2"])
@pytest.mark.parametrize(("Lmax", "lmax"), [(2, 3), (3, 2)])
def test_o2_representation_uses_common_angular_coverage(Lmax, lmax, interaction):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["node_embedding"]["type"] = "linear"
    config["atomic_basis"]["type"] = [interaction]
    config["atomic_basis"]["nonlinear"] = ["gate"]
    config["atomic_basis"]["edge_nonlinear"] = ["gate"]
    config["atomic_basis"]["use_radial_rotary_attention"] = interaction != "uu_o2"
    config["product_basis"]["type"] = ["cgtp"]
    config["product_basis"]["correlation"] = [2]

    representation = NodeRepresentation(
        num_layers=1,
        atomic_numbers=[1],
        cutoff=3.0,
        avg_num_neighbors=2.0,
        magnetic_scale=None,
        mmax=2,
        Lmax=Lmax,
        lmax=lmax,
        angular_basis={
            "magnetic_Lmax": 1,
        },
        num_channel=2,
        target_irreps=o3.Irreps("0e"),
        node_embedding=config["node_embedding"],
        edge_embedding=config["edge_embedding"],
        edge_update=config["edge_update"],
        node_update=config["node_update"],
        radial_basis=config["radial_basis"],
        atomic_basis=config["atomic_basis"],
        resnet=config["resnet"],
        product_basis=config["product_basis"],
        invariant_property=[],
        equivariant_property=[],
        universal_embedding=config["universal_embedding"],
        layer_norm=config["layer_norm"],
        dropout=config["dropout"],
        parity=False,
    )

    common_lmax = max(Lmax, lmax)
    assert representation.use_o2
    assert not representation.use_so2
    assert not representation.use_time_reversal
    assert representation.o2_angular_basis.lmax == common_lmax
    rejector = representation.interactions[0].rejector
    assert rejector.local_frame_in.lmax == rejector.irreps_in.lmax

    model = torch.nn.ModuleDict({"representation": representation})
    state_dict = model.state_dict()
    assert not any("o2_angular_basis" in key for key in state_dict)
    for angular_basis in ("so2_angular_basis", "o2_angular_basis"):
        for name in ("wigner_index_to_m_array", "wigner_inv_rescale"):
            state_dict[f"representation.{angular_basis}.{name}"] = torch.empty(0)
    incompatible = model.load_state_dict(state_dict, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


@pytest.mark.parametrize(
    "magnetic_info_type",
    [None, "node"],
    ids=["default", "node"],
)
@pytest.mark.parametrize("magnetic_type", ["identity", "element", "element2"])
def test_magnetic_info_is_independent_per_interaction(
    magnetic_info_type,
    magnetic_type,
    monkeypatch,
):
    if magnetic_info_type is not None:
        monkeypatch.setattr(
            O2MagneticInteraction,
            "magnetic_info_type",
            magnetic_info_type,
        )
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["node_update"]["magnetic_type"] = magnetic_type
    config["atomic_basis"]["type"] = ["o2_mag", "o2_mag"]
    config["atomic_basis"]["nonlinear"] = ["gate", "gate"]
    config["atomic_basis"]["edge_nonlinear"] = ["gate", "gate"]
    config["product_basis"]["type"] = ["cgtp", "cgtp"]
    config["product_basis"]["correlation"] = [2, 2]

    representation = NodeRepresentation(
        num_layers=2,
        atomic_numbers=[26],
        cutoff=3.0,
        avg_num_neighbors=2.0,
        magnetic_scale=[{26: 2.0}],
        mmax=1,
        Lmax=1,
        lmax=1,
        angular_basis={
            "magnetic_Lmax": 1,
            "use_spin_orbit_coupling": False,
        },
        num_channel=2,
        target_irreps=o3.Irreps("0e"),
        node_embedding=config["node_embedding"],
        edge_embedding=config["edge_embedding"],
        edge_update=config["edge_update"],
        node_update=config["node_update"],
        radial_basis=config["radial_basis"],
        atomic_basis=config["atomic_basis"],
        resnet=config["resnet"],
        product_basis=config["product_basis"],
        invariant_property=[],
        equivariant_property=[],
        universal_embedding=config["universal_embedding"],
        layer_norm=config["layer_norm"],
        dropout=config["dropout"],
        parity=True,
    )

    num_mag_radial_basis = config["radial_basis"]["num_mag_radial_basis"]
    assert len(representation.node_updates) == 2
    assert representation.magnetic_edge_irreps_out.num_irreps == 2
    assert all(
        ir.l == 0 and ir.p == 1 for _, ir in representation.magnetic_edge_irreps_out
    )
    assert representation.node_updates[0] is not (representation.node_updates[1])
    for update, interaction in zip(
        representation.node_updates,
        representation.interactions,
    ):
        assert isinstance(update, NodeUpdate)
        assert isinstance(update, NODE_UPDATE[magnetic_type])
        assert interaction.edge_info.dims[0] == representation.edge_updates[0].out_dim
        expected_info_dims = [
            update.out_dim,
            *config["radial_basis"]["hidden"],
            interaction.magnetic_linear.weight_numel,
        ]
        assert interaction.source_magnetic_info.dims == expected_info_dims
        assert interaction.target_magnetic_info.dims == expected_info_dims
        assert all(
            mul == representation.num_channel
            for mul, _ in interaction.magnetic_edge_irreps_out
        )
        assert interaction.magnetic_linear.bias is None

        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        magnetic_radial_basis = torch.randn(3, num_mag_radial_basis, requires_grad=True)
        node_attrs = torch.ones(3, 1)
        magnetic_info = update(
            magnetic_radial_basis,
            node_attrs,
        )
        magnetic_edge_attrs = torch.randn(
            edge_index.size(1),
            interaction.magnetic_edge_irreps.dim,
        )
        magnetic_weights = interaction._magnetic_weights(
            magnetic_info,
            edge_index,
        )
        source_info, target_info = magnetic_info
        source, target = edge_index
        expected_weights = (
            interaction.source_magnetic_info(source_info)[source]
            * interaction.target_magnetic_info(target_info)[target]
        )
        torch.testing.assert_close(magnetic_weights, expected_weights)
        projected = interaction.magnetic_linear(
            magnetic_edge_attrs,
            magnetic_weights,
        )
        assert projected.shape == (
            edge_index.size(1),
            interaction.magnetic_edge_irreps_out.dim,
        )
        projected.square().sum().backward()
        assert torch.isfinite(magnetic_radial_basis.grad).all()
        for num_nodes in (0, 3):
            empty_edge_weights = interaction._magnetic_weights(
                update(magnetic_radial_basis[:num_nodes], node_attrs[:num_nodes]),
                edge_index[:, :0],
            )
            assert empty_edge_weights.shape == (
                0,
                interaction.magnetic_linear.weight_numel,
            )


def _scatter_module(use_attention, linear_type="uv"):
    irreps = o3.Irreps("2x0e+2x0o+2x1o+2x1e")
    if linear_type == "uu":
        return UuO2TensorProductConv(irreps, irreps, num_channel=2, mmax=1).to(
            DEVICE, DTYPE
        )
    return UvO2TensorProductConv(
        irreps,
        irreps,
        num_channel=2,
        mmax=1,
        even_scalar_act=torch.nn.SiLU(),
        odd_scalar_act=torch.nn.Tanh(),
        tensor_act=torch.nn.Sigmoid(),
        num_head=1,
        num_radial_basis=4,
        use_radial_rotary_attention=use_attention,
    ).to(DEVICE, DTYPE)


@pytest.mark.parametrize(
    ("use_attention", "linear_type"), [(False, "uv"), (True, "uv"), (False, "uu")]
)
@pytest.mark.parametrize("reflected", [False, True])
def test_o2_scatter_is_o3_equivariant(use_attention, linear_type, reflected):
    torch.manual_seed(7)
    module = _scatter_module(use_attention, linear_type)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], device=DEVICE)
    node_features = module.irreps_in.randn(3, -1, dtype=DTYPE, device=DEVICE)
    edge_vectors = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(4, module.weight_numel, dtype=DTYPE, device=DEVICE)
    radial = torch.randn(4, 4, dtype=DTYPE, device=DEVICE)
    kwargs = {"edge_radial_basis": radial} if linear_type == "uv" else {}
    cutoff = torch.rand(4, 1, dtype=DTYPE, device=DEVICE)
    wigner_module = o2.WignerD(1, 1).to(DEVICE, DTYPE)
    wigner, wigner_inv = wigner_module(edge_vectors)
    output = module(
        node_features,
        weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff=cutoff,
        **kwargs,
    )
    if linear_type == "uu":
        torch.testing.assert_close(
            module(
                node_features,
                2 * weights,
                edge_index,
                wigner,
                wigner_inv,
                edge_cutoff=cutoff,
                **kwargs,
            ),
            2 * output,
        )

    rotation = o3.rand_matrix(dtype=DTYPE, device=DEVICE) * (-1 if reflected else 1)
    matrix = module.irreps_in.D_from_matrix(rotation.cpu()).to(node_features)
    rotated_features = node_features @ matrix.T
    rotated_vectors = edge_vectors @ rotation.T
    rotated_wigner, rotated_wigner_inv = wigner_module(rotated_vectors)
    rotated_output = module(
        rotated_features,
        weights,
        edge_index,
        rotated_wigner,
        rotated_wigner_inv,
        edge_cutoff=cutoff,
        **kwargs,
    )
    torch.testing.assert_close(
        rotated_output,
        output @ matrix.T,
        atol=2.0e-6,
        rtol=2.0e-5,
    )


def test_uu_o2_scatter_uses_only_source_features(monkeypatch):
    torch.manual_seed(8)
    module = _scatter_module(False, "uu")
    assert module.linear.irreps_in == module.local_frame_in.irreps_out
    assert not hasattr(module, "attention")
    assert not hasattr(module, "linear_type")
    assert (
        sum(isinstance(layer, (o2.Linear, o2.UuLinear)) for layer in module.modules())
        == 1
    )
    edge_index = torch.tensor([[0, 0], [1, 2]], device=DEVICE)
    node_features = module.irreps_in.randn(
        3, -1, dtype=DTYPE, device=DEVICE, requires_grad=True
    )
    weights = torch.randn(2, module.weight_numel, dtype=DTYPE, device=DEVICE)
    cutoff = torch.rand(2, 1, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(1, 1).to(DEVICE, DTYPE)(
        torch.randn(2, 3, dtype=DTYPE, device=DEVICE)
    )
    to_local = Mock(wraps=module.local_frame_in.to_local)
    monkeypatch.setattr(module.local_frame_in, "to_local", to_local)
    output = module(
        node_features,
        weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff=cutoff,
    )
    to_local.assert_called_once()
    torch.testing.assert_close(
        to_local.call_args.args[0], module.reshape_in(node_features)[edge_index[0]]
    )
    changed_features = node_features.detach().clone()
    changed_features[1:] = torch.randn_like(changed_features[1:])
    torch.testing.assert_close(
        module(
            changed_features,
            weights,
            edge_index,
            wigner,
            wigner_inv,
            edge_cutoff=cutoff,
        ),
        output,
        atol=0,
        rtol=0,
    )
    gradients = torch.autograd.grad(output.square().sum(), node_features)[0]
    assert gradients[0].abs().sum() > 0
    torch.testing.assert_close(gradients[1:], torch.zeros_like(gradients[1:]))


@pytest.mark.parametrize("linear_type", ["uv", "uu"])
@pytest.mark.parametrize("num_nodes", [0, 3])
def test_o2_scatter_supports_empty_edges(linear_type, num_nodes):
    module = _scatter_module(False, linear_type)
    assert "reshape_in" not in repr(module)
    assert "reshape_out" not in repr(module)
    edge_index = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(1, 1).to(DEVICE, DTYPE)(
        torch.empty(0, 3, dtype=DTYPE, device=DEVICE)
    )
    output = module(
        module.irreps_in.randn(num_nodes, -1, dtype=DTYPE, device=DEVICE),
        torch.empty(0, module.weight_numel, dtype=DTYPE, device=DEVICE),
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff=torch.empty(0, 1, dtype=DTYPE, device=DEVICE),
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.parametrize("use_attention", [False, True])
def test_uu_o2_interaction_trains_forces_and_uses_external_weights(
    double_precision, use_attention
):
    from tace.models._e3nn.inter import UuO2Interaction, UvO2Interaction
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=4.0,
        max_neighbors=None,
        num_layers=2,
        num_channel=2,
        Lmax=1,
        lmax=2,
        mmax=1,
        parity=True,
        statistics=[
            dict(atomic_numbers=[1], avg_num_neighbors=2.0, atomic_energy={1: 0.0})
        ],
        target_property=["energy", "forces", "stress", "virials"],
    )
    config["atomic_basis"].update(
        type="uu_o2",
        edge_nonlinear=None,
        use_radial_rotary_attention=use_attention,
        num_head=1,
    )
    config["node_embedding"]["type"] = "linear"
    config["readout_emlp"]["use_one_body_magmoms"] = False
    config["readout_emlp"]["hidden"] = [2]
    config["radial_basis"]["hidden"] = [4]
    config["radial_basis"]["apply_cutoff"] = False
    config["scale_shift"]["enable"] = False
    if use_attention:
        with pytest.raises(
            ValueError, match="uu_o2 does not support radial rotary attention"
        ):
            e3nnTACE(**config)
        return
    model = TensorModel(e3nnTACE(**config)).train()
    assert model.readout_fn.representation.use_o2
    assert model.readout_fn.representation._can_pack_wigner
    for interaction in model.readout_fn.representation.interactions:
        assert isinstance(interaction, UuO2Interaction)
        assert not isinstance(interaction, UvO2Interaction)
        assert isinstance(interaction.rejector, UuO2TensorProductConv)
        assert isinstance(interaction.rejector.linear, o2.UuLinear)
        assert not hasattr(interaction.rejector, "nonlinearity")
        assert not hasattr(interaction.rejector, "linear_up")
        assert not hasattr(interaction.rejector, "linear_down")
        assert not hasattr(interaction.rejector, "attention")
        assert list(interaction.rejector.linear.parameters()) == []
        edge_features = torch.randn(3, interaction.edge_feats_channel)
        weights = interaction.edge_info(edge_features)
        assert weights.shape == (3, interaction.rejector.linear.weight_numel)

    data = dict(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]]),
        node_attrs=torch.ones(3, 1),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
        edge_shifts=torch.zeros(6, 3),
        lattice=torch.eye(3).unsqueeze(0) * 8,
        batch=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3]),
        fidelity_idx=torch.zeros(1, dtype=torch.long),
    )
    output = model(data)
    for name in ("energy", "forces", "stress", "virials"):
        assert torch.isfinite(output[name]).all()
    output["forces"].square().sum().backward()
    for interaction in model.readout_fn.representation.interactions:
        gradients = [p.grad for p in interaction.edge_info.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


def _magnetic_scatter_module(use_attention: bool):
    irreps = o3.Irreps("2x0ee+2x1oe")
    magnetic_edge_irreps = o3.Irreps("2x0ee+2x1eo+2x1ee")
    module = MagneticUvO2TensorProductConv(
        irreps,
        irreps,
        magnetic_edge_irreps,
        num_channel=2,
        mmax=1,
        even_scalar_act=torch.nn.SiLU(),
        odd_scalar_act=torch.nn.Tanh(),
        tensor_act=torch.nn.Sigmoid(),
        num_head=1,
        num_radial_basis=4,
        use_radial_rotary_attention=use_attention,
    ).to(DEVICE, DTYPE)
    return module, irreps, magnetic_edge_irreps


@pytest.mark.parametrize("use_attention", [False, True])
@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed e3nn does not expose time-reversal irreps.",
)
def test_o2_magnetic_scatter_is_time_reversal_invariant(use_attention):
    module, irreps, magnetic_edge_irreps = _magnetic_scatter_module(use_attention)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], device=DEVICE)
    num_edges = edge_index.size(1)
    node_features = torch.randn(4, irreps.dim, dtype=DTYPE, device=DEVICE)
    magnetic_edge_attrs = torch.randn(
        num_edges, magnetic_edge_irreps.dim, dtype=DTYPE, device=DEVICE
    )
    conv_weights = torch.randn(
        num_edges, module.weight_numel, dtype=DTYPE, device=DEVICE
    )
    edge_vectors = torch.randn(num_edges, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(mmax=1, lmax=1).to(DEVICE, DTYPE)(edge_vectors)
    edge_cutoff = torch.rand(num_edges, 1, dtype=DTYPE, device=DEVICE)
    edge_radial_basis = torch.randn(num_edges, 4, dtype=DTYPE, device=DEVICE)

    def apply(edge_attrs):
        return module(
            node_features,
            edge_attrs,
            conv_weights,
            edge_index,
            wigner,
            wigner_inv,
            edge_radial_basis=edge_radial_basis,
            edge_cutoff=edge_cutoff,
        )

    output = apply(magnetic_edge_attrs)
    torch.testing.assert_close(
        apply(time_reverse_features(magnetic_edge_attrs, magnetic_edge_irreps)),
        output,
    )
    assert not torch.allclose(output, apply(torch.zeros_like(magnetic_edge_attrs)))


@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed e3nn does not expose time-reversal irreps.",
)
def test_o2_magnetic_scatter_supports_empty_edges():
    module, irreps, magnetic_edge_irreps = _magnetic_scatter_module(False)
    assert "reshape_in" not in repr(module)
    assert "reshape_out" not in repr(module)
    edge_index = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(mmax=1, lmax=1).to(DEVICE, DTYPE)(
        torch.empty(0, 3, dtype=DTYPE, device=DEVICE)
    )
    output = module(
        torch.randn(3, irreps.dim, dtype=DTYPE, device=DEVICE),
        torch.empty(
            0,
            magnetic_edge_irreps.dim,
            dtype=DTYPE,
            device=DEVICE,
        ),
        torch.empty(0, module.weight_numel, dtype=DTYPE, device=DEVICE),
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff=torch.empty(0, 1, dtype=DTYPE, device=DEVICE),
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.parametrize(
    ("irreps_in", "irreps_sh", "irreps_out"),
    [
        ("2x3o", "0e+1o", "2x2e"),
        ("2x1o", "0e+1o+2e+3o", "2x2e"),
        ("2x1o", "0e+1o", "2x2e"),
    ],
)
def test_o2_cgtp_infers_degrees_and_accepts_larger_shared_wigner(
    double_precision, monkeypatch, irreps_in, irreps_sh, irreps_out
):
    from types import SimpleNamespace

    from tace.models._e3nn.fused import (
        O2TensorProductConv,
        O3TensorProductConv,
    )

    for name in ("TACE_USE_OEQ", "TACE_USE_CUE"):
        monkeypatch.setenv(name, "0")
    module = O2TensorProductConv(irreps_in, irreps_sh, irreps_out)
    reference = O3TensorProductConv(irreps_in, irreps_sh, irreps_out)
    lmax = max(o3.Irreps(irreps).lmax for irreps in (irreps_in, irreps_out))
    assert module.tp.lmax == lmax
    assert module.irreps_out == reference.irreps_out
    assert module.weight_numel == reference.weight_numel

    x = torch.randn(3, o3.Irreps(irreps_in).dim, requires_grad=True)
    r = torch.randn(6, 3, requires_grad=True)
    w = torch.randn(6, module.weight_numel, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    graph = SimpleNamespace(
        edge_vector=r,
        edge_length=r.norm(dim=-1, keepdim=True),
    )
    d, di = o2.WignerD(lmax + 2, lmax + 2)(r)
    actual = module(x, w, edge_index, d, di, graph)
    expected = reference(
        x,
        o3.spherical_harmonics(
            o3.Irreps(irreps_sh), r / graph.edge_length, False, "component"
        ),
        w,
        edge_index,
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.square().sum(), (x, r, w), retain_graph=True),
        torch.autograd.grad(expected.square().sum(), (x, r, w)),
    ):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-8, rtol=2e-9)
