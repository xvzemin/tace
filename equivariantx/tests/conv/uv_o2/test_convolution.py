"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx import conv as eqx_conv
from equivariantx import o2
from equivariantx.tests.conv.testing import assert_native_autograd


class RadialConvolution(torch.nn.Module):
    """Small non-attention UV operator with explicit learned radial layers."""

    def __init__(self, backend="torch", radial_mode="last"):
        super().__init__()
        frame = o2.LocalFrame("2x0e+2x1e+2x1o")
        gate = o2.Gate(
            "2x0e+2x0o",
            [torch.nn.SiLU(), torch.nn.Tanh()],
            "2x0o",
            [torch.nn.Tanh()],
            "2x1m",
        )
        self.up = o2.Linear((frame.irreps_out * 2).regroup(), gate.irreps_in)
        self.down = o2.Linear(gate.irreps_out, frame.irreps_out)
        self.conv = eqx_conv.UvO2TensorProductConv(
            frame,
            self.up,
            gate,
            self.down,
            frame,
            backend=backend,
            radial_mode=radial_mode,
        )
        self.radial = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, self.conv.weight_numel),
        )

    def forward(self, features, vectors, index):
        radius = vectors.norm(dim=-1, keepdim=True)
        basis = torch.cat((radius, radius.sin(), radius.cos()), dim=-1)
        return self.conv(
            features,
            None,
            basis,
            index,
            None,
            None,
            radius.sigmoid(),
            (self.up.weight, self.up.bias, self.down.weight, self.down.bias),
            vectors=vectors,
            radial_network=self.radial,
        )


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("radial_mode", ["last", "full"])
def test_transverse_functional(backend, radial_mode, monkeypatch, double_precision):
    from equivariantx.ops import config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = RadialConvolution(backend, radial_mode).to(device)
    index = torch.tensor([[0, 1, 1, 2, 0], [1, 2, 0, 1, 0]], device=device)
    inputs = (
        torch.randn(3, module.conv.input_dim, device=device),
        torch.randn(5, 3, device=device),
    )
    directions = tuple(torch.randn_like(value) for value in inputs)

    def function(x, v):
        return module(x, v, index)

    actual = torch.func.jvp(function, inputs, directions)
    batched = tuple(torch.stack((value, value * 0.9)) for value in inputs)
    batch_actual = torch.vmap(function)(*batched)
    module.conv.backend = "torch"
    for frame in module.conv.transverse_frames:
        if frame is not None:
            frame.backend = "torch"
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    torch.testing.assert_close(
        actual, torch.func.jvp(function, inputs, directions), atol=2e-10, rtol=2e-10
    )
    torch.testing.assert_close(
        batch_actual, torch.vmap(function)(*batched), atol=2e-10, rtol=2e-10
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor package")
@pytest.mark.parametrize("radial_mode", ["last", "full"])
def test_transverse_aoti(radial_mode, tmp_path, monkeypatch, double_precision):
    import os
    import subprocess
    import sys

    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config

    module = RadialConvolution("cuda", radial_mode).cuda()
    index = torch.tensor([[0, 1, 1, 2, 0], [1, 2, 0, 1, 0]], device="cuda")
    x = torch.randn(3, module.conv.input_dim, device="cuda", requires_grad=True)
    vectors = torch.randn(5, 3, device="cuda", requires_grad=True)
    names, parameters = zip(*module.named_parameters())

    def joint(x, vectors, *parameters):
        output = torch.func.functional_call(
            module, dict(zip(names, parameters)), (x, vectors, index)
        )
        energy = output.square().mean()
        force = -torch.autograd.grad(energy, vectors, create_graph=True)[0]
        gradients = torch.autograd.grad(
            energy + (force - 0.1).square().sum(), parameters
        )
        return energy, force, *gradients

    inputs = x, vectors, *parameters
    graph = make_fx(joint)(*inputs)
    assert not any(str(node.target).startswith("eqx.") for node in graph.graph.nodes)
    exported = torch.export.export(graph, tuple(value.detach() for value in inputs))
    package = tmp_path / "transverse.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    module.conv.backend = "torch"
    for frame in module.conv.transverse_frames:
        if frame is not None:
            frame.backend = "torch"
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    cases = []
    for scale in (1.0, 0.8):
        values = tuple((value.detach() * scale).requires_grad_() for value in inputs)
        cases.append(
            {
                "inputs": tuple(value.detach().cpu() for value in values),
                "expected": tuple(value.detach().cpu() for value in joint(*values)),
            }
        )
    data = tmp_path / "cases.pt"
    torch.save(cases, data)
    script = """
import sys
import torch
runner = torch._inductor.aoti_load_package(sys.argv[1])
for case in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad():
        actual = runner(*(value.cuda() for value in case['inputs']))
    torch.testing.assert_close(tuple(value.cpu() for value in actual), case['expected'], atol=1e-9, rtol=1e-9)
assert 'equivariantx' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(package), str(data)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tiled compilation")
@pytest.mark.parametrize("compiler", ["inductor", "aoti"])
def test_full_radial_native_joint(compiler, tmp_path, monkeypatch, double_precision):
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config

    monkeypatch.setattr(config, "EDGE_PROGRAM_INLINE", False)
    module = RadialConvolution("cuda", "full").cuda()
    index = torch.tensor([[0, 1, 1, 2, 0], [1, 2, 0, 1, 0]], device="cuda")
    names, parameters = zip(*module.named_parameters())
    inputs = (
        torch.randn(3, module.conv.input_dim, device="cuda", requires_grad=True),
        torch.randn(5, 3, device="cuda", requires_grad=True),
        *parameters,
    )

    def joint(x, vectors, *parameters):
        output = torch.func.functional_call(
            module, dict(zip(names, parameters)), (x, vectors, index)
        )
        energy = output.square().mean()
        force = -torch.autograd.grad(energy, vectors, create_graph=True)[0]
        loss = energy + (force - 0.1).square().sum()
        return loss, force, *torch.autograd.grad(loss, parameters)

    graph = make_fx(joint)(*inputs)
    assert any(
        node.target == torch.ops.eqx.edge_program.default for node in graph.graph.nodes
    )
    if compiler == "aoti":
        exported = torch.export.export(graph, tuple(value.detach() for value in inputs))
        package = torch._inductor.aoti_compile_and_package(
            exported, package_path=str(tmp_path / "tiled.pt2")
        )
        run = torch._inductor.aoti_load_package(package)
    else:
        run = torch.compile(graph, fullgraph=True)
    module.conv.backend = "torch"
    for frame in module.conv.transverse_frames:
        if frame is not None:
            frame.backend = "torch"
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for scale in (1.0, 0.8):
        values = tuple((value.detach() * scale).requires_grad_() for value in inputs)
        expected = joint(*values)
        with torch.no_grad():
            actual = run(*(value.detach() for value in values))
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize(
    "backend,reference", [("torch", False), ("torch", True), ("cuda", False)]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "mmax,magnetic,attention,odd",
    [
        (0, False, False, False),
        (2, False, False, False),
        (2, True, False, True),
        (1, True, True, False),
        (2, True, True, True),
    ],
)
def test_uv_o2_transverse(
    backend,
    device,
    reference,
    mmax,
    magnetic,
    attention,
    odd,
    double_precision,
    monkeypatch,
):
    from equivariantx.ops import config

    monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
    monkeypatch.setattr(config, "TORCH_RECOMPUTE", not reference)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.conv.attention import graph_softmax

    if backend == "cuda" and device == "cuda":
        import equivariantx.ops.common.cuda.execution as execution

        launch = execution.launch
        monkeypatch.setattr(
            execution, "launch", lambda *args: launch(*args, tile_size=3)
        )

    frame = o2.LocalFrame("2x0e+2x0o+2x1o+2x1e+2x2e+2x2o", mmax).to(device)
    scalars = frame.irreps_out.filter(keep=lambda ir_mul: ir_mul.ir.m == 0)
    tensors = frame.irreps_out.filter(keep=lambda ir_mul: ir_mul.ir.m > 0)
    gates = (
        o2.Irreps([(o2.Irrep("0o" if odd else "0e"), tensors.num_irreps)])
        if tensors
        else o2.Irreps("")
    )
    gate = o2.Gate(
        scalars,
        [
            torch.nn.Tanh() if ir.is_odd_scalar() else torch.nn.SiLU()
            for ir, _ in scalars
        ],
        gates,
        [torch.nn.Tanh() if odd else torch.nn.Sigmoid()] if tensors else [],
        tensors,
    ).to(device)
    inputs = (frame.irreps_out * (3 if magnetic else 2)).regroup()
    up = o2.Linear(inputs, gate.irreps_in, biases=True).to(device)
    down = o2.Linear(gate.irreps_out, frame.irreps_out, biases=True).to(device)
    query, key = [
        o2.Linear(frame.irreps_out, frame.irreps_out).to(device) for _ in range(2)
    ]
    conv = eqx_conv.UvO2TensorProductConv(
        frame,
        up,
        gate,
        down,
        frame,
        frame_edge=frame if magnetic else None,
        query=query if attention else None,
        key=key if attention else None,
        num_heads=2,
        attention_scale=0.2,
        backend=backend,
    ).to(device)
    if backend == "cuda" and device == "cuda" and not attention:
        from equivariantx.ops.torch.conv.uv_o2 import convolution as uv_convolution

        def no_full_edge_linear(*args, **kwargs):
            raise AssertionError(
                "Non-attention O2 must keep channel maps inside edge tiles"
            )

        monkeypatch.setattr(uv_convolution, "linear_blocks", no_full_edge_linear)
    parameters = tuple(
        p
        for module in (up, down, *((query, key) if attention else ()))
        for p in (module.weight, module.bias)
    )
    rotation = o2.WignerD(2, 2, method="recursive").to(device)
    for edges in (8, 0):

        def rand(*shape):
            return (torch.randn(*shape, device=device) * 0.3).requires_grad_()

        index = torch.randint(4, (2, edges), device=device)
        x, extra = (
            rand(4, frame.input_dim),
            rand(edges, frame.input_dim) if magnetic else None,
        )
        vectors = (
            torch.cat((torch.eye(3), -torch.eye(3), torch.randn(2, 3)))
            .to(device)[:edges]
            .requires_grad_()
        )
        weights, cutoff, radial = (
            rand(edges, conv.weight_numel),
            rand(edges, 1).sigmoid(),
            rand(edges, 4),
        )
        wigner, inverse = rotation(vectors)
        source, target = (
            frame.to_local(x[index[0]], wigner),
            frame.to_local(x[index[1]], wigner),
        )
        local = [(frame.irreps_out, target), (frame.irreps_out, source)]
        if magnetic:
            local.append((frame.irreps_out, frame.to_local(extra, wigner)))
        parts, offset = [], 0
        for ir, mul in inputs:
            value = torch.cat(
                [
                    value[:, s].reshape(edges, ir.dim, width)
                    for irreps, value in local
                    for (ir_in, width), s in zip(irreps, irreps.slices())
                    if ir == ir_in
                ],
                -1,
            )
            parts.append((value * weights[:, None, offset : offset + mul]).flatten(1))
            offset += mul
        message = down(gate(up(torch.cat(parts, -1))))
        if attention:
            q, k = query(target), key(source)
            score = sum(
                (q[:, s] * k[:, s]).reshape(edges, ir.dim, 2, mul // 2).sum((1, 3))
                for (ir, mul), s in zip(frame.irreps_out, frame.irreps_out.slices())
            )
            score = score * 0.2 * (2 * radial[:, :2].sigmoid()) + radial[:, 2:]
            scale = graph_softmax(score, index[1], 4, cutoff, fused=False) * cutoff
            message = torch.cat(
                [
                    (
                        message[:, s].reshape(edges, ir.dim, 2, mul // 2)
                        * scale[:, None, :, None]
                    ).flatten(1)
                    for (ir, mul), s in zip(frame.irreps_out, frame.irreps_out.slices())
                ],
                -1,
            )
        else:
            message = message * cutoff
        expected = x.new_zeros(4, frame.input_dim).index_add(
            0, index[1], frame.to_global(message, inverse)
        )
        arguments = (
            x,
            vectors,
            weights,
            cutoff,
            *((extra,) if magnetic else ()),
            *((radial,) if attention else ()),
        ) + tuple(p for p in parameters if p.requires_grad)
        for transverse in (False, True):
            actual = conv(
                x,
                extra,
                weights,
                index,
                None if transverse else wigner,
                None if transverse else inverse,
                cutoff,
                parameters,
                radial if attention else None,
                vectors=vectors if transverse else None,
            )
            torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-10)
            if backend == "torch" and reference:
                assert_native_autograd(actual)
            a, b = actual, expected
            for order in range(3 if edges else 0):
                gradients = [
                    torch.autograd.grad(
                        value.sin().sum(),
                        arguments,
                        create_graph=True,
                        retain_graph=True,
                    )
                    for value in (a, b)
                ]
                for slot, (ga, gb) in enumerate(zip(*gradients)):
                    torch.testing.assert_close(
                        ga,
                        gb,
                        atol=3e-8,
                        rtol=3e-8,
                        msg=lambda message: (
                            f"transverse={transverse}, order={order + 1}, input={slot}: {message}"
                        ),
                    )
                a, b = [torch.cat([g.flatten() for g in gs]) / 30 for gs in gradients]
                if backend == "torch" and reference:
                    assert_native_autograd(a)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("geometry", ["vectors", "packed", "dense"])
def test_uv_o2_transverse_compile(backend, geometry, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    frame = o2.LocalFrame("2x0e+2x1e+2x1o").cuda()
    gate = o2.Gate(
        "2x0e+2x0o",
        [torch.nn.SiLU(), torch.nn.Tanh()],
        "2x0o",
        [torch.nn.Tanh()],
        "2x1m",
    ).cuda()
    up = o2.Linear((frame.irreps_out * 2).regroup(), gate.irreps_in).cuda()
    down = o2.Linear(gate.irreps_out, frame.irreps_out).cuda()
    module = eqx_conv.UvO2TensorProductConv(
        frame, up, gate, down, frame, backend=backend
    ).cuda()
    rotation = o2.WignerD(1, 1, method="recursive").cuda()

    def evaluate(x, vectors, weights, cutoff, up_weight, down_weight, index):
        if geometry == "packed":
            wigner, inverse = rotation.forward_packed(vectors), None
        elif geometry == "dense":
            wigner, inverse = rotation(vectors)
        else:
            wigner = inverse = None
        return module(
            x,
            None,
            weights,
            index,
            wigner,
            inverse,
            cutoff,
            (up_weight, up.bias, down_weight, down.bias),
            vectors=vectors if geometry == "vectors" else None,
        )

    compiled = torch.compile(
        evaluate, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for nodes, edges in ((4, 8), (5, 11), (3, 0), (0, 0)):
        x = torch.randn(nodes, frame.input_dim, device="cuda", requires_grad=True)
        vectors = torch.randn(edges, 3, device="cuda", requires_grad=True)
        weights = torch.randn(
            edges, module.weight_numel, device="cuda", requires_grad=True
        )
        cutoff = torch.rand(edges, 1, device="cuda", requires_grad=True)
        index = torch.randint(max(1, nodes), (2, edges), device="cuda")
        args = x, vectors, weights, cutoff, up.weight, down.weight, index
        actual, expected = compiled(*args), evaluate(*args)
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
        gradients = [
            torch.autograd.grad(
                result.square().sum(),
                (x, vectors, weights, up.weight, down.weight),
            )
            for result in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
        if edges == 8:
            from torch.fx.experimental.proxy_tensor import make_fx

            def energy_forces(x, vectors, weights, cutoff, up_weight, down_weight):
                output = evaluate(
                    x,
                    vectors,
                    weights,
                    cutoff,
                    up_weight,
                    down_weight,
                    index,
                )
                energy = output.square().sum()
                forces = -torch.autograd.grad(energy, vectors, create_graph=True)[0]
                return energy, forces

            inputs = x, vectors, weights, cutoff, up.weight, down.weight
            graph = make_fx(energy_forces)(*inputs)
            # This function has no intentional detach. make_fx inserts
            # detaches for saved activation outputs; remove them before
            # differentiating the force graph, as in model compilation.
            for node in list(graph.graph.nodes):
                if (
                    node.op == "call_function"
                    and node.target == torch.ops.aten.detach.default
                ):
                    node.replace_all_uses_with(node.args[0])
                    graph.graph.erase_node(node)
            graph.recompile()
            force_model = torch.compile(graph, backend="aot_eager", fullgraph=True)

            actual, expected = force_model(*inputs), energy_forces(*inputs)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
            gradients = [
                torch.autograd.grad(
                    energy + forces.square().sum(), (weights, up.weight, down.weight)
                )
                for energy, forces in (actual, expected)
            ]
            for a, b in zip(*gradients):
                torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("radial_mode", ["last", "full"])
@pytest.mark.parametrize("width", [64, 128, 1024])
def test_radial_projection(backend, radial_mode, width, monkeypatch, double_precision):
    from equivariantx.ops import config
    from equivariantx.ops.common import radial as radial_ops

    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    frame = o2.LocalFrame("2x0e+2x1e+2x1o").to(device)
    gate = o2.Gate(
        "2x0e+2x0o",
        [torch.nn.SiLU(), torch.nn.Tanh()],
        "2x0o",
        [torch.nn.Tanh()],
        "2x1m",
    ).to(device)
    up = o2.Linear((frame.irreps_out * 2).regroup(), gate.irreps_in).to(device)
    down = o2.Linear(gate.irreps_out, frame.irreps_out).to(device)
    module = eqx_conv.UvO2TensorProductConv(
        frame, up, gate, down, frame, backend=backend, radial_mode=radial_mode
    ).to(device)
    network = torch.nn.Sequential(
        torch.nn.Linear(3, width),
        torch.nn.SiLU(),
        torch.nn.Sequential(
            torch.nn.Linear(width, module.weight_numel), torch.nn.Tanh()
        ),
    ).to(device)
    x = torch.randn(3, frame.input_dim, device=device, requires_grad=True)
    vectors = torch.randn(5, 3, device=device, requires_grad=True)
    radial = torch.randn(5, 3, device=device, requires_grad=True)
    index = torch.randint(3, (2, 5), device=device)
    cutoff = torch.rand(5, 1, device=device)
    parameters = up.weight, up.bias, down.weight, down.bias
    materialized_widths = []
    original = radial_ops.radial_features

    def evaluate_hidden(*args, **kwargs):
        value = original(*args, **kwargs)
        materialized_widths.append(value.shape[-1])
        return value

    monkeypatch.setattr(radial_ops, "radial_features", evaluate_hidden)
    actual = module(
        x,
        None,
        radial,
        index,
        None,
        None,
        cutoff,
        parameters,
        vectors=vectors,
        radial_network=network,
    )
    assert materialized_widths == ([width] if radial_mode == "last" else [])
    module.backend = "torch"
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = module(
        x,
        None,
        network(radial),
        index,
        None,
        None,
        cutoff,
        parameters,
        vectors=vectors,
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    variables = x, vectors, radial, up.weight, down.weight, *network.parameters()
    for _ in range(3):
        seed = torch.randn_like(actual) * 0.1
        gradients = [
            torch.autograd.grad(
                (value.sin() * seed).sum(), variables, create_graph=True
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-8, rtol=1e-8)
        actual, expected = [
            torch.cat([g.flatten() for g in values]) for values in gradients
        ]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("promoted", [False, True])
@pytest.mark.parametrize("radial_mode", ["last", "full"])
def test_reference_bypasses_geometry_children(
    device, promoted, radial_mode, monkeypatch, double_precision
):
    from equivariantx.ops import config

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    dtype = torch.float32 if promoted else torch.float64
    tolerance = 3e-5 if promoted else 2e-10
    module = RadialConvolution("cuda", radial_mode).to(device=device, dtype=dtype)
    module.conv.backend = "torch"
    module.conv.wigner.backend = "cuda"
    reference = RadialConvolution("torch", radial_mode).to(device=device, dtype=dtype)
    reference.load_state_dict(module.state_dict())

    def forbidden(*args, **kwargs):
        raise AssertionError("The parent reference called a geometry child.")

    monkeypatch.setattr(module.conv.wigner, "forward_packed", forbidden)
    for frame in module.conv.transverse_frames:
        if frame is not None:
            monkeypatch.setattr(frame, "forward", forbidden)
            for coupling in frame.couplings.values():
                coupling.backend = "cuda"
                monkeypatch.setattr(coupling, "forward", forbidden)
    backends = {
        name: child.backend
        for name, child in module.named_modules()
        if hasattr(child, "backend")
    }
    for edges in (5, 0):
        index = torch.stack(
            (
                torch.arange(edges, device=device) % 3,
                (torch.arange(edges, device=device) + 1) % 3,
            )
        )
        inputs = (
            torch.randn(
                3, module.conv.input_dim, device=device, dtype=dtype
            ).requires_grad_(),
            (
                torch.randn(edges, 3, device=device, dtype=torch.float64) + 1
            ).requires_grad_(),
        )
        a, b = module(*inputs, index), reference(*inputs, index)
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
        assert_native_autograd(a)
        for _ in range(3):
            gradients = [
                torch.autograd.grad(
                    value.sin().sum(), inputs, create_graph=True, allow_unused=True
                )
                for value in (a, b)
            ]
            gradients = [
                tuple(
                    torch.zeros_like(x) if g is None else g
                    for x, g in zip(inputs, values)
                )
                for values in gradients
            ]
            torch.testing.assert_close(*gradients, atol=tolerance, rtol=tolerance)
            a, b = [
                sum(g.sum() / max(1, g.numel()) for g in values)
                + sum(x.sum() * 0 for x in inputs)
                for values in gradients
            ]
            assert_native_autograd(a)
        tangents = tuple(torch.randn_like(x) for x in inputs)
        with torch.autograd.forward_ad.dual_level():
            duals = tuple(
                torch.autograd.forward_ad.make_dual(x, v)
                for x, v in zip(inputs, tangents)
            )
            actual = torch.autograd.forward_ad.unpack_dual(module(*duals, index))
        expected = torch.autograd.functional.jvp(
            lambda *args: reference(*args, index), inputs, tangents
        )
        assert actual.tangent is not None
        torch.testing.assert_close(
            tuple(actual), expected, atol=tolerance, rtol=tolerance
        )
        assert backends == {
            name: child.backend
            for name, child in module.named_modules()
            if hasattr(child, "backend")
        }
        assert config.TORCH_REFERENCE is True


def test_reference_mixed_children_dynamic_aoti(monkeypatch, tmp_path, double_precision):
    import os
    import subprocess
    import sys
    import zipfile

    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config

    class ScalarConvolution(RadialConvolution):
        def __init__(self, backend):
            torch.nn.Module.__init__(self)
            frame = o2.LocalFrame("0e+1o", mmax=0)
            gate = o2.Gate(frame.irreps_out, [torch.nn.SiLU()], "", [], "")
            self.up = o2.Linear((frame.irreps_out * 2).regroup(), gate.irreps_in)
            self.down = o2.Linear(gate.irreps_out, frame.irreps_out)
            self.conv = eqx_conv.UvO2TensorProductConv(
                frame,
                self.up,
                gate,
                self.down,
                frame,
                backend=backend,
                radial_mode="full",
            )
            self.radial = torch.nn.Sequential(
                torch.nn.Linear(3, 2),
                torch.nn.SiLU(),
                torch.nn.Linear(2, self.conv.weight_numel),
            )

    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    module = ScalarConvolution("cuda").double()
    module.conv.backend = "torch"
    reference = ScalarConvolution("torch").double()
    reference.load_state_dict(module.state_dict())
    backends = {
        name: child.backend
        for name, child in module.named_modules()
        if hasattr(child, "backend")
    }
    names, parameters = zip(*module.named_parameters())

    def joint(layer, x, vectors, index, parameters):
        output = torch.func.functional_call(
            layer, dict(zip(names, parameters)), (x, vectors, index)
        )
        energy = output.square().sum() / 10
        force = -torch.autograd.grad(energy, vectors, create_graph=True)[0]
        loss = energy + (force - 0.1).square().sum() / 10
        return energy, force, *torch.autograd.grad(loss, parameters)

    def actual(x, vectors, index, *parameters):
        return joint(module, x, vectors, index, parameters)

    def expected(x, vectors, index, *parameters):
        return joint(reference, x, vectors, index, parameters)

    def inputs(edges):
        return (
            (torch.randn(3, module.conv.input_dim) / 5).requires_grad_(),
            (torch.randn(edges, 3) / 5 + 1).requires_grad_(),
            torch.stack((torch.arange(edges) % 3, (torch.arange(edges) + 1) % 3)),
        )

    values = (*inputs(7), *parameters)
    probe = actual(*values)
    assert torch.count_nonzero(probe[1]) > 0
    assert all(torch.count_nonzero(gradient) > 0 for gradient in probe[2:])
    edges = ShapeVar("edges", min=2, optimization_hint=7)
    shapes = ShapesSpec(
        ParamsSpec(
            {
                "x": TensorSpec([None, None]),
                "vectors": TensorSpec([edges, None]),
                "index": TensorSpec([None, edges]),
            }
        )
    )
    graph = make_fx(
        actual,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=shapes,
    )(*values)
    oracle = make_fx(
        expected,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=shapes,
    )(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    assert not any("eqx." in str(node.target) for node in graph.graph.nodes)
    for count in (0, 1, 7, 11):
        batch = (*inputs(count), *parameters)
        torch.testing.assert_close(
            actual(*batch), expected(*batch), atol=2e-9, rtol=2e-9
        )
        if count >= 2:
            torch.testing.assert_close(
                graph(*batch), oracle(*batch), atol=2e-9, rtol=2e-9
            )
    dynamic_edges = torch.export.Dim("edges", min=2, max=16)
    arguments = tuple(x.detach() for x in values)
    dynamic_shapes = (
        None,
        {0: dynamic_edges},
        {1: dynamic_edges},
        *(None for _ in parameters),
    )
    exported = torch.export.export(
        graph,
        arguments,
        strict=True,
        dynamic_shapes=dynamic_shapes,
    )
    reference_program = torch.export.export(
        oracle,
        arguments,
        strict=True,
        dynamic_shapes=dynamic_shapes,
    )
    package = tmp_path / "reference.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    with zipfile.ZipFile(package) as archive:
        assert any(name.endswith(".so") for name in archive.namelist())
        assert not any(name.endswith((".cubin", ".ptx")) for name in archive.namelist())
        source = "\n".join(
            archive.read(name).decode()
            for name in archive.namelist()
            if name.endswith(".cpp")
        )
        assert source and "eqx::" not in source
    torch.export.save(reference_program, tmp_path / "oracle.pt2")
    torch.save(
        (
            tuple(p.detach() for p in parameters),
            [tuple(x.detach() for x in inputs(n)) for n in (7, 11, 5)],
            [tuple(x.detach() for x in inputs(n)) for n in (0, 1, 17)],
        ),
        tmp_path / "inputs.pt",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys, torch
torch.set_num_threads(2)
run = torch._inductor.aoti_load_package(sys.argv[1])
oracle = torch.export.load(sys.argv[2]).module()
weights, cases, invalid = torch.load(sys.argv[3], weights_only=True)
expected_weights = tuple(p.clone() for p in weights)
updates = 0
for inputs in cases:
    for _ in range(2):
        actual = run(*inputs, *weights)
        expected = oracle(*inputs, *expected_weights)
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        previous = weights
        if inputs[1].numel():
            assert torch.count_nonzero(actual[1]) > 0
            assert all(torch.count_nonzero(gradient) > 0 for gradient in actual[2:])
        weights = tuple(p - 0.001 * g for p, g in zip(weights, actual[2:]))
        if inputs[1].numel():
            assert any(not torch.equal(p, q) for p, q in zip(previous, weights))
            updates += 1
        expected_weights = tuple(p - 0.001 * g for p, g in zip(expected_weights, expected[2:]))
        torch.testing.assert_close(weights, expected_weights, atol=2e-9, rtol=2e-9)
for inputs in invalid:
    try:
        run(*inputs, *weights)
    except RuntimeError as error:
        assert ('too small' if inputs[1].shape[0] < 2 else 'too large') in str(error)
    else:
        raise AssertionError('The exported edge bounds were not enforced')
assert 'equivariantx' not in sys.modules and 'e3nn' not in sys.modules
print(f'{len(cases) * 2} steps checked, {updates} nonzero force/parameter updates')
""",
            str(package),
            str(tmp_path / "oracle.pt2"),
            str(tmp_path / "inputs.pt"),
        ],
        cwd=tmp_path,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "AOTI_RUNTIME_CHECK_INPUTS": "1",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    (tmp_path / "independent.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert backends == {
        name: child.backend
        for name, child in module.named_modules()
        if hasattr(child, "backend")
    }
    assert config.TORCH_REFERENCE is True
