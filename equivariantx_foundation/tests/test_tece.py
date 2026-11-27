import pytest
import torch


def _rotary_model():
    from equivariantx_foundation.tests.test_tace_plan import make_model
    from tace.models.spherical.tace import SphericalTACE

    def build(**cfg):
        cfg.update(parity=False, num_channel=4, mmax=1, lmax=1, Lmax=1)
        cfg["atomic_basis"].update(type=["cgtp", "so2"], num_head=2, edge_ace_hidden=3)
        cfg["edge_update"]["type"] = "element2"
        return SphericalTACE(**cfg)

    return make_model(build)


def test_rotary_rules_preserve_host_and_unsupported_activations(monkeypatch):
    from equivariantx_foundation.models.tace.tece_oam_rra.regional import rotary_rule

    monkeypatch.setenv("TACE_USE_EQX", "0")
    block = _rotary_model().readout_fn.representation.interactions[1]

    def reason():
        lower, detail = rotary_rule(block, backend="cuda", task="training")
        assert lower is None
        return detail

    assert "FP32 or FP64" in reason()
    assert block.rejector._eqx_metadata is None
    with monkeypatch.context() as mp:
        mp.setenv("TACE_MEMORY_FIRST", "0")
        assert "speed-first" in reason()
    with block.edge_info.register_forward_hook(lambda *args: None):
        assert "customized" in reason()
    with monkeypatch.context() as mp:
        # A name match is insufficient to inline an unknown activation.
        mp.setattr(block.rejector, "scalar_act", type("SiLU", (torch.nn.ReLU,), {})())
        assert "unsupported" in reason()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="regional rotary CUDA stream")
def test_regional_rotary_preserves_frames_and_force_training(
    monkeypatch, double_precision
):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace.tece_oam_rra.regional import (
        StreamedRotaryInteraction,
    )
    from equivariantx_foundation.tests.test_tace_plan import EFS, efs_loss, graph

    monkeypatch.setenv("TACE_USE_EQX", "0")
    host = _rotary_model().cuda().double().train()
    original = host.readout_fn.representation.interactions[1].rejector
    original.prepare_eqx()
    original.scalar_act = torch.nn.Tanh()
    fast = optimize(host, family="tace", backend="cuda", regional_only=True)
    rep = fast.readout_fn.representation
    assert type(rep.interactions[1]) is StreamedRotaryInteraction
    assert (
        not rep.use_packed_wigner
        and not host.readout_fn.representation.use_packed_wigner
    )
    assert tuple(fast.state_dict()) == tuple(host.state_dict())
    assert optimize(fast, family="tace", backend="cuda", regional_only=True) is fast
    edge_index = graph("cuda")["edge_index"]
    features = torch.ones(1, device="cuda")
    route = rep.interactions[1]._rotary_radial
    parts = ((features, None), (features, edge_index[1]), (features, edge_index[0]))
    assert route(features, parts, edge_index) is not None
    assert route(features, (parts[0], parts[2], parts[1]), edge_index) is None
    assert route(features, (*parts, parts[-1]), edge_index) is None

    def eval_step(model):
        out = model(graph("cuda"))
        grads = torch.autograd.grad(
            efs_loss(out), tuple(host.parameters()), allow_unused=True
        )
        return tuple(out[k] for k in EFS), grads

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        actual = eval_step(fast)
    calls = {e.key: e.count for e in prof.key_averages()}
    assert calls["eqx_foundation::tece_interaction"] == 1
    assert calls["eqx_foundation::tece_contraction"] > 0
    torch.testing.assert_close(actual, eval_step(host), rtol=1e-7, atol=5e-8)
    restored = optimize(fast, family="tace", backend="torch", regional_only=True)
    assert type(restored.readout_fn.representation.interactions[1]) is type(
        host.readout_fn.representation.interactions[1]
    )


@pytest.mark.parametrize(
    "device,provider",
    [("cpu", "native"), ("cuda", "native"), ("cuda", "eqx"), ("cuda", "regional")],
)
def test_rotary_owned_receiver_messages(
    monkeypatch, double_precision, device, provider
):
    """Ghosts supply keys/values; softmax and scatter normalize only owner queries."""
    from types import SimpleNamespace

    from equivariantx_foundation.models.tace.foundation import shared_parameter_copy
    from equivariantx_foundation.models.tace.tece_oam_rra.regional import rotary_rule

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("TACE_USE_EQX", "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    rep = _rotary_model().double().to(device).readout_fn.representation
    layer = rep.interactions[1]
    n, owners = 6, 3
    idx = torch.tensor([[5, 3, 4, 5, 1, 0, 3], [1, 1, 0, 1, 0, 1, 0]], device=device)
    c = rep.edge_updates[1].source_embedding.embedding_dim

    def rand(*dims):
        return torch.randn(
            *dims, dtype=torch.float64, device=device, requires_grad=True
        )

    x, src, dst = rand(n, layer.irreps_in.dim), rand(n, c), rand(n, c)
    e = idx.shape[1]
    edge = rand(e, layer.edge_info.mlp[0].in_dim - 2 * c)
    vec, basis, cutoff = rand(e, 3), rand(e, layer.num_radial_basis), rand(e, 1)
    hidden = ((edge, None), (dst, idx[1]), (src, idx[0]))
    inp = (
        x,
        vec,
        basis,
        cutoff,
        edge,
        src,
        dst,
        *layer.edge_info.parameters(),
        *layer.rejector.parameters(),
    )
    w, wi = rep.o2_angular_basis(vec)

    def run(block, count):
        graph = SimpleNamespace(
            lmp_natoms=(owners, n - owners),
            lmp_data=object() if count is not None else None,
        )
        y = block._compute_messages(
            x,
            None,
            basis,
            hidden,
            None,
            idx,
            cutoff.sigmoid(),
            w.contiguous(),
            wi.contiguous(),
            graph=graph,
        )
        assert y.shape[0] == (owners if count is not None else n)
        assert y.untyped_storage().nbytes() == y.numel() * y.element_size()
        y = y[:owners]
        assert torch.count_nonzero(y[2]) == 0
        first = torch.autograd.grad(
            y.sin().sum(), inp, create_graph=True, retain_graph=True, allow_unused=True
        )
        assert first[0][5].abs().sum() > 0
        second = torch.autograd.grad(
            first[1].square().sum(), inp, retain_graph=True, allow_unused=True
        )
        return y, first, second

    oracle = run(layer, None)
    if provider == "regional":
        lower, why = rotary_rule(layer, backend="cuda", task="inference")
        assert lower is not None, why
        layer = lower(shared_parameter_copy(layer))
    elif provider == "eqx":
        monkeypatch.setenv("TACE_USE_EQX", "1")
        assert layer.use_eqx
    torch.testing.assert_close(run(layer, owners), oracle, atol=2e-8, rtol=2e-8)
    empty = idx[:, :0]
    for count in (0, owners):
        graph = SimpleNamespace(lmp_natoms=(count, n - count), lmp_data=object())
        # Dense radial inputs also admit empty graphs; they need no dummy edge.
        output = layer._compute_messages(
            x,
            None,
            basis[:0],
            rand(0, layer.edge_info.mlp[0].in_dim),
            None,
            empty,
            cutoff[:0],
            w[:0],
            wi[:0],
            graph=graph,
        )
        assert output.shape == (count, oracle[0].shape[1])
        assert output.count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rotary row maps")
def test_exported_rotary_arbitrary_rows(monkeypatch, double_precision):
    """Changing a feature-table row must not change the physical endpoints."""
    from equivariantx_foundation import optimize

    monkeypatch.setenv("TACE_USE_EQX", "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    native = _rotary_model().cuda().train()
    optimized = optimize(
        native, family="tace", backend="cuda", task="training", regional_only=True
    )
    rep = native.readout_fn.representation

    class Messages(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x, edge, a, b, index, rows_a, rows_b, basis, cutoff, w, winv):
            return self.layer._compute_messages(
                x,
                None,
                basis,
                ((edge, None), (a, rows_a), (b, rows_b)),
                None,
                index,
                cutoff,
                w,
                winv,
            )

    layer = rep.interactions[1]
    eager = Messages(layer)
    streamed = Messages(optimized.readout_fn.representation.interactions[1])

    def sample(e):
        n, c = 5, rep.edge_updates[1].source_embedding.embedding_dim

        def rand(*s):
            return torch.randn(s, device="cuda")

        index = torch.randint(n, (2, e), device="cuda")
        w, winv = rep.o2_angular_basis(rand(e, 3))
        return (
            rand(n, layer.irreps_in.dim),
            rand(e, layer.edge_info.mlp[0].in_dim - 2 * c),
            rand(7, c),
            rand(8, c),
            index,
            index[1],
            index[0],
            rand(e, layer.num_radial_basis),
            rand(e, 1).sigmoid(),
            w.contiguous(),
            winv.contiguous(),
        )

    edges = torch.export.Dim("edges", min=2, max=32)
    dims = tuple(
        ({0: edges} if i in (1, 5, 6, 7, 8, 9, 10) else {1: edges} if i == 4 else {})
        for i in range(11)
    )
    ep = torch.export.export(streamed, sample(7), dynamic_shapes=dims, strict=True)
    calls = [n for n in ep.graph.nodes if n.op == "call_function"]
    from equivariantx.utils.metadata import parse_metadata

    (stream,) = [
        n
        for n in calls
        if n.target == torch.ops.eqx_foundation.tece_interaction.default
    ]
    _, base = parse_metadata(stream.args[0])
    tables = [
        (stream.args[-1][slot], stream.args[-1][kind[1]])
        for op, _, _, data in base[0]
        if op == "input"
        for slot, kind in [data]
        if isinstance(kind, tuple) and kind[0] == "indexed"
    ]
    assert [(v.name, rows.name) for v, rows in tables] == [
        ("a", "rows_a"),
        ("b", "rows_b"),
    ]
    # Each row input goes directly to the streamed operation. The remaining
    # index_selects permute angular columns and must not be mistaken for row gathers.
    assert all(set(rows.users) == {stream} for _, rows in tables)

    for e in (7, 11):
        args = list(sample(e))
        # Independently changed maps use duplicate rows beyond the physical
        # node range, proving that the recorded endpoint views were not frozen.
        args[5] = torch.arange(e, device="cuda") % 2 + 5
        args[6] = (torch.arange(e, device="cuda") % 3 + 5).flip(0)
        args = [v.detach().requires_grad_(v.is_floating_point()) for v in args]
        checked = [v for v in args if v.requires_grad] + list(
            layer.edge_info.parameters()
        )

        def terms(m):
            out = m(*args)
            dx = torch.autograd.grad(out.square().mean(), args[0], create_graph=True)[0]
            grads = torch.autograd.grad(
                dx.square().sum() + out.square().mean(), checked
            )
            return tuple(v.detach() for v in (out, dx, *grads))

        expected = terms(eager)
        for route in (streamed, ep.module()):
            torch.testing.assert_close(terms(route), expected, atol=1.0e-8, rtol=2.0e-7)


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("edges,owners", [(0, None), (9, None), (0, 4), (9, 4), (0, 0)])
def test_rotary_interaction_providers(
    edges, indexed, owners, double_precision, monkeypatch
):
    from equivariantx.ops import config, get_kernel
    from equivariantx.utils.program import Program
    from equivariantx_foundation.models.tace.tece_oam_rra.expressions import (
        first_adjoint,
    )
    from equivariantx_foundation.ops import tece

    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = Program()
    specs = (
        (4, "source"),
        (2, "edge"),
        (4, "shared"),
        (4, "shared"),
        (1, "edge"),
        (1, "edge"),
        (1, "edge"),
    )
    x, score, scale, *_ = [
        p.input(i, kind, width) for i, (width, kind) in enumerate(specs)
    ]
    if indexed:
        # The row map is an operand, not an expression leaf. Its table need
        # not have one row per node, and repeated rows must accumulate gradients.
        kind = ("indexed", 8, 7)
        score = p.binary("add", score, p.input(7, kind, 2))
        specs += ((2, kind), (1, "row_map"))
    receiver = 0 if owners is None else len(specs)
    if owners is not None:
        specs += ((4, "target"),)
    v = p.binary("add", p.binary("mul", x, scale), p.input(receiver, "target", 4))
    base = (tuple(p.nodes), score, v, specs, 2, 4, 1e-6)
    description = (repr((0, 0, 4, 2)), base)
    meta = repr(description if owners is None else (*description, receiver))
    inputs = [
        torch.randn(*shape, device=device, dtype=torch.float64).requires_grad_()
        for shape in (
            (5, 1, 4),
            (edges, 2),
            (2, 2),
            (4,),
            (edges, 1),
            (edges, 1),
            (edges, 1),
        )
    ]
    inputs[4] = inputs[4].sigmoid()
    src = torch.arange(edges, device=device) % 5
    dst = (src + 1) % 3  # receiver 3 is isolated; sender 4 is a ghost
    if indexed:
        inputs.extend([torch.randn(3, 2, device=device, requires_grad=True), src % 3])
    if owners is not None:
        inputs.append(inputs[0].narrow(0, 0, owners))
    results = []
    for ordinary in (True, False):
        with monkeypatch.context() as switch:
            switch.setattr(config, "TORCH_REFERENCE", ordinary)
            results.append(
                get_kernel("tece_interaction", "torch")(meta, src, dst, inputs)
            )
    if device == "cuda":
        results.append(tece.interaction(meta, src, dst, inputs))
    for message, denominator, maximum in results:
        count = 5 if owners is None else owners
        assert message.shape == (count, 1, 4)
        assert denominator.shape == maximum.shape == (count, 2)
        if owners is not None and edges:
            (dx,) = torch.autograd.grad(
                message.square().sum(), inputs[0], retain_graph=True
            )
            assert dx[4].abs().sum() > 0 and message[3].count_nonzero() == 0
    selected = (0, 1, 2, 4, 7) if indexed else (0, 1, 2, 4)
    for result in results[1:]:
        for a, b in zip(result, results[0]):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
        if edges:
            actual, expected = [
                v[0].sin().sum() + v[1].square().sum() for v in (result, results[0])
            ]
            for _ in range(3):
                actual, expected = [
                    torch.cat(
                        [
                            g.flatten()
                            for g in torch.autograd.grad(
                                v.sin().sum(),
                                [inputs[i] for i in selected],
                                create_graph=True,
                                retain_graph=True,
                            )
                        ]
                    )
                    for v in (actual, expected)
                ]
                torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    expanded = [
        *inputs,
        *results[0],
        torch.randn_like(results[0][0]),
        torch.randn_like(results[0][1]),
    ]
    adjoint = first_adjoint(base, selected)
    contractions = []
    for ordinary in (True, False):
        with monkeypatch.context() as switch:
            switch.setattr(config, "TORCH_REFERENCE", ordinary)
            contractions.append(
                get_kernel("tece_contraction", "torch")(adjoint, src, dst, expanded)
            )
    if device == "cuda":
        contractions.append(tece.contraction(adjoint, src, dst, expanded))
    for result in contractions[1:]:
        torch.testing.assert_close(result, contractions[0], rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rotary AOTInductor")
def test_rotary_owned_dynamic_joint_aoti(tmp_path, double_precision, monkeypatch):
    """One compiled attention/force joint for independent source, owner and edge axes."""
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops.compiler import compile_and_package
    from equivariantx_foundation.models.tace.tece_oam_rra.interaction import stream

    monkeypatch.setenv("TACE_USE_EQX", "0")
    rep = _rotary_model().cuda().eval().readout_fn.representation
    layer = rep.interactions[1]
    c = layer.rejector
    c.prepare_eqx()
    assert c._eqx_metadata is not None
    hidden = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.SiLU()).cuda()
    projection = torch.randn(4, c.weight_numel, device="cuda") * 0.1
    bias = torch.randn(c.weight_numel, device="cuda") * 0.1

    def run(compact, x, owned, v, idx):
        radius = v.square().sum(-1, keepdim=True)
        cut = (-radius / 3).exp()
        basis = radius.expand(-1, layer.num_radial_basis)
        w, wi = rep.o2_angular_basis(v)
        if compact:
            radial = ((radius, "edge"), (x[:, :1], "source"), (x[:, 1:2], "target"))
            y = stream(
                c,
                x,
                radial,
                projection,
                bias,
                idx,
                cut,
                w,
                wi,
                basis,
                radial_network=hidden,
                backend="cuda",
                num_nodes=owned.shape[0],
            )
        else:
            terms = torch.cat((radius, x[idx[0], :1], x[idx[1], 1:2]), 1)
            y = c(x, hidden(terms) @ projection + bias, idx, cut, w, wi, basis)[
                : owned.shape[0]
            ]
        dx, dv = torch.autograd.grad(y.square().sum(), (x, v), create_graph=True)
        (mixed,) = torch.autograd.grad(dv.square().sum(), (x,))
        return y, dx, -dv, mixed

    def joint(x, owned, v, idx):
        return run(True, x, owned, v, idx)

    def inputs(n, r, e):
        ids = torch.arange(e, device="cuda")
        return (
            torch.randn(n, layer.irreps_in.dim, device="cuda").requires_grad_(),
            torch.empty(r, device="cuda"),
            torch.randn(e, 3, device="cuda").requires_grad_(),
            torch.stack(((ids + n - 1) % n, ids % r)),
        )

    # Equality in the example must not specialize away the separate target view.
    values = inputs(7, 7, 11)
    dimensions = (("senders", 7, 15, 7), ("owners", 2, 7, 7), ("edges", 8, 30, 11))
    n, r, e = [
        ShapeVar(name, min=lo, max=hi, optimization_hint=hint)
        for name, lo, hi, hint in dimensions
    ]
    spec = ShapesSpec(
        ParamsSpec(
            {
                "x": TensorSpec([n, None]),
                "owned": TensorSpec([r]),
                "v": TensorSpec([e, None]),
                "idx": TensorSpec([None, e]),
            }
        )
    )
    graph = make_fx(
        joint, tracing_mode="fake", _allow_non_fake_inputs=True, dynamic_shapes=spec
    )(*values)
    ops = {node.target for node in graph.graph.nodes}
    assert torch.ops.eqx_foundation.tece_interaction.default in ops
    assert torch.ops.eqx_foundation.tece_contraction.default in ops
    n, r, e = [torch.export.Dim(name, min=lo, max=hi) for name, lo, hi, _ in dimensions]
    ep = torch.export.export(
        graph,
        tuple(z.detach() for z in values),
        strict=True,
        dynamic_shapes=({0: n}, {0: r}, {0: e}, {1: e}),
    )
    package = compile_and_package(ep, package_path=str(tmp_path / "rotary-owned.pt2"))
    runner = torch._inductor.aoti_load_package(package)
    for case in (values, inputs(12, 4, 19)):
        expected = run(False, *case)
        actual = runner(*(z.detach() for z in case))
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)
        if case[0].shape[0] != case[1].shape[0]:
            assert expected[1][case[1].shape[0] :].count_nonzero()
