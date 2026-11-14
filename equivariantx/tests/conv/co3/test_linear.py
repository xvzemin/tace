"""Tests for the corresponding EQX source module."""

import os
import pickle
import subprocess
import sys

import pytest
import torch

from equivariantx import conv as eqx_conv
from equivariantx.ops import config


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("ordinary", [False, True])
@pytest.mark.parametrize("nodes", [0, 3])
def test_empty_output_broadcast(backend, ordinary, nodes, device, monkeypatch):
    from equivariantx.conv.co3 import Linear
    from equivariantx.ops.torch.conv.co3.linear import reference

    monkeypatch.setattr(config, "TORCH_REFERENCE", ordinary)
    module = Linear(
        "1x1o",
        "",
        [0, 1, 2],
        backend=backend,
        shared_weights=False,
        internal_weights=False,
    ).to(device=device, dtype=torch.float64)
    values = tuple(
        torch.randn(*shape, device=device, dtype=torch.float64).requires_grad_()
        for shape in ((nodes, 3), (2, 1, 0), (1, 1, 0))
    )
    for operation in (module, lambda *args: reference(module, *args)):
        output = operation(*values)
        assert output.shape == (2, nodes, 0)
        gradients = torch.autograd.grad(output.sum(), values)
        for gradient, value in zip(gradients, values):
            torch.testing.assert_close(
                gradient, torch.zeros_like(value), atol=0, rtol=0
            )
        _, tangent = torch.func.jvp(
            operation, values, tuple(torch.ones_like(value) for value in values)
        )
        assert tangent.shape == output.shape


def partial_linear(
    channels=128,
    channels_out=None,
    rank=3,
    output_basis="spherical",
    shared=True,
    mixed=False,
):
    from equivariantx.co3.symmetric import symmetric_basis
    from equivariantx.conv.co3.linear import Linear

    channels_out = channels if channels_out is None else channels_out
    degrees = range(4) if mixed else (rank,)
    irreps_in = "+".join(f"{channels}x{l}{'o' if l % 2 else 'e'}" for l in degrees)
    irreps_out = "+".join(f"{channels_out}x{l}{'o' if l % 2 else 'e'}" for l in degrees)
    if not mixed:
        irreps_out += "+1x0e"
    indices = []
    offset = 0
    for degree in degrees:
        base = (
            (symmetric_basis(2)["indices"][:, None] * 3 + torch.arange(3)).flatten()
            if degree == 3
            else torch.arange(3**degree)
        )
        indices.append(
            (
                offset + base[None, :] * channels + torch.arange(channels)[:, None]
            ).flatten()
        )
        offset += (int(base.max()) + 1) * channels
    return Linear(
        irreps_in,
        irreps_out,
        torch.cat(indices),
        internal_weights=False,
        shared_weights=shared,
        biases=True,
        output_basis=output_basis,
    ).double()


@pytest.mark.parametrize("channels", [64, 128, 192, 256, 512, 1024])
def test_compact_basis_first_channels(channels, monkeypatch):
    from equivariantx.ops.torch.conv.co3 import linear as kernel

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_COMPACT_LINEAR_BASIS_FIRST", True)
    module = partial_linear(channels)
    values = (
        torch.randn(3, module.input_dim, dtype=torch.float64),
        torch.randn(module.weight_numel, dtype=torch.float64),
        torch.randn(module.bias_numel, dtype=torch.float64),
    )
    assert kernel.basis_first_available(module, *values[:2], 0) == (
        channels in (128, 256, 512, 1024)
    )
    torch.testing.assert_close(
        module(*values), kernel.reference(module, *values), atol=2e-12, rtol=2e-12
    )
    monkeypatch.setattr(config, "TORCH_COMPACT_LINEAR_BASIS_FIRST", False)
    assert not kernel.basis_first_available(module, *values[:2], 0)
    torch.testing.assert_close(
        module(*values), kernel.reference(module, *values), atol=2e-12, rtol=2e-12
    )


@pytest.mark.parametrize(
    "options",
    [
        {"rank": 2},
        {"shared": False},
        {"channels_out": 64},
        {"output_basis": "cartesian"},
    ],
)
def test_compact_basis_first_preserves_other_shapes(options, monkeypatch):
    from equivariantx.ops.torch.conv.co3 import linear as kernel

    monkeypatch.setattr(config, "TORCH_COMPACT_LINEAR_BASIS_FIRST", True)
    module = partial_linear(**options)
    prefix = () if module.shared_weights else (3,)
    values = (
        torch.randn(3, module.input_dim, dtype=torch.float64),
        torch.randn(*prefix, module.weight_numel, dtype=torch.float64),
        torch.randn(*prefix, module.bias_numel, dtype=torch.float64),
    )
    assert not kernel.basis_first_available(module, *values[:2], 0)
    torch.testing.assert_close(
        module(*values), kernel.reference(module, *values), atol=2e-12, rtol=2e-12
    )


@pytest.mark.parametrize("layout", ["permuted", "leading_dimensions"])
def test_compact_basis_first_preserves_other_layouts(layout, monkeypatch):
    from equivariantx.conv.co3.linear import Linear
    from equivariantx.ops.torch.conv.co3 import linear as kernel

    monkeypatch.setattr(config, "TORCH_COMPACT_LINEAR_BASIS_FIRST", True)
    module = partial_linear()
    if layout == "permuted":
        module = Linear(
            module.irreps_in,
            module.irreps_out,
            (module.input_expansion + 1) % module.input_dim,
            internal_weights=False,
            biases=True,
        ).double()
    shape = (2, 3) if layout == "leading_dimensions" else (3,)
    values = (
        torch.randn(*shape, module.input_dim, dtype=torch.float64),
        torch.randn(module.weight_numel, dtype=torch.float64),
        torch.randn(module.bias_numel, dtype=torch.float64),
    )
    assert not kernel.basis_first_available(module, *values[:2], 0)
    torch.testing.assert_close(
        module(*values), kernel.reference(module, *values), atol=2e-12, rtol=2e-12
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_compact_basis_first_high_order(mixed, monkeypatch):
    from equivariantx.ops.torch.conv.co3 import linear as kernel

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_COMPACT_LINEAR_BASIS_FIRST", True)
    module = partial_linear(mixed=mixed)
    values = tuple(
        (torch.randn(*shape[:-1], 2 * shape[-1], dtype=torch.float64) * 0.1)[
            ..., ::2
        ].requires_grad_()
        for shape in (
            (3, module.input_dim),
            (module.weight_numel,),
            (module.bias_numel,),
        )
    )
    choices = [
        kernel.basis_first_available(module, *values[:2], i)
        for i in range(len(module.compact_groups))
    ]
    assert choices == ([False, False, False, True] if mixed else [True])
    directions = tuple(torch.randn_like(value) * 0.1 for value in values)
    torch.testing.assert_close(
        torch.func.jvp(module, values, directions),
        torch.func.jvp(
            lambda *args: kernel.reference(module, *args), values, directions
        ),
        atol=2e-12,
        rtol=2e-12,
    )
    losses = []
    for output in (module(*values), kernel.reference(module, *values)):
        energy = output.sin().square().mean()
        force = -torch.autograd.grad(energy, values[0], create_graph=True)[0]
        losses.append(energy.square() + force.square().sum())
    for _ in range(5):
        gradients = [
            torch.autograd.grad(loss, values, create_graph=True) for loss in losses
        ]
        torch.testing.assert_close(*gradients, atol=3e-11, rtol=3e-9)
        losses = [
            sum((g * d).sum() for g, d in zip(result, directions))
            for result in gradients
        ]
    for count in (0, 1, 7):
        x = values[0].new_empty(count, module.input_dim).normal_()
        torch.testing.assert_close(
            module(x, *values[1:]),
            kernel.reference(module, x, *values[1:]),
            atol=2e-12,
            rtol=2e-12,
        )


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_compact_basis_first_preserves_cuda(backend, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.torch.conv.co3 import linear as kernel

    monkeypatch.setattr(config, "TORCH_COMPACT_LINEAR_BASIS_FIRST", True)
    module = partial_linear().cuda()
    module.backend = backend
    values = (
        torch.randn(3, module.input_dim, dtype=torch.float64, device="cuda"),
        torch.randn(module.weight_numel, dtype=torch.float64, device="cuda"),
        torch.randn(module.bias_numel, dtype=torch.float64, device="cuda"),
    )
    assert not kernel.basis_first_available(module, *values[:2], 0)
    torch.testing.assert_close(
        module(*values), kernel.reference(module, *values), atol=2e-12, rtol=2e-12
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("output_basis", ["spherical", "cartesian"])
def test_compact_bias_broadcast(device, backend, output_basis, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3
    from equivariantx.ops.torch.conv.co3 import linear as kernel

    module = partial_linear(mixed=True, shared=False, output_basis=output_basis).to(
        device
    )
    module.backend = backend
    full = (
        co3.Linear(
            module.irreps_in,
            module.irreps_out,
            internal_weights=False,
            shared_weights=False,
            biases=True,
            output_basis=output_basis,
            project=True,
        )
        .double()
        .to(device)
    )
    for count, prefix in ((3, (2, 1)), (1, (5,)), (0, (2, 1))):
        values = tuple(
            (
                torch.randn(*shape, dtype=torch.float64, device=device) * 0.1
            ).requires_grad_()
            for shape in (
                (count, module.input_dim),
                (module.weight_numel,),
                (*prefix, module.bias_numel),
            )
        )
        monkeypatch.setattr(config, "TORCH_REFERENCE", False)
        actual = module(*values)
        monkeypatch.setattr(config, "TORCH_REFERENCE", True)
        expected = full(values[0][..., module.input_expansion], *values[1:])
        reference = kernel.reference(module, *values)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(reference, expected, atol=2e-12, rtol=2e-12)
        losses = []
        for output in (actual, reference, expected):
            energy = output.sin().square().sum() / max(1, output.numel())
            force = -torch.autograd.grad(energy, values[0], create_graph=True)[0]
            losses.append(energy.square() + force.square().sum())
        directions = tuple(torch.randn_like(value) * 0.1 for value in values)
        for _ in range(5):
            gradients = [
                torch.autograd.grad(loss, values, create_graph=True) for loss in losses
            ]
            for result in gradients[:2]:
                torch.testing.assert_close(result, gradients[2], atol=3e-11, rtol=3e-9)
            losses = [
                sum((g * d).sum() for g, d in zip(result, directions))
                for result in gradients
            ]


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_compact_basis_first_fullgraph(reference, mixed, monkeypatch):
    from equivariantx.ops.torch.conv.co3 import linear as kernel

    monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
    monkeypatch.setattr(config, "TORCH_COMPACT_LINEAR_BASIS_FIRST", True)
    module = partial_linear(mixed=mixed, shared=False)
    graphs = []

    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward

    run = torch.compile(module, backend=capture, fullgraph=True, dynamic=True)
    weight = torch.randn(module.weight_numel, dtype=torch.float64)
    bias = torch.randn(2, 1, module.bias_numel, dtype=torch.float64)
    for count in (7, 11, 5):
        x = torch.randn(count, module.input_dim, dtype=torch.float64)
        torch.testing.assert_close(
            run(x, weight, bias),
            kernel.reference(module, x, weight, bias),
            atol=2e-12,
            rtol=2e-12,
        )
    assert len(graphs) == 1


def test_compact_basis_first_dynamic_fx_and_aoti_contract(monkeypatch, tmp_path):
    pytest.importorskip("torch.fx.experimental.dynamic_spec")
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops.torch.conv.co3 import linear as kernel

    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    monkeypatch.setattr(config, "TORCH_COMPACT_LINEAR_BASIS_FIRST", True)
    module = partial_linear(mixed=True, shared=False)

    def operation(x, weight, bias, reference=False):
        y = (
            kernel.reference(module, x, weight, bias)
            if reference
            else module(x, weight, bias)
        )
        energy = y.sin().square().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = 0.1 * energy.square() + (force - x.detach().cos() * 0.003).square().sum()
        return y, force, loss, *torch.autograd.grad(loss, (weight, bias))

    def captured(x, weight, bias):
        return operation(x, weight, bias)

    parameters = [
        (torch.randn(*shape, dtype=torch.float64) * 0.1).requires_grad_()
        for shape in ((module.weight_numel,), (2, 1, module.bias_numel))
    ]
    x = (torch.randn(7, module.input_dim, dtype=torch.float64) * 0.2).requires_grad_()
    rows = ShapeVar("rows", min=0, max=16)
    graph = make_fx(
        captured,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=ShapesSpec(ParamsSpec({"x": TensorSpec((rows, None))})),
    )(x, *parameters)
    for count in (0, 1):
        sample = (
            torch.randn(count, module.input_dim, dtype=torch.float64) * 0.2
        ).requires_grad_()
        torch.testing.assert_close(
            graph(sample, *parameters),
            operation(sample, *parameters, reference=True),
            atol=3e-10,
            rtol=3e-9,
        )
    exported = torch.export.export(
        graph,
        tuple(value.detach() for value in (x, *parameters)),
        strict=True,
        dynamic_shapes=({0: torch.export.Dim("rows", min=2, max=16)}, None, None),
    )
    package = tmp_path / "basis-first.pt2"
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(package),
        inductor_configs={
            "aot_inductor.check_lowerbound": True,
            "aot_inductor.check_upperbound": True,
        },
    )
    initial = tuple(value.detach().clone() for value in parameters)
    records = []
    for count in (7, 11, 5):
        x = (
            torch.randn(count, module.input_dim, dtype=torch.float64) * 0.2
        ).requires_grad_()
        expected = operation(x, *parameters, reference=True)
        records.append(
            {
                "x": x.detach(),
                "outputs": tuple(value.detach().clone() for value in expected),
            }
        )
        for value, gradient in zip(parameters, expected[-2:]):
            value.grad = gradient
        assert torch.nn.utils.clip_grad_norm_(parameters, 0.05, foreach=True) > 0.05
        with torch.no_grad():
            for value in parameters:
                value.add_(value.grad, alpha=-0.1)
    assert all(
        not torch.equal(before, after) for before, after in zip(initial, parameters)
    )
    payload = tmp_path / "states.pt"
    torch.save(
        {
            "initial": initial,
            "steps": records,
            "final": tuple(v.detach() for v in parameters),
        },
        payload,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
torch.set_num_threads(2)
run = torch._inductor.aoti_load_package(sys.argv[1])
record = torch.load(sys.argv[2], weights_only=True)
parameters = list(record['initial'])
for step in record['steps']:
    actual = run(step['x'], *parameters)
    torch.testing.assert_close(tuple(actual), step['outputs'], atol=3e-10, rtol=3e-9)
    intervals = [(g.data_ptr(), g.data_ptr() + g.element_size() * g.numel()) for g in actual[-2:]]
    assert intervals[0][1] <= intervals[1][0] or intervals[1][1] <= intervals[0][0]
    for value, gradient in zip(parameters, actual[-2:]):
        value.grad = gradient
    assert torch.nn.utils.clip_grad_norm_(parameters, 0.05, foreach=True) > 0.05
    for value in parameters:
        value.add_(value.grad, alpha=-0.1)
torch.testing.assert_close(tuple(parameters), record['final'], atol=2e-12, rtol=2e-11)
for count in (0, 1, 17):
    try:
        run(step['x'].new_zeros(count, step['x'].shape[1]), *parameters)
    except RuntimeError as error:
        assert ('too small' if count < 2 else 'too large') in str(error)
        assert ('>= 2' if count < 2 else '<= 16') in str(error)
    else:
        raise AssertionError('Declared AOTI row bounds must remain enforced.')
assert 'equivariantx' not in sys.modules
""",
            str(package),
            str(payload),
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
    assert result.returncode == 0, result.stdout + result.stderr


def compact_reference_modules(output_basis, backend="torch", device="cpu", mixed=False):
    from equivariantx import co3
    from equivariantx.conv.co3.linear import Linear

    symmetric = torch.tensor([0, 1, 2, 1, 3, 4, 2, 4, 5])
    tail = torch.arange(9) if mixed else symmetric
    indices = torch.cat((symmetric, symmetric + 6, torch.arange(3) + 12, tail + 15))
    indices = torch.cat((indices, indices.new_tensor([int(indices.max()) + 1])))
    options = dict(
        internal_weights=False,
        shared_weights=False,
        instructions=[(0, 0), (1, 1), (2, 0), (3, 2), (0, 0)],
        biases=True,
        path_normalization="path",
        output_basis=output_basis,
    )
    irreps_in = "2x2e+1x1o+1x2e+1x0e"
    irreps_out = "3x2e+2x1o+2x0e+1x3o"
    compact = Linear(irreps_in, irreps_out, indices, backend=backend, **options)
    reference = co3.Linear(irreps_in, irreps_out, project=True, **options)
    return (
        compact.to(device=device, dtype=torch.float64),
        reference.to(device=device, dtype=torch.float64),
        indices.to(device),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("output_basis", ["spherical", "cartesian"])
@pytest.mark.parametrize("row_weights", [False, True])
def test_compact_reference_independent(
    device, backend, output_basis, row_weights, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module, full, indices = compact_reference_modules(output_basis, backend, device)
    prefix = (5,) if row_weights else ()
    shapes = (
        (5, module.input_dim),
        (*prefix, module.weight_numel),
        (*prefix, module.bias_numel),
    )
    values = tuple(
        (
            torch.randn(*shape[:-1], 2 * shape[-1], dtype=torch.float64, device=device)
            * 0.1
        )[..., ::2].requires_grad_()
        for shape in shapes
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", False)
    actual = module(*values)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = full(values[0][..., indices], *values[1:])
    reference = module(*values)
    if backend == "cuda" and device == "cuda":
        from equivariantx.ops.torch.conv.co3 import linear as torch_kernel

        def forbidden(*args):
            raise AssertionError("Explicit CUDA must not dispatch the Torch reference.")

        monkeypatch.setattr(torch_kernel, "reference", forbidden)
    else:
        # The ordinary path must not read optimized compaction metadata or matrices.
        monkeypatch.setattr(module, "compact_groups", None)
        monkeypatch.setattr(
            module, "basis_0", torch.full_like(module.basis_0, torch.nan)
        )
    torch.testing.assert_close(module(*values), expected, atol=2e-12, rtol=2e-12)
    losses = []
    for value in (actual, reference, expected):
        torch.testing.assert_close(value, expected, atol=2e-12, rtol=2e-12)
        energy = value.sin().square().mean()
        force = torch.autograd.grad(energy, values[0], create_graph=True)[0]
        losses.append(energy.square() + force.square().sum())
    directions = tuple(torch.randn_like(value) * 0.1 for value in values)
    for _ in range(5):
        gradients = [
            torch.autograd.grad(loss, values, create_graph=True) for loss in losses
        ]
        for result in gradients[:2]:
            torch.testing.assert_close(result, gradients[2], atol=3e-11, rtol=3e-9)
        losses = [
            sum((g * d).sum() for g, d in zip(result, directions))
            for result in gradients
        ]


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("output_basis", ["spherical", "cartesian"])
def test_compact_reference_legacy(output_basis, mixed, monkeypatch):
    module, full, indices = compact_reference_modules(output_basis, mixed=mixed)
    original = module.input_expansion.clone()
    del module._buffers["input_expansion"]
    for name in tuple(module._buffers):
        if name.startswith("reference_basis_"):
            del module._buffers[name]
    restored = pickle.loads(pickle.dumps(module)).float().double()
    torch.testing.assert_close(restored.input_expansion, original)
    values = (
        torch.randn(3, restored.input_dim, dtype=torch.float64),
        torch.randn(restored.weight_numel, dtype=torch.float64),
        torch.randn(restored.bias_numel, dtype=torch.float64),
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    torch.testing.assert_close(
        restored(*values),
        full(values[0][..., indices], *values[1:]),
        atol=2e-12,
        rtol=2e-12,
    )


def test_compact_reference_empty_and_broadcast(monkeypatch):
    module, full, indices = compact_reference_modules("spherical")
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (0, 1, 7):
        values = (
            torch.randn(2, 1, count, module.input_dim, dtype=torch.float64),
            torch.randn(1, 3, 1, module.weight_numel, dtype=torch.float64),
            torch.randn(3, 1, module.bias_numel, dtype=torch.float64),
        )
        torch.testing.assert_close(
            module(*values),
            full(values[0][..., indices], *values[1:]),
            atol=2e-12,
            rtol=2e-12,
        )


@pytest.mark.parametrize("output_basis", ["spherical", "cartesian"])
def test_compact_reference_dynamic_aoti(output_basis, monkeypatch, tmp_path):
    pytest.importorskip("torch.fx.experimental.dynamic_spec")
    from torch.fx.experimental.dynamic_spec import (
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    module, full, expansion = compact_reference_modules(output_basis)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)

    def operation(x, weight, bias, elements, reference=False):
        y = (
            full(x[..., expansion], weight[elements], bias[elements])
            if reference
            else module(x, weight[elements], bias[elements])
        )
        energy = y.sin().square().mean()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = energy.square() + force.square().sum()
        return y, force, loss, *torch.autograd.grad(loss, (weight, bias))

    def captured(x, weight, bias, elements):
        return operation(x, weight, bias, elements)

    parameters = [
        (torch.randn(2, size, dtype=torch.float64) * 0.1).requires_grad_()
        for size in (module.weight_numel, module.bias_numel)
    ]
    x = torch.randn(7, module.input_dim, dtype=torch.float64).requires_grad_()
    elements = torch.arange(7) % 2
    values = (x, *parameters, elements)
    tangents = tuple(torch.randn_like(v) * 0.1 for v in values[:3])
    torch.testing.assert_close(
        torch.func.jvp(
            lambda a, w, b: module(a, w[elements], b[elements]), values[:3], tangents
        ),
        torch.func.jvp(
            lambda a, w, b: full(a[..., expansion], w[elements], b[elements]),
            values[:3],
            tangents,
        ),
        atol=2e-12,
        rtol=2e-12,
    )
    rows = ShapeVar("rows", min=2)
    shapes = ShapesSpec(
        ParamsSpec({"x": TensorSpec((rows, None)), "elements": TensorSpec((rows,))})
    )
    graph = make_fx(
        captured,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        dynamic_shapes=shapes,
    )(*values)
    rows = torch.export.Dim("rows", min=2)
    exported = torch.export.export(
        graph,
        tuple(value.detach() for value in values),
        strict=True,
        dynamic_shapes=({0: rows}, None, None, {0: rows}),
    )
    package = tmp_path / "compact-linear.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    initial = tuple(value.detach().clone() for value in parameters)
    for count in (7, 11, 5):
        x = torch.randn(count, module.input_dim, dtype=torch.float64).requires_grad_()
        elements = torch.arange(count) % 2
        expected = operation(x, *parameters, elements, reference=True)
        records.append(
            {
                "x": x.detach(),
                "elements": elements,
                "outputs": tuple(value.detach().clone() for value in expected),
            }
        )
        for value, gradient in zip(parameters, expected[-2:]):
            value.grad = gradient
        torch.nn.utils.clip_grad_norm_(parameters, 1e-3, foreach=True)
        with torch.no_grad():
            for value in parameters:
                value.add_(value.grad, alpha=-0.01)
    payload = tmp_path / "states.pt"
    torch.save(
        {
            "initial": initial,
            "steps": records,
            "final": tuple(v.detach() for v in parameters),
        },
        payload,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
torch.set_num_threads(2)
run = torch._inductor.aoti_load_package(sys.argv[1])
record = torch.load(sys.argv[2], weights_only=True)
parameters = list(record['initial'])
for step in record['steps']:
    actual = run(step['x'], *parameters, step['elements'])
    torch.testing.assert_close(tuple(actual), step['outputs'], atol=2e-10, rtol=2e-9)
    for value, gradient in zip(parameters, actual[-2:]):
        value.grad = gradient
    torch.nn.utils.clip_grad_norm_(parameters, 1e-3, foreach=True)
    for value in parameters:
        value.add_(value.grad, alpha=-0.01)
torch.testing.assert_close(tuple(parameters), record['final'], atol=2e-12, rtol=2e-11)
assert 'equivariantx' not in sys.modules
""",
            str(package),
            str(payload),
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("output_basis", ["spherical", "cartesian"])
def test_cartesian_compact_linear(backend, output_basis, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx import co3
    from equivariantx.conv.co3.linear import Linear

    tp = co3.TensorProduct(
        "2x0e+2x1o+2x2e",
        "1o+2e",
        "2x2e+2x2e+2x1e+2x3o+2x2o+2x1o",
        [
            (0, 1, 0, "uvu", True),
            (1, 0, 1, "uvu", True),
            (1, 0, 2, "uvu", True),
            (2, 0, 3, "uvu", True),
            (2, 0, 4, "uvu", True),
            (2, 0, 5, "uvu", True),
        ],
        project=False,
        internal_weights=False,
        shared_weights=False,
    )
    compact = eqx_conv.CartesianTensorProductConv(
        tp,
        backend=backend,
        normalize=False,
        symmetric_inputs=True,
        input_basis="spherical",
        compact_output=True,
    )
    linear = Linear(
        tp.irreps_out.simplify(),
        "3x1o+3x1e+3x2e+3x2o+3x3o",
        compact.harmonic_output_index,
        internal_weights=False,
        backend=backend,
        output_basis=output_basis,
    )
    # Reconstruct constants when changing model precision.
    compact, linear = [m.float().cuda().double() for m in (compact, linear)]
    reference = eqx_conv.CartesianTensorProductConv(
        tp, backend="torch", normalize=False
    ).cuda()
    basis = co3.ChangeOfBasis(tp.irreps_in1).cuda()
    reference_linear = co3.Linear(
        linear.irreps_in,
        linear.irreps_out,
        internal_weights=False,
        output_basis=output_basis,
        project=output_basis == "cartesian",
    ).cuda()
    edges = torch.randint(4, (2, 11), device="cuda")
    inputs = [
        torch.randn(4, tp.irreps_in1.spherical().dim, device="cuda"),
        torch.randn(11, 3, device="cuda") * 0.2,
        torch.randn(11, 2, device="cuda"),
        torch.randn(2, tp.weight_numel, device="cuda"),
        torch.randn(linear.weight_numel, device="cuda"),
    ]
    inputs[1][0] = 0
    inputs = [x.requires_grad_() for x in inputs]
    x, vectors, radial, projection, weight = inputs
    full = reference(basis(x), None, radial, projection, edges, vectors=vectors)
    packed = compact(x, None, radial, projection, edges, vectors=vectors)
    assert packed.shape[-1] < full.shape[-1]
    torch.testing.assert_close(
        packed[:, compact.harmonic_output_index], full, atol=2e-12, rtol=2e-12
    )
    outputs = [linear(packed, weight), reference_linear(full, weight)]
    torch.testing.assert_close(*outputs, atol=2e-12, rtol=2e-12)
    losses = [out.sin().sum() for out in outputs]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, inputs, create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
        tangents = [torch.randn_like(x) for x in inputs]
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]
