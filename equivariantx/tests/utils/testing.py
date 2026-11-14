"""Tests for the corresponding EQX source module."""

import ast
import doctest
from pathlib import Path

import pytest
import torch
from e3nn import o3

import equivariantx
from equivariantx import co2, co3, o2
from equivariantx import o3 as eqx_o3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPACES = {2: o2, 3: eqx_o3}
ROOT = Path(equivariantx.__file__).resolve().parent


def check_method_precision(module, cases):
    """Reject unsupported operands before public methods cast or combine them."""
    from torch.utils._pytree import tree_flatten, tree_unflatten

    for name, args, kwargs in cases:
        function = getattr(module, name)
        function(*args, **kwargs)
        values, spec = tree_flatten((args, kwargs))
        floating = [
            i
            for i, value in enumerate(values)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ]
        assert floating
        for index in floating:
            for dtype in (torch.float16, torch.bfloat16, torch.complex64):
                changed = list(values)
                changed[index] = values[index].to(dtype)
                inputs, options = tree_unflatten(changed, spec)
                with pytest.raises(TypeError, match="float32 or float64"):
                    function(*inputs, **options)
        device = values[floating[0]].device.type
        with torch.autocast(device, dtype=torch.bfloat16):
            with pytest.raises(TypeError, match="autocast"):
                function(*args, **kwargs)


def check_ictd_method_precision(d, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = SPACES[d].ICTD(2, device=device)
    features = torch.randn(3, d**2, device=device)
    check_method_precision(
        module,
        [
            ("inverse", (module(features),), {}),
            ("project", (features, 0), {}),
        ],
    )


def has_documentation_examples(path):
    """Collect executable examples, excluding commented usage sketches."""
    parser = doctest.DocTestParser()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if parser.get_examples(ast.get_docstring(node) or ""):
                return True
    return False


def check_ictd_basis(rank, d, double_precision):
    module = SPACES[d].ICTD(rank, device=DEVICE)
    matrix = module.change_of_basis
    identity = torch.eye(d**rank, device=DEVICE)
    torch.testing.assert_close(matrix.T @ matrix, identity, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(matrix @ matrix.T, identity, atol=3e-14, rtol=3e-14)
    assert module.irreps_out.dim == module.dim == d**rank
    assert len(module.paths) == len(module.irreps_out)
    assert all(len(path) == rank + 1 for path in module.paths)
    basis = co3.path_matrix(rank) if d == 3 else co2.path_matrix(rank)
    torch.testing.assert_close(
        module.path_matrix(0), basis.to(DEVICE), atol=3e-15, rtol=3e-15
    )
    x = torch.randn(2, d**rank, 3, device=DEVICE).transpose(-1, -2)
    h = module(x)
    torch.testing.assert_close(module.inverse(h), x, atol=3e-14, rtol=3e-14)
    assert module(x[:0]).shape == x[:0].shape
    assert module.inverse(h[:0]).shape == h[:0].shape
    assert not tuple(module.parameters())


def check_ictd_projectors(d, double_precision):
    module = SPACES[d].ICTD(3, device=DEVICE)
    x = torch.randn(5, d**3, device=DEVICE)
    projections = [module.project(x, i) for i in range(len(module.paths))]
    torch.testing.assert_close(sum(projections), x, atol=3e-14, rtol=3e-14)
    for i, projected in enumerate(projections):
        torch.testing.assert_close(module.project(projected, i), projected)
        for j in range(i):
            torch.testing.assert_close(
                module.project(projected, j), torch.zeros_like(x), atol=3e-14, rtol=0
            )
    assert module.project(x[:0], 0).shape == x[:0].shape


def check_ictd_equivariance(rank, d, reflected, double_precision):
    module = SPACES[d].ICTD(rank, device=DEVICE)
    if d == 3:
        rotation = o3.rand_matrix(device=DEVICE) * (-1 if reflected else 1)
        output_rotation = module.irreps_out.D_from_matrix(rotation.cpu()).to(DEVICE)
    else:
        angle = torch.rand((), device=DEVICE)
        rotation = o2.Irrep("1m").D_from_angle(angle, reflected)
        output_rotation = module.irreps_out.D_from_angle(angle, reflected)
    cartesian_rotation = torch.ones(1, 1, device=DEVICE)
    for _ in range(rank):
        cartesian_rotation = torch.kron(cartesian_rotation, rotation.contiguous())
    x = torch.randn(3, d**rank, device=DEVICE)
    torch.testing.assert_close(
        module(x @ cartesian_rotation.T),
        module(x) @ output_rotation.T,
        atol=3e-12,
        rtol=3e-12,
    )


def check_ictd_precision_and_derivatives(d):
    module = SPACES[d].ICTD(3, dtype=torch.float32, device=DEVICE).double()
    reference = SPACES[d].ICTD(3, dtype=torch.float64, device=DEVICE)
    torch.testing.assert_close(
        module.change_of_basis, reference.change_of_basis, atol=0, rtol=0
    )
    x = torch.randn(2, d**3, dtype=torch.float64, device=DEVICE, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: module(x).sin(), (x,), fast_mode=True)
    assert torch.autograd.gradgradcheck(lambda x: module(x).sin(), (x,), fast_mode=True)
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), module(x))


def check_ictd_invalid_rank(rank, d):
    with pytest.raises(ValueError):
        SPACES[d].ICTD(rank)
    with pytest.raises(ValueError):
        list(SPACES[d].path_matrices(rank))


def check_cartesian_representation_precision(space, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    d, label = (2, "2m") if space is co2 else (3, "2e")
    representations = (
        space.Irrep("0e"),
        space.Irrep(label),
        space.Irreps("2x" + label),
        space.Irreps(),
    )
    for representation in representations:
        for dtype in (torch.float16, torch.bfloat16, torch.complex64):
            matrix = torch.eye(d, dtype=dtype, device=device)
            with pytest.raises(TypeError, match="float32 or float64"):
                representation.D_from_matrix(matrix)
        for dtype in (torch.float32, torch.float64):
            matrix = torch.eye(d, dtype=dtype, device=device)
            assert representation.D_from_matrix(matrix).dtype == dtype
            with torch.autocast(device, dtype=torch.bfloat16):
                with pytest.raises(TypeError, match="autocast"):
                    representation.D_from_matrix(matrix)
        for shape in ((d,), (d, d + 1), (1, d + 1, d + 1)):
            with pytest.raises(ValueError, match="trailing shape"):
                representation.D_from_matrix(torch.ones(shape, device=device))


def check_cartesian_representation_transforms(space, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    d, label = (2, "2m") if space is co2 else (3, "2e")
    matrix = (torch.randn(3, d, d, device=device) * 0.2).requires_grad_()
    for direct_sum in (False, True):
        representation = (
            space.Irreps("2x" + label) if direct_sum else space.Irrep(label)
        )

        def oracle(matrix):
            blocks = [torch.kron(value, value) for value in matrix]
            if direct_sum:
                blocks = [torch.block_diag(value, value) for value in blocks]
            return torch.stack(blocks)

        actual, expected = representation.D_from_matrix(matrix), oracle(matrix)
        for _ in range(4):
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            actual, expected = [
                torch.autograd.grad(value.sin().sum(), matrix, create_graph=True)[0]
                for value in (actual, expected)
            ]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        direction = torch.randn_like(matrix)
        torch.testing.assert_close(
            torch.func.jvp(representation.D_from_matrix, (matrix,), (direction,)),
            torch.func.jvp(oracle, (matrix,), (direction,)),
        )
        torch.testing.assert_close(
            torch.func.vmap(representation.D_from_matrix)(matrix), oracle(matrix)
        )
        compiled = torch.compile(
            representation.D_from_matrix, backend="aot_eager", fullgraph=True
        )
        torch.testing.assert_close(compiled(matrix), oracle(matrix))


def transverse_reference(tensor, direction, rank):
    """Independent transverse projection using the two helicity projectors."""
    if rank == 0:
        return tensor + 0 * direction[..., :1]
    x, y, z = direction.unbind(-1)
    zero = torch.zeros_like(x)
    cross = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).unflatten(
        -1, (3, 3)
    )
    plane = torch.eye(
        3, device=direction.device, dtype=direction.dtype
    ) - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    matrix = (plane + 1j * cross) * 0.5
    value = tensor.to(matrix.dtype)
    for axis in range(rank):
        value = torch.einsum(
            "...aib,...ji->...ajb",
            value.unflatten(-1, (3**axis, 3, 3 ** (rank - axis - 1))),
            matrix,
        ).flatten(-3)
    return 2 * value.real


def check_ictd_sparse_backend(d, backend, device, monkeypatch):
    """Check sparse state, derivatives and forward-mode transforms."""
    from equivariantx.ops import config

    monkeypatch.setattr(config, "ICTD_DENSE_MAX_WIDTH", 0)
    module = SPACES[d].ICTD(4, backend=backend, dtype=torch.float64, device=device)
    oracle = SPACES[d].ICTD(4, dtype=torch.float64, device=device)
    matrix = oracle.change_of_basis
    assert module._buffers["change_of_basis"] is None
    x = (torch.randn(2, 3, 2 * d**4, dtype=torch.float64, device=device) * 0.1)[
        ..., ::2
    ].requires_grad_()
    actual, expected = module(x), x @ matrix
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        actual, expected = [
            torch.autograd.grad(y.sin().sum(), x, create_graph=True)[0] * 0.01
            for y in (actual, expected)
        ]
    values = module(x)
    torch.testing.assert_close(module.inverse(values), x, atol=3e-12, rtol=3e-12)
    module.project(x, 0)
    assert module._buffers["change_of_basis"] is None
    contiguous = x.detach().contiguous().requires_grad_()
    actual, expected = module(contiguous), contiguous @ matrix
    for _ in range(5):
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        actual, expected = [
            torch.autograd.grad(y.sin().sum(), contiguous, create_graph=True)[0] * 0.01
            for y in (actual, expected)
        ]
    tangent = torch.randn_like(x)
    torch.testing.assert_close(
        torch.func.jvp(module, (x,), (tangent,)),
        torch.func.jvp(lambda x: x @ matrix, (x,), (tangent,)),
        atol=3e-11,
        rtol=3e-11,
    )
    torch.testing.assert_close(torch.vmap(module)(x), module(x))
    with torch.autograd.forward_ad.dual_level():
        value = torch.autograd.forward_ad.make_dual(x, tangent)
        for operation, reference in (
            (module, lambda x: x @ matrix),
            (module.inverse, lambda x: x @ matrix.T),
            (
                lambda x: module.project(x, 0),
                lambda x: (
                    (x @ matrix[:, module.slices[0]]) @ matrix[:, module.slices[0]].T
                ),
            ),
        ):
            actual = torch.autograd.forward_ad.unpack_dual(operation(value))
            expected = torch.autograd.forward_ad.unpack_dual(reference(value))
            assert actual.tangent is not None
            torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    assert not module.state_dict()


def ictd_training_problem(d, rank, backend, device, dtype=torch.float64):
    """Build a nonlinear complete/inverse/project chain and its force loss."""
    module = SPACES[d].ICTD(rank, backend=backend, dtype=dtype, device=device)
    x = (torch.randn(2, 2, d**rank, device=device, dtype=dtype) * 0.1).requires_grad_()
    w = (torch.randn(2, 1, device=device, dtype=dtype) * 0.1).requires_grad_()

    def operation(x, w):
        value = module.inverse(module(x * w).sin())
        value = module.project(value, 0) + value * 0.1
        energy = 1 + value.sin().sum()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = force.square().sum() + energy.square()
        return loss, force, torch.autograd.grad(loss, w)[0]

    return module, (x, w), operation


def check_ictd_lazy_compile(d, device, monkeypatch):
    """Compile the first dense reference call without preloading its basis."""
    from equivariantx.ops import config

    module = SPACES[d].ICTD(3, dtype=torch.float64, device=device)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    assert module._buffers["change_of_basis"] is None
    compiled = torch.compile(module, backend="eager", fullgraph=True, dynamic=True)
    for count in (2, 5, 0):
        x = torch.randn(count, 2, d**3, dtype=torch.float64, device=device)
        torch.testing.assert_close(compiled(x), module(x))
    assert module._buffers["change_of_basis"] is not None
    from torch.fx.experimental.proxy_tensor import make_fx

    fresh = SPACES[d].ICTD(3, dtype=torch.float64, device=device)
    x = torch.randn(2, d**3, dtype=torch.float64, device=device)
    traced = make_fx(fresh)(x)
    # Basis construction must be lifted as a constant on the very first call.
    assert sum(n.op == "call_function" for n in traced.graph.nodes) < 5
    torch.testing.assert_close(traced(x), fresh(x))


def check_ictd_path_cache(d, backend, device, monkeypatch):
    """One-path maps stay rectangular and regain FP64 constants after casting."""
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config

    monkeypatch.setattr(config, "ICTD_DENSE_MAX_WIDTH", 1024)
    module = SPACES[d].ICTD(4, backend=backend, dtype=torch.float32, device=device)
    oracle = SPACES[d].ICTD(4, dtype=torch.float64, device=device)
    x = torch.randn(2, 3, d**4, dtype=torch.float32, device=device) * 0.1
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        traced = make_fx(lambda value: module.project(value, -1))(x)
        assert module._buffers["change_of_basis"] is None
        assert (
            sum(getattr(module, name) is not None for name in module.path_cache_names)
            == 1
        )
        assert sum(n.op == "call_function" for n in traced.graph.nodes) < 15
        matrix = oracle.path_matrix(-1)
        expected = (x.double() @ matrix) @ matrix.T
        torch.testing.assert_close(traced(x).double(), expected, atol=1e-6, rtol=1e-6)
        module.double()
        from equivariantx.ops.common.ictd import path_basis

        torch.testing.assert_close(
            path_basis(module, -1), matrix, atol=3e-14, rtol=3e-14
        )
        compiled = torch.compile(module.project, backend="eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(x.double(), -1), expected, atol=3e-13, rtol=3e-13
        )
    finally:
        torch.set_float32_matmul_precision(precision)


def check_ictd_native_aoti(d, dtype, monkeypatch, tmp_path):
    """Load an exported force-loss chain and update its weights."""
    import os
    import subprocess
    import sys

    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx.ops import config

    monkeypatch.setattr(config, "ICTD_INLINE", False)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        _, (x, w), run = ictd_training_problem(d, 3, "cuda", "cuda", dtype)
        _, _, reference = ictd_training_problem(d, 3, "torch", "cuda", torch.float64)
        graph = make_fx(run)(x, w)
        graph.graph.eliminate_dead_code()
        graph.recompile()
        exported = torch.export.export(graph, (x.detach(), w.detach()), strict=True)
        package = tmp_path / "ictd.pt2"
        torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
        records = []
        monkeypatch.setattr(config, "TORCH_REFERENCE", True)
        for _ in range(3):
            expected = reference(x.double(), w.double())
            records.append(
                dict(
                    inputs=(x.detach().cpu(), w.detach().cpu()),
                    outputs=tuple(v.detach().cpu() for v in expected),
                )
            )
            w = (w - 0.01 * expected[-1].to(w)).detach().requires_grad_()
        data = tmp_path / "records.pt"
        torch.save(records, data)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                package_script("""
import sys
import torch
run=torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2],weights_only=True):
    with torch.no_grad():actual=run(*(v.cuda() for v in row['inputs']))
    torch.testing.assert_close(tuple(v.double().cpu() for v in actual),row['outputs'],atol=float(sys.argv[3]),rtol=float(sys.argv[3]))
assert 'equivariantx' in sys.modules
"""),
                str(package),
                str(data),
                "3e-5" if dtype == torch.float32 else "3e-9",
            ],
            cwd=tmp_path,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        torch.set_float32_matmul_precision(precision)


def check_forward_ad(operation, reference, inputs):
    """Compare ordinary dual contexts and their reverse derivatives to Torch JVP."""
    from torch.utils._pytree import tree_flatten

    tangents = tuple(torch.randn_like(value) for value in inputs)
    expected = torch.func.jvp(reference, inputs, tangents)
    with torch.autograd.forward_ad.dual_level():
        duals = tuple(
            torch.autograd.forward_ad.make_dual(x, t) for x, t in zip(inputs, tangents)
        )
        values, _ = tree_flatten(operation(*duals))
        actual = [torch.autograd.forward_ad.unpack_dual(value) for value in values]
        primals = [value.primal for value in actual]
        directions = [
            torch.zeros_like(value.primal) if value.tangent is None else value.tangent
            for value in actual
        ]
    expected_primals, _ = tree_flatten(expected[0])
    expected_directions, _ = tree_flatten(expected[1])
    torch.testing.assert_close(
        (primals, directions),
        (expected_primals, expected_directions),
        atol=3e-10,
        rtol=3e-10,
    )
    if all(value.requires_grad for value in inputs):
        losses = [
            sum(v.sin().sum() for v in leaves)
            for leaves in (primals + directions, expected_primals + expected_directions)
        ]
        gradients = [
            torch.autograd.grad(loss, inputs, allow_unused=True) for loss in losses
        ]
        torch.testing.assert_close(gradients[0], gradients[1], atol=3e-9, rtol=3e-9)


def check_linear_forward_ad(space, kind, device):
    """Include feature, shared weight and bias tangents across both backends."""
    vector = "1o" if space.__name__.endswith(".co3") else "1m"
    a, b = f"4x0e+4x{vector}", f"2x0e+4x{vector}"
    modules = {}
    for backend in ("torch", "cuda"):
        layer = (
            space.UuLinear(a, b, 2, backend=backend)
            if kind == "uu"
            else space.Linear(
                a,
                b,
                biases=True,
                shared_weights=True,
                internal_weights=False,
                backend=backend,
            )
        )
        modules[backend] = layer.to(device=device, dtype=torch.float64)
    layer = modules["torch"]
    values = [
        torch.randn(3, layer.irreps_in.dim, device=device, dtype=torch.float64),
        torch.randn(layer.weight_numel, device=device, dtype=torch.float64),
    ]
    if kind == "linear":
        values.append(torch.randn(layer.bias_numel, device=device, dtype=torch.float64))
    check_forward_ad(
        modules["cuda"], modules["torch"], tuple(v.requires_grad_() for v in values)
    )


def package_script(script):
    """Load the checked-out EQX operators before executing an AOTI package."""
    from pathlib import Path

    import equivariantx

    root = str(Path(equivariantx.__file__).resolve().parent.parent)
    return f"import sys\nsys.path.insert(0, {root!r})\nimport equivariantx\n" + script
