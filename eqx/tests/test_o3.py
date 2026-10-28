"""Spherical O(3) operators and derivatives."""

import math

import pytest
import torch
from e3nn import o3

from eqx import o3 as eqx_o3
from eqx.o3.s2grid import S2Grid

IRREPS = "0e+0o+1e+1o+2e+2o"


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize(
    "irrep_normalization,path_normalization",
    [("component", "element"), ("norm", "path"), ("none", "none")],
)
def test_wigner6j_tensor_product(
    backend, irrep_normalization, path_normalization, double_precision
):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = eqx_o3.Wigner6jTensorProduct(
        "2x1o",
        "3x1o",
        "2x1e",
        "2x0e+2x1e+2x2e",
        "2x0e+2x1e+2x2e",
        [(0, 0, k, "uvu", k != 1) for k in range(3)] + [(0, 0, 0, "uvu", True, 0.5)],
        [
            (i, 0, k, "uvu", i != 2)
            for i in range(3)
            for k in range(3)
            if abs(i - 1) <= k <= i + 1
        ],
        irrep_normalization=irrep_normalization,
        path_normalization=path_normalization,
        internal_weights=False,
        shared_weights=False,
        backend=backend,
    ).to(device)
    inputs = [
        torch.randn(4, width, device=device, requires_grad=True)
        for width in (6, 9, 6, module.weight_numel1, module.weight_numel2)
    ]
    actual, expected = module(*inputs), module.forward_reference(*inputs)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for _ in range(3):
        seed = torch.randn_like(actual)
        gradients = [
            torch.autograd.grad((value.sin() * seed).sum(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
        actual, expected = [torch.cat([g.flatten() for g in gs]) for gs in gradients]


@pytest.mark.parametrize("degrees", [(0, 2, 1), (2, 3, 2), (4, 3, 2)])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_wigner6j_shared_weights(degrees, backend, double_precision):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    l1, l2, l3 = degrees
    ir1, ir2, ir3 = o3.Irrep(l1, -1), o3.Irrep(l2, 1), o3.Irrep(l3, -1)
    mid = o3.Irreps([(2, ir) for ir in ir1 * ir2])
    out = o3.Irreps([(2, o3.Irrep(l, 1)) for l in range(l1 + l2 + l3 + 1)])
    module = eqx_o3.Wigner6jTensorProduct(
        [(2, ir1)],
        [(1, ir2)],
        [(1, ir3)],
        mid,
        out,
        [(0, 0, i, "uvu", True) for i in range(len(mid))],
        [
            (i, 0, k, "uvu", True)
            for i, (_, ir) in enumerate(mid)
            for k, (_, ir_out) in enumerate(out)
            if ir_out in ir * ir3
        ],
        backend=backend,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device)
    x1 = torch.randn(2, 1, 2 * ir1.dim, device=device)
    x2 = torch.randn(3, ir2.dim, device=device)
    x3 = torch.randn(ir3.dim, device=device)
    actual, expected = module(x1, x2, x3), module.forward_reference(x1, x2, x3)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), tuple(module.parameters())),
        torch.autograd.grad(expected.square().sum(), tuple(module.parameters())),
    ):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("degree", range(6))
def test_rotation_generators_match_wigner(degree, double_precision):
    angle = torch.zeros((), dtype=torch.float64)
    half_pi = angle.new_tensor(torch.pi / 2)
    matrices = (
        lambda a: o3.wigner_D(degree, angle, a, angle),
        lambda a: o3.wigner_D(degree, a, angle, angle),
        lambda a: o3.wigner_D(degree, -half_pi, a, half_pi),
    )
    expected = torch.stack(
        [torch.autograd.functional.jacobian(matrix, angle) for matrix in matrices]
    )
    torch.testing.assert_close(
        eqx_o3.so3_generators(degree), expected, atol=1e-13, rtol=1e-13
    )


@pytest.mark.parametrize("num_nodes", [0, 5])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("gate_scalars", [False, True])
def test_gate_derivatives(num_nodes, device, gate_scalars, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from e3nn.nn import Gate

    args = (
        "2x0e+0o+0e",
        [torch.nn.SiLU(), torch.nn.Tanh(), None],
        "2x0e+2x0o",
        [torch.nn.Sigmoid(), torch.nn.Tanh()],
        "2x0e+1o+3e" if gate_scalars else "2x1o+1e+3e",
    )
    module = eqx_o3.Gate(*args).to(device)
    reference = Gate(*args).to(device)
    module.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(module.irreps_in.dim, num_nodes, device=device).T.requires_grad_()
    actual, expected = module(x), reference(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for _ in range(3):
        seed = torch.randn_like(actual)
        actual, expected = [
            torch.autograd.grad((y.sin() * seed).sum(), x, create_graph=True)[0]
            for y in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("scalar_only", [False, True])
def test_gate_compile(scalar_only, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from e3nn.nn import Gate

    args = (
        ("2x0e", [torch.nn.SiLU()], "", [], "")
        if scalar_only
        else ("", [], "2x0e", [torch.nn.Sigmoid()], "1o+2e")
    )
    module = eqx_o3.Gate(*args).cuda()
    reference = Gate(*args).cuda()
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = torch.randn(
            2, count, module.irreps_in.dim, device="cuda", requires_grad=True
        )
        actual, expected = compiled(x), reference(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().sum(), x)[0],
            torch.autograd.grad(expected.square().sum(), x)[0],
            atol=1e-12,
            rtol=1e-12,
        )


def test_gate_fallback(double_precision):
    from e3nn.nn import Gate

    for backend, act in (("torch", torch.nn.SiLU()), ("cuda", torch.nn.ReLU())):
        args = ("2x0e", [act], "0e", [None], "1o")
        module = eqx_o3.Gate(*args, backend=backend)
        assert module.metadata is None
        x = torch.randn(4, module.irreps_in.dim, requires_grad=True)
        actual, expected = module(x), Gate(*args)(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().sum(), x)[0],
            torch.autograd.grad(expected.square().sum(), x)[0],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize(
    "irreps_in,irreps_out", [("", "2x0e"), ("2x1o", "3x0e"), ("0x0e+2x1o", "0x1o+0e")]
)
def test_indexed_linear_disconnected(irreps_in, irreps_out):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linear = o3.Linear(
        irreps_in, irreps_out, internal_weights=False, shared_weights=False
    )
    module = eqx_o3.ElementLinear(linear)
    x = torch.randn(3, linear.irreps_in.dim, device=device, requires_grad=True)
    w = torch.randn(2, linear.weight_numel, device=device, requires_grad=True)
    y = module(x, w, torch.tensor([0, 1, 0], device=device))
    assert y.shape == (3, linear.irreps_out.dim)
    assert not y.count_nonzero()
    for gradient in torch.autograd.grad(y.sum(), (x, w)):
        assert not gradient.count_nonzero()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_experts", [1, 2])
@pytest.mark.parametrize("num_nodes", [0, 5])
def test_indexed_linear_derivatives(device, num_experts, num_nodes, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    linear = o3.Linear(
        "2x0e+0e+2x1o+1e",
        "3x0e+2x1o+1o+2o",
        internal_weights=False,
        shared_weights=False,
    )
    cls = eqx_o3.ElementLinear if num_experts == 1 else eqx_o3.MoEElementLinear
    kwargs = {} if num_experts == 1 else {"num_experts": num_experts}
    module = cls(linear, **kwargs)
    reference = cls(linear, backend="torch", **kwargs)
    x = torch.randn(module.irreps_in.dim, num_nodes, device=device).T.requires_grad_()
    shape = (
        (3, linear.weight_numel)
        if num_experts == 1
        else (3, num_experts, linear.weight_numel)
    )
    weight = torch.randn(*shape, device=device, requires_grad=True)
    types = (torch.arange(2 * num_nodes, device=device) % 3)[::2]
    actual, expected = module(x, weight, types), reference(x, weight, types)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for _ in range(3):
        seed = torch.randn_like(actual)
        gradients = [
            torch.autograd.grad((y.sin() * seed).sum(), (x, weight), create_graph=True)
            for y in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        actual, expected = [
            torch.cat([g.flatten() for g in grads]) for grads in gradients
        ]
    assert not module.state_dict()


@pytest.mark.parametrize("experts", [1, 2])
def test_indexed_linear_compile_and_capture(experts, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    linear = o3.Linear(
        "2x0e+2x1o", "3x0e+2x1o", internal_weights=False, shared_weights=False
    )
    module = eqx_o3.MoEElementLinear(linear, experts)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for count in (3, 7, 0):
        x = torch.randn(count, module.irreps_in.dim, device="cuda", requires_grad=True)
        weight = torch.randn(
            2, experts, linear.weight_numel, device="cuda", requires_grad=True
        )
        types = torch.arange(count, device="cuda") % 2
        actual, expected = compiled(x, weight, types), module(x, weight, types)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), (x, weight)),
            torch.autograd.grad(expected.square().sum(), (x, weight)),
        ):
            torch.testing.assert_close(a, b)
    x = torch.randn(7, module.irreps_in.dim, device="cuda", requires_grad=True)
    weight = torch.randn(
        2, experts, linear.weight_numel, device="cuda", requires_grad=True
    )
    types = torch.arange(7, device="cuda") % 2

    def evaluate():
        y = module(x, weight, types)
        first = torch.autograd.grad(y.sin().sum(), (x, weight), create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), (x, weight))
        return y, *first, *second

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            evaluate()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = evaluate()
    with torch.no_grad():
        x.normal_()
        weight.normal_()
        types.random_(2)
    graph.replay()
    for a, b in zip(actual, evaluate()):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_product_equivariance(cls, device, dtype, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = cls(IRREPS, IRREPS, IRREPS, shared_weights=False).to(
        device=device, dtype=dtype
    )
    x = torch.randn(3, module.irreps_in1.dim, device=device, dtype=dtype)
    y = torch.randn_like(x)
    weight = torch.randn(3, module.weight_numel, device=device, dtype=dtype)
    for matrix in (o3.rand_matrix(), -torch.eye(3), -o3.rand_matrix()):
        rotation = module.irreps_in1.D_from_matrix(matrix).to(
            device=device, dtype=dtype
        )
        actual = module(x @ rotation.T, y @ rotation.T, weight)
        expected = module(x, y, weight) @ rotation.T
        tolerance = 2e-5 if dtype == torch.float32 else 2e-12
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("normalization", ["none", "component", "norm"])
def test_gaunt_quadrature_matches_coefficients(normalization, double_precision):
    args = ("2x2o+0e+2x1o+0e", "1e+2o+0e", "2x0o+2x1e+2x2o+2x3e")
    kwargs = dict(irrep_normalization=normalization, shared_weights=False)
    grid = eqx_o3.GauntTensorProduct(*args, **kwargs)
    coefficients = eqx_o3.GauntTensorProduct(*args, method="coefficients", **kwargs)
    x = torch.randn(2, 1, grid.irreps_in1.dim, requires_grad=True)
    y = torch.randn(1, 3, grid.irreps_in2.dim, requires_grad=True)
    weight = torch.randn(2, 3, grid.weight_numel, requires_grad=True)
    actual, expected = grid(x, y, weight), coefficients(x, y, weight)
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        seed = torch.randn_like(actual)
        gradients = [
            torch.autograd.grad(
                (value.sin() * seed).sum(), (x, y, weight), create_graph=True
            )
            for value in (actual, expected)
        ]
        actual, expected = [
            torch.cat([g.flatten() for g in items]) for items in gradients
        ]


def test_matrix_matches_cg_reference(double_precision):
    module = eqx_o3.MatrixTensorProduct(
        IRREPS, IRREPS, IRREPS, irrep_normalization="none"
    )
    basis = {l: math.sqrt(2 * l + 1) * o3.wigner_3j(1, 1, l) for l in range(3)}
    instructions, weights = [], []
    for i, (_, ir1) in enumerate(module.irreps_in1):
        for j, (_, ir2) in enumerate(module.irreps_in2):
            for k, (_, ir) in enumerate(module.irreps_out):
                if ir not in ir1 * ir2:
                    continue
                coefficient = torch.einsum(
                    "abi,bcj,ack->ijk", basis[ir1.l], basis[ir2.l], basis[ir.l]
                )
                weights.append((coefficient * o3.wigner_3j(ir1.l, ir2.l, ir.l)).sum())
                instructions.append((i, j, k, "uuu", True))
    reference = o3.TensorProduct(
        IRREPS,
        IRREPS,
        IRREPS,
        instructions,
        internal_weights=False,
        irrep_normalization="none",
        path_normalization="none",
    )
    x, y = torch.randn(2, 4, module.irreps_in1.dim)
    torch.testing.assert_close(
        module(x, y, torch.ones(module.weight_numel)),
        reference(x, y, torch.stack(weights)),
        atol=2e-13,
        rtol=2e-13,
    )


def test_antisymmetric_coupling(double_precision):
    x, y = torch.randn(2, 7, 3)
    matrix = eqx_o3.MatrixTensorProduct("1o", "1o", "1e")
    weight = torch.ones(matrix.weight_numel)
    result = matrix(x, y, weight)
    assert result.norm() > 0.1
    torch.testing.assert_close(result, -matrix(y, x, weight), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(
        matrix(x, x, weight), torch.zeros_like(result), atol=1e-13, rtol=0
    )
    for method in ("quadrature", "coefficients"):
        gaunt = eqx_o3.GauntTensorProduct("1o", "1o", "1e", method=method)
        assert not gaunt(x, y).count_nonzero()
        assert not gaunt.output_mask.count_nonzero()


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
@pytest.mark.parametrize("normalization", ["component", "norm"])
def test_product_variance_normalization(cls, normalization, double_precision):
    module = cls(IRREPS, IRREPS, IRREPS, irrep_normalization=normalization)
    features = torch.eye(module.irreps_in1.dim)
    if normalization == "norm":
        scale = torch.cat(
            [torch.full((ir.dim,), math.sqrt(ir.dim)) for _, ir in module.irreps_in1]
        )
        features = features / scale
    coefficients = module(
        features[:, None], features[None], torch.ones(module.weight_numel)
    )
    variances = coefficients.square().sum((0, 1))
    expected = torch.ones_like(variances)
    if normalization == "norm":
        expected /= scale.square()
    torch.testing.assert_close(variances, expected, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
def test_regroup_and_broadcast(cls, double_precision):
    module = cls("1o+2x0e+1o", "0e+1o", "0e+2x1o+0e", shared_weights=False)
    reference = cls("2x0e+2x1o", "0e+1o", "2x0e+2x1o", shared_weights=False)
    x = torch.randn(3, 1, 8)
    y = torch.randn(1, 2, 4)
    weight = torch.randn(3, 2, module.weight_numel)
    packed = torch.cat((x[..., 3:5], x[..., :3], x[..., 5:]), -1)
    expected = reference(packed, y, weight)
    expected = torch.cat((expected[..., :1], expected[..., 2:], expected[..., 1:2]), -1)
    torch.testing.assert_close(module(x, y, weight), expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradcheck_and_gradgradcheck(cls, device, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = cls("0e+1o", "0o+1e", "0o+1e+2o", shared_weights=False).to(device)
    inputs = (
        torch.randn(2, 4, device=device, requires_grad=True),
        torch.randn(2, 4, device=device, requires_grad=True),
        torch.randn(2, module.weight_numel, device=device, requires_grad=True),
    )
    assert torch.autograd.gradcheck(module, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(module, inputs, fast_mode=True)


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
def test_precision_state_and_empty_batch(cls, double_precision):
    args = ("0e+1o+3o", "0e+1o", "0e+1o+2e+3o")
    module = cls(*args)
    torch.set_default_dtype(torch.float32)
    restored = cls(*args).double()
    torch.set_default_dtype(torch.float64)
    restored.load_state_dict(module.state_dict())
    for name, value in module.named_buffers():
        torch.testing.assert_close(
            value, dict(restored.named_buffers())[name], atol=0, rtol=0
        )
    x = torch.randn(2, module.irreps_in1.dim)
    y = torch.randn(2, module.irreps_in2.dim)
    torch.testing.assert_close(restored(x, y), module(x, y), atol=0, rtol=0)
    empty = module(x[:0], y[:0])
    assert empty.shape == (0, module.irreps_out.dim)
    assert torch.autograd.grad(empty.sum(), module.weight)[0].count_nonzero() == 0


def test_matrix_degree_and_structural_zero(double_precision):
    module = eqx_o3.MatrixTensorProduct("3o", "5o", "5e", matrix_degree=3)
    assert not module.output_mask.count_nonzero()
    assert not module(torch.randn(2, 7), torch.randn(2, 11)).count_nonzero()
    assert eqx_o3.MatrixTensorProduct("3o", "0e", "3o").matrix_degree == 2
    with pytest.raises(ValueError, match="matrix_degree"):
        eqx_o3.MatrixTensorProduct("3o", "0e", "3o", matrix_degree=1)


@pytest.mark.parametrize("degree", [5, 6])
def test_high_degree_products(degree, double_precision):
    irreps = f"0e+{degree}e+{degree}o"
    gaunt = eqx_o3.GauntTensorProduct(irreps, irreps, irreps)
    reference = eqx_o3.GauntTensorProduct(irreps, irreps, irreps, method="coefficients")
    reference.load_state_dict(gaunt.state_dict())
    x, y = torch.randn(2, 3, gaunt.irreps_in1.dim)
    torch.testing.assert_close(gaunt(x, y), reference(x, y), atol=2e-12, rtol=2e-12)
    matrix = eqx_o3.MatrixTensorProduct(irreps, irreps, irreps)
    rotation = matrix.irreps_in1.D_from_matrix(-o3.rand_matrix())
    torch.testing.assert_close(
        matrix(x @ rotation.T, y @ rotation.T),
        matrix(x, y) @ rotation.T,
        atol=2e-11,
        rtol=2e-11,
    )


@pytest.mark.parametrize("cls", (eqx_o3.GauntTensorProduct, eqx_o3.MatrixTensorProduct))
def test_invalid_layout_and_weights(cls):
    with pytest.raises(ValueError, match="uniform"):
        cls("2x0e+1o", "0e", "2x0e")
    with pytest.raises(ValueError, match="multiplicities"):
        cls("2x0e", "3x0e", "3x0e")
    with pytest.raises(ValueError, match="shared"):
        cls("0e", "0e", "0e", internal_weights=True, shared_weights=False)
    module = cls("0e", "0e", "0e", internal_weights=False)
    with pytest.raises(ValueError, match="External weights"):
        module(torch.ones(1), torch.ones(1))


@pytest.fixture(params=["lebedev", "gauss_legendre", "equiangular"])
def quadrature(request):
    return request.param


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize("lmax", [0, 6, 13])
def test_s2_round_trip(lmax, normalization, quadrature, device):
    grid = S2Grid(lmax, normalization=normalization, quadrature=quadrature)
    grid = grid.to(device, torch.float64)
    for count in (0, 4):
        x = torch.randn(grid.dim, count, 3, dtype=torch.float64, device=device)
        x = x.movedim(0, -1)
        torch.testing.assert_close(grid.from_grid(grid(x)), x, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(grid.weights.sum(), grid.weights.new_tensor(1))


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_s2_e3nn(normalization, double_precision, device):
    grid = S2Grid(3, normalization=normalization, quadrature="equiangular").to(device)
    shape = (10, 10)
    synthesis = o3.ToS2Grid(3, shape, normalization=normalization).to(device)
    analysis = o3.FromS2Grid(shape, 3, normalization=normalization).to(device)
    x = torch.randn(2, 4, grid.dim, device=device, requires_grad=True)
    torch.testing.assert_close(
        grid(x), synthesis(x).flatten(-2), atol=1e-13, rtol=1e-13
    )
    actual = grid.from_grid(torch.nn.functional.silu(grid(x)))
    expected = analysis(torch.nn.functional.silu(synthesis(x)))
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        seed = torch.randn_like(actual)
        actual, expected = [
            torch.autograd.grad((value.sin() * seed).sum(), x, create_graph=True)[0]
            for value in (actual, expected)
        ]


@pytest.mark.parametrize("power", [2, 3])
@pytest.mark.parametrize("inversion", [False, True])
def test_s2_polynomial_equivariance(
    power, inversion, quadrature, double_precision, device
):
    grid = S2Grid(3, 3 * (power + 1), quadrature=quadrature).to(device)
    irreps = o3.Irreps.spherical_harmonics(3)
    rotation = o3.rand_matrix() * (-1 if inversion else 1)
    matrix = irreps.D_from_matrix(rotation).to(device)
    x = torch.randn(3, grid.dim, device=device)
    actual = grid.from_grid(grid(x @ matrix.T).pow(power))
    expected = grid.from_grid(grid(x).pow(power)) @ matrix.T
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_s2_dtype_and_derivatives(double_precision, device):
    grid = S2Grid(1, dtype=torch.float32, quadrature="gauss_legendre").to(
        device, torch.float64
    )
    reference = S2Grid(
        1, dtype=torch.float64, device=device, quadrature="gauss_legendre"
    )
    for actual, expected in zip(grid.buffers(), reference.buffers()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not grid.state_dict()
    x = torch.randn(1, grid.dim, device=device, requires_grad=True)

    def nonlinear(x):
        return grid.from_grid(torch.nn.functional.silu(grid(x)))

    assert torch.autograd.gradcheck(nonlinear, (x,))
    assert torch.autograd.gradgradcheck(nonlinear, (x,))
    value = nonlinear(x)
    for _ in range(3):
        value = torch.autograd.grad(value.sin().sum(), x, create_graph=True)[0]
        assert torch.isfinite(value).all()


def test_s2_compile(double_precision, device):
    grid = S2Grid(2, device=device, quadrature="gauss_legendre")

    def nonlinear(x):
        return grid.from_grid(torch.nn.functional.silu(grid(x)))

    compiled = torch.compile(
        nonlinear, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for count in (3, 7, 0):
        x = torch.randn(count, 4, grid.dim, device=device, requires_grad=True)
        actual, expected = compiled(x), nonlinear(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().sum(), x)[0],
            torch.autograd.grad(expected.square().sum(), x)[0],
            atol=1e-12,
            rtol=1e-12,
        )


def test_extended_quadrature(device):
    for resolution, quadrature in [(131, "lebedev"), (140, "gauss_legendre")]:
        grid = S2Grid(2, resolution, quadrature=quadrature, dtype=torch.float64).to(
            device
        )
        assert grid.degree >= resolution
        for power in (2, 20, 130):
            integral = (grid.grid[:, 0].pow(power) * grid.weights).sum()
            torch.testing.assert_close(
                integral, integral.new_tensor(1 / (power + 1)), atol=1e-13, rtol=1e-13
            )
    with pytest.raises(ValueError, match="131"):
        S2Grid(2, 132)


def test_s2_invalid_inputs():
    with pytest.raises(ValueError):
        S2Grid(-1)
    with pytest.raises(ValueError):
        S2Grid(3, resolution=2)
    with pytest.raises(ValueError):
        S2Grid(2, normalization="invalid")
    grid = S2Grid(2)
    with pytest.raises(ValueError):
        grid(torch.empty(2, grid.dim + 1))
    with pytest.raises(ValueError):
        grid.from_grid(torch.empty(2, 1))
