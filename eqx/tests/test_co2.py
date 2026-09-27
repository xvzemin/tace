"""Cartesian O(2) operators and coordinate-free CGTP equivalence."""

import math

import pytest
import torch
from e3nn import o3

from eqx import co2, co3, o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_basis_and_representations(double_precision):
    irreps = co2.Irreps("2x0e+1x0oo+3x1mo+2x3m+1x1mo")
    assert list(irreps)[0] == (2, co2.Irrep("0e"))
    assert irreps.count("1mo") == 4
    assert irreps.regroup().num_irreps == irreps.num_irreps
    assert irreps.filter(mmax=0).dim == 3
    assert co2.Irreps(irreps.circular()) == irreps
    for m in range(9):
        matrix = co2.path_matrix(m).to(DEVICE)
        torch.testing.assert_close(
            matrix.T @ matrix, torch.eye(matrix.shape[1], device=DEVICE)
        )
    embed = co2.ChangeOfBasis(irreps).to(DEVICE)
    extract = co2.ChangeOfBasis(irreps, inverse=True).to(DEVICE)
    compact = irreps.circular().randn(4, -1, device=DEVICE)
    torch.testing.assert_close(extract(embed(compact)), compact)
    for reflected in (False, True):
        for time_reversal in (False, True):
            angle = torch.rand(4, device=DEVICE)
            d = irreps.D_from_angle(angle, reflected, time_reversal)
            expected = torch.einsum(
                "bij,bj->bi",
                irreps.circular().D_from_angle(angle, reflected, time_reversal),
                compact,
            )
            actual = extract(torch.einsum("bij,bj->bi", d, embed(compact)))
            torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    x = irreps.randn(3, -1, device=DEVICE)
    torch.testing.assert_close(co2.Projector(irreps).to(DEVICE)(x), x)
    assert irreps.randn(0, -1, device=DEVICE).shape == (0, irreps.dim)
    assert irreps.randn(-1, 0, device=DEVICE).shape == (irreps.dim, 0)


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_harmonics(double_precision, normalization):
    module = co2.CartesianHarmonics(
        [0, 1, 3, 5, 2], normalization=normalization, time_reversal=True
    ).to(DEVICE)
    x = torch.randn(4, 2, device=DEVICE, requires_grad=True) * 0.2
    values = o2.circular_harmonics(x, 5, normalize=False)
    expected = []
    for m in module.orders:
        scale = 1 if normalization == "norm" or m == 0 else math.sqrt(2)
        if normalization == "integral":
            scale /= math.sqrt(2 * math.pi)
        expected.append(
            (values[..., :1] if m == 0 else values[..., 2 * m - 1 : 2 * m + 1]) * scale
        )
    actual = co2.ChangeOfBasis(module.irreps_out, True).to(DEVICE)(module(x))
    torch.testing.assert_close(actual, torch.cat(expected, -1))
    zero = torch.zeros(1, 2, device=DEVICE, requires_grad=True)
    grad = torch.autograd.grad(module(zero).sum(), zero, create_graph=True)[0]
    assert torch.isfinite(grad).all()
    assert torch.isfinite(torch.autograd.grad(grad.sum(), zero)[0]).all()


@pytest.mark.parametrize("path_normalization", ["element", "path"])
def test_linear_and_gate(double_precision, path_normalization):
    a, b = "2x0e+3x1mo+1x1mo+2x3m", "1x0e+2x1mo+2x3m+1x0o"
    ref = o2.Linear(
        a,
        b,
        biases=True,
        internal_weights=False,
        shared_weights=False,
        path_normalization=path_normalization,
    ).to(DEVICE)
    module = co2.Linear(
        a,
        b,
        biases=True,
        internal_weights=False,
        shared_weights=False,
        path_normalization=path_normalization,
        output_basis="circular",
    ).to(DEVICE)
    x = torch.randn(4, ref.irreps_in.dim, device=DEVICE)
    weight = torch.randn(1, ref.weight_numel, device=DEVICE)
    bias = torch.randn(4, ref.bias_numel, device=DEVICE)
    embed = co2.ChangeOfBasis(a).to(DEVICE)
    torch.testing.assert_close(module(embed(x), weight, bias), ref(x, weight, bias))
    args = (
        "2x0e+1x0oo",
        [torch.nn.functional.silu, torch.tanh],
        "2x0e+2x0oo",
        [torch.sigmoid, torch.tanh],
        "2x1mo+2x3m",
    )
    gate = co2.Gate(*args).to(DEVICE)
    compact_gate = o2.Gate(*args).to(DEVICE)
    x = compact_gate.irreps_in.randn(5, -1, device=DEVICE)
    actual = co2.ChangeOfBasis(gate.irreps_out, True).to(DEVICE)(
        gate(co2.ChangeOfBasis(gate.irreps_in).to(DEVICE)(x))
    )
    torch.testing.assert_close(actual, compact_gate(x))


@pytest.mark.parametrize("mode", ["u1u", "uuu", "uvw"])
@pytest.mark.parametrize("irrep_normalization", ["component", "norm", "none"])
@pytest.mark.parametrize("path_normalization", ["element", "path", "none"])
def test_tensor_product(
    double_precision, mode, irrep_normalization, path_normalization
):
    v = 1 if mode == "u1u" else 2
    a, b = co2.Irreps("2x0oo+2x1mo+2x2m"), co2.Irreps(f"{v}x0o+{v}x1m+{v}x2mo")
    output, ins = [], []
    for i, (_, ir) in enumerate(a):
        for j, (_, jr) in enumerate(b):
            for kr in ir * jr:
                ins.append((i, j, len(output), mode, True, 0.7))
                output.append((3 if mode == "uvw" else 2, kr))
    c = co2.Irreps(output)
    ins.append((*ins[0][:5], 1.3))
    kwargs = dict(
        internal_weights=False,
        shared_weights=False,
        irrep_normalization=irrep_normalization,
        path_normalization=path_normalization,
        in1_var=[0.7, 1.2, 1.1],
        in2_var=[1.2, 0.9, 1.3],
    )
    ref = o2.TensorProduct(a.circular(), b.circular(), c.circular(), ins, **kwargs).to(
        DEVICE
    )
    module = co2.TensorProduct(a, b, c, ins, project=False, **kwargs).to(DEVICE)
    x = a.circular().randn(2, -1, device=DEVICE, requires_grad=True)
    y = b.circular().randn(1, -1, device=DEVICE, requires_grad=True)
    weight = torch.randn(2, module.weight_numel, device=DEVICE, requires_grad=True)
    embed_a, embed_b = co2.ChangeOfBasis(a).to(DEVICE), co2.ChangeOfBasis(b).to(DEVICE)
    extract = co2.ChangeOfBasis(c, True).to(DEVICE)
    expected = ref(x, y, weight)
    actual = extract(module(embed_a(x), embed_b(y), weight))
    torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)
    for _ in range(2):
        probe = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        (value * probe).sum(),
                        (x, y, weight),
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    assert module(embed_a(x[:0]), embed_b(y), weight[:0]).shape == (0, c.dim)


@pytest.mark.parametrize("l", range(7))
def test_restriction(double_precision, l):
    module = co2.Restriction(l).to(DEVICE)
    spherical = torch.randn(3, 2 * l + 1, device=DEVICE)
    x = spherical @ co3.path_matrix(l).to(DEVICE).T
    n = torch.randn(3, 3, device=DEVICE)
    n = n / n.norm(dim=-1, keepdim=True)
    values = module(x, n)
    torch.testing.assert_close(module.inverse(values, n), x, atol=8e-13, rtol=8e-13)
    torch.testing.assert_close(
        sum(v.square().sum(-1) for v in values),
        spherical.square().sum(-1),
        atol=8e-13,
        rtol=8e-13,
    )
    for m, value in enumerate(values):
        if m:
            contraction = torch.einsum(
                "bik,bk->bi", value.reshape(3, 3 ** (m - 1), 3), n
            )
            torch.testing.assert_close(
                contraction, torch.zeros_like(contraction), atol=5e-13, rtol=0
            )
        if m > 1:
            trace = (
                value.reshape(3, 3 ** (m - 2), 3, 3).diagonal(dim1=-2, dim2=-1).sum(-1)
            )
            torch.testing.assert_close(
                trace, torch.zeros_like(trace), atol=5e-13, rtol=0
            )


@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize("path_normalization", ["element", "path", "none"])
def test_o3_equivalence(double_precision, mode, normalization, path_normalization):
    a, b = o3.Irreps("2x1o+3x2e"), o3.Irreps.spherical_harmonics(2)
    output, instructions = [], []
    for i, (mul, ir) in enumerate(a):
        for j, (_, jr) in enumerate(b):
            for kr in ir * jr:
                instructions.append((i, j, len(output), mode, True, 0.8))
                output.append((mul if mode == "uvu" else 4, kr))
    c = o3.Irreps(output)
    instructions.append((*instructions[0][:5], 1.3))
    options = dict(
        internal_weights=False,
        shared_weights=False,
        path_normalization=path_normalization,
        irrep_normalization="norm",
        in1_var=[0.8, 1.3],
        in2_var=[0.7, 1.2, 1.1],
    )
    ref = o3.TensorProduct(a, b, c, instructions, **options).to(DEVICE)
    module = co2.O3TensorProduct(
        a,
        b,
        c,
        instructions,
        normalization=normalization,
        input_basis="spherical",
        output_basis="spherical",
        **options,
    ).to(DEVICE)
    cartesian = co3.TensorProduct(a, b, c, instructions, **options).to(DEVICE)
    x = torch.randn(3, a.dim, device=DEVICE, requires_grad=True)
    vectors = torch.randn(3, 3, device=DEVICE, requires_grad=True)
    weight = torch.randn(3, ref.weight_numel, device=DEVICE, requires_grad=True)
    harmonics = o3.spherical_harmonics(b, vectors, True, normalization)
    expected = ref(x, harmonics, weight)
    actual = module(x, vectors, weight)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    transforms = [
        co3.ChangeOfBasis(irreps, inverse).to(DEVICE)
        for irreps, inverse in ((a, False), (b, False), (c, True))
    ]
    torch.testing.assert_close(
        actual,
        transforms[2](cartesian(transforms[0](x), transforms[1](harmonics), weight)),
        atol=3e-12,
        rtol=3e-12,
    )
    for _ in range(2):
        probe = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        (value * probe).sum(),
                        (x, vectors, weight),
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=5e-11, rtol=5e-11)
    assert module(x[:0], vectors[:0], weight[:0]).shape == (0, c.dim)


def test_scatter_compile_and_high_degree(double_precision):
    a, b, c = "2x5e", "1x3o", "2x4o+2x5o"
    ins = [(0, 0, 0, "uvu", True), (0, 0, 1, "uvu", True)]
    module = co2.O3TensorProduct(
        a,
        b,
        c,
        ins,
        shared_weights=False,
        internal_weights=False,
        input_basis="spherical",
        output_basis="spherical",
    ).to(DEVICE)
    ref = o3.TensorProduct(
        a, b, c, ins, shared_weights=False, internal_weights=False
    ).to(DEVICE)
    x = torch.randn(4, o3.Irreps(a).dim, device=DEVICE)
    edge_index = torch.tensor([[0, 1, 2, 3, 2], [1, 0, 1, 1, 0]], device=DEVICE)
    vectors = torch.randn(5, 3, device=DEVICE)
    weight = torch.randn(5, ref.weight_numel, device=DEVICE)
    sh = o3.spherical_harmonics(o3.Irreps(b), vectors, True, normalization="component")
    expected = ref(x[edge_index[0]], sh, weight)
    actual = module.forward_scatter(x, vectors, edge_index, weight)
    torch.testing.assert_close(
        actual,
        expected.new_zeros(4, expected.shape[-1]).index_add(0, edge_index[1], expected),
        atol=3e-12,
        rtol=3e-12,
    )
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(x[edge_index[0]], vectors, weight), expected, atol=3e-12, rtol=3e-12
    )


@pytest.mark.parametrize("cartesian", [False, True])
def test_conversion_and_deferred_projection(double_precision, cartesian):
    a, b, c = "2x2e", "1x2e", "2x2e+2x3e"
    ins = [
        (0, 0, 0, "uvu", True, 0.4),
        (0, 0, 0, "uvu", True, 1.7),
        (0, 0, 1, "uvu", True, 0.8),
    ]
    cls = co3.TensorProduct if cartesian else o3.TensorProduct
    ref = cls(
        a, b, c, ins, path_normalization="path", in1_var=[0.7], out_var=[1.3, 0.8]
    ).to(DEVICE)
    ref.eval()
    ref.weight.requires_grad_(False)
    # Conversion must not lose precision through the ambient default dtype.
    torch.set_default_dtype(torch.float32)
    try:
        module = co2.O3TensorProduct.from_tensor_product(
            ref, input_basis="spherical", project=False
        )
    finally:
        torch.set_default_dtype(torch.float64)
    assert not module.training and not module.weight.requires_grad
    torch.testing.assert_close(module.weight, ref.weight)
    x = torch.randn(4, o3.Irreps(a).dim, device=DEVICE)
    r = torch.randn(4, 3, device=DEVICE)
    if cartesian:
        cx = co3.ChangeOfBasis(a).to(DEVICE)(x)
        y = co3.CartesianHarmonics(co3.Irreps(b), True, "component").to(DEVICE)(r)
        expected = co3.ChangeOfBasis(c, True).to(DEVICE)(ref(cx, y))
    else:
        expected = ref(x, o3.spherical_harmonics(o3.Irreps(b), r, True, "component"))
    raw = module(x, r)
    actual = co3.ChangeOfBasis(c, True).to(DEVICE)(raw)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    down = co3.Linear(c, "1x2e+1x3e", output_basis="spherical").to(DEVICE)
    torch.testing.assert_close(
        down(raw), down(co3.Projector(c).to(DEVICE)(raw)), atol=2e-12, rtol=2e-12
    )
    for sign in (-1, 1):
        q = sign * o3.rand_matrix(device=DEVICE)
        rotated_x = x @ o3.Irreps(a).D_from_matrix(q.cpu()).to(DEVICE).T
        rotated = co3.ChangeOfBasis(c, True).to(DEVICE)(module(rotated_x, r @ q.T))
        transformed = actual @ o3.Irreps(c).D_from_matrix(q.cpu()).to(DEVICE).T
        torch.testing.assert_close(rotated, transformed, atol=1e-11, rtol=1e-11)


def test_native_compile_and_empty(double_precision):
    module = co2.FullyConnectedTensorProduct("2x1m", "2x2m", "3x1m+3x3m").to(DEVICE)
    x = module.irreps_in1.randn(3, -1, device=DEVICE)
    y = module.irreps_in2.randn(3, -1, device=DEVICE)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(x, y), module(x, y))
    linear = co2.Linear(module.irreps_out, "2x1m+2x3m", output_basis="circular").to(
        DEVICE
    )
    value = module(x, y)
    torch.testing.assert_close(
        torch.compile(linear, backend="aot_eager", fullgraph=True)(value), linear(value)
    )
    elementwise = co2.ElementwiseTensorProduct("1x0o+2x1m", "2x2m+1x0o").to(DEVICE)
    assert elementwise.irreps_out.num_irreps == 4
    for count in (0, 3):
        empty = co2.TensorProduct("", "", "1x0e", []).to(DEVICE)
        x = torch.empty(count, 0, device=DEVICE, requires_grad=True)
        y = torch.empty(1, 0, device=DEVICE, requires_grad=True)
        z = empty(x, y)
        assert z.shape == (count, 1)
        assert not z.count_nonzero()
        assert torch.autograd.grad(z.sum(), x)[0].shape == x.shape
        linear = co2.Linear("2x1m", "1x0e").to(DEVICE)
        assert linear(torch.randn(count, 4, device=DEVICE)).shape == (count, 1)


def test_third_derivative(double_precision):
    a, b, c = "1x3o", "1x2e", "1x3o"
    options = dict(internal_weights=False, shared_weights=False)
    ins = [(0, 0, 0, "uvu", True)]
    module = co2.O3TensorProduct(
        a, b, c, ins, input_basis="spherical", output_basis="spherical", **options
    ).to(DEVICE)
    reference = o3.TensorProduct(a, b, c, ins, **options).to(DEVICE)
    x = torch.randn(2, 7, device=DEVICE, requires_grad=True)
    r = torch.randn(2, 3, device=DEVICE, requires_grad=True)
    w = torch.randn(2, 1, device=DEVICE, requires_grad=True)
    actual = module(x, r, w)
    expected = reference(
        x, o3.spherical_harmonics(o3.Irreps(b), r, True, "component"), w
    )
    for _ in range(3):
        probe = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        (v * probe).sum(),
                        (x, r, w),
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for v in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)


def test_single_precision_and_poles():
    torch.set_default_dtype(torch.float32)
    a, b, c = "2x4e", "1x3o", "2x3o+2x4o+2x5o"
    ins = [(0, 0, i, "uvu", True) for i in range(3)]
    ref = o3.TensorProduct(a, b, c, ins).to(DEVICE)
    module = co2.O3TensorProduct.from_tensor_product(
        ref, input_basis="spherical", output_basis="spherical"
    )
    vectors = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1e-8, 1.0, 1e-8],
        ],
        device=DEVICE,
        requires_grad=True,
    )
    x = torch.randn(5, o3.Irreps(a).dim, device=DEVICE)
    actual = module(x, vectors)
    expected = ref(x, o3.spherical_harmonics(o3.Irreps(b), vectors, True, "component"))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual, expected = [
        torch.autograd.grad(v.square().sum(), vectors, retain_graph=True)[0]
        for v in (actual, expected)
    ]
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)


def test_unweighted_paths_and_validation(double_precision):
    a, b, c = "1x0e+2x1o", "1x0e+1x1o", "1x0e+2x1o+2x0e"
    instructions = [
        (0, 0, 0, "uvu", False),
        (1, 0, 1, "uvu", True),
        (1, 1, 2, "uvu", False),
    ]
    reference = o3.TensorProduct(a, b, c, instructions).to(DEVICE)
    module = co2.O3TensorProduct.from_tensor_product(
        reference, input_basis="spherical", output_basis="spherical"
    )
    x = reference.irreps_in1.randn(3, -1, device=DEVICE)
    r = torch.randn(3, 3, device=DEVICE)
    sh = o3.spherical_harmonics(reference.irreps_in2, r, True, "component")
    torch.testing.assert_close(module(x, r), reference(x, sh), atol=2e-12, rtol=2e-12)
    for mode, weighted in (("uvu", True), ("uvw", False)):
        with pytest.raises(ValueError):
            co2.O3TensorProduct("1x1o", "1x1o", "2x0e", [(0, 0, 0, mode, weighted)])
