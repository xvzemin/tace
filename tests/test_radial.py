"""Spherical Bessel values, derivatives and radial normalization."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch
from scipy.special import spherical_jn

from tace.models.radial import (
    C2PolynomialCutoff,
    C3PolynomialCutoff,
    CosineCutoff,
    MollifierCutoff,
    RadialBasis,
    ZBLBasis,
    compute_jn_zeros,
    j0SphericalBesselBasis,
    jnSphericalBesselBasis,
    spherical_bessel_jn,
    spherical_bessel_parameters,
)

pytestmark = pytest.mark.usefixtures("double_precision")


@pytest.mark.parametrize(
    "cutoff_cls",
    [CosineCutoff, MollifierCutoff, C2PolynomialCutoff, C3PolynomialCutoff],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cutoff_boundary_and_shape(cutoff_cls, dtype, column, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = cutoff_cls(6.0).to(device=device, dtype=dtype)
    cutoff = module.cutoff
    below = torch.nextafter(cutoff, torch.zeros_like(cutoff))
    above = torch.nextafter(cutoff, 2 * cutoff)
    r = torch.stack([cutoff * 0, cutoff / 2, below, cutoff, above, cutoff * 2])
    if column:
        r = r[:, None]
    r.requires_grad_()
    value = module(r)
    assert value.shape == r.shape and value.dtype == dtype
    torch.testing.assert_close(value.flatten()[0], value.new_tensor(1.0))
    assert torch.isfinite(value).all()
    assert bool((value >= 0).all())
    for order in range(7):
        assert torch.isfinite(value).all(), order
        assert torch.equal(value[r >= cutoff], torch.zeros_like(value[r >= cutoff]))
        if order < 6:
            value = torch.autograd.grad(value.sum(), r, create_graph=True)[0]
    empty = torch.empty((0, 1) if column else (0,), device=device, dtype=dtype)
    assert module(empty).shape == empty.shape


@pytest.mark.parametrize(
    "cutoff_cls",
    [CosineCutoff, MollifierCutoff, C2PolynomialCutoff, C3PolynomialCutoff],
)
def test_cutoff_broadcast_and_derivatives(cutoff_cls):
    module = cutoff_cls(6.0)
    cutoff = torch.tensor([[2.0], [3.0], [6.0]], requires_grad=True)
    r = torch.tensor([[0.3], [2.0], [5.7]], requires_grad=True)
    actual = module(r, cutoff)
    assert actual.shape == r.shape
    expected = torch.stack([module(r[i], cutoff[i]) for i in range(3)])
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(module, (r, cutoff))
    assert torch.autograd.gradgradcheck(module, (r, cutoff))
    _, tangent = torch.func.jvp(
        module, (r, cutoff), (torch.ones_like(r), torch.zeros_like(cutoff))
    )
    expected_tangent = torch.autograd.grad(actual.sum(), r)[0]
    torch.testing.assert_close(tangent, expected_tangent)


@pytest.mark.parametrize("cutoff_cls", [C2PolynomialCutoff, C3PolynomialCutoff])
@pytest.mark.parametrize("p", [1, 2, 5, 6, 50])
def test_polynomial_cutoff_legacy_agreement(cutoff_cls, p):
    module = cutoff_cls(6.0, p)
    r = torch.linspace(0.01, 5.7, 19, requires_grad=True)
    actual = module(r)
    expected = module.legacy_forward(r)
    torch.testing.assert_close(actual, module.calculate_envelope(r, module.cutoff, p))
    for order in range(4):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        if order < 3:
            actual = torch.autograd.grad(actual.sum(), r, create_graph=True)[0]
            expected = torch.autograd.grad(expected.sum(), r, create_graph=True)[0]


@pytest.mark.parametrize(
    "cutoff_cls,order", [(C2PolynomialCutoff, 3), (C3PolynomialCutoff, 4)]
)
@pytest.mark.parametrize("p", [5, 6, 50])
def test_polynomial_cutoff_near_boundary(cutoff_cls, order, p):
    module = cutoff_cls(6.0, p)
    r = torch.tensor([6.0 - 1e-3, 6.0 - 1e-6, 6.0 - 1e-8])
    with localcontext() as context:
        context.prec = 80
        expected = []
        for distance in r.tolist():
            x = Decimal.from_float(distance) / 6
            polynomial = sum(
                Decimal(math.comb(k + order - 1, order - 1)) * x**k for k in range(p)
            )
            expected.append(float((1 - x) ** order * polynomial))
    actual = module(r)
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=2e-13, atol=0)
    assert bool((actual > 0).all())


@pytest.mark.parametrize("cutoff_name", ["cosine", "mollifier", "c2poly", "c3poly"])
def test_zbl_cutoff_boundary(cutoff_name):
    module = ZBLBasis(cutoff_name)
    cutoff = 2 * module.covalent_radii[1]
    # Edges can be outside the element-dependent ZBL cutoff while still being
    # inside the model's graph cutoff.
    r = (cutoff + torch.tensor([0.0, 1e-8, 0.01])).requires_grad_()
    edges = torch.arange(6).reshape(3, 2)
    edge_index = torch.stack((edges.flatten(), edges.flip(1).flatten()))
    energy = module(
        r.repeat_interleave(2)[:, None],
        torch.zeros(6, dtype=torch.long),
        edge_index,
        torch.tensor([1]),
    ).sum()
    assert float(energy.detach()) == 0.0
    for order in range(6):
        energy = torch.autograd.grad(energy.sum(), r, create_graph=True)[0]
        assert torch.equal(energy, torch.zeros_like(energy)), order


@pytest.mark.parametrize(
    "cutoff_cls",
    [CosineCutoff, MollifierCutoff, C2PolynomialCutoff, C3PolynomialCutoff],
)
def test_cutoff_compilation(cutoff_cls):
    module = cutoff_cls(6.0)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for count in (4, 7, 0):
        r = torch.linspace(0.0, 6.01, count).unsqueeze(-1).requires_grad_()
        actual = compiled(r)
        expected = module(r)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), r)[0],
            torch.autograd.grad(expected.sum(), r)[0],
        )


@pytest.mark.parametrize("cutoff_cls", [C2PolynomialCutoff, C3PolynomialCutoff])
@pytest.mark.parametrize("p", [2, 5, 50])
def test_polynomial_cutoff_functional_capture(cutoff_cls, p):
    from torch.fx.experimental.proxy_tensor import make_fx

    module = cutoff_cls(6.0, p)

    def evaluate(r, cutoff):
        return torch.func.functional_call(module, {"cutoff": cutoff}, (r,))

    r = torch.linspace(0.1, 6.01, 8)
    captured = make_fx(evaluate, tracing_mode="symbolic")(r, module.cutoff)
    for cutoff in (module.cutoff, module.cutoff + 1):
        torch.testing.assert_close(captured(r, cutoff), module(r, cutoff))


@pytest.mark.parametrize("cutoff_cls", [C2PolynomialCutoff, C3PolynomialCutoff])
def test_polynomial_cutoff_cuda_graph(cutoff_cls):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = cutoff_cls(6.0).cuda()
    r = torch.tensor([0.0, 2.0, 5.99999999, 6.0, 6.1], device="cuda")
    expected = module(r)
    capture = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(capture):
        actual = module(r)
    capture.replay()
    torch.testing.assert_close(actual, expected)


def reference_derivative(n, x, derivative):
    if derivative == 0:
        return spherical_jn(n, np.abs(x)) * np.where(x < 0, (-1) ** n, 1)
    if n == 0:
        return -reference_derivative(1, x, derivative - 1)
    return (
        n * reference_derivative(n - 1, x, derivative - 1)
        - (n + 1) * reference_derivative(n + 1, x, derivative - 1)
    ) / (2 * n + 1)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 9, 10, 16, 32, 64, 128])
def test_spherical_bessel_values_and_derivatives(n):
    boundaries = [
        boundary
        for derivative in range(4)
        for boundary in (
            min(max(n + derivative, 1), math.sqrt(8 * n + 12)),
            max(n + derivative, 1),
        )
    ]
    positive = torch.cat(
        (
            torch.logspace(-12, math.log10(max(100, 3 * n)), 180),
            torch.tensor(boundaries)[:, None]
            .mul(torch.tensor([1 - 1e-8, 1.0, 1 + 1e-8]))
            .flatten(),
            # Include zeros of j0, used to normalize the old downward recurrence.
            torch.arange(1, 10) * math.pi,
        )
    )
    x = torch.cat((-positive, torch.zeros(1), positive)).requires_grad_()
    value = spherical_bessel_jn(n, x)
    for derivative in range(4):
        expected = torch.from_numpy(
            reference_derivative(n, x.detach().numpy(), derivative)
        )
        torch.testing.assert_close(value, expected, atol=2e-13, rtol=2e-11)
        if derivative == 0:
            mask = (x.abs() <= max(n, 1)) & (expected.abs() > 1e-250)
            torch.testing.assert_close(value[mask], expected[mask], atol=0, rtol=2e-11)
        if derivative < 3:
            value = torch.autograd.grad(value.sum(), x, create_graph=True)[0]


@pytest.mark.parametrize("n", [0, 1, 5, 10, 64, 128])
def test_spherical_bessel_zeros_and_normalization(n):
    roots = compute_jn_zeros(n, 5)
    assert roots.shape == (5,)
    assert roots[0] > n
    assert np.all(np.diff(roots) > 3.0)
    np.testing.assert_allclose(spherical_jn(n, roots), 0.0, atol=3e-15)
    if n == 0:
        np.testing.assert_array_equal(roots, np.arange(1, 6) * math.pi)
    if n == 1:
        np.testing.assert_allclose(
            roots[:3], [4.493409457909064, 7.725251836937707, 10.904121659428899]
        )
    # Interlacing checks that the first roots were not skipped.
    next_roots = compute_jn_zeros(n + 1, 4)
    assert np.all(roots[:-1] < next_roots)
    assert np.all(next_roots < roots[1:])

    basis = jnSphericalBesselBasis(cutoff=5.0, order=n, num_basis=5)
    points, weights = np.polynomial.legendre.leggauss(256)
    r = torch.tensor((points + 1) * 2.5).unsqueeze(-1)
    values = basis(r, None, None)
    integral = values.T @ (values * r.square() * torch.tensor(weights * 2.5)[:, None])
    torch.testing.assert_close(integral, torch.eye(5), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        basis(torch.tensor([[5.0]]), None, None), torch.zeros(1, 5), atol=2e-13, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("trainable", [False, True])
def test_spherical_bessel_basis_layout_and_state(dtype, trainable):
    basis = jnSphericalBesselBasis(
        order=[2, 0, 10], num_basis=[3, 2, 1], trainable=trainable
    ).to(dtype)
    r = torch.tensor([[0.0], [1e-10], [0.2], [3.0]], dtype=dtype, requires_grad=True)
    value = basis(r, None, None)
    argument = (
        basis.zeros.double().numpy(force=True)
        * r.double().numpy(force=True)
        / basis.cutoff.item()
    )
    expected = np.concatenate(
        [
            spherical_jn(n, x)
            for n, x in zip(basis.order, np.split(argument, [3, 5], axis=-1))
        ],
        axis=-1,
    ) * basis.normalizer.double().numpy(force=True)
    torch.testing.assert_close(value, torch.tensor(expected, dtype=dtype))
    assert value.shape == (4, 6) and value.dtype == dtype
    assert basis(torch.empty(0, 1, dtype=dtype), None, None).shape == (0, 6)
    assert set(basis.state_dict()) == {"zeros", "normalizer", "cutoff"}
    clone = jnSphericalBesselBasis(
        order=[2, 0, 10], num_basis=[3, 2, 1], trainable=trainable
    ).to(dtype)
    clone.load_state_dict(basis.state_dict(), strict=True)
    torch.testing.assert_close(value, clone(r, None, None), atol=0, rtol=0)
    first = torch.autograd.grad(value.sum(), r, create_graph=True)[0]
    parameters = (r, basis.zeros) if trainable else (r,)
    for derivative in torch.autograd.grad(first.square().sum(), parameters):
        assert torch.isfinite(derivative).all()


def test_spherical_bessel_trainable_gradients():
    basis = jnSphericalBesselBasis(
        order=[0, 1, 12], num_basis=[1, 1, 1], trainable=True
    )
    r = torch.tensor([[0.0], [0.2], [3.7]], requires_grad=True)

    def evaluate(r, zeros):
        return torch.func.functional_call(basis, {"zeros": zeros}, (r, None, None))

    assert torch.autograd.gradcheck(evaluate, (r, basis.zeros))
    assert torch.autograd.gradgradcheck(evaluate, (r, basis.zeros))
    # Trainable frequencies may pass through zero or change sign.
    with torch.no_grad():
        basis.zeros.copy_(torch.tensor([[0.0, -2.0, -12.0]]))
    assert torch.autograd.gradcheck(evaluate, (r, basis.zeros))
    assert torch.autograd.gradgradcheck(evaluate, (r, basis.zeros))


def test_spherical_bessel_precomputed_parameters():
    spherical_bessel_parameters.cache_clear()
    basis = jnSphericalBesselBasis(order=[1, 3, 12], num_basis=[2, 2, 2])
    cache = spherical_bessel_parameters.cache_info()
    assert cache.misses == 9
    r = torch.tensor([[0.0], [0.2], [3.0]], requires_grad=True)
    value = basis(r, None, None)
    for _ in range(2):
        value = torch.autograd.grad(value.sum(), r, create_graph=True)[0]
        assert torch.isfinite(value).all()
    assert spherical_bessel_parameters.cache_info() == cache
    # Derivatives beyond force training remain available through the shared cache.
    value = torch.autograd.grad(value.sum(), r, create_graph=True)[0]
    assert torch.isfinite(value).all()
    assert spherical_bessel_parameters.cache_info().misses == 12


@pytest.mark.parametrize("name", ["j0", "j1", "j3", "j12"])
def test_spherical_bessel_compilation(name):
    basis = RadialBasis(radial_basis=name, num_basis=4)
    compiled = torch.compile(basis, backend="aot_eager", fullgraph=True, dynamic=True)
    for size in (4, 7, 0):
        r = torch.linspace(0, 5.9, size).unsqueeze(-1).requires_grad_()
        args = (r, torch.empty(0), torch.empty(2, 0, dtype=torch.long), torch.empty(0))
        actual, _ = compiled(*args)
        expected, _ = basis(*args)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual.sum(), r)[0]
        expected_grad = torch.autograd.grad(expected.sum(), r)[0]
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("name", ["j0", "bessel", "j1", "j3", "j12"])
def test_spherical_bessel_selection(name):
    basis = RadialBasis(radial_basis=name, num_basis=4, apply_cutoff=False)
    if name in ("j0", "bessel"):
        assert isinstance(basis.radial_fn, j0SphericalBesselBasis)
        reference = j0SphericalBesselBasis(num_basis=4)
    else:
        assert isinstance(basis.radial_fn, jnSphericalBesselBasis)
        reference = jnSphericalBesselBasis(order=int(name[1:]), num_basis=4)
        assert basis.radial_fn.order == [int(name[1:])]
    r = torch.tensor([[0.0], [0.1], [2.0], [5.9]], requires_grad=True)
    actual, _ = basis(r, None, None, None)
    expected = reference(r, None, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (4, 4)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), r)[0],
        torch.autograd.grad(expected.sum(), r)[0],
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("n", [0, 1, 3, 12])
def test_spherical_bessel_differentiation(n):
    x = torch.tensor([0.0, 0.01, 1.0, 4.0, 12.0, 30.0], requires_grad=True)
    value = spherical_bessel_jn(n, x)
    for derivative in range(1, 7):
        value = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
        expected = torch.from_numpy(
            reference_derivative(n, x.detach().numpy(), derivative)
        )
        torch.testing.assert_close(value, expected, atol=2e-13, rtol=2e-11)
    tangent = torch.linspace(0.1, 1.0, x.numel())
    _, actual = torch.func.jvp(lambda x: spherical_bessel_jn(n, x), (x,), (tangent,))
    expected = (
        torch.from_numpy(reference_derivative(n, x.detach().numpy(), 1)) * tangent
    )
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-11)
    torch.testing.assert_close(
        torch.vmap(lambda x: spherical_bessel_jn(n, x))(x), spherical_bessel_jn(n, x)
    )


@pytest.mark.parametrize("name", ["jn", "j-1", "j1.5", "jx"])
def test_spherical_bessel_invalid_name(name):
    with pytest.raises(ValueError, match="Unknown radial_basis"):
        RadialBasis(radial_basis=name)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": -1},
        {"order": []},
        {"num_basis": 0},
        {"num_basis": []},
        {"order": [0, 1], "num_basis": [3]},
        {"cutoff": 0.0},
        {"cutoff": float("inf")},
    ],
)
def test_spherical_bessel_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        jnSphericalBesselBasis(**kwargs)


@pytest.mark.parametrize("counts", [None, [3, 1, 2]])
def test_physical_basis(counts, double_precision):
    reference = pytest.importorskip("physical_basis").PhysicalBasis()
    from tace.models.physical_basis import PhysicalRadialBasis

    radial = RadialBasis(
        cutoff=6.0,
        radial_basis="physical",
        num_basis=counts,
        lmax=2,
        max_eigenvalue=25 if counts is None else 0.0,
        cutoff_fn="c2poly",
        apply_cutoff=counts is not None,
    )
    mod = radial.radial_fn
    assert mod.widths == tuple(counts or [2, 1, 1])
    distance = torch.tensor([[1.4], [1.4], [20.0]], requires_grad=True)
    edges = torch.tensor([[0, 1, 0], [1, 0, 1]])
    z = torch.tensor([6, 8])
    y = mod(distance, z, edges)
    features, cutoff = radial(distance, torch.tensor([0, 1]), edges, z)
    if radial.apply_cutoff:
        assert cutoff is None
        torch.testing.assert_close(features, y * radial.cutoff_fn(distance))
    else:
        torch.testing.assert_close(features, y)
        torch.testing.assert_close(cutoff, radial.cutoff_fn(distance))
    x = np.array([1.4 / 1.094])
    modes = [(n, l) for l, count in enumerate(mod.widths) for n in range(count)]
    scale = (6**3 / 3) ** 0.5
    expected = np.array([reference.compute(n, l, x)[0] for n, l in modes]) * scale
    np.testing.assert_allclose(y[0].detach(), expected, rtol=2e-7, atol=2e-7)
    torch.testing.assert_close(y[0], y[1])
    assert y[2].count_nonzero() == 0
    grad = torch.autograd.grad(y.sum(), distance, create_graph=True)[0]
    derivative = sum(reference.compute_derivative(n, l, x)[0] for n, l in modes)
    assert grad[0].item() == pytest.approx(derivative * scale / 1.094, abs=2e-5)
    assert torch.autograd.grad(grad.sum(), distance)[0].isfinite().all()
    with pytest.raises(ValueError, match="list"):
        PhysicalRadialBasis(6.0, 2, 8)
    with pytest.raises(ValueError, match="cover lmax"):
        PhysicalRadialBasis(6.0, 3, None, 25)


def test_pt_physical_basis():
    from scipy.special import betaln, eval_jacobi

    from tace.models.physical_basis import PtPhysicalRadialBasis

    basis = PtPhysicalRadialBasis(5.0, 3)
    assert basis.widths == (3, 2, 1, 1)
    assert PtPhysicalRadialBasis(5.0, 5).widths[-1] == 1
    assert PtPhysicalRadialBasis(5.0, 0, max_eigenvalue=35.0).widths == (2,)
    assert PtPhysicalRadialBasis(5.0, 1, [1, 3], 0.0).widths == (1, 3)
    with pytest.raises(ValueError, match="list"):
        PtPhysicalRadialBasis(5.0, 1, 8)
    with pytest.raises(ValueError, match="cover lmax"):
        PtPhysicalRadialBasis(5.0, 3, max_eigenvalue=35)

    r = torch.tensor([0.0, 0.5, 1.4, 4.8], requires_grad=True)
    z = torch.tensor([6, 8])
    edges = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    values = basis(r, z, edges)
    x = r.detach().numpy() / 1.094
    expected = []
    for l, count in enumerate(basis.widths):
        b = (1 + math.sqrt(4 * l * (l + 1) + 25)) / 2
        alpha, beta = l + 0.5, b - 0.5
        for n in range(count):
            norm = math.sqrt(
                2
                * 5**3
                / 3
                * (2 * n + alpha + beta + 1)
                * math.exp(
                    betaln(n + 1, alpha + beta + n + 1)
                    - betaln(n + alpha + 1, n + beta + 1)
                )
            )
            expected.append(
                norm
                * x**l
                / (1 + x * x) ** ((l + b + 2) / 2)
                * eval_jacobi(n, alpha, beta, (1 - x * x) / (1 + x * x))
            )
    np.testing.assert_allclose(values.detach(), np.stack(expected, axis=1), atol=1e-12)
    assert torch.autograd.gradcheck(lambda d: basis(d, z, edges), r)
    assert torch.autograd.gradgradcheck(lambda d: basis(d, z, edges), r)


@pytest.mark.parametrize("version", ["mlp_v1", "mlp_v2"])
@pytest.mark.parametrize("hidden", [[], [4, 4]])
def test_batch_mlp_matches_independent_paths(version, hidden, double_precision):
    from types import SimpleNamespace

    from e3nn import o3

    from tace.models.mlp import BATCH_MLP

    product = SimpleNamespace(
        irreps_in1=o3.Irreps("2x0e"),
        irreps_in2=o3.Irreps("0e+1o+2e"),
        instructions=[(0, l, i, "uvu", True) for i, l in enumerate([1, 0, 1, 2])],
        weight_numel=8,
    )
    net = BATCH_MLP[version].from_product(
        product, [2, 1, 1], hidden, context_features=2, bias=True, rms_norm=True
    )
    x = torch.randn(5, 4, requires_grad=True)
    context = torch.randn(3, 2, requires_grad=True)
    ids = torch.tensor([1, 0, 1, 2, 0])
    actual = net(((x, None), (context, ids)))
    a, b, c = [
        m(torch.cat((v, context[ids]), 1))
        for m, v in zip(net.branches, x.split([2, 1, 1], 1))
    ]
    expected = torch.cat((b[:, :2], a, b[:, 2:], c), 1)
    inputs = (x, context, *net.parameters())

    def derivatives(y):
        grads = torch.autograd.grad(y.square().sum(), inputs, create_graph=True)
        return grads, torch.autograd.grad(grads[0].sum(), inputs, allow_unused=True)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(derivatives(actual), derivatives(expected))
