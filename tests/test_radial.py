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


@pytest.mark.parametrize("interpolate", [False, True])
def test_physical_radial_basis(interpolate):
    from physical_basis import PhysicalBasis

    from tace.models.physical_basis import PhysicalRadialBasis

    module = PhysicalRadialBasis(6.0, 3, use_interpolation=interpolate)
    assert module.widths == (8, 7, 6, 6)
    z, edges = torch.tensor([6, 8]), torch.tensor([[0, 1, 0], [1, 0, 1]])
    r = torch.tensor([1.4, 1.4, 20.0], requires_grad=True)
    y = module(r, z, edges)
    scale = (6**3 / 3) ** 0.5
    official = PhysicalBasis()
    modes = [(n, l) for l, width in enumerate(module.widths) for n in range(width)]
    x = np.array([1.4 / 1.094])
    expected = np.array([official.compute(n, l, x)[0] for n, l in modes]) * scale
    slope = (
        sum(official.compute_derivative(n, l, x)[0] for n, l in modes) * scale / 1.094
    )
    np.testing.assert_allclose(
        y[0].detach(), expected, atol=2e-7 if interpolate else 1e-12, rtol=0
    )
    torch.testing.assert_close(y[0], y[1])
    other = PhysicalRadialBasis(6.0, 3, use_interpolation=not interpolate)
    other.load_state_dict(module.state_dict(), strict=True)
    torch.testing.assert_close(other(r, z, edges), y, rtol=0, atol=2e-7)
    assert not y[2].count_nonzero()
    grad = torch.autograd.grad(y.sum(), r, create_graph=True)[0]
    assert grad[0].item() == pytest.approx(
        slope, abs=2e-5 if interpolate else 1e-12, rel=0
    )
    assert torch.autograd.grad(grad.sum(), r)[0].isfinite().all()
    assert PhysicalRadialBasis(4.0, 0, [3])(r, z, edges).shape == (3, 3)


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


@pytest.mark.parametrize("name", ["j0", "j1", "j3", "j12", "jn"])
def test_spherical_bessel_compilation(name):
    torch.compiler.reset()
    basis = RadialBasis(
        radial_basis=name, lmax=2, num_basis=[3, 2, 1] if name == "jn" else 4
    )
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


@pytest.mark.parametrize("counts", [[3], [4, 3, 2, 1]])
def test_degree_bessel_basis(counts):
    cutoff = 6.0
    basis = RadialBasis(
        radial_basis="jn",
        lmax=len(counts) - 1,
        num_basis=counts,
        cutoff=cutoff,
        apply_cutoff=False,
    )
    r = torch.tensor([[0.0], [1e-8], [1.23], [5.999], [cutoff]], requires_grad=True)
    actual, _ = basis(r, None, None, None)
    assert basis.widths == tuple(counts)
    assert actual.shape == (len(r), sum(counts))
    roots = [compute_jn_zeros(degree, n) for degree, n in enumerate(counts)]
    x = r.detach().numpy()
    probe = torch.arange(1, actual.shape[1] + 1, dtype=r.dtype)
    for d in range(3):
        expected = np.concatenate(
            [
                np.sqrt(2 / cutoff**3)
                / np.abs(spherical_jn(degree + 1, z))
                * (z / cutoff) ** d
                * reference_derivative(degree, x * z / cutoff, d)
                for degree, z in enumerate(roots)
            ],
            axis=1,
        )
        ref = torch.from_numpy(expected)
        if d:
            ref = (ref * probe).sum(dim=1, keepdim=True)
        torch.testing.assert_close(actual, ref, rtol=1e-10, atol=1e-12)
        if d < 2:
            value = (actual * probe).sum() if d == 0 else actual.sum()
            actual = torch.autograd.grad(value, r, create_graph=True)[0]


@pytest.mark.parametrize(
    "counts", [2, [], [1], [1, 2, 3], [1, 0], [True, 1], [1.5, 2]]
)
def test_degree_bessel_counts(counts):
    with pytest.raises(ValueError, match="num_radial_basis"):
        RadialBasis(radial_basis="jn", lmax=1, num_basis=counts)


def test_radial_spectral_defaults():
    from physical_basis import PhysicalBasis

    threshold = (8 * math.pi / 6) ** 2
    def basis(name, cutoff, limit=None):
        return RadialBasis(cutoff=cutoff, radial_basis=name, lmax=3, max_eigenvalue=limit)

    for cutoff in (5.0, 6.0, 8.0):
        widths = basis("jn", cutoff).widths
        assert widths == (8, 7, 7, 6)
        assert [basis(f"j{i}", cutoff).num_basis for i in range(4)] == list(widths)
        assert basis("bessel", cutoff).num_basis == 8
        explicit = basis("jn", cutoff, threshold).widths
        assert explicit == {5: (6, 6, 5, 5), 6: widths, 8: (10, 10, 9, 9)}[cutoff]
        assert basis("j0", cutoff, threshold).num_basis == explicit[0]
        assert basis("physical_basis", cutoff).widths == (8, 7, 6, 6)
        assert basis("physical_basis", cutoff, 36.0).widths == (3, 2, 1, 1)

    assert basis("j0", 4.76).num_basis == 8

    # Include a mode exactly at the threshold, but not below it.
    physical_limit = PhysicalBasis().return_eigenvalue(7, 0)
    for name, limit, n in [
        ("j0", threshold, 8),
        ("j2", (compute_jn_zeros(2, 3)[-1] / 6) ** 2, 3),
        ("physical_basis", physical_limit, 8),
    ]:
        for value, expected in [(limit, n), (math.nextafter(limit, 0.0), n - 1)]:
            selected = RadialBasis(cutoff=6, radial_basis=name, max_eigenvalue=value)
            assert selected.num_basis == expected


@pytest.mark.parametrize(
    "name,counts,limit,match",
    [
        ("j0", 8, 20.0, "only one"),
        ("jn", [2, 1], 20.0, "only one"),
        ("physical_basis", [2, 1], 36.0, "only one"),
        ("j0", None, float("nan"), "finite positive"),
        ("jn", None, True, "finite positive"),
        ("physical_basis", None, 0, "finite positive"),
        ("jn", None, 0.4, "empty"),
        ("physical_basis", None, 8.0, "empty"),
        ("gaussian", None, 20, "eigenvalue"),
    ],
)
def test_radial_spectral_errors(name, counts, limit, match):
    with pytest.raises(ValueError, match=match):
        RadialBasis(radial_basis=name, lmax=1, num_basis=counts, max_eigenvalue=limit)


@pytest.mark.parametrize("name", ["j-1", "j1.5", "jx"])
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
