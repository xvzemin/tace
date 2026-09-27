"""Cartesian operators and spherical basis equivalence."""

import pytest
import torch
from e3nn import nn, o3

from eqx import co3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_irreps_and_basis(double_precision):
    irreps = co3.Irreps("2x0e+1x1o+2x2e+1x2o+1x1o")
    assert irreps.dim == 2 + 3 + 27 + 3
    assert irreps.regroup() == co3.Irreps("2x0e+2x1o+1x2o+2x2e")
    assert co3.Irreps([co3.Irrep("1o"), co3.Irrep("2e")]).dim == 12
    assert irreps.filter(lmax=1).dim == 8
    with pytest.raises(ValueError):
        irreps.index("0o")
    to_cart = co3.ChangeOfBasis(irreps).to(DEVICE)
    to_sph = co3.ChangeOfBasis(irreps, inverse=True).to(DEVICE)
    x = torch.randn(4, irreps.spherical().dim, device=DEVICE)
    torch.testing.assert_close(to_sph(to_cart(x)), x, atol=2e-14, rtol=2e-14)
    for sign in (1, -1):
        matrix = sign * o3.rand_matrix(device=DEVICE)
        expected = to_cart(
            x @ irreps.spherical().D_from_matrix(matrix.cpu()).to(DEVICE).T
        )
        actual = to_cart(x) @ irreps.D_from_matrix(matrix).T
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    raw = torch.randn(4, irreps.dim, device=DEVICE)
    project = co3.Projector(irreps).to(DEVICE)
    torch.testing.assert_close(project(raw), to_cart(to_sph(raw)))
    torch.testing.assert_close(project(project(raw)), project(raw))
    assert irreps.randn(0, -1, device=DEVICE).shape == (0, irreps.dim)


@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
@pytest.mark.parametrize("normalize", [False, True])
def test_harmonics(double_precision, normalization, normalize):
    irreps = co3.Irreps.spherical_harmonics(6)
    module = co3.CartesianHarmonics(irreps, normalize, normalization).to(DEVICE)
    inverse = co3.ChangeOfBasis(irreps, inverse=True).to(DEVICE)
    x = torch.randn(8, 3, device=DEVICE, requires_grad=True) * 0.5
    actual = inverse(module(x))
    expected = o3.spherical_harmonics(list(range(7)), x, normalize, normalization)
    torch.testing.assert_close(actual, expected, atol=5e-13, rtol=5e-13)
    for _ in range(2):
        seed = torch.randn_like(actual)
        actual, expected = [
            torch.autograd.grad((v * seed).sum(), x, create_graph=True)[0]
            for v in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    assert module(x[:0]).shape == (0, irreps.dim)
    if not normalize:
        zero = torch.zeros(1, 3, device=DEVICE, requires_grad=True)
        result = module(zero)
        assert torch.isfinite(torch.autograd.grad(result.sum(), zero)[0]).all()


@pytest.mark.parametrize("normalization", ["element", "path"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize(
    "instructions", [None, [(1, 0), (0, 0), (3, 2), (2, 1), (0, 0)]]
)
def test_linear(double_precision, normalization, shared, instructions):
    kwargs = dict(
        path_normalization=normalization,
        internal_weights=False,
        shared_weights=shared,
        biases=shared,
        instructions=instructions,
    )
    ref = o3.Linear("2x0e+1x0e+2x1o+1x2e", "3x0e+2x1o+2x2e+1x0o", **kwargs).to(DEVICE)
    module = co3.Linear(
        ref.irreps_in, ref.irreps_out, output_basis="spherical", **kwargs
    ).to(DEVICE)
    x = torch.randn(5, ref.irreps_in.dim, device=DEVICE)
    shape = () if shared else (5,)
    weights = torch.randn(*shape, ref.weight_numel, device=DEVICE)
    bias = torch.randn(*shape, ref.bias_numel, device=DEVICE)
    cx = co3.ChangeOfBasis(ref.irreps_in).to(DEVICE)(x)
    torch.testing.assert_close(
        module(cx, weights, bias), ref(x, weights, bias), atol=2e-14, rtol=2e-14
    )
    for a, b in zip(module.weight_views(weights), ref.weight_views(weights)):
        torch.testing.assert_close(a, b)
    raw = torch.randn_like(cx)
    inverse = co3.ChangeOfBasis(ref.irreps_in, inverse=True).to(DEVICE)
    torch.testing.assert_close(
        module(raw, weights, bias),
        ref(inverse(raw), weights, bias),
        atol=2e-14,
        rtol=2e-14,
    )
    inputs = [x.requires_grad_() for x in (raw, weights, bias)]
    losses = [
        module(*inputs).square().sum(),
        ref(inverse(inputs[0]), inputs[1], inputs[2]).square().sum(),
    ]
    differentiable = [x for x in inputs if x.numel()]
    for _ in range(2):
        gradients = [
            torch.autograd.grad(loss, differentiable, create_graph=True)
            for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)
        tangents = [torch.randn_like(x) for x in differentiable]
        losses = [sum((g * t).sum() for g, t in zip(gs, tangents)) for gs in gradients]


@pytest.mark.parametrize(
    "mode,multiplicities",
    [
        ("uvu", (2, 3, 2)),
        ("uvv", (2, 3, 3)),
        ("uuu", (2, 2, 2)),
        ("uuw", (2, 2, 3)),
        ("uvw", (2, 3, 4)),
        ("uvuv", (2, 3, 6)),
    ],
)
@pytest.mark.parametrize("normalization", ["component", "norm", "none"])
@pytest.mark.parametrize("path_normalization", ["element", "path", "none"])
def test_tensor_product(
    double_precision, mode, multiplicities, normalization, path_normalization
):
    u, v, w = multiplicities
    a, b, c = f"{u}x1o+{u}x2e", f"{v}x1o", f"{w}x1e+{w}x2e+{w}x1o+{w}x3o"
    instructions = [
        (0, 0, 0, mode, True),
        (0, 0, 1, mode, True),
        (1, 0, 2, mode, True, 0.7),
        (1, 0, 3, mode, True),
        (1, 0, 2, mode, True, 1.3),
    ]
    kwargs = dict(
        internal_weights=False,
        shared_weights=False,
        irrep_normalization=normalization,
        path_normalization=path_normalization,
        in1_var=[0.8, 1.3],
        in2_var=[1.2],
        out_var=[0.5, 1.0, 0.7, 0.9],
    )
    ref = o3.TensorProduct(a, b, c, instructions, **kwargs).to(DEVICE)
    module = co3.TensorProduct(a, b, c, instructions, **kwargs).to(DEVICE)
    x = torch.randn(2, ref.irreps_in1.dim, device=DEVICE, requires_grad=True)
    y = torch.randn(2, ref.irreps_in2.dim, device=DEVICE, requires_grad=True)
    weights = torch.randn(2, ref.weight_numel, device=DEVICE, requires_grad=True)
    transforms = [
        co3.ChangeOfBasis(ir, inverse=inverse).to(DEVICE)
        for ir, inverse in ((a, False), (b, False), (c, True))
    ]
    actual = transforms[2](module(transforms[0](x), transforms[1](y), weights))
    expected = ref(x, y, weights)
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    for _ in range(2):
        seed = torch.randn_like(actual)
        actual, expected = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(
                        (val * seed).sum(),
                        (x, y, weights),
                        create_graph=True,
                        retain_graph=True,
                    )
                ]
            )
            for val in (actual, expected)
        ]
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    assert module(transforms[0](x[:0]), transforms[1](y[:0]), weights[:0]).shape == (
        0,
        module.irreps_out.dim,
    )


def test_gate_and_elementwise(double_precision):
    args = (
        "2x0e+1x0o",
        [torch.nn.functional.silu, torch.tanh],
        "2x0e+2x0o",
        [torch.sigmoid, torch.tanh],
        "2x1o+1x2e+1x2o",
    )
    ref = nn.Gate(*args).to(DEVICE)
    module = co3.Gate(*args).to(DEVICE)
    assert module.irreps_in.spherical() == ref.irreps_in
    assert module.irreps_out.spherical() == ref.irreps_out
    x = torch.randn(8, ref.irreps_in.dim, device=DEVICE, requires_grad=True)
    actual = co3.ChangeOfBasis(module.irreps_out, True).to(DEVICE)(
        module(co3.ChangeOfBasis(ref.irreps_in).to(DEVICE)(x))
    )
    torch.testing.assert_close(actual, ref(x), atol=3e-14, rtol=3e-14)
    with pytest.raises(ValueError):
        co3.Gate("", [], "1x0o", [torch.sigmoid], "1x1e")


def test_high_degree_and_compile(double_precision):
    assert co3.path_normalization(0, 2000, 2000) == pytest.approx((4001) ** 0.5)
    kwargs = dict(internal_weights=False, shared_weights=False)
    module = co3.TensorProduct(
        "1x4e", "1x3o", "1x4o", [(0, 0, 0, "uuu", True)], **kwargs
    ).to(DEVICE)
    ref = o3.TensorProduct(
        "1x4e", "1x3o", "1x4o", [(0, 0, 0, "uuu", True)], **kwargs
    ).to(DEVICE)
    x = torch.randn(2, 9, device=DEVICE)
    y = torch.randn(2, 7, device=DEVICE)
    w = torch.randn(2, 1, device=DEVICE)
    cx = co3.ChangeOfBasis("1x4e").to(DEVICE)(x)
    cy = co3.ChangeOfBasis("1x3o").to(DEVICE)(y)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    actual = co3.ChangeOfBasis("1x4o", True).to(DEVICE)(compiled(cx, cy, w))
    torch.testing.assert_close(actual, ref(x, y, w), atol=1e-13, rtol=1e-13)


def test_all_angular_paths(double_precision):
    for l1 in range(5):
        for l2 in range(5):
            a, b = f"1x{l1}e", f"1x{l2}o"
            x = torch.randn(3, 2 * l1 + 1, device=DEVICE)
            y = torch.randn(3, 2 * l2 + 1, device=DEVICE)
            cx = co3.ChangeOfBasis(a).to(DEVICE)(x)
            cy = co3.ChangeOfBasis(b).to(DEVICE)(y)
            for l3 in range(abs(l1 - l2), min(l1 + l2, 6) + 1):
                out = f"1x{l3}o"
                tp = co3.TensorProduct(a, b, out, [(0, 0, 0, "uuu", False)]).to(DEVICE)
                actual = co3.ChangeOfBasis(out, True).to(DEVICE)(tp(cx, cy))
                cg = o3.wigner_3j(l1, l2, l3, dtype=torch.float64, device=DEVICE)
                expected = torch.einsum("bi,bj,ijk->bk", x, y, cg) * (2 * l3 + 1) ** 0.5
                torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)


def test_broadcast_weights_and_disconnected_paths(double_precision):
    kwargs = dict(internal_weights=False, shared_weights=False)
    a, b, c = "2x1e", "2x2o", "2x1o+2x2o"
    ins = [(0, 0, 0, "uuu", True), (0, 0, 1, "uuu", True)]
    tp = co3.TensorProduct(a, b, c, ins, **kwargs).to(DEVICE)
    ref = o3.TensorProduct(a, b, c, ins, **kwargs).to(DEVICE)
    x = torch.randn(o3.Irreps(a).dim, 3, device=DEVICE).T
    y = torch.randn(1, o3.Irreps(b).dim, device=DEVICE)
    w = torch.randn(1, tp.weight_numel, device=DEVICE)
    result = tp(
        co3.ChangeOfBasis(a).to(DEVICE)(x), co3.ChangeOfBasis(b).to(DEVICE)(y), w
    )
    torch.testing.assert_close(
        co3.ChangeOfBasis(c, True).to(DEVICE)(result), ref(x, y, w)
    )
    for count in (0, 3):
        linear = co3.Linear("0x0e+2x1e", "1x0e+0x1e").to(DEVICE)
        feats = torch.randn(
            count, linear.irreps_in.dim, device=DEVICE, requires_grad=True
        )
        result = linear(feats)
        assert result.shape == (count, 1)
        assert not result.count_nonzero()
        assert not linear.output_mask.count_nonzero()
        assert not torch.autograd.grad(result.sum(), feats)[0].count_nonzero()


def test_time_reversal_labels(double_precision):
    try:
        o3.Irrep("1eo")
    except ValueError:
        pytest.skip("Time-reversal representations are not installed")
    a, b, c = "2x1eo", "1x1oe", "2x0oo+2x1oo+2x2oo"
    ins = [(0, 0, i, "uvu", True) for i in range(3)]
    tp = co3.TensorProduct(a, b, c, ins).to(DEVICE)
    x, y = (
        tp.irreps_in1.randn(3, -1, device=DEVICE),
        tp.irreps_in2.randn(3, -1, device=DEVICE),
    )
    torch.testing.assert_close(tp(-x, y), -tp(x, y))
    gate = co3.Gate("1x0eo", [torch.tanh], "1x0ee", [torch.sigmoid], "1x1eo").to(DEVICE)
    ref = nn.Gate("1x0eo", [torch.tanh], "1x0ee", [torch.sigmoid], "1x1eo").to(DEVICE)
    h = torch.randn(3, ref.irreps_in.dim, device=DEVICE)
    actual = co3.ChangeOfBasis(gate.irreps_out, True).to(DEVICE)(
        gate(co3.ChangeOfBasis(gate.irreps_in).to(DEVICE)(h))
    )
    torch.testing.assert_close(actual, ref(h))
