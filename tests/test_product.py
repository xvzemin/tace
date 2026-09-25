"""ACE paths, element coefficients and separate standard/bilinear execution."""

import itertools
from functools import partial

import pytest
import torch
from e3nn import o3

from tace.models._e3nn.fused import uuuTensorProduct
from tace.models._e3nn.paths import SymmetricProductPaths, generate_paths
from tace.models._e3nn.prod import BilinearMoEACE, CgtpACE
from tace.models.linear import e3nnElementLinear, e3nnMoEElementLinear


def test_standard_ace_eqx_precedes_eqt(monkeypatch):
    monkeypatch.setenv("TACE_USE_EQX", "1")
    monkeypatch.setenv("TACE_USE_EQT", "1")
    product = make_product("2x0e+2x1o", 3, nonlinear=None)
    assert hasattr(product, "eqx_ace")
    assert all(not ace.use_eqt for ace in product.aces)


@pytest.mark.parametrize("correlation", [2, 3, 4])
@pytest.mark.parametrize("agnostic", [False, True])
@pytest.mark.parametrize("matrix_weight", ["0", "1"])
def test_fused_ace_cuda(monkeypatch, correlation, agnostic, matrix_weight):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setenv("TACE_USE_EQT", "0")
    import tace.models._e3nn.prod as prod

    for name in ("e3nnLinear", "e3nnElementLinear"):
        monkeypatch.setattr(
            prod, name, partial(getattr(prod, name), use_matrix_weight=matrix_weight)
        )
    kwargs = dict(
        layer=0,
        num_layers=1,
        num_elements=2,
        Lmax=1,
        lmax=1,
        num_channel=2,
        num_expert=None,
        num_channel_per_expert=None,
        target_irreps=o3.Irreps("0e+1o"),
        irreps_in=o3.Irreps("2x0e+2x1o"),
        correlation=[correlation],
        l1l2=None,
        bias=True,
        nonlinear=None,
        agnostic=agnostic,
        parity=True,
    )
    monkeypatch.setenv("TACE_USE_EQX", "0")
    reference = CgtpACE(**kwargs).to(device="cuda", dtype=torch.float64)
    monkeypatch.setenv("TACE_USE_EQX", "1")
    actual = CgtpACE(**kwargs).to(device="cuda", dtype=torch.float64)
    actual.load_state_dict(reference.state_dict(), strict=True)
    assert actual.state_dict().keys() == reference.state_dict().keys()
    x = torch.randn(3, 8, device="cuda", dtype=torch.float64, requires_grad=True)
    attrs = x.new_tensor([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])
    # Include nonzero biases, including the soft-attribute bias convention.
    with torch.no_grad():
        for a, b in zip(actual.coefs, reference.coefs):
            if a.bias is not None:
                a.bias.normal_()
                b.bias.copy_(a.bias)
    predictions, derivatives = [], []
    for model in (actual, reference):
        output = model(x, attrs, None, torch.zeros(3, device="cuda", dtype=torch.long))
        first = torch.autograd.grad(output.square().sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(
            first.square().sum(),
            (x, *model.parameters()),
            create_graph=True,
            allow_unused=True,
        )
        third = torch.autograd.grad(second[0].square().sum(), x)[0]
        predictions.append(output)
        derivatives.append((first, *second, third))
    torch.testing.assert_close(*predictions, atol=1e-10, rtol=1e-10)
    for a, b in zip(*derivatives):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-9)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("matrix_weight", ["0", "1"])
@pytest.mark.parametrize("num_nodes", [0, 5])
def test_precomputed_element_indices(moe, matrix_weight, num_nodes):
    """Reuse element indices without changing soft-attribute bias or gradients."""
    cls = e3nnMoEElementLinear if moe else e3nnElementLinear
    module = cls(
        "4x0e+4x1o",
        "6x0e+2x1o",
        num_elements=3,
        bias=True,
        use_matrix_weight=matrix_weight,
        **({"num_experts": 2} if moe else {}),
    )
    with torch.no_grad():
        module.bias.normal_()
    x = torch.randn(num_nodes, 16, requires_grad=True)
    attrs = torch.rand(num_nodes, 3, requires_grad=True)
    node_type = attrs.argmax(-1)
    expected = module(x, attrs)
    actual = module(x, attrs, node_type)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (x, attrs, *module.parameters())
    derivatives = [
        torch.autograd.grad(
            y.square().sum(), inputs, retain_graph=True, allow_unused=True
        )
        for y in (actual, expected)
    ]
    for actual_grad, expected_grad in zip(*derivatives):
        if expected_grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.fixture(autouse=True)
def double_precision():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)


def product_irreps(irreps_in1, irreps_in2, lmax):
    return o3.Irreps(
        [
            (1, ir)
            for ir in sorted(
                {
                    ir
                    for _, ir1 in irreps_in1
                    for _, ir2 in irreps_in2
                    for ir in ir1 * ir2
                    if ir.l <= lmax
                }
            )
        ]
    )


@pytest.mark.parametrize(
    "irreps,correlation,lmax",
    [
        ("1o", 5, 5),
        ("2e", 3, 6),
        ("1o+1o", 3, 3),
        ("0e+1o+1e", 3, 3),
    ],
)
def test_zero_paths_match_symmetrized_cg(irreps, correlation, lmax):
    """Compare against dense CG maps symmetrized over every original input."""
    base = o3.Irreps(irreps)
    symmetric_paths = SymmetricProductPaths(base)
    irreps_in = base
    selectors = [
        torch.eye(base.dim, dtype=torch.float64)[:, sl] for sl in base.slices()
    ]
    coefficients = selectors
    for nu in range(2, correlation + 1):
        instructions, irreps_out = generate_paths(
            product_irreps(irreps_in, base, lmax),
            irreps_in,
            base,
            e3nn_mode="uuu",
        )
        permutations = list(itertools.permutations(range(nu)))
        expected, nonzero = [], []
        for i, j, k, mode, trainable in instructions:
            cg = o3.wigner_3j(
                irreps_in[i].ir.l,
                base[j].ir.l,
                irreps_out[k].ir.l,
                dtype=torch.float64,
            )
            value = torch.einsum(
                "...a,xb,abm->...xm", coefficients[i], selectors[j], cg
            )
            value = sum(value.permute(*perm, nu) for perm in permutations) / len(
                permutations
            )
            if value.abs().max() > 1e-12:
                expected.append((i, j, len(nonzero), mode, trainable))
                nonzero.append(value)
        actual, irreps_in = symmetric_paths.filter(instructions, irreps_out, base)
        assert actual == expected
        coefficients = nonzero

        if irreps == "1o" and nu == 3:
            # These two vector paths are proportional, not zero: keep both.
            assert sum(ir.l == 1 for _, ir in irreps_in) == 2
            assert {ir.l for _, ir in irreps_in} == {1, 3}
        if irreps == "2e" and nu == 3:
            # Odd angular degree is not in general forbidden at odd correlation.
            assert {ir.l for _, ir in irreps_in} == {0, 2, 3, 4, 6}


@pytest.mark.parametrize("use_eqt", [False, True])
def test_filtered_products_preserve_values_and_derivatives(monkeypatch, use_eqt):
    monkeypatch.setenv("TACE_USE_EQT", str(int(use_eqt)))
    torch.manual_seed(0)
    base = o3.Irreps("2x0e+2x1o+2x1o")
    symmetric_paths = SymmetricProductPaths(base)
    irreps_in = base
    features = torch.randn(4, base.dim, dtype=torch.float64, requires_grad=True)
    reference = features.detach().clone().requires_grad_()
    actual, expected = features, reference
    for nu in range(2, 5):
        requested = product_irreps(irreps_in, base, 2)
        instructions, irreps_out = generate_paths(
            requested,
            irreps_in,
            base,
            e3nn_mode="uuu",
        )
        unfiltered = o3.TensorProduct(
            irreps_in,
            base,
            irreps_out,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        filtered = uuuTensorProduct(
            irreps_in,
            base,
            requested,
            symmetric_paths=symmetric_paths,
        ).double()
        kept = {
            (i, j, filtered.irreps_out[k].ir) for i, j, k, _, _ in filtered.instructions
        }
        slices = [
            sl
            for (i, j, k, _, _), sl in zip(instructions, irreps_out.slices())
            if (i, j, irreps_out[k].ir) in kept
        ]
        assert len(slices) == len(filtered.instructions)
        expected = unfiltered(expected, reference)
        expected = torch.cat([expected[:, sl] for sl in slices], dim=-1)
        actual = filtered(actual, features)
        tolerance = 1e-12
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        irreps_in = filtered.irreps_out

    for _ in range(2):
        actual = torch.autograd.grad(
            actual.square().sum(), features, create_graph=True
        )[0]
        expected = torch.autograd.grad(
            expected.square().sum(), reference, create_graph=True
        )[0]
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def make_product(irreps, correlation, **kwargs):
    return CgtpACE(
        layer=0,
        num_layers=1,
        num_elements=1,
        Lmax=0,
        lmax=2,
        num_channel=2,
        num_expert=None,
        num_channel_per_expert=None,
        target_irreps=o3.Irreps("0e"),
        irreps_in=o3.Irreps(irreps),
        correlation=[correlation],
        l1l2=None,
        bias=False,
        parity=True,
        **kwargs,
    ).double()


@pytest.mark.parametrize("correlation", [1, 2, 3])
@pytest.mark.parametrize("agnostic", [False, True])
@pytest.mark.parametrize("hidden,num_nodes", [(2, 0), (3, 4)])
def test_standard_product_matches_extended_path(
    monkeypatch, correlation, agnostic, hidden, num_nodes
):
    monkeypatch.setenv("TACE_USE_EQT", "0")
    kwargs = dict(
        layer=0,
        num_layers=1,
        num_elements=2,
        Lmax=1,
        lmax=1,
        num_channel=2,
        num_expert=None,
        num_channel_per_expert=hidden,
        target_irreps=o3.Irreps("0e+1o"),
        irreps_in=o3.Irreps("2x0e+2x1o"),
        correlation=[correlation],
        l1l2=None,
        bias=True,
        nonlinear=None,
        agnostic=agnostic,
        parity=True,
        stochastic_depth=0.2 if num_nodes else 0.0,
        use_first_dropout=True,
    )
    actual = CgtpACE(**kwargs)
    reference = BilinearMoEACE(**kwargs)
    reference.load_state_dict(actual.state_dict(), strict=True)
    assert actual.state_dict().keys() == reference.state_dict().keys()
    assert not hasattr(actual, "_linear_up_features")
    assert not hasattr(actual, "_merge_shared_expert")
    x = torch.randn(num_nodes, 8, requires_grad=True)
    attrs = torch.rand(num_nodes, 2)
    sc = torch.randn(num_nodes, actual.irreps_out.dim, requires_grad=True)
    batch = torch.arange(num_nodes) // 2
    for training in (False, True):
        outputs, gradients = [], []
        for model in (actual, reference):
            model.train(training)
            torch.manual_seed(7)
            output = model(x, attrs, sc, batch, attrs.argmax(-1))
            outputs.append(output)
            first = torch.autograd.grad(output.square().sum(), x, create_graph=True)[0]
            gradients.append(
                torch.autograd.grad(
                    output.square().sum() + first.square().sum(),
                    (x, sc, *model.parameters()),
                    allow_unused=True,
                )
            )
        torch.testing.assert_close(*outputs, atol=0, rtol=0)
        for a, b in zip(*gradients):
            if a is None:
                assert b is None
            else:
                torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("correlation", [2, 3, 4, 6, 8])
def test_product_supports_higher_correlation(monkeypatch, correlation):
    monkeypatch.setenv("TACE_USE_EQT", "0")
    model = make_product("2x1o", correlation, nonlinear=None)
    features = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    out = model(
        features,
        torch.ones(3, 1, dtype=torch.float64),
        None,
        torch.zeros(3, dtype=torch.long),
    )
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()
    # Even self powers of one vector have only even degrees, odd powers only odd.
    for nu, ace in enumerate(model.aces, 2):
        assert all(ir.l % 2 == nu % 2 for _, ir in ace.irreps_out)
    if correlation % 2 == 0:
        assert model.aces[-1].instructions
        grad = torch.autograd.grad(out.square().sum(), features, create_graph=True)[0]
        assert torch.isfinite(
            torch.autograd.grad(grad.square().sum(), features)[0]
        ).all()


@pytest.mark.parametrize("irreps", ["2x0e", "2x1e"])
def test_eqt_without_polar_vectors_and_with_empty_paths(monkeypatch, irreps):
    monkeypatch.setenv("TACE_USE_EQT", "1")
    model = make_product(irreps, 3, nonlinear=None)
    features = torch.randn(3, model.irreps_in.dim, dtype=torch.float64)
    out = model(
        features,
        torch.ones(3, 1, dtype=torch.float64),
        None,
        torch.zeros(3, dtype=torch.long),
    )
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()
    if irreps == "2x1e":
        assert not model.aces[-1].instructions
        assert not model.aces[-1].use_eqt


def test_independent_glu_inputs_keep_antisymmetric_paths(monkeypatch):
    monkeypatch.setenv("TACE_USE_EQT", "0")
    # An axial vector output can contain the cross product of two independent
    # vector fields. Do not use the self-product mask for a bilinear gate.
    model = BilinearMoEACE(
        layer=0,
        num_layers=1,
        num_elements=1,
        Lmax=1,
        lmax=1,
        num_channel=2,
        num_expert=None,
        num_channel_per_expert=None,
        target_irreps=o3.Irreps("1e"),
        irreps_in=o3.Irreps("2x1o"),
        correlation=[2],
        l1l2=None,
        bias=False,
        nonlinear="bilinear_gate",
        parity=True,
    )
    ace = model.aces[0]
    assert any(
        i == j == 0 and ace.irreps_out[k].ir.l == 1
        for i, j, k, _, _ in ace.instructions
    )


def test_time_reversal_labels_preserved(monkeypatch):
    if not hasattr(o3.Irrep("0e"), "t"):
        pytest.skip("Time-reversal e3nn is not installed")
    monkeypatch.setenv("TACE_USE_EQT", "0")
    base = o3.Irreps("1eo+1ee+0ee")
    symmetric_paths = SymmetricProductPaths(base)
    irreps_in = base
    for _ in range(3):
        instructions, irreps_out = generate_paths(
            product_irreps(irreps_in, base, 2),
            irreps_in,
            base,
            e3nn_mode="uuu",
        )
        paths, out = symmetric_paths.filter(instructions, irreps_out, base)
        for i, j, k, _, _ in paths:
            assert out[k].ir.p == irreps_in[i].ir.p * base[j].ir.p
            assert out[k].ir.t == irreps_in[i].ir.t * base[j].ir.t
        # Equal l with opposite time labels denotes independent fields.
        if irreps_in == base:
            assert any(
                i == 0 and j == 1 and out[k].ir.l == 1 for i, j, k, _, _ in paths
            )
        irreps_in = out

    model = make_product("2x1eo", 4, nonlinear=None, use_time_reversal=True)
    for nu, ace in enumerate(model.aces, 2):
        assert ace.instructions
        assert all(ir.t == (-1) ** nu for _, ir in ace.irreps_out)
    features = torch.randn(3, model.irreps_in.dim, requires_grad=True)
    out = model(features, torch.ones(3, 1), None, torch.zeros(3, dtype=torch.long))
    assert torch.isfinite(torch.autograd.grad(out.square().sum(), features)[0]).all()
