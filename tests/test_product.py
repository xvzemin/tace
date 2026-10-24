"""Tensor products, recoupling, and ACE contractions."""

import itertools
from functools import partial

import pytest
import torch
from e3nn import o3

import tace.models.eqt.equitorch.nn.sparse_product as sparse_product
import tace.models.eqt.equitorch.nn.tensor_products as tensor_products
from tace.models._e3nn.fused import UuuTensorProduct, UvuTensorProduct
from tace.models._e3nn.paths import (
    SymmetricProductPaths,
    generate_paths,
    satisfy,
    to_possible_tp_irreps,
)
from tace.models._e3nn.prod import BilinearMoEACE, CgtpACE
from tace.models._e3nn.wigner6j import (
    O3Wigner6jTensorProductConv,
    wigner_6j,
)
from tace.models.eqt.equitorch.nn import TensorProduct
from tace.models.linear import e3nnElementLinear, e3nnMoEElementLinear
from tace.models.time_reversal import spherical_harmonics_irreps, supports_time_reversal
from tace.utils.env import EQX_KERNELS, acceleration_enabled

pytestmark = pytest.mark.usefixtures("double_precision")


@pytest.mark.parametrize(
    "parity,expected",
    [
        (False, "0e + 1o + 2e + 3o"),
        (True, "0e + 1o + 1e + 2o + 2e + 3o"),
        (None, "0e + 1o + 1e + 2e + 3o"),
    ],
)
def test_possible_tp_irreps(parity, expected):
    irreps_in1 = o3.Irreps("2x2e + 3x1o + 0e + 1o")
    irreps_in2 = o3.Irreps("1o + 2x0e")
    target = o3.Irreps("4x1e + 0o + 3e + 4e")
    actual = to_possible_tp_irreps(
        irreps_in1, irreps_in2, parity, lmax=3, target_irreps=target
    )
    assert actual == o3.Irreps(expected)
    assert all(mul == 1 for mul, _ in actual)
    assert all(
        any(ir in ir1 * ir2 for _, ir1 in irreps_in1 for _, ir2 in irreps_in2)
        for _, ir in actual
    )
    truncated = to_possible_tp_irreps(
        irreps_in1, irreps_in2, parity, lmax=0, target_irreps=target
    )
    assert truncated == o3.Irreps("0e")
    default_degree = to_possible_tp_irreps(
        irreps_in1, irreps_in2, parity, target_irreps=target
    )
    assert default_degree == o3.Irreps([(mul, ir) for mul, ir in actual if ir.l <= 1])
    assert to_possible_tp_irreps(irreps_in1, irreps_in2) == o3.Irreps("0e + 1o")
    assert to_possible_tp_irreps(
        o3.Irreps("0e"), o3.Irreps("0e"), parity, target_irreps=target
    ) == o3.Irreps("0e")


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_possible_tp_irreps_time_reversal():
    actual = to_possible_tp_irreps(
        o3.Irreps("1oo + 1oe"),
        o3.Irreps("1oe"),
        lmax=2,
        target_irreps=o3.Irreps("1ee"),
    )
    assert {(ir.l, ir.p, ir.t) for _, ir in actual} == {
        (ell, 1, time_reversal)
        for ell in range(3)
        for time_reversal in (-1, 1)
    }


@pytest.mark.parametrize("mode", ["uuu", "uvu"])
@pytest.mark.parametrize("identical_inputs", [False, True])
@pytest.mark.parametrize("trainable", [False, True])
def test_generate_paths_order(mode, identical_inputs, trainable):
    irreps_in1 = o3.Irreps("2x0e + 3x1o + 1o + 2x2e")
    irreps_in2 = irreps_in1 if identical_inputs else o3.Irreps("1o + 0e + 2e + 1o")
    irreps_out = o3.Irreps("2e + 0e + 1e + 1o + 0e + 3o")
    for restriction in (None, "<", "<=", ">", ">=", "==", "!="):
        paths, outputs = [], []
        for _, ir_out in irreps_out:
            for i, (mul, ir1) in enumerate(irreps_in1):
                for j, (_, ir2) in enumerate(irreps_in2):
                    if (
                        ir_out not in ir1 * ir2
                        or not satisfy(ir1.l, ir2.l, restriction)
                        or not satisfy(ir2.l, ir_out.l, "<=")
                        or not satisfy(ir_out.l, ir1.l, ">=")
                        or (
                            identical_inputs
                            and i == j
                            and (ir1.l + ir2.l - ir_out.l) % 2 == 1
                        )
                    ):
                        continue
                    paths.append((i, j, len(outputs), mode, mode == "uvu" or trainable))
                    outputs.append((mul, ir_out))
        actual = generate_paths(
            irreps_out,
            irreps_in1,
            irreps_in2,
            l1l2=restriction,
            l2l3="<=",
            l3l1=">=",
            e3nn_mode=mode,
            trainable=trainable,
            identical_inputs=identical_inputs,
        )
        assert actual == (paths, o3.Irreps(outputs))


@pytest.mark.parametrize(
    "master,conv,product,expected",
    [
        (None, True, True, (None, None)),
        ("1", True, True, (True, True)),
        ("1", False, True, (False, True)),
        ("1", True, False, (True, False)),
        ("0", True, True, (False, False)),
    ],
)
def test_internal_eqx_selection(monkeypatch, master, conv, product, expected):
    if master is None:
        monkeypatch.delenv("TACE_USE_EQX", raising=False)
    else:
        monkeypatch.setenv("TACE_USE_EQX", master)
    monkeypatch.setitem(EQX_KERNELS, "conv", conv)
    monkeypatch.setitem(EQX_KERNELS, "product", product)
    assert (
        acceleration_enabled("eqx", kernel="conv"),
        acceleration_enabled("eqx", kernel="product"),
    ) == expected


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
        filtered = UuuTensorProduct(
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


@pytest.mark.parametrize(
    ("setting", "correlation", "expected_setting", "expected_product"),
    [
        (None, 2, None, False),
        (None, 3, None, True),
        ("0", 3, False, False),
        ("1", 2, True, True),
    ],
)
def test_product_eqt_selection(
    monkeypatch: pytest.MonkeyPatch,
    setting: str | None,
    correlation: int,
    expected_setting: bool | None,
    expected_product: bool,
) -> None:
    if setting is None:
        monkeypatch.delenv("TACE_USE_EQT", raising=False)
    else:
        monkeypatch.setenv("TACE_USE_EQT", setting)

    product = make_product("2x0e+2x1o", correlation, nonlinear=None)
    assert acceleration_enabled("eqt") is expected_setting
    assert all(ace.use_eqt is expected_product for ace in product.aces)
    assert all(hasattr(ace, "fused_tp") is expected_product for ace in product.aces)


def _reference_segment(
    source: torch.Tensor,
    indptr: torch.Tensor,
    *,
    reduce: str,
) -> torch.Tensor:
    assert reduce == "sum"
    indptr = indptr.squeeze(0)
    return torch.stack(
        [
            source[:, start:end].sum(dim=1)
            for start, end in zip(indptr[:-1].tolist(), indptr[1:].tolist())
        ],
        dim=1,
    )


def test_eqt_first_derivative_skips_unused_intermediate_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor_product = TensorProduct(
        "0e+1o",
        "0e+1o",
        "0e+1o",
        channels_in1=2,
        channels_in2=2,
        channels_out=2,
        internal_weights=True,
    ).to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    tensor_product.weight.requires_grad_(False)
    input1 = torch.randn(
        2,
        tensor_product.irreps_in1_dim,
        2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float64,
        requires_grad=True,
    )
    input2 = torch.randn_like(input1, requires_grad=True)

    sparse_mul_calls = 0
    sparse_mul = tensor_products.sparse_mul

    def tracked_sparse_mul(*args, **kwargs):
        nonlocal sparse_mul_calls
        sparse_mul_calls += 1
        return sparse_mul(*args, **kwargs)

    monkeypatch.setattr(tensor_products, "sparse_mul", tracked_sparse_mul)
    output = tensor_product(input1, input2)
    torch.autograd.grad(output.sum(), (input1, input2))

    assert sparse_mul_calls == 6


@pytest.mark.parametrize("channels", [4, 64])
def test_eqt_scatter_values_and_derivatives(monkeypatch, channels):
    tensor_product = TensorProduct(
        "0e+1o+2e",
        "0e+1o+2e",
        "0e+1o+2e",
        channels_in1=channels,
        channels_in2=channels,
        channels_out=channels,
        internal_weights=True,
    ).to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    input1 = torch.randn(
        3,
        tensor_product.irreps_in1_dim,
        channels,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float64,
        requires_grad=True,
    )
    input2 = torch.randn_like(input1, requires_grad=True)
    inputs = (input1, input2, tensor_product.weight)
    installed_segment = sparse_product.segment_csr

    def derivatives(segment):
        monkeypatch.setattr(sparse_product, "segment_csr", segment)
        value = tensor_product(input1, input2)
        result = [value.detach()]
        for _ in range(3):
            gradients = torch.autograd.grad(
                value.sin().sum(), inputs, create_graph=True
            )
            result.extend(g.detach() for g in gradients)
            value = torch.cat([g.flatten() for g in gradients]) / channels
        return result

    expected = derivatives(_reference_segment)
    segments = [None] if installed_segment is None else [None, installed_segment]
    for segment in segments:
        for actual, reference in zip(derivatives(segment), expected):
            torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_wigner_6j_recoupling_coefficient():
    assert wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1.0 / 2.0)


@pytest.mark.parametrize("shared_weights", [False, True])
def test_uvu_tensor_product_oeq_matches_e3nn(shared_weights, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("OEQ requires CUDA")
    pytest.importorskip("openequivariance")
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    irreps_in1 = o3.Irreps("2x0e + 2x1o")
    irreps_in2 = o3.Irreps("1x1e")
    irreps_out = o3.Irreps("2x1e + 2x0o + 2x1o + 2x2o")
    instructions = [
        (0, 0, 0, "uvu", True, 1.0),
        (1, 0, 1, "uvu", True, 1.0),
        (1, 0, 2, "uvu", True, 1.0),
        (1, 0, 3, "uvu", True, 1.0),
    ]

    monkeypatch.setenv("TACE_USE_OEQ", "0")
    reference = UvuTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights=shared_weights,
    ).to(DEVICE)
    monkeypatch.setenv("TACE_USE_OEQ", "1")
    actual = UvuTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights=shared_weights,
    ).to(DEVICE)
    assert actual.use_oeq
    assert hasattr(actual, "fused_tp")
    assert not hasattr(reference, "fused_tp")

    num_nodes = 7
    x = torch.randn(num_nodes, irreps_in1.dim, device=DEVICE, requires_grad=True)
    y = torch.randn(num_nodes, irreps_in2.dim, device=DEVICE, requires_grad=True)
    weight_shape = (
        (reference.weight_numel,)
        if shared_weights
        else (num_nodes, reference.weight_numel)
    )
    weights = torch.randn(*weight_shape, device=DEVICE, requires_grad=True)

    expected = reference(x, y, weights)
    observed = actual(x, y, weights)
    torch.testing.assert_close(observed, expected, atol=2.0e-12, rtol=2.0e-12)

    grad_output = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(
        (expected * grad_output).sum(), (x, y, weights), retain_graph=True
    )
    observed_grads = torch.autograd.grad(
        (observed * grad_output).sum(), (x, y, weights)
    )
    for observed_grad, expected_grad in zip(observed_grads, expected_grads):
        torch.testing.assert_close(
            observed_grad,
            expected_grad,
            atol=3.0e-12,
            rtol=3.0e-12,
        )


def _build_tensor_product(*, parity=True):
    irreps_edge_attrs = o3.Irreps.spherical_harmonics(2, p=-1)
    if parity:
        irreps_node_feats = o3.Irreps("2x0e + 2x1o + 2x1e")
        irreps_out = o3.Irreps("2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o")
        extra_irreps = o3.Irreps("0o + 0e + 1o + 1e + 2o + 2e")
    else:
        irreps_node_feats = o3.Irreps("2x0e + 2x1o + 2x2e")
        irreps_out = irreps_node_feats
        extra_irreps = irreps_edge_attrs
    module = O3Wigner6jTensorProductConv(
        irreps_node_feats,
        irreps_edge_attrs,
        irreps_out,
        extra_irreps_node_attrs=extra_irreps,
        register_reference=True,
    )
    return module.to(DEVICE)


def _random_inputs(module, *, requires_grad=False, num_nodes=5, num_edges=11):
    edge_index = (
        torch.randint(num_nodes, (2, num_edges), device=DEVICE)
        if num_edges
        else torch.empty(2, 0, dtype=torch.int64, device=DEVICE)
    )
    node_feats = torch.randn(
        num_nodes,
        module.irreps_node_feats.dim,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    edge_attrs = torch.randn(
        num_edges,
        module.irreps_edge_attrs.dim,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    extra_node_attrs = torch.randn(
        num_nodes,
        module.extra_irreps_node_attrs.dim,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    edge_weights = torch.randn(
        num_edges,
        module.edge_weight_numel,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    extra_weights = torch.randn(
        num_edges,
        module.extra_weight_numel,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    return (
        node_feats,
        edge_attrs,
        extra_node_attrs,
        edge_weights,
        extra_weights,
        edge_index,
    )


@pytest.mark.parametrize("improper", [False, True])
def test_wigner6j_matches_reference_gradients_and_o3(improper):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    module = _build_tensor_product()
    assert {"node_tp", "edge_tp"}.issubset(dict(module.named_children()))
    node_tp = module.node_tp
    assert node_tp.irreps_out == node_tp.irreps_out.sort().irreps
    if not node_tp.use_oeq:
        for tp in (node_tp.tp, module.edge_tp.tp):
            forward = tp._compiled_main_left_right
            if isinstance(forward, torch.fx.GraphModule):
                assert any(
                    node.op == "call_function" and node.target is torch.split
                    for node in forward.graph.nodes
                )
    couplings = [
        (ins.i_in1, ins.i_in2, node_tp.irreps_out[ins.i_out].ir)
        for ins in node_tp.instructions
    ]
    assert len(couplings) == len(set(couplings))
    assert len(couplings) < len(module.edge_tp.instructions)
    assert node_tp.shared_weights
    assert module.edge_weight_numel == module.reference_node_edge_tp.weight_numel
    assert module.extra_weight_numel == module.reference_edge_edge_tp.weight_numel
    state = module.state_dict()
    # The old node TP stored one output per recoupling term, before sharing.
    old_intermediate_dim = sum(
        node_tp.irreps_out[ins[0]].dim for ins in module.edge_tp.instructions
    )
    state["node_tp.tp.output_mask"] = torch.ones(old_intermediate_dim)
    module.load_state_dict(state, strict=True)
    assert module.node_tp.tp.output_mask.numel() == node_tp.irreps_out.dim
    inputs = _random_inputs(module, requires_grad=True)

    recoupled = module(*inputs)
    reference = module.forward_reference(*inputs)
    torch.testing.assert_close(recoupled, reference, atol=2.0e-12, rtol=2.0e-12)

    grad_output = torch.randn_like(recoupled)
    differentiable_inputs = inputs[:-1]
    recoupled_grads = torch.autograd.grad(
        (recoupled * grad_output).sum(),
        differentiable_inputs,
        create_graph=True,
    )
    reference_grads = torch.autograd.grad(
        (reference * grad_output).sum(),
        differentiable_inputs,
        create_graph=True,
    )
    for recoupled_grad, reference_grad in zip(recoupled_grads, reference_grads):
        torch.testing.assert_close(
            recoupled_grad,
            reference_grad,
            atol=3.0e-12,
            rtol=3.0e-12,
        )

    directions = [torch.randn_like(value) for value in differentiable_inputs]
    recoupled_second_grads = torch.autograd.grad(
        sum(
            (grad * direction).sum()
            for grad, direction in zip(recoupled_grads, directions)
        ),
        differentiable_inputs,
    )
    reference_second_grads = torch.autograd.grad(
        sum(
            (grad * direction).sum()
            for grad, direction in zip(reference_grads, directions)
        ),
        differentiable_inputs,
    )
    for observed, expected in zip(recoupled_second_grads, reference_second_grads):
        torch.testing.assert_close(observed, expected, atol=2.0e-11, rtol=2.0e-11)

    node_feats, edge_attrs, extra_node_attrs, edge_weights, extra_weights, _ = inputs
    rotation = o3.rand_matrix(dtype=torch.float64)
    if improper:
        rotation = -rotation
    node_rotation = module.irreps_node_feats.D_from_matrix(rotation).to(DEVICE)
    edge_rotation = module.irreps_edge_attrs.D_from_matrix(rotation).to(DEVICE)
    extra_rotation = module.extra_irreps_node_attrs.D_from_matrix(rotation).to(DEVICE)
    output_rotation = module.irreps_out.D_from_matrix(rotation).to(DEVICE)
    rotated_inputs = (
        node_feats @ node_rotation.T,
        edge_attrs @ edge_rotation.T,
        extra_node_attrs @ extra_rotation.T,
        edge_weights,
        extra_weights,
        inputs[-1],
    )

    rotated_output = module(*rotated_inputs)
    expected = recoupled @ output_rotation.T
    torch.testing.assert_close(
        rotated_output,
        expected,
        atol=3.0e-11,
        rtol=3.0e-11,
    )


def test_wigner6j_repeated_irreps_remain_independent():
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    module = O3Wigner6jTensorProductConv(
        "2x1o + 3x1o",
        "0e + 1o",
        "0o + 1e + 1o + 2e + 2o",
        "1e + 1e",
        l1l2="==",
        register_reference=True,
    ).to(DEVICE)
    inputs = _random_inputs(module, requires_grad=True)
    observed = module(*inputs)
    expected = module.forward_reference(*inputs)
    torch.testing.assert_close(observed, expected, atol=2.0e-12, rtol=2.0e-12)
    probe = torch.randn_like(expected)
    observed_grads = torch.autograd.grad((observed * probe).sum(), inputs[:-1])
    expected_grads = torch.autograd.grad((expected * probe).sum(), inputs[:-1])
    for observed, expected in zip(observed_grads, expected_grads):
        torch.testing.assert_close(observed, expected, atol=3.0e-12, rtol=3.0e-12)


@pytest.mark.parametrize("method", ["forward", "forward_reference"])
def test_wigner6j_requires_edge_weights(method):
    module = O3Wigner6jTensorProductConv(
        "2x0e", "0e", "0e", "0e", register_reference=True
    ).to(DEVICE)
    inputs = list(_random_inputs(module, num_nodes=3, num_edges=5))
    inputs[4] = inputs[4][:3]
    with pytest.raises(ValueError, match="one row per graph edge"):
        getattr(module, method)(*inputs)


def test_wigner6j_export_preserves_split_inputs():
    module = O3Wigner6jTensorProductConv(
        "2x0e + 2x1o",
        "0e + 1o",
        "0e + 1o",
        "0e + 1e",
    ).to(DEVICE)
    inputs = _random_inputs(module, requires_grad=True)
    exported = torch.export.export(module, inputs, strict=True).module()
    expected = module(*inputs)
    observed = exported(*inputs)
    torch.testing.assert_close(observed, expected)
    expected_grads = torch.autograd.grad(expected.sum(), inputs[:-1])
    observed_grads = torch.autograd.grad(observed.sum(), inputs[:-1])
    for observed, expected in zip(observed_grads, expected_grads):
        torch.testing.assert_close(observed, expected)


@pytest.mark.parametrize("num_nodes", [0, 3])
def test_wigner6j_empty_edges(num_nodes):
    module = O3Wigner6jTensorProductConv(
        "2x0e + 2x1o",
        "0e + 1o",
        "0e + 1o",
        "0e + 1e",
        register_reference=True,
    ).to(DEVICE)
    inputs = _random_inputs(
        module, num_nodes=num_nodes, num_edges=0, requires_grad=True
    )
    observed = module(*inputs)
    expected = module.forward_reference(*inputs)
    assert observed.shape == (num_nodes, module.irreps_out.dim)
    torch.testing.assert_close(observed, expected)
    observed_grads = torch.autograd.grad(observed.sum(), inputs[:-1])
    expected_grads = torch.autograd.grad(expected.sum(), inputs[:-1])
    for observed, expected in zip(observed_grads, expected_grads):
        torch.testing.assert_close(observed, expected)


@pytest.mark.parametrize("parity", [False, True])
def test_wigner6j_oeq_matches_reference(parity, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("OEQ requires CUDA")
    pytest.importorskip("openequivariance")
    torch.manual_seed(2)
    torch.set_default_dtype(torch.float64)
    monkeypatch.setenv("TACE_USE_CUE", "0")
    monkeypatch.setenv("TACE_USE_OEQ", "0")
    native = _build_tensor_product(parity=parity)
    monkeypatch.setenv("TACE_USE_OEQ", "1")
    module = _build_tensor_product(parity=parity)
    assert module.node_tp.use_oeq
    assert module.edge_tp.use_oeq
    assert module.irreps_out == native.irreps_out
    assert module.edge_weight_numel == native.edge_weight_numel
    assert module.extra_weight_numel == native.extra_weight_numel
    for name in ("node_tp", "edge_tp"):
        assert getattr(module, name).instructions == getattr(native, name).instructions
    assert dict(module.named_parameters()) == dict(native.named_parameters()) == {}
    inputs = _random_inputs(module, requires_grad=True)
    observed = module(*inputs)
    expected = module.forward_reference(*inputs)
    torch.testing.assert_close(observed, expected, atol=3.0e-12, rtol=3.0e-12)
    torch.testing.assert_close(observed, native(*inputs), atol=3.0e-12, rtol=3.0e-12)
    probe = torch.randn_like(expected)
    observed_grads = torch.autograd.grad(
        (observed * probe).sum(), inputs[:-1], create_graph=True
    )
    expected_grads = torch.autograd.grad(
        (expected * probe).sum(), inputs[:-1], create_graph=True
    )
    for observed, expected in zip(observed_grads, expected_grads):
        torch.testing.assert_close(observed, expected, atol=5.0e-12, rtol=5.0e-12)
    directions = [torch.randn_like(value) for value in inputs[:-1]]
    observed_second = torch.autograd.grad(
        sum(
            (grad * direction).sum()
            for grad, direction in zip(observed_grads, directions)
        ),
        inputs[:-1],
    )
    expected_second = torch.autograd.grad(
        sum(
            (grad * direction).sum()
            for grad, direction in zip(expected_grads, directions)
        ),
        inputs[:-1],
    )
    for observed, expected in zip(observed_second, expected_second):
        torch.testing.assert_close(observed, expected, atol=3.0e-11, rtol=3.0e-11)


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
def test_wigner6j_matches_reference_and_time_reversal():
    node_irreps = o3.Irreps("2x0ee + 2x1eo")
    edge_irreps = spherical_harmonics_irreps(1, p=-1)
    magnetic_irreps = spherical_harmonics_irreps(
        1,
        p=1,
        time_reversal=-1,
    )
    intermediate_irreps = to_possible_tp_irreps(
        node_irreps,
        edge_irreps,
        parity=True,
        lmax=2,
    )
    output_irreps = (
        to_possible_tp_irreps(
            intermediate_irreps,
            magnetic_irreps,
            parity=True,
            lmax=1,
        )
        * 2
    ).regroup()
    tensor_product = O3Wigner6jTensorProductConv(
        node_irreps,
        edge_irreps,
        output_irreps,
        magnetic_irreps,
        register_reference=True,
    )
    assert (
        tensor_product.edge_tp.irreps_out
        == tensor_product.reference_edge_edge_tp.irreps_out
        == tensor_product.irreps_out
    )

    num_nodes = 4
    num_edges = 7
    edge_index = torch.randint(num_nodes, (2, num_edges))
    node_features = torch.randn(num_nodes, node_irreps.dim)
    edge_features = torch.randn(num_edges, edge_irreps.dim)
    magnetic_features = torch.randn(num_nodes, magnetic_irreps.dim)
    edge_weights = torch.randn(num_edges, tensor_product.edge_weight_numel)
    magnetic_weights = torch.randn(num_edges, tensor_product.extra_weight_numel)

    def reverse(features, irreps):
        matrix = irreps.D_from_matrix(
            -torch.eye(3),
            parity=False,
            time_reversal=True,
        )
        return features @ matrix.T

    output = tensor_product(
        node_features,
        edge_features,
        magnetic_features,
        edge_weights,
        magnetic_weights,
        edge_index,
    )
    reference = tensor_product.forward_reference(
        node_features,
        edge_features,
        magnetic_features,
        edge_weights,
        magnetic_weights,
        edge_index,
    )
    torch.testing.assert_close(output, reference)
    observed = tensor_product(
        reverse(node_features, node_irreps),
        reverse(edge_features, edge_irreps),
        reverse(magnetic_features, magnetic_irreps),
        edge_weights,
        magnetic_weights,
        edge_index,
    )
    torch.testing.assert_close(
        observed,
        reverse(output, tensor_product.irreps_out),
    )


@pytest.mark.skipif(
    not supports_time_reversal(),
    reason="the installed e3nn does not represent time-reversal parity",
)
@pytest.mark.parametrize(
    ("kernel", "environment"),
    [("CUE", "TACE_USE_CUE"), ("OEQ", "TACE_USE_OEQ")],
)
def test_time_reversal_wigner6j_rejects_scatter_acceleration(
    monkeypatch,
    kernel,
    environment,
):
    for variable in ("TACE_USE_CUE", "TACE_USE_OEQ"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(environment, "1")
    with pytest.raises(ValueError, match=f"{kernel} does not support"):
        O3Wigner6jTensorProductConv(
            o3.Irreps("2x0ee + 2x1eo"),
            spherical_harmonics_irreps(1, p=-1),
            o3.Irreps("2x0ee + 2x1eo"),
            spherical_harmonics_irreps(1, p=1, time_reversal=-1),
        )
