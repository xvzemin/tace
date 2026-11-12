"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import co2, co3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
