################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import doctest
import re
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from e3nn import o3

from eqx import o2
from tace.models.layout import LayoutTransform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


@pytest.mark.parametrize("allow_slice", [False, True])
def test_native_o2_import_is_independent(allow_slice):
    script = f"""
import sys
import torch
if {allow_slice}:
    torch.serialization.add_safe_globals([slice])
safe_globals = set(torch.serialization.get_safe_globals())
load = torch.load
import eqx
from eqx import o2

assert safe_globals.issubset(torch.serialization.get_safe_globals())
assert (slice in torch.serialization.get_safe_globals()) == (slice in safe_globals)
assert torch.load is load
linear = o2.Linear('2x0e+2x1m', '2x0e+2x1m')
linear(linear.irreps_in.randn(3, -1)).sum().backward()
frame = o2.LocalFrame('2x0e+2x1o')
rotation = o2.WignerD(mmax=1, lmax=1, method='recursive')
d, di = rotation(torch.randn(3, 3))
x = torch.randn(3, frame.input_dim)
frame.to_global(frame.to_local(x, d), di)
assert all('eqx.' + name not in sys.modules for name in ('conv', 'kernels', 'ace', 'o3'))
assert 'tace' not in sys.modules
assert set(eqx.__all__).issubset(dir(eqx))
assert eqx.conv.O3TensorProductConv is not None
assert 'eqx.kernels.cuda' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("page", ["equivariantx", "convolutions"])
def test_documentation_examples(page):
    path = Path(__file__).resolve().parents[1] / "eqx/docs/source" / f"{page}.rst"
    blocks = re.findall(
        r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*|)\n)+)",
        path.read_text(),
        re.MULTILINE,
    )
    assert blocks
    namespace = {"__name__": "__main__"}
    dtype = torch.get_default_dtype()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            for block in blocks:
                exec(compile(textwrap.dedent(block), str(path), "exec"), namespace)
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize("operator", [o2.Linear, o2.TensorProduct])
def test_operator_docstrings(operator):
    examples = doctest.DocTestParser().get_doctest(
        operator.__doc__,
        {operator.__name__: operator, "torch": torch},
        operator.__name__,
        None,
        None,
    )
    doctest.DebugRunner().run(examples)


def _transform(features, irreps, angle, reflected=False, time_reversal=False):
    matrix = irreps.D_from_angle(
        angle,
        reflected=reflected,
        time_reversal=time_reversal,
        dtype=features.dtype,
        device=features.device,
    )
    return torch.matmul(features, matrix.transpose(-1, -2))


def _time_reverse(features: torch.Tensor, irreps) -> torch.Tensor:
    output = features.clone()
    for ir_mul, ir_slice in zip(irreps, irreps.slices()):
        output[..., ir_slice] *= ir_mul.ir.t
    return output


@pytest.mark.parametrize(
    ("rotation_matrix", "axis"),
    [
        (o2.rotation_matrix_to_x_axis, 0),
        (o2.rotation_matrix_to_y_axis, 1),
        (o2.rotation_matrix_to_z_axis, 2),
    ],
)
def test_rotation_matrix_to_axis(rotation_matrix, axis):
    generator = torch.Generator().manual_seed(20260903)
    vectors = torch.randn(64, 3, dtype=DTYPE, generator=generator).to(DEVICE)
    rotation = rotation_matrix(vectors)
    rotated = torch.einsum("bij,bj->bi", rotation, vectors)
    expected = torch.zeros_like(vectors)
    expected[:, axis] = torch.linalg.vector_norm(vectors, dim=-1)

    torch.testing.assert_close(rotated, expected)
    identity = torch.eye(3, dtype=DTYPE, device=DEVICE).expand_as(rotation)
    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), identity)
    torch.testing.assert_close(
        torch.linalg.det(rotation),
        torch.ones(vectors.size(0), dtype=DTYPE, device=DEVICE),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_eqx_without_optional_backends(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    script = r"""
import sys
import torch

# Match both import failure and find_spec returning None for absent packages.
sys.modules.update({name: None for name in ("triton", "torch_geometric", "torch_scatter")})
from eqx import conv as eqx_conv
from eqx import o2

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
device = sys.argv[1]
linear = o2.Linear("2x0e+2x0o+2x1m", "3x0e+2x0o+2x1m").to(device)
linear(linear.irreps_in.randn(2, -1, device=device)).sum().backward()
tp = o2.O3TensorProduct(
    "2x1o", "1o", "2x0e+2x1e+2x2e",
    [(0, 0, i, "uvu", True) for i in range(3)],
    internal_weights=False, shared_weights=False,
).to(device)
frame = o2.WignerD(2, 2, method="recursive").to(device)
x = torch.randn(3, 6, device=device, requires_grad=True)
vectors = torch.randn(4, 3, device=device, requires_grad=True)
weights = torch.randn(4, tp.weight_numel, device=device, requires_grad=True)
edges = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], device=device)
d, di = frame(vectors)
expected = x.new_zeros(3, tp.irreps_out.dim).index_add(
    0, edges[1], tp(x[edges[0]], d, di, weights)
)
packed = frame.forward_packed(vectors)
actual = tp.forward_scatter(x, edges, packed, weights)
torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
inputs = x, vectors, weights
for _ in range(2):
    a = torch.autograd.grad(actual.square().sum(), inputs, create_graph=True)
    b = torch.autograd.grad(expected.square().sum(), inputs, create_graph=True)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-8, rtol=1e-8)
    actual = torch.cat([value.flatten() for value in a])
    expected = torch.cat([value.flatten() for value in b])
assert "eqx.conv.o2_o3.cuda" not in sys.modules
assert "eqx.kernels.cuda" not in sys.modules
assert "tace" not in sys.modules
accelerated = eqx_conv.O2O3TensorProductConv(tp, backend="cuda").to(device)
arguments = (
    x, weights, x.new_empty(0, tp.weight_numel), packed,
    x.new_ones(1, tp.num_harmonics), edges, 3,
)
if device == "cpu":
    torch.testing.assert_close(accelerated(*arguments), tp.forward_scatter(x, edges, packed, weights))
else:
    import torch.utils.cpp_extension
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    torch.utils.cpp_extension.CUDA_HOME = None
    try:
        accelerated(*arguments)
    except ImportError as error:
        assert "CUDA_HOME" in str(error)
    else:
        raise AssertionError("CUDA execution must report the missing toolkit")
    finally:
        torch.utils.cpp_extension.CUDA_HOME = cuda_home
    # The default CUDA construction also works without Triton or PyG.
    quaternion_frame = o2.WignerD(2, 2).to(device)
    for actual, expected in zip(quaternion_frame(vectors), (d, di)):
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    assert "tace" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, device],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("wigner_lmax", [2, 4])
@pytest.mark.parametrize("basis_change", [True, False])
def test_local_frame_roundtrip_flattened_ir_mul(wigner_lmax, basis_change):
    irreps = o3.Irreps("2x0e+1x0o+3x1e+2x1o+1x2e+3x2o")
    frame = o2.LocalFrame(irreps, basis_change=basis_change).to(DEVICE, DTYPE)
    layout = LayoutTransform(
        irreps,
        layout_in="flatten_mul_ir",
        layout_out="flatten_ir_mul",
    ).to(DEVICE)
    vectors = torch.randn(7, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(wigner_lmax, wigner_lmax).to(DEVICE, DTYPE)(vectors)
    features = torch.randn(7, irreps.dim, dtype=DTYPE, device=DEVICE)

    local = frame(layout(features), wigner)
    assert frame.irreps_out == o2.Irreps("5x0e+7x0o+9x1m+4x2m")
    options = "mmax=2" + ("" if basis_change else ", basis_change=False")
    assert repr(frame) == (
        f"LocalFrame({frame.global_irreps} -> {frame.local_irreps})({options})"
    )
    reverse_frame = o2.LocalFrame(irreps, reverse=True, basis_change=basis_change)
    assert repr(reverse_frame) == (
        f"LocalFrame({frame.local_irreps} -> {frame.global_irreps})({options})"
    )
    if basis_change:
        default = o2.LocalFrame(irreps).to(DEVICE, DTYPE)
        torch.testing.assert_close(
            default(layout(features), wigner), local, atol=0, rtol=0
        )
    assert local.shape == (7, frame.irreps_out.dim)
    torch.testing.assert_close(
        layout.inverse(frame.to_global(local, wigner_inv)),
        features,
    )


@pytest.mark.parametrize("basis_change", [True, False])
def test_local_frame_matches_degreewise_rotation(double_precision, basis_change):
    irreps = o3.Irreps("2x0e+0o+2x1e+3x1o+2e+2x2o")
    frame = o2.LocalFrame(irreps, basis_change=basis_change)
    vectors = torch.randn(5, 3, generator=torch.Generator().manual_seed(7))
    features = torch.randn(5, irreps.dim)
    rotation = o2.rotation_matrix_to_y_axis(vectors)
    wigner, _ = o2.WignerD(2, 2)(vectors)
    expected = {ir: [] for ir, _ in frame.irreps_out}
    for entry, ir_slice in zip(irreps, irreps.slices()):
        ir, mul = entry.ir, entry.mul
        values = features[:, ir_slice].reshape(5, ir.dim, mul)
        values = torch.einsum("bij,bju->biu", ir.D_from_matrix(rotation), values)
        parity = ir.p * (-1) ** ir.l
        expected[o2.Irrep(0, parity)].append(values[:, ir.l : ir.l + 1])
        for m in range(1, ir.l + 1):
            pair = values[:, [ir.l + m, ir.l - m]]
            if basis_change and parity == -1:
                pair = torch.stack((-pair[:, 1], pair[:, 0]), dim=1)
            expected[o2.Irrep(m, 0)].append(pair)
    expected = torch.cat(
        [torch.cat(expected[ir], dim=-1).flatten(1) for ir, _ in frame.irreps_out],
        dim=-1,
    )
    torch.testing.assert_close(frame(features, wigner), expected, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("mmax", [0, 1])
@pytest.mark.parametrize("basis_change", [True, False])
@pytest.mark.parametrize(
    ("wigner_mmax", "wigner_lmax"), [(1, 2), (1, 4), (2, 4), (4, 4)]
)
def test_local_frame_trailing_axes_and_empty_batch(
    mmax, basis_change, wigner_mmax, wigner_lmax
):
    irreps = o3.Irreps("2x0e+2x1o+1x1e+2x2e+3x2o")
    frame = o2.LocalFrame(irreps, mmax=mmax, basis_change=basis_change).to(
        DEVICE, DTYPE
    )
    vectors = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(wigner_mmax, wigner_lmax).to(DEVICE, DTYPE)(vectors)
    reference, reference_inv = o2.WignerD(mmax, 2).to(DEVICE, DTYPE)(vectors)
    features = torch.randn(4, 2, irreps.dim, dtype=DTYPE, device=DEVICE)

    local = frame.to_local(features, wigner)
    assert local.shape == (4, 2, frame.irreps_out.dim)
    torch.testing.assert_close(local, frame.to_local(features, reference))
    torch.testing.assert_close(
        frame.to_global(local, wigner_inv), frame.to_global(local, reference_inv)
    )

    empty = frame.to_local(features[:0], wigner[:0])
    assert empty.shape == (0, 2, frame.irreps_out.dim)
    assert frame.to_global(empty, wigner_inv[:0]).shape == (0, 2, irreps.dim)


@pytest.mark.parametrize(
    ("local_dim", "global_dim", "message"),
    [(4, 4, "degree"), (9, 10, "degree"), (5, 9, "orders"), (6, 9, "orders")],
)
def test_local_frame_rejects_incompatible_wigner_layout(local_dim, global_dim, message):
    frame = o2.LocalFrame("2x2e", mmax=4)
    assert frame.mmax == 2
    wigner = torch.zeros(3, local_dim, global_dim)
    with pytest.raises(ValueError, match=message):
        frame.to_local(torch.zeros(3, frame.input_dim), wigner)
    with pytest.raises(ValueError, match=message):
        frame.to_global(torch.zeros(3, frame.output_dim), wigner.transpose(1, 2))


def test_local_frame_empty_irreps():
    frame = o2.LocalFrame("")
    d, di = o2.WignerD(2, 2)(torch.randn(3, 3))
    for batch_size in (3, 0):
        features = torch.empty(batch_size, 2, 0)
        local = frame(features, d[:batch_size])
        assert local.shape == features.shape
        assert frame.to_global(local, di[:batch_size]).shape == features.shape


@pytest.mark.parametrize("basis_change", [False, True])
@pytest.mark.parametrize("mmax", [0, 1, 3])
def test_local_frame_packed_derivatives(double_precision, basis_change, mmax):
    frame = o2.LocalFrame(
        "2x0e+1x1o+3x1e+2x3o+1x1o", mmax=mmax, basis_change=basis_change
    )
    wigner = o2.WignerD(4, 4, method="recursive")
    for batch_size in (0, 2):
        x = torch.randn(batch_size, 2, frame.input_dim, requires_grad=True)
        r = torch.randn(batch_size, 3, requires_grad=True)
        dense, inverse = wigner(r)
        packed = wigner.forward_packed(r)
        actual = frame.to_local(x, packed)
        expected = frame.to_local(x, dense)
        torch.testing.assert_close(actual, expected)
        actual = frame.to_global(actual, packed)
        expected = frame.to_global(expected, inverse)
        for _ in range(3):
            torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
            gradients = [
                torch.autograd.grad(
                    value.sin().sum(), (x, r), create_graph=True, retain_graph=True
                )
                for value in (actual, expected)
            ]
            actual, expected = [
                torch.cat([g.flatten() for g in values]) / 10 for values in gradients
            ]


def test_gate_derivatives(double_precision):
    gate = o2.Gate(
        "0e+0o",
        [torch.nn.SiLU(), torch.nn.Tanh()],
        "2x0o+3x0e",
        [torch.nn.Tanh(), torch.nn.Sigmoid()],
        "1m+0o+2x2m+0e",
    )
    features = torch.randn(2, gate.irreps_in.dim, requires_grad=True)
    assert torch.autograd.gradcheck(gate, (features,), fast_mode=True)
    assert torch.autograd.gradgradcheck(gate, (features,), fast_mode=True)

    tangent = torch.randn_like(features)
    _, expected = torch.autograd.functional.jvp(gate, features, tangent)
    _, actual = torch.func.jvp(gate, (features,), (tangent,))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.vmap(gate)(features), gate(features))


@pytest.mark.parametrize("normalization", ["element", "path"])
@pytest.mark.parametrize("weight_batch", [(), (1,), (3,)])
def test_linear_grouped_instructions(double_precision, normalization, weight_batch):
    module = o2.Linear(
        "2x1m+3x1m+0e",
        "3x1m+2x1m+0e",
        internal_weights=False,
        shared_weights=False,
        path_normalization=normalization,
    )
    x = torch.randn(3, module.irreps_in.dim, requires_grad=True)
    weight = torch.randn(*weight_batch, module.weight_numel, requires_grad=True)
    inputs = [
        x[..., s].reshape(3, ir.dim, mul)
        for (ir, mul), s in zip(module.irreps_in, module.irreps_in.slices())
    ]
    outputs = []
    for i_out, (ir, mul) in enumerate(module.irreps_out):
        terms = [
            torch.matmul(
                inputs[ins.i_in], module.weight_view_for_instruction(i, weight)
            )
            * ins.path_weight
            for i, ins in enumerate(module._weight_instructions)
            if ins.i_out == i_out
        ]
        outputs.append(sum(terms).reshape(3, ir.dim * mul))
    actual, expected = module(x, weight), torch.cat(outputs, dim=-1)
    for _ in range(2):
        torch.testing.assert_close(actual, expected)
        gradients = [
            torch.autograd.grad(
                value.sin().sum(), (x, weight), create_graph=True, retain_graph=True
            )
            for value in (actual, expected)
        ]
        actual, expected = [
            torch.cat([g.flatten() for g in values]) for values in gradients
        ]


@pytest.mark.parametrize("basis_change", [True, False])
def test_local_frame_truncation_compiles_with_shared_wigner(
    double_precision, basis_change
):
    torch.compiler.reset()
    frame = o2.LocalFrame("2x0e+1x1e+2x2o", mmax=1, basis_change=basis_change)

    def roundtrip(features, wigner, wigner_inv):
        return frame.to_global(frame.to_local(features, wigner), wigner_inv)

    compiled = torch.compile(
        roundtrip, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for mmax, lmax in ((1, 2), (2, 4)):
        for batch_size in (3, 1, 0):
            x = torch.randn(batch_size, 2, frame.input_dim, requires_grad=True)
            r = torch.randn(batch_size, 3, requires_grad=True)
            d, di = o2.WignerD(mmax, lmax)(r)
            actual = compiled(x, d, di)
            x_ref = x.detach().requires_grad_()
            r_ref = r.detach().requires_grad_()
            d, di = o2.WignerD(mmax, lmax)(r_ref)
            expected = roundtrip(x_ref, d, di)
            torch.testing.assert_close(actual, expected)
            for actual_grad, expected_grad in zip(
                torch.autograd.grad(actual.square().sum(), (x, r)),
                torch.autograd.grad(expected.square().sum(), (x_ref, r_ref)),
            ):
                torch.testing.assert_close(actual_grad, expected_grad)


def test_o2_irrep_and_irreps_metadata():
    assert o2.Irrep("0e") == o2.Irrep("0ee") == o2.Irrep(0, 1)
    assert o2.Irrep("0o") == o2.Irrep("0oe") == o2.Irrep((0, -1))
    assert o2.Irrep("0eo") == o2.Irrep((0, 1, -1))
    assert o2.Irrep("1mo") == o2.Irrep(1, 0, -1)
    assert str(o2.Irrep("0eo")) == "0eo"
    assert str(o2.Irrep("2mo")) == "2mo"
    assert o2.Irrep("3m").dim == 2
    assert o2.Irrep("0e").is_invariant_scalar()

    angle = torch.tensor(0.37, dtype=DTYPE)
    time_odd = o2.Irrep("1mo")
    torch.testing.assert_close(
        time_odd.D_from_angle(angle, time_reversal=True),
        -time_odd.D_from_angle(angle),
    )

    irreps = o2.Irreps("2x0e+0o+3x1m+2m")
    assert irreps.dim == 11
    assert irreps.num_irreps == 7
    assert irreps.mmax == 2
    assert irreps.slices() == (
        slice(0, 2),
        slice(2, 3),
        slice(3, 9),
        slice(9, 11),
    )
    assert irreps.regroup() == o2.Irreps("2x0e+0o+3x1m+2m")


def test_o2_irreps_sort_and_serialization():
    import pickle

    irreps = o2.Irreps("2x1mo+0oo+3x0ee+1me+0eo")
    result = irreps.sort()
    assert result.irreps == o2.Irreps("3x0ee+0eo+0oo+1me+2x1mo")
    for i, ir_mul in enumerate(irreps):
        assert result.irreps[result.p[i]] == ir_mul
        assert result.inv[result.p[i]] == i
    assert deepcopy(irreps) == pickle.loads(pickle.dumps(irreps)) == irreps
    assert o2.Irrep("1mo") in irreps
    assert "2me" not in irreps
    assert 3 * o2.Irrep("1mo") == o2.Irreps("3x1mo")
    assert o2.Irrep("0ee") + o2.Irrep("1mo") == o2.Irreps("0ee+1mo")
    assert 0 * irreps == o2.Irreps()
    with pytest.raises(AttributeError, match="immutable"):
        irreps._irreps = ()

    module = o2.Linear(irreps, irreps)
    features = irreps.randn(4, -1)
    torch.testing.assert_close(deepcopy(module)(features), module(features))
    torch.testing.assert_close(
        pickle.loads(pickle.dumps(module))(features), module(features)
    )


def test_o2_irrep_products_and_restriction():
    assert o2.Irrep("0o") * o2.Irrep("0o") == (o2.Irrep("0e"),)
    assert o2.Irrep("1m") * o2.Irrep("2m") == (
        o2.Irrep("1m"),
        o2.Irrep("3m"),
    )
    assert o2.Irrep("2m") * o2.Irrep("2m") == (
        o2.Irrep("0e"),
        o2.Irrep("0o"),
        o2.Irrep("4m"),
    )
    assert o2.Irrep("0oo") * o2.Irrep("1mo") == (o2.Irrep("1me"),)
    assert o2.Irrep("1mo") * o2.Irrep("1mo") == (
        o2.Irrep("0ee"),
        o2.Irrep("0oe"),
        o2.Irrep("2me"),
    )
    assert o2.Irrep("1me") * o2.Irrep("2mo") == (
        o2.Irrep("1mo"),
        o2.Irrep("3mo"),
    )
    assert o2.LocalFrame.restrict("2x1e+0o") == o2.Irreps("3x0o+2x1m")


@pytest.mark.parametrize("normalization", ["component", "norm"])
def test_o2_irreps_randn_uses_flattened_ir_mul(normalization):
    irreps = o2.Irreps("2x0e+0o+3x1m+2m")
    sample = irreps.randn(
        5,
        -1,
        4,
        normalization=normalization,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    assert sample.shape == (5, irreps.dim, 4)
    if normalization == "norm":
        for (ir, mul), ir_slice in zip(irreps, irreps.slices()):
            values = sample[:, ir_slice].reshape(5, ir.dim, mul, 4)
            torch.testing.assert_close(
                values.norm(dim=1),
                torch.ones(5, mul, 4, dtype=DTYPE, device=DEVICE),
            )


def test_o2_direct_sum_matrix_uses_ir_mul_layout():
    irreps = o2.Irreps("0e+0o+2x1m")
    angle = torch.tensor([0.2, -0.7], dtype=DTYPE, device=DEVICE)
    actual = irreps.D_from_angle(angle, reflected=True)
    one = o2.Irrep("1m").D_from_angle(angle, reflected=True)
    identity = torch.eye(2, dtype=DTYPE, device=DEVICE)
    expected = torch.einsum("bij,uv->biujv", one, identity).reshape(2, 4, 4)

    torch.testing.assert_close(actual[:, 0, 0], torch.ones_like(angle))
    torch.testing.assert_close(actual[:, 1, 1], -torch.ones_like(angle))
    torch.testing.assert_close(actual[:, 2:, 2:], expected)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("reflected", [False, True])
def test_circular_harmonics_is_equivariant(normalize, reflected):
    module = o2.CircularHarmonics(4, normalize=normalize).to(DEVICE, DTYPE)
    vectors = torch.randn(9, 2, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.37, dtype=DTYPE, device=DEVICE)
    transformed = _transform(vectors, module.irreps_in, angle, reflected)
    expected = _transform(module(vectors), module.irreps_out, angle, reflected)
    torch.testing.assert_close(module(transformed), expected)


@pytest.mark.parametrize("reflected", [False, True])
def test_o2_linear_is_equivariant(reflected):
    module = o2.Linear(
        "2x0e+3x1m",
        "4x0e+2x1m+0o",
        biases=True,
    ).to(DEVICE, DTYPE)
    features = module.irreps_in.randn(6, -1, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.41, dtype=DTYPE, device=DEVICE)
    expected = _transform(module(features), module.irreps_out, angle, reflected)
    actual = module(_transform(features, module.irreps_in, angle, reflected))
    torch.testing.assert_close(actual, expected)


def test_o2_linear_external_weights_broadcast_and_zero_pad():
    module = o2.Linear(
        "2x0e+3x1m",
        "4x0e+2x1m+0o",
        internal_weights=False,
        shared_weights=False,
    ).to(DEVICE, DTYPE)
    features = module.irreps_in.randn(5, -1, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(5, module.weight_numel, dtype=DTYPE, device=DEVICE)
    output = module(features, weights)
    reference = torch.stack(
        [module(features[index], weights[index]) for index in range(5)]
    )
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(output[:, -1], torch.zeros_like(output[:, -1]))

    singleton = torch.randn(1, module.weight_numel, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        module(features, singleton), module(features, singleton[0])
    )


@pytest.mark.parametrize("batch_size", [0, 5])
@pytest.mark.parametrize("weight_batch", ["shared", "singleton", "edge"])
def test_o2_uu_linear_matches_channel_diagonal_linear(
    double_precision, batch_size, weight_batch
):
    module = o2.UuLinear(
        "4x0ee+2x0ee+4x0oe+4x1me+2x1mo",
        "4x0ee+2x0oe+2x1me+4x1mo+2x2me",
        num_channel=2,
    ).to(DEVICE, DTYPE)
    reference = o2.Linear(
        module.irreps_in,
        module.irreps_out,
        internal_weights=False,
        shared_weights=False,
    ).to(DEVICE, DTYPE)
    features = module.irreps_in.randn(
        batch_size,
        -1,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )
    leading = (
        ()
        if weight_batch == "shared"
        else (1 if weight_batch == "singleton" else batch_size,)
    )
    weight = torch.randn(
        *leading,
        module.weight_numel,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )
    matrices = []
    offset = 0
    for instruction in module.instructions:
        copies_in, copies_out, channels = instruction.path_shape
        size = copies_in * copies_out * channels
        matrix = weight[..., offset : offset + size].reshape(
            *leading, copies_in, copies_out, channels
        )
        matrix = torch.einsum(
            "...iju,uv->...iujv",
            matrix,
            torch.eye(channels, device=DEVICE, dtype=DTYPE),
        )
        matrices.append(
            matrix.reshape(*leading, copies_in * copies_out * channels**2)
            * channels**0.5
        )
        offset += size
    expected = reference(features, torch.cat(matrices, dim=-1))
    actual = module(features, weight)
    torch.testing.assert_close(actual, expected)
    assert list(module.parameters()) == []
    for _ in range(2):
        gradients = [
            torch.autograd.grad(
                output.square().sum(),
                (features, weight),
                create_graph=True,
                retain_graph=True,
            )
            for output in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)
        actual, expected = [
            torch.cat([g.flatten() for g in grads]) for grads in gradients
        ]

    angle = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)
    for reflected, time_reversal in ((False, False), (True, False), (False, True)):
        torch.testing.assert_close(
            module(
                _transform(features, module.irreps_in, angle, reflected, time_reversal),
                weight,
            ),
            _transform(
                module(features, weight),
                module.irreps_out,
                angle,
                reflected,
                time_reversal,
            ),
        )


def test_o2_uu_linear_compiles_with_dynamic_batches(double_precision):
    module = o2.UuLinear("4x0e+4x1m", "2x0e+6x1m+2x0o", 2)
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for batch_size in (3, 7, 0):
        features = torch.randn(batch_size, module.irreps_in.dim, requires_grad=True)
        weight = torch.randn(batch_size, module.weight_numel, requires_grad=True)
        actual, expected = compiled(features, weight), module(features, weight)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), (features, weight)),
            torch.autograd.grad(expected.square().sum(), (features, weight)),
        ):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("reflected", [False, True])
def test_o2_gate_is_equivariant(reflected):
    module = o2.Gate(
        "2x0e+0o",
        [torch.nn.SiLU(), torch.nn.Tanh()],
        "3x0e",
        [torch.nn.Sigmoid()],
        "3x1m",
    ).to(DEVICE, DTYPE)
    features = module.irreps_in.randn(5, -1, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.29, dtype=DTYPE, device=DEVICE)
    expected = _transform(module(features), module.irreps_out, angle, reflected)
    actual = module(_transform(features, module.irreps_in, angle, reflected))
    torch.testing.assert_close(actual, expected)


def test_o2_odd_scalar_activation_is_odd():
    activation = o2.Activation("2x0o", [torch.nn.Tanh()])
    features = torch.randn(4, 2, dtype=DTYPE)
    torch.testing.assert_close(activation(-features), -activation(features))


def _tensor_product_case(mode):
    if mode == "u1u":
        return "2x1m", "0e", "2x1m"
    if mode == "uuu":
        return "2x1m", "2x1m", "2x0e"
    return "2x1m", "3x1m", "4x0e"


@pytest.mark.parametrize("mode", ["u1u", "uuu", "uvw"])
@pytest.mark.parametrize("reflected", [False, True])
def test_o2_tensor_product_is_equivariant(mode, reflected):
    irreps_in1, irreps_in2, irreps_out = _tensor_product_case(mode)
    module = o2.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        [(0, 0, 0, mode, True)],
        internal_weights=False,
        shared_weights=False,
    ).to(DEVICE, DTYPE)
    input1 = module.irreps_in1.randn(5, -1, dtype=DTYPE, device=DEVICE)
    input2 = module.irreps_in2.randn(5, -1, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(5, module.weight_numel, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.37, dtype=DTYPE, device=DEVICE)
    output = module(input1, input2, weights)
    expected = _transform(output, module.irreps_out, angle, reflected)
    actual = module(
        _transform(input1, module.irreps_in1, angle, reflected),
        _transform(input2, module.irreps_in2, angle, reflected),
        weights,
    )
    torch.testing.assert_close(actual, expected)


def test_o2_tensor_product_zero_pads_missing_outputs():
    module = o2.TensorProduct(
        "2x1m",
        "0e",
        "2x1m+0o",
        [(0, 0, 0, "u1u", False)],
    )
    output = module(torch.randn(3, 4), torch.randn(3, 1))
    torch.testing.assert_close(output[:, -1], torch.zeros_like(output[:, -1]))


def _asymmetric_contractions(correlation=3, path_mode="sum"):
    irreps_in = o2.Irreps("2x0e+2x0o+2x1m")
    irreps_out = o2.Irreps("2x0e+2x0o+2x1m+2x2m")
    recursive = o2.AsymmetricContraction(
        irreps_in,
        irreps_out,
        correlation,
        algorithm="recursive",
        path_mode=path_mode,
    ).to(DEVICE, DTYPE)
    dense = o2.AsymmetricContraction(
        irreps_in,
        irreps_out,
        correlation,
        algorithm="dense",
        path_mode=path_mode,
    ).to(DEVICE, DTYPE)
    return recursive, dense


@pytest.mark.parametrize("correlation", [2, 3])
@pytest.mark.parametrize("path_mode", ["sum", "expand"])
@pytest.mark.parametrize("batch_size", [0, 4])
def test_o2_asymmetric_contraction_algorithms_match(correlation, path_mode, batch_size):
    recursive, dense = _asymmetric_contractions(correlation, path_mode)
    inputs = [
        recursive.irreps_in.randn(
            batch_size, -1, dtype=DTYPE, device=DEVICE, requires_grad=True
        )
        for _ in range(correlation)
    ]
    weights = torch.randn(
        batch_size,
        recursive.weight_numel,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    assert recursive.order_num_paths == dense.order_num_paths
    actual, expected = recursive(inputs, weights), dense(inputs, weights)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (*inputs, weights)),
        torch.autograd.grad(expected.square().sum(), (*inputs, weights)),
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("algorithm", ["recursive", "dense"])
@pytest.mark.parametrize("reflected", [False, True])
def test_o2_asymmetric_contraction_is_equivariant(algorithm, reflected):
    module = o2.AsymmetricContraction(
        "2x0e+2x0o+2x1m",
        "2x0e+2x0o+2x1m+2x2m",
        2,
        algorithm=algorithm,
    ).to(DEVICE, DTYPE)
    inputs = [
        module.irreps_in.randn(4, -1, dtype=DTYPE, device=DEVICE) for _ in range(2)
    ]
    weights = torch.randn(4, module.weight_numel, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.31, dtype=DTYPE, device=DEVICE)
    output = module(inputs, weights)
    expected = _transform(output, module.irreps_out, angle, reflected)
    transformed = [
        _transform(features, module.irreps_in, angle, reflected) for features in inputs
    ]
    torch.testing.assert_close(module(transformed, weights), expected)


@pytest.mark.parametrize("path_normalization", ["element", "path"])
@pytest.mark.parametrize("shared_weights", [False, True])
def test_o2_scalar_linear_matches_o3_normalization(
    double_precision, path_normalization, shared_weights
):
    irreps_in, irreps_out = "2x0e+3x0e+2x0o", "4x0e+2x0o"
    kwargs = dict(
        internal_weights=False,
        shared_weights=shared_weights,
        biases=shared_weights,
        path_normalization=path_normalization,
    )
    module = o2.Linear(irreps_in, irreps_out, **kwargs)
    reference = o3.Linear(irreps_in, irreps_out, **kwargs)
    assert module.weight_numel == reference.weight_numel
    x = torch.randn(5, module.irreps_in.dim, requires_grad=True)
    weight_shape = (
        (module.weight_numel,) if shared_weights else (5, module.weight_numel)
    )
    weight = torch.randn(weight_shape, requires_grad=True)
    bias = (
        torch.randn(module.bias_numel, requires_grad=True) if shared_weights else None
    )
    actual, expected = module(x, weight, bias), reference(x, weight, bias)
    torch.testing.assert_close(actual, expected)
    inputs = (x, weight, bias) if shared_weights else (x, weight)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), inputs),
        torch.autograd.grad(expected.square().sum(), inputs),
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("batch_size", [0, 3])
def test_o2_linear_broadcasts_bias_and_unconnected_outputs(
    double_precision, batch_size
):
    module = o2.Linear(
        "2x0e+1m",
        "0e+1m+0o",
        internal_weights=False,
        shared_weights=False,
        biases=True,
    )
    x = torch.randn(1, 1, module.irreps_in.dim, requires_grad=True)
    weight = torch.randn(2, 1, module.weight_numel, requires_grad=True)
    bias = torch.randn(1, batch_size, module.bias_numel, requires_grad=True)
    output = module(x, weight, bias)
    expected = module(
        x.expand(2, batch_size, -1),
        weight.expand(2, batch_size, -1),
        bias.expand(2, batch_size, -1),
    )
    assert output.shape == (2, batch_size, module.irreps_out.dim)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(output[..., -1], torch.zeros_like(output[..., -1]))
    for grad in torch.autograd.grad(output.square().sum(), (x, weight, bias)):
        assert torch.isfinite(grad).all()


def test_o2_gated_linear_compiles_with_dynamic_batches(double_precision):
    gate = o2.Gate(
        "2x0ee+0oo",
        [torch.nn.SiLU(), torch.tanh],
        "4x0ee",
        [torch.sigmoid],
        "2x1me+2x1mo",
    )
    linear_up = o2.Linear(
        "2x0ee+0oo+2x1me+2x1mo",
        gate.irreps_in,
        internal_weights=False,
        shared_weights=False,
    )
    linear_down = o2.Linear(gate.irreps_out, linear_up.irreps_in)

    def forward(features, weight):
        return linear_down(gate(linear_up(features, weight)))

    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True, dynamic=True)
    for batch_size in (3, 1, 0):
        features = linear_up.irreps_in.randn(batch_size, -1, requires_grad=True)
        weight = torch.randn(batch_size, linear_up.weight_numel, requires_grad=True)
        actual, expected = compiled(features, weight), forward(features, weight)
        torch.testing.assert_close(actual, expected)
        inputs = (features, weight, linear_down.weight)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), inputs),
            torch.autograd.grad(expected.square().sum(), inputs),
        ):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("irreps_out", ["", "0o"])
def test_o2_disconnected_operators_have_zero_gradients(double_precision, irreps_out):
    linear = o2.Linear("0e", irreps_out, instructions=[])
    tp = o2.TensorProduct("0e", "0e", irreps_out, [])
    for batch_size in (3, 0):
        x = torch.randn(batch_size, 1, requires_grad=True)
        y = torch.randn(batch_size, 1, requires_grad=True)
        for output, inputs in ((linear(x), (x,)), (tp(x, y), (x, y))):
            assert output.shape == (batch_size, linear.irreps_out.dim)
            torch.testing.assert_close(output, torch.zeros_like(output))
            for grad in torch.autograd.grad(output.sum(), inputs):
                torch.testing.assert_close(grad, torch.zeros_like(grad))


@pytest.mark.parametrize(
    ("scalars", "gates", "gated"),
    [("", "", ""), ("0o", "", ""), ("", "0o", "1m")],
)
def test_o2_gate_supports_empty_sectors(double_precision, scalars, gates, gated):
    module = o2.Gate(
        scalars,
        [torch.tanh] if scalars else [],
        gates,
        [torch.tanh] if gates else [],
        gated,
    )
    for batch_size in (4, 0):
        features = module.irreps_in.randn(batch_size, -1, requires_grad=True)
        output = module(features)
        assert output.shape == (batch_size, module.irreps_out.dim)
        transformed = _transform(features, module.irreps_in, torch.tensor(0.3), True)
        torch.testing.assert_close(
            module(transformed),
            _transform(output, module.irreps_out, torch.tensor(0.3), True),
        )
        (grad,) = torch.autograd.grad(output.sum(), (features,))
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize("normalization", ["component", "norm", "none"])
@pytest.mark.parametrize(
    ("ir1", "ir2", "ir_out"),
    [
        ("0o", "1m", "1m"),
        ("1m", "1m", "0e"),
        ("1m", "1m", "0o"),
        ("1m", "2m", "1m"),
        ("1m", "2m", "3m"),
    ],
)
def test_o2_tensor_product_coupling_normalization(
    double_precision, normalization, ir1, ir2, ir_out
):
    ir1, ir2, ir_out = o2.Irrep(ir1), o2.Irrep(ir2), o2.Irrep(ir_out)
    module = o2.TensorProduct(
        ir1,
        ir2,
        ir_out,
        [(0, 0, 0, "uuu", False)],
        irrep_normalization=normalization,
        path_normalization="none",
    )
    coefficients = module(torch.eye(ir1.dim)[:, None], torch.eye(ir2.dim)[None, :])
    squared_norm = {"component": ir_out.dim, "norm": ir1.dim * ir2.dim, "none": 1}
    torch.testing.assert_close(
        coefficients.square().sum(), torch.tensor(float(squared_norm[normalization]))
    )


@pytest.mark.parametrize("batch_size", [0, 4])
@pytest.mark.parametrize(
    ("ir1", "ir2"), [("0oo", "1mo"), ("1me", "1mo"), ("1mo", "2me")]
)
def test_o2_uuu_matches_diagonal_uvw(double_precision, batch_size, ir1, ir2):
    channels = 16
    ir1, ir2 = o2.Irrep(ir1), o2.Irrep(ir2)
    irreps_out = o2.Irreps([(ir, channels) for ir in ir1 * ir2])
    kwargs = dict(
        internal_weights=False, shared_weights=False, path_normalization="none"
    )
    module = o2.TensorProduct(
        channels * ir1,
        channels * ir2,
        irreps_out,
        [(0, 0, i, "uuu", True) for i in range(len(irreps_out))],
        **kwargs,
    )
    reference = o2.TensorProduct(
        channels * ir1,
        channels * ir2,
        irreps_out,
        [(0, 0, i, "uvw", True) for i in range(len(irreps_out))],
        **kwargs,
    )
    x = torch.randn(batch_size, ir1.dim * channels, requires_grad=True)
    y = torch.randn(batch_size, ir2.dim * channels, requires_grad=True)
    weight = torch.randn(batch_size, module.weight_numel, requires_grad=True)
    reference_weight = torch.einsum(
        "bpu,uv,uw->bpuvw",
        weight.reshape(batch_size, len(irreps_out), channels),
        torch.eye(channels),
        torch.eye(channels),
    ).flatten(-4)
    actual, expected = module(x, y, weight), reference(x, y, reference_weight)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (x, y, weight)),
        torch.autograd.grad(expected.square().sum(), (x, y, weight)),
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize(("lmax", "mmax"), [(0, 0), (1, 1), (3, 1), (4, 2), (3, 3)])
def test_wigner_matches_o3_rotation_matrices(double_precision, lmax, mmax):
    vectors = torch.randn(3, 3, generator=torch.Generator().manual_seed(7))
    rotation = o2.rotation_matrix_to_y_axis(vectors)
    module = o2.WignerD(mmax, lmax)
    actual, inverse = module(vectors)
    full = torch.stack(
        [
            torch.block_diag(
                *[o3.Irrep(l, 1).D_from_matrix(r) for l in range(lmax + 1)]
            )
            for r in rotation
        ]
    )
    expected = full.index_select(1, module.local_indices)
    # The reference converts the rotation matrix through Euler angles.
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(
        inverse, expected.transpose(1, 2) * module.inverse_scale, atol=2e-9, rtol=2e-9
    )
    assert not module.state_dict()
    empty, empty_inverse = module(vectors[:0])
    assert empty.shape == (0, *actual.shape[1:])
    assert empty_inverse.shape == (0, *inverse.shape[1:])


@pytest.mark.parametrize("lmax", [2, 4])
def test_wigner_recursive_derivatives(double_precision, lmax):
    module = o2.WignerD(lmax, lmax, method="recursive")
    vectors = torch.randn(2, 3, requires_grad=True)
    assert torch.autograd.gradcheck(module.forward_packed, (vectors,), fast_mode=True)
    assert torch.autograd.gradgradcheck(
        module.forward_packed, (vectors,), fast_mode=True
    )


def test_time_odd_circular_harmonics_alternate_time_parity():
    harmonics = o2.CircularHarmonics(4, time_reversal=True).to(DEVICE, DTYPE)
    assert harmonics.irreps_in == o2.Irreps("1mo")
    assert harmonics.irreps_out == o2.Irreps("0ee+1mo+2me+3mo+4me")
    vectors = torch.randn(8, 2, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        harmonics(-vectors),
        _time_reverse(harmonics(vectors), harmonics.irreps_out),
    )


def test_o2_linear_preserves_time_parity():
    irreps = o2.Irreps("2x0ee+3x0eo+2x1me+3x1mo")
    linear = o2.Linear(irreps, irreps, biases=True).to(DEVICE, DTYPE)
    assert linear.bias_numel == 2
    assert all(
        linear.irreps_in[instruction.i_in].ir.t
        == linear.irreps_out[instruction.i_out].ir.t
        for instruction in linear._weight_instructions
    )
    features = irreps.randn(5, -1, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        linear(_time_reverse(features, irreps)),
        _time_reverse(linear(features), irreps),
    )


def test_o2_activation_and_gate_preserve_time_parity():
    odd_activation = o2.Activation("3x0eo", [torch.nn.Tanh()])
    features = torch.randn(4, 3, dtype=DTYPE)
    torch.testing.assert_close(odd_activation(-features), -odd_activation(features))

    gate = o2.Gate(
        "2x0ee",
        [torch.nn.SiLU()],
        "3x0oo",
        [torch.nn.Tanh()],
        "3x1mo",
    ).to(DEVICE, DTYPE)
    features = gate.irreps_in.randn(7, -1, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        gate(_time_reverse(features, gate.irreps_in)),
        _time_reverse(gate(features), gate.irreps_out),
    )


def test_o2_tensor_product_preserves_time_parity():
    irreps_in = o2.Irreps("2x1mo")
    irreps_out = o2.Irreps("2x0ee+2x0oe+2x2me")
    tensor_product = o2.TensorProduct(
        irreps_in,
        irreps_in,
        irreps_out,
        [(0, 0, i_out, "uuu", False) for i_out in range(len(irreps_out))],
    ).to(DEVICE, DTYPE)
    first = irreps_in.randn(6, -1, dtype=DTYPE, device=DEVICE)
    second = irreps_in.randn(6, -1, dtype=DTYPE, device=DEVICE)
    output = tensor_product(first, second)
    torch.testing.assert_close(
        tensor_product(
            _time_reverse(first, irreps_in),
            _time_reverse(second, irreps_in),
        ),
        _time_reverse(output, irreps_out),
    )


def test_o2_asymmetric_contraction_preserves_time_parity():
    irreps_in = o2.Irreps("2x0ee+2x0oo+2x1mo")
    contraction = o2.AsymmetricContraction(
        irreps_in,
        "2x0ee+2x1me",
        correlation=2,
        algorithm="recursive",
    ).to(DEVICE, DTYPE)
    inputs = [irreps_in.randn(5, -1, dtype=DTYPE, device=DEVICE) for _ in range(2)]
    weights = torch.randn(5, contraction.weight_numel, dtype=DTYPE, device=DEVICE)
    output = contraction(inputs, weights)
    torch.testing.assert_close(
        contraction([_time_reverse(value, irreps_in) for value in inputs], weights),
        _time_reverse(output, contraction.irreps_out),
    )


@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed e3nn does not expose time-reversal irreps.",
)
@pytest.mark.parametrize("basis_change", [True, False])
def test_local_frame_preserves_time_parity(basis_change):
    irreps = o3.Irreps("2x1eo+2x2ee")
    frame = o2.LocalFrame(irreps, basis_change=basis_change).to(DEVICE, DTYPE)
    edge_vectors = torch.randn(6, 3, dtype=DTYPE, device=DEVICE)
    wigner, _ = o2.WignerD(mmax=2, lmax=2).to(DEVICE, DTYPE)(edge_vectors)
    features = torch.randn(6, irreps.dim, dtype=DTYPE, device=DEVICE)
    local = frame.to_local(features, wigner)
    torch.testing.assert_close(
        frame.to_local(_time_reverse(features, irreps), wigner),
        _time_reverse(local, frame.irreps_out),
    )


@pytest.mark.parametrize("mode", ["uvu", "uvw"])
@pytest.mark.parametrize("normalization", ["component", "integral", "norm"])
@pytest.mark.parametrize("batch_size", [0, 5])
def test_o3_tensor_product_matches_edge_cgtp(
    double_precision, mode, normalization, batch_size
):
    from tace.models._e3nn.paths import generate_paths

    irreps_in = o3.Irreps("2x0e+2x1o+3x1e+3x2o+2x3e")
    irreps_sh = o3.Irreps.spherical_harmonics(4)
    irreps_out = o3.Irreps("2x0e+2x0o+2x1e+2x1o+2x2e+2x2o+2x3e+2x3o")
    instructions, irreps_out = generate_paths(
        irreps_out,
        irreps_in,
        irreps_sh,
        e3nn_mode=mode,
        trainable=True,
    )
    kwargs = dict(internal_weights=False, shared_weights=False)
    reference = o3.TensorProduct(
        irreps_in, irreps_sh, irreps_out, instructions, **kwargs
    )
    module = o2.O3TensorProduct(
        irreps_in,
        irreps_sh,
        irreps_out,
        instructions,
        normalization=normalization,
        **kwargs,
    )
    assert not module.local_frame_in.basis_change
    assert not module.local_frame_out.basis_change
    x = torch.randn(batch_size, irreps_in.dim, requires_grad=True)
    r = torch.randn(batch_size, 3, requires_grad=True)
    w = torch.randn(batch_size, module.weight_numel, requires_grad=True)
    layout_in = LayoutTransform(
        irreps_in,
        layout_in="flatten_mul_ir",
        layout_out="flatten_ir_mul",
    )
    layout_out = LayoutTransform(
        irreps_out,
        layout_in="flatten_ir_mul",
        layout_out="flatten_mul_ir",
    )
    d, di = o2.WignerD(3, 3)(r)
    actual = layout_out(module(layout_in(x), d, di, w))
    expected = reference(
        x, o3.spherical_harmonics(irreps_sh, r, True, normalization), w
    )
    assert module.weight_numel == reference.weight_numel
    assert not any(isinstance(child, o3.TensorProduct) for child in module.modules())
    torch.testing.assert_close(module.output_mask, reference.output_mask)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    if batch_size:
        for actual_grad, expected_grad in zip(
            torch.autograd.grad(actual.square().sum(), (x, r, w)),
            torch.autograd.grad(expected.square().sum(), (x, r, w)),
        ):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-8, rtol=2e-9)


@pytest.mark.parametrize("irrep_normalization", ["component", "norm", "none"])
@pytest.mark.parametrize("path_normalization", ["element", "path", "none"])
def test_o3_tensor_product_normalization_and_second_derivatives(
    double_precision,
    irrep_normalization,
    path_normalization,
):
    irreps_in = o3.Irreps("2x1o")
    irreps_sh = o3.Irreps.spherical_harmonics(2)
    irreps_out = o3.Irreps("2x1o+2x1e+2x2e+2x0o")
    instructions = [
        (0, 0, 0, "uvu", True, 0.7),
        (0, 2, 0, "uvu", True, 1.3),
        (0, 1, 1, "uvu", False, 0.9),
        (0, 1, 2, "uvu", True, 1.1),
    ]
    kwargs = dict(
        irrep_normalization=irrep_normalization,
        path_normalization=path_normalization,
        in1_var=[0.8],
        in2_var=[1.2, 0.9, 1.5],
        out_var=[0.7, 1.1, 1.3, 1.0],
        internal_weights=False,
        shared_weights=False,
    )
    reference = o3.TensorProduct(
        irreps_in, irreps_sh, irreps_out, instructions, **kwargs
    )
    module = o2.O3TensorProduct(
        irreps_in, irreps_sh, irreps_out, instructions, **kwargs
    )
    x = torch.randn(4, irreps_in.dim, requires_grad=True)
    r = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(1, module.weight_numel, requires_grad=True)
    layout_in = LayoutTransform(
        irreps_in, layout_in="flatten_mul_ir", layout_out="flatten_ir_mul"
    )
    layout_out = LayoutTransform(
        irreps_out, layout_in="flatten_ir_mul", layout_out="flatten_mul_ir"
    )
    d, di = o2.WignerD(2, 2)(r)
    # A nonunit input vector tests the optional per-degree amplitude and its gradient.
    scale = r.square().sum(-1, keepdim=True).sqrt().pow(torch.arange(3))
    actual = layout_out(module(layout_in(x), d, di, weight, scale))
    expected = reference(
        x, o3.spherical_harmonics(irreps_sh, r, False, "component"), weight
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    grads = [
        torch.autograd.grad(value.square().sum(), (x, r, weight), create_graph=True)
        for value in (actual, expected)
    ]
    for actual_grad, expected_grad in zip(*grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-8, rtol=2e-9)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(grads[0][1].square().sum(), (x, r, weight)),
        torch.autograd.grad(grads[1][1].square().sum(), (x, r, weight)),
    ):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-8)


def test_o3_tensor_product_internal_weights_axes_and_truncated_frames(double_precision):
    module = o2.O3TensorProduct(
        "2x1o",
        "1x1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
    )
    reference = o3.TensorProduct(
        module.irreps_in1,
        module.irreps_in2,
        module.irreps_out,
        [(0, 0, i, "uvu", True) for i in range(3)],
    )
    r = torch.cat((torch.eye(3), -torch.eye(3)))
    x = torch.randn(6, 3, 6)
    layout_in = LayoutTransform(
        module.irreps_in1, layout_in="flatten_mul_ir", layout_out="flatten_ir_mul"
    )
    layout_out = LayoutTransform(
        module.irreps_out, layout_in="flatten_ir_mul", layout_out="flatten_mul_ir"
    )
    d, di = o2.WignerD(2, 2)(r)
    actual = layout_out(module(layout_in(x), d, di))
    expected = reference(
        x, o3.spherical_harmonics([1], r, True, "component")[:, None], module.weight
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    d, di = o2.WignerD(1, 2)(r)
    with pytest.raises(ValueError, match="orders"):
        module(layout_in(x), d, di)
    with pytest.raises(ValueError, match="spherical harmonics"):
        o2.O3TensorProduct("1o", "1e", "0o", [(0, 0, 0, "uvu", True)])


@pytest.mark.parametrize("wigner_lmax", [2, 4])
def test_o3_tensor_product_compiles_with_dynamic_and_empty_batches(
    double_precision, wigner_lmax
):
    module = o2.O3TensorProduct(
        "2x1o",
        "1o",
        "2x0e+2x1e+2x2e",
        [(0, 0, i, "uvu", True) for i in range(3)],
        internal_weights=False,
        shared_weights=False,
    )
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for batch_size in (3, 1, 0):
        x = torch.randn(batch_size, 6, requires_grad=True)
        w = torch.randn(batch_size, module.weight_numel, requires_grad=True)
        d, di = o2.WignerD(2, wigner_lmax)(torch.randn(batch_size, 3))
        actual, expected = compiled(x, d, di, w), module(x, d, di, w)
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(
            torch.autograd.grad(actual.square().sum(), (x, w)),
            torch.autograd.grad(expected.square().sum(), (x, w)),
        ):
            torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed O(3) irreps do not expose time-reversal parity.",
)
def test_o3_tensor_product_time_reversal(double_precision):
    from tace.models.time_reversal import with_time_reversal

    irreps_in = with_time_reversal(o3.Irreps("2x1e"), -1)
    irreps_out = with_time_reversal(o3.Irreps("2x0o+2x1o+2x2o"), -1)
    module = o2.O3TensorProduct(
        irreps_in,
        o3.Irreps.spherical_harmonics(1)[1:],
        irreps_out,
        [(0, 0, i, "uvu", True) for i in range(3)],
    )
    x = torch.randn(4, irreps_in.dim)
    d, di = o2.WignerD(2, 2)(torch.randn(4, 3))
    torch.testing.assert_close(module(-x, d, di), -module(x, d, di))
