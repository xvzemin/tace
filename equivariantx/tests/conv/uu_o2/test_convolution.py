"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx import conv as eqx_conv
from equivariantx import o2
from equivariantx.tests.conv.testing import assert_native_autograd


@pytest.mark.parametrize(
    "device,backend", [("cpu", "torch"), ("cuda", "torch"), ("cuda", "cuda")]
)
@pytest.mark.parametrize("direction", [False, True, "wigner"])
@pytest.mark.parametrize(
    "mmax,shared,projected,radial_channels",
    [
        (0, False, False, 3),
        (1, True, False, 3),
        (2, False, True, 3),
        (1, True, True, 3),
        (2, False, True, 1024),
        (1, True, True, 1024),
    ],
)
def test_uu_o2_convolution_derivatives(
    double_precision,
    monkeypatch,
    device,
    backend,
    direction,
    mmax,
    shared,
    projected,
    radial_channels,
):
    from equivariantx.ops import config

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if backend == "torch":
        monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    frame_in = o2.LocalFrame("2x0e+2x1o+2x1e+2x2o", mmax=mmax).to(device)
    frame_out = o2.LocalFrame("2x0o+2x1e+2x1o+2x2e", mmax=mmax).to(device)
    linear = o2.UuLinear(frame_in.irreps_out, frame_out.irreps_out, 2)
    module = eqx_conv.UuO2TensorProductConv(
        frame_in, linear, frame_out, backend=backend
    ).to(device)
    frame = o2.WignerD(2, 2, method="recursive", backend="torch").to(device)
    index = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 2, 0]], device=device)
    rows = 1 if shared else index.size(1)

    def rand(*shape):
        return (torch.randn(*shape, device=device) * 0.2).requires_grad_()

    x, vectors = rand(3, frame_in.input_dim), rand(rows, 3)
    radial = rand(rows, radial_channels if projected else linear.weight_numel)
    projection = (
        rand(radial_channels, linear.weight_numel) / radial_channels**0.5
        if projected
        else torch.empty(0, linear.weight_numel, device=device)
    )
    cutoff = rand(rows, 1)
    packed = frame.forward_packed(vectors)
    inputs = (x, vectors, radial, cutoff) + ((projection,) if projected else ())
    evaluate = module.forward_wigner if direction == "wigner" else module
    actual = evaluate(
        x,
        radial,
        projection,
        packed.detach() if direction == "wigner" else None if direction else packed,
        cutoff,
        index,
        3,
        vectors=vectors if direction else None,
    )
    wigner, inverse = frame(vectors.expand(index.size(1), -1))
    local = frame_in.to_local(x[index[0]], wigner)
    message = frame_out.to_global(
        linear(local, radial @ projection if projected else radial), inverse
    )
    expected = x.new_zeros(3, frame_out.input_dim).index_add(
        0, index[1], message * cutoff
    )
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
    if backend == "torch":
        assert_native_autograd(actual)
    for _ in range(3 if direction and mmax == 1 and projected else 2):
        gradients = [
            torch.autograd.grad(value.sin().sum(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-8)
        actual, expected = [
            torch.cat([g.flatten() for g in grads]) / 20 for grads in gradients
        ]
        if backend == "torch":
            assert_native_autograd(actual)


@pytest.mark.parametrize("channels,shared", [(3, False), (33, True)])
def test_uu_o2_chunked_reduction(monkeypatch, double_precision, channels, shared):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import equivariantx.ops.common.cuda.o2_o3.convolution as cuda
    import equivariantx.ops.common.cuda.o3.convolution as o3_cuda

    monkeypatch.setattr(cuda, "CHUNK_SIZE", 1025)
    monkeypatch.setattr(o3_cuda, "CHUNK_SIZE", 1025)
    frame_in = o2.LocalFrame(f"{channels}x1e+{channels}x1o").cuda()
    frame_out = o2.LocalFrame(f"{channels}x0o+{channels}x1o").cuda()
    linear = o2.UuLinear(frame_in.irreps_out, frame_out.irreps_out, channels)
    module = eqx_conv.UuO2TensorProductConv(
        frame_in, linear, frame_out, backend="cuda"
    ).cuda()
    reference = eqx_conv.UuO2TensorProductConv(
        frame_in, linear, frame_out, backend="torch"
    ).cuda()
    frame = o2.WignerD(1, 1).cuda()
    edges, nodes = 2051, 129
    index = torch.randint(nodes, (2, edges), device="cuda")
    rows = 1 if shared else edges
    inputs = [
        (0.2 * torch.randn(shape, device="cuda")).requires_grad_()
        for shape in (
            (nodes, frame_in.input_dim),
            (rows, 3),
            (rows, 4),
            (4, linear.weight_numel),
            (rows, 1),
        )
    ]
    x, vectors, radial, projection, cutoff = inputs
    packed = frame.forward_packed(vectors)
    args = x, radial, projection, packed, cutoff, index, nodes
    actual, expected = module(*args, vectors=vectors), reference(*args)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-10)
    for _ in range(2):
        gradients = [
            torch.autograd.grad(value.sin().mean(), inputs, create_graph=True)
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=3e-10, rtol=3e-9)
        actual, expected = [
            torch.cat([g.flatten() for g in values]) for values in gradients
        ]


def test_uu_o2_time_parity_and_duplicate_entries(double_precision):
    if not hasattr(o3.Irrep("0e"), "t"):
        pytest.skip("Time-reversal representations are unavailable")
    irreps_in = "2x0ee+2x1eo+2x1oe+2x1eo"
    irreps_out = "2x0oo+2x1oe+2x1oo+2x1eo"
    frame_in, frame_out = o2.LocalFrame(irreps_in), o2.LocalFrame(irreps_out)
    linear = o2.UuLinear(frame_in.irreps_out, frame_out.irreps_out, 2)
    module = eqx_conv.UuO2TensorProductConv(frame_in, linear, frame_out, backend="cuda")
    index = torch.tensor([[0, 1, 0], [1, 0, 1]])
    features = torch.randn(2, frame_in.input_dim)
    vectors = torch.randn(3, 3)
    weights = torch.randn(3, linear.weight_numel)
    cutoff = torch.rand(3, 1)
    frame = o2.WignerD(1, 1)
    wigner, inverse = frame(vectors)
    expected = frame_out.to_global(
        linear(frame_in.to_local(features[index[0]], wigner), weights), inverse
    )
    expected = features.new_zeros(2, frame_out.input_dim).index_add(
        0, index[1], expected * cutoff
    )
    actual = module(
        features,
        weights,
        weights.new_empty(0, linear.weight_numel),
        frame.forward_packed(vectors),
        cutoff,
        index,
        2,
        vectors=vectors,
    )
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("degree,mmax", [(5, 2), (7, 7)])
def test_uu_o2_high_degree(degree, mmax, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    frame = o2.LocalFrame(f"2x{degree}e+2x{degree}o", mmax=mmax).cuda()
    linear = o2.UuLinear(frame.irreps_out, frame.irreps_out, 2)
    # Construct in float32, then promote: constants must retain their precision.
    torch.set_default_dtype(torch.float32)
    module = (
        eqx_conv.UuO2TensorProductConv(frame, linear, frame, backend="cuda")
        .cuda()
        .double()
    )
    torch.set_default_dtype(torch.float64)
    wigner = o2.WignerD(degree, degree).cuda()
    vectors = torch.cat((torch.eye(3), -torch.eye(3), torch.randn(3, 3))).cuda()
    vectors.requires_grad_()
    features = torch.randn(3, frame.input_dim, device="cuda", requires_grad=True)
    weights = torch.randn(9, linear.weight_numel, device="cuda", requires_grad=True)
    projection = weights.new_empty((0, linear.weight_numel))
    cutoff = weights.new_ones((9, 1))
    index = torch.randint(3, (2, 9), device="cuda")
    actual = module(
        features, weights, projection, None, cutoff, index, 3, vectors=vectors
    )
    expected = module(
        features, weights, projection, wigner.forward_packed(vectors), cutoff, index, 3
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    # Check Cartesian direction derivatives against the packed rotation reference.
    for value in (actual, expected):
        assert torch.isfinite(
            torch.autograd.grad(value.square().sum(), vectors, retain_graph=True)[0]
        ).all()
    gradients = [
        torch.autograd.grad(value.square().sum(), (features, weights, vectors))
        for value in (actual, expected)
    ]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-8)


def test_uu_o2_compile_and_empty_edges(double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    frame = o2.LocalFrame("2x0e+2x1o").cuda()
    linear = o2.UuLinear(frame.irreps_out, frame.irreps_out, 2)
    module = eqx_conv.UuO2TensorProductConv(frame, linear, frame, backend="cuda").cuda()
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    for nodes, edges in ((3, 5), (3, 0), (0, 0)):
        x = torch.randn(nodes, frame.input_dim, device="cuda", requires_grad=True)
        vectors = torch.randn(edges, 3, device="cuda", requires_grad=True)
        radial = torch.randn(edges, 3, device="cuda", requires_grad=True)
        projection = torch.randn(
            3, linear.weight_numel, device="cuda", requires_grad=True
        )
        index = torch.randint(max(1, nodes), (2, edges), device="cuda")
        args = (x, radial, projection, None, radial.new_ones((edges, 1)), index, nodes)
        actual, expected = [f(*args, vectors=vectors) for f in (compiled, module)]
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
        gradients = [
            torch.autograd.grad(y.square().sum(), (x, vectors, radial, projection))
            for y in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("promoted", [False, True])
def test_reference_bypasses_wigner_child(
    device, promoted, monkeypatch, double_precision
):
    import copy

    from equivariantx.ops import config

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    dtype = torch.float32 if promoted else torch.float64
    tolerance = 3e-5 if promoted else 2e-10
    frame = o2.LocalFrame("2x0e+2x1o+2x2e", mmax=1)
    linear = o2.UuLinear(frame.irreps_out, frame.irreps_out, 2)
    module = eqx_conv.UuO2TensorProductConv(frame, linear, frame, backend="torch").to(
        device=device, dtype=dtype
    )
    reference = copy.deepcopy(module)
    module.frame.backend = "cuda"

    def forbidden(*args, **kwargs):
        raise AssertionError("The parent reference called its Wigner child.")

    monkeypatch.setattr(module.frame, "forward_packed", forbidden)
    backends = {
        name: child.backend
        for name, child in module.named_modules()
        if hasattr(child, "backend")
    }
    for edges in (5, 0):
        index = torch.stack(
            (
                torch.arange(edges, device=device) % 3,
                (torch.arange(edges, device=device) + 1) % 3,
            )
        )
        inputs = (
            torch.randn(
                3, frame.input_dim, device=device, dtype=dtype
            ).requires_grad_(),
            (
                torch.randn(edges, 3, device=device, dtype=torch.float64) + 1
            ).requires_grad_(),
            torch.randn(
                edges, linear.weight_numel, device=device, dtype=dtype
            ).requires_grad_(),
        )

        def run(layer, x, vectors, weights):
            return layer.forward_wigner(
                x,
                weights,
                weights.new_empty((0, linear.weight_numel)),
                None,
                weights.new_ones((edges, 1)),
                index,
                3,
                vectors=vectors,
            )

        a, b = run(module, *inputs), run(reference, *inputs)
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
        assert_native_autograd(a)
        for _ in range(3):
            gradients = [
                torch.autograd.grad(
                    value.sin().sum(), inputs, create_graph=True, allow_unused=True
                )
                for value in (a, b)
            ]
            gradients = [
                tuple(
                    torch.zeros_like(x) if g is None else g
                    for x, g in zip(inputs, values)
                )
                for values in gradients
            ]
            torch.testing.assert_close(*gradients, atol=tolerance, rtol=tolerance)
            a, b = [
                sum(g.sum() / max(1, g.numel()) for g in values)
                + sum(x.sum() * 0 for x in inputs)
                for values in gradients
            ]
            assert_native_autograd(a)
        tangents = tuple(torch.randn_like(x) for x in inputs)
        with torch.autograd.forward_ad.dual_level():
            duals = tuple(
                torch.autograd.forward_ad.make_dual(x, v)
                for x, v in zip(inputs, tangents)
            )
            actual = torch.autograd.forward_ad.unpack_dual(run(module, *duals))
        expected = torch.autograd.functional.jvp(
            lambda *args: run(reference, *args), inputs, tangents
        )
        assert actual.tangent is not None
        torch.testing.assert_close(
            tuple(actual), expected, atol=tolerance, rtol=tolerance
        )
        assert backends == {
            name: child.backend
            for name, child in module.named_modules()
            if hasattr(child, "backend")
        }
        assert config.TORCH_REFERENCE is True
