"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
from equivariantx.tests.o2.testing import _time_reverse


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


@pytest.mark.parametrize("edges", [0, 7])
def test_local_frame_indexed_torch(double_precision, edges):
    frame = o2.LocalFrame("2x0e+3x1e+1x2o", mmax=1)
    rotation = o2.WignerD(2, 2, method="recursive")
    index = torch.randint(4, (edges,))
    nodes = torch.randn(5, frame.input_dim, requires_grad=True)
    vectors = torch.randn(edges, 3, requires_grad=True)
    wigner = rotation.forward_packed(vectors)
    local = frame.gather(nodes, wigner, index)
    expected = frame.to_local(nodes[index], wigner)
    torch.testing.assert_close(local, expected)
    output = frame.scatter(local, wigner, index, 5)
    expected = nodes.new_zeros(nodes.shape).index_add(
        0, index, frame.to_global(expected, wigner)
    )
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(output[4], torch.zeros_like(output[4]))
    for a, b in zip(
        torch.autograd.grad(output.square().sum(), (nodes, vectors), retain_graph=True),
        torch.autograd.grad(expected.square().sum(), (nodes, vectors)),
    ):
        torch.testing.assert_close(a, b)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "mmax,basis_change,packed", [(0, True, True), (1, False, False), (3, True, True)]
)
def test_indexed_frame(dtype, mmax, basis_change, packed):
    from equivariantx import o2

    frame = o2.LocalFrame(
        "2x0e+3x1o+2x1e+1x3o+2x1o", mmax, basis_change=basis_change
    ).to("cuda", dtype)
    rotation = o2.WignerD(4, 4, method="recursive").to("cuda", dtype)
    for edges in (9, 0):
        nodes = torch.randn(
            6, frame.input_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        vectors = torch.randn(edges, 3, device="cuda", dtype=dtype, requires_grad=True)
        index = torch.arange(edges, device="cuda") % 5
        wigner, inverse = rotation(vectors)
        if packed:
            wigner = inverse = rotation.forward_packed(vectors)
        for inverse_map in (False, True):
            values = (
                torch.randn(
                    edges,
                    frame.output_dim,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                if inverse_map
                else nodes
            )
            outputs = [
                frame.scatter(values, inverse, index, 6, backend=backend)
                if inverse_map
                else frame.gather(values, wigner, index, backend=backend)
                for backend in ("cuda", "torch")
            ]
            for _ in range(3):
                torch.testing.assert_close(
                    *outputs,
                    rtol=3e-4 if dtype == torch.float32 else 1e-9,
                    atol=3e-5 if dtype == torch.float32 else 1e-9,
                )
                gradients = [
                    torch.autograd.grad(
                        value.sin().sum(),
                        (values, vectors),
                        create_graph=True,
                        retain_graph=True,
                    )
                    for value in outputs
                ]
                outputs = [
                    torch.cat([g.flatten() for g in grads]) / 10 for grads in gradients
                ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_indexed_o2_pipeline_compile(double_precision):
    from equivariantx import o2

    frame = o2.LocalFrame("2x0e+2x1o+2x1e", mmax=1).cuda()
    args = ("2x0e+2x0o", [torch.nn.SiLU(), torch.tanh], "4x0e", [torch.sigmoid], "4x1m")
    gates = [o2.Gate(*args, backend=backend).cuda() for backend in ("torch", "cuda")]
    up = o2.Linear(frame.irreps_out, gates[0].irreps_in).cuda()
    down = o2.Linear(gates[0].irreps_out, frame.irreps_out).cuda()

    def evaluate(nodes, wigner, index, backend):
        local = frame.gather(nodes, wigner, index[0], backend=backend)
        local = down(gates[backend == "cuda"](up(local)))
        return frame.scatter(local, wigner, index[1], nodes.size(0), backend=backend)

    torch.compiler.reset()
    compiled = torch.compile(
        evaluate, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for edges in (7, 11, 0):
        nodes = torch.randn(5, frame.input_dim, device="cuda", requires_grad=True)
        wigner = torch.randn(edges, 10, device="cuda", requires_grad=True)
        index = torch.randint(5, (2, edges), device="cuda")
        outputs = [
            compiled(nodes, wigner, index, "cuda"),
            evaluate(nodes, wigner, index, "torch"),
        ]
        torch.testing.assert_close(*outputs)
        inputs = (nodes, wigner, up.weight, down.weight)
        gradients = [
            torch.autograd.grad(value.square().sum(), inputs) for value in outputs
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)

    vectors = torch.randn(7, 3, device="cuda", requires_grad=True)
    rotation = o2.WignerD(1, 1, method="recursive").cuda()
    nodes = torch.randn(5, frame.input_dim, device="cuda", requires_grad=True)
    index = torch.randint(5, (2, 7), device="cuda")
    losses = []
    for backend in ("cuda", "torch"):
        output = evaluate(nodes, rotation.forward_packed(vectors), index, backend)
        forces = -torch.autograd.grad(
            output.square().mean(), vectors, create_graph=True
        )[0]
        losses.append(forces.square().mean())
    gradients = [
        torch.autograd.grad(loss, (nodes, up.weight, down.weight)) for loss in losses
    ]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-9)
