import pytest
import torch
from e3nn import o3

from equivariantx import co2
from equivariantx.o2 import rotation_matrix_to_y_axis
from equivariantx.ops import config


@pytest.mark.parametrize("kind", ["uu", "uv"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("radial_mode", ["full", "last"])
def test_cartesian_convolution(
    kind, backend, radial_mode, monkeypatch, double_precision
):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = co2.LocalFrame("2x0e+2x1e+2x2e", mmax=1, backend="torch")
    if kind == "uu":
        linear = co2.UuLinear(frame.irreps_out, frame.irreps_out, 2, backend="torch")
        layer = co2.UuTensorProductConv(
            frame, linear, frame, backend=backend, radial_mode=radial_mode
        )
    else:
        inputs = co2.Irreps([(2 * m, ir) for m, ir in frame.irreps_out])
        gate = co2.Gate(
            "2x0e", [torch.tanh], "2x0e", [torch.sigmoid], "2x1m", backend="torch"
        )
        up = co2.Linear(inputs, gate.irreps_in, biases=True, backend="torch")
        down = co2.Linear(
            gate.irreps_out, frame.irreps_out, biases=True, backend="torch"
        )
        hidden = (
            ((co2.Linear(gate.irreps_out, gate.irreps_in, backend="torch"), gate),)
            if radial_mode == "last"
            else ()
        )
        layer = co2.UvTensorProductConv(
            frame,
            up,
            gate,
            down,
            frame,
            hidden=hidden,
            backend=backend,
            radial_mode=radial_mode,
        )
    layer = layer.to(device)
    if radial_mode == "last":
        layer.float().double()
    net = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.SiLU()).to(device)
    edges = torch.tensor([[2, 1, 2, 0, 0, 1], [0, 0, 1, 2, 1, 1]], device=device)

    def sample(*shape):
        return torch.randn(shape, device=device).requires_grad_()

    x, rot, cut = sample(4, frame.input_dim), sample(6, 3, 3), sample(6, 1)
    source, edge = sample(4, 3), sample(6, 2)
    projection = sample(5, layer.weight_numel)
    vs = (x, rot, cut, source, edge, projection, *net.parameters(), *layer.parameters())

    monkeypatch.setattr(config, "RADIAL_PROJECT_MIN_WIDTH", 4)

    def run():
        result = layer(
            x,
            ((source, "source"), (edge, "edge")),
            edges,
            rot,
            cut,
            radial_network=net,
            projection=projection,
        )
        g = torch.autograd.grad(result.square().mean(), vs, create_graph=True)
        gg = torch.autograd.grad(sum(v.square().mean() for v in g), vs)
        return (result, *g, *gg)

    actual = run()
    empty = layer(
        x,
        ((source, "source"), (edge[:0], "edge")),
        edges[:, :0],
        rot[:0],
        cut[:0],
        radial_network=net,
        projection=projection,
    )
    torch.testing.assert_close(empty, x.new_zeros(4, frame.input_dim))
    for grad in torch.autograd.grad(empty.sum(), vs):
        torch.testing.assert_close(grad, torch.zeros_like(grad))
    monkeypatch.setattr(layer, "backend", "torch")
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    expected = run()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("kind", ["uu", "uv"])
def test_frame_independent_message(kind, device, monkeypatch, double_precision):
    labels = (
        "2x0ee+2x1eo+2x3ee+2x4oe"
        if hasattr(o3.Irrep("0e"), "t")
        else "2x0e+2x1e+2x3e+2x4o"
    )
    frame = co2.LocalFrame(labels, mmax=None if kind == "uu" else 2)
    ir = frame.irreps_out
    if kind == "uu":
        layer = co2.UuTensorProductConv(frame, co2.UuLinear(ir, ir, 2), frame)
    else:
        scalars = co2.Irreps([(m, i) for m, i in ir if not i.m])
        tensors = co2.Irreps([(m, i) for m, i in ir if i.m])
        gate = co2.Gate(
            scalars,
            [torch.tanh] * len(scalars),
            f"{sum(m for m, _ in tensors)}x0e",
            [torch.sigmoid],
            tensors,
        )
        up = co2.Linear(co2.Irreps([(2 * m, i) for m, i in ir]), gate.irreps_in)
        layer = co2.UvTensorProductConv(
            frame, up, gate, co2.Linear(gate.irreps_out, ir), frame
        )
    layer.to(device)
    edges = torch.tensor([[0, 1, 2, 0], [1, 0, 1, 2]], device=device)
    x = frame.irreps_in.randn(3, -1, device=device)
    pos = torch.randn(3, 3, device=device)
    weights = torch.randn(1, layer.weight_numel, device=device)
    cutoff = torch.rand(4, 1, device=device)

    def message(features, positions):
        rot = rotation_matrix_to_y_axis(positions[edges[1]] - positions[edges[0]])
        return layer(features, weights, edges, rot, cutoff)

    y = message(x, pos)
    q, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    for mat, reverse in ((q, False), (-q, True)):
        d = frame.irreps_in.D_from_matrix(mat, time_reversal=reverse)
        torch.testing.assert_close(
            message(x @ d.T, pos @ mat.T), y @ d.T, atol=2e-9, rtol=2e-9
        )
    x.requires_grad_()
    pos.requires_grad_()

    def derivatives():
        value = message(x, pos)
        grad = torch.autograd.grad(value.square().mean(), (x, pos), create_graph=True)
        return (
            value,
            *grad,
            *torch.autograd.grad(sum(g.square().mean() for g in grad), (x, pos)),
        )

    actual = derivatives()
    monkeypatch.setattr(layer, "backend", "torch")
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for optimized, reference in zip(actual, derivatives()):
        torch.testing.assert_close(optimized, reference, atol=2e-9, rtol=2e-8)
