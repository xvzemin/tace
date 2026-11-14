"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3
from equivariantx import o2


@pytest.mark.parametrize("degree,mmax", [(0, 0), (1, 1), (2, 1), (3, 2), (5, 2)])
@pytest.mark.parametrize("inverse", [False, True])
def test_polynomial_program(degree, mmax, inverse, double_precision):
    from equivariantx.conv.uv_o2.transverse import TransverseFrame
    from equivariantx.ops.common.torch.edge import evaluate_torch
    from equivariantx.utils.program import Program

    frame = o2.LocalFrame(f"2x{degree}e+3x{degree}o", mmax)
    module = TransverseFrame(frame)
    width = module.output_dim if inverse else module.input_dim
    features = torch.randn(4, width, requires_grad=True)
    vectors = torch.cat((torch.eye(3), -torch.eye(3))).requires_grad_()
    source, target = torch.arange(6) % 4, torch.arange(6) % 3
    program = Program()
    output = module.program(
        program,
        program.input(0, "source", width),
        program.input(1, "edge", 3),
        inverse=inverse,
        native=False,
    )
    metadata = repr((tuple(program.nodes), ((output, 0, "target"),)))
    direction = vectors / vectors.norm(dim=-1, keepdim=True)
    actual = evaluate_torch(metadata, [features, direction], source, target, 3)[0]
    expected = module(features, direction, source, target, 3, inverse=inverse)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    if not degree:
        return
    a, b = actual.sin().sum(), expected.sin().sum()
    tangents = torch.randn_like(features) / 10, torch.randn_like(vectors) / 10
    for _ in range(4):
        ga, gb = [
            torch.autograd.grad(value, (features, vectors), create_graph=True)
            for value in (a, b)
        ]
        torch.testing.assert_close(ga, gb, atol=2e-9, rtol=2e-9)
        a, b = [sum((g * v).sum() for g, v in zip(gs, tangents)) for gs in (ga, gb)]


@pytest.mark.parametrize("degree,mmax", [(5, 2), (7, 7)])
def test_transverse_frame(degree, mmax, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.conv.uv_o2.transverse import TransverseFrame

    time = hasattr(o3.Irrep("0e"), "t")
    suffix = "o" if time else ""
    frame = o2.LocalFrame(
        f"1x{degree}e{suffix}+2x{degree}o{suffix}+3x{degree}e{suffix}", mmax
    ).cuda()
    vectors = (
        torch.cat((torch.eye(3), -torch.eye(3), torch.randn(2, 3)))
        .cuda()
        .requires_grad_()
    )
    direction = vectors / vectors.norm(dim=-1, keepdim=True)
    features = torch.randn(8, frame.input_dim, device="cuda", requires_grad=True)
    index = torch.arange(8, device="cuda")
    wigner, inverse = o2.WignerD(degree, degree).cuda()(vectors)
    compact = frame.to_local(features, wigner)
    expected = frame.to_global(compact, inverse)
    for backend in ("torch", "cuda"):
        torch.set_default_dtype(torch.float32)
        module = TransverseFrame(frame, backend).cuda().double()
        torch.set_default_dtype(torch.float64)
        transverse = module(features, direction, index, index, 8)
        for (dim, mul, start), (ir, _), section in zip(
            module.layout, frame.irreps_out, frame.irreps_out.slices()
        ):
            norm = (
                transverse[:, start : start + dim * mul]
                .reshape(8, dim, mul)
                .square()
                .sum(1)
            )
            reference = compact[:, section].reshape(8, ir.dim, mul).square().sum(1)
            torch.testing.assert_close(norm, reference, atol=3e-10, rtol=3e-10)
        actual = module(transverse, direction, index, index, 8, inverse=True)
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
        gradients = [
            torch.autograd.grad(
                value.sin().sum(), (features, vectors), retain_graph=True
            )
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-8)
