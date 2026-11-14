"""Tests for the corresponding EQX source module."""

import torch

from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


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


def _tensor_product_case(mode):
    if mode == "u1u":
        return "2x1m", "0e", "2x1m"
    if mode == "uuu":
        return "2x1m", "2x1m", "2x0e"
    return "2x1m", "3x1m", "4x0e"


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
        path_mode=path_mode, backend="torch",
    ).to(DEVICE, DTYPE)
    return recursive, dense


def spherical_layout(features, irreps, inverse=False):
    values = []
    for (mul, ir), section in zip(irreps, irreps.slices()):
        shape = (ir.dim, mul) if inverse else (mul, ir.dim)
        values.append(
            features[..., section]
            .reshape(*features.shape[:-1], *shape)
            .transpose(-1, -2)
            .flatten(-2)
        )
    return torch.cat(values, dim=-1)


def operator_training_problem(kind, backend, device):
    """Return one smooth force-loss chain for the corresponding O(2) module."""
    from equivariantx.o2.clebsch_gordan import clebsch_gordan_product
    from equivariantx.ops.torch.o2.clebsch_gordan import reference as coupling_reference
    from equivariantx.ops.torch.o2.rotation_matrix import (
        reference as rotation_reference,
    )

    dtype = torch.float64
    if kind == "rotation_matrix":
        x = (torch.randn(3, 3, device=device, dtype=dtype) * 0.1 + 0.5).requires_grad_()
        w = (torch.randn(3, device=device, dtype=dtype) * 0.1 + 0.8).requires_grad_()

        def forward(a, b, ordinary):
            return (
                rotation_reference(a * b, 1)
                if ordinary
                else o2.rotation_matrix_to_y_axis(a * b, backend=backend)
            )
    elif kind == "clebsch_gordan":
        x = (torch.randn(3, 2, 3, device=device, dtype=dtype) * 0.1).requires_grad_()
        w = (torch.randn(1, 2, 3, device=device, dtype=dtype) * 0.1).requires_grad_()
        ir1, ir2, ir = o2.Irrep("1m"), o2.Irrep("2m"), o2.Irrep("1m")

        def forward(a, b, ordinary):
            return (
                coupling_reference(a, ir1, b, ir2, ir, elementwise=True)
                if ordinary
                else clebsch_gordan_product(
                    a, ir1, b, ir2, ir, elementwise=True, backend=backend
                )
            )
    else:
        layer = (
            o2.SymmetricContraction("2x0e+2x1m", "2x0e+2x1m", 3, backend=backend)
            .to(device)
            .double()
        )
        reference = (
            o2.SymmetricContraction("2x0e+2x1m", "2x0e+2x1m", 3, algorithm="dense", backend="torch")
            .to(device)
            .double()
        )
        x = (
            torch.randn(3, layer.irreps_in.dim, device=device, dtype=dtype) * 0.1
        ).requires_grad_()
        w = (
            torch.randn(layer.weight_numel, device=device, dtype=dtype) * 0.1
        ).requires_grad_()

        def forward(a, b, ordinary):
            return (reference if ordinary else layer)(a, b)

    def run(a, b, ordinary=False):
        y = forward(a, b, ordinary)
        energy = 1 + y.sin().mean()
        force = -torch.autograd.grad(energy, a, create_graph=True)[0]
        loss = energy.square() + force.square().mean()
        return loss, force, torch.autograd.grad(loss, b)[0]

    return run, (x, w)


def check_joint_o2_aoti(kind, tmp_path):
    """Load a native force-loss package with Torch only and update its parameters."""
    import os
    import subprocess
    import sys

    from torch.fx.experimental.proxy_tensor import make_fx

    operation, values = operator_training_problem(kind, "cuda", "cuda")
    graph = make_fx(lambda *args: operation(*args))(*values)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    exported = torch.export.export(
        graph, tuple(t.detach() for t in values), strict=True
    )
    package = tmp_path / (kind + ".pt2")
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    records = []
    x, w = values
    for _ in range(3):
        expected = operation(x, w, True)
        records.append(
            {
                "inputs": tuple(t.detach().cpu() for t in (x, w)),
                "outputs": tuple(t.detach().cpu() for t in expected),
            }
        )
        w = (w - 0.01 * expected[-1]).detach().requires_grad_()
        x = (x.detach() * 0.95 + 0.02).requires_grad_()
    data = tmp_path / "data.pt"
    torch.save(records, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
run = torch._inductor.aoti_load_package(sys.argv[1])
for row in torch.load(sys.argv[2], weights_only=True):
    with torch.no_grad(): actual = run(*(v.cuda() for v in row['inputs']))
    torch.testing.assert_close(tuple(v.cpu() for v in actual), row['outputs'], atol=3e-9, rtol=3e-9)
assert 'equivariantx' not in sys.modules
""",
            str(package),
            str(data),
        ],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
