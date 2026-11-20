"""Measure one complete TACE interaction with and without reciprocal weight reuse.

Use TACE_USE_EQX=1 and set TACE_MEMORY_FIRST=0 (last) or 1 (full). Inputs to
each block are captured from the 3BPA O(3) model, with a nonzero element table.
Both routes include geometry, radial weights, convolution, scatter, linears,
gate and residuals. Product and readout are outside the measurement. F+B+B is
a mixed position/parameter derivative with a fixed probe, not an MD step.
The pair map is reused unless --rebuild-pairs is set. Its standalone cost is
also reported. Memory is peak allocated CUDA memory above the warm inputs.
"""

import argparse
import csv
import inspect
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
import yaml
from ase.io import read
from omegaconf import OmegaConf

from equivariantx.conv import undirected_edge_pairs
from tace.lightning.torch_model import create_model
from tace.utils.env import memory_first
from tace.utils.precision import PrecisionPolicy, configure_model_precision


def molecular_batch(path, structures, cutoff):
    atoms = read(path, f":{structures}")
    ptr = np.cumsum([0, *map(len, atoms)])
    arcs = []
    for a, start in zip(atoms, ptr):
        if a.pbc.any():
            raise ValueError("This benchmark uses nonperiodic 3BPA molecules.")
        d2 = np.square(a.positions[:, None] - a.positions[None, :]).sum(-1)
        arcs.append(np.stack(np.where((d2 > 0) & (d2 < cutoff**2))) + start)

    def tensor(x, dtype):
        return torch.as_tensor(np.asarray(x), device="cuda", dtype=dtype)

    index = tensor(np.concatenate(arcs, 1), torch.long)
    return dict(
        positions=tensor(np.concatenate([a.positions for a in atoms]), torch.float64),
        atomic_numbers=tensor(np.concatenate([a.numbers for a in atoms]), torch.long),
        edge_index=index,
        edge_shifts=index.new_zeros(index.size(1), 3),
        lattice=tensor([a.cell.array for a in atoms], torch.float64),
        ptr=tensor(ptr, torch.long),
        batch=tensor(np.repeat(np.arange(len(atoms)), np.diff(ptr)), torch.long),
        fidelity_idx=index.new_zeros(len(atoms)),
    )


def benchmark(args, data, cfg, stats, embedding, width):
    cfg["edge_embedding"]["type"] = embedding
    cfg["radial_basis"]["hidden"][-1] = width
    torch.manual_seed(42)
    model = create_model(cfg, stats, ["energy"], [])
    dtype = getattr(torch, args.dtype)
    configure_model_precision(model, dtype, "cuda")
    rep = model.readout_fn.representation
    assert rep.reciprocal_radial and all(b.use_eqx for b in rep.interactions)
    with torch.no_grad():
        for name in rep.edge_embedding.table_names:
            getattr(rep.edge_embedding, name).weight.normal_(std=0.1)
    positions = data["positions"].detach().requires_grad_(True)
    index = data["edge_index"]
    pairs = undirected_edge_pairs(
        index, data["edge_shifts"], num_nodes=positions.size(0)
    )
    assert pairs.inverse.numel() == index.shape[1]
    for layer in args.layers:
        inter = rep.interactions[layer - 1]
        snapshots = []

        def capture(module, a, kw):
            snapshots.append(
                dict(inspect.signature(module.forward).bind(*a, **kw).arguments)
            )

        hook = inter.register_forward_pre_hook(capture, with_kwargs=True)
        with torch.no_grad():
            for paired in (False, True):
                rep.reciprocal_radial = paired
                model(dict(data))
        hook.remove()
        assert len(snapshots) == 2 and snapshots[0].get("edge_pairs") is None
        assert (
            snapshots[1]["edge_pairs"].representatives.shape
            == pairs.representatives.shape
        )
        x = snapshots[0]["node_feats"].detach().requires_grad_(True)
        p = [x, positions, *inter.parameters(), *rep.edge_embedding.parameters()]

        def forward(paired):
            kw = snapshots[paired].copy()
            active = pairs
            if paired and args.rebuild_pairs:
                active = undirected_edge_pairs(
                    index, data["edge_shifts"], num_nodes=len(positions)
                )
                kw["edge_pairs"] = active
            dest = (
                index if not paired else index.index_select(1, active.representatives)
            )
            vector = positions[index[1]] - positions[index[0]]
            length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
            radial_length = (
                length.index_select(0, active.representatives) if paired else length
            )
            basis, cutoff = rep.radial_basis(
                radial_length, kw["node_types"], dest, rep.atomic_numbers
            )
            basis = basis.to(dtype)
            if cutoff is not None:
                cutoff = cutoff.to(dtype)
            feats = rep.edge_embedding(x, kw["node_types"], basis, dest, cutoff)
            cutoff = active.expand(cutoff) if paired and cutoff is not None else cutoff
            kw.update(
                graph=kw["graph"]._replace(edge_vector=vector, edge_length=length),
                node_feats=x,
                edge_radial_basis=active.expand(basis) if paired else basis,
                edge_feats=feats,
                edge_cutoff=cutoff,
                message_cutoff=None,
                element=(rep.edge_embedding.layer_weights(layer - 1), kw["element"][1]),
            )
            if rep.use_o3_angular_basis:
                kw["edge_attrs"] = rep.o3_angular_basis(
                    vector / length.clamp_min(1e-12)
                ).to(dtype)
            return tuple(z for z in inter(**kw) if z is not None)

        with torch.no_grad():
            torch.manual_seed(771)
            probes = [torch.randn_like(z) / z.numel() ** 0.5 for z in forward(False)]
            force_probe = torch.randn_like(positions) / positions.numel() ** 0.5

        def step(paired, phase):
            with torch.set_grad_enabled(phase != "F"):
                out = forward(paired)
                if phase == "F":
                    return out
                score = sum((z * v).sum() for z, v in zip(out, probes))
                if phase == "F+dR":
                    return torch.autograd.grad(score, positions)
                if phase == "F+B+B":
                    force = torch.autograd.grad(score, positions, create_graph=True)[0]
                    score = score + (force * force_probe).sum()
                return torch.autograd.grad(score, p, allow_unused=True)

        for phase in args.phases:
            a, b = (step(flag, phase) for flag in (False, True))
            assert all((u is None) == (v is None) for u, v in zip(a, b))
            values = [
                (u.detach(), v.detach())
                for u, v in zip(a, b)
                if u is not None and u.numel()
            ]
            err = max((u - v).abs().max().item() for u, v in values)
            rel = (
                (
                    sum((u.double() - v.double()).square().sum() for u, v in values)
                    / sum(u.double().square().sum() for u, v in values).clamp_min(
                        1.0e-30
                    )
                )
                .sqrt()
                .item()
            )
            tol = 3.0e-9 if dtype == torch.float64 else (0.005 if args.tf32 else 0.0001)
            if not rel < tol:
                raise AssertionError((embedding, width, layer, phase, err, rel))
            del a, b, values
            t, m = [[], []], [[], []]
            for flag in (False, True):
                for _ in range(3):
                    step(flag, phase)
            for i in range(args.rounds):
                for flag in (False, True) if i % 2 == 0 else (True, False):
                    torch.cuda.synchronize()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    start = perf_counter()
                    for _ in range(args.steps):
                        step(flag, phase)
                    torch.cuda.synchronize()
                    t[flag].append((perf_counter() - start) * 1000 / args.steps)
                    m[flag].append((torch.cuda.max_memory_allocated() - base) / 2**20)
            direct, paired = map(median, t)
            yield [
                embedding,
                "full" if memory_first() else "last",
                width,
                layer,
                phase,
                len(x),
                index.size(1),
                pairs.representatives.numel(),
                direct,
                paired,
                direct / paired,
                *map(median, m),
                err,
                rel,
            ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--statistics", type=Path, required=True)
    ap.add_argument(
        "--data", type=Path, default=Path.home() / "dataset/3BPA/train_300K.xyz"
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--structures", type=int, default=5)
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 128, 1024])
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument(
        "--embeddings",
        nargs="+",
        choices=["element", "element2"],
        default=["element", "element2"],
    )
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument(
        "--rebuild-pairs",
        action="store_true",
        help="Charge topology construction to each paired call.",
    )
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument(
        "--phases",
        nargs="+",
        choices=["F", "F+dR", "F+B", "F+B+B"],
        default=["F", "F+B", "F+B+B"],
    )
    args = ap.parse_args()
    PrecisionPolicy(model=args.dtype, matmul="tf32" if args.tf32 else "ieee").apply()
    root = Path(__file__).resolve().parents[2]
    base = OmegaConf.load(root / "TACE/tace.yaml").model.config
    override = OmegaConf.load(root / "benchmark/3BPA/o3.yaml").model.config
    cfg = OmegaConf.to_container(OmegaConf.merge(base, override), resolve=True)
    stats = [yaml.safe_load(args.statistics.read_text())]
    data = molecular_batch(args.data, args.structures, cfg["cutoff"])
    topology = []
    for i in range(15):
        torch.cuda.synchronize()
        start = perf_counter()
        undirected_edge_pairs(
            data["edge_index"], data["edge_shifts"], num_nodes=len(data["positions"])
        )
        torch.cuda.synchronize()
        if i >= 3:
            topology.append(1000 * (perf_counter() - start))
    print(
        torch.cuda.get_device_name(),
        args.dtype,
        "tf32",
        args.tf32,
        "pair map (once per graph):",
        round(median(topology), 3),
        "ms",
        flush=True,
    )
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            "embedding mode width layer phase nodes edges pairs direct_ms paired_ms speedup direct_MiB paired_MiB max_abs relative_L2 map_ms rebuild_pairs".split()
        )
        for width in args.widths:
            for embedding in args.embeddings:
                for row in benchmark(args, data, cfg, stats, embedding, width):
                    print(
                        *[round(v, 6) if isinstance(v, float) else v for v in row],
                        flush=True,
                    )
                    writer.writerow([*row, median(topology), args.rebuild_pairs])
                    f.flush()
