"""Static or shared dynamic E/F/S packages, unequal shards, one replicated optimizer."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from equivariantx_foundation.targets import aoti
from tace.models.adapter import TensorModel
from tace.utils.loss.normal import NormalLoss


@pytest.mark.parametrize("num_graphs", [1, 2])
def test_single_graph_node_sum(num_graphs):
    from torch.fx.experimental.proxy_tensor import make_fx

    batch = torch.arange(3) % num_graphs
    x = torch.randn(3, 2, dtype=torch.float64)
    dest = x.new_ones((num_graphs, 2))

    def aggregate(ids, values):
        return dest.index_add(0, ids, values + 0.21, alpha=2)

    graph = make_fx(aggregate)(batch, x)
    aoti._sum_single_graph(graph.graph, next(iter(graph.graph.nodes)))
    graph.recompile()
    assert any(
        n.target == torch.ops.aten.index_add.default for n in graph.graph.nodes
    ) == (num_graphs > 1)
    torch.testing.assert_close(graph(batch, x), aggregate(batch, x))
    if num_graphs == 1:
        with pytest.raises(RuntimeError, match="Expected one graph"):
            graph(batch + 1, x)


class PairEnergy(torch.nn.Module):
    """Small conservative potential with tied, unused, and frozen coefficients."""

    embedding_property = []
    target_property = ["energy", "forces", "stress", "noncollinear_magnetic_forces"]
    fidelity = ("standard", "excited")

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
        self.alias = self.w
        self.bias = torch.nn.Parameter(torch.tensor(0.21, dtype=torch.float64))
        self.unused = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.fixed = torch.nn.Parameter(
            torch.ones(1, dtype=torch.float64), requires_grad=False
        )
        self.register_buffer("atomic_numbers", torch.tensor([1, 8]))

    def forward(self, data, graph):
        r2 = graph.edge_vector.square().sum(-1)
        interaction = self.w * r2 + self.alias * r2.square() * 0.1
        interaction = interaction * (1 + graph.node_fidelity[data["edge_index"][1]])
        node = (
            r2.new_zeros(graph.positions.shape[0]).index_add(
                0, data["edge_index"][1], interaction
            )
            + self.bias * self.fixed
        )
        node = node + self.w * data["initial_noncollinear_magmoms"].square().sum(-1)
        energy = node.new_zeros(graph.num_graphs).index_add(0, data["batch"], node)
        return {"energy": energy, "node_energy": node}


def objective():
    return NormalLoss(
        PairEnergy.target_property,
        [
            "mse_energy_per_atom",
            "mse_forces",
            "mse_stress",
            "mse_noncollinear_magnetic_forces",
        ],
        [1.0, 0.7, 4.0, 2.5],
    )


def batch(sizes, offset, device):
    indices = torch.arange(sum(sizes), device=device)
    xyz = torch.stack(
        (
            (indices + offset + 2).double().sin(),
            (indices + 2 * offset).double().cos(),
            indices * 0.3,
        ),
        1,
    )
    groups = indices.new_tensor(sizes)
    ids = torch.repeat_interleave(torch.arange(len(sizes), device=device), groups)
    edge = (
        ((ids[:, None] == ids) & (indices[:, None] != indices)).nonzero().T.contiguous()
    )
    props = torch.arange(
        offset, offset + len(sizes), device=device, dtype=torch.float64
    )
    cell = torch.eye(3, device=device, dtype=xyz.dtype).repeat(len(sizes), 1, 1) * 4
    mask = (props > 0).double()
    return dict(
        positions=xyz,
        lattice=cell,
        pbc=cell[:, 0].new_ones((len(sizes), 3), dtype=torch.bool),
        node_type=indices % 2,
        edge_index=edge,
        edge_shifts=indices.new_zeros(edge.size(1), 3),
        ptr=torch.cat((indices.new_zeros(1), groups.cumsum(0))),
        batch=ids,
        entropy=1.0 + props / 10,
        energy_weight=1 + props,
        forces_weight=1.2 + props / 5,
        stress_weight=mask,
        initial_noncollinear_magmoms=torch.cos(xyz + offset),
        noncollinear_magnetic_forces=0.2 * xyz.sin(),
        noncollinear_magnetic_forces_weight=mask * (0.6 + props),
        energy=props - 0.5,
        forces=xyz.sin() * 0.3,
        stress=torch.where(mask[:, None, None] > 0, cell * 0.13, torch.nan),
    )


def combine(a, b):
    out = {
        k: torch.cat((a[k], b[k])) for k in a if k not in {"ptr", "edge_index", "batch"}
    }
    out["ptr"] = torch.cat((a["ptr"], a["ptr"][-1] + b["ptr"][1:]))
    out["edge_index"] = torch.cat(
        (a["edge_index"], a["ptr"][-1] + b["edge_index"]), dim=1
    )
    out["batch"] = torch.cat((a["batch"], len(a["ptr"]) - 1 + b["batch"]))
    return out


def worker(rank, directory, right_sizes, package_names):
    directory = Path(directory)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        model = TensorModel(PairEnergy()).to(device).train()
        obj = objective()
        # Create the user's optimizer before obtaining external input mappings.
        opt = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=0.1)
        step = aoti.load_training(
            directory / package_names[rank],
            device_index=rank,
        )
        shard = batch(([2], right_sizes)[rank], rank, device)
        oracle = torch.load(
            directory / "oracle.pt", map_location=device, weights_only=True
        )
        for i, expected in enumerate(oracle):
            if i == 1:
                # The export samples lacked IDs. Ranks may subsequently
                # supply absent or explicit heads without changing the ABI.
                if rank == 0:
                    model.reset_fidelity_idx(1)
                else:
                    shard["fidelity_idx"] = torch.arange(
                        len(right_sizes), dtype=torch.int32, device=device
                    )
            if i == 0:
                raw = aoti.training_inputs(model, obj, shard)
                _, naive_loss, naive_grads = step(*raw)
                naive_loss, naive_grads = aoti.mean_training_result(
                    naive_loss, naive_grads, raw[1]
                )
                assert not torch.isclose(naive_loss, expected["loss"])
                assert not torch.isclose(
                    naive_grads["model.readout_fn.w"],
                    expected["grads"]["model.readout_fn.w"],
                )
            data, params, buffers = aoti.distributed_training_inputs(model, obj, shard)
            predictions, local_loss, local_grad = step(data, params, buffers)
            assert model.readout_fn.w is model.readout_fn.alias
            assert set(local_grad) == {
                "model.readout_fn." + k for k in ("w", "bias", "unused")
            }
            sl = slice(0, 1) if rank == 0 else slice(1, None)
            atoms = slice(0, 2) if rank == 0 else slice(2, None)
            for k in PairEnergy.target_property:
                torch.testing.assert_close(
                    predictions[k], expected[k][atoms if k.endswith("forces") else sl]
                )
            loss, grads = aoti.mean_training_result(local_loss, local_grad, params)
            torch.testing.assert_close(loss, expected["loss"])
            opt.zero_grad(set_to_none=True)
            for name, g in grads.items():
                ref = expected["grads"][name]
                assert (g is None) == (ref is None)
                if ref is not None:
                    torch.testing.assert_close(g, ref)
                params[name].grad = g
            opt.step()
            for name, p in model.named_parameters():
                torch.testing.assert_close(p, expected["state"][name])
        # None on one rank contributes zero, without dropping the other
        # rank's derivative. A mismatched per-rank ABI fails on *both* ranks.
        partial = {
            **local_grad,
            "model.readout_fn.w": None
            if not rank
            else torch.ones_like(model.readout_fn.w),
        }
        _, mixed = aoti.mean_training_result(local_loss, partial, params)
        assert mixed["model.readout_fn.w"].item() == 0.5
        mismatched = dict(local_grad)
        if rank == 1:
            mismatched.pop("model.readout_fn.bias")
        with pytest.raises(ValueError, match="Inconsistent distributed"):
            aoti.mean_training_result(local_loss, mismatched, params)
        wrong_weights = (
            {**shard, "forces_weight": shard["forces_weight"].long()} if rank else shard
        )
        with pytest.raises(ValueError, match="label weights"):
            aoti.distributed_training_inputs(model, obj, wrong_weights)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="requires two CUDA devices and NCCL",
)
@pytest.mark.parametrize(
    "shared_dynamic", [False, True], ids=["static", "shared-dynamic"]
)
def test_unequal_two_rank_packaged_joint_training(tmp_path, shared_dynamic):
    # Compile outside the process group: only the explicitly requested
    # input preparation and reduction participate in collectives.
    device = torch.device("cuda", 0)
    model = TensorModel(PairEnergy()).to(device).train()
    obj = objective()
    right_sizes = [3] if shared_dynamic else [3, 2]
    a, b = batch([2], 0, device), batch(right_sizes, 1, device)
    if shared_dynamic:
        # Both ranks have G=1. Rank 0 uses N=2/E=2, while rank 1 uses
        # N=3/E=6; neither rank compiles or selects a shape-specific package.
        package_names = ("shared.pt2", "shared.pt2")
        aoti.export_training(
            model, obj, b, tmp_path / package_names[0], dynamic_shapes=True, max_edges=6
        )
        shared = aoti.load_training(tmp_path / package_names[0])
        with pytest.raises(ValueError, match="shared edges axis"):
            shared(*aoti.training_inputs(model, obj, batch([4], 0, device)))
    else:
        package_names = ("rank0.pt2", "rank1.pt2")
        for shard, name in zip((a, b), package_names, strict=True):
            aoti.export_training(model, obj, shard, tmp_path / name)
    full = combine(a, b)
    opt = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=0.1)
    oracle = []
    for i in range(2):
        if i == 1:
            full["fidelity_idx"] = torch.tensor(
                [1, *range(len(right_sizes))], device=device
            )
        out = model(full)
        loss = obj(out, full)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = {
            "model." + k: (p.grad.detach().cpu() if p.grad is not None else None)
            for k, p in model.named_parameters()
            if p.requires_grad
        }
        opt.step()
        oracle.append(
            dict(
                loss=loss.detach().cpu(),
                grads=grads,
                state={
                    k: p.detach().to("cpu", copy=True)
                    for k, p in model.named_parameters()
                },
                **{k: out[k].detach().cpu() for k in PairEnergy.target_property},
            )
        )
    torch.save(oracle, tmp_path / "oracle.pt")
    torch.multiprocessing.spawn(
        worker,
        args=(str(tmp_path), right_sizes, package_names),
        nprocs=2,
        join=True,
    )


class Prepared(torch.nn.Module):
    """Already-prepared tensors; large logical sizes can use zero-stride views."""

    def prepare_property_data(self, batch):
        if "fail" in batch:
            raise RuntimeError("local preparation failed")
        return dict(batch)


def _gloo_checks(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model = Prepared()
        objective = NormalLoss(["forces"], ["mse_forces"], [1.0])
        shard = batch([2], 0, "cpu")
        # 2**24+1 does not fit in FP32. Only tensor metadata is needed for
        # normalization; test the real integer arithmetic without a huge allocation.
        sizes = (2**24 + 1, 2**24 + 7)
        n = sizes[rank]
        shard["positions"] = torch.zeros(1, 3, dtype=torch.float32).expand(n, 3)
        shard["forces"] = shard["positions"]
        for key in ("node_type", "batch"):
            shard[key] = torch.zeros(1, dtype=torch.long).expand(n)
        shard["ptr"] = torch.tensor([0, n])
        data, _, _ = aoti.distributed_training_inputs(model, objective, shard)
        expected = float(shard["forces_weight"].item()) * 2 * n / sum(sizes)
        assert data["forces_weight"].item() == pytest.approx(expected, abs=2e-15, rel=0)
        fp32_weights = shard["forces_weight"].float()
        data, _, _ = aoti.distributed_training_inputs(
            model, objective, {**shard, "forces_weight": fp32_weights}
        )
        expected_fp32 = fp32_weights * fp32_weights.new_tensor(2 * n / sum(sizes))
        torch.testing.assert_close(data["forces_weight"], expected_fp32, rtol=0, atol=0)
        for key in (
            "ptr",
            "positions",
            "forces_weight",
            "entropy",
            "forces",
            "edge_index",
        ):
            bad = dict(shard)
            if rank == 1:
                del bad[key]
            with pytest.raises(ValueError, match="rank 1"):
                aoti.distributed_training_inputs(model, objective, bad)
        with pytest.raises(
            ValueError, match="rank 1: RuntimeError: local preparation failed"
        ):
            aoti.distributed_training_inputs(
                model, objective, {**shard, "fail": True} if rank == 1 else shard
            )
        prop = "noncollinear_magnetic_forces"
        spin_loss = NormalLoss([prop], ["mse_" + prop], [1.0])
        for key in (prop, "initial_noncollinear_magmoms"):
            spin_batch = batch([2], 0, "cpu")
            if rank == 1:
                spin_batch[key] = spin_batch[key][:, :1]
            with pytest.raises(ValueError, match="rank 1"):
                aoti.distributed_training_inputs(model, spin_loss, spin_batch)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_collective_preflight_and_int64_counts(tmp_path):
    torch.multiprocessing.spawn(
        _gloo_checks, args=((tmp_path / "gloo").as_uri(),), nprocs=2
    )


# Rank 0 has one two-atom structure; rank 1 has three and four atoms.
HIP_CASES = (
    ("structure", (1.7, 0.0, 0.4)),
    ("component", (1.7, 0.0, 0.4)),
    ("component", (0.0, 0.7, 2.1)),  # rank 0 has no Hessian label
    ("structure", (0.0, 0.0, 0.0)),
)


def _hip_shard(rank, weights, device="cpu"):
    sizes = (2,) if rank == 0 else (3, 4)
    data = batch(sizes, rank, device)
    blocks = []
    data["direct_hessian_weight"] = data["entropy"].new_tensor(
        weights[:1] if rank == 0 else weights[1:]
    )
    for p, w in zip(
        data["positions"].split(sizes), data["direct_hessian_weight"], strict=True
    ):
        if w > 0:
            u = (0.4 + p).cos()
            blocks.append(torch.einsum("ia,jb->ijab", u, u).reshape(-1, 3, 3))
        else:
            blocks.append(p.new_full((len(p) ** 2, 3, 3), float("nan")))
    data["direct_hessian"] = torch.cat(blocks)
    return data


def _hip_objective(reduction, mae=False):
    if mae:
        return NormalLoss(
            ["energy", "forces", "direct_hessian"],
            ["mae_energy", "mae_forces", "mae_direct_hessian"],
            [1.0, 10.0, 25.0],
            [{}, {}, {"reduction": reduction}],
        )
    return NormalLoss(
        ["direct_hessian"], ["mse_direct_hessian"], [2.3], [{"reduction": reduction}]
    )


def _hip_cases(mae):
    # The shipped HIP objective uses a component mean at every optimizer step.
    return HIP_CASES if not mae else [("component", w) for _, w in HIP_CASES[1:]]


def _hip_optimizer(model):
    from tace.lightning.optimizer import get_optimizer_param_groups

    return torch.optim.AdamW(get_optimizer_param_groups(model, 1e-8), lr=0.001)


def _hip_worker(rank, directory, mae):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.tests.test_tace_plan import physics_model

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    directory = Path(directory)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    records = torch.load(directory / "hip.pt", weights_only=True, map_location=device)
    host = physics_model("direct_hessian").to(device).train()
    host.reset_target_property(_hip_objective("component", mae).loss_property)
    host.load_state_dict(records["state"])
    opt = _hip_optimizer(host) if mae else None
    model = optimize(host, family="tace", backend="torch", task="training")
    packaged = aoti.load_training(directory / f"hip{rank}.pt2", device_index=rank)
    dist.init_process_group(
        "nccl",
        init_method=(directory / "gloo").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        for (reduction, weights), ref in zip(
            _hip_cases(mae), records["cases"], strict=True
        ):
            shard = _hip_shard(rank, weights, device)
            obj = _hip_objective(reduction, mae)
            data, state, buffers = aoti.distributed_training_inputs(model, obj, shard)
            valid = [n for n, w in zip((2, 3, 4), weights, strict=True) if w > 0]
            assert data["_valid_direct_hessian_count"].dtype == torch.int64
            assert data["_valid_direct_hessian_pair_count"].dtype == torch.int64
            assert data["_valid_direct_hessian_count"] == len(valid)
            assert data["_valid_direct_hessian_pair_count"] == sum(n * n for n in valid)
            pred = model(dict(shard))
            local_loss = obj(pred, data)
            params = {k: p for k, p in state.items() if p.requires_grad}
            first = torch.autograd.grad(
                local_loss, tuple(params.values()), allow_unused=True
            )
            loss, grads = aoti.mean_training_result(
                local_loss, dict(zip(params, first, strict=True)), params
            )
            torch.testing.assert_close(loss, ref["loss"], rtol=1e-8, atol=1e-10)
            torch.testing.assert_close(grads, ref["grads"], rtol=1e-7, atol=1e-9)
            if reduction == "component":
                out, local_loss, local_grad = packaged(data, state, buffers)
                torch.testing.assert_close(
                    out["direct_hessian"], pred["direct_hessian"]
                )
                loss, grads = aoti.mean_training_result(local_loss, local_grad, params)
                torch.testing.assert_close(loss, ref["loss"], rtol=1e-8, atol=1e-10)
                torch.testing.assert_close(grads, ref["grads"], rtol=1e-7, atol=1e-9)
                if mae:
                    for k, split in (
                        ("energy", 1),
                        ("forces", 2),
                        ("direct_hessian", 4),
                    ):
                        reference = ref["pred"][k]
                        torch.testing.assert_close(
                            out[k], reference[split:] if rank else reference[:split]
                        )
                    opt.zero_grad(set_to_none=True)
                    for name, g in grads.items():
                        params[name].grad = g
                    opt.step()
                    torch.testing.assert_close(
                        dict(host.named_parameters()),
                        ref["updated"],
                        rtol=1e-8,
                        atol=1e-10,
                    )
        # A rank with missing *keys*, instead of NaNs + zero weights, must
        # fail collectively before any count reduction or package invocation.
        for key in ("direct_hessian", "direct_hessian_weight"):
            bad = dict(shard)
            if rank:
                del bad[key]
            with pytest.raises(ValueError, match="rank 1"):
                aoti.distributed_training_inputs(model, obj, bad)
        bad = (
            {**shard, "direct_hessian": shard["direct_hessian"][:-1]} if rank else shard
        )
        with pytest.raises(
            ValueError, match=r"rank 1: ValueError: direct_hessian labels"
        ):
            aoti.distributed_training_inputs(model, obj, bad)
        with pytest.raises(ValueError, match="rank 0"):
            aoti.distributed_training_inputs(
                model, _hip_objective("unsupported", mae), shard
            )
        for k, fn in (
            ("energy", "huber_energy"),
            ("forces", "huber_forces"),
            ("direct_hessian", "l2mae_direct_hessian"),
        ):
            # A single unsupported term must fail on both peers before a valid
            # peer tries to reduce its Hessian counts.
            with pytest.raises(ValueError, match="rank 1: unsupported objective"):
                aoti.distributed_training_inputs(
                    model,
                    NormalLoss([k], [fn], [1.0]) if rank else obj,
                    shard,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="requires two CUDA devices and NCCL",
)
@pytest.mark.parametrize("mae", [False, True], ids=["hessian-mse", "efh-mae-adamw"])
def test_distributed_hip_against_global_native(tmp_path, monkeypatch, mae):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.tests.test_tace_plan import physics_model

    monkeypatch.setenv("TACE_USE_EQX", "0")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(271)
    host = physics_model("direct_hessian").train()
    host.reset_target_property(_hip_objective("component", mae).loss_property)
    initial = {n: t.clone() for n, t in host.state_dict().items()}
    opt = _hip_optimizer(host) if mae else None
    cases = []
    for reduction, weights in _hip_cases(mae):
        full = combine(_hip_shard(0, weights), _hip_shard(1, weights))
        pred = host(dict(full))
        assert pred["direct_hessian"].shape == (4 + 9 + 16, 3, 3)
        loss = _hip_objective(reduction, mae)(pred, full)
        params = {
            "model." + k: p for k, p in host.named_parameters() if p.requires_grad
        }
        grads = dict(
            zip(
                params,
                torch.autograd.grad(loss, tuple(params.values()), allow_unused=True),
                strict=True,
            )
        )
        if any(weights):
            assert any(
                "direct_hessian_readout.edge_readout" in k
                and g is not None
                and g.abs().max() > 1e-6
                for k, g in grads.items()
            )
        elif not mae:
            assert loss == 0
        cases.append({"loss": loss.detach(), "grads": grads})
        if mae:
            cases[-1]["pred"] = {
                k: pred[k].detach() for k in host.get_target_property()
            }
            opt.zero_grad(set_to_none=True)
            for name, g in grads.items():
                params[name].grad = g
            opt.step()
            cases[-1]["updated"] = {
                n: p.detach().clone() for n, p in host.named_parameters()
            }
    # Atom counts and errors differ, so the two averaging policies must
    # actually be distinguishable by this global reference.
    if not mae:
        assert not torch.isclose(cases[0]["loss"], cases[1]["loss"])
    host.load_state_dict(initial)
    torch.save({"state": initial, "cases": cases}, tmp_path / "hip.pt")
    # Fixed-shape joint programs take the aggregated counts as live scalar
    # inputs. Export outside the group, then run a different shape per rank.
    model = optimize(host.cuda(), family="tace", backend="torch", task="training")
    for rank in range(2):
        aoti.export_training(
            model,
            _hip_objective("component", mae),
            _hip_shard(rank, HIP_CASES[1][1], "cuda"),
            tmp_path / f"hip{rank}.pt2",
        )
    torch.multiprocessing.spawn(_hip_worker, args=(str(tmp_path), mae), nprocs=2)
