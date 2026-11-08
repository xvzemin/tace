"""Coverage, distribution, bounded memory and checkpoint semantics of sampling."""

import itertools
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DistributedSampler
from torch_geometric.data import Data

from tace.dataset.augmentation import AugmentedDataset
from tace.dataset.dataloader import build_graph_dataloader
from tace.dataset.datamodule import GraphDataModule
from tace.dataset.filter import filter_graphs, filter_options
from tace.dataset.quantity import KeySpecification
from tace.dataset.sampler import EpochSampler
from tace.dataset.store import write_graphs
from tace.utils.callbacks import EpochSamplerCallback


@pytest.mark.parametrize("size", list(range(1, 70)) + [127, 128, 129, 1023, 4097])
@pytest.mark.parametrize("seed", [0, 17, -3])
def test_permutation(size, seed):
    sampler = EpochSampler(range(size), seed=seed)
    indices = list(sampler)
    assert sorted(indices) == list(range(size))
    assert all(type(index) is int for index in indices)
    assert isinstance(sampler, DistributedSampler)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("size, count", [(11, 4), (13, 8), (17, 16), (12, 5)])
def test_coverage_across_epoch_and_pass_boundaries(size, count, shuffle):
    sampler = EpochSampler(range(size), epoch_size=count, shuffle=shuffle, seed=11)
    stream = []
    for epoch in range(3 * size):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        assert len(indices) == count
        assert len(set(indices)) == count
        stream.extend(indices)
    for start in range(len(stream) - size + 1):
        assert set(stream[start : start + size]) == set(range(size))
    if not shuffle:
        assert stream == [index % size for index in range(len(stream))]


@pytest.mark.parametrize("world_size", [2, 3, 4])
@pytest.mark.parametrize("shuffle", [False, True])
def test_distributed_global_count_and_disjoint_ranks(world_size, shuffle):
    count = world_size * 7
    dataset = range(count + 5)
    samplers = [
        EpochSampler(
            dataset, count, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        for rank in range(world_size)
    ]
    reference = EpochSampler(dataset, count, shuffle=shuffle)
    for epoch in range(6):
        reference.set_epoch(epoch)
        expected = list(reference)
        actual = []
        for rank, sampler in enumerate(samplers):
            sampler.set_epoch(epoch)
            indices = list(sampler)
            assert len(sampler) == len(indices) == 7
            assert indices == expected[rank::world_size]
            actual.extend(indices)
        assert len(actual) == len(set(actual)) == count


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize(
    "size, count, world_size, batch_size, partial_total, full_total",
    [
        (23, 9, 2, 3, 10, 12),
        (23, 1, 4, 3, 4, 12),
        (5, 5, 4, 3, 8, 12),
        (1, 1, 4, 3, 4, 12),
        (7, 3, 2, 4, 4, 8),
        (23, 12, 2, 3, 12, 12),
    ],
)
def test_rounded_epochs_preserve_complete_global_stream(
    size, count, world_size, batch_size, partial_total, full_total, drop_last, shuffle
):
    total = full_total if drop_last else partial_total
    samplers = [
        EpochSampler(
            range(size),
            count,
            num_replicas=world_size,
            rank=rank,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=17,
        )
        for rank in range(world_size)
    ]
    order = list(EpochSampler(range(size), shuffle=shuffle, seed=17))
    stream = []
    for epoch in range(6):
        local_indices = []
        for sampler in samplers:
            sampler.set_epoch(epoch)
            assert sampler.epoch_size == count
            assert sampler.total_size == total
            assert len(sampler) == total // world_size
            indices = list(sampler)
            assert len(indices) == len(sampler)
            if drop_last:
                assert len(indices) % batch_size == 0
            local_indices.append(indices)
            sampler.advance(len(indices))
        global_indices = list(itertools.chain.from_iterable(zip(*local_indices)))
        assert global_indices == [
            order[i % size] for i in range(epoch * total, (epoch + 1) * total)
        ]
        stream.extend(global_indices)
    for start in range(len(stream) - size + 1):
        assert len(set(stream[start : start + size])) == size


@pytest.mark.parametrize("drop_last, total", [(False, 10), (True, 12)])
def test_epoch_jump_uses_effective_global_size(drop_last, total):
    sampler = EpochSampler(
        range(23),
        9,
        num_replicas=2,
        rank=1,
        batch_size=3,
        drop_last=drop_last,
        shuffle=False,
    )
    sampler.set_epoch(7)
    assert list(sampler) == [i % 23 for i in range(7 * total + 1, 8 * total, 2)]


@pytest.mark.parametrize(
    "size, fraction, expected",
    [(13, 1.0, 13), (13, 0.5, 6), (13, 0.001, 1), (10, 0.3, 3)],
)
def test_fraction(size, fraction, expected):
    sampler = EpochSampler(range(size), fraction)
    assert len(sampler) == sampler.epoch_size == expected


def test_full_fraction_preserves_integer_length():
    sampler = EpochSampler(range(2**53 + 1), 1.0)
    assert sampler.epoch_size == 2**53 + 1


@pytest.mark.parametrize(
    "count", [0, -1, 11, 0.0, -0.5, 1.1, float("nan"), float("inf")]
)
def test_invalid_epoch_size_value(count):
    with pytest.raises(ValueError, match="epoch_size"):
        EpochSampler(range(10), count)


@pytest.mark.parametrize("count", [True, False, None, "2", [2]])
def test_invalid_epoch_size_type(count):
    with pytest.raises(TypeError, match="epoch_size"):
        EpochSampler(range(10), count)


@pytest.mark.parametrize(
    "options, error",
    [
        ({"num_replicas": 0}, ValueError),
        ({"num_replicas": -1}, ValueError),
        ({"num_replicas": 1.0}, TypeError),
        ({"num_replicas": True}, TypeError),
        ({"rank": -1}, ValueError),
        ({"rank": 1}, ValueError),
        ({"rank": False}, TypeError),
        ({"seed": 0.5}, TypeError),
        ({"seed": True}, TypeError),
        ({"shuffle": 1}, TypeError),
        ({"batch_size": 0}, ValueError),
        ({"batch_size": -1}, ValueError),
        ({"batch_size": 1.0}, TypeError),
        ({"batch_size": True}, TypeError),
        ({"drop_last": 1}, TypeError),
    ],
)
def test_invalid_options(options, error):
    with pytest.raises(error):
        EpochSampler(range(12), **options)


def test_empty_dataset():
    with pytest.raises(ValueError, match="nonempty"):
        EpochSampler([])


def test_seed_and_default_dtype():
    expected = list(EpochSampler(range(103), 37, seed=12))
    assert expected == list(EpochSampler(range(103), 37, seed=12))
    assert expected != list(EpochSampler(range(103), 37, seed=13))
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert expected == list(EpochSampler(range(103), 37, seed=12))
    finally:
        torch.set_default_dtype(previous)
    with torch.device("meta"):
        assert expected == list(EpochSampler(range(103), 37, seed=12))


@pytest.mark.parametrize("drop_last", [False, True])
def test_large_dataset_bounded_memory(monkeypatch, drop_last):
    def forbidden(*args, **kwargs):
        pytest.fail("A full dataset permutation must never be materialized")

    monkeypatch.setattr(torch, "randperm", forbidden)
    monkeypatch.setattr(np.random, "permutation", forbidden)
    tracemalloc.start()
    try:
        sampler = EpochSampler(
            range(100_000_000),
            10_000_001,
            seed=91,
            num_replicas=3,
            rank=2,
            batch_size=7,
            drop_last=drop_last,
        )
        sampler.set_epoch(10**30)
        indices = list(itertools.islice(sampler, 23))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(set(indices)) == 23
    assert all(0 <= index < 100_000_000 for index in indices)
    assert peak < 2_000_000


def test_prefetch_does_not_commit_and_iterator_captures_progress():
    sampler = EpochSampler(range(19), 12, seed=6)
    full = list(sampler)
    iterator = iter(sampler)
    sampler.advance(3)
    assert list(iterator) == full
    assert sampler.consumed == 3
    assert list(sampler) == full[3:]
    sampler.set_epoch(0)
    assert sampler.consumed == 3
    assert len(sampler) == 12
    sampler.advance(9)
    assert list(sampler) == []
    assert len(sampler) == 12
    sampler.set_epoch(1)
    assert sampler.consumed == 0
    assert len(list(sampler)) == 12


def test_checkpoint_restores_every_rank_from_rank_zero():
    dataset = range(31)
    zero = EpochSampler(dataset, 24, num_replicas=3, rank=0, seed=28)
    zero.set_epoch(4)
    zero.advance(3)
    state = zero.state_dict()
    assert "rank" not in state
    for rank in range(3):
        sampler = EpochSampler(dataset, 24, num_replicas=3, rank=rank, seed=28)
        sampler.set_epoch(4)
        expected = list(sampler)[3:]
        list(sampler)  # Pretend the entire epoch was prefetched.
        sampler.load_state_dict(state)
        sampler.set_epoch(4)
        assert list(sampler) == expected
        assert sampler.consumed == 3
        assert len(sampler) == 8


@pytest.mark.parametrize(
    "name, value, error",
    [
        ("dataset_size", 21, ValueError),
        ("epoch_size", 12, ValueError),
        ("seed", 1, ValueError),
        ("shuffle", False, ValueError),
        ("num_replicas", 2, ValueError),
        ("batch_size", 2, ValueError),
        ("drop_last", True, ValueError),
        ("drop_last", 1, TypeError),
        ("epoch", -1, ValueError),
        ("epoch", True, TypeError),
        ("consumed", -1, ValueError),
        ("consumed", 11, ValueError),
        ("consumed", 0.5, TypeError),
        ("start", -1, ValueError),
    ],
)
def test_checkpoint_validation_is_atomic(name, value, error):
    sampler = EpochSampler(range(20), 10)
    sampler.advance(2)
    before = sampler.state_dict()
    state = dict(before, epoch=7)
    state[name] = value
    with pytest.raises(error):
        sampler.load_state_dict(state)
    assert sampler.state_dict() == before


@pytest.mark.parametrize("drop_last", [False, True])
def test_checkpoint_rejects_changed_request_with_same_effective_count(drop_last):
    options = dict(num_replicas=2, rank=0, batch_size=3, drop_last=drop_last)
    original = EpochSampler(range(23), 9, **options)
    changed = EpochSampler(range(23), 10, **options)
    assert original.total_size == changed.total_size
    with pytest.raises(ValueError, match="epoch_size"):
        changed.load_state_dict(original.state_dict())


@pytest.mark.parametrize("method", ["advance", "set_epoch"])
@pytest.mark.parametrize(
    "value, error", [(-1, ValueError), (0.5, TypeError), (True, TypeError)]
)
def test_invalid_progress(method, value, error):
    sampler = EpochSampler(range(10))
    with pytest.raises(error):
        getattr(sampler, method)(value)
    assert sampler.consumed == 0 and sampler.epoch == 0


def test_advance_cannot_overrun():
    sampler = EpochSampler(range(10), 6, num_replicas=2, rank=0)
    sampler.advance(0)
    sampler.advance(2)
    with pytest.raises(ValueError, match="exceeds"):
        sampler.advance(2)
    assert sampler.consumed == 2


def test_partial_epoch_continues_from_completed_samples():
    sampler = EpochSampler(range(23), 10, shuffle=False)
    sampler.advance(6)
    state = sampler.state_dict()
    sampler = EpochSampler(range(23), 10, shuffle=False)
    sampler.load_state_dict(state)
    sampler.set_epoch(1)
    assert list(sampler) == list(range(6, 16))
    sampler.advance(10)
    sampler.set_epoch(2)
    assert list(sampler) == list(range(16, 23)) + [0, 1, 2]


@pytest.mark.parametrize(
    "changes, match, error",
    [
        ({"in_order": False}, "in_order", ValueError),
        ({"batch_size": None}, "batch_size", TypeError),
        ({"sampler": "other"}, "another sampler", ValueError),
        ({"_target_": "torch.utils.data.DataLoader"}, "torch_geometric", ValueError),
    ],
)
def test_loader_rejects_unsupported_or_unordered_batches(changes, match, error):
    config = {
        "_target_": "torch_geometric.loader.DataLoader",
        "batch_size": 3,
        **changes,
    }
    with pytest.raises(error, match=match):
        build_graph_dataloader(config, range(23), epoch_size=10)


@pytest.mark.parametrize(
    "drop_last, total, batch_lengths", [(False, 4, [3, 1]), (True, 6, [3, 3])]
)
def test_loader_rounding_and_resume_at_batch_boundaries(
    drop_last, total, batch_lengths
):
    config = {
        "_target_": "torch_geometric.loader.DataLoader",
        "batch_size": 3,
        "shuffle": False,
        "drop_last": drop_last,
    }
    loader = build_graph_dataloader(config, range(5), epoch_size=4)
    iterator = iter(loader)
    first = next(iterator).tolist()
    assert first == [0, 1, 2]
    loader.mark_batch_complete()
    partial_state = loader.state_dict()
    remaining = []
    for batch in iterator:
        remaining.append(batch.tolist())
        loader.mark_batch_complete()
    assert [len(first)] + [len(batch) for batch in remaining] == batch_lengths
    assert first + list(itertools.chain.from_iterable(remaining)) == [
        i % 5 for i in range(total)
    ]
    assert loader.sampler.consumed == total

    resumed = build_graph_dataloader(config, range(5), epoch_size=4)
    resumed.load_state_dict(partial_state)
    assert len(resumed) == 2
    actual = []
    for batch in resumed:
        actual.append(batch.tolist())
        resumed.mark_batch_complete()
    assert actual == remaining
    assert resumed.state_dict() == loader.state_dict()

    completed = build_graph_dataloader(config, range(5), epoch_size=4)
    completed.load_state_dict(resumed.state_dict())
    assert list(completed) == []
    completed.sampler.set_epoch(1)
    assert list(
        itertools.chain.from_iterable(batch.tolist() for batch in completed)
    ) == [i % 5 for i in range(total, 2 * total)]


def test_loader_validates_resume_batch_shape():
    config = {"_target_": "torch_geometric.loader.DataLoader", "batch_size": 2}
    loader = build_graph_dataloader(config, range(23), epoch_size=10)
    state = loader.state_dict()
    for name, value in (("batch_size", 5), ("drop_last", True)):
        with pytest.raises(ValueError, match=name):
            loader.load_state_dict(dict(state, **{name: value}))
    loader.load_state_dict(state)


@pytest.mark.parametrize("shuffle", [False, True])
def test_filtered_lmdb_sampling_coverage_ranks_and_resume(tmp_path, shuffle):
    graphs = [
        Data(
            num_nodes=1 + i % 3,
            atomic_numbers=torch.ones(1 + i % 3, dtype=torch.int64),
            x=torch.tensor([[float(i)]], dtype=torch.float64),
            edge_index=torch.zeros(2, i, dtype=torch.int64),
        )
        for i in range(12)
    ]
    source = write_graphs(graphs, [tmp_path], "train", shard_size=3)
    selected = filter_graphs(source, filter_options({"enabled": True, "max_atoms": 2}))
    dataset = AugmentedDataset(selected, ["time_reversal"])
    expected = [i for i in range(12) if i % 3 < 2]
    assert len(dataset) == len(expected)
    reference = EpochSampler(dataset, 0.5, shuffle=shuffle, seed=41)
    ranks = [
        EpochSampler(dataset, 0.5, num_replicas=2, rank=rank, shuffle=shuffle, seed=41)
        for rank in range(2)
    ]
    stream = []
    for epoch in range(5):
        reference.set_epoch(epoch)
        for sampler in ranks:
            sampler.set_epoch(epoch)
        indices = list(itertools.chain.from_iterable(zip(*(list(s) for s in ranks))))
        assert indices == list(reference)
        stream.extend(int(graph.x.item()) for graph in dataset.__getitems__(indices))
        for sampler in ranks:
            sampler.advance(len(sampler))
        reference.advance(len(reference))
    config = {
        "_target_": "torch_geometric.loader.DataLoader",
        "batch_size": 2,
        "shuffle": shuffle,
        "num_workers": 1,
        "multiprocessing_context": "spawn",
    }
    loader = build_graph_dataloader(config, dataset, epoch_size=0.5, seed=41)
    iterator = iter(loader)
    first = next(iterator).x.flatten().tolist()
    loader.mark_batch_complete()
    state = loader.state_dict()
    rest = [batch.x.flatten().tolist() for batch in iterator]
    resumed = build_graph_dataloader(config, dataset, epoch_size=0.5, seed=41)
    resumed.load_state_dict(state)
    assert [batch.x.flatten().tolist() for batch in resumed] == rest
    assert len(set(first + list(itertools.chain.from_iterable(rest)))) == 4
    different = source.select([0, 1, 2, 3, 4, 6, 7, 9])
    assert len(different) == len(selected)
    changed = build_graph_dataloader(
        config, AugmentedDataset(different, []), epoch_size=0.5, seed=41
    )
    with pytest.raises(ValueError, match="dataset selection"):
        changed.load_state_dict(state)
    for view in (source, selected, different):
        view.close()


class SamplingModel(pl.LightningModule):
    """Small deterministic training loop for loader checkpoint tests."""

    def __init__(self, output_dir=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.output_dir = output_dir
        self.seen = []
        self.cfg = {"dataset": {"epoch_size": 10}}

    def configure_callbacks(self):
        from tace.lightning.lit_model import LightningWrapperModel

        return LightningWrapperModel.configure_callbacks(self)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.002)

    def training_step(self, batch, batch_idx):
        self.seen.extend(batch.x.flatten().long().tolist())
        return (self.weight * batch.x / 23 - 0.1).square().mean()

    def validation_step(self, batch, batch_idx):
        return (self.weight * batch.x / 23 - 0.1).square().mean()

    def on_train_end(self):
        if self.output_dir is not None:
            torch.save(
                {"seen": self.seen, "weight": self.weight.detach().cpu()},
                Path(self.output_dir) / f"rank-{self.global_rank}.pt",
            )


def sampling_data(
    workers=0, epoch_size=10, batch_size=3, validation=False, drop_last=False
):
    loader = {
        "_target_": "torch_geometric.loader.DataLoader",
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": workers,
        "drop_last": drop_last,
    }
    if workers:
        loader.update(prefetch_factor=3, persistent_workers=True)
    data = GraphDataModule(
        {
            "dataset": {
                "epoch_size": epoch_size,
                "train_dataloader": loader,
                "valid_dataloader": dict(loader, shuffle=False, drop_last=False),
                "no_valid_set": not validation,
            },
            "model": {"config": {}},
            "misc": {"global_seed": 11},
        },
        None,
        [],
        KeySpecification.from_defaults(),
        [],
    )
    data.train_dataset = [
        Data(x=torch.tensor([[i]], dtype=torch.float32)) for i in range(23)
    ]
    data._elements_ready = True
    if validation:
        data.val_dataset = data.train_dataset[:4]
    return data


def test_default_loader_and_checkpoint_origin():
    ordinary = sampling_data(epoch_size=None)
    assert len(ordinary.train_dataloader()) == 8
    assert ordinary.state_dict() == {}
    assert not isinstance(ordinary.train_dataloader().sampler, EpochSampler)
    from tace.lightning.lit_model import LightningWrapperModel

    assert (
        LightningWrapperModel.configure_callbacks(SimpleNamespace(cfg=ordinary.cfg))
        == []
    )

    data = sampling_data()
    data.train_dataloader().mark_batch_complete()
    data.train_dataloader()  # In-memory rebuilding is not checkpoint recovery.
    trainer = SimpleNamespace(state=SimpleNamespace(fn="fit"), datamodule=data)
    callback = EpochSamplerCallback()
    for checkpoint in ({}, {type(data).__qualname__: {}}):
        with pytest.raises(ValueError, match="no epoch sampler progress"):
            callback.on_load_checkpoint(trainer, None, checkpoint)
    checkpoint = {type(data).__qualname__: data.state_dict()}
    callback.on_load_checkpoint(trainer, None, checkpoint)
    trainer.state.fn = "test"
    callback.on_load_checkpoint(trainer, None, {})
    ordinary.trainer = trainer
    ordinary.load_state_dict(data.state_dict())
    assert ordinary.state_dict() == {}


def sampling_trainer(directory, **kwargs):
    options = dict(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=directory,
        callbacks=[
            ModelCheckpoint(
                dirpath=directory,
                filename="{step}",
                every_n_train_steps=1,
                save_top_k=-1,
                save_last=True,
                save_on_train_epoch_end=False,
            )
        ],
    )
    options.update(kwargs)
    return pl.Trainer(**options)


@pytest.mark.parametrize("workers, accumulation", [(0, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize("drop_last, total", [(False, 10), (True, 12)])
def test_lightning_resume_with_prefetch(
    tmp_path, workers, accumulation, drop_last, total
):
    reference = SamplingModel()
    trainer = sampling_trainer(tmp_path / "full", accumulate_grad_batches=accumulation)
    trainer.fit(reference, datamodule=sampling_data(workers, drop_last=drop_last))
    expected = list(reference.seen)
    for step in (1, 4 // accumulation):
        checkpoint = tmp_path / "full" / f"step={step}.ckpt"
        state = torch.load(checkpoint, weights_only=False)
        loader_state = state["loops"]["fit_loop"]["state_dict"]["combined_loader"][0]
        completed = min(step * accumulation * 3, total)
        assert loader_state["sampler"]["consumed"] == completed
        resumed = SamplingModel()
        sampling_trainer(
            tmp_path / f"resumed-{step}", accumulate_grad_batches=accumulation
        ).fit(
            resumed,
            datamodule=sampling_data(workers, drop_last=drop_last),
            ckpt_path=checkpoint,
        )
        assert resumed.seen == expected[completed:]
        torch.testing.assert_close(resumed.weight, reference.weight, rtol=0, atol=0)


@pytest.mark.parametrize("max_steps, reload", [(4, 0), (-1, 0), (4, 1)])
def test_resume_after_max_steps(tmp_path, max_steps, reload):
    stopped = SamplingModel()
    trainer = sampling_trainer(tmp_path / "stopped", max_steps=2)
    trainer.fit(stopped, datamodule=sampling_data())
    checkpoint = tmp_path / "stopped" / "after-fit.ckpt"
    trainer.save_checkpoint(checkpoint)
    resumed = SamplingModel()
    sampling_trainer(
        tmp_path / "resumed",
        max_steps=max_steps,
        reload_dataloaders_every_n_epochs=reload,
    ).fit(resumed, datamodule=sampling_data(), ckpt_path=checkpoint)
    sampler = EpochSampler(range(23), seed=11)
    order = list(sampler)
    actual = stopped.seen + resumed.seen
    expected = [order[i % len(order)] for i in range(len(actual))]
    assert actual == expected


def test_validation_checkpoint_and_loader_reload(tmp_path):
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / "full",
        filename="{epoch}",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
    )
    full = SamplingModel()
    sampling_trainer(
        tmp_path / "full", callbacks=[checkpoint], reload_dataloaders_every_n_epochs=1
    ).fit(full, datamodule=sampling_data(validation=True))
    resumed = SamplingModel()
    sampling_trainer(tmp_path / "resumed", reload_dataloaders_every_n_epochs=1).fit(
        resumed,
        datamodule=sampling_data(validation=True),
        ckpt_path=tmp_path / "full" / "epoch=0.ckpt",
    )
    assert resumed.seen == full.seen[10:]
    torch.testing.assert_close(resumed.weight, full.weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options", [{"overfit_batches": 1}, {"limit_train_batches": 2}]
)
def test_trainer_rejects_implicit_subsampling(tmp_path, options):
    with pytest.raises(ValueError, match="epoch_size"):
        sampling_trainer(tmp_path, **options).fit(
            SamplingModel(), datamodule=sampling_data()
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
@pytest.mark.parametrize("drop_last, total", [(False, 10), (True, 12)])
def test_distributed_lightning_resume(tmp_path, drop_last, total):
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    sampling_trainer(
        full_dir, accelerator="gpu", devices=2, strategy="ddp_spawn", max_epochs=3
    ).fit(
        SamplingModel(output_dir=full_dir),
        datamodule=sampling_data(
            workers=1, epoch_size=9, batch_size=3, drop_last=drop_last
        ),
    )
    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    sampling_trainer(
        resumed_dir, accelerator="gpu", devices=2, strategy="ddp_spawn", max_epochs=3
    ).fit(
        SamplingModel(output_dir=resumed_dir),
        datamodule=sampling_data(
            workers=1, epoch_size=9, batch_size=3, drop_last=drop_last
        ),
        ckpt_path=full_dir / "step=1.ckpt",
    )
    rank_streams = []
    for rank in range(2):
        full = torch.load(full_dir / f"rank-{rank}.pt", weights_only=True)
        resumed = torch.load(resumed_dir / f"rank-{rank}.pt", weights_only=True)
        assert resumed["seen"] == full["seen"][3:]
        torch.testing.assert_close(resumed["weight"], full["weight"], rtol=0, atol=0)
        sampler = EpochSampler(
            range(23),
            9,
            num_replicas=2,
            rank=rank,
            seed=11,
            batch_size=3,
            drop_last=drop_last,
        )
        expected = []
        for epoch in range(3):
            sampler.set_epoch(epoch)
            expected.extend(sampler)
        assert full["seen"] == expected
        assert len(expected) == 3 * total // 2
        rank_streams.append(full["seen"])
    global_stream = list(itertools.chain.from_iterable(zip(*rank_streams)))
    order = list(EpochSampler(range(23), seed=11))
    assert global_stream == [order[i % 23] for i in range(3 * total)]
