"""Graph shard metadata and statistics cache keys."""

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

from .read import discover_sources, source_specs, stage_readers

GRAPH_SCHEMA = 3
RECORD_SCHEMA = 1
STATISTICS_SCHEMA = 2
ARRAY_CODEC = "typed-array-v1"
COMPRESSED_ARRAY_CODEC = "typed-array-zlib-v1"
LMDB_MODES = ("lmdb", "clmdb")


def digest(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:20]


def validate_stage(stage):
    if not isinstance(stage, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", stage):
        raise ValueError(f"Invalid dataset stage: {stage!r}")
    return stage


def stage_source(dataset, stage):
    if stage == "train":
        return dataset.get("train_file")
    if stage == "valid":
        return dataset.get("valid_file") or dataset.get("train_file")
    if re.fullmatch(r"test\d+", stage):
        sources = dataset.get("test_files")
        sources = (
            sources
            if isinstance(sources, Sequence) and not isinstance(sources, str)
            else [sources]
        )
        index = int(stage[4:])
        if index >= len(sources):
            raise ValueError(f"No source configured for {stage}")
        return sources[index]
    raise ValueError(f"Unknown dataset stage: {stage!r}")


def source_cache_key(cfg, stage="train", keyspec=None):
    """Raw records are independent of splits, cutoffs, and requested labels."""
    dataset = cfg["dataset"]
    reader = stage_readers(dataset)[stage]
    fidelity_key = (
        keyspec.info_keys.get("fidelity_idx", "fidelity_idx")
        if keyspec is not None
        else dataset.get("keys", {}).get("fidelity_idx_key", "fidelity_idx")
    )
    return digest(
        {
            "schema": RECORD_SCHEMA,
            "codec": ARRAY_CODEC,
            "source": source_signature(
                stage_source(dataset, stage),
                reader,
            ),
            "fidelity_key": fidelity_key,
        }
    )


def split_signature(dataset, stage):
    if stage.startswith("test"):
        return None
    if dataset.get("no_valid_set", False):
        return {"mode": "all", "stage": stage}
    if dataset.get("valid_file") is not None:
        return {"mode": "external", "stage": stage}
    if dataset.get("valid_from_index", False):
        return {
            "mode": "indices",
            "stage": stage,
            "indices": {
                str(path): index_signature(path)
                for path in (Path("train.index"), Path("valid.index"))
            },
        }
    return {
        "mode": "random",
        "stage": stage,
        "valid_ratio": dataset.get("valid_ratio"),
        "split_seed": dataset.get("split_seed", 1),
    }


def index_signature(path):
    """Fingerprint ordered integer indices without retaining the index text."""
    digest = hashlib.sha256()
    count = 0
    try:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    digest.update(f"{int(line)}\n".encode())
                    count += 1
    except FileNotFoundError:
        return None
    return {"count": count, "sha256": digest.hexdigest()}


def source_signature(source, reader="ase"):
    """Fingerprint the effective files, readers, and source fidelities."""
    if source is None:
        return None
    existing, missing = [], []
    for spec in source_specs(source, reader):
        path = Path(spec.path)
        if path.exists():
            existing.append(spec)
            continue
        # Missing explicit files remain representable for cache-only startup.
        missing.append(spec)
    specs = (
        discover_sources(
            [
                {"path": spec.path, "fidelity_idx": spec.fidelity_idx}
                for spec in existing
            ],
            [spec.reader for spec in existing],
        )
        if existing
        else []
    )
    found = {spec.path: spec for spec in specs}
    for spec in missing:
        previous = found.get(spec.path)
        if previous is not None:
            if previous.reader != spec.reader:
                raise ValueError(f"{spec.path}: conflicting source readers")
            if previous.fidelity_idx is not None:
                if (
                    spec.fidelity_idx is not None
                    and previous.fidelity_idx != spec.fidelity_idx
                ):
                    raise ValueError(
                        f"{spec.path}: conflicting source fidelity_idx values"
                    )
                spec.fidelity_idx = previous.fidelity_idx
        found[spec.path] = spec
    signature = []
    for path, spec in sorted(found.items()):
        file = Path(path)
        stat = file.stat() if file.exists() else None
        signature.append(
            {
                "path": path,
                "reader": spec.reader,
                "fidelity_idx": spec.fidelity_idx,
                "size": stat.st_size if stat else None,
                "mtime_ns": stat.st_mtime_ns if stat else None,
            }
        )
    return signature


def statistics_cache_key(cfg, target_property, embedding_property, keyspec) -> str:
    """Return a key for the training labels, statistics options, and references."""
    from .quantity import get_need_property

    dataset = cfg["dataset"]
    model = cfg["model"]["config"]
    names = set(
        get_need_property(list(target_property), list(embedding_property), True)
    )
    names &= {"energy", "forces", "direct_forces", "initial_noncollinear_magmoms"}
    config = {
        "schema": STATISTICS_SCHEMA,
        "cutoff": float(model.get("cutoff", 6.0)),
        "max_neighbors": model.get("max_neighbors"),
        "properties": sorted(names),
        "atomic_numbers": model.get("atomic_numbers"),
        "references": [item.get("atomic_energy") for item in model.get("fidelity", [])],
    }
    metadata = (
        read_cache_metadata(graph_cache_directories(cfg), "train")
        if dataset.get("storage_mode") in LMDB_MODES
        else None
    )
    if metadata is not None:
        config["graphs"] = metadata["shards"]
        if not dataset.get("no_valid_set", False):
            valid = read_cache_metadata(graph_cache_directories(cfg), "valid")
            config["validation_elements"] = valid["atomic_numbers"] if valid else None
    else:
        config.update(
            source=source_cache_key(cfg, "train", keyspec),
            split=split_signature(dataset, "train"),
            info_keys={k: v for k, v in keyspec.info_keys.items() if k in names},
            arrays_keys={k: v for k, v in keyspec.arrays_keys.items() if k in names},
            neighborlist_backend=dataset.get("neighborlist_backend", "matscipy"),
        )
    return digest(config)


def graph_cache_directories(cfg):
    """Return graph directories without inspecting the original structures."""
    directories = cfg["dataset"].get("shard_dirs", ["graphCache"])
    if isinstance(directories, (str, Path)):
        directories = [directories]
    return [Path(directory).expanduser() for directory in directories]


def read_cache_metadata(directories, stage: str, *, allow_building=False):
    """Combine metadata from completed LMDB shards in the supplied directories."""
    validate_stage(stage)
    directories = list(map(Path, directories))
    if not allow_building and any(
        (directory / f"{stage}.build.lock").exists()
        or (directory / f"{stage}_failed.json").exists()
        for directory in directories
    ):
        return None
    paths = sorted(
        (
            path
            for directory in directories
            for path in directory.glob(f"{stage}_shard*.lmdb")
        ),
        key=lambda path: path.name,
    )
    if not paths:
        return None
    metadata = None
    elements, fields = set(), None
    shards, shard_stats = {}, {}
    for path in paths:
        if path.name in shards:
            raise ValueError(f"Duplicate graph shard: {path.name}")
        try:
            current = json.loads((path / "metadata.json").read_text())
            kind = current["kind"]
            expected = GRAPH_SCHEMA if kind == "graphs" else RECORD_SCHEMA
            if (
                kind not in {"graphs", "records"}
                or current["schema"] != expected
                or current["codec"] not in {ARRAY_CODEC, COMPRESSED_ARRAY_CODEC}
                or current["dtype"] != "float64"
                or current["length"] <= 0
                or not (path / "data.mdb").is_file()
                or (path / "counts.bin").stat().st_size != current["length"] * 16
            ):
                raise ValueError("Invalid shard format or incomplete data")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot read graph shard {path}: {exc}") from exc
        if metadata is None:
            metadata = {
                key: current[key] for key in ("schema", "kind", "codec", "dtype")
            }
            metadata["graph_config"] = current.get("graph_config")
        elif any(current[key] != metadata[key] for key in ("schema", "kind", "dtype")):
            raise ValueError(f"Incompatible shard format in {path}")
        elif current.get("graph_config") != metadata["graph_config"]:
            raise ValueError(f"Different cutoffs or neighbor limits in {path}")
        shards[path.name] = current["length"]
        shard_stats[path.name] = {
            key: current[key] for key in ("counts", "num_nodes", "num_edges")
        }
        elements.update(current["atomic_numbers"])
        fields = (
            set(current["fields"])
            if fields is None
            else fields.intersection(current["fields"])
        )
    metadata.update(
        shards=shards,
        shard_stats=shard_stats,
        atomic_numbers=sorted(elements),
        fields=sorted(fields),
        num_nodes=sum(item["num_nodes"] for item in shard_stats.values()),
        num_edges=sum(item["num_edges"] for item in shard_stats.values()),
    )
    return metadata
