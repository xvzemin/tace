"""Versioned graph and statistics cache metadata."""

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

from .read import HOW_TO_READ, SourceSpec, discover_sources

GRAPH_SCHEMA = 3
RECORD_SCHEMA = 1
STATISTICS_SCHEMA = 2
ARRAY_CODEC = "typed-array-v1"


def _digest(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:20]


def validate_stage(stage):
    if not isinstance(stage, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", stage):
        raise ValueError(f"Invalid dataset stage: {stage!r}")
    return stage


def _stage_source(dataset, stage):
    if stage == "train":
        return dataset.get("train_file")
    if stage == "valid":
        return dataset.get("valid_file") or dataset.get("train_file")
    if re.fullmatch(r"test\d+", stage):
        sources = dataset.get("test_files")
        sources = sources if isinstance(sources, (list, tuple)) else [sources]
        index = int(stage[4:])
        if index >= len(sources):
            raise ValueError(f"No source configured for {stage}")
        return sources[index]
    raise ValueError(f"Unknown dataset stage: {stage!r}")


def source_cache_key(cfg, stage="train", keyspec=None):
    """Raw records are independent of splits, cutoffs, and requested labels."""
    dataset = cfg["dataset"]
    backend = dataset.get("type", "ase")
    fidelity_key = (
        keyspec.info_keys.get("fidelity_idx", "fidelity_idx")
        if keyspec is not None
        else dataset.get("keys", {}).get("fidelity_idx_key", "fidelity_idx")
    )
    return _digest(
        {
            "schema": RECORD_SCHEMA,
            "codec": ARRAY_CODEC,
            "reader": dataset.get("reader") or backend,
            "format": dataset.get("format"),
            "options": dataset.get("options") or {},
            "source": source_signature(
                _stage_source(dataset, stage),
                backend,
                reader=dataset.get("reader"),
                format=dataset.get("format"),
                options=dataset.get("options"),
            ),
            "fidelity_key": fidelity_key,
        }
    )


def _split_signature(dataset, stage):
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
                str(path): path.read_text() if path.exists() else None
                for path in (Path("train.index"), Path("valid.index"))
            },
        }
    return {
        "mode": "random",
        "stage": stage,
        "valid_ratio": dataset.get("valid_ratio"),
        "split_seed": dataset.get("split_seed", 1),
    }


def source_signature(source, backend: str, *, reader=None, format=None, options=None):
    """Fingerprint the same effective file set and options used by the reader."""
    if source is None:
        return None
    entries = source if isinstance(source, (list, tuple)) else [source]
    existing, missing = [], []
    for item in entries:
        entry = dict(item) if isinstance(item, Mapping) else {"path": item}
        if "path" not in entry or entry.keys() - {
            "path",
            "reader",
            "format",
            "options",
            "fidelity_idx",
        }:
            raise ValueError("Invalid dataset source configuration")
        path = Path(entry["path"]).expanduser().resolve()
        if path.exists():
            existing.append(entry)
            continue
        # Missing explicit files remain representable for cache-only startup.
        source_reader = entry.get("reader", reader or backend)
        if source_reader not in HOW_TO_READ:
            raise ValueError(f"Unknown dataset reader: {source_reader}")
        missing.append(
            SourceSpec(
                str(path),
                source_reader,
                entry.get("format", format),
                {**(options or {}), **(entry.get("options") or {})},
                entry.get("fidelity_idx"),
            )
        )
    specs = (
        discover_sources(
            existing, backend, reader=reader, format=format, options=options
        )
        if existing
        else []
    )
    found = {spec.path: spec for spec in specs}
    for spec in missing:
        previous = found.get(spec.path)
        if previous is not None:
            if (previous.reader, previous.format, previous.options) != (
                spec.reader,
                spec.format,
                spec.options,
            ):
                raise ValueError(f"{spec.path}: conflicting reader, format or options")
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
                "format": spec.format,
                "options": spec.options,
                "fidelity_idx": spec.fidelity_idx,
                "size": stat.st_size if stat else None,
                "mtime_ns": stat.st_mtime_ns if stat else None,
            }
        )
    return signature


def graph_cache_key(
    cfg, target_property, embedding_property, keyspec, stage="train"
) -> str:
    """Identify graph data independently of network precision and element ordering."""
    dataset = cfg["dataset"]
    model = cfg["model"]["config"]
    config = {
        "schema": GRAPH_SCHEMA,
        "codec": ARRAY_CODEC,
        "dtype": "float64",
        "cutoff": float(model.get("cutoff", 6.0)),
        "max_neighbors": model.get("max_neighbors"),
        "neighborlist_backend": dataset.get("neighborlist_backend", "matscipy"),
        "source": source_cache_key(cfg, stage, keyspec),
        "split": _split_signature(dataset, stage),
        "target_property": sorted(target_property),
        "embedding_property": sorted(embedding_property),
        "info_keys": keyspec.info_keys,
        "arrays_keys": keyspec.arrays_keys,
    }
    return _digest(config)


def statistics_cache_key(cfg, target_property, embedding_property, keyspec) -> str:
    """Training statistics depend on labels, masks, topology, and references."""
    from .quantity import get_need_property

    dataset = cfg["dataset"]
    model = cfg["model"]["config"]
    needed = set(
        get_need_property(list(target_property), list(embedding_property), True)
    )
    names = needed & {
        "energy",
        "forces",
        "direct_forces",
        "initial_noncollinear_magmoms",
    }
    config = {
        "schema": STATISTICS_SCHEMA,
        "data_schema": GRAPH_SCHEMA,
        "source": source_cache_key(cfg, "train", keyspec),
        "split": _split_signature(dataset, "train"),
        "cutoff": float(model.get("cutoff", 6.0)),
        "max_neighbors": model.get("max_neighbors"),
        "neighborlist_backend": dataset.get("neighborlist_backend", "matscipy"),
        "properties": sorted(names),
        "info_keys": {k: v for k, v in keyspec.info_keys.items() if k in names},
        "arrays_keys": {k: v for k, v in keyspec.arrays_keys.items() if k in names},
        "atomic_numbers": model.get("atomic_numbers"),
        "references": [item.get("atomic_energy") for item in model.get("fidelity", [])],
        "weights": {
            "policy": "positive-mask",
            "keys": sorted(f"{p}_weight" for p in names),
        },
    }
    return _digest(config)


def graph_cache_directories(
    cfg, target_property, embedding_property, keyspec, stage="train"
):
    """Return precision-independent versioned cache directories."""
    key = graph_cache_key(cfg, target_property, embedding_property, keyspec, stage)
    return [
        Path(directory) / f"graphs-v{GRAPH_SCHEMA}-{key}"
        for directory in cfg["dataset"].get("shard_dirs", ["graphCache"])
    ]


def read_cache_metadata(directories, stage: str):
    """Read completed shard metadata, or return None for an incomplete cache."""
    validate_stage(stage)
    directories = list(map(Path, directories))
    metadata = None
    for directory in directories:
        path = directory / f"{stage}_metadata.json"
        if not path.is_file():
            return None
        try:
            current = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(current, dict):
            return None
        kind = current.get("kind", "graphs")
        if not isinstance(kind, str):
            return None
        schema = {"graphs": GRAPH_SCHEMA, "records": RECORD_SCHEMA}.get(kind)
        if (
            schema is None
            or current.get("schema") != schema
            or current.get("dtype") != "float64"
            or current.get("codec") != ARRAY_CODEC
        ):
            return None
        elements = current.get("atomic_numbers")
        fields = current.get("fields", [])
        if (
            not isinstance(elements, list)
            or any(type(z) is not int or z < 0 for z in elements)
            or not isinstance(fields, list)
            or any(not isinstance(name, str) for name in fields)
            or not isinstance(current.get("shard_stats", {}), dict)
        ):
            return None
        if metadata is not None and current != metadata:
            return None
        metadata = current
    if metadata is None or not isinstance(metadata.get("shards"), dict):
        return None
    paths = [
        path
        for directory in directories
        for path in directory.glob(f"{stage}_shard*.lmdb")
    ]
    names = [path.name for path in paths]
    if len(set(names)) != len(names) or set(names) != set(metadata.get("shards", {})):
        return None
    for path in paths:
        length = metadata["shards"][path.name]
        if type(length) is not int or length < 0 or not (path / "data.mdb").is_file():
            return None
        shard_stats = metadata.get("shard_stats", {}).get(path.name, {})
        if not isinstance(shard_stats, dict):
            return None
        if shard_stats.get("counts") == "counts.bin":
            try:
                if (path / "counts.bin").stat().st_size != length * 16:
                    return None
            except OSError:
                return None
    return metadata


def write_cache_metadata(
    directories, stage: str, lengths: dict, atomic_numbers: set, **extra
):
    """Atomically mark all shards of a graph dataset as complete."""
    validate_stage(stage)
    kind = extra.pop("kind", "graphs")
    if kind not in {"graphs", "records"}:
        raise ValueError(f"Unknown store kind: {kind}")
    metadata = {
        **extra,
        "schema": GRAPH_SCHEMA if kind == "graphs" else RECORD_SCHEMA,
        "kind": kind,
        "codec": ARRAY_CODEC,
        "dtype": "float64",
        "atomic_numbers": sorted(atomic_numbers),
        "shards": lengths,
    }
    for directory in map(Path, directories):
        path = directory / f"{stage}_metadata.json"
        temporary = directory / f"{stage}_metadata.json.tmp-{os.getpid()}"
        with temporary.open("w") as stream:
            json.dump(metadata, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    return metadata
