"""Versioned graph and statistics cache metadata."""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from .read import RGLOB

GRAPH_SCHEMA = 2


def source_signature(source, backend: str):
    """Return source paths, fidelities, sizes, and modification times."""
    if source is None:
        return None
    if isinstance(source, (list, tuple)):
        return [source_signature(item, backend) for item in source]
    fidelity = source.get("fidelity_idx") if isinstance(source, Mapping) else None
    path = Path(source["path"] if isinstance(source, Mapping) else source).resolve()
    files = (
        sorted({file for pattern in RGLOB[backend] for file in path.rglob(pattern)})
        if path.is_dir()
        else [path]
    )
    signature = []
    for file in files:
        stat = file.stat() if file.exists() else None
        signature.append(
            [
                str(file),
                stat.st_size if stat else None,
                stat.st_mtime_ns if stat else None,
            ]
        )
    return {"path": str(path), "fidelity_idx": fidelity, "files": signature}


def graph_cache_key(cfg, target_property, embedding_property, keyspec) -> str:
    """Identify graph data independently of network precision and element ordering."""
    dataset = cfg["dataset"]
    model = cfg["model"]["config"]
    backend = dataset.get("type", "ase")
    config = {
        "schema": GRAPH_SCHEMA,
        "dtype": "float64",
        "cutoff": float(model.get("cutoff", 6.0)),
        "max_neighbors": model.get("max_neighbors"),
        "neighborlist_backend": dataset.get("neighborlist_backend", "matscipy"),
        "reader": backend,
        "sources": {
            name: source_signature(dataset.get(name), backend)
            for name in ("train_file", "valid_file", "test_files")
        },
        "split": {
            name: dataset.get(name, default)
            for name, default in (
                ("no_valid_set", False),
                ("valid_ratio", None),
                ("valid_from_index", False),
                ("split_seed", 1),
            )
        },
        "target_property": sorted(target_property),
        "embedding_property": sorted(embedding_property),
        "info_keys": keyspec.info_keys,
        "arrays_keys": keyspec.arrays_keys,
    }
    if dataset.get("valid_from_index", False):
        config["split_indices"] = {
            str(path): path.read_text() if path.exists() else None
            for path in (Path("train.index"), Path("valid.index"))
        }
    encoded = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def statistics_cache_key(cfg, target_property, embedding_property, keyspec) -> str:
    """Identify statistics by graph data, element selection, and energy references."""
    model = cfg["model"]["config"]
    config = {
        "graph": graph_cache_key(cfg, target_property, embedding_property, keyspec),
        "atomic_numbers": model.get("atomic_numbers"),
        "fidelity": model.get("fidelity"),
    }
    encoded = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def graph_cache_directories(cfg, target_property, embedding_property, keyspec):
    """Return precision-independent versioned cache directories."""
    key = graph_cache_key(cfg, target_property, embedding_property, keyspec)
    return [
        Path(directory) / f"graphs-v{GRAPH_SCHEMA}-{key}"
        for directory in cfg["dataset"].get("shard_dirs", ["graphCache"])
    ]


def read_cache_metadata(directories, stage: str):
    """Read completed shard metadata, or return None for an incomplete cache."""
    metadata = None
    for directory in directories:
        path = directory / f"{stage}_metadata.json"
        if not path.is_file():
            return None
        try:
            current = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if current.get("schema") != GRAPH_SCHEMA or current.get("dtype") != "float64":
            return None
        if metadata is not None and current != metadata:
            return None
        metadata = current
    if metadata is None:
        return None
    paths = [
        path
        for directory in directories
        for path in directory.glob(f"{stage}_shard*.lmdb")
    ]
    names = [path.name for path in paths]
    if len(set(names)) != len(names) or set(names) != set(metadata.get("shards", {})):
        return None
    return metadata


def write_cache_metadata(directories, stage: str, lengths: dict, atomic_numbers: set):
    """Atomically mark all shards of a graph dataset as complete."""
    metadata = {
        "schema": GRAPH_SCHEMA,
        "dtype": "float64",
        "atomic_numbers": sorted(atomic_numbers),
        "shards": lengths,
    }
    for directory in directories:
        path = directory / f"{stage}_metadata.json"
        temporary = directory / f"{stage}_metadata.json.tmp-{os.getpid()}"
        temporary.write_text(json.dumps(metadata, sort_keys=True))
        os.replace(temporary, path)
