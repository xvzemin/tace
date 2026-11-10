"""Graph-cache metadata, statistics keys, and lossless shard copying.

Examples
--------
>>> # Run in a shell; only the new destination is written.
>>> # python -m tace.dataset.cache --source graphCache1 graphCache2 \
>>> #     --destination /local/graphs --workers 8 --reserve-gib 32

Copying requires complete, immutable source shards. Names, record keys and
statistics fingerprints are preserved. Rerunning the command reuses verified
completed shards and restarts unfinished shards without loading Python tensors.
"""

import argparse
import hashlib
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import struct
import tempfile
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path

import lmdb
from filelock import FileLock, Timeout

from .read import discover_sources, source_specs, stage_readers

GRAPH_SCHEMA = 3
RECORD_SCHEMA = 1
STATISTICS_SCHEMA = 2
ARRAY_CODEC = "typed-array-v1"
COMPRESSED_ARRAY_CODEC = "typed-array-zlib-v1"
LMDB_MODES = ("lmdb", "clmdb")

LOGGER = logging.getLogger(__name__)
MANIFEST = "copy-manifest.json"
RECEIPT = "copy-receipt.json"
STAGING = ".copy-staging"
SHARD_LOCKS = ".copy-shard-locks"
STOP = ".copy-stop"
RECORD_HEADER = struct.Struct("<8sII")
COMPRESSED_HEADER = struct.Struct("<8sQ")
MAX_RECORD_BYTES = 2 << 30
MIB = 1 << 20


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


def json_digest(value):
    """Return a SHA256 for a deterministic JSON representation."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def sync_directory(path):
    """Flush directory entries on platforms supporting directory descriptors."""
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, value):
    """Publish a complete JSON file after flushing its contents."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_inventory(sources):
    """Inspect completed graph shards without reading their LMDB payloads.

    Parameters
    ----------
    sources : sequence of path-like
        Directories containing published ``*_shard*.lmdb`` graph shards.

    Returns
    -------
    list of dict
        Sorted source specifications, including immutable file identities and
        the complete original metadata. Duplicate shard names are rejected.
    """
    roots = [Path(path).expanduser().resolve(strict=True) for path in sources]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Source directories must be nonempty and distinct")
    result, names = [], set()
    for root in sorted(roots):
        if not root.is_dir():
            raise ValueError(f"Source is not a directory: {root}")
        paths = sorted(root.glob("*_shard*.lmdb"))
        if not paths:
            raise ValueError(f"No graph shards in {root}")
        for path in paths:
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"Invalid source shard directory: {path}")
            if path.name in names:
                raise ValueError(f"Duplicate source shard name: {path.name}")
            names.add(path.name)
            metadata = json.loads((path / "metadata.json").read_text())
            stage = path.name.split("_shard", 1)[0]
            metadata_stage = metadata.get("stage")
            length = metadata.get("length")
            if (
                re.fullmatch(r"[A-Za-z0-9_-]+", stage) is None
                or not isinstance(metadata_stage, str)
                or re.fullmatch(r"[A-Za-z0-9_-]+", metadata_stage) is None
                or not path.name.startswith(f"{stage}_shard")
                or metadata.get("kind") != "graphs"
                or metadata.get("schema") != GRAPH_SCHEMA
                or metadata.get("codec") not in {ARRAY_CODEC, COMPRESSED_ARRAY_CODEC}
                or metadata.get("dtype") != "float64"
                or type(length) is not int
                or length <= 0
                or metadata.get("counts") != "counts.bin"
            ):
                raise ValueError(f"Unsupported graph shard metadata: {path}")
            if any(
                (root / name).exists()
                for current_stage in {stage, metadata_stage}
                for name in (
                    f"{current_stage}.build.lock",
                    f"{current_stage}_failed.json",
                )
            ):
                raise ValueError(f"Source stage is not complete: {root}/{stage}")
            files = {
                name: file_identity(path / name)
                for name in ("data.mdb", "counts.bin", "metadata.json")
            }
            if files["counts.bin"]["size"] != length * 16:
                raise ValueError(f"Invalid count index size: {path}")
            spec = {
                "source": str(path),
                "name": path.name,
                "stage": stage,
                "metadata": metadata,
                "metadata_sha256": file_digest(path / "metadata.json"),
                "files": files,
                "target_codec": COMPRESSED_ARRAY_CODEC,
            }
            spec["source_fingerprint"] = json_digest(spec)
            result.append(spec)
    return sorted(result, key=lambda spec: spec["name"])


def check_source(spec):
    path = Path(spec["source"])
    for name, identity in spec["files"].items():
        if file_identity(path / name) != identity:
            raise ValueError(f"Source changed during copy: {path / name}")
    if file_digest(path / "metadata.json") != spec["metadata_sha256"]:
        raise ValueError(f"Source metadata changed: {path}")
    if any(
        (path.parent / name).exists()
        for stage in {spec["stage"], spec["metadata"]["stage"]}
        for name in (f"{stage}.build.lock", f"{stage}_failed.json")
    ):
        raise ValueError(
            f"Source stage became incomplete: {path.parent}/{spec['stage']}"
        )


def raw_record(value):
    """Validate the binary envelope and return the original typed-array bytes."""
    view = memoryview(value)
    if len(view) < RECORD_HEADER.size:
        raise ValueError("Truncated typed-array record")
    if view[:8] == b"TACEZLB\0":
        _, size = COMPRESSED_HEADER.unpack_from(view)
        if not RECORD_HEADER.size <= size <= MAX_RECORD_BYTES:
            raise ValueError("Invalid compressed typed-array record size")
        decoder = zlib.decompressobj()
        try:
            payload = decoder.decompress(view[COMPRESSED_HEADER.size :], size + 1)
        except zlib.error as error:
            raise ValueError("Invalid compressed typed-array record") from error
        if len(payload) != size or not decoder.eof or decoder.unused_data:
            raise ValueError("Truncated or oversized compressed typed-array record")
        view = memoryview(payload)
    if not RECORD_HEADER.size <= len(view) <= MAX_RECORD_BYTES:
        raise ValueError("Invalid typed-array record size")
    magic, version, header_size = RECORD_HEADER.unpack_from(view)
    if magic != b"TACEARR\0" or version != 1:
        raise ValueError("Only typed-array-v1 graph records can be copied")
    if header_size > len(view) - RECORD_HEADER.size:
        raise ValueError("Truncated typed-array record header")
    return view


def record_digest(digest, key, value):
    digest.update(struct.pack("<QQ", len(key), len(value)))
    digest.update(key)
    digest.update(value)


def require_space(destination, reserve_bytes, pending_bytes):
    free = shutil.disk_usage(destination).free
    if free < reserve_bytes + pending_bytes:
        raise OSError(
            f"Insufficient free space in {destination}: {free} bytes available; "
            f"reserve {reserve_bytes} plus in-flight budget {pending_bytes}"
        )


def check_counts(path, metadata):
    digest = hashlib.sha256()
    nodes = edges = count = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 65536), b""):
            if len(block) % 16:
                raise ValueError(f"Truncated count index: {path}")
            digest.update(block)
            for n, e in struct.iter_unpack("<qq", block):
                if n < 0 or e < 0:
                    raise ValueError(f"Negative count in {path}")
                nodes += n
                edges += e
                count += 1
    if (count, nodes, edges) != (
        metadata["length"],
        metadata["num_nodes"],
        metadata["num_edges"],
    ):
        raise ValueError(f"Count index does not match metadata: {path}")
    return digest.hexdigest()


def verify_shard(path, spec, receipt=None, *, full=True):
    """Check a copied shard, optionally verifying every decompressed payload."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Invalid destination shard: {path}")
    if receipt is None:
        receipt = json.loads((path / RECEIPT).read_text())
    if receipt.get("source_fingerprint") != spec["source_fingerprint"]:
        raise ValueError(f"Destination shard belongs to a different source: {path}")
    expected = dict(spec["metadata"], codec=spec["target_codec"])
    if json.loads((path / "metadata.json").read_text()) != expected:
        raise ValueError(f"Destination metadata mismatch: {path}")
    for name, identity in receipt["files"].items():
        if file_identity(path / name) != identity:
            raise ValueError(f"Destination file changed: {path / name}")
    if full:
        if check_counts(path / "counts.bin", expected) != receipt["counts_sha256"]:
            raise ValueError(f"Destination count index hash mismatch: {path}")
    digest, count = hashlib.sha256(), 0
    with lmdb.open(
        str(path), readonly=True, create=False, lock=False, readahead=False
    ) as env:
        with env.begin(buffers=True) as transaction:
            if transaction.stat()["entries"] != expected["length"]:
                raise ValueError(f"Destination record count mismatch: {path}")
            if full:
                for key, value in transaction.cursor():
                    if key != f"{count:09d}".encode("ascii"):
                        raise ValueError(f"Destination record key mismatch: {path}")
                    if value[:8] != b"TACEZLB\0":
                        raise ValueError(
                            f"Destination record is not compressed: {path}"
                        )
                    record_digest(digest, key, raw_record(value))
                    count += 1
    if full and digest.hexdigest() != receipt["raw_sha256"]:
        raise ValueError(f"Destination payload hash mismatch: {path}")
    return receipt


def copy_shard(
    spec,
    destination,
    *,
    transaction_bytes,
    reserve_bytes,
    workers,
    progress_interval=30.0,
):
    """Copy one shard under a stable lock, including orphan-worker recovery.

    The full-cache controller owns the manifest and stage barriers. Direct
    callers may use this helper for an isolated single-shard pilot only.
    """
    destination = Path(destination).resolve()
    source = Path(spec["source"]).resolve()
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("Source and destination paths must not overlap")
    destination.mkdir(parents=True, exist_ok=True)
    directory = destination / SHARD_LOCKS
    directory.mkdir(exist_ok=True)
    lock_path = directory / f"{spec['name']}.lock"
    if directory.is_symlink() or lock_path.is_symlink():
        raise ValueError("Shard lock paths must not be symlinks")
    lock = FileLock(str(lock_path))
    while True:
        if (destination / STOP).exists():
            raise InterruptedError("Copy stopped while waiting for a shard writer")
        try:
            lock.acquire(timeout=1)
            break
        except Timeout:
            continue
    try:
        check_source(spec)
        final = destination / spec["name"]
        if final.exists():
            return verify_shard(final, spec, full=False)
        return copy_shard_locked(
            spec,
            destination,
            transaction_bytes=transaction_bytes,
            reserve_bytes=reserve_bytes,
            workers=workers,
            progress_interval=progress_interval,
        )
    finally:
        lock.release()


def copy_shard_locked(
    spec,
    destination,
    *,
    transaction_bytes,
    reserve_bytes,
    workers,
    progress_interval=30.0,
):
    """Write and verify a shard while its stable destination lock is held."""
    destination = Path(destination)
    final = destination / spec["name"]
    if final.exists():
        raise FileExistsError(f"Refusing to replace a destination shard: {final}")
    check_source(spec)
    stage = destination / STAGING / spec["name"]
    if stage.exists():
        owner = stage / "source.json"
        if stage.is_symlink() or not owner.is_file():
            raise ValueError(f"Unrecognized staging directory: {stage}")
        if json.loads(owner.read_text()) != spec:
            raise ValueError(
                f"Staging directory belongs to a different source: {stage}"
            )
        shutil.rmtree(stage)
    # Publish an initialized staging directory atomically. A crash before the
    # rename leaves only a small, unused bootstrap directory, never a canonical
    # unfinished shard without its source ownership record.
    stage.parent.mkdir(parents=True, exist_ok=True)
    bootstrap = Path(tempfile.mkdtemp(prefix=".initialize-", dir=stage.parent))
    try:
        write_json(bootstrap / "source.json", spec)
        os.rename(bootstrap, stage)
        sync_directory(stage.parent)
    finally:
        if bootstrap.exists():
            shutil.rmtree(bootstrap)
    pending_budget = workers * transaction_bytes * 4
    require_space(destination, reserve_bytes, pending_budget)
    digest = hashlib.sha256()
    records = raw_bytes = compressed_bytes = buffered = 0
    buffer = []
    last_progress = 0.0
    last_report = time.monotonic()

    def progress(force=False):
        nonlocal last_progress, last_report
        now = time.monotonic()
        if force or now - last_progress >= 5:
            write_json(
                stage / "progress.json",
                {
                    "records": records,
                    "raw_bytes": raw_bytes,
                    "compressed_bytes": compressed_bytes,
                },
            )
            last_progress = now
        if workers == 1 and now - last_report >= progress_interval:
            LOGGER.info(
                "copy %s: %d records; %.3f GiB raw -> %.3f GiB compressed",
                spec["name"],
                records,
                raw_bytes / (1 << 30),
                compressed_bytes / (1 << 30),
            )
            last_report = now

    with lmdb.open(
        str(stage), map_size=64 * MIB, sync=True, metasync=True, max_readers=32
    ) as target:

        def flush():
            nonlocal buffered
            if not buffer:
                return
            if (destination / STOP).exists():
                raise InterruptedError("Copy stopped; unfinished shards can be resumed")
            require_space(
                destination,
                reserve_bytes,
                workers * max(transaction_bytes, buffered) * 4,
            )
            for attempt in range(32):
                try:
                    with target.begin(write=True) as transaction:
                        consumed, added = transaction.cursor().putmulti(
                            buffer, append=True
                        )
                        if consumed != len(buffer) or added != len(buffer):
                            raise RuntimeError(
                                "Failed to write every record in a transaction"
                            )
                    break
                except lmdb.MapFullError:
                    if attempt == 31:
                        raise
                    target.set_mapsize(target.info()["map_size"] * 2)
            buffer.clear()
            buffered = 0
            progress()

        with lmdb.open(
            spec["source"],
            readonly=True,
            create=False,
            lock=False,
            readahead=False,
        ) as source:
            with source.begin(buffers=True) as transaction:
                if transaction.stat()["entries"] != spec["metadata"]["length"]:
                    raise ValueError(f"Source record count mismatch: {spec['source']}")
                for key, value in transaction.cursor():
                    key = bytes(key)
                    if key != f"{records:09d}".encode("ascii"):
                        raise ValueError(f"Unsupported source record key: {key!r}")
                    raw = raw_record(value)
                    record_digest(digest, key, raw)
                    encoded = (
                        bytes(value)
                        if value[:8] == b"TACEZLB\0"
                        else COMPRESSED_HEADER.pack(b"TACEZLB\0", len(raw))
                        + zlib.compress(raw, 1)
                    )
                    if buffer and buffered + len(encoded) > transaction_bytes:
                        flush()
                    buffer.append((key, encoded))
                    buffered += len(encoded)
                    raw_bytes += len(raw)
                    compressed_bytes += len(encoded)
                    records += 1
                    if buffered >= transaction_bytes:
                        flush()
                flush()
        target.sync()
    source_path = Path(spec["source"])
    shutil.copyfile(source_path / "counts.bin", stage / "counts.bin")
    # Windows requires a writable descriptor for fsync.
    with (stage / "counts.bin").open("r+b") as stream:
        os.fsync(stream.fileno())
    metadata = dict(spec["metadata"], codec=spec["target_codec"])
    write_json(stage / "metadata.json", metadata)
    receipt = {
        "source_fingerprint": spec["source_fingerprint"],
        "records": records,
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "raw_sha256": digest.hexdigest(),
        "counts_sha256": check_counts(stage / "counts.bin", metadata),
        "files": {
            name: file_identity(stage / name)
            for name in ("data.mdb", "counts.bin", "metadata.json")
        },
    }
    if receipt["counts_sha256"] != file_digest(source_path / "counts.bin"):
        raise ValueError(f"Count index copy differs from source: {source_path}")
    verify_shard(stage, spec, receipt, full=True)
    check_source(spec)
    write_json(stage / RECEIPT, receipt)
    progress(force=True)
    if final.exists():
        raise FileExistsError(f"Refusing to replace a destination shard: {final}")
    os.rename(stage, final)
    sync_directory(destination)
    return receipt


def report_progress(destination, completed, active, total, started):
    values = list(completed.values())
    for spec in active:
        try:
            values.append(
                json.loads(
                    (destination / STAGING / spec["name"] / "progress.json").read_text()
                )
            )
        except (OSError, ValueError):
            pass
    LOGGER.info(
        "copy %d/%d shards; %d records; %.3f GiB raw -> %.3f GiB compressed; %.1f s",
        len(completed),
        total,
        sum(value["records"] for value in values),
        sum(value["raw_bytes"] for value in values) / (1 << 30),
        sum(value["compressed_bytes"] for value in values) / (1 << 30),
        time.monotonic() - started,
    )


def copy_graph_cache(
    sources,
    destination,
    *,
    workers=1,
    transaction_bytes=16 * MIB,
    reserve_bytes=32 << 30,
    progress_interval=30.0,
    verify_existing=False,
):
    """Make a streaming compressed copy without writing to the source cache.

    Parameters
    ----------
    sources : sequence of path-like
        Complete, immutable source graph-cache directories.
    destination : path-like
        New/empty directory, or a previous copy with an identical source plan.
        All source shard names must be unique; they are preserved in this root.
    workers : int, default=1
        Number of independently spawned shard-copy processes.
    transaction_bytes : int, default=16777216
        Maximum compressed payload bytes buffered per worker transaction,
        except that a single larger record is written in its own transaction.
    reserve_bytes : int, default=34359738368
        Free disk space to retain, in addition to a concurrent transaction
        budget. External writers can still exhaust the disk; failures remain
        resumable and never publish an unverified shard.
    progress_interval : float, default=30
        Seconds between progress messages.
    verify_existing : bool, default=False
        Re-read/decompress completed destination shards during resume. Without
        this, verify their file identities, metadata and LMDB entry counts.

    Returns
    -------
    dict
        Complete manifest with source plan and per-shard verification receipts.

    Notes
    -----
    Every newly copied shard receives a complete local read/decompression/hash
    check before publication. Statistics YAML files are not rewritten or moved;
    retain the existing files in the training working directory. Use this target
    as ``dataset.shard_dirs`` with ``storage_mode='clmdb'`` after completion.
    """
    if (
        type(workers) is not int
        or workers < 1
        or type(transaction_bytes) is not int
        or transaction_bytes < 1
        or type(reserve_bytes) is not int
        or reserve_bytes < 0
        or not math.isfinite(progress_interval)
        or progress_interval <= 0
    ):
        raise ValueError("Invalid worker, transaction, reserve or progress budget")
    roots = [Path(path).expanduser().resolve(strict=True) for path in sources]
    destination = Path(destination).expanduser().resolve()
    for source in roots:
        if (
            source == destination
            or source in destination.parents
            or destination in source.parents
        ):
            raise ValueError("Source and destination paths must not overlap")
    plan = {"format_version": 1, "shards": source_inventory(roots)}
    plan_hash = json_digest(plan)
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise ValueError("Destination must be a directory")
    if any(path.is_symlink() for path in destination.iterdir()):
        raise ValueError("Destination entries must not be symlinks")
    started = time.monotonic()
    stages = sorted({spec["stage"] for spec in plan["shards"]})
    writer_names = {f".{stage}.writer.lock" for stage in stages}
    with (
        FileLock(str(destination / ".cache-copy.lock"), timeout=0),
        ExitStack() as locks,
    ):
        # Share the normal cache writer locks, on the destination only. A
        # concurrent training builder must wait rather than build into a copy.
        for name in sorted(writer_names):
            locks.enter_context(FileLock(str(destination / name), timeout=0))
        manifest_path = destination / MANIFEST
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("plan") != plan or manifest.get("plan_sha256") != plan_hash:
                raise ValueError(
                    "Destination manifest belongs to a different source plan"
                )
        else:
            if set(path.name for path in destination.iterdir()) != {
                ".cache-copy.lock",
                *writer_names,
            }:
                raise FileExistsError(
                    "Refusing a nonempty destination without a copy manifest"
                )
            manifest = {
                "plan": plan,
                "plan_sha256": plan_hash,
                "status": "building",
                "completed": {},
            }
            write_json(manifest_path, manifest)
        allowed = {
            MANIFEST,
            ".cache-copy.lock",
            STAGING,
            SHARD_LOCKS,
            STOP,
            *writer_names,
            *(spec["name"] for spec in plan["shards"]),
            *(f"{stage}.build.lock" for stage in stages),
        }
        unknown = {path.name for path in destination.iterdir()} - allowed
        if unknown:
            raise ValueError(f"Unrecognized destination contents: {sorted(unknown)}")
        completed, pending = {}, []
        for spec in plan["shards"]:
            path = destination / spec["name"]
            if path.exists():
                receipt = verify_shard(path, spec, full=verify_existing)
                recorded = manifest["completed"].get(spec["name"])
                if recorded is not None and recorded != receipt:
                    raise ValueError(f"Completed shard receipt changed: {path}")
                completed[spec["name"]] = receipt
            elif spec["name"] in manifest["completed"]:
                raise ValueError(
                    f"Previously completed destination shard is missing: {path}"
                )
            else:
                pending.append(spec)
        # The owner is a stable on-disk identifier for existing interrupted copies.
        marker = {"owner": "tace.dataset.cache_copy", "plan_sha256": plan_hash}
        for stage in stages:
            path = destination / f"{stage}.build.lock"
            if path.exists() and json.loads(path.read_text()) != marker:
                raise ValueError(f"Unrecognized destination stage marker: {path}")
            if pending:
                write_json(path, marker)
        if pending:
            (destination / STAGING).mkdir(exist_ok=True)
            if (destination / STAGING).is_symlink():
                raise ValueError("Staging directory must not be a symlink")
        (destination / STOP).unlink(missing_ok=True)
        manifest.update(
            status="building" if pending else "complete", completed=completed
        )
        write_json(manifest_path, manifest)
        options = {
            "transaction_bytes": transaction_bytes,
            "reserve_bytes": reserve_bytes,
            "workers": workers,
            "progress_interval": progress_interval,
        }
        active, executor = {}, None
        last_progress = time.monotonic()
        try:
            if workers == 1:
                for spec in pending:
                    completed[spec["name"]] = copy_shard(spec, destination, **options)
                    write_json(manifest_path, manifest)
                    if time.monotonic() - last_progress >= progress_interval:
                        report_progress(
                            destination, completed, [], len(plan["shards"]), started
                        )
                        last_progress = time.monotonic()
            elif pending:
                executor = ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                )
                remaining = iter(pending)
                for _ in range(min(workers, len(pending))):
                    spec = next(remaining)
                    active[
                        executor.submit(copy_shard, spec, destination, **options)
                    ] = spec
                while active:
                    finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    for future in finished:
                        spec = active.pop(future)
                        completed[spec["name"]] = future.result()
                        write_json(manifest_path, manifest)
                        spec = next(remaining, None)
                        if spec is not None:
                            active[
                                executor.submit(
                                    copy_shard, spec, destination, **options
                                )
                            ] = spec
                    if time.monotonic() - last_progress >= progress_interval:
                        report_progress(
                            destination,
                            completed,
                            active.values(),
                            len(plan["shards"]),
                            started,
                        )
                        last_progress = time.monotonic()
            for spec in plan["shards"]:
                check_source(spec)
            manifest.update(
                status="complete", elapsed_seconds=time.monotonic() - started
            )
            write_json(manifest_path, manifest)
            for stage in stages:
                (destination / f"{stage}.build.lock").unlink(missing_ok=True)
            sync_directory(destination)
            report_progress(destination, completed, [], len(plan["shards"]), started)
            return manifest
        except BaseException as error:
            (destination / STOP).touch()
            for future in active:
                future.cancel()
            manifest.update(
                status="interrupted", error=f"{type(error).__name__}: {error}"
            )
            write_json(manifest_path, manifest)
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--transaction-mib", type=int, default=16)
    parser.add_argument("--reserve-gib", type=float, default=32)
    parser.add_argument("--progress-seconds", type=float, default=30)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < 0:
        parser.error("--reserve-gib must be finite and nonnegative")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    result = copy_graph_cache(
        args.source,
        args.destination,
        workers=args.workers,
        transaction_bytes=args.transaction_mib * MIB,
        reserve_bytes=int(args.reserve_gib * (1 << 30)),
        progress_interval=args.progress_seconds,
        verify_existing=args.verify_existing,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "shards": len(result["completed"]),
                "destination": str(args.destination.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
