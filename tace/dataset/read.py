"""Lazy, indexed dataset readers. Source failures are never treated as empty data."""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from .quantity import get_need_property
from .records import (
    record_from_atoms,
    record_from_row,
    resolve_property,
    validate_fidelity,
)
from .split import IndexedSource, random_split


class DatasetsSplit:
    def __init__(self, train, valid, test):
        self.train, self.valid, self.test = train, valid, test


class ThreeDataset:
    def __init__(self, train, valid, test=None):
        self.data = DatasetsSplit(train, valid, [] if test is None else test)

    def __getitem__(self, idx):
        return (self.data.train, self.data.valid, self.data.test)[idx]

    def __setitem__(self, idx, value):
        if idx not in (0, 1, 2):
            raise IndexError("Index out of range for ThreeDataset")
        setattr(self.data, ("train", "valid", "test")[idx], value)

    def __len__(self):
        return 3

    train = property(lambda self: self.data.train)
    valid = property(lambda self: self.data.valid)
    test = property(lambda self: self.data.test)


def check_keys(
    atomsList, target_property, keyspec, embedding_property=None, training=True
):
    """Fill missing requested aliases without overwriting an explicit label."""
    names = get_need_property(target_property, embedding_property, training)
    for atoms in atomsList:
        record = record_from_atoms(atoms)
        for name in names:
            value = resolve_property(record, name, keyspec)
            if value is None:
                continue
            if name in keyspec.info_keys:
                atoms.info.setdefault(keyspec.info_keys[name], value)
            elif name in keyspec.arrays_keys:
                atoms.arrays.setdefault(keyspec.arrays_keys[name], np.asarray(value))
    return atomsList


class SourceSequence(Sequence):
    """Random-access Atoms compatibility plus a calculator-free record fast path."""

    atomic_numbers = None  # Optional source metadata; never trigger a data scan.

    def close(self):
        """Release open readers while retaining indexes for lazy, read-only reuse."""

    def _index(self, index):
        if not isinstance(index, (int, np.integer)) or isinstance(
            index, (bool, np.bool_)
        ):
            raise TypeError("Dataset indices must be integers")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return index

    def get_record(self, index):
        return self.get_records([index])[0]

    def get_records(self, indices):
        indices = [self._index(index) for index in indices]
        try:
            return self._read_records(indices)
        except Exception as exc:
            message = f"Failed to read {getattr(self, 'path', type(self).__name__)} at structure indices {indices[:8]}: {exc}"
            if isinstance(exc, ValueError):
                raise ValueError(message) from exc
            raise RuntimeError(message) from exc

    def _atoms(self, record):
        return record.to_atoms()

    def __getitem__(self, index):
        if isinstance(index, slice):
            return IndexedSource(self, range(*index.indices(len(self))))
        return self._atoms(self.get_record(index))

    def getitems(self, indices):
        return [self._atoms(record) for record in self.get_records(indices)]

    __getitems__ = getitems

    def __iter__(self):
        for start in range(0, len(self), 128):
            yield from self.getitems(range(start, min(start + 128, len(self))))


class ASEFileSource(SourceSequence):
    """Read structures from an ASE-supported file.

    Parameters
    ----------
    path : path-like
        Source file. Its format is detected by ASE.

    Examples
    --------
    >>> source = ASEFileSource("structures.xyz")
    >>> atoms = source[0]
    """

    def __init__(self, path):
        from ase.io.formats import filetype

        self.path = str(Path(path).expanduser().resolve())
        self.format = filetype(self.path)
        self._offsets = None
        self._length = None
        self._trajectory = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_trajectory=None, _pid=None)
        return state

    def close(self):
        if self._trajectory is not None:
            self._trajectory.close()
            self._trajectory = None
        self._pid = None

    def _build_index(self):
        if self._length is not None:
            return
        try:
            if self.format in {"xyz", "extxyz"} and Path(self.path).suffix not in {
                ".gz",
                ".bz2",
                ".xz",
            }:
                offsets = []
                with open(self.path, "rb") as stream:
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line:
                            break
                        if not line.strip():
                            continue
                        count = int(line)
                        if count < 0:
                            raise ValueError("Negative atom count in XYZ frame")
                        for _ in range(count + 1):
                            if not stream.readline():
                                raise ValueError("Truncated XYZ frame")
                        # ASE extended XYZ can append up to three VEC cell lines.
                        while True:
                            end = stream.tell()
                            line = stream.readline()
                            if not line.startswith(b"VEC"):
                                stream.seek(end)
                                break
                        offsets.append((start, end - start))
                self._offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
                self._length = len(offsets)
            elif self.format == "traj":
                self._length = len(self._open_trajectory())
            else:
                from ase.io import iread

                self._length = sum(
                    1 for _ in iread(self.path, index=":", format=self.format)
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to index {self.path} ({self.format}): {exc}"
            ) from exc

    def _open_trajectory(self):
        if self._trajectory is None or self._pid != os.getpid():
            from ase.io.trajectory import TrajectoryReader

            self.close()
            self._trajectory = TrajectoryReader(self.path)
            self._pid = os.getpid()
        return self._trajectory

    def __len__(self):
        self._build_index()
        return self._length

    def _read_records(self, indices):
        from ase.io import read

        if self._offsets is not None:
            records = []
            with open(self.path, "rb") as stream:
                for index in indices:
                    offset, length = self._offsets[index]
                    stream.seek(int(offset))
                    frame = io.StringIO(stream.read(int(length)).decode("utf-8"))
                    records.append(
                        record_from_atoms(read(frame, index=0, format=self.format))
                    )
            return records
        if self.format == "traj":
            trajectory = self._open_trajectory()
            return [record_from_atoms(trajectory[index]) for index in indices]
        return [
            record_from_atoms(read(self.path, index=index, format=self.format))
            for index in indices
        ]

    @property
    def fast_random_access(self):
        return self.format == "traj" or (
            self.format in {"xyz", "extxyz"}
            and Path(self.path).suffix not in {".gz", ".bz2", ".xz"}
        )

    def iter_records(self):
        """One sequential decoder pass for formats without an offset index."""
        from ase.io import iread

        for atoms in iread(self.path, index=":", format=self.format):
            yield record_from_atoms(atoms)


class RecordSource(SourceSequence):
    def __init__(self, dataset, path):
        self.dataset, self.path = dataset, path

    def __len__(self):
        return len(self.dataset)

    def _read_records(self, indices):
        return self.dataset.get_records(indices)

    def close(self):
        self.dataset.close()


def cached_random_source(source, directory):
    """Materialize only non-random ASE formats in the typed raw-record store."""
    if (
        directory is None
        or not isinstance(source, ASEFileSource)
        or source.fast_random_access
    ):
        return source
    from .cache import read_cache_metadata
    from .store import RecordDataset, write_records

    stat = Path(source.path).stat()
    identity = {
        "schema": 1,
        "path": source.path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "format": source.format,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :24
    ]
    cache_directory = Path(directory) / f"source-{digest}"
    if read_cache_metadata([cache_directory], "source") is None:
        logging.info(
            "Indexing %s into bounded raw-record cache %s", source.path, cache_directory
        )
        dataset = write_records(
            source.iter_records(), [cache_directory], stage="source"
        )
    else:
        dataset = RecordDataset.from_directories([cache_directory], "source")
    return RecordSource(dataset, source.path)


# LMDB forbids opening the same file twice in a process. Share read-only handles;
# worker processes use their own PID key and never pickle live database handles.
_LMDB_HANDLES = OrderedDict()
_LMDB_LOCK = threading.RLock()


def close_inherited_lmdb_handles():
    global _LMDB_LOCK
    for environment in _LMDB_HANDLES.values():
        environment.close()
    _LMDB_HANDLES.clear()
    _LMDB_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=close_inherited_lmdb_handles)


class ASEDatabaseSource(SourceSequence):
    def __init__(self, path, *, lmdb=False):
        self.path = str(Path(path).expanduser().resolve())
        self.lmdb = lmdb
        self._ids = None
        self._sql_connection = None
        self._sql_db = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_sql_connection=None, _sql_db=None, _pid=None)
        return state

    def close(self):
        if self._sql_connection is not None:
            self._sql_connection.close()
            self._sql_connection = None
        self._sql_db = None
        self._pid = None
        if self.lmdb:
            # Transactions hold this lock, and every reader resolves the shared
            # handle anew for each transaction. Other sources can reopen it.
            with _LMDB_LOCK:
                for key in list(_LMDB_HANDLES):
                    if key == (os.getpid(), self.path):
                        _LMDB_HANDLES.pop(key).close()

    def _lmdb(self):
        import lmdb

        key = (os.getpid(), self.path)
        with _LMDB_LOCK:
            if key not in _LMDB_HANDLES:
                _LMDB_HANDLES[key] = lmdb.open(
                    self.path,
                    subdir=False,
                    readonly=True,
                    lock=False,
                    meminit=False,
                    readahead=False,
                )
            _LMDB_HANDLES.move_to_end(key)
            while len(_LMDB_HANDLES) > 32:
                _, environment = _LMDB_HANDLES.popitem(last=False)
                environment.close()
            return _LMDB_HANDLES[key]

    @contextmanager
    def _lmdb_transaction(self):
        # Keep an environment alive for the whole transaction during LRU eviction.
        with _LMDB_LOCK:
            with self._lmdb().begin(write=False) as transaction:
                yield transaction

    def _sqlite(self):
        if self._sql_connection is None or self._pid != os.getpid():
            from urllib.parse import quote

            from ase.db.sqlite import SQLite3Database

            self.close()
            self._sql_db = SQLite3Database(self.path, use_lock_file=False, serial=True)
            self._sql_connection = sqlite3.connect(
                f"file:{quote(self.path)}?mode=ro", uri=True
            )
            # ASE row conversion queries external-table metadata through its
            # managed connection. Reuse this reader instead of opening a new
            # default (read/write) SQLite connection for every converted row.
            self._sql_db.connection = self._sql_connection
            self._sql_db.change_count = 0
            self._sql_db._initialize(self._sql_connection)
            self._pid = os.getpid()
        return self._sql_connection, self._sql_db

    def __len__(self):
        if self._ids is None:
            if self.lmdb:
                from ase_db_backends.aselmdb import decode_bytestream, encode_key

                with self._lmdb_transaction() as transaction:
                    next_id = transaction.get(encode_key("nextid"))
                    deleted = transaction.get(encode_key("deleted_ids"))
                    if next_id is None and transaction.stat()["entries"]:
                        raise ValueError(
                            f"{self.path}: missing ASE-LMDB nextid metadata"
                        )
                next_id = (
                    1
                    if next_id is None
                    else decode_bytestream(next_id, decompress=True, json_decode=True)
                )
                if (
                    not isinstance(next_id, int)
                    or isinstance(next_id, bool)
                    or next_id < 1
                ):
                    raise ValueError(f"{self.path}: invalid ASE-LMDB nextid metadata")
                deleted = (
                    []
                    if deleted is None
                    else decode_bytestream(deleted, decompress=True, json_decode=True)
                )
                self._ids = np.arange(1, next_id, dtype=np.int64)
                if deleted:
                    self._ids = self._ids[~np.isin(self._ids, deleted)]
            else:
                connection, _ = self._sqlite()
                self._ids = np.fromiter(
                    (
                        row[0]
                        for row in connection.execute(
                            "SELECT id FROM systems ORDER BY id"
                        )
                    ),
                    dtype=np.int64,
                )
        return len(self._ids)

    def _read_records(self, indices):
        if self.lmdb:
            from ase.db.row import AtomsRow
            from ase_db_backends.aselmdb import decode_bytestream, encode_key

            with self._lmdb_transaction() as transaction:
                records = []
                for index in indices:
                    row_id = int(self._ids[index])
                    payload = transaction.get(encode_key(str(row_id)))
                    if payload is None:
                        raise KeyError(f"Database row {row_id} no longer exists")
                    values = decode_bytestream(
                        payload, decompress=True, json_decode=True
                    )
                    records.append(record_from_row(AtomsRow(values)))
            return records
        connection, db = self._sqlite()
        # One SELECT per bounded chunk, retaining requested order and duplicates.
        ids = [int(self._ids[index]) for index in indices]
        rows = {}
        for start in range(0, len(ids), 900):
            chunk = ids[start : start + 900]
            query = (
                "SELECT * FROM systems WHERE id IN ("
                + ",".join("?" for _ in chunk)
                + ")"
            )
            for values in connection.execute(query, chunk):
                row = db._convert_tuple_to_row(values)
                rows[row.id] = row
        return [record_from_row(rows[row_id]) for row_id in ids]


class BoundSource(SourceSequence):
    """Apply source fidelity lazily, while keeping raw labels independent of targets."""

    def __init__(
        self, source, keyspec, target_property, embedding_property, fidelity_idx=None
    ):
        self.source, self.keyspec = source, keyspec
        self.path = source.path
        self.target_property, self.embedding_property = (
            target_property,
            embedding_property,
        )
        self.fidelity_idx = (
            None
            if fidelity_idx is None
            else validate_fidelity(fidelity_idx, context=f"{self.path}: fidelity_idx")
        )

    def __len__(self):
        return len(self.source)

    def close(self):
        self.source.close()

    def _read_records(self, indices):
        records = self.source.get_records(indices)
        for index, record in zip(indices, records):
            value = resolve_property(record, "fidelity_idx", self.keyspec)
            if value is not None:
                value = validate_fidelity(
                    value, context=f"{self.path}: structure {index} fidelity_idx"
                )
            if self.fidelity_idx is not None:
                if value is not None and value != self.fidelity_idx:
                    raise ValueError(
                        f"{self.path}: structure {index} (zero-based) has fidelity_idx={value}, but the data source specifies fidelity_idx={self.fidelity_idx}"
                    )
                record.info[
                    self.keyspec.info_keys.get("fidelity_idx", "fidelity_idx")
                ] = self.fidelity_idx
        return records

    def _atoms(self, record):
        return check_keys(
            [record.to_atoms()],
            self.target_property,
            self.keyspec,
            self.embedding_property,
        )[0]


class ConcatSource(SourceSequence):
    def __init__(self, sources):
        self.sources = list(sources)
        self._ends = np.cumsum([len(source) for source in self.sources], dtype=np.int64)

    def __len__(self):
        return int(self._ends[-1]) if len(self._ends) else 0

    def close(self):
        for source in self.sources:
            source.close()

    def _read_records(self, indices):
        groups = {}
        output = [None] * len(indices)
        for position, index in enumerate(indices):
            source_index = bisect.bisect_right(self._ends, index)
            offset = int(self._ends[source_index - 1]) if source_index else 0
            groups.setdefault(source_index, []).append((position, index - offset))
        for source_index, items in groups.items():
            records = self.sources[source_index].get_records(
                [item[1] for item in items]
            )
            for (position, _), record in zip(items, records):
                output[position] = record
        return output

    def getitems(self, indices):
        indices = [self._index(index) for index in indices]
        records = self.get_records(indices)
        return [
            self.sources[bisect.bisect_right(self._ends, index)]._atoms(record)
            for index, record in zip(indices, records)
        ]

    __getitems__ = getitems

    def __getitem__(self, index):
        if isinstance(index, slice):
            return IndexedSource(self, range(*index.indices(len(self))))
        return self.getitems([index])[0]


@dataclass
class SourceSpec:
    path: str
    reader: str = "ase"
    fidelity_idx: int | None = None


RGLOB = {
    "ase_db": ["*.db"],
    "fair_aselmdb": ["*.aselmdb"],
    "aselmdb": ["*.aselmdb"],
}


def stage_readers(dataset):
    """Assign readers in train, validation, then test source-entry order."""
    removed = {"type", "format", "options"}.intersection(dataset)
    if removed:
        raise ValueError(
            f"Unsupported dataset settings: {sorted(removed)}. "
            "Use dataset.reader; file formats are detected automatically."
        )
    groups = {
        "train": dataset.get("train_file"),
        "valid": dataset.get("valid_file"),
    }
    tests = dataset.get("test_files")
    if tests is not None:
        tests = [tests] if isinstance(tests, (str, Path, Mapping)) else tests
        groups.update((f"test{index}", source) for index, source in enumerate(tests))
    counts = {}
    for stage, source in groups.items():
        if source is None:
            counts[stage] = 0
        elif isinstance(source, (str, Path, Mapping)):
            counts[stage] = 1
        else:
            counts[stage] = len(source)
    reader = dataset.get("reader", "ase")
    if not isinstance(reader, Sequence):
        raise ValueError("dataset.reader must be a string or a list of strings")
    if isinstance(reader, str):
        values = [reader]
    else:
        total = sum(counts.values())
        if len(reader) != total:
            raise ValueError(
                f"dataset.reader has {len(reader)} values, expected {total} "
                f"for train/valid/test source entries: {counts}"
            )
        values = reader
    for value in values:
        if not isinstance(value, str) or value not in HOW_TO_READ:
            raise ValueError(f"Unknown dataset reader: {value!r}")
    if isinstance(reader, str):
        return {stage: reader for stage in groups}
    result = {}
    offset = 0
    for stage, count in counts.items():
        result[stage] = list(reader[offset : offset + count])
        offset += count
    if dataset.get("valid_file") is None:
        result["valid"] = result["train"]
    return result


def source_specs(filename, reader="ase"):
    """Assign a reader to each source before directories are expanded or sorted."""
    sources = (
        [filename] if isinstance(filename, (str, Path, Mapping)) else list(filename)
    )
    if not isinstance(reader, Sequence):
        raise ValueError("dataset.reader must be a string or a list of strings")
    if not isinstance(reader, str) and len(reader) != len(sources):
        raise ValueError("reader must have one value per source entry")
    specs = []
    for index, entry in enumerate(sources):
        entry = dict(entry) if isinstance(entry, Mapping) else {"path": entry}
        if "path" not in entry or entry.keys() - {"path", "fidelity_idx"}:
            raise ValueError("Dataset entries require path and optionally fidelity_idx")
        path = Path(entry["path"]).expanduser().resolve()
        source_reader = reader if isinstance(reader, str) else reader[index]
        if not isinstance(source_reader, str) or source_reader not in HOW_TO_READ:
            raise ValueError(f"Unknown dataset reader: {source_reader!r}")
        fidelity = entry.get("fidelity_idx")
        if fidelity is not None:
            fidelity = validate_fidelity(fidelity, context=f"{path}: fidelity_idx")
        specs.append(SourceSpec(str(path), source_reader, fidelity))
    return specs


def discover_sources(filename, reader="ase"):
    """Resolve deterministic unique files with automatic ASE format detection."""
    found = {}
    for source in source_specs(filename, reader):
        path = Path(source.path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if path.is_file():
            files = [path]
        elif path.is_dir():
            if source.reader == "ase":
                from ase.io.formats import UnknownFileTypeError, filetype, ioformats

                files = []
                for file in path.rglob("*"):
                    if not file.is_file():
                        continue
                    try:
                        fmt = ioformats.get(filetype(str(file), read=False))
                    except UnknownFileTypeError:
                        continue
                    if fmt is not None and fmt.can_read:
                        files.append(file)
                files.sort()
            else:
                files = sorted(
                    {
                        file
                        for pattern in RGLOB[source.reader]
                        for file in path.rglob(pattern)
                        if file.is_file()
                    }
                )
        else:
            raise ValueError(f"Unsupported path type: {path}")
        for file in files:
            spec = SourceSpec(str(file.resolve()), source.reader, source.fidelity_idx)
            previous = found.get(spec.path)
            if previous is not None:
                if previous.reader != spec.reader:
                    raise ValueError(f"{file}: conflicting source readers")
                if (
                    previous.fidelity_idx is not None
                    and spec.fidelity_idx is not None
                    and previous.fidelity_idx != spec.fidelity_idx
                ):
                    raise ValueError(
                        f"{file}: conflicting source fidelity_idx values "
                        f"{previous.fidelity_idx} and {spec.fidelity_idx}"
                    )
                if spec.fidelity_idx is None:
                    continue
            found[spec.path] = spec
    if not found:
        raise FileNotFoundError("No dataset files found in the provided paths")
    return [found[path] for path in sorted(found)]


HOW_TO_READ = {
    "ase": ASEFileSource,
    "ase_db": ASEDatabaseSource,
    "fair_aselmdb": partial(ASEDatabaseSource, lmdb=True),
    "aselmdb": partial(ASEDatabaseSource, lmdb=True),
}


def read_single_file(
    fpath,
    target_property,
    keyspec,
    embedding_property=None,
    reader="ase",
    fidelity_idx=None,
    *,
    record_cache_dir=None,
):
    source = HOW_TO_READ[reader](str(fpath))
    source = cached_random_source(source, record_cache_dir)
    return BoundSource(
        source, keyspec, target_property, embedding_property or [], fidelity_idx
    )


def read_all_files(
    filename,
    target_property,
    keyspec,
    embedding_property=None,
    reader="ase",
    *,
    record_cache_dir=None,
):
    specs = discover_sources(filename, reader)
    sources = [
        read_single_file(
            spec.path,
            target_property,
            keyspec,
            embedding_property,
            spec.reader,
            spec.fidelity_idx,
            record_cache_dir=record_cache_dir,
        )
        for spec in specs
    ]
    combined = ConcatSource(sources)
    logging.info(
        "Indexed %d structures across %d source files (lazy, no full Atoms list)",
        len(combined),
        len(sources),
    )
    return combined


def tace_read_all_files(
    cfg,
    target_property,
    embedding_property,
    keyspec,
    *,
    stages=("train", "valid", "test"),
):
    """Read requested stages, with validation disabled before any split selection."""
    stages = set(stages)
    if stages - {"train", "valid", "test"}:
        raise ValueError(f"Unknown dataset stages: {sorted(stages)}")
    dataset = cfg["dataset"]
    readers = stage_readers(dataset)
    directories = dataset.get("shard_dirs", ["graphCache"])
    record_directory = (
        Path(
            directories if isinstance(directories, (str, Path)) else directories[0]
        ).expanduser()
        / "records"
    )

    def read(source, stage):
        return read_all_files(
            source,
            target_property,
            keyspec,
            embedding_property,
            readers[stage],
            record_cache_dir=dataset.get(
                "record_cache_dir",
                record_directory,
            ),
        )

    train = valid = None
    if stages == {"valid"} and dataset.get("valid_file") is not None:
        valid = read(dataset["valid_file"], "valid")
        if len(valid) == 0:
            raise ValueError("Validation set is empty !")
    elif stages & {"train", "valid"}:
        train_file = dataset.get("train_file")
        if not train_file:
            raise ValueError(
                "No valid training dataset provided. Please check cfg.dataset.train_file"
            )
        train = read(train_file, "train")
        no_valid = dataset.get("no_valid_set", False)
        if no_valid:
            pass
        elif dataset.get("valid_file") is not None:
            if "valid" in stages:
                valid = read(dataset["valid_file"], "valid")
        elif dataset.get("valid_from_index", False):
            valid_path, train_path = Path("valid.index"), Path("train.index")
            valid_indices = [
                int(line)
                for line in valid_path.read_text().splitlines()
                if line.strip()
            ]
            if train_path.exists():
                train_indices = [
                    int(line)
                    for line in train_path.read_text().splitlines()
                    if line.strip()
                ]
                overlap = set(train_indices) & set(valid_indices)
                if overlap:
                    raise ValueError(
                        "train.index and valid.index overlap at structure indices "
                        f"{sorted(overlap)[:8]}"
                    )
            else:
                validation = set(valid_indices)
                train_indices = [
                    index for index in range(len(train)) if index not in validation
                ]
            valid, train = (
                IndexedSource(train, valid_indices),
                IndexedSource(train, train_indices),
            )
        else:
            ratio = dataset.get("valid_ratio")
            if not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
                raise ValueError(
                    "valid_ratio must be in (0, 1) when no validation file is given"
                )
            train, valid = random_split(train, ratio, dataset.get("split_seed", 1))
        if len(train) == 0:
            raise ValueError("Training set is empty !")
        if "valid" in stages and not no_valid and (valid is None or len(valid) == 0):
            raise ValueError("Validation set is empty !")

    test_files = dataset.get("test_files") if "test" in stages else None
    tests = (
        None
        if test_files is None
        else [
            read(source, f"test{index}")
            for index, source in enumerate(
                [test_files]
                if isinstance(test_files, (str, Path, Mapping))
                else test_files
            )
        ]
    )
    return ThreeDataset(train, valid, tests)
