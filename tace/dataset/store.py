################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import bisect
import hashlib
import json
import math
import operator
import os
import struct
import tempfile
import threading
import zlib
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import lmdb
import numpy as np
import torch
from filelock import FileLock, Timeout
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .cache import (
    ARRAY_CODEC,
    COMPRESSED_ARRAY_CODEC,
    GRAPH_SCHEMA,
    RECORD_SCHEMA,
    digest,
    read_cache_metadata,
    validate_stage,
)

_MAGIC = b"TACEARR\0"
_HEADER = struct.Struct("<8sII")
_COMPRESSED_MAGIC = b"TACEZLB\0"
_COMPRESSED_HEADER = struct.Struct("<8sQ")
_MAX_COMPRESSED_RECORD_BYTES = 2 << 30
_COUNTS = struct.Struct("<qq")
_FORBIDDEN = {"node_attrs", "node_type"}
_INDEX_FIELDS = {"atomic_numbers", "edge_index", "edge_shifts", "fidelity_idx"}
_MIB = 1024 * 1024
DEFAULT_SHARD_SIZE = 100_000
DEFAULT_SHARD_BYTES = 2048 * _MIB
DEFAULT_MAX_BUFFER_BYTES = 64 * _MIB
DEFAULT_TRANSACTION_BYTES = 16 * _MIB
DEFAULT_MAP_SIZE = 64 * _MIB
DEFAULT_CACHE_BYTES = 128 * _MIB
DEFAULT_MAX_OPEN_SHARDS = 4096
try:
    import resource

    _descriptor_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if _descriptor_limit != resource.RLIM_INFINITY:
        # Leave descriptors for other stores, worker queues and shared tensors.
        DEFAULT_MAX_OPEN_SHARDS = min(
            DEFAULT_MAX_OPEN_SHARDS, max(1, _descriptor_limit // 8)
        )
except ImportError:
    # Windows has no POSIX descriptor limit; LMDB owns native file handles.
    pass


def check_record_bytes(size, maximum):
    if maximum is not None and size > maximum:
        raise ValueError(
            f"A record requires at least {size} bytes, exceeding max_buffer_bytes={maximum}"
        )


def encode_array_record(mapping, kind, max_bytes=None):
    arrays, payloads = [], []
    offset = 0

    def pack(value, path):
        nonlocal offset
        tensor = torch.is_tensor(value)
        if tensor or isinstance(value, np.ndarray):
            if tensor:
                if (
                    kind == "graphs"
                    and value.is_floating_point()
                    and value.dtype != torch.float64
                ):
                    raise ValueError(f"{path} uses {value.dtype}, not float64")
                if (
                    kind == "graphs"
                    and path in _INDEX_FIELDS
                    and value.dtype != torch.int64
                ):
                    raise ValueError(f"{path} must use int64 indices")
                itemsize = 8 if value.is_floating_point() else value.element_size()
                check_record_bytes(
                    _HEADER.size + offset + value.numel() * itemsize, max_bytes
                )
                if kind == "records" and value.is_floating_point():
                    value = value.to(dtype=torch.float64)
                value = value.detach().cpu().contiguous().numpy()
            array = np.asarray(value)
            if array.dtype.kind not in "biufcSU":
                raise TypeError(f"Unsupported array dtype at {path}: {array.dtype}")
            if array.dtype.kind == "f":
                if kind == "graphs" and array.dtype.itemsize != 8:
                    raise ValueError(f"{path} uses {array.dtype}, not float64")
                dtype = np.dtype("<f8")
            else:
                dtype = array.dtype.newbyteorder("<")
            if (
                kind == "graphs"
                and path in _INDEX_FIELDS
                and (array.dtype.kind != "i" or array.dtype.itemsize != 8)
            ):
                raise ValueError(f"{path} must use int64 indices")
            check_record_bytes(
                _HEADER.size + offset + array.size * dtype.itemsize, max_bytes
            )
            array = np.ascontiguousarray(array.astype(dtype, copy=False)).reshape(
                array.shape
            )
            raw = array.tobytes(order="C")
            index = len(arrays)
            arrays.append(
                [dtype.str, list(array.shape), offset, len(raw), bool(tensor)]
            )
            payloads.append(raw)
            offset += len(raw)
            return ["a", index]
        if isinstance(value, np.floating):
            return pack(float(value), path)
        if isinstance(value, np.generic):
            scalar = value.item()
            if isinstance(scalar, np.generic):
                raise TypeError(f"Unsupported scalar dtype at {path}: {value.dtype}")
            return pack(scalar, path)
        if isinstance(value, Mapping):
            entries = []
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool)) and key is not None:
                    raise TypeError(
                        f"Unsupported mapping key at {path}: {type(key).__name__}"
                    )
                entries.append([key, pack(item, f"{path}.{key}")])
            return ["d", entries]
        if isinstance(value, (list, tuple)):
            return [
                "t" if isinstance(value, tuple) else "l",
                [pack(item, f"{path}[{i}]") for i, item in enumerate(value)],
            ]
        if kind == "records" and isinstance(value, float) and not math.isfinite(value):
            return [
                "nonfinite",
                "nan" if math.isnan(value) else "inf" if value > 0 else "-inf",
            ]
        if value is None or isinstance(value, (str, int, float, bool)):
            return ["s", value]
        if kind == "records":
            from ase.spacegroup import Spacegroup

            if isinstance(value, Spacegroup):
                return ["spacegroup", [int(value.no), int(value.setting)]]
        raise TypeError(f"Unsupported value at {path}: {type(value).__name__}")

    if kind == "graphs" and _FORBIDDEN.intersection(mapping):
        raise ValueError(
            "Graph stores must not contain model-dependent node_attrs/node_type"
        )
    fields = {name: pack(value, name) for name, value in mapping.items()}
    header = json.dumps(
        {"kind": kind, "fields": fields, "arrays": arrays},
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    check_record_bytes(_HEADER.size + len(header) + offset, max_bytes)
    return b"".join((_HEADER.pack(_MAGIC, 1, len(header)), header, *payloads))


def compress_array_record(value):
    """Losslessly compress an encoded record with a bounded decoded length."""
    view = memoryview(value)
    if not _HEADER.size <= len(view) <= _MAX_COMPRESSED_RECORD_BYTES:
        raise ValueError("Invalid typed-array record size for compression")
    magic, version, _ = _HEADER.unpack_from(view)
    if magic != _MAGIC or version != 1:
        raise ValueError("Compression requires an uncompressed typed-array record")
    return _COMPRESSED_HEADER.pack(_COMPRESSED_MAGIC, len(view)) + zlib.compress(
        view, 1
    )


def decode_array_record(value, kind, fields=None):
    view = memoryview(value)
    if len(view) < _HEADER.size:
        raise ValueError("Truncated typed-array record")
    if view[:8] == _COMPRESSED_MAGIC:
        _, size = _COMPRESSED_HEADER.unpack_from(view)
        if not _HEADER.size <= size <= _MAX_COMPRESSED_RECORD_BYTES:
            raise ValueError("Invalid compressed typed-array record size")
        decoder = zlib.decompressobj()
        try:
            payload = decoder.decompress(view[_COMPRESSED_HEADER.size :], size + 1)
        except zlib.error as exc:
            raise ValueError("Invalid compressed typed-array record") from exc
        if len(payload) != size or not decoder.eof or decoder.unused_data:
            raise ValueError("Truncated or oversized compressed typed-array record")
        view = memoryview(payload)
    magic, version, header_size = _HEADER.unpack_from(view)
    if magic != _MAGIC or version != 1:
        raise ValueError(
            "Unsupported store record; legacy pickle caches must be rebuilt"
        )
    payload_start = _HEADER.size + header_size
    if payload_start > len(view):
        raise ValueError("Truncated typed-array record header")
    header = json.loads(bytes(view[_HEADER.size : payload_start]))
    if (
        not isinstance(header, dict)
        or not isinstance(header.get("fields"), dict)
        or not isinstance(header.get("arrays"), list)
    ):
        raise ValueError("Invalid typed-array record metadata")
    if header.get("kind") != kind:
        raise ValueError(f"Expected {kind} record, found {header.get('kind')!r}")
    if kind == "graphs" and _FORBIDDEN.intersection(header["fields"]):
        raise ValueError("Model-dependent features are not allowed in graph stores")
    descriptors = header["arrays"]
    decoded = {}

    def unpack(node):
        tag, content = node
        if tag == "a":
            if (
                not isinstance(content, int)
                or content < 0
                or content >= len(descriptors)
            ):
                raise ValueError("Invalid typed-array reference")
            if content in decoded:
                return decoded[content]
            dtype, shape, offset, nbytes, tensor = descriptors[content]
            dtype = np.dtype(dtype)
            if dtype.kind not in "biufcSU" or dtype.hasobject or dtype.itemsize <= 0:
                raise ValueError("Invalid typed-array dtype")
            if dtype.kind == "f" and dtype.itemsize != 8:
                raise ValueError("Stored floating point arrays must use float64")
            if not isinstance(shape, list) or any(
                type(n) is not int or n < 0 for n in shape
            ):
                raise ValueError("Invalid typed-array shape")
            count = math.prod(shape)
            if (
                type(offset) is not int
                or type(nbytes) is not int
                or type(tensor) is not bool
                or offset < 0
                or nbytes < 0
                or nbytes != count * dtype.itemsize
                or payload_start + offset + nbytes > len(view)
            ):
                raise ValueError("Invalid typed-array payload bounds")
            # Own the returned storage: no tensor can outlive a transaction buffer.
            array = (
                np.frombuffer(
                    view, dtype=dtype, count=count, offset=payload_start + offset
                )
                .copy()
                .reshape(shape)
            )
            if not array.dtype.isnative:
                array = array.astype(array.dtype.newbyteorder("="))
            result = torch.from_numpy(array) if tensor else array
            decoded[content] = result
            return result
        if tag == "d":
            return {key: unpack(item) for key, item in content}
        if tag in {"l", "t"}:
            result = [unpack(item) for item in content]
            return tuple(result) if tag == "t" else result
        if tag == "s":
            return content
        if tag == "nonfinite" and kind == "records":
            if content not in {"nan", "inf", "-inf"}:
                raise ValueError("Invalid nonfinite scalar tag")
            return float(content)
        if tag == "spacegroup" and kind == "records":
            from ase.spacegroup import Spacegroup

            number, setting = content
            return Spacegroup(number, setting=setting)
        raise ValueError(f"Invalid typed-array node: {tag!r}")

    selected = None if fields is None else set(fields) | {"num_nodes"}
    return {
        name: unpack(node)
        for name, node in header["fields"].items()
        if selected is None or name in selected
    }


def encode_graph(graph, max_bytes=None):
    """Encode a model-independent FP64 graph without Python object serialization."""
    mapping = graph.to_dict() if hasattr(graph, "to_dict") else dict(graph)
    if "num_nodes" not in mapping:
        mapping["num_nodes"] = int(graph.num_nodes)
    return encode_array_record(mapping, "graphs", max_bytes=max_bytes)


def decode_graph(value, fields=None):
    return Data(**decode_array_record(value, "graphs", fields))


def encode_record(record, max_bytes=None):
    return encode_array_record(
        record.to_dict() if hasattr(record, "to_dict") else dict(record),
        "records",
        max_bytes=max_bytes,
    )


def decode_record(value):
    from .records import AtomicRecord

    return AtomicRecord.from_dict(decode_array_record(value, "records"))


_ENVIRONMENTS = {}
_ENV_LOCK = threading.RLock()


def close_entry(entry):
    entry["txn"].abort()
    entry["env"].close()


def before_fork():
    _ENV_LOCK.acquire()
    # The parent also lazily reopens. A child never inherits a live LMDB handle.
    for entry in _ENVIRONMENTS.values():
        close_entry(entry)
    _ENVIRONMENTS.clear()


def after_fork_parent():
    _ENV_LOCK.release()


def after_fork_child():
    global _ENV_LOCK
    _ENV_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=before_fork,
        after_in_parent=after_fork_parent,
        after_in_child=after_fork_child,
    )


class GraphDatasetLMDB(Dataset):
    """A process-local, batch-readable view of immutable typed-array shards."""

    kind = "graphs"

    def __init__(
        self,
        lmdb_paths,
        in_memory=False,
        cache_size=0,
        shard_lengths=None,
        fields=None,
        cache_bytes=DEFAULT_CACHE_BYTES,
        max_open_shards=DEFAULT_MAX_OPEN_SHARDS,
        readahead=False,
    ):
        self.lmdb_paths = [Path(path).resolve() for path in lmdb_paths]
        self.in_memory = in_memory
        self.cache_size = max(0, int(cache_size))
        self.cache_bytes = max(0, int(cache_bytes))
        self.max_open_shards = max(1, int(max_open_shards))
        self.readahead = readahead
        self.fields = None if fields is None else tuple(sorted(set(fields)))
        self.cache = OrderedDict()
        self._cached_bytes = 0
        self._opened = OrderedDict()
        self._pid = os.getpid()
        self.metadata = None
        self.indices = None
        self.sample_fingerprint = None
        self.data_list = []
        self.lengths = []
        self.length = 0
        if not in_memory:
            if shard_lengths is None:
                for shard in range(len(self.lmdb_paths)):
                    with _ENV_LOCK:
                        self.lengths.append(self._transaction(shard).stat()["entries"])
                self.close()
            else:
                if len(shard_lengths) != len(self.lmdb_paths):
                    raise ValueError(
                        "shard_lengths must match the number of LMDB paths"
                    )
                self.lengths = [operator.index(length) for length in shard_lengths]
            if any(length < 0 for length in self.lengths):
                raise ValueError("Shard lengths must be non-negative")
            self.length = sum(self.lengths)
        self._ends = np.cumsum(self.lengths, dtype=np.int64).tolist()

    @classmethod
    def from_directories(cls, directories, stage, *, allow_building=False, **kwargs):
        directories = list(directories)
        metadata = read_cache_metadata(
            directories, stage, allow_building=allow_building
        )
        if metadata is None or metadata["kind"] != cls.kind:
            raise ValueError(f"No completed {cls.kind} store for {stage}")
        paths = sorted(
            (
                path
                for directory in map(Path, directories)
                for path in directory.glob(f"{stage}_shard*.lmdb")
            ),
            key=lambda path: path.name,
        )
        dataset = cls(
            paths,
            shard_lengths=[metadata["shards"][path.name] for path in paths],
            **kwargs,
        )
        dataset.metadata = metadata
        return dataset

    def __len__(self):
        return self.length if self.indices is None else len(self.indices)

    def _ensure_process(self):
        if self._pid != os.getpid():
            self._opened = OrderedDict()
            self.cache = OrderedDict()
            self._cached_bytes = 0
            self._pid = os.getpid()

    def _release(self, path):
        entry = _ENVIRONMENTS.get(path)
        if entry is not None:
            entry["owners"].discard(id(self))
            if not entry["owners"]:
                close_entry(entry)
                del _ENVIRONMENTS[path]
        self._opened.pop(path, None)

    def _transaction(self, shard):
        self._ensure_process()
        path = str(self.lmdb_paths[shard])
        entry = _ENVIRONMENTS.get(path)
        if entry is None:
            env = lmdb.open(
                path,
                readonly=True,
                create=False,
                lock=False,
                readahead=self.readahead,
                max_readers=512,
            )
            try:
                entry = {"env": env, "txn": env.begin(buffers=True), "owners": set()}
            except BaseException:
                env.close()
                raise
            _ENVIRONMENTS[path] = entry
        entry["owners"].add(id(self))
        self._opened[path] = None
        self._opened.move_to_end(path)
        while len(self._opened) > self.max_open_shards:
            self._release(next(iter(self._opened)))
        return entry["txn"]

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of length {len(self)}"
            )
        return index

    def _decode(self, value):
        return decode_graph(value, self.fields)

    def _project(self, graph):
        if self.fields is None:
            return graph
        return Data(
            num_nodes=graph.num_nodes,
            **{
                name: value
                for name, value in graph.items()
                if name in self.fields and name != "num_nodes"
            },
        )

    def _remember(self, index, graph):
        if not self.cache_size or not self.cache_bytes:
            return
        size = (
            sum(
                value.numel() * value.element_size()
                for _, value in graph.items()
                if torch.is_tensor(value)
            )
            + 1024
        )
        if size > self.cache_bytes:
            return
        previous = self.cache.pop(index, None)
        if previous is not None:
            self._cached_bytes -= previous[1]
        self.cache[index] = (graph, size)
        self._cached_bytes += size
        while (
            len(self.cache) > self.cache_size or self._cached_bytes > self.cache_bytes
        ):
            _, (_, evicted_size) = self.cache.popitem(last=False)
            self._cached_bytes -= evicted_size

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        self._ensure_process()
        indices = [self._index(index) for index in indices]
        if self.indices is not None:
            indices = [int(self.indices[index]) for index in indices]
        if self.in_memory:
            return [self._project(self.data_list[index]) for index in indices]
        output = [None] * len(indices)
        groups = {}
        for position, index in enumerate(indices):
            cached = self.cache.get(index)
            if cached is not None:
                self.cache.move_to_end(index)
                output[position] = cached[0]
                continue
            shard = bisect.bisect_right(self._ends, index)
            local = index - (self._ends[shard - 1] if shard else 0)
            groups.setdefault(shard, {}).setdefault(local, []).append((position, index))
        for shard, requests in groups.items():
            with _ENV_LOCK:
                txn = self._transaction(shard)
                with txn.cursor() as cursor:
                    keys = [
                        f"{local:09d}".encode("ascii") for local in sorted(requests)
                    ]
                    values = cursor.getmulti(keys)
                    if len(values) != len(keys):
                        raise IndexError(
                            f"Missing record in LMDB shard {self.lmdb_paths[shard]}"
                        )
                    for key, value in values:
                        local = int(bytes(key))
                        graph = self._decode(value)
                        for position, index in requests[local]:
                            output[position] = graph
                        self._remember(requests[local][0][1], graph)
        return output

    def project(self, fields):
        """Select decoded fields, or use ``None`` to reopen all stored fields."""
        if self.fields is not None and fields is not None:
            fields = set(fields).intersection(self.fields)
        dataset = type(self)(
            self.lmdb_paths,
            in_memory=self.in_memory,
            cache_size=self.cache_size,
            shard_lengths=self.lengths,
            fields=fields,
            cache_bytes=self.cache_bytes,
            max_open_shards=self.max_open_shards,
            readahead=self.readahead,
        )
        dataset.data_list, dataset.length = self.data_list, self.length
        dataset.metadata = self.metadata
        dataset.indices = self.indices
        dataset.sample_fingerprint = self.sample_fingerprint
        return dataset

    def select(self, indices):
        """Return an immutable view of increasing, unique sample indices.

        Selection is applied before sampling and preserves batched LMDB reads,
        field projection, and the original shard files.
        """
        indices = np.asarray(indices)
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
            raise ValueError(
                "Selection indices must be a one-dimensional integer array"
            )
        if indices.size and (
            indices[0] < 0
            or indices[-1] >= len(self)
            or np.any(indices[1:] <= indices[:-1])
        ):
            raise ValueError(
                "Selection indices must be increasing, unique and in range"
            )
        indices = indices.astype(np.int64, copy=True)
        selected = self.project(self.fields)
        selected.indices = indices if self.indices is None else self.indices[indices]
        selected.indices.flags.writeable = False
        fingerprint = hashlib.sha256()
        fingerprint.update(
            digest(
                [(path.name, n) for path, n in zip(self.lmdb_paths, self.lengths)]
            ).encode()
        )
        fingerprint.update(selected.indices)
        selected.sample_fingerprint = fingerprint.hexdigest()
        return selected

    def get_sizes(self, indices=None):
        """Read atom/edge counts without loading or decoding graph records."""
        if indices is None and not self.in_memory:
            output = np.empty((len(self), 2), dtype=np.int64)
            start = 0
            for path, length in zip(self.lmdb_paths, self.lengths):
                if length:
                    counts = np.memmap(
                        path / "counts.bin", mode="r", dtype="<i8", shape=(length, 2)
                    )
                    if self.indices is None:
                        output[start : start + length] = counts
                    else:
                        first, last = np.searchsorted(
                            self.indices, [start, start + length]
                        )
                        output[first:last] = counts[self.indices[first:last] - start]
                    del counts
                start += length
            return output
        indices = range(len(self)) if indices is None else indices
        indices = [self._index(index) for index in indices]
        if self.indices is not None:
            indices = [int(self.indices[index]) for index in indices]
        if self.in_memory:
            return np.asarray(
                [
                    (self.data_list[i].num_nodes, self.data_list[i].num_edges)
                    for i in indices
                ],
                dtype=np.int64,
            ).reshape(-1, 2)
        output = np.empty((len(indices), 2), dtype=np.int64)
        groups = {}
        for row, index in enumerate(indices):
            shard = bisect.bisect_right(self._ends, index)
            local = index - (self._ends[shard - 1] if shard else 0)
            groups.setdefault(shard, []).append((row, local))
        for shard, requests in groups.items():
            counts = np.memmap(
                self.lmdb_paths[shard] / "counts.bin",
                mode="r",
                dtype="<i8",
                shape=(self.lengths[shard], 2),
            )
            rows, local = zip(*requests)
            output[list(rows)] = counts[list(local)]
            del counts
        return output

    @property
    def num_nodes(self):
        return self.get_sizes()[:, 0]

    @property
    def num_edges(self):
        return self.get_sizes()[:, 1]

    def close(self):
        self._ensure_process()
        with _ENV_LOCK:
            for path in list(self._opened):
                self._release(path)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(
            _opened=OrderedDict(), cache=OrderedDict(), _cached_bytes=0, _pid=None
        )
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RecordDataset(GraphDatasetLMDB):
    """Random access to canonical raw records, independent of graph settings."""

    kind = "records"

    def _decode(self, value):
        return decode_record(value)

    def _project(self, record):
        return record

    def _remember(self, index, record):
        return

    def get_record(self, index):
        return self[index]

    def get_records(self, indices):
        return self.__getitems__(indices)


def write_store_locked(
    items,
    shard_dirs,
    stage,
    *,
    kind,
    compressed=True,
    shard_size=DEFAULT_SHARD_SIZE,
    max_buffer_bytes=DEFAULT_MAX_BUFFER_BYTES,
    transaction_bytes=DEFAULT_TRANSACTION_BYTES,
    shard_bytes=DEFAULT_SHARD_BYTES,
    cache_size=0,
    max_open_shards=DEFAULT_MAX_OPEN_SHARDS,
    map_size=DEFAULT_MAP_SIZE,
    progress=None,
    graph_config=None,
):
    if (
        shard_size <= 0
        or max_buffer_bytes <= 0
        or transaction_bytes <= 0
        or shard_bytes <= 0
    ):
        raise ValueError("Store size and byte budgets must be positive")
    directories = shard_dirs
    env = counts_stream = None
    temporary = None
    build_id = uuid4().hex
    atomic_numbers, all_fields = set(), set()
    transaction_bytes = min(transaction_bytes, max_buffer_bytes)
    buffer, buffered_bytes = [], 0
    shard_index = shard_length = shard_total_bytes = 0
    shard_nodes = shard_edges = 0

    def flush():
        nonlocal buffer, buffered_bytes
        if not buffer:
            return
        for attempt in range(16):
            try:
                with env.begin(write=True) as txn:
                    with txn.cursor() as cursor:
                        consumed, added = cursor.putmulti(
                            ((key, value) for key, value, _, _ in buffer),
                            append=True,
                        )
                        if consumed != len(buffer) or added != len(buffer):
                            raise RuntimeError(
                                "Failed to write all records in a transaction"
                            )
                break
            except lmdb.MapFullError:
                if attempt == 15:
                    raise
                env.set_mapsize(env.info()["map_size"] * 2)
        for _, _, nodes, edges in buffer:
            counts_stream.write(_COUNTS.pack(nodes, edges))
        written = len(buffer)
        buffer = []
        buffered_bytes = 0
        if progress is not None:
            progress(written)

    def finish_shard():
        nonlocal env, counts_stream, temporary, shard_index
        nonlocal shard_length, shard_total_bytes, shard_nodes, shard_edges
        if env is None:
            return
        flush()
        counts_stream.flush()
        os.fsync(counts_stream.fileno())
        counts_stream.close()
        counts_stream = None
        env.sync()
        env.close()
        env = None
        metadata = {
            "schema": GRAPH_SCHEMA if kind == "graphs" else RECORD_SCHEMA,
            "kind": kind,
            "codec": COMPRESSED_ARRAY_CODEC if compressed else ARRAY_CODEC,
            "dtype": "float64",
            "stage": stage,
            "length": shard_length,
            "atomic_numbers": sorted(atomic_numbers),
            "fields": sorted(all_fields),
            "counts": "counts.bin",
            "num_nodes": shard_nodes,
            "num_edges": shard_edges,
        }
        if graph_config is not None:
            metadata["graph_config"] = {
                key: graph_config[key]
                for key in ("cutoff", "max_neighbors")
                if key in graph_config
            }
        with (temporary / "metadata.json").open("w") as stream:
            json.dump(metadata, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        destination = (
            directories[shard_index % len(directories)]
            / f"{stage}_shard{build_id}_{shard_index:06d}.lmdb"
        )
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing shard {destination}")
        os.rename(temporary, destination)
        temporary = None
        atomic_numbers.clear()
        all_fields.clear()
        shard_index += 1
        shard_length = shard_total_bytes = shard_nodes = shard_edges = 0

    try:
        for item in items:
            if kind == "graphs":
                value = encode_graph(item, max_bytes=max_buffer_bytes)
                numbers = item["atomic_numbers"]
                nodes, edges = int(item.num_nodes), int(item.num_edges)
                fields = set(item.keys()) | {"num_nodes"}
            else:
                value = encode_record(item, max_bytes=max_buffer_bytes)
                numbers = item.numbers
                nodes, edges = len(item), 0
                fields = item.to_dict()
            if compressed:
                value = compress_array_record(value)
                check_record_bytes(len(value), max_buffer_bytes)
            if env is not None and (
                shard_length >= shard_size
                or shard_total_bytes + len(value) > shard_bytes
            ):
                finish_shard()
            atomic_numbers.update(
                int(number)
                for number in np.unique(
                    numbers.detach().cpu().numpy()
                    if torch.is_tensor(numbers)
                    else numbers
                )
            )
            all_fields.update(fields)
            if env is None:
                directory = directories[shard_index % len(directories)]
                temporary = Path(
                    tempfile.mkdtemp(
                        prefix=f".{stage}_shard{build_id}_{shard_index:06d}-",
                        suffix=".tmp",
                        dir=directory,
                    )
                )
                env = lmdb.open(
                    str(temporary),
                    map_size=max(int(map_size), _MIB),
                    subdir=True,
                    sync=True,
                    metasync=True,
                    max_readers=512,
                )
                counts_stream = (temporary / "counts.bin").open("wb")
            if buffer and buffered_bytes + len(value) > transaction_bytes:
                flush()
            key = f"{shard_length:09d}".encode("ascii")
            buffer.append((key, value, nodes, edges))
            buffered_bytes += len(value)
            shard_length += 1
            shard_total_bytes += len(value)
            shard_nodes += nodes
            shard_edges += edges
            if buffered_bytes >= transaction_bytes:
                flush()
        finish_shard()
        if not shard_index:
            raise ValueError(f"Cannot write an empty {stage} store")
    finally:
        if counts_stream is not None:
            counts_stream.close()
        if env is not None:
            env.close()
    dataset_class = GraphDatasetLMDB if kind == "graphs" else RecordDataset
    return dataset_class.from_directories(
        directories,
        stage,
        allow_building=True,
        cache_size=cache_size,
        max_open_shards=max_open_shards,
    )


def build_store(builder, shard_dirs, stage, *, kind, lock_timeout=86400, **kwargs):
    """Build once across jobs; waiters attach without consuming their iterators."""
    validate_stage(stage)
    try:
        lock_timeout = float(lock_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("lock_timeout must be a finite non-negative number") from exc
    if not math.isfinite(lock_timeout) or lock_timeout < 0:
        raise ValueError("lock_timeout must be a finite non-negative number")
    directories = [Path(directory).resolve() for directory in shard_dirs]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("shard_dirs must contain distinct directories")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    control = min(directories)
    owner_path = control / f"{stage}.build.lock"
    failure_path = control / f"{stage}_failed.json"
    # Keep the native lock file: unlinking it can create two independently locked
    # inodes while another process is waiting. The owner marker is only metadata.
    lock = FileLock(str(control / f".{stage}.writer.lock"))
    try:
        lock.acquire(timeout=lock_timeout, poll_interval=0.1)
    except Timeout as exc:
        try:
            owner = owner_path.read_text()
        except OSError:
            owner = "owner metadata unavailable"
        raise TimeoutError(
            f"Timed out after {lock_timeout}s waiting for {stage} cache writer in "
            f"{control} ({owner}). The waiting process did not modify the cache."
        ) from exc
    try:
        metadata = read_cache_metadata(directories, stage)
        dataset_class = GraphDatasetLMDB if kind == "graphs" else RecordDataset
        if metadata is not None:
            if metadata["kind"] != kind:
                raise ValueError(
                    f"Existing {stage} store contains {metadata.get('kind')}, not {kind}"
                )
            return dataset_class.from_directories(
                directories,
                stage,
                cache_size=kwargs.get("cache_size", 0),
                max_open_shards=kwargs.get("max_open_shards", DEFAULT_MAX_OPEN_SHARDS),
            )
        if failure_path.exists():
            try:
                failure = json.loads(failure_path.read_text())
                reason = failure.get("error", "unknown writer error")
            except (OSError, ValueError, AttributeError):
                reason = "unreadable writer failure metadata"
            raise RuntimeError(
                f"The previous {stage} cache writer failed: {reason}. "
                f"See {failure_path}. Partial files were preserved; use a new "
                "cache directory or explicitly move the failed build before retrying."
            )
        if owner_path.exists():
            raise RuntimeError(
                f"The previous {stage} writer was interrupted before publishing a "
                f"complete cache. See {owner_path}. Partial files were preserved; "
                "use a new cache directory or explicitly move the failed build."
            )
        owner_path.write_text(json.dumps({"pid": os.getpid(), "stage": stage}))
        remove_owner = False
        try:
            if any(
                any(directory.glob(f"{stage}_shard*.lmdb")) for directory in directories
            ):
                raise FileExistsError(
                    f"Existing {stage} shards must not be overwritten"
                )
            result = builder(directories, stage, kwargs)
            remove_owner = True
            return result
        except BaseException as exc:
            temporary = control / f"{stage}_failed.json.tmp-{os.getpid()}"
            try:
                temporary.write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                )
                os.replace(temporary, failure_path)
                remove_owner = True
            except OSError:
                pass
            raise
        finally:
            if remove_owner:
                owner_path.unlink(missing_ok=True)
    finally:
        lock.release()


def write_graphs(graphs, shard_dirs, stage, **kwargs):
    """Write losslessly compressed graphs with bounded, atomic shard publication."""
    return build_store(
        lambda directories, stage, options: write_store_locked(
            graphs, directories, stage, kind="graphs", **options
        ),
        shard_dirs,
        stage,
        kind="graphs",
        **kwargs,
    )


def write_records(records, shard_dirs, stage="source", **kwargs):
    """Compress complete raw namespaces for reuse with different graph requests."""
    return build_store(
        lambda directories, stage, options: write_store_locked(
            records, directories, stage, kind="records", **options
        ),
        shard_dirs,
        stage,
        kind="records",
        **kwargs,
    )


def build_graph_store(builder, shard_dirs, stage, **kwargs):
    """Build independent, ordered partitions without a parent shard manifest.

    ``builder(directories, build_stage, writer_options)`` writes partitions
    using stages ``f"{build_stage}_shard{part:06d}_part"`` in source order.
    A unique build prefix keeps that order when shards from different builds
    are collected into the same directory. Existing stores skip the builder.
    """

    def publish(directories, stage, options):
        completed = builder(directories, f"{stage}_shard{uuid4().hex}", options)
        if completed is not None:
            for _ in completed:
                pass
        return GraphDatasetLMDB.from_directories(
            directories,
            stage,
            allow_building=True,
            cache_size=options.get("cache_size", 0),
            max_open_shards=options.get("max_open_shards", DEFAULT_MAX_OPEN_SHARDS),
        )

    return build_store(publish, shard_dirs, stage, kind="graphs", **kwargs)
