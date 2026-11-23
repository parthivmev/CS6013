"""Small streaming safetensors reader/writer used by the offline codec."""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np


NP_DTYPES = {
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "I16": np.dtype("<i2"),
    "U16": np.dtype("<u2"),
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "I64": np.dtype("<i8"),
    "U64": np.dtype("<u8"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "BOOL": np.dtype("?"),
    # NumPy has no portable native bfloat16. Keep it as raw uint16 words.
    "BF16": np.dtype("<u2"),
}


def bf16_to_f32(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    words = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> 16) & 1)
    return (rounded >> 16).astype("<u2")


class SafeTensorReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated safetensors file: {self.path}")
            self.header_length = struct.unpack("<Q", raw)[0]
            self.header = json.loads(handle.read(self.header_length))
        self.data_start = 8 + self.header_length

    def keys(self) -> list[str]:
        return [key for key in self.header if key != "__metadata__"]

    def spec(self, key: str) -> dict:
        return self.header[key]

    def array(self, key: str) -> np.memmap:
        spec = self.spec(key)
        dtype_name = spec["dtype"]
        if dtype_name not in NP_DTYPES:
            raise ValueError(f"unsupported safetensors dtype {dtype_name!r}")
        start, end = map(int, spec["data_offsets"])
        dtype = NP_DTYPES[dtype_name]
        expected = int(np.prod(spec["shape"], dtype=np.int64)) * dtype.itemsize
        if end - start != expected:
            raise ValueError(f"invalid byte length for {key}")
        return np.memmap(
            self.path,
            mode="r",
            dtype=dtype,
            offset=self.data_start + start,
            shape=tuple(spec["shape"]),
            order="C",
        )


def write_streaming(
    path: str | Path,
    specs: Iterable[tuple[str, str, list[int]]],
    producers: dict[str, Callable[[], Iterable[bytes]]],
    metadata: dict[str, str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {}
    offset = 0
    normalized = []
    for name, dtype_name, shape in specs:
        if dtype_name not in NP_DTYPES:
            raise ValueError(f"unsupported output dtype {dtype_name!r}")
        size = int(np.prod(shape, dtype=np.int64)) * NP_DTYPES[dtype_name].itemsize
        header[name] = {"dtype": dtype_name, "shape": shape, "data_offsets": [offset, offset + size]}
        normalized.append((name, size))
        offset += size
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            for name, expected in normalized:
                written = 0
                for chunk in producers[name]():
                    handle.write(chunk)
                    written += len(chunk)
                if written != expected:
                    raise ValueError(f"producer for {name} wrote {written} bytes, expected {expected}")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
