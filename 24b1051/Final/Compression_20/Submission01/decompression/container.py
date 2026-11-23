from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path


MAGIC = b"CS6013CZ"


class Container:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        size = self.path.stat().st_size
        if size < 36:
            raise ValueError("container is truncated")
        with self.path.open("rb") as handle:
            if handle.read(8) != MAGIC or struct.unpack("<I", handle.read(4))[0] != 1:
                raise ValueError("invalid CS6013CZ prefix/version")
            handle.seek(-24, os.SEEK_END)
            header_offset, header_length = struct.unpack("<QQ", handle.read(16))
            if handle.read(8) != MAGIC or header_offset + header_length > size - 24:
                raise ValueError("invalid CS6013CZ footer/header bounds")
            handle.seek(header_offset)
            self.header = json.loads(handle.read(header_length))
        if self.header.get("format") != "cs6013-czip-packed-v1":
            raise ValueError("unsupported CS6013CZ format")
        names = [item.get("name") for item in self.header.get("tensors", [])]
        if len(names) != 738 or len(set(names)) != len(names):
            raise ValueError("container must contain 738 unique tensors")
        intervals = []
        for item in self.header["tensors"]:
            for block in item.get("blocks", []):
                start, length = int(block["offset"]), int(block["size"])
                if start < 12 or length < 0 or start + length > header_offset:
                    raise ValueError(f"invalid block bounds for {item['name']}")
                intervals.append((start, start + length))
        intervals.sort()
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError("overlapping compressed blocks")

    def extract(self, block: dict, directory: Path) -> Path:
        fd, name = tempfile.mkstemp(prefix="czip-", suffix=".raw", dir=directory)
        output = Path(name)
        digest = hashlib.sha256()
        decompressor = zlib.decompressobj()
        written = 0
        try:
            with os.fdopen(fd, "wb") as target, self.path.open("rb") as source:
                source.seek(int(block["offset"]))
                remaining = int(block["size"])
                while remaining:
                    chunk = source.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("truncated compressed block")
                    remaining -= len(chunk)
                    digest.update(chunk)
                    try:
                        decoded = decompressor.decompress(chunk)
                    except zlib.error as error:
                        raise ValueError("corrupt DEFLATE block") from error
                    target.write(decoded)
                    written += len(decoded)
                decoded = decompressor.flush()
                target.write(decoded)
                written += len(decoded)
                target.flush()
                os.fsync(target.fileno())
            if digest.hexdigest() != block["sha256"]:
                raise ValueError("compressed block SHA256 mismatch")
            if written != int(block["raw_size"]):
                raise ValueError("decompressed block length mismatch")
            return output
        except BaseException:
            output.unlink(missing_ok=True)
            raise
