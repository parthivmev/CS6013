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
            if (handle.read(8) != MAGIC or header_offset < 12 or header_length == 0
                    or header_offset + header_length > size - 24):
                raise ValueError("invalid CS6013CZ footer/header bounds")
            handle.seek(header_offset)
            self.header = json.loads(handle.read(header_length))
        if self.header.get("format") != "cs6013-czip-packed-v1":
            raise ValueError("unsupported CS6013CZ format")
        names = [item.get("name") for item in self.header.get("tensors", [])]
        if len(names) != 738 or len(set(names)) != len(names):
            raise ValueError("container must contain 738 unique tensors")
        intervals = []
        self._labels = {}
        for item in self.header["tensors"]:
            for block in item.get("blocks", []):
                start, length = int(block["offset"]), int(block["size"])
                if start < 12 or length <= 0 or start + length > header_offset:
                    raise ValueError(f"invalid block bounds for {item['name']}")
                self._labels[(start, length)] = f"tensor {item['name']}, block {block.get('role')}, offset {start}"
                intervals.append((start, start + length))
        intervals.sort()
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError("overlapping compressed blocks")

    def extract(self, block: dict, directory: Path) -> Path:
        start, size = int(block["offset"]), int(block["size"])
        label = self._labels.get((start, size))
        if label is None:
            raise ValueError(f"block is outside validated container bounds: offset {start}, size {size}")
        domain = block.get("sha256_domain", self.header.get("sha256_domain"))
        if domain not in (None, "raw", "compressed"):
            raise ValueError(f"{label}: unsupported SHA256 domain {domain!r}")
        if block.get("codec") != "zlib":
            raise ValueError(f"{label}: unsupported codec {block.get('codec')!r}")
        expected_size = int(block["raw_size"])
        if expected_size < 0:
            raise ValueError(f"{label}: invalid raw size {expected_size}")
        fd, name = tempfile.mkstemp(prefix="czip-", suffix=".raw", dir=directory)
        output = Path(name)
        compressed_digest = hashlib.sha256()
        raw_digest = hashlib.sha256()
        decompressor = zlib.decompressobj()
        written = 0
        try:
            with os.fdopen(fd, "wb") as target, self.path.open("rb") as source:
                source.seek(start)
                remaining = size
                while remaining:
                    chunk = source.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"{label}: truncated compressed block")
                    remaining -= len(chunk)
                    compressed_digest.update(chunk)
                    try:
                        decoded = decompressor.decompress(chunk)
                    except zlib.error as error:
                        raise ValueError(f"{label}: corrupt DEFLATE block: {error}") from error
                    raw_digest.update(decoded)
                    target.write(decoded)
                    written += len(decoded)
                    if written > expected_size:
                        raise ValueError(f"{label}: decompressed block length exceeds expected {expected_size}: {written}")
                    if decompressor.unused_data or decompressor.unconsumed_tail:
                        raise ValueError(f"{label}: unexpected trailing compressed data")
                decoded = decompressor.flush()
                raw_digest.update(decoded)
                target.write(decoded)
                written += len(decoded)
                target.flush()
                os.fsync(target.fileno())
            if not decompressor.eof:
                raise ValueError(f"{label}: incomplete zlib stream")
            if written != expected_size:
                raise ValueError(f"{label}: decompressed block length mismatch: expected {expected_size}, observed {written}")
            observed = {"compressed": compressed_digest.hexdigest(), "raw": raw_digest.hexdigest()}
            # The published v1 checkpoint stores raw-byte hashes; the v1 writer
            # in this submission stored compressed-byte hashes without a tag.
            # Untagged legacy blocks accept only these two verified conventions.
            allowed = (domain,) if domain is not None else ("raw", "compressed")
            if not any(observed[key] == block["sha256"] for key in allowed):
                raise ValueError(
                    f"{label}: SHA256 mismatch (domain={domain or 'legacy raw/compressed'}): "
                    f"expected {block['sha256']}, compressed {observed['compressed']}, raw {observed['raw']}"
                )
            return output
        except BaseException:
            output.unlink(missing_ok=True)
            raise
