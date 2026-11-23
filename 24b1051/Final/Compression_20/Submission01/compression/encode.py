from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import zlib
from pathlib import Path

import numpy as np

from decompression.container import MAGIC
from decompression.safetensors_io import SafeTensorReader, bf16_to_f32, f32_to_bf16


REFERENCE = Path(__file__).with_name("reference_header.json")


def _source_f32(reader: SafeTensorReader, name: str, start: int, end: int) -> np.ndarray:
    raw = reader.array(name).reshape(-1)[start:end]
    if reader.spec(name)["dtype"] == "BF16":
        return bf16_to_f32(np.asarray(raw, dtype="<u2"))
    return np.asarray(raw, dtype=np.float32)


def _quantized_groups(reader: SafeTensorReader, entry: dict, groups_per_chunk: int = 2048):
    group_size = int(entry["group_size"])
    numel = int(entry["numel"])
    padded = int(entry["padded_numel"])
    bits = int(entry["bits"])
    offset = int(entry["code_offset"])
    qmin, qmax = -offset, (1 << bits) - 1 - offset
    groups = padded // group_size
    for group0 in range(0, groups, groups_per_chunk):
        group1 = min(groups, group0 + groups_per_chunk)
        begin, finish = group0 * group_size, min(numel, group1 * group_size)
        values = np.zeros((group1 - group0) * group_size, dtype=np.float32)
        if finish > begin:
            values[: finish - begin] = _source_f32(reader, entry["name"], begin, finish)
        values = values.reshape(group1 - group0, group_size)
        absmax = np.max(np.abs(values), axis=1)
        best_error = np.full(values.shape[0], np.inf, dtype=np.float32)
        best_scale = np.ones(values.shape[0], dtype=np.float32)
        for fraction in np.linspace(0.80, 1.00, 100, dtype=np.float32):
            scale = np.maximum(absmax * fraction / max(qmax, 1), np.float32(1e-12))
            scale_bf16 = bf16_to_f32(f32_to_bf16(scale))
            quant = np.clip(np.rint(values / scale_bf16[:, None]), qmin, qmax)
            error = np.mean((values - quant * scale_bf16[:, None]) ** 2, axis=1)
            improve = error < best_error
            best_error[improve] = error[improve]
            best_scale[improve] = scale_bf16[improve]
        codes = np.clip(np.rint(values / best_scale[:, None]), qmin, qmax).astype(np.int16) + offset
        yield codes.astype(np.uint8), f32_to_bf16(best_scale)


def _qweight_chunks(reader: SafeTensorReader, entry: dict):
    bits = int(entry["bits"])
    shifts = np.arange(bits, dtype=np.uint8)
    for codes, _ in _quantized_groups(reader, entry):
        bit_matrix = ((codes.reshape(-1, 1) >> shifts) & 1).astype(np.uint8)
        yield np.packbits(bit_matrix.reshape(-1), bitorder="little").tobytes()


def _scale_chunks(reader: SafeTensorReader, entry: dict):
    for _, scale_words in _quantized_groups(reader, entry):
        yield scale_words.tobytes()


def _raw_chunks(reader: SafeTensorReader, entry: dict, chunk_elements: int = 4_000_000):
    raw = reader.array(entry["name"]).reshape(-1)
    for start in range(0, raw.size, chunk_elements):
        yield np.asarray(raw[start:start + chunk_elements]).tobytes()


def _write_zlib_block(handle, producer, *, role: str, dtype: str, shape: list[int]) -> dict:
    offset = handle.tell()
    compressor = zlib.compressobj(level=9)
    digest = hashlib.sha256()
    compressed_size = 0
    raw_size = 0
    for raw in producer:
        raw_size += len(raw)
        encoded = compressor.compress(raw)
        if encoded:
            handle.write(encoded)
            digest.update(encoded)
            compressed_size += len(encoded)
    encoded = compressor.flush()
    handle.write(encoded)
    digest.update(encoded)
    compressed_size += len(encoded)
    return {
        "codec": "zlib", "component_shape": shape, "dtype": dtype,
        "offset": offset, "raw_size": raw_size, "role": role,
        "sha256": digest.hexdigest(), "size": compressed_size,
    }


def compress(checkpoint_path: str | Path, output_path: str | Path) -> None:
    source = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if len(index["weight_map"]) != 738:
        raise ValueError(f"expected 738 source tensors, found {len(index['weight_map'])}")
    expected_names = {entry["name"] for entry in reference["tensors"]}
    if set(index["weight_map"]) != expected_names:
        missing = sorted(expected_names - set(index["weight_map"]))[:5]
        extra = sorted(set(index["weight_map"]) - expected_names)[:5]
        raise ValueError(f"source tensor census mismatch; missing={missing}, extra={extra}")
    output.mkdir(parents=True, exist_ok=True)
    metadata_hashes = {}
    for name in reference["metadata_files"]:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, output / name)
        metadata_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    readers = {shard: SafeTensorReader(source / shard) for shard in sorted(set(index["weight_map"].values()))}
    temporary = output / "model.czip.partial"
    exported = []
    text_bytes = 0
    visual_bytes = 0
    with temporary.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", 1))
        for policy in reference["tensors"]:
            entry = {key: value for key, value in policy.items() if key != "blocks"}
            name = entry["name"]
            shard = index["weight_map"][name]
            reader = readers[shard]
            spec = reader.spec(name)
            shape = [int(v) for v in spec["shape"]]
            if shape != [int(v) for v in entry["shape"]]:
                raise ValueError(f"shape mismatch for {name}")
            entry["original_shard"] = shard
            entry["original_dtype"] = spec["dtype"]
            entry["dtype"] = spec["dtype"]
            if entry["mode"] == "raw":
                block = _write_zlib_block(
                    handle, _raw_chunks(reader, entry), role="raw", dtype=spec["dtype"], shape=shape
                )
                entry["blocks"] = [block]
            elif entry["mode"] == "packed_groupwise":
                qbytes = (int(entry["padded_numel"]) * int(entry["bits"]) + 7) // 8
                groups = int(entry["padded_numel"]) // int(entry["group_size"])
                qblock = _write_zlib_block(
                    handle, _qweight_chunks(reader, entry), role="qweight", dtype="U8", shape=[qbytes]
                )
                sblock = _write_zlib_block(
                    handle, _scale_chunks(reader, entry), role="scales", dtype="BF16", shape=[groups]
                )
                entry["blocks"] = [qblock, sblock]
            else:
                raise ValueError(f"unsupported mode {entry['mode']}")
            component_bytes = sum(int(block["size"]) for block in entry["blocks"])
            if entry.get("visual"):
                visual_bytes += component_bytes
            else:
                text_bytes += component_bytes
            exported.append(entry)
        header = {
            "allocation_id": reference["allocation_id"],
            "architecture": reference["architecture"],
            "compressed_text_bytes": text_bytes,
            "compressed_total_bytes": text_bytes + visual_bytes,
            "compressed_visual_bytes": visual_bytes,
            "format": reference["format"],
            "metadata_files": reference["metadata_files"],
            "metadata_sha256": metadata_hashes,
            "model_name": reference["model_name"],
            "source_index_metadata": index.get("metadata", {}),
            "tensors": exported,
            "version": 1,
        }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_offset = handle.tell()
        handle.write(encoded)
        handle.write(struct.pack("<QQ", header_offset, len(encoded)))
        handle.write(MAGIC)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / "model.czip")
