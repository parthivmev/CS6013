from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from decompression.safetensors_io import SafeTensorReader, bf16_to_f32, write_streaming


AUXILIARY = [
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


def _as_f32(reader: SafeTensorReader, key: str) -> np.ndarray:
    spec = reader.spec(key)
    value = reader.array(key)
    if spec["dtype"] == "BF16":
        return bf16_to_f32(np.asarray(value, dtype="<u2"))
    return np.asarray(value, dtype=np.float32)


def _pack_sign_chunks(reader: SafeTensorReader, key: str, row_size: int, rows_per_chunk: int = 4096):
    value = reader.array(key).reshape(-1, row_size)
    for row0 in range(0, value.shape[0], rows_per_chunk):
        rows = _as_slice_f32(reader, key, row0 * row_size, min(value.size, (row0 + rows_per_chunk) * row_size))
        yield np.packbits(rows >= 0, bitorder="little").tobytes()


def _scale_chunks(reader: SafeTensorReader, key: str, row_size: int, rows_per_chunk: int = 4096):
    numel = int(np.prod(reader.spec(key)["shape"], dtype=np.int64))
    rows = numel // row_size
    for row0 in range(0, rows, rows_per_chunk):
        row1 = min(rows, row0 + rows_per_chunk)
        values = _as_slice_f32(reader, key, row0 * row_size, row1 * row_size).reshape(row1 - row0, row_size)
        yield np.mean(np.abs(values), axis=1, dtype=np.float32).astype("<f2").tobytes()


def _as_slice_f32(reader: SafeTensorReader, key: str, start: int, end: int) -> np.ndarray:
    spec = reader.spec(key)
    raw = reader.array(key).reshape(-1)[start:end]
    if spec["dtype"] == "BF16":
        return bf16_to_f32(np.asarray(raw, dtype="<u2"))
    return np.asarray(raw, dtype=np.float32)


def _fp16_chunks(reader: SafeTensorReader, key: str, chunk_elements: int = 4_000_000):
    numel = int(np.prod(reader.spec(key)["shape"], dtype=np.int64))
    for start in range(0, numel, chunk_elements):
        yield _as_slice_f32(reader, key, start, min(numel, start + chunk_elements)).astype("<f2").tobytes()


def compress(checkpoint_path: str | Path, output_path: str | Path) -> None:
    source = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    source_index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = source_index["weight_map"]
    shards = sorted(set(weight_map.values()))
    if len(weight_map) != 738:
        raise ValueError(f"expected 738 source tensors, found {len(weight_map)}")
    output.mkdir(parents=True, exist_ok=True)
    for name in AUXILIARY:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
        shutil.copy2(source / name, output / name)
    tensor_meta: dict[str, dict] = {}
    layouts = []
    for shard_number, source_shard in enumerate(shards, start=1):
        compressed_shard = f"compressed-{shard_number:05d}-of-{len(shards):05d}.safetensors"
        reader = SafeTensorReader(source / source_shard)
        names = [name for name, shard in weight_map.items() if shard == source_shard]
        specs = []
        producers = {}
        sequence = 0
        for name in names:
            source_spec = reader.spec(name)
            shape = [int(v) for v in source_spec["shape"]]
            numel = int(np.prod(shape, dtype=np.int64))
            prefix = f"t{sequence:06d}"
            if len(shape) >= 2:
                row_size = shape[-1]
                if numel % row_size:
                    raise ValueError(f"non-integral rows for {name}")
                packed_size = (numel + 7) // 8
                rows = numel // row_size
                specs.extend([(prefix + ".bits", "U8", [packed_size]), (prefix + ".scale", "F16", [rows])])
                producers[prefix + ".bits"] = lambda reader=reader, name=name, row_size=row_size: _pack_sign_chunks(reader, name, row_size)
                producers[prefix + ".scale"] = lambda reader=reader, name=name, row_size=row_size: _scale_chunks(reader, name, row_size)
                tensor_meta[name] = {
                    "compressed_shard": compressed_shard, "encoding": "rowwise_sign_1bit", "numel": numel,
                    "packed_key": prefix + ".bits", "row_size": row_size, "scale_key": prefix + ".scale",
                    "shape": shape, "source_dtype": source_spec["dtype"].lower(), "source_shard": source_shard,
                }
            else:
                specs.append((prefix + ".data", "F16", shape))
                producers[prefix + ".data"] = lambda reader=reader, name=name: _fp16_chunks(reader, name)
                tensor_meta[name] = {
                    "compressed_shard": compressed_shard, "data_key": prefix + ".data", "encoding": "fp16",
                    "numel": numel, "shape": shape, "source_dtype": source_spec["dtype"].lower(), "source_shard": source_shard,
                }
            sequence += 1
        write_streaming(output / compressed_shard, specs, producers, reader.header.get("__metadata__", {}))
        layouts.append({
            "compressed_shard": compressed_shard,
            "source_safetensors_metadata": reader.header.get("__metadata__", {}),
            "source_shard": source_shard,
            "tensor_names": names,
        })
    meta = {
        "auxiliary_files": AUXILIARY, "bit_width": 1, "bitorder": "little",
        "decompressed_floating_dtype": "bfloat16", "format": "cs6013-rowwise-sign",
        "model_name_argument": "Qwen-3.5-4B", "original_index": source_index,
        "original_model_id": "Qwen/Qwen3.5-4B", "original_shards": shards,
        "quantization": "per-output-row mean-absolute scale and packed sign",
        "scale_dtype": "float16", "shard_layout": layouts, "tensors": tensor_meta, "version": 1,
    }
    temporary = output / "compression_meta.json.partial"
    temporary.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "compression_meta.json")
