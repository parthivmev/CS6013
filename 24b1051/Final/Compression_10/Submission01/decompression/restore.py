from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from .safetensors_io import SafeTensorReader, f32_to_bf16, write_streaming


def _copy_auxiliary(checkpoint: Path, output: Path, names: list[str]) -> None:
    for name in names:
        source = checkpoint / name
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint is missing required file {name}")
        shutil.copy2(source, output / name)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["torch_dtype"] = "bfloat16"
    config["dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _fp16_to_bf16_chunks(array: np.ndarray, chunk_elements: int = 4_000_000):
    flat = array.reshape(-1)
    for start in range(0, flat.size, chunk_elements):
        yield f32_to_bf16(np.asarray(flat[start : start + chunk_elements], dtype=np.float32)).tobytes()


def _sign_to_bf16_chunks(bits: np.ndarray, scales: np.ndarray, row_size: int, rows_per_chunk: int = 4096):
    packed = np.asarray(bits).reshape(-1)
    row_scales = np.asarray(scales, dtype=np.float16).reshape(-1)
    rows = row_scales.size
    for row0 in range(0, rows, rows_per_chunk):
        row1 = min(rows, row0 + rows_per_chunk)
        bit0, bit1 = row0 * row_size, row1 * row_size
        byte0, byte1 = bit0 // 8, (bit1 + 7) // 8
        unpacked = np.unpackbits(packed[byte0:byte1], bitorder="little")
        unpacked = unpacked[bit0 - byte0 * 8 : bit1 - byte0 * 8]
        signs = unpacked.astype(np.float32) * 2.0 - 1.0
        values = signs.reshape(row1 - row0, row_size) * row_scales[row0:row1, None].astype(np.float32)
        yield f32_to_bf16(values.reshape(-1)).tobytes()


def restore(checkpoint_path: str | Path, output_path: str | Path) -> None:
    checkpoint = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    meta_path = checkpoint / "compression_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != "cs6013-rowwise-sign" or meta.get("version") != 1:
        raise ValueError("unsupported compressed checkpoint format")
    tensors = meta["tensors"]
    if len(tensors) != 738:
        raise ValueError(f"expected 738 tensors, found {len(tensors)}")
    output.mkdir(parents=True, exist_ok=True)
    _copy_auxiliary(checkpoint, output, meta["auxiliary_files"])
    readers = {
        layout["compressed_shard"]: SafeTensorReader(checkpoint / layout["compressed_shard"])
        for layout in meta["shard_layout"]
    }
    weight_map: dict[str, str] = {}
    total_size = 0
    for layout in meta["shard_layout"]:
        compressed_name = layout["compressed_shard"]
        source_name = layout["source_shard"]
        reader = readers[compressed_name]
        specs = []
        producers = {}
        for tensor_name in layout["tensor_names"]:
            info = tensors[tensor_name]
            shape = [int(v) for v in info["shape"]]
            specs.append((tensor_name, "BF16", shape))
            total_size += int(info["numel"]) * 2
            weight_map[tensor_name] = source_name
            if info["encoding"] == "fp16":
                data = reader.array(info["data_key"])
                producers[tensor_name] = lambda data=data: _fp16_to_bf16_chunks(data)
            elif info["encoding"] == "rowwise_sign_1bit":
                bits = reader.array(info["packed_key"])
                scales = reader.array(info["scale_key"])
                row_size = int(info["row_size"])
                producers[tensor_name] = lambda bits=bits, scales=scales, row_size=row_size: _sign_to_bf16_chunks(bits, scales, row_size)
            else:
                raise ValueError(f"unsupported encoding for {tensor_name}: {info['encoding']}")
        write_streaming(output / source_name, specs, producers, layout.get("source_safetensors_metadata"))
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    temporary = output / "model.safetensors.index.json.partial"
    temporary.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "model.safetensors.index.json")
    (output / "DECOMPRESSION_COMPLETE.json").write_text(
        json.dumps({"format": meta["format"], "tensor_count": len(tensors), "dtype": "bfloat16"}, indent=2) + "\n",
        encoding="utf-8",
    )
