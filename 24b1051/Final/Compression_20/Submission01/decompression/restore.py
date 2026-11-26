from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from .container import Container
from .safetensors_io import bf16_to_f32, f32_to_bf16, write_streaming


def _copy_metadata(container: Container, checkpoint: Path, output: Path) -> None:
    expected_hashes = container.header.get("metadata_sha256", {})
    import hashlib
    for name in container.header["metadata_files"]:
        source = checkpoint / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if name in expected_hashes and hashlib.sha256(source.read_bytes()).hexdigest() != expected_hashes[name]:
            raise ValueError(f"metadata hash mismatch: {name}")
        shutil.copy2(source, output / name)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["torch_dtype"] = "bfloat16"
    config["dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _raw_producer(container: Container, entry: dict, temporary: Path):
    block = entry["blocks"][0]
    path = container.extract(block, temporary)
    try:
        dtype = block["dtype"]
        if dtype == "BF16":
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    yield chunk
        elif dtype == "F16":
            source = np.memmap(path, mode="r", dtype="<f2")
            for start in range(0, source.size, 4_000_000):
                yield f32_to_bf16(np.asarray(source[start:start + 4_000_000], dtype=np.float32)).tobytes()
        elif dtype == "F32":
            source = np.memmap(path, mode="r", dtype="<f4")
            for start in range(0, source.size, 4_000_000):
                yield f32_to_bf16(np.asarray(source[start:start + 4_000_000], dtype=np.float32)).tobytes()
        else:
            raise ValueError(f"unsupported raw dtype {dtype}")
    finally:
        path.unlink(missing_ok=True)


def _packed_producer(container: Container, entry: dict, temporary: Path):
    blocks = {block["role"]: block for block in entry["blocks"]}
    qpath = container.extract(blocks["qweight"], temporary)
    spath = None
    try:
        spath = container.extract(blocks["scales"], temporary)
        packed = np.memmap(qpath, mode="r", dtype="u1")
        scale_words = np.memmap(spath, mode="r", dtype="<u2")
        scales = bf16_to_f32(np.asarray(scale_words))
        bits = int(entry["bits"])
        group_size = int(entry["group_size"])
        numel = int(entry["numel"])
        padded = int(entry["padded_numel"])
        groups = padded // group_size
        if scales.size != groups:
            raise ValueError(f"scale count mismatch for {entry['name']}")
        powers = (np.uint16(1) << np.arange(bits, dtype=np.uint16))
        emitted = 0
        groups_per_chunk = max(1, 4_000_000 // group_size)
        for group0 in range(0, groups, groups_per_chunk):
            group1 = min(groups, group0 + groups_per_chunk)
            code0, code1 = group0 * group_size, group1 * group_size
            bit0, bit1 = code0 * bits, code1 * bits
            byte0, byte1 = bit0 // 8, (bit1 + 7) // 8
            unpacked = np.unpackbits(np.asarray(packed[byte0:byte1]), bitorder="little")
            unpacked = unpacked[bit0 - byte0 * 8 : bit1 - byte0 * 8]
            codes = (unpacked.reshape(-1, bits).astype(np.uint16) * powers).sum(axis=1, dtype=np.uint16)
            values = (codes.reshape(group1 - group0, group_size).astype(np.float32) - float(entry["code_offset"]))
            values *= scales[group0:group1, None]
            flat = values.reshape(-1)
            keep = min(flat.size, numel - emitted)
            if keep > 0:
                yield f32_to_bf16(flat[:keep]).tobytes()
                emitted += keep
        if emitted != numel:
            raise ValueError(f"decoded element count mismatch for {entry['name']}")
    finally:
        qpath.unlink(missing_ok=True)
        if spath is not None:
            spath.unlink(missing_ok=True)


def restore(checkpoint_path: str | Path, output_path: str | Path) -> None:
    checkpoint = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    container = Container(checkpoint / "model.czip")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".decompress_tmp"
    temporary.mkdir(exist_ok=True)
    tensors = container.header["tensors"]
    shards = sorted({entry["original_shard"] for entry in tensors})
    weight_map = {}
    total_size = 0
    created = []
    partial = output / "model.safetensors.index.json.partial"
    complete_partial = output / "DECOMPRESSION_COMPLETE.json.partial"
    try:
        _copy_metadata(container, checkpoint, output)
        for shard in shards:
            entries = [entry for entry in tensors if entry["original_shard"] == shard]
            specs = [(entry["name"], "BF16", [int(v) for v in entry["shape"]]) for entry in entries]
            producers = {}
            for entry in entries:
                name = entry["name"]
                weight_map[name] = shard
                total_size += int(entry["numel"]) * 2
                if entry["mode"] == "raw":
                    producers[name] = lambda entry=entry: _raw_producer(container, entry, temporary)
                elif entry["mode"] == "packed_groupwise":
                    producers[name] = lambda entry=entry: _packed_producer(container, entry, temporary)
                else:
                    raise ValueError(f"unsupported tensor mode {entry['mode']}")
            write_streaming(output / shard, specs, producers)
            created.append(output / shard)
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        partial.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, output / "model.safetensors.index.json")
        created.append(output / "model.safetensors.index.json")
        complete_partial.write_text(
            json.dumps({"format": container.header["format"], "tensor_count": len(tensors), "dtype": "bfloat16"}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(complete_partial, output / "DECOMPRESSION_COMPLETE.json")
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        partial.unlink(missing_ok=True)
        complete_partial.unlink(missing_ok=True)
        shutil.rmtree(temporary, ignore_errors=True)
