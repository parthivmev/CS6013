# CS6013 Final Compression 20 Submission01

Public compressed checkpoint: <https://huggingface.co/ParthivMEV/24b1051-Final-Compression20-Submission01>

## Setup

Use Python 3.10 or newer. The only dependency is NumPy and the lock is CUDA-independent:

```bash
uv sync
```

## Compress

The input directory must be the complete local `Qwen/Qwen3.5-4B` checkpoint. `--model_name` is only a validated name and is never used to download weights.

```bash
uv run python compress.py --model_name Qwen-3.5-4B --checkpoint_path /path/to/dense --output_path /path/to/compressed
```

Actual dtype path: BF16/FP32 base weights → packed U8/BF16 and lossless compressed components → BF16 reconstructed weights.

## Decompress

Decompression is completely offline and reads only the downloaded compressed checkpoint:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python decompress.py --model_name Qwen-3.5-4B --checkpoint_path /path/to/compressed --output_path /path/to/restored
```

The decoder reconstructs all tensors as BF16 safetensors in the original two-shard Hugging Face layout and writes the index plus required tokenizer, processor, template, and config files.

Block integrity uses `sha256_domain` (`raw` or `compressed`) at block level,
falling back to the container-level field when present. Untagged v1 blocks
support exactly the two verified historical conventions: the published checkpoint
stores SHA256 of decoded bytes, while the original submitted writer stored SHA256
of compressed bytes. Either convention still requires an exact digest match, the
declared decoded length, and a complete single zlib stream without trailing data.
Newly written blocks explicitly declare `sha256_domain: compressed`.

Run integrity and failure-cleanup regressions in the submission environment:

```bash
uv run python -m unittest discover -s tests -v
```
