#!/usr/bin/env python3
import argparse

from compression.encode import compress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen-3.5-4B")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    if args.model_name not in {"Qwen-3.5-4B", "Qwen/Qwen3.5-4B"}:
        raise ValueError("this submission supports Qwen-3.5-4B only")
    compress(args.checkpoint_path, args.output_path)


if __name__ == "__main__":
    main()
