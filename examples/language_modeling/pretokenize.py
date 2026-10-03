"""Tokenize a text dataset into flat token shards for pretraining.

Documents are shuffled once with a fixed seed, tokenized, each followed by
the tokenizer's EOS token, and concatenated into one stream.  The stream is
cut into ``--shard_tokens``-token shards of raw token ids (``.bin``), uint16
when the vocabulary fits and uint32 otherwise; the first shard is the
validation split.  ``meta.json`` records how the shards were made.  The
dataset is streamed, so only the files the first ``--max_tokens`` tokens come
from are downloaded.

Example::

    python examples/language_modeling/pretokenize.py \\
        --tokenizer_name amd/AMD-Llama-135m \\
        --output_dir ~/claude/datasets/fineweb-edu-10BT-llama2 \\
        --max_tokens 3000000000
"""

import argparse
import json
import os
from itertools import chain

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize a text dataset into token shards."
    )
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config_name", default="sample-10BT")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--shard_tokens",
        type=int,
        default=100_000_000,
        help="Tokens per shard; the first shard is the validation split.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Stop after this many tokens; the whole dataset if unset.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the document shuffle."
    )
    parser.add_argument(
        "--num_threads", type=int, default=8, help="Tokenizer threads."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # The Rust tokenizer otherwise runs a thread on every core.
    os.environ["RAYON_NUM_THREADS"] = str(args.num_threads)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    eos = tokenizer.eos_token_id
    dtype = np.uint16 if len(tokenizer) <= 2**16 else np.uint32
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split="train",
        streaming=True,
    ).shuffle(seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    shards = []

    def write(tokens):
        split = "train" if shards else "val"
        name = f"{split}_{len(shards):06d}.bin"
        tokens.tofile(os.path.join(args.output_dir, name))
        shards.append({"file": name, "tokens": len(tokens)})
        print(f"wrote {name}: {len(tokens)} tokens", flush=True)

    shard = np.empty(args.shard_tokens, dtype=dtype)
    filled = 0
    total = 0
    for batch in dataset.iter(batch_size=1000):
        docs = tokenizer(batch[args.text_column], add_special_tokens=False)
        tokens = np.fromiter(
            chain.from_iterable(ids + [eos] for ids in docs["input_ids"]),
            dtype=dtype,
        )
        if args.max_tokens is not None:
            tokens = tokens[: args.max_tokens - total]
        total += len(tokens)
        while len(tokens) > 0:
            take = min(len(tokens), args.shard_tokens - filled)
            shard[filled : filled + take] = tokens[:take]
            filled += take
            tokens = tokens[take:]
            if filled == args.shard_tokens:
                write(shard)
                filled = 0
        if total == args.max_tokens:
            break
    if filled > 0:
        write(shard[:filled])

    meta = {
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config_name,
        "seed": args.seed,
        "tokenizer": args.tokenizer_name,
        "vocab_size": len(tokenizer),
        "eos_token_id": eos,
        "dtype": np.dtype(dtype).name,
        "shards": shards,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
    # pyarrow's IO threads may still be reading ahead in the stream through
    # Python, which crashes the interpreter's teardown; exit without it.
    os._exit(0)
