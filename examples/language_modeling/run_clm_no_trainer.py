#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pretrain a causal language model from scratch, optionally quantized.

Adapted from Hugging Face's ``run_clm_no_trainer.py``, with the data loading
and optimizer setup of Karpathy's build-nanogpt.  The model is built from
``--config_name`` with random weights and trains on the shards
``pretokenize.py`` writes, read in order in fixed-size batches, so every run
sees the same tokens and a resumed run continues exactly.

The quantization flags pick the mode::

    flags                          forward GEMMs  backward GEMMs
    -----------------------------  -------------  --------------
    none                           unquantized    unquantized
    --weight --activation          quantized      unquantized
    --weight --activation --error  quantized      quantized

``lm_head`` and attention's QK^T and PV stay unquantized unless
``--quantize_lm_head`` / ``--quantize_attention``, as do the decoder layers
``--bf16_layers_at_start`` / ``--bf16_layers_at_end`` name.  The optimizer
updates FP32 master weights, from gradients accumulated in FP32 and clipped
to ``--max_grad_norm``; the model computes in bf16 with ``--bf16``.

Example::

    python examples/language_modeling/run_clm_no_trainer.py \\
        --data_dir ~/claude/datasets/fineweb-edu-10BT-llama2 \\
        --output_dir ~/claude/pretrain/mxint8 --max_train_steps 5000 --bf16 \\
        --activation int8,qs=microscaling,bs=32 \\
        --weight int8,qs=microscaling,bs=32 \\
        --error int8,qs=microscaling,bs=32
"""

import argparse
import json
import logging
import math
import os
import time

import numpy as np
import torch
import wandb
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    get_scheduler,
    set_seed,
)

from voyager_compiler import (
    add_experiment_args,
    get_default_quantizer,
    prepare_from_args,
    with_execution_context,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain a causal language model from scratch."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory of token shards written by pretokenize.py.",
    )
    parser.add_argument(
        "--config_name",
        default="amd/AMD-Llama-135m",
        help="Model config to build the model from, with random weights.",
    )
    parser.add_argument(
        "--config_overrides",
        default=None,
        help=(
            "Config fields to override, e.g. "
            '"num_hidden_layers=6,hidden_size=512".'
        ),
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument(
        "--block_size", type=int, default=1024, help="Tokens per sequence."
    )
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--lr_scheduler_type",
        default="cosine_with_min_lr",
        choices=["cosine_with_min_lr", "warmup_stable_decay"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=715,
        help="GPT-3's 375M warmup tokens at 0.5M tokens per step.",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="Final learning rate as a fraction of --learning_rate.",
    )
    parser.add_argument(
        "--decay_ratio",
        type=float,
        default=0.2,
        help="Fraction of the steps warmup_stable_decay decays over.",
    )
    parser.add_argument("--max_train_steps", type=int, required=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=20,
        help="Validation batches each evaluation averages the loss over.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help="Save a checkpoint every this many steps; always at the end.",
    )
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quantize_lm_head",
        action="store_true",
        help="Quantize lm_head like the other linears.",
    )
    parser.add_argument(
        "--quantize_attention",
        action="store_true",
        help="Quantize attention's QK^T and PV matmuls.",
    )
    parser.add_argument(
        "--bf16_layers_at_start",
        type=int,
        default=0,
        help="Number of first decoder layers left unquantized.",
    )
    parser.add_argument(
        "--bf16_layers_at_end",
        type=int,
        default=0,
        help="Number of last decoder layers left unquantized.",
    )
    add_experiment_args(parser)
    parser.set_defaults(log_level="INFO")
    return parser.parse_args()


class TokenStream:
    """Batches of token rows read in order from token shards.

    The shards are read one after another, and from the first again after
    the last; the tail of a shard too short for a whole batch is skipped.
    ``shard`` and ``offset`` are the read position.

    Args:
        files: Shard paths.
        dtype: Dtype the shards store token ids in.
        batch_size: Rows per batch.
        block_size: Tokens per row.
    """

    def __init__(self, files, dtype, batch_size, block_size):
        self.shards = [np.memmap(f, dtype=dtype, mode="r") for f in files]
        self.shape = (batch_size, block_size)
        self.shard = 0
        self.offset = 0

    def next_batch(self) -> torch.Tensor:
        size = self.shape[0] * self.shape[1]
        if self.offset + size > len(self.shards[self.shard]):
            self.shard = (self.shard + 1) % len(self.shards)
            self.offset = 0
        tokens = self.shards[self.shard][self.offset : self.offset + size]
        self.offset += size
        return torch.from_numpy(tokens.astype(np.int64)).view(self.shape)


def build_quantizer(args, num_layers):
    """The quantizer the flags build, with the unquantized GEMMs excluded.

    Args:
        args: Parsed command-line arguments.
        num_layers: Number of decoder layers in the model.

    Returns:
        The configured quantizer.
    """
    quantizer = get_default_quantizer(
        input_activation=args.activation,
        weight=args.weight,
        bias=args.bias,
        error=args.error,
        force_scale_power_of_two=args.force_scale_power_of_two,
    )
    if not args.quantize_lm_head:
        quantizer.set_module_name("lm_head", None)
    if not args.quantize_attention:
        quantizer.set_module_name_object_type(
            r"self_attn$", torch.ops.aten.matmul.default, None
        )
    end = num_layers - args.bf16_layers_at_end
    for layer in [*range(args.bf16_layers_at_start), *range(end, num_layers)]:
        quantizer.set_module_name(rf"layers\.{layer}$", None)
    return quantizer


def evaluate(model, stream, num_batches, device):
    """Mean loss of ``model`` over ``stream``'s next ``num_batches``."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            batch = stream.next_batch().to(device)
            total += model(input_ids=batch, labels=batch).loss.item()
    model.train()
    return total / num_batches


@with_execution_context
def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    files = [os.path.join(args.data_dir, s["file"]) for s in meta["shards"]]
    shape = (args.per_device_train_batch_size, args.block_size)
    # The first shard is the validation split.
    train_stream = TokenStream(files[1:], meta["dtype"], *shape)

    if torch.cuda.is_available():
        device = torch.device(
            "cuda" if args.gpu is None else f"cuda:{args.gpu}"
        )
    else:
        logger.warning("CUDA not available; training on the CPU.")
        device = torch.device("cpu")

    config = AutoConfig.from_pretrained(args.config_name)
    if args.config_overrides is not None:
        config.update_from_string(args.config_overrides)
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation="eager"
    )
    if args.bf16:
        model.bfloat16()
    model.to(device)
    logger.info(f"#Params: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Tokens: {meta['tokenizer']} ids from {args.data_dir}")

    if args.activation is not None or args.weight is not None:
        example = torch.zeros(shape, dtype=torch.long, device=device)
        model = prepare_from_args(
            model,
            args,
            (),
            {"input_ids": example, "labels": example},
            quantizer=build_quantizer(args, config.num_hidden_layers),
        )

    params = [p for p in model.parameters() if p.requires_grad]
    masters = [p.detach().to(torch.float32, copy=True) for p in params]
    for master in masters:
        master.grad = torch.zeros_like(master)
    # Weight decay applies to matrices, not to norm gains.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [m for m in masters if m.dim() >= 2],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [m for m in masters if m.dim() < 2],
                "weight_decay": 0.0,
            },
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    if args.lr_scheduler_type == "warmup_stable_decay":
        schedule_kwargs = {
            "num_decay_steps": round(args.decay_ratio * args.max_train_steps),
            "min_lr_ratio": args.min_lr_ratio,
        }
    else:
        schedule_kwargs = {"min_lr_rate": args.min_lr_ratio}
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
        scheduler_specific_kwargs=schedule_kwargs,
    )

    completed_steps = 0
    if args.resume_from_checkpoint is not None:
        state = torch.load(
            args.resume_from_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(state["model"])
        for master, saved in zip(masters, state["masters"]):
            master.copy_(saved)
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        train_stream.shard, train_stream.offset = state["stream"]
        completed_steps = state["step"]
        logger.info(f"Resumed from {args.resume_from_checkpoint}")

    tokens_per_step = math.prod(shape) * args.gradient_accumulation_steps
    logger.info(f"Tokens per step: {tokens_per_step}")
    model.train()
    total_loss = 0.0
    start = time.time()
    while completed_steps < args.max_train_steps:
        for _ in range(args.gradient_accumulation_steps):
            batch = train_stream.next_batch().to(device)
            loss = model(input_ids=batch, labels=batch).loss
            (loss / args.gradient_accumulation_steps).backward()
            total_loss += loss.detach().float()
            for param, master in zip(params, masters):
                master.grad += param.grad
                param.grad = None
        grad_norm = torch.nn.utils.clip_grad_norm_(masters, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=False)
        with torch.no_grad():
            for param, master in zip(params, masters):
                param.copy_(master)
        completed_steps += 1

        metrics = {}
        if completed_steps % args.logging_steps == 0:
            batches = args.logging_steps * args.gradient_accumulation_steps
            metrics["train_loss"] = total_loss.item() / batches
            metrics["lr"] = lr_scheduler.get_last_lr()[0]
            metrics["grad_norm"] = grad_norm.item()
            metrics["tokens_per_second"] = (
                args.logging_steps * tokens_per_step / (time.time() - start)
            )
            total_loss = 0.0
            start = time.time()
        last = completed_steps == args.max_train_steps
        if completed_steps % args.eval_steps == 0 or last:
            val_stream = TokenStream(files[:1], meta["dtype"], *shape)
            val_loss = evaluate(model, val_stream, args.eval_batches, device)
            metrics["val_loss"] = val_loss
            metrics["val_perplexity"] = math.exp(val_loss)
        if metrics:
            logger.info(f"step {completed_steps}: {metrics}")
            if wandb.run is not None:
                wandb.log(metrics, step=completed_steps)

        checkpoint = (
            args.checkpointing_steps is not None
            and completed_steps % args.checkpointing_steps == 0
        )
        if checkpoint or last:
            path = os.path.join(args.output_dir, f"step_{completed_steps}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "masters": masters,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "stream": (train_stream.shard, train_stream.offset),
                    "step": completed_steps,
                },
                path,
            )
            logger.info(f"Saved {path}")


if __name__ == "__main__":
    main(parse_args())
