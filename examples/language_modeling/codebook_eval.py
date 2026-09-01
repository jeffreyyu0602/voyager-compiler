"""Fit codebooks, GPTQ-compensate, and score, driven by the graph.

One process over one prepared graph. It scores wikitext-2 perplexity at
the end unless told not to; what happens to the tables first is opt-in:

    --fit         fit the model's own codebooks (``fit_codebooks``)
    --dump PATH   write the fitted tables to JSON
    --codebooks PATH   install tables from JSON instead of fitting
    --gptq        compensate the weights against the deployed grid (``gptq``)
    --no_eval     stop after that, without scoring

Both the fit and the GPTQ Hessian read every tensor off the fake-quant the
quantizer attached, so ``--config`` picks the *structure* to work in, not
the tables. The fit takes the first N windows by default (which reproduces
the committed table dumps); ``--spread`` diversifies a small budget by
drawing them across the whole corpus. The GPTQ Hessian is contiguous.

Examples:
    # fit + compensate + score in one process (loss-aware, per-head Q/K/V)
    python codebook_eval.py --gpu 1 --config mxnf4 --fit \\
        --weighting fisher_activations \\
        --granularity '|q:-1,1,1' '|k:-1,1,1' --gptq --shrink \\
        --fit_windows 128 --gptq_windows 480 --c4_docs 4000

    # fit and dump only
    python codebook_eval.py --gpu 1 --config mxcb --fit --no_eval \\
        --dump codebooks/fitted.json

    # install dumped tables, compensate, and score
    python codebook_eval.py --gpu 1 --config mxnf4 --codebooks \\
        codebooks/fitted.json --gptq --shrink --gptq_windows 480
"""

import argparse
import logging

import torch
from datasets import load_dataset
from quantization_configs import QUANTIZATION_CONFIGS, set_qconfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from wikitext import evaluate_perplexity

from voyager_compiler import (
    get_default_quantizer,
    prepare_pt2e,
    sink_obs_or_fq,
)
from voyager_compiler.quantization import (
    CACHE_RESERVE,
    Weighting,
    fit_codebooks,
    gptq,
    load_codebooks,
)

logger = logging.getLogger(__name__)

WINDOW = 1024
BF16_PERPLEXITY = 5.924286365509033


def calibration_text(corpus, docs):
    """Return the text the fit and the Hessian are accumulated over.

    Args:
        corpus: ``wikitext`` for wikitext-2 validation -- never the test
            split the model is scored on -- or ``c4`` for a stream of C4
            English validation documents.
        docs: How many C4 documents to take; ignored for wikitext.

    Returns:
        The documents joined into one string.

    Raises:
        SystemExit: C4 did not stream, which would pull whole shards onto a
            shared filesystem.
    """
    if corpus == "wikitext":
        return "\n\n".join(
            load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")[
                "text"
            ]
        )
    stream = load_dataset(
        "allenai/c4", "en", split="validation", streaming=True
    )
    if not hasattr(stream, "_ex_iterable"):
        raise SystemExit("C4 did not stream - refusing to download shards")
    collected = []
    for example in stream:
        collected.append(example["text"])
        if len(collected) >= docs:
            break
    return "\n\n".join(collected)


def parse_args(parser=None):
    """Parse this script's command line, or a driver's superset of it.

    Args:
        parser: Parser to add the fit/GPTQ/eval arguments to, already
            carrying a driver's own. A fresh one when None.

    Returns:
        The parsed arguments.
    """
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument(
        "--config",
        default="mxnf4",
        help=(
            "Scheme giving the block size, contraction axis and range to fit "
            "into and compensate against. Its tables are only a seed."
        ),
    )
    parser.add_argument(
        "--calib_corpus",
        choices=["wikitext", "c4"],
        default="c4",
        help="Corpus the fit and the Hessian are sampled from.",
    )
    parser.add_argument(
        "--c4_docs",
        type=int,
        default=4000,
        help="C4 validation documents to stream for --calib_corpus c4.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Substrings of tensor names to leave unfitted/uncompensated.",
    )

    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit the model's own codebooks before compensating.",
    )
    parser.add_argument(
        "--fit_windows",
        type=int,
        default=128,
        help=(
            f"Fit calibration windows of {WINDOW} tokens. The first N by "
            "default; a seeded spread across the corpus under --spread."
        ),
    )
    parser.add_argument(
        "--spread",
        action="store_true",
        help=(
            "Draw the fit windows as a seeded spread across the whole corpus "
            "instead of its first N. Diversifies a small fit budget; the "
            "first-N default reproduces the committed table dumps exactly."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for --spread's draw of fit windows across the corpus.",
    )
    parser.add_argument(
        "--weighting",
        choices=[choice.value for choice in Weighting],
        default=Weighting.PARTNER.value,
        help=(
            "Which sensitivity weights each sample: the partner's channel "
            "energy, the operand's own loss-gradient Fisher diagonal on the "
            "activations, that diagonal extended to the weights as well "
            "(measured worse), or nothing at all."
        ),
    )
    parser.add_argument(
        "--granularity",
        nargs="*",
        default=[],
        metavar="PATTERN:COUNTS",
        help=(
            "How many codebooks a tensor is fitted along each axis, as "
            "'<name substring>:<comma-separated counts>'. The counts align "
            "to the trailing axes the way Tensor.expand's do: -1 gives one "
            "table per item of that axis, n splits it into n, 1 shares one "
            'across it. "|q:-1,1,1" is one table per attention head; '
            '"up_proj|wgt:28,16" a grid over a weight\'s last two axes.'
        ),
    )
    parser.add_argument(
        "--weight_by",
        nargs="*",
        default=[],
        help=(
            "Substrings picking which partner weights a shared tensor's fit. "
            "Ignored under --fisher."
        ),
    )
    parser.add_argument(
        "--scale_weighted",
        action="store_true",
        help=(
            "Also weight each sample by its block's squared scale, so the "
            "fit minimizes the error deployment makes rather than the "
            "error in normalized units."
        ),
    )
    parser.add_argument(
        "--quantized_scale",
        action="store_true",
        help=(
            "Bin each value divided by the block scale the decoder can "
            "store -- pushed through the fp8 scale codebook -- rather "
            "than the exact amax/quant_max it never sees."
        ),
    )
    parser.add_argument(
        "--dump",
        default=None,
        help="Where to write the fitted tables as JSON. Requires --fit.",
    )
    parser.add_argument(
        "--codebooks",
        default=None,
        help=(
            "JSON of fitted tables to install before compensating, as --dump "
            "writes. Fit them under the same --config. Excludes --fit."
        ),
    )

    parser.add_argument(
        "--gptq",
        action="store_true",
        help="Compensate the quantized weights with GPTQ.",
    )
    parser.add_argument(
        "--gptq_windows",
        type=int,
        default=480,
        help=(
            f"Contiguous GPTQ Hessian windows of {WINDOW} tokens. Give it "
            "more tokens than the largest input dimension."
        ),
    )
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument(
        "--cache_reserve",
        type=float,
        default=CACHE_RESERVE / 2**30,
        help=(
            "GiB of device memory held back from the activation cache. Raise "
            "it when sharing the GPU: the cache spills to host rather than "
            "failing."
        ),
    )
    parser.add_argument(
        "--shrink",
        action="store_true",
        help=(
            "Search each block's scale below amax/quant_max, clipping the "
            "largest value to buy resolution for the other 63."
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Let every weight see its predecessors' compensation, instead of "
            "holding a stretch's weights back until the walk leaves it."
        ),
    )
    parser.add_argument(
        "--no_within_order",
        action="store_true",
        help="Leave columns inside a block in index order, not by salience.",
    )
    parser.add_argument(
        "--no_act_order",
        action="store_true",
        help="Quantize blocks left to right, not most-salient-first.",
    )

    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Stop after the pipeline instead of scoring wikitext-2.",
    )

    args = parser.parse_args()
    if args.dump and not args.fit:
        parser.error("--dump writes the fitted tables, so it needs --fit")
    if args.codebooks and args.fit:
        parser.error(
            "--codebooks and --fit are two sources of tables; pick one"
        )
    return args


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device(f"cuda:{args.gpu}")
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokens = tokenizer(
        calibration_text(args.calib_corpus, args.c4_docs), return_tensors="pt"
    ).input_ids
    total_windows = tokens.shape[1] // WINDOW
    if args.gptq and total_windows < args.gptq_windows:
        raise SystemExit(
            f"{args.calib_corpus} gave {total_windows} windows, short of the "
            f"{args.gptq_windows} that GPTQ needs"
        )

    def pick(indices):
        return [
            (
                tokens[:, i * WINDOW : (i + 1) * WINDOW].to(device),
                tokens[:, i * WINDOW : (i + 1) * WINDOW].to(device),
                False,
            )
            for i in indices
        ]

    def fit_windows():
        count = min(args.fit_windows, total_windows)
        if not args.spread:
            logger.info(
                "fit on the first %d of %d windows", count, total_windows
            )
            return pick(range(count))
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(total_windows, generator=generator)
        chosen = sorted(order[:count].tolist())
        logger.info(
            "fit on %d of %d windows (seeded spread, seed %d)",
            len(chosen),
            total_windows,
            args.seed,
        )
        return pick(chosen)

    quantizer = get_default_quantizer()
    quantizer.set_module_name("model.rotary_emb", None)
    set_qconfig(quantizer, QUANTIZATION_CONFIGS[args.config], False)
    example = torch.randint(
        0, model.config.vocab_size, (1, WINDOW), device=device
    )
    chunk = torch.export.Dim("chunk_dim", min=2, max=16)
    with torch.no_grad():
        graph = prepare_pt2e(
            model,
            quantizer,
            (example,),
            {"labels": example.clone(), "use_cache": False},
            {
                "input_ids": {1: chunk * 64},
                "labels": {1: chunk * 64},
                "use_cache": None,
            },
        )
    sink_obs_or_fq(graph)

    if args.fit:
        fit_codebooks(
            graph,
            fit_windows(),
            skip=args.skip,
            weighting=Weighting(args.weighting),
            weight_by=args.weight_by,
            granularity={
                pattern: tuple(int(part) for part in counts.split(","))
                for pattern, _, counts in (
                    pick.partition(":") for pick in args.granularity
                )
            },
            scale_weighted=args.scale_weighted,
            quantized_scale=args.quantized_scale,
            dump=args.dump,
        )
    elif args.codebooks is not None:
        load_codebooks(graph, args.codebooks)

    if args.gptq:
        logger.info(
            "%s, GPTQ over %d contiguous %s windows",
            args.config,
            args.gptq_windows,
            args.calib_corpus,
        )
        gptq(
            graph,
            pick(range(args.gptq_windows)),
            skip=args.skip,
            damping=args.damping,
            act_order=not args.no_act_order,
            within_block=not args.no_within_order,
            shrink=args.shrink,
            folded=not args.sequential,
            reserve=int(args.cache_reserve * 2**30),
        )

    if args.no_eval:
        return

    test = tokenizer(
        "\n\n".join(
            load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
        ),
        return_tensors="pt",
    )
    ppl = evaluate_perplexity(graph, test, WINDOW, 512, device).item()
    source = (
        args.weighting if args.fit else ("json" if args.codebooks else "seed")
    )
    gptq_note = " + GPTQ" if args.gptq else ""
    print(
        f"\n{args.config} + {source} codebooks{gptq_note}: {ppl:.6f}  "
        f"(gap {ppl - BF16_PERPLEXITY:+.4f} over bf16)"
    )


if __name__ == "__main__":
    main(parse_args())
