"""LongBench evaluation of a Hugging Face causal language model.

Loads ``--model_id`` with the transformers API, greedily generates an answer
for every sample of the English LongBench tasks (LongBench-E with ``--e``),
and scores the predictions with the official LongBench metrics. Predictions
are written to ``<output_dir>/<task>.jsonl`` in the layout the official
``eval.py`` reads and the scores to ``<output_dir>/result.json``.

Example:
    python longbench.py --model_id meta-llama/Llama-3.1-8B-Instruct \\
        --max_length 31500 --gpu 0
"""

import argparse
import json
import math
import logging
import os
import re
import string
import zipfile
from collections import Counter

import torch
from fuzzywuzzy import fuzz
from huggingface_hub import hf_hub_download
from rouge import Rouge
from tqdm import tqdm

from common import (
    BF16_CONFIG,
    SEQ_BLOCK,
    add_inference_args,
    add_model_args,
    add_quantization_args,
    build_generator,
    generate_answer,
    load_model_and_tokenizer,
)

LONGBENCH_REPO = "THUDM/LongBench"

DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

DATASETS_E = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

# Few-shot and code tasks: the prompt is used verbatim, without a chat
# template, and only the first generated line is scored.
FEW_SHOT_DATASETS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}
FIRST_LINE_DATASETS = {"trec", "triviaqa", "samsum"}

DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie "
        "script, and a question. Answer the question asconcisely as you can, "
        "using a single phrase if possible. Do not provide any explanation."
        "\n\nStory: {context}\n\nNow, answer the question based on the story "
        "asconcisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the "
        "question as concisely as you can, using a single phrase or sentence "
        "if possible. If the question cannot be answered based on the "
        'information in the article, write "unanswerable". If the question '
        'is a yes/no question, answer "yes", "no", or "unanswerable". Do not '
        "provide any explanation.\n\nArticle: {context}\n\n Answer the "
        "question based on the above article as concisely as you can, using "
        "a single phrase or sentence if possible. If the question cannot be "
        "answered based on the information in the article, write "
        '"unanswerable". If the question is a yes/no question, answer "yes", '
        '"no", or "unanswerable". Do not provide any explanation.\n\n'
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, "
        "answer the following question based on the above text, only give "
        "me the answer and do not output any other words.\n\nQuestion: "
        "{input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are "
        "given passages.\n{context}\n\nAnswer the question based on the "
        "given passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are "
        "given passages.\n{context}\n\nAnswer the question based on the "
        "given passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are "
        "given passages.\n{context}\n\nAnswer the question based on the "
        "given passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n{context}\n\nNow, write a "
        "one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a "
        "question or instruction. Answer the query in one or more sentences."
        "\n\nTranscript:\n{context}\n\nNow, answer the query based on the "
        "above meeting transcript in one or more sentences.\n\nQuery: "
        "{input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of "
        "all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of "
        "all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some "
        "examples of questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the "
        "answer and do not output any other words. The following are some "
        "examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following "
        "are some examples.\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of "
        "them may be duplicates. Please carefully read these paragraphs and "
        "determine how many unique paragraphs there are after removing "
        "duplicates. In other words, how many non-repeating paragraphs are "
        "there in total?\n\n{context}\n\nPlease enter the final count of "
        "unique paragraphs after removing duplicates. The output format "
        "should only contain the number, such as 1, 2, 3, and so on.\n\nThe "
        "final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease "
        "enter the number of the paragraph that the abstract is from. The "
        'answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\n'
        "The answer is: "
    ),
    "lcc": (
        "Please complete the code given below. \n{context}Next line of "
        "code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below. \n{context}{input}Next line "
        "of code:\n"
    ),
}

DATASET2MAXGEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}


def add_longbench_args(parser):
    """Add the task selection options to ``parser``."""
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="LongBench tasks to run (default: every English task)",
    )
    parser.add_argument(
        "--e",
        action="store_true",
        help="Evaluate on LongBench-E and report scores per length bucket",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="LongBench evaluation.")
    add_model_args(parser)
    add_inference_args(parser)
    add_longbench_args(parser)
    add_quantization_args(parser)
    args = parser.parse_args()
    if (args.qconfig or args.kivi) and args.max_length is None:
        parser.error("--max_length sizes the quantized graphs' KV cache")
    return args


def load_longbench(dataset, longbench_e):
    """Return the samples of one LongBench task as a list of dicts.

    The hub repo ships every task as ``data/<task>.jsonl`` inside
    ``data.zip`` behind a loading script that recent ``datasets`` versions
    no longer execute, so the archive is read directly.

    Args:
        dataset: Task name, e.g. ``"narrativeqa"``.
        longbench_e: Read the LongBench-E split (``<task>_e``) instead.
    """
    archive = hf_hub_download(LONGBENCH_REPO, "data.zip", repo_type="dataset")
    member = (
        f"data/{dataset}_e.jsonl" if longbench_e else f"data/{dataset}.jsonl"
    )
    with zipfile.ZipFile(archive) as zf, zf.open(member) as f:
        return [json.loads(line) for line in f]


def predict(
    model,
    tokenizer,
    samples,
    dataset,
    max_length,
    max_gen,
    chat_template,
    out=None,
):
    """Greedily generate an answer for every sample of one task.

    Few-shot and code tasks get the prompt verbatim; the others go through
    the tokenizer's chat template when ``chat_template`` is set. ``samsum``
    additionally stops at the first newline, as in the official runner.

    Args:
        model: Causal LM whose ``generate`` produces the answers.
        tokenizer: Its tokenizer.
        samples: Task samples from ``load_longbench``.
        dataset: Task name, selecting the prompt template.
        max_length: Prompt token budget.
        max_gen: Maximum number of generated tokens.
        chat_template: Chat-template the prompts of non-few-shot tasks.
        out: File each record is appended to as it is produced, so an
            interrupted run resumes where it stopped.

    Returns:
        One record per sample with ``pred``, ``answers``, ``all_classes``
        and ``length``: the layout the official LongBench ``eval.py`` reads.
    """
    prompt_format = DATASET2PROMPT[dataset]
    generate_kwargs = {"max_new_tokens": max_gen}
    if dataset == "samsum":
        newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
        generate_kwargs["min_new_tokens"] = 1
        generate_kwargs["eos_token_id"] = [tokenizer.eos_token_id, newline_id]
    chat_template = chat_template and dataset not in FEW_SHOT_DATASETS

    records = []
    for sample in tqdm(samples, desc=dataset):
        pred = generate_answer(
            model,
            tokenizer,
            prompt_format.format(**sample),
            max_length,
            chat_template,
            **generate_kwargs,
        )
        record = {
            "pred": pred,
            "answers": sample["answers"],
            "all_classes": sample["all_classes"],
            "length": sample["length"],
        }
        records.append(record)
        if out is not None:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    return records


def normalize_answer(text):
    """Lower-case and strip punctuation, articles and extra whitespace."""
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1_score(prediction, ground_truth, **kwargs):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_score(prediction, ground_truth, **kwargs):
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except ValueError:
        return 0.0
    return scores["rouge-l"]["f"]


def classification_score(prediction, ground_truth, **kwargs):
    all_classes = kwargs["all_classes"]
    matches = [name for name in all_classes if name in prediction]
    for name in list(matches):
        if name in ground_truth and name != ground_truth:
            matches.remove(name)
    if ground_truth in matches:
        return 1.0 / len(matches)
    return 0.0


def retrieval_score(prediction, ground_truth, **kwargs):
    ground_truth_id = re.findall(r"Paragraph (\d+)", ground_truth)[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return numbers.count(ground_truth_id) / len(numbers)


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return numbers.count(str(ground_truth)) / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs):
    # The first generated line that is not a comment or a code fence.
    prediction = next(
        (
            line
            for line in prediction.lstrip("\n").split("\n")
            if "`" not in line and "#" not in line and "//" not in line
        ),
        "",
    )
    return fuzz.ratio(prediction, ground_truth) / 100


DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def score(dataset, records, longbench_e):
    """Score one task's predictions with the official LongBench metric.

    Each prediction takes the best score over its reference answers; the
    task score is the mean in percent. On LongBench-E the mean is reported
    per prompt-length bucket (``0-4k``, ``4-8k``, ``8k+``) instead.

    Args:
        dataset: Task name, selecting the metric.
        records: Prediction records from ``predict``.
        longbench_e: Bucket the scores by ``length``.

    Returns:
        A float, or a dict of bucket name to float for LongBench-E.
    """
    metric = DATASET2METRIC[dataset]
    buckets = {}
    for record in records:
        prediction = record["pred"]
        if dataset in FIRST_LINE_DATASETS:
            prediction = prediction.lstrip("\n").split("\n")[0]
        best = max(
            (
                metric(prediction, answer, all_classes=record["all_classes"])
                for answer in record["answers"]
            ),
            default=0.0,
        )
        if not longbench_e:
            bucket = "all"
        elif record["length"] < 4000:
            bucket = "0-4k"
        elif record["length"] < 8000:
            bucket = "4-8k"
        else:
            bucket = "8k+"
        buckets.setdefault(bucket, []).append(best)
    means = {
        name: round(100 * sum(values) / len(values), 2)
        for name, values in buckets.items()
    }
    return means if longbench_e else means["all"]


def evaluate(model, tokenizer, args, output_dir):
    """Run and score every selected task, writing predictions and scores.

    Existing predictions in ``output_dir`` are reused unless
    ``args.overwrite`` is set, so an interrupted run resumes after the last
    sample it wrote.

    Args:
        model: Anything with ``generate`` and ``device``: an HF causal LM,
            or the quantized generator ``common.build_generator`` returns.
            Without ``args.max_length`` its ``config`` supplies the prompt
            budget.
        tokenizer: Its tokenizer.
        args: Parsed options carrying the inference and task settings.
        output_dir: Where ``<task>.jsonl`` and ``result.json`` go.

    Returns:
        ``{task: score}``, as written to ``result.json``.
    """
    chat_template = (
        not args.no_chat_template and tokenizer.chat_template is not None
    )
    datasets = args.datasets or (DATASETS_E if args.e else DATASETS)
    os.makedirs(output_dir, exist_ok=True)

    scores = {}
    for dataset in datasets:
        max_gen = DATASET2MAXGEN[dataset]
        max_length = args.max_length
        if max_length is None:
            max_length = model.config.max_position_embeddings - max_gen
        pred_path = os.path.join(output_dir, f"{dataset}.jsonl")
        records = []
        if os.path.exists(pred_path) and not args.overwrite:
            with open(pred_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
        samples = load_longbench(dataset, args.e)[: args.num_samples]
        if len(records) >= len(samples):
            print(f"{dataset}: reusing predictions in {pred_path}")
        else:
            if records:
                print(f"{dataset}: resuming after {len(records)} predictions")
            with open(
                pred_path, "a" if records else "w", encoding="utf-8"
            ) as f:
                records += predict(
                    model,
                    tokenizer,
                    samples[len(records) :],
                    dataset,
                    max_length,
                    max_gen,
                    chat_template,
                    f,
                )
        scores[dataset] = score(dataset, records, args.e)
        print(f"{dataset}: {scores[dataset]}")

    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    return scores


def main(args):
    quantized = args.qconfig is not None or args.kivi
    if quantized:
        # The compiler's graphs run on one device; without --gpu, the CPU.
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        device = (
            torch.device(f"cuda:{args.gpu}")
            if args.gpu is not None
            else torch.device("cpu")
        )
        if device.type == "cuda":
            torch.cuda.set_device(device)
        datasets = args.datasets or (DATASETS_E if args.e else DATASETS)
        longest_generation = max(DATASET2MAXGEN[name] for name in datasets)
        # Whole prefill blocks and whole residual chunks.
        block = math.lcm(SEQ_BLOCK, args.residual_length)
        cache_len = args.max_length + SEQ_BLOCK + longest_generation
        cache_len = -(-cache_len // block) * block
        model, tokenizer = build_generator(
            args.model_id,
            args.torch_dtype,
            args.qconfig or BF16_CONFIG,
            args.decode_qconfig,
            args.kivi,
            args.residual_length,
            args.bake,
            args.codebooks,
            cache_len,
            device,
        )
    elif args.gpu is None:
        model, tokenizer = load_model_and_tokenizer(
            args.model_id, args.torch_dtype, device_map="auto"
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.model_id, args.torch_dtype
        )
        model.to(f"cuda:{args.gpu}")

    tag = [args.qconfig] if args.qconfig else []
    tag += [f"decode-{args.decode_qconfig}"] if args.decode_qconfig else []
    tag += ["kivi"] if args.kivi else []
    tag += ["baked"] if args.bake else []
    output_dir = args.output_dir or os.path.join(
        "pred_e" if args.e else "pred",
        "-".join([args.model_id.rstrip("/").split("/")[-1], *tag]),
    )
    evaluate(model, tokenizer, args, output_dir)
    print(f"model:      {args.model_id}")
    if quantized:
        print(f"scheme:     {'-'.join(tag)}")
    print(f"max length: {args.max_length or 'model context'}")
    print(f"results:    {os.path.join(output_dir, 'result.json')}")


if __name__ == "__main__":
    main(parse_args())
