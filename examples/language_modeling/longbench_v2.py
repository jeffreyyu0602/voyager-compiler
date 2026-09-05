"""LongBench v2 evaluation of a Hugging Face causal language model.

Loads ``--model_id`` with the transformers API and answers the 503
multiple-choice questions of LongBench v2 by greedy generation, either
directly or reasoning step by step first (``--cot``), then reports accuracy
overall and by difficulty, context length and domain. Predictions are
written to ``<output_dir>/<mode>.jsonl`` and the scores to
``<output_dir>/<mode>_result.json``.

Example:
    python longbench_v2.py --model_id meta-llama/Llama-3.1-8B-Instruct \\
        --max_length 65536 --gpu 0
"""

import argparse
import json
import os
import re
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

from common import (
    add_inference_args,
    add_model_args,
    generate_answer,
    load_model_and_tokenizer,
)

LONGBENCH_V2_REPO = "THUDM/LongBench-v2"
MAX_NEW_TOKENS = 128
MAX_NEW_TOKENS_COT = 1024

PROMPT = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n\n"
    'Format your response as follows: "The correct answer is (insert answer '
    'here)".'
)

PROMPT_COT = (
    "Please read the following text and answer the questions below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n\n"
    "Let’s think step by step:"
)

PROMPT_COT_ANSWER = (
    "Please read the following text and answer the questions below.\n\n"
    "The text is too long and omitted here.\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n\n"
    "Let’s think step by step: {cot}\n\n"
    "Based on the above, what is the single, most likely answer choice? "
    'Format your response as follows: "The correct answer is (insert answer '
    'here)".'
)

PROMPT_FIELDS = (
    "context",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
)
RECORD_FIELDS = (
    "_id",
    "domain",
    "sub_domain",
    "difficulty",
    "length",
    "question",
    "answer",
)
ANSWER_PATTERN = re.compile(r"The correct answer is \(?([A-D])\)?")


def parse_args():
    parser = argparse.ArgumentParser(description="LongBench v2 evaluation.")
    add_model_args(parser)
    add_inference_args(parser)
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Reason step by step before answering (two generation passes)",
    )
    return parser.parse_args()


def predict(model, tokenizer, samples, max_length, chat_template, cot):
    """Answer every question, directly or through a chain of thought.

    Direct mode asks for the answer letter in one pass. Chain-of-thought
    mode first generates the reasoning against the full context, then asks
    for the letter in a second, context-free pass that quotes the reasoning.
    The letter is read from ``"The correct answer is (X)"`` in the response.

    Args:
        model: Causal LM whose ``generate`` produces the answers.
        tokenizer: Its tokenizer.
        samples: LongBench v2 rows.
        max_length: Prompt token budget.
        chat_template: Wrap prompts in the tokenizer's chat template.
        cot: Use chain-of-thought mode.

    Returns:
        One record per sample with the sample's metadata, the model's
        ``response`` (and ``response_cot``), the extracted ``pred`` letter
        and whether it matches the answer (``judge``).
    """
    records = []
    for sample in tqdm(samples, desc="longbench_v2"):
        fields = {name: sample[name].strip() for name in PROMPT_FIELDS}
        record = {name: sample[name] for name in RECORD_FIELDS}
        if cot:
            reasoning = generate_answer(
                model,
                tokenizer,
                PROMPT_COT.format(**fields),
                max_length,
                chat_template,
                max_new_tokens=MAX_NEW_TOKENS_COT,
            ).strip()
            record["response_cot"] = reasoning
            prompt = PROMPT_COT_ANSWER.format(**fields, cot=reasoning)
        else:
            prompt = PROMPT.format(**fields)
        response = generate_answer(
            model,
            tokenizer,
            prompt,
            max_length,
            chat_template,
            max_new_tokens=MAX_NEW_TOKENS,
        ).strip()
        match = ANSWER_PATTERN.search(response.replace("*", ""))
        pred = match.group(1) if match else None
        record["response"] = response
        record["pred"] = pred
        record["judge"] = pred == sample["answer"]
        records.append(record)
    return records


def score(records):
    """Accuracy in percent: overall and per difficulty, length and domain.

    Args:
        records: Prediction records from ``predict``.

    Returns:
        ``{"overall": float, "difficulty": {...}, "length": {...},
        "domain": {...}}`` with one entry per category value.
    """
    groups = defaultdict(list)
    for record in records:
        for key in ("difficulty", "length", "domain"):
            groups[key, record[key]].append(record["judge"])
    result = {
        "overall": round(
            100 * sum(record["judge"] for record in records) / len(records), 1
        )
    }
    for (key, value), judges in groups.items():
        result.setdefault(key, {})[value] = round(
            100 * sum(judges) / len(judges), 1
        )
    return result


def main(args):
    if args.gpu is None:
        model, tokenizer = load_model_and_tokenizer(
            args.model_id, args.torch_dtype, device_map="auto"
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.model_id, args.torch_dtype
        )
        model.to(f"cuda:{args.gpu}")
    chat_template = (
        not args.no_chat_template and tokenizer.chat_template is not None
    )

    max_new_tokens = MAX_NEW_TOKENS_COT if args.cot else MAX_NEW_TOKENS
    max_length = args.max_length
    if max_length is None:
        max_length = model.config.max_position_embeddings - max_new_tokens
    mode = "cot" if args.cot else "direct"
    output_dir = args.output_dir or os.path.join(
        "pred_v2", args.model_id.split("/")[-1]
    )
    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(output_dir, f"{mode}.jsonl")
    if os.path.exists(pred_path) and not args.overwrite:
        print(f"reusing predictions in {pred_path}")
        with open(pred_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    else:
        samples = load_dataset(LONGBENCH_V2_REPO, split="train")
        if args.num_samples is not None:
            samples = samples.select(range(args.num_samples))
        records = predict(
            model, tokenizer, samples, max_length, chat_template, args.cot
        )
        with open(pred_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = score(records)
    result_path = os.path.join(output_dir, f"{mode}_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"model:      {args.model_id}")
    print(f"mode:       {mode}")
    print(f"max length: {max_length}")
    print(f"overall:    {result['overall']}")
    for key in ("difficulty", "length", "domain"):
        for value, accuracy in result[key].items():
            print(f"  {value}: {accuracy}")
    print(f"results:    {result_path}")


if __name__ == "__main__":
    main(parse_args())
