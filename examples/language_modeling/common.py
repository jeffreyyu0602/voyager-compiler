"""Pieces shared by the language-modeling evaluation scripts.

Model loading, the CLI options every script takes, sliding-window
perplexity, and greedy prompt completion for the LongBench scripts.
"""

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_model_args(parser):
    """Add the ``--model_id`` and ``--torch_dtype`` options to ``parser``."""
    parser.add_argument(
        "--model_id", required=True, help="Pretrained model identifier"
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help=(
            "Override the default `torch.dtype` and load the model under this "
            "dtype. If `auto` is passed, the dtype will be automatically "
            "derived from the model's weights."
        ),
    )


def add_inference_args(parser):
    """Add the generation options shared by the LongBench scripts."""
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Single GPU to run on; omit to shard across all visible GPUs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help=(
            "Prompt token budget; longer prompts are truncated in the "
            "middle. Defaults to the model's context length minus the "
            "generation length."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where predictions and scores go (default: pred*/<model>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate predictions that already exist in output_dir",
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Do not wrap prompts in the tokenizer's chat template",
    )


def load_model_and_tokenizer(model_id, torch_dtype, **from_pretrained_kwargs):
    """Load a causal LM and its tokenizer with the transformers API.

    Args:
        model_id: Hub id or local path of the checkpoint.
        torch_dtype: ``"auto"`` or the name of a torch dtype, as given on
            the command line.
        **from_pretrained_kwargs: Forwarded to
            ``AutoModelForCausalLM.from_pretrained`` (``device_map``,
            ``attn_implementation``, ...).

    Returns:
        ``(model, tokenizer)``.
    """
    dtype = (
        torch_dtype if torch_dtype == "auto" else getattr(torch, torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, **from_pretrained_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def evaluate_perplexity(
    model, encodings, max_length, stride, device, num_steps=None
):
    """Sliding-window perplexity evaluation. Returns a scalar tensor.

    If num_steps is set, exits early after that many windows — useful for
    activation observer calibration (caller can ignore the returned value).
    """
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    # Subtract max_length from seq_len so the last window is max_length long
    for i, begin_loc in enumerate(tqdm(range(0, seq_len - max_length, stride))):
        end_loc = min(begin_loc + max_length, seq_len)
        # May differ from stride on the last window.
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids, use_cache=False)

            # The loss is a CrossEntropyLoss mean over the valid labels. The
            # model scores only trg_len - 1 of them, because it shifts the
            # labels left by one internally.
            nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len or (num_steps is not None and i == num_steps - 1):
            break

    return torch.exp(torch.stack(nlls).mean())


def generate_answer(
    model, tokenizer, prompt, max_length, chat_template, **generate_kwargs
):
    """Greedily complete ``prompt`` and return the decoded completion.

    A prompt longer than ``max_length`` tokens is truncated in the middle so
    the instructions at both ends survive. With ``chat_template`` set, the
    prompt is sent as a single user turn through the tokenizer's chat
    template, which then supplies the special tokens.

    Args:
        model: Causal LM whose ``generate`` produces the completion.
        tokenizer: Its tokenizer.
        prompt: The prompt text.
        max_length: Prompt token budget.
        chat_template: Wrap the prompt in the chat template.
        **generate_kwargs: Forwarded to ``model.generate``; at least
            ``max_new_tokens``.
    """
    prompt_ids = tokenizer(prompt, truncation=False).input_ids
    if len(prompt_ids) > max_length:
        half = max_length // 2
        prompt = tokenizer.decode(
            prompt_ids[:half], skip_special_tokens=True
        ) + tokenizer.decode(prompt_ids[-half:], skip_special_tokens=True)
    if chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(
        prompt,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=not chat_template,
    ).to(model.device)
    context_length = inputs.input_ids.shape[-1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )[0]
    return tokenizer.decode(output[context_length:], skip_special_tokens=True)
