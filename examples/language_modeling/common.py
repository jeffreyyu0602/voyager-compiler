"""Pieces shared by the language-modeling evaluation scripts.

Model loading, the CLI options every script takes, sliding-window
perplexity, greedy prompt completion for the LongBench scripts, and the
quantized generator: the compiler's prefill and decode graphs, prepared
with their weights and KV-cache reads frozen, driven through a greedy loop
behind an HF-style ``generate``.
"""

import json
import logging
import time

import torch
from torch._export.utils import _disable_aten_to_metadata_assertions
from torch.utils._pytree import register_pytree_node
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.cache_utils import DynamicCache
from transformers.integrations.executorch import convert_and_export_with_cache

from quantization_configs import (
    KIVI_BLOCK_SIZE,
    KIVI_RESIDUAL_LENGTH,
    QUANTIZATION_CONFIGS,
    annotate_kivi_cache,
    set_kivi_attention_qconfig,
    set_qconfig,
    set_residual_attention_qconfig,
)
from voyager_compiler import (
    export_model,
    freeze_cache_reads,
    freeze_weights,
    get_default_quantizer,
    prepare_pt2e,
    sink_obs_or_fq,
    split_kv_cache,
)
from voyager_compiler.codegen import (
    remove_fp32_casts,
    replace_rmsnorm_with_layer_norm,
)
from voyager_compiler.quantization import load_codebooks
from voyager_compiler.quantization.codebook_optimizer import _quantized_operands

logger = logging.getLogger(__name__)


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
    """Load a text-generation model and its tokenizer.

    A vision-language checkpoint registers only under the
    image-text-to-text auto class, so it is loaded through that one and
    driven text-only; everything else goes through the causal-LM class.

    Args:
        model_id: Hub id or local path of the checkpoint.
        torch_dtype: ``"auto"`` or the name of a torch dtype, as given on
            the command line.
        **from_pretrained_kwargs: Forwarded to ``from_pretrained``
            (``device_map``, ``attn_implementation``, ...).

    Returns:
        ``(model, tokenizer)``.
    """
    dtype = (
        torch_dtype if torch_dtype == "auto" else getattr(torch, torch_dtype)
    )
    config = AutoConfig.from_pretrained(model_id)
    auto_class = (
        AutoModelForCausalLM
        if type(config) in AutoModelForCausalLM._model_mapping
        else AutoModelForImageTextToText
    )
    model = auto_class.from_pretrained(
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


# The scheme that quantizes nothing; the other names in QUANTIZATION_CONFIGS
# turn on the compiler's graph rewrites.  The softmax and layer norm stay
# unquantized: the fp8 spec the compiler path puts on the softmax input sees
# the causal mask's ``finfo.min`` and scales every real score to zero.
BF16_CONFIG = "bf16"
# The rotary embedding's ``inv_freq @ position`` matmul stays unquantized; a
# regex, so it matches the prefill scope and the decode wrapper's.
ROTARY_SCOPE = r"model\.rotary_emb"
# Prompt lengths the prefill graph is exported for come in blocks of this
# many tokens: the microscaling block along the sequence axis.
SEQ_BLOCK = 64
# A prompt is padded to a multiple of this many tokens, so the prefill
# graph meets few lengths: its fake-quants compile once per length.  The
# padding sits after the prompt, where the causal mask hides it.
PROMPT_BUCKET = 512


def add_quantization_args(parser):
    """Add the options that pick the quantized generator over the HF model:
    any of them set means the compiler's graphs run instead."""
    parser.add_argument(
        "--qconfig",
        default=None,
        choices=sorted(QUANTIZATION_CONFIGS),
        help=(
            "Scheme for the linears, attention and lm_head; with --kivi "
            "alone everything but the KV cache stays bf16"
        ),
    )
    parser.add_argument(
        "--decode_qconfig",
        default=None,
        choices=sorted(QUANTIZATION_CONFIGS),
        help=(
            "Scheme for the decode graph when it differs from --qconfig, "
            "e.g. mxlut4_int6_e5m3_a_int6 to decode with int6 activations"
        ),
    )
    parser.add_argument(
        "--kivi",
        action="store_true",
        help="Quantize the decode KV cache to KIVI's 2-bit layout",
    )
    parser.add_argument(
        "--residual_length",
        type=int,
        default=KIVI_RESIDUAL_LENGTH,
        help=(
            "Positions of the KIVI cache kept in full precision while their "
            "chunk fills; a multiple of the 64-token group"
        ),
    )
    parser.add_argument(
        "--codebooks",
        default=None,
        help="JSON of fitted tables to install, as codebook_eval.py dumps",
    )


def _flatten_cache(cache):
    return [
        t for layer in cache.layers for t in (layer.keys, layer.values)
    ], None


def _unflatten_cache(tensors, context):
    cache = DynamicCache()
    for layer, index in enumerate(range(0, len(tensors), 2)):
        cache.update(tensors[index], tensors[index + 1], layer)
    return cache


def build_quantizer(qconfig, decode, kivi):
    """Return the quantizer for one graph.

    Args:
        qconfig: Name in ``QUANTIZATION_CONFIGS``.
        decode: The decode graph's quantizer: its split caches' residual
            halves of the attention stay unquantized.
        kivi: Point the attention matmuls at the int6 re-encode of a KIVI
            cache (decode only).
    """
    quantizer = get_default_quantizer()
    quantizer.set_module_name_object_type_order(
        ROTARY_SCOPE, torch.ops.aten.matmul.default, 0, None
    )
    set_qconfig(quantizer, QUANTIZATION_CONFIGS[qconfig])
    if kivi:
        set_kivi_attention_qconfig(quantizer)
    elif decode:
        set_residual_attention_qconfig(quantizer)
    return quantizer


def rewrite_for_compiler(gm, model, seq):
    """Apply the graph rewrites the compiler's Llama path makes before
    quantizing: RMSNorm as the layer-norm op, the softmax in bf16."""
    # get_decoder() is the text decoder, wherever the model nests it.
    layernorm = model.get_decoder().layers[0].input_layernorm
    hidden = layernorm.weight.shape[-1]
    example = torch.randn(
        1, seq, hidden, dtype=model.dtype, device=model.device
    )
    replace_rmsnorm_with_layer_norm(gm, layernorm, (example,))
    remove_fp32_casts(gm)


def build_prefill(model, quantizer, quantized, max_tokens):
    """Export and prepare the prefill graph: a prompt of any multiple of
    ``SEQ_BLOCK`` tokens up to ``max_tokens`` in, logits and the KV cache out.

    Args:
        model: The causal LM, on the device the graph runs on.
        quantizer: From ``build_quantizer``.
        quantized: Apply ``rewrite_for_compiler``.
        max_tokens: Longest prompt the graph accepts.
    """
    vocab_size = model.config.get_text_config().vocab_size
    example = torch.randint(
        0, vocab_size, (1, 2 * SEQ_BLOCK), device=model.device
    )
    kwargs = {"use_cache": True}
    chunk = torch.export.Dim("chunk", min=2, max=max_tokens // SEQ_BLOCK)
    dynamic = {"input_ids": {1: chunk * SEQ_BLOCK}, "use_cache": None}
    gm = export_model(model, (example,), kwargs, dynamic_shapes=dynamic)
    if quantized:
        rewrite_for_compiler(gm, model, 2 * SEQ_BLOCK)
    gm = prepare_pt2e(gm, quantizer, (example,), kwargs, dynamic)
    sink_obs_or_fq(gm)
    return gm


def build_decode(model, quantizer, quantized, split, kivi, cache_len):
    """Export and prepare the decode graph: one token at ``cache_position``
    over a static KV cache of ``cache_len`` slots.

    Args:
        model: The causal LM, on the device the graph runs on.
        quantizer: From ``build_quantizer``.
        quantized: Apply ``rewrite_for_compiler``.
        split: Split each cache into completed chunks and a full-precision
            residual of this many positions; ``None`` leaves the cache
            whole.
        kivi: Quantize the split caches' chunks to KIVI's 2-bit layout.
        cache_len: Slots in the cache.
    """
    model.generation_config = GenerationConfig(
        use_cache=True,
        cache_implementation="static",
        cache_config={"batch_size": 1, "max_cache_len": cache_len},
    )
    input_ids = torch.tensor([[1]], device=model.device)
    cache_position = torch.tensor([0], device=model.device)
    with _disable_aten_to_metadata_assertions():
        ep = convert_and_export_with_cache(
            model,
            example_input_ids=input_ids,
            example_cache_position=cache_position,
        )
    gm = ep.module()
    if quantized:
        rewrite_for_compiler(gm, model, 1)
    if split is not None:
        # The buffers are empty at export; ``load_cache`` fills both halves.
        split_kv_cache(gm, split, 0)
    if kivi:
        caches = annotate_kivi_cache(gm)
        expected = 2 * model.config.get_text_config().num_hidden_layers
        if caches != expected:
            raise RuntimeError(
                f"annotated {caches} KV caches, expected {expected}"
            )
    kwargs = {"input_ids": input_ids, "cache_position": cache_position}
    gm = prepare_pt2e(gm, quantizer, (), kwargs)
    sink_obs_or_fq(gm)
    return gm


def install_codebooks(prefill, decode, path):
    """Install fitted tables into both prepared graphs.

    The decode graph names every operand under the export wrapper's extra
    ``model.`` scope, and quantizes its attention operands its own way, so
    the tables are re-keyed and only those it has an operand for go in.

    Args:
        prefill: Prepared prefill graph.
        decode: Prepared decode graph.
        path: JSON of tables, keyed by the prefill graph's operand names.
    """
    with open(path) as handle:
        tables = json.load(handle)
    installed = load_codebooks(prefill, tables)
    logger.info("prefill: installed %d codebooks", len(installed))
    operands = _quantized_operands(decode, ())
    retagged = {f"model.{name}": entry for name, entry in tables.items()}
    retagged = {name: e for name, e in retagged.items() if name in operands}
    installed = load_codebooks(decode, retagged)
    logger.info("decode: installed %d codebooks", len(installed))


class StaticCacheGenerator:
    """Greedy generation over a prefill graph and a static-cache decode
    graph, behind an HF-style ``generate``.

    On a GPU the decode step is captured once as a CUDA graph over static
    input and output tensors and replayed from then on, which takes the
    Python and launch overhead of its few thousand nodes off every token.
    A split cache's main half is written here, not by the decode graph:
    each chunk as its residual fills, then ``refresh`` recomputes what the
    graph reads from the main caches (``freeze_cache_reads``).

    Args:
        prefill: Graph returning ``(logits, cache)`` for a prompt whose
            length is a multiple of ``SEQ_BLOCK``.
        decode: Static-cache graph taking ``input_ids`` and
            ``cache_position`` for one token.
        cache_len: Slots in the decode cache.
        residual_length: Slots in the decode graph's KIVI residual buffers,
            or ``None`` when its caches are not split.
        refresh: What ``freeze_cache_reads`` returned for the decode
            graph's main caches, or ``None`` when they are not split.
        pad_token_id: Token the prompt is padded with, up to the next
            multiple of ``PROMPT_BUCKET``.
        eos_token_ids: Tokens that end generation unless ``generate`` is
            given others.
        device: Where both graphs live.
    """

    def __init__(
        self,
        prefill,
        decode,
        cache_len,
        residual_length,
        refresh,
        pad_token_id,
        eos_token_ids,
        device,
    ):
        self.prefill = prefill
        self.decode = decode
        self.cache_len = cache_len
        self.residual_length = residual_length
        self.refresh = refresh
        self.pad_token_id = pad_token_id
        self.eos_token_ids = eos_token_ids
        self.device = device
        self.graph = None
        self.replay = False
        self.samples = 0
        self.prefill_seconds = 0.0
        self.tokens = 0
        self.decode_seconds = 0.0

    def fold(self, chunk):
        """Copy every residual into chunk ``chunk`` of its main cache, the
        write the decode graph leaves to the caller, and recompute what the
        graph reads from the main caches."""
        start = chunk * self.residual_length
        for name, residual in self.decode.named_buffers():
            if name.endswith("_residual"):
                main = self.decode.get_buffer(name[: -len("_residual")])
                main[:, :, start : start + self.residual_length] = residual
        self.refresh()

    def load_cache(self, cache, length):
        """Copy the first ``length`` positions of a prefilled cache into the
        decode graph's buffers, zeroing the slots beyond them.  A split
        cache takes its completed chunks; the tail goes to the residual."""
        done = length
        if self.residual_length is not None:
            done = length // self.residual_length * self.residual_length
        for layer, entry in enumerate(cache.layers):
            for kind, tensor in (("key", entry.keys), ("value", entry.values)):
                buffer = self.decode.get_buffer(f"{kind}_cache_{layer}")
                buffer.zero_()
                buffer[:, :, :done] = tensor[:, :, :done]
                if self.residual_length is None:
                    continue
                residual = self.decode.get_buffer(
                    f"{kind}_cache_{layer}_residual"
                )
                residual.zero_()
                residual[:, :, : length - done] = tensor[:, :, done:length]
        if self.refresh is not None:
            self.refresh()

    def capture(self, token, position):
        """Capture one decode step as a CUDA graph.  The warmup steps it
        needs rewrite ``token`` at ``position``, which is what the captured
        step and its first replay do as well."""
        self.static_ids = torch.tensor([[token]], device=self.device)
        self.static_pos = torch.tensor([position], device=self.device)
        # Capture on the graphs' own device: ``torch.cuda.graph`` otherwise
        # captures the current device's stream, which is GPU 0 by default.
        with torch.cuda.device(self.device):
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.decode(
                        input_ids=self.static_ids,
                        cache_position=self.static_pos,
                    )
            torch.cuda.current_stream(self.device).wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                self.graph, stream=torch.cuda.Stream(self.device)
            ):
                out = self.decode(
                    input_ids=self.static_ids, cache_position=self.static_pos
                )
        self.static_logits = (
            out[0] if isinstance(out, (tuple, list)) else out
        )[0, -1]
        logger.info("captured the decode step as a CUDA graph")

    def step(self, token, position):
        """Write ``token`` at ``position`` and return the next logits.  On
        a split cache the chunk ``position`` completes is folded after the
        step."""
        if self.device.type != "cuda" or not self.replay:
            out = self.decode(
                input_ids=torch.tensor([[token]], device=self.device),
                cache_position=torch.tensor([position], device=self.device),
            )
            logits = (out[0] if isinstance(out, (tuple, list)) else out)[0, -1]
        else:
            if self.graph is None:
                self.capture(token, position)
            self.static_ids.fill_(token)
            self.static_pos.fill_(position)
            with torch.cuda.device(self.device):
                self.graph.replay()
            logits = self.static_logits
        if (
            self.refresh is not None
            and position % self.residual_length == self.residual_length - 1
        ):
            self.fold(position // self.residual_length)
        return logits

    def generate(
        self,
        input_ids,
        max_new_tokens,
        min_new_tokens=0,
        eos_token_id=None,
        **unused,
    ):
        """Greedily continue ``input_ids`` and return prompt plus generation,
        as ``model.generate`` does.

        Args:
            input_ids: ``[1, prompt]`` token ids.
            max_new_tokens: Generation budget.
            min_new_tokens: Steps during which the stop tokens are suppressed.
            eos_token_id: Stop token or tokens; the model's by default.
            **unused: The sampling flags ``generate_answer`` passes, which
                greedy decoding has no use for.
        """
        if eos_token_id is None:
            stops = list(self.eos_token_ids)
        elif isinstance(eos_token_id, int):
            stops = [eos_token_id]
        else:
            stops = list(eos_token_id)
        length = input_ids.shape[1]
        if length + max_new_tokens > self.cache_len:
            raise ValueError(
                f"{length} prompt + {max_new_tokens} new tokens exceed the "
                f"{self.cache_len}-slot cache"
            )
        padded = min(
            -(-length // PROMPT_BUCKET) * PROMPT_BUCKET,
            self.cache_len // SEQ_BLOCK * SEQ_BLOCK,
        )
        prompt = torch.nn.functional.pad(
            input_ids, (0, padded - length), value=self.pad_token_id
        )
        started = time.perf_counter()
        logits, cache = self.prefill(prompt, use_cache=True)[:2]
        self.load_cache(cache, length)
        next_logits = logits[0, length - 1]
        next_logits.max().item()
        prefilled = time.perf_counter()
        generated = []
        for step in range(max_new_tokens):
            if step < min_new_tokens:
                next_logits = next_logits.clone()
                next_logits[stops] = float("-inf")
            token = next_logits.argmax().item()
            generated.append(token)
            if token in stops or step == max_new_tokens - 1:
                break
            next_logits = self.step(token, length + step)
        self.samples += 1
        self.prefill_seconds += prefilled - started
        self.tokens += len(generated)
        self.decode_seconds += time.perf_counter() - prefilled
        if self.samples % 20 == 0:
            logger.info(
                "%d samples: prefill %.2f s each, decode %.0f ms per token "
                "over %d tokens",
                self.samples,
                self.prefill_seconds / self.samples,
                1000 * self.decode_seconds / max(self.tokens, 1),
                self.tokens,
            )
        return torch.cat(
            [input_ids, torch.tensor([generated], device=input_ids.device)], 1
        )


@torch.no_grad()
def build_generator(
    model_id,
    torch_dtype,
    qconfig,
    decode_qconfig,
    kivi,
    residual_length,
    codebooks,
    cache_len,
    device,
):
    """Load the model, build and freeze both graphs, and wrap them.

    Exports the model twice: a prefill graph over a dynamic-length prompt
    that returns the KV cache, and a single-token decode graph over a
    static KV cache, quantized the way ``test_codegen.py``'s
    ``llama_prefill`` / ``llama_decode_kivi`` paths quantize them, except
    that the softmax and layer norm stay unquantized.  Both stay prepared:
    every weight is fake-quantized once (``freeze_weights``), and what the
    decode graph reads from a split cache's main half once per chunk
    (``freeze_cache_reads``).

    Args:
        model_id: Hub id or local path of the checkpoint.
        torch_dtype: ``"auto"`` or a torch dtype name, as on the CLI.
        qconfig: Name in ``QUANTIZATION_CONFIGS``; ``BF16_CONFIG`` leaves
            everything but the cache alone.
        decode_qconfig: The decode graph's scheme, or None for ``qconfig``.
        kivi: Quantize the decode KV cache to KIVI's 2-bit layout.
        residual_length: Positions of that cache kept in full precision
            while their chunk fills.
        codebooks: JSON of fitted tables to install, or None.
        cache_len: Slots in the decode cache, a multiple of ``SEQ_BLOCK``
            covering the longest prompt plus generation.
        device: Where everything runs.

    Returns:
        ``(generator, tokenizer)``.
    """
    started = time.monotonic()

    def mark(what):
        logger.info(
            "### %s (%.0f min)", what, (time.monotonic() - started) / 60
        )

    register_pytree_node(DynamicCache, _flatten_cache, _unflatten_cache)
    quantized = qconfig != BF16_CONFIG
    decode_qconfig = decode_qconfig or qconfig
    decode_quantized = decode_qconfig != BF16_CONFIG

    mark("loading the checkpoint")
    model, tokenizer = load_model_and_tokenizer(
        model_id, torch_dtype, attn_implementation="eager"
    )
    model.to(device)
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_token_ids = [eos] if isinstance(eos, int) else list(eos or [])

    mark("exporting the prefill graph")
    prefill = build_prefill(
        model, build_quantizer(qconfig, False, False), quantized, cache_len
    )
    mark(f"exporting the decode graph over {cache_len} cache slots")
    # A quantized cache -- KIVI's 2-bit one, or the int6 re-encode a
    # quantized qconfig's attention matmuls read -- is split; bf16 is not.
    split = residual_length if (kivi or decode_quantized) else None
    decode = build_decode(
        model,
        build_quantizer(decode_qconfig, True, kivi and decode_quantized),
        decode_quantized,
        split,
        kivi,
        cache_len,
    )
    if codebooks:
        install_codebooks(prefill, decode, codebooks)
    mark("freezing the weights")
    freeze_weights(prefill, decode)
    refresh = None
    if split is not None:
        caches = [
            name[: -len("_residual")]
            for name, _ in decode.named_buffers()
            if name.endswith("_residual")
        ]
        refresh = freeze_cache_reads(decode, caches)
        mark(f"froze the reads of {len(caches)} split caches")
    generator = StaticCacheGenerator(
        prefill,
        decode,
        cache_len,
        split,
        refresh,
        tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_ids,
        device,
    )
    del model
    generator.replay = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.empty_cache()
        mark(f"{torch.cuda.memory_allocated(device) / 2**30:.1f} GiB resident")
    return generator, tokenizer
