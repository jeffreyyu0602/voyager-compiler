"""Transcribe LibriSpeech test-clean with Whisper and report the WER.

The encoder and a static-KV-cache decoder are exported separately, following
transformers' executorch recipe, and prepared as the quantization flags ask.
A greedy loop that forces Whisper's English transcription prompt drives them,
one clip at a time.
"""

import argparse
import io
import logging
import os
from types import SimpleNamespace

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from evaluate import load
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.integrations.executorch import (
    Seq2SeqLMDecoderExportableModuleWithStaticCache,
)

from voyager_compiler import (
    add_experiment_args,
    disable_observers,
    export_model,
    get_default_quantizer,
    prepare_pt2e,
    with_execution_context,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform whisper model inference on LibriSpeech dataset."
    )
    parser.add_argument(
        "--model_id",
        default="openai/whisper-tiny",
        help="Model to perform evaluation.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Evaluation batch size."
    )
    parser.add_argument(
        "--output_dir", default=None, help="Output directory for scores."
    )
    add_experiment_args(parser)
    return parser.parse_args()


class WhisperEncoder(torch.nn.Module):
    """Whisper's encoder over log-mel features, returning its states."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.get_encoder()

    def forward(self, input_features):
        return self.encoder(input_features).last_hidden_state


def whisper_decoder(model):
    """HF's static-cache seq2seq decoder wrapper, with Whisper's head.

    The wrapper reads the head as ``model.lm_head``, which Whisper names
    ``proj_out``; a proxy supplies it without adding a module to ``model``.
    Export it before ever calling it: a call caches the cross-attention keys
    and values, which export would then bake in as constants.

    Args:
        model: The Whisper model.

    Returns:
        The wrapper, in eval mode, its cache in ``model``'s dtype.
    """
    proxy = SimpleNamespace(
        get_decoder=model.get_decoder,
        lm_head=model.proj_out,
        config=model.config,
        parameters=model.parameters,
    )
    decoder = Seq2SeqLMDecoderExportableModuleWithStaticCache(
        proxy,
        max_static_cache_length=model.config.max_target_positions,
        batch_size=1,
    )
    # The wrapper allocates the cache in float32.  Converting through ``data``
    # keeps each buffer the same tensor the cache layers hold.
    for buffer in decoder.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(model.dtype)
    return decoder.eval()


def prepare_whisper(model, args, features):
    """Export Whisper's encoder and decoder and prepare each one.

    Args:
        model: The Whisper model, in eval mode.
        args: Parsed command-line arguments.
        features: One clip's log-mel features to export with.

    Returns:
        The prepared encoder and decoder graphs.
    """
    quantizer = get_default_quantizer(
        input_activation=args.activation,
        weight=args.weight,
        bias=args.bias,
        force_scale_power_of_two=args.force_scale_power_of_two,
    )
    encoder = WhisperEncoder(model).eval()
    states = encoder(features)
    device = features.device
    example = (
        torch.zeros(1, 1, dtype=torch.long, device=device),
        states,
        torch.zeros(1, dtype=torch.long, device=device),
    )
    return (
        prepare_pt2e(export_model(encoder, (features,)), quantizer),
        prepare_pt2e(export_model(whisper_decoder(model), example), quantizer),
    )


@torch.no_grad()
def transcribe(encoder, decoder, features, generation_config):
    """Greedily decode one clip, forcing Whisper's English prompt.

    The prompt goes in one token per call, as the static cache takes it, and
    the tokens ``generate`` suppresses are suppressed here too.

    Args:
        encoder: Prepared encoder graph.
        decoder: Prepared decoder graph.
        features: The clip's log-mel features, batch of one.
        generation_config: The model's generation config.

    Returns:
        The generated token ids, the prompt and eos excluded.
    """
    config = generation_config
    prompt = [
        config.decoder_start_token_id,
        config.lang_to_id["<|en|>"],
        config.task_to_id["transcribe"],
        config.no_timestamps_token_id,
    ]
    device = features.device
    states = encoder(features)

    def step(token, position):
        logits = decoder(
            torch.tensor([[token]], device=device),
            states,
            torch.tensor([position], device=device),
        )
        return logits[0, -1].float()

    for position, token in enumerate(prompt[:-1]):
        step(token, position)
    token, tokens = prompt[-1], []
    for position in range(len(prompt) - 1, config.max_length - 1):
        logits = step(token, position)
        logits[config.suppress_tokens] = -float("inf")
        if position == len(prompt) - 1:
            logits[config.begin_suppress_tokens] = -float("inf")
        token = int(logits.argmax())
        if token == config.eos_token_id:
            break
        tokens.append(token)
    return tokens


@with_execution_context
def main(args):
    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu}" if args.gpu is not None else "cuda"
        )
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")

    # The audio is decoded with soundfile: datasets needs torchcodec for it.
    librispeech_test_clean = load_dataset(
        "openslr/librispeech_asr", "clean", split="test"
    ).cast_column("audio", Audio(decode=False))

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()
    if args.bf16:
        model.bfloat16()

    processor = AutoProcessor.from_pretrained(args.model_id)
    normalize = processor.tokenizer.normalize

    def features(audio):
        array, sampling_rate = sf.read(io.BytesIO(audio["bytes"]))
        input_features = processor(
            array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features
        return input_features.to(device, model.dtype)

    def predict(audio):
        tokens = transcribe(
            encoder, decoder, features(audio), model.generation_config
        )
        return normalize(processor.decode(tokens, skip_special_tokens=True))

    example = librispeech_test_clean[0]["audio"]
    encoder, decoder = prepare_whisper(model, args, features(example))
    if args.calibration_steps > 0:
        calibration = librispeech_test_clean[: args.calibration_steps]
        for audio in calibration["audio"]:
            predict(audio)
        disable_observers(encoder)
        disable_observers(decoder)

    def map_to_pred(batch):
        batch["reference"] = [normalize(text) for text in batch["text"]]
        batch["prediction"] = [predict(audio) for audio in batch["audio"]]
        return batch

    result = librispeech_test_clean.map(
        map_to_pred, batched=True, batch_size=args.batch_size
    )

    wer = load("wer")
    logger.info(
        100
        * wer.compute(
            references=result["reference"], predictions=result["prediction"]
        )
    )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
            f.write("\n".join(result["prediction"]) + "\n")
        with open(os.path.join(args.output_dir, "references.txt"), "w") as f:
            f.write("\n".join(result["reference"]) + "\n")


if __name__ == "__main__":
    main(parse_args())
