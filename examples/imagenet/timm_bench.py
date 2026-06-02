"""
Case-1 TIMM benchmark: evaluate accuracy impact of approx nonlinear ops.

Runs two passes over the validation set:
  1. Baseline (standard float model)
  2. With piecewise approximations bound — only the functions the model
     actually uses, as declared in model_registry.py.

Usage:
    # Run all registered (non-skipped) models:
    python timm_bench.py /home/zhouhua/nas/imagenet --all --tag formal

    # Run a specific model by arch name:
    python timm_bench.py /home/zhouhua/nas/imagenet --arch efficientnet_b0 --tag formal

    # Run all models in a family:
    python timm_bench.py /home/zhouhua/nas/imagenet --family transformer --tag kartik-thesis

    # Limit dataset size for quick iteration:
    python timm_bench.py /home/zhouhua/nas/imagenet --all --tag formal --num_eval_samples 5000
"""

import argparse
import os
import random
import time
from enum import Enum

import timm
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
from timm.data import resolve_model_data_config, create_transform
from torch.utils.data import Subset

from voyager_compiler.approx.app_binding import bind_approx, unbind_all
from model_registry import MODELS, get_active_models, get_models_by_family, ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="TIMM approx benchmark (Case 1)")
    parser.add_argument("data", metavar="DIR", help="path to ImageNet dataset")

    # Model selection — mutually exclusive modes
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true",
                     help="run all non-skipped models from model_registry.py")
    sel.add_argument("--arch", nargs="+",
                     help="one or more timm arch names (looked up in registry)")
    sel.add_argument("--family", help="run all models in a registry family")

    parser.add_argument(
        "--tag", default="formal",
        help="approx registry tag to use (default: formal)",
    )
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("-p", "--print-freq", default=20, type=int)
    parser.add_argument("--num_eval_samples", default=None, type=int,
                        help="limit validation set size")
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--mode", default="approx", choices=["approx", "bf16", "bf16-clamp"],
                        help="approx = piecewise polynomial, bf16 = cast to bf16 (no clamp), bf16-clamp = bf16 + clamp to tag segment range")
    parser.add_argument("--output", default=None, type=str,
                        help="save results to CSV file")
    parser.add_argument("--patch-softmax", action="store_true",
                        help="also patch softmax/SDPA to route exp through approx")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="skip baseline run (use stored results)")
    parser.add_argument("--skip-models", nargs="*", default=[],
                        help="models to skip (e.g. --skip-models efficientnet_b4,crossvit_base_240)")
    args = parser.parse_args()
    # Support both comma-separated and space-separated skip lists
    expanded = []
    for s in args.skip_models:
        expanded.extend(s.split(","))
    args.skip_models = expanded
    return args


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

class Summary(Enum):
    NONE = 0
    AVERAGE = 1


class AverageMeter:
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return ('{name} {val' + self.fmt + '} ({avg' + self.fmt + '})').format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.AVERAGE:
            return f"{self.name} {self.avg:.3f}"
        return ""


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        num_digits = len(str(num_batches))
        self.fmt = f"[{{:{num_digits}d}}/{num_batches}]"
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.fmt.format(batch)]
        entries += [str(m) for m in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        print(" * " + "  ".join(m.summary() for m in self.meters))


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size) for k in topk]


def validate(val_loader, model, criterion, device, args):
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ")

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i + 1)

    progress.display_summary()
    return top1.avg, top5.avg


def build_val_loader(arch, args):
    """Build val loader using TIMM's recommended transforms for the given arch."""
    model_for_config = timm.create_model(arch, pretrained=False)
    data_config = resolve_model_data_config(model_for_config)
    transform = create_transform(**data_config, is_training=False)

    val_dataset = datasets.ImageFolder(os.path.join(args.data, "val"), transform=transform)

    if args.num_eval_samples is not None and args.num_eval_samples < len(val_dataset):
        indices = random.sample(range(len(val_dataset)), args.num_eval_samples)
        val_dataset = Subset(val_dataset, indices)

    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )


def bench_model(cfg: ModelConfig, val_loader, device, args):
    print(f"\n{'='*60}")
    print(f"Model:       {cfg.arch}")
    print(f"Family:      {cfg.family}")
    print(f"Activations: {cfg.activations}  (tag={args.tag})")
    if cfg.note:
        print(f"Note:        {cfg.note}")
    print(f"{'='*60}")

    # Activations this model uses that are also registered under the tag
    from voyager_compiler.approx.app_param import get_all_names_by_tag
    registered = set(get_all_names_by_tag(args.tag))
    to_bind = [fn for fn in cfg.activations if fn in registered]
    not_covered = [fn for fn in cfg.activations if fn not in registered]

    if not_covered:
        print(f"  [warn] not in registry under '{args.tag}': {not_covered}")
    if not to_bind:
        print(f"  [skip] no activations for this model covered by tag '{args.tag}'")
        return None

    # When --patch-softmax is set, also bind "exp" so that softmax/SDPA
    # route through the patched exp.  This covers ViT/DeiT (SDPA),
    # Swin/Lambda (F.softmax), HaloNet/SeBotNet/CrossViT (Tensor.softmax).
    if args.patch_softmax and "exp" in registered and "exp" not in to_bind and cfg.has_exp_softmax:
        to_bind.append("exp")

    exp_note = ""
    if args.patch_softmax and cfg.has_exp_softmax:
        exp_note = " (+softmax/SDPA via exp)"
    elif args.patch_softmax and not cfg.has_exp_softmax:
        exp_note = " (no interceptable softmax)"
    print(f"  Binding: {to_bind}{exp_note}")

    model = timm.create_model(cfg.arch, pretrained=True).to(device).eval()
    criterion = nn.CrossEntropyLoss().to(device)

    import voyager_compiler.approx.app_binding as ab

    # When --patch-softmax, replace SDPA/softmax with manual impl for BOTH
    # baseline and test runs, so the only variable is float32 vs bf16/approx.
    if args.patch_softmax:
        ab.softmax_via_exp = True
        ab._bind_softmax_via_exp()

    if args.skip_baseline:
        print("\n--- Baseline (skipped) ---")
        base_top1, base_top5 = 0.0, 0.0
    else:
        print("\n--- Baseline (float) ---")
        base_top1, base_top5 = validate(val_loader, model, criterion, device, args)

    if args.mode in ("bf16", "bf16-clamp"):
        clamp = args.mode == "bf16-clamp"
        mode_label = f"{args.mode}: {to_bind}"
        print(f"\n--- {args.mode} ({to_bind}) ---")
        for fn in to_bind:
            if clamp:
                ab.bind_quantize(fn, "bf16", tag=args.tag, clamp=True)
            else:
                ab.bind_quantize(fn, "bf16")
        try:
            approx_top1, approx_top5 = validate(val_loader, model, criterion, device, args)
        finally:
            ab.unbind_all()
    else:
        mode_label = f"{args.tag}: {to_bind}"
        print(f"\n--- Approx ({args.tag}: {to_bind}) ---")
        for fn in to_bind:
            bind_approx(fn, args.tag)
        try:
            approx_top1, approx_top5 = validate(val_loader, model, criterion, device, args)
        finally:
            unbind_all()

    # unbind_all above also removed the softmax/SDPA patches; reset the flag
    ab.softmax_via_exp = False

    delta1 = approx_top1 - base_top1
    delta5 = approx_top5 - base_top5
    print(f"\nSummary [{cfg.arch}]  mode={args.mode}  bound={to_bind}")
    print(f"  Acc@1: {base_top1:.2f} -> {approx_top1:.2f}  ({delta1:+.2f})")
    print(f"  Acc@5: {base_top5:.2f} -> {approx_top5:.2f}  ({delta5:+.2f})")
    return {"arch": cfg.arch, "family": cfg.family, "mode": args.mode,
            "activations": "+".join(to_bind),
            "base_top1": base_top1, "approx_top1": approx_top1,
            "base_top5": base_top5, "approx_top5": approx_top5,
            "delta_top1": delta1, "delta_top5": delta5}


def resolve_model_configs(args) -> list:
    """Resolve CLI flags to a list of ModelConfig objects."""
    registry = {m.arch: m for m in MODELS}
    if args.all:
        return get_active_models()
    if args.family:
        cfgs = get_models_by_family(args.family)
        if not cfgs:
            raise ValueError(f"No active models for family '{args.family}'")
        return cfgs
    if args.arch:
        cfgs = []
        for arch in args.arch:
            if arch not in registry:
                raise ValueError(f"'{arch}' not found in model_registry.py")
            cfg = registry[arch]
            if cfg.skip:
                print(f"[warn] {arch} is marked skip: {cfg.skip_reason}")
            cfgs.append(cfg)
        return cfgs
    # default: resnet50
    return [registry["resnet50"]]


def main():
    args = parse_args()

    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    configs = resolve_model_configs(args)
    print(f"Device: {device}  |  Tag: {args.tag}  |  Models: {[c.arch for c in configs]}")

    results = []
    for cfg in configs:
        if cfg.arch in args.skip_models:
            print(f"\nSkipping {cfg.arch}")
            continue
        val_loader = build_val_loader(cfg.arch, args)
        result = bench_model(cfg, val_loader, device, args)
        if result is not None:
            results.append(result)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Final Summary")
        print(f"{'='*60}")
        print(f"{'Model':<35} {'Family':<14} {'Bound':<20} {'Base@1':>7} {'Approx@1':>9} {'Delta@1':>8}")
        for r in results:
            print(f"{r['arch']:<35} {r['family']:<14} {r['activations']:<20} "
                  f"{r['base_top1']:>7.2f} {r['approx_top1']:>9.2f} {r['delta_top1']:>+8.2f}")

    if args.output and results:
        import csv
        fields = ["arch", "family", "mode", "activations",
                  "base_top1", "approx_top1", "delta_top1",
                  "base_top5", "approx_top5", "delta_top5"]
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
