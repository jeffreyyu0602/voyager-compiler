from voyager_compiler.hardware_config import AcceleratorConfig


def _config(args):
    """The accelerator the compiler plans against.

    A target's profile carries values ``args`` cannot: capabilities like
    ``partial_matrix_input_channels`` are facts about the datapath, not CLI
    knobs, so they exist only on the config the target builds. Going through
    the target here means a workload adapter plans against the declared
    machine rather than a config reassembled from flags -- which silently drops
    every capability the target declares.
    """
    target = getattr(args, "target", None)
    if target is not None:
        from utils import targets

        return targets.resolve(target).accelerator_config(args)
    return AcceleratorConfig.from_args(args)


def get_transform_args(args, vector_stages):
    return {
        "patterns": vector_stages,
        "config": _config(args),
        "layout_policy": args.layout_policy,
        "gemv_weight_layout": args.gemv_weight_layout,
        "fuse_reshape": not args.disable_reshape_fusion,
        "target": getattr(args, "target", None),
        "rename_fused_operands": getattr(
            args, "rename_fused_operands", False
        ),
    }


def get_compile_args(args):
    return {
        "config": _config(args),
        "output_dir": args.model_output_dir,
        "output_file": args.model,
        "dump_tensors": args.dump_tensors,
        "runtime_tolerance": args.runtime_tolerance,
        "target": getattr(args, "target", None),
        "local_scheduler": getattr(args, "local_scheduler", None),
        "scratchpad_mapping": getattr(args, "scratchpad_mapping", None),
        "execution_order": getattr(args, "execution_order", None),
    }
