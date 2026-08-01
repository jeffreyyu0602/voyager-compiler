from voyager_compiler.hardware_config import AcceleratorConfig


def get_transform_args(args, vector_stages):
    return {
        "patterns": vector_stages,
        "config": AcceleratorConfig.from_args(args),
        "layout_policy": args.layout_policy,
        "gemv_weight_layout": args.gemv_weight_layout,
        "fuse_reshape": not args.disable_reshape_fusion,
    }


def get_compile_args(args):
    return {
        "config": AcceleratorConfig.from_args(args),
        "output_dir": args.model_output_dir,
        "output_file": args.model,
        "dump_tensors": args.dump_tensors,
        "runtime_tolerance": args.runtime_tolerance,
    }
