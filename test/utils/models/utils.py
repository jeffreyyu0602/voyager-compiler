from voyager_compiler.hardware import AcceleratorConfig


def get_transform_args(args, vector_stages):
    fuse_reshape = not args.disable_reshape_fusion and (
        args.bufferize
        or args.pe_array_size is None
        or max(args.pe_array_size) < 64
    )

    return {
        "patterns": vector_stages,
        "config": AcceleratorConfig.from_args(args),
        "transform_layout": args.transform_layout,
        "transpose_fc": args.transpose_fc,
        "fuse_reshape": fuse_reshape,
        "split_spmm": args.split_spmm,
        "use_interstellar_tiling": args.use_interstellar_tiling,
        "bufferize": args.bufferize,
    }


def get_compile_args(args):
    return {
        "config": AcceleratorConfig.from_args(args),
        "output_dir": args.model_output_dir,
        "output_file": args.model,
        "dump_tensors": args.dump_tensors,
        "bufferize": args.bufferize,
    }
