def get_transform_args(args, vector_stages):
    fuse_reshape = (
        not args.disable_reshape_fusion
        and (
            args.hardware_unrolling is None
            or max(args.hardware_unrolling) < 64
        )
    )

    return {
        "patterns": vector_stages,
        "transform_layout": args.transform_layout,
        "transpose_fc": args.transpose_fc,
        "unroll_dims": args.hardware_unrolling,
        "cache_size": args.cache_size,
        "num_banks": args.num_banks,
        "fuse_reshape": fuse_reshape,
    }


def get_compile_args(args):
    return {
        "cache_size": args.cache_size,
        "num_banks": args.num_banks,
        "bank_width": args.bank_width,
        "unroll_dims": args.hardware_unrolling,
        "output_dir": args.model_output_dir,
        "output_file": args.model,
        "dump_tensors": args.dump_tensors,
    }
