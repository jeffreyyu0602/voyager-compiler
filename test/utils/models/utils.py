import re


def compute_interstellar_dram(args):
    """Build the interstellar 4-level-DRAM arch/schedule and the DRAM bandwidth
    (in input elements per cycle) from the CLI args.

    Returns ``(interstellar_dram_arch, interstellar_dram_schedule,
    dram_bandwidth_elem_per_cycle)``.  The arch/schedule are ``None`` unless both
    ``--dram_size`` and ``--hardware_unrolling`` are set.
    """
    def _parse_bits(dtype_arg, default=8):
        # First integer in the dtype string (e.g. "nf4_6,..." -> 4); the typed
        # L0/L1 capacities are built in bytes from the model's (uniform)
        # activation/weight widths so they cancel against the byte-valued block
        # sizes in the feasibility check.
        if dtype_arg is None:
            return default
        m = re.search(r"(\d+)", str(dtype_arg).split(",")[0])
        return int(m.group(1)) if m else default

    # DRAM bandwidth GB/s -> input elements per cycle.  GB/s and GHz share the
    # 1e9 factor, so they cancel: elem/cycle = bw_GBs / (freq_GHz * elem_bytes).
    if args.dram_bandwidth > 0 and args.activation is not None:
        input_elem_bytes = _parse_bits(args.activation) / 8
        dram_bandwidth = int(args.dram_bandwidth / (args.frequency * input_elem_bytes))
    else:
        dram_bandwidth = int(args.dram_bandwidth)

    arch = schedule = None
    if args.dram_size is not None and args.hardware_unrolling is not None:
        from voyager_compiler.codegen.tiler import (
            build_architecture_and_schedule_with_dram,
        )

        arch, schedule = build_architecture_and_schedule_with_dram(
            ic_dim=args.hardware_unrolling[0],
            oc_dim=args.hardware_unrolling[1],
            l2_cache_size=args.cache_size * 2,
            input_buffer_size=args.input_buffer_size,
            weight_buffer_size=args.weight_buffer_size,
            accum_buffer_size=args.accum_buffer_size,
            dram_size=args.dram_size,
            double_buffered_l2=args.double_buffered_l2,
            input_dtype_bits=_parse_bits(args.activation),
            weight_dtype_bits=_parse_bits(args.weight),
        )
    return arch, schedule, dram_bandwidth


def get_transform_args(args, vector_stages):
    fuse_reshape = (
        not args.disable_reshape_fusion
        and (
            args.hardware_unrolling is None
            or max(args.hardware_unrolling) < 64
        )
    )
    arch, schedule, dram_bandwidth = compute_interstellar_dram(args)

    return {
        "patterns": vector_stages,
        "transform_layout": args.transform_layout,
        "transpose_fc": args.transpose_fc,
        "unroll_dims": args.hardware_unrolling,
        "cache_size": args.cache_size,
        "num_banks": args.num_banks,
        "fuse_reshape": fuse_reshape,
        "split_spmm": args.split_spmm,
        "double_buffered_accum_buffer": args.double_buffered_accum_buffer,
        "double_buffered_l2": args.double_buffered_l2,
        "interstellar_dram_arch": arch,
        "interstellar_dram_schedule": schedule,
        "dram_bandwidth": dram_bandwidth,
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
        "input_buffer_size": args.input_buffer_size,
        "weight_buffer_size": args.weight_buffer_size,
        "accum_buffer_size": args.accum_buffer_size,
        "double_buffered_accum_buffer": args.double_buffered_accum_buffer,
        "bufferize": args.bufferize,
    }
