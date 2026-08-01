"""Hardware lowering: the ``transform()`` passes and the bufferized backend.

Every pass ``transform()`` runs is exported from here, in the order the
pipeline applies them.
"""

from voyager_compiler.codegen.subgraph import rename_nodes_with_param_names
from voyager_compiler.codegen.transform.bufferize import gen_compute_graph
from voyager_compiler.codegen.transform.data_layout import (
    eliminate_reshape_with_no_effect,
    normalize_conv2d_layout,
    normalize_gemm_weight_layout,
)
from voyager_compiler.codegen.transform.operator_fusion import fuse_operator
from voyager_compiler.codegen.transform.padding import (
    pad_matrix_op_dimensions,
    pad_vector_op_dimensions,
    pad_vit_embeddings_output,
)
from voyager_compiler.codegen.transform.quant_folding import (
    fuse_dequantize_quantize,
    fuse_quantize_dequantize_with_producer,
)
from voyager_compiler.codegen.transform.rewrites import (
    deduplicate_nodes,
    extract_input_preprocessor,
    fold_constant_generators,
    inline_autocast_modules,
    remove_prunable_ops,
    remove_softmax_dtype_cast,
    remove_zero_attention_mask,
    replace_conv2d_with_im2col,
    replace_interpolate,
    replace_rmsnorm_with_layer_norm,
)
from voyager_compiler.codegen.transform.tiling.search import (
    gemv_op_tiling,
    pool_op_tiling,
    vector_op_tiling,
)

__all__ = [
    "deduplicate_nodes",
    "eliminate_reshape_with_no_effect",
    "extract_input_preprocessor",
    "fold_constant_generators",
    "fuse_dequantize_quantize",
    "fuse_operator",
    "fuse_quantize_dequantize_with_producer",
    "gen_compute_graph",
    "inline_autocast_modules",
    "normalize_conv2d_layout",
    "normalize_gemm_weight_layout",
    "pad_matrix_op_dimensions",
    "pad_vector_op_dimensions",
    "pad_vit_embeddings_output",
    "remove_prunable_ops",
    "remove_softmax_dtype_cast",
    "remove_zero_attention_mask",
    "rename_nodes_with_param_names",
    "replace_conv2d_with_im2col",
    "replace_interpolate",
    "replace_rmsnorm_with_layer_norm",
    "gemv_op_tiling",
    "vector_op_tiling",
    "pool_op_tiling",
]
