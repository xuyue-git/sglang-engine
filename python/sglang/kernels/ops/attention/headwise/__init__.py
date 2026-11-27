"""MuxServe's head-wise KV cache ops (from MuxServe-vLLM eaca02e), for the MuxServe runtimes.

MuxServe's FlexStore keeps one KV pool per GPU for all the models it serves and cuts it
"head-wise": a block holds one layer's one attention head for ``block_size`` tokens, so a
block fits a model of any depth or head count. SGLang's attention backends do not read
that layout. These two ops do, with MuxServe-vLLM's arguments:

    headwise_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
        store this step's K and V in the head-wise blocks given by slot_mapping
        (tokens x layers x heads, one slot per token, layer and head)
    single_query_cached_kv_headwise_attention(out, query, key_cache, value_cache,
            head_mapping, scale, block_tables, context_lens, block_size,
            max_context_len, alibi_slopes)
        decode attention of one query token per sequence over its head-wise blocks

``csrc/`` holds the sources: MuxServe-vLLM's files unchanged (SOURCE_MANIFEST.json gives
each one's origin and sha256; LICENSE is vLLM's Apache-2.0) and the Python bindings. As
MuxServe-vLLM's install compiles its kernels, they are compiled ahead of time, once, into
the extension ``_C`` next to this file:

    python -m sglang.kernels.ops.attention.headwise

(nvcc on PATH; -O2 -std=c++20; TORCH_CUDA_ARCH_LIST, or else 8.0 for the A100). Importing
this package loads that extension, so nothing is compiled or locked while a server runs.
"""

import importlib

import torch  # noqa: F401  (the extension links against torch's libraries)

try:
    _C = importlib.import_module(f"{__name__}._C")
except ImportError as error:  # not built yet
    _C, _NOT_BUILT = None, error


def _ops():
    if _C is None:
        raise RuntimeError(
            "MuxServe's head-wise ops are not built: build them once with "
            f"`python -m sglang.kernels.ops.attention.headwise` ({_NOT_BUILT})"
        )
    return _C


def headwise_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    _ops().headwise_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)


def single_query_cached_kv_headwise_attention(
    out,
    query,
    key_cache,
    value_cache,
    head_mapping,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    _ops().single_query_cached_kv_headwise_attention(
        out,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
    )

