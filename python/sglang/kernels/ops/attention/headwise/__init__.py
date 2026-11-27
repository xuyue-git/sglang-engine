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
each one's origin and sha256; LICENSE is vLLM's Apache-2.0) and the Python bindings. The
extension is compiled on first use with torch's C++ extension loader (nvcc on PATH,
-O2 -std=c++20, for the visible GPUs or TORCH_CUDA_ARCH_LIST) and cached; build it ahead
of time with ``python -m sglang.kernels.ops.attention.headwise``.
"""

from functools import lru_cache
from pathlib import Path

_CSRC = Path(__file__).resolve().parent / "csrc"
_SOURCES = ("bindings.cpp", "cache_headwise.cu", "attention_headwise_kernels.cu")
_FLAGS = ["-O2", "-std=c++20"]


@lru_cache(maxsize=None)
def _ops():
    from torch.utils.cpp_extension import load

    return load(
        name="sglang_muxserve_headwise",
        sources=[str(_CSRC / name) for name in _SOURCES],
        extra_include_paths=[str(_CSRC)],
        extra_cflags=_FLAGS,
        extra_cuda_cflags=_FLAGS,
    )


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


if __name__ == "__main__":
    print(f"built {_ops().__file__}")
