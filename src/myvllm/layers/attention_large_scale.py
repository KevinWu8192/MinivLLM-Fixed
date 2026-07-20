import triton
import triton.language as tl
from myvllm.utils import get_context
import torch
import torch.nn as nn

@triton.jit
def store_kvcache_kernel(
    key_ptr, # pointer to what we want to store
    value_ptr,
    k_cache_ptr, # pointer to where we want to store
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr
):
    """
    Store keys and values into paged KV cache.
    Each token is mapped to a slot via slot_mapping.
    Grid layout: (num_tokens, num_kv_heads)
    Cache layout: (num_blocks, block_size, num_kv_heads, head_dim)
    """
    # thread ID, in dimension 0
    token_idx = tl.program_id(0) # each GPU thread processes one token
    # slot ID, where in cache to store this token
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    if slot_idx == -1:
        return

    # Calculate which block and position within block
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # Process each head
    # program_id(0) = which token
    # program_id(1) = which head
    head_idx = tl.program_id(1)

    # it creates a vector [0, 1, ..., head_dim-1]
    # Load key and value for this token and head
    head_offsets = tl.arange(0, head_dim)
    # Input: (num_tokens, num_kv_heads, head_dim)
    # example: input_offset = 5 * (8 * 128) + 3 * 128 + [0, 1, 2, ..., 127]
    #         = 5120 + 384 + [0, 1, 2, ..., 127]
    #         = [5504, 5505, 5506, ..., 5631]
    input_offset = (token_idx * num_kv_heads * head_dim + # skip previous tokens
                    head_idx * head_dim + # skip previous heads
                    head_offsets)

    # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim + # skip previous blocks
                   block_offset * num_kv_heads * head_dim + # skip previous positions in block
                   head_idx * head_dim + # skip previous heads
                   head_offsets)

    # load key and value value floats from the pointers's memory
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)

    # store into cache
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int
):
    """
    Store key-value pairs into paged cache.

    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - maps each token to a cache slot
        block_size: number of tokens per block
    """
    num_tokens, num_kv_heads, head_dim = key.shape

    # Make contiguous if needed
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"

    grid = (num_tokens, num_kv_heads)
    # launch num_tokens x num_kv_heads threads
    store_kvcache_kernel[grid](
        key, # tensors are automatically converted to pointers by triton
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size
    )


@triton.jit
def flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    USE_PAGED_KV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel for variable-length sequences.
    Each program processes one block of queries for one head in one sequence.
    """
    # Program IDs
    start_m = tl.program_id(0) # block index
    off_h = tl.program_id(1) # head index
    seq_idx = tl.program_id(2) # sequence index

    # Determine which KV head to use (for GQA)
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # Load sequence boundaries
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len_q = seq_end - seq_start
    seq_start_k = tl.load(cu_seqlens_k_ptr + seq_idx)
    seq_end_k = tl.load(cu_seqlens_k_ptr + seq_idx + 1)
    seq_len_k = seq_end_k - seq_start_k
    # Prefix-cache hits remove full prefix blocks from Q, while K/V still
    # represent the full sequence through the paged KV cache.
    num_cached_tokens = seq_len_k - seq_len_q

    # Early exit if this block is beyond sequence length
    if start_m * BLOCK_M >= seq_len_q:
        return

    # Offset for this block of queries
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    # Query pointers: Q has shape (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]

    # Load Q block - shape (BLOCK_M, head_dim)
    mask_m = offs_m < seq_len_q
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize output accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # Number of blocks to process
    num_blocks = tl.cdiv(seq_len_k, BLOCK_N)

    # Loop over K, V blocks
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Mask for valid positions
        mask_n = offs_n < seq_len_k

        if USE_PAGED_KV:
            # Prefix hit: current K/V were stored into the paged cache before
            # this kernel launch, so both the cached prefix and new suffix can
            # be read through the request's block table.
            logical_block_idx = offs_n // block_size
            block_offset = offs_n % block_size
            block_ids = tl.load(
                block_tables_ptr
                + seq_idx * max_num_blocks
                + logical_block_idx,
                mask=mask_n,
                other=0,
            )
            cache_offsets = (
                block_ids * block_size * num_kv_heads * head_dim
                + block_offset * num_kv_heads * head_dim
                + kv_head_idx * head_dim
            )
            k_ptrs = k_cache_ptr + cache_offsets[None, :] + offs_d[:, None]
        else:
            # No prefix hit: K is contiguous and has the same sequence layout
            # as Q.
            k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]

        # Load K block - shape (head_dim, BLOCK_N)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # Compute QK^T - shape (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * scale

        # Apply causal mask: only attend to positions <= current position
        # offs_m is relative to the uncached suffix, while offs_n is relative
        # to the full context. Convert Q positions to full-context positions.
        mask_causal = (num_cached_tokens + offs_m[:, None]) >= offs_n[None, :]
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        # Rescale previous accumulator
        acc = acc * alpha[:, None]

        # Load V block - shape (BLOCK_N, head_dim)
        if USE_PAGED_KV:
            v_ptrs = v_cache_ptr + cache_offsets[:, None] + offs_d[None, :]
        else:
            v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Accumulate weighted values
        acc = acc + tl.dot(p.to(v.dtype), v)

        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output: O has shape (total_tokens, num_heads, head_dim)
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor | None,
    block_size: int,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Optimized Flash Attention for prefill phase with variable-length sequences.

    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens_q: cumulative lengths of newly computed query tokens
        cu_seqlens_k: cumulative lengths of the full KV context
        scale: attention scale factor

    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Allocate output
    output = torch.empty_like(q)

    # Conservative block sizes to avoid OOM on shared memory
    # Shared memory usage ~ BLOCK_M * BLOCK_N * 4 bytes (for float32 attention scores)
    # + BLOCK_M * head_dim * 4 (for Q)
    # + BLOCK_N * head_dim * 4 (for K, V)
    # Want to keep total < 48KB for most GPUs

    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    # Number of sequences
    num_seqs = cu_seqlens_q.shape[0] - 1

    # Find max sequence length to determine grid size
    cu_seqlens_cpu = cu_seqlens_q.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()

    # Calculate grid dimensions - launch all kernels at once
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)

    use_paged_kv = block_tables is not None
    # Triton still requires pointer arguments for constexpr-disabled branches.
    # These placeholders are never dereferenced when USE_PAGED_KV is False.
    block_tables_ptr = block_tables if use_paged_kv else cu_seqlens_q
    k_cache_ptr = k_cache if use_paged_kv else k
    v_cache_ptr = v_cache if use_paged_kv else v
    max_num_blocks = block_tables.shape[1] if use_paged_kv else 1

    flash_attention_varlen_kernel[grid](
        q, k, v, output,
        cu_seqlens_q,
        cu_seqlens_k,
        k_cache_ptr,
        v_cache_ptr,
        block_tables_ptr,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        USE_PAGED_KV=use_paged_kv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output


@triton.jit
def paged_attention_decode_stage1_kernel(
    output_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    num_splits: tl.constexpr,
    q_per_kv: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_KV: tl.constexpr,
):
    """
    First stage of split-KV decode attention.

    Each program owns one ``(batch, kv_head, context_split)`` tuple.  All GQA
    query heads which share that KV head are processed together so that a K/V
    tile is loaded only once and both QK^T and PV can use tl.dot.

    With one context split, the program normalizes ``acc`` directly into the
    final output.  With multiple splits, it stores ``(m_i, l_i, acc)`` for the
    reduction kernel.
    """
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    context_len = tl.load(context_lens_ptr + batch_idx)
    # each tile is BLOCK_N = 32
    num_n_tiles = tl.cdiv(context_len, BLOCK_N)
    # how many tiles should each block (triton program) process
    tiles_per_split = tl.cdiv(num_n_tiles, num_splits)
    split_tile_start = split_idx * tiles_per_split
    split_tile_end = tl.minimum(split_tile_start + tiles_per_split, num_n_tiles)
    offs_d = tl.arange(0, head_dim)

    offs_h = tl.arange(0, BLOCK_H)
    # 16 heads in total with 8 of them are paddings
    head_idx = kv_head_idx * q_per_kv + offs_h
    # 8 padded heads will be masked
    mask_h = offs_h < q_per_kv
    q_offsets = (
        batch_idx * num_heads * head_dim
        + head_idx[:, None] * head_dim
        + offs_d[None, :]
    )
    q = tl.load(
        query_ptr + q_offsets,
        mask=mask_h[:, None],
        other=0.0,
    )
    # flash attention for 8 heads using tiled gemm
    m_i = tl.full([BLOCK_H], -1.0e20, tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, head_dim], tl.float32)

    # Empty splits retain (m=sentinel, l=0, acc=0) and stage 2 ignores them.
    if split_tile_end > split_tile_start:
        for tile_idx in tl.range(split_tile_start, split_tile_end):
            # tokens which need to be processed in this split tile
            offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len
            # load paged kv-caches in different physical blocks at once
            logical_block_idx = offs_n // block_size
            block_offset = offs_n % block_size
            physical_block_idx = tl.load(
                block_tables_ptr
                + batch_idx * max_num_blocks
                + logical_block_idx,
                mask=mask_n & (logical_block_idx < max_num_blocks),
                other=0,
            )
            cache_row_offsets = (
                physical_block_idx * block_size * num_kv_heads * head_dim
                + block_offset * num_kv_heads * head_dim
                + kv_head_idx * head_dim
            )

            k = tl.load(
                k_cache_ptr
                + cache_row_offsets[None, :]
                + offs_d[:, None],
                mask=mask_n[None, :],
                other=0.0,
            )
            qk = tl.dot(q, k) * scale
            qk = tl.where(mask_n[None, :], qk, -1.0e20)
            m_ij = tl.max(qk, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])

            v = tl.load(
                v_cache_ptr
                + cache_row_offsets[:, None]
                + offs_d[None, :],
                mask=mask_n[:, None],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_i_new

    # num_splits > 1: collect partial result, wait for reduce
    if SPLIT_KV:
        partial_offsets = (
          batch_idx * num_heads * num_splits # skip previous batches
          # skip previous kv heads (1 kv head corresponds to 16 q heads)
          + head_idx * num_splits
          # skip prevoius split (blocks a.k.a triton programs)
          + split_idx
      )
        tl.store(partial_m_ptr + partial_offsets, m_i, mask=mask_h)
        tl.store(partial_l_ptr + partial_offsets, l_i, mask=mask_h)
        partial_acc_offsets = partial_offsets[:, None] * head_dim + offs_d[None, :]
        tl.store(
            partial_acc_ptr + partial_acc_offsets,
            acc,
            mask=mask_h[:, None],
        )
    # num_splits == 1: return final result
    else:
        output_offsets = (
            batch_idx * num_heads * head_dim
            + head_idx[:, None] * head_dim
            + offs_d[None, :]
        )
        # avoid CUDA graph padding
        denominator = tl.where(l_i > 0.0, l_i, 1.0)
        tl.store(
            output_ptr + output_offsets,
            acc / denominator[:, None],
            mask=mask_h[:, None],
        )


@triton.jit
def paged_attention_decode_reduce_kernel(
    output_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """
    Merge split-local online-softmax states.

    For split ``s``, stage 1 produces ``(m_s, l_s, acc_s)``.  The final
    online-softmax state is reconstructed as::

        m = max_s(m_s)
        scale_s = exp(m_s - m)
        l = sum_s(scale_s * l_s)
        acc = sum_s(scale_s * acc_s)
        output = acc / l

    Empty splits have ``l_s == 0`` and contribute nothing.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, head_dim)
    mask_s = offs_s < num_splits
    partial_offsets = (
        (batch_idx * num_heads + head_idx) * num_splits + offs_s
    )

    partial_l = tl.load(
        partial_l_ptr + partial_offsets, mask=mask_s, other=0.0
    )
    valid = mask_s & (partial_l > 0.0)
    partial_m = tl.load(
        partial_m_ptr + partial_offsets, mask=valid, other=-1.0e20
    )
    global_m = tl.max(partial_m, axis=0)

    # Empty splits use a finite sentinel so the subtraction is always defined.
    safe_m = tl.where(valid, partial_m, global_m)
    correction = tl.exp(safe_m - global_m)
    correction = tl.where(valid, correction, 0.0)
    global_l = tl.sum(correction * partial_l, axis=0)

    partial_acc = tl.load(
        partial_acc_ptr
        + partial_offsets[:, None] * head_dim
        + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    global_acc = tl.sum(correction[:, None] * partial_acc, axis=0)
    denominator = tl.where(global_l > 0.0, global_l, 1.0)
    output = global_acc / denominator
    output_offsets = (
        batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    )
    tl.store(output_ptr + output_offsets, output)


def _choose_decode_num_splits(
    batch_size: int,
    num_kv_heads: int,
    max_context_tokens: int,
    block_n: int,
) -> int:
    """Choose a fixed, CUDA-graph-compatible split count for decode."""
    max_num_n_tiles = triton.cdiv(max_context_tokens, block_n)
    # no more than 128 blocks (triton programs) in total
    splits_for_parallelism = triton.cdiv(
        128, batch_size * num_kv_heads
    )
    # no more than 32 blocks (triton programs) for N direction parallelism
    return max(
        1,
        min(32, max_num_n_tiles, splits_for_parallelism),
    )


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    """
    Compute attention in decode mode using paged KV cache.

    Args:
        query: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks)
        context_lens: (batch_size,)
        scale: attention scale factor

    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]

    # Make contiguous
    query = query.contiguous()

    output = torch.empty_like(query)

    if num_heads <= num_kv_heads or num_heads % num_kv_heads != 0:
        raise ValueError(
            "GQA decode requires num_heads to be greater than and divisible "
            "by num_kv_heads"
        )

    # Split the maximum logical context represented by this block table.  The
    # actual context length is masked in the kernel.  A fixed, shape-derived
    # split count also keeps this path compatible with CUDA graph capture.
    BLOCK_N = 32
    max_context_tokens = max_num_blocks * block_size
    num_splits = _choose_decode_num_splits(
        batch_size,
        num_kv_heads,
        max_context_tokens,
        BLOCK_N,
    )
    q_per_kv = num_heads // num_kv_heads
    # actual BLOCK_H should be 8 (each kv head will be used by 8 q heads)
    # we pad it to 16 since mma atom needs at least 16 rows in A matrix
    BLOCK_H = 16
    # tl.arange needs input of power of 2
    BLOCK_S = triton.next_power_of_2(num_splits)

    if num_splits > 1:
        # Shape order keeps splits adjacent for the reduction kernel.
        partial_acc = torch.empty(
            batch_size,
            num_heads,
            num_splits,
            head_dim,
            device=query.device,
            dtype=torch.float32,
        )
        partial_m = torch.empty(
            batch_size,
            num_heads,
            num_splits,
            device=query.device,
            dtype=torch.float32,
        )
        partial_l = torch.empty_like(partial_m)
    else:
        # These pointers are compile-time unused when SPLIT_KV is false.
        # if num_splits == 1, only stage 1 is needed
        partial_acc = output
        partial_m = output
        partial_l = output

    stage1_grid = (batch_size, num_kv_heads, num_splits)
    paged_attention_decode_stage1_kernel[stage1_grid](
        output,
        partial_acc,
        partial_m,
        partial_l,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        num_splits=num_splits,
        q_per_kv=q_per_kv,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        SPLIT_KV=num_splits > 1,
        num_warps=4,
        num_stages=2,
    )

    if num_splits > 1:
        reduce_grid = (batch_size, num_heads)
        paged_attention_decode_reduce_kernel[reduce_grid](
            output,
            partial_acc,
            partial_m,
            partial_l,
            num_heads=num_heads,
            head_dim=head_dim,
            num_splits=num_splits,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )

    return output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,
        block_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Store current k, v into cache if cache is allocated
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            # Ensure k, v are in the right shape: (num_tokens, num_kv_heads, head_dim)
            if k.dim() == 4:
                # Batched: (B, N, num_kv_heads, head_dim) -> reshape to (B*N, num_kv_heads, head_dim)
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                # Already in correct shape (num_tokens, num_kv_heads, head_dim)
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()

            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        scale = self.scale / (self.head_dim ** 0.5)

        if context.is_prefill:
            # Prefill: use flash attention
            # Varlen mode: (total_tokens, num_heads, head_dim)
            cu_seqlens_q = context.cu_seqlens_q
            cu_seqlens_k = context.cu_seqlens_k
            if cu_seqlens_q is None or cu_seqlens_k is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")

            o = flash_attention_prefill(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                k_cache,
                v_cache,
                context.block_tables,
                self.block_size,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
            # Output: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            o = paged_attention_decode(
                q,
                k_cache,
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size
            )
            # o: (batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    # Example usage
    layer = Attention(num_heads=8, head_dim=64).cuda()
    B, N, D = 4, 1024, 512
    q = torch.randn(B, N, D).cuda()
    k = torch.randn(B, N, D).cuda()
    v = torch.randn(B, N, D).cuda()
    layer.k_cache = torch.zeros(B, N, D).cuda()
    layer.v_cache = torch.zeros(B, N, D).cuda()
    slot_mapping = torch.arange(N).cuda()

    for _ in range(10):  # Warm-up iterations
        _ = layer(q, k, v)

    import time
    times = []
    for _ in range(100):  # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
