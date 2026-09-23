import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    triton = None


@torch.compile
def pytorch_forward(Q, K, V, is_causal):
    """Dense attention: materialize the full query-by-key score matrix."""
    S = (Q @ K.transpose(-2, -1)) * (Q.shape[-1] ** -0.5)
    if is_causal:
        query_rows = torch.arange(Q.shape[-2], device=Q.device)
        key_rows = torch.arange(K.shape[-2], device=K.device)
        S = S.masked_fill(query_rows[:, None] < key_rows[None, :], float("-inf"))
    L = torch.logsumexp(S, dim=-1)
    O = torch.softmax(S, dim=-1) @ V
    return O, L


@torch.compile
def pytorch_backward(Q, K, V, O, dO, L, is_causal):
    d = Q.shape[-1]

    D = (O * dO).sum(-1)
    S = Q @ K.transpose(-2, -1) / (d ** 0.5)


    if is_causal:
        q_indexes = torch.arange(Q.shape[-2], device=Q.device)
        k_indexes = torch.arange(K.shape[-2], device=K.device)
        # valid = S.new_ones(S.shape)
        valid = (q_indexes[..., None] >= k_indexes[..., None, :])
        S = torch.where(valid, S, float("-inf"))

    P = torch.exp(S - L.unsqueeze(-1))
    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)
    dS = P * (dP - D.unsqueeze(-1))
    dQ = dS @ K / (d ** 0.5)
    dK = dS.transpose(-2, -1) @ Q / (d ** 0.5)
    return dQ, dK, dV

def split_into_tiles(x, tile_size=16):
    """Split the sequence axis while preserving all features and batch axes."""
    return x.split(tile_size, dim=-2)


class NaiveAttention(torch.autograd.Function):
    """Dense compiled forward and backward, with quadratic intermediates."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, L = pytorch_forward(Q, K, V, is_causal)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = pytorch_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None


def tiled_pytorch_backward(Q, K, V, O, dO, L, is_causal, tile_size=16):
    """Two tiled sweeps, matching the ownership of the Triton kernels.

    Only tile-sized score/probability matrices are materialized. PyTorch
    executes the outer loops sequentially; Triton parallelizes those tiles.
    """
    scale = Q.shape[-1] ** -0.5
    D_vec = (O * dO).sum(-1)
    dQ, dK, dV = torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V)

    def recompute_tile(q_start, k_start):
        Qi = Q[..., q_start:q_start + tile_size, :]
        Kj = K[..., k_start:k_start + tile_size, :]
        Vj = V[..., k_start:k_start + tile_size, :]
        dOi = dO[..., q_start:q_start + tile_size, :]
        Li = L[..., q_start:q_start + tile_size]
        Di = D_vec[..., q_start:q_start + tile_size]
        Sij = (Qi @ Kj.transpose(-2, -1)) * scale
        if is_causal:
            query_rows = torch.arange(q_start, q_start + Qi.shape[-2], device=Q.device)
            key_rows = torch.arange(k_start, k_start + Kj.shape[-2], device=K.device)
            Sij = Sij.masked_fill(query_rows[:, None] < key_rows[None, :], float("-inf"))
        Pij = torch.exp(Sij - Li.unsqueeze(-1))
        dPij = dOi @ Vj.transpose(-2, -1)
        dSij = Pij * (dPij - Di.unsqueeze(-1))
        return Qi, Kj, dOi, Pij, dSij

    # Like flash_bwd_kernel1: each query tile owns its complete dQ sum.
    for q_start in range(0, Q.shape[-2], tile_size):
        dQi = torch.zeros_like(Q[..., q_start:q_start + tile_size, :])
        for k_start in range(0, K.shape[-2], tile_size):
            _, Kj, _, _, dSij = recompute_tile(q_start, k_start)
            dQi = dQi + (dSij @ Kj) * scale
        dQ[..., q_start:q_start + tile_size, :] = dQi

    # Like flash_bwd_kernel2: each key tile owns its complete dK/dV sums.
    for k_start in range(0, K.shape[-2], tile_size):
        dKj = torch.zeros_like(K[..., k_start:k_start + tile_size, :])
        dVj = torch.zeros_like(V[..., k_start:k_start + tile_size, :])
        for q_start in range(0, Q.shape[-2], tile_size):
            Qi, _, dOi, Pij, dSij = recompute_tile(q_start, k_start)
            dKj = dKj + (dSij.transpose(-2, -1) @ Qi) * scale
            dVj = dVj + Pij.transpose(-2, -1) @ dOi
        dK[..., k_start:k_start + tile_size, :] = dKj
        dV[..., k_start:k_start + tile_size, :] = dVj

    return dQ, dK, dV


def tiled_pytorch_forward(Q, K, V, is_causal=False, tile_size=16):
    """Online-softmax tiled forward shared by eager and compiled variants."""
    d = Q.shape[-1]

    Q_tiles = split_into_tiles(Q, tile_size)
    K_tiles = split_into_tiles(K, tile_size)
    V_tiles = split_into_tiles(V, tile_size)

    O = torch.empty_like(Q)
    L = Q.new_empty(Q.shape[:-1])

    for i, Q_i in enumerate(Q_tiles):
        O_i = torch.zeros_like(Q_i)
        l = Q_i.new_zeros(Q_i.shape[:-1])
        m_prev = Q_i.new_full(Q_i.shape[:-1], float("-inf"))

        q_indexes = torch.arange(i * tile_size, i * tile_size + Q_i.shape[-2], device=Q.device)

        for j, (K_j, V_j) in enumerate(zip(K_tiles, V_tiles)):
            S = (Q_i @ K_j.transpose(-2, -1)) / (d ** 0.5)

            k_indexes = torch.arange(j * tile_size, j * tile_size + K_j.shape[-2], device=K.device)

            if is_causal:
                # valid = S.new_ones(S.shape)
                valid = (q_indexes[:, None] >= k_indexes[None, :])
                S = torch.where(valid, S, float("-inf"))

            m = torch.maximum(m_prev, S.max(dim=-1).values)
            P = torch.exp(S - m.unsqueeze(-1))

            # Rescale earlier contributions to the new running maximum.
            alpha = torch.exp(m_prev - m)
            l = alpha * l + P.sum(dim=-1)
            O_i = alpha.unsqueeze(-1) * O_i + P @ V_j
            m_prev = m

        start = i * tile_size
        end = start + Q_i.shape[-2]
        O[..., start:end, :] = O_i / l.unsqueeze(-1)
        L[..., start:end] = m_prev + torch.log(l)

    return O, L


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, tile_size=16):
        if not isinstance(tile_size, int) or tile_size < 1:
            raise ValueError("tile_size must be a positive integer.")
        ctx.tile_size = tile_size
        output, L = tiled_pytorch_forward(Q, K, V, is_causal, tile_size)

        ctx.save_for_backward(L, Q, K, V, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = tiled_pytorch_backward(Q, K, V, O, dO, L, ctx.is_causal, ctx.tile_size)
        return (dQ, dK, dV, None, None)[:len(ctx.needs_input_grad)]

# Compile the complete tiled loops, including the manually derived backward.
# Full-graph capture makes unsupported operations fail instead of running eager
# fragments under a compiled label. Shapes and tile sizes specialize per case.
compiled_tiled_pytorch_forward = torch.compile(
    tiled_pytorch_forward, fullgraph=True, dynamic=False, backend="inductor",
)
compiled_tiled_pytorch_backward = torch.compile(
    tiled_pytorch_backward, fullgraph=True, dynamic=False, backend="inductor",
)


class CompiledFlashAttention(torch.autograd.Function):
    """The same tiled algorithm as FlashAttention, compiled in both directions."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, tile_size=16):
        if not isinstance(tile_size, int) or tile_size < 1:
            raise ValueError("tile_size must be a positive integer.")
        output, L = compiled_tiled_pytorch_forward(Q, K, V, is_causal, tile_size)
        ctx.save_for_backward(L, Q, K, V, output)
        ctx.is_causal = is_causal
        ctx.tile_size = tile_size
        return output

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, output = ctx.saved_tensors
        dQ, dK, dV = compiled_tiled_pytorch_backward(
            Q, K, V, output, dO, L, ctx.is_causal, ctx.tile_size,
        )
        return (dQ, dK, dV, None, None)[:len(ctx.needs_input_grad)]


# Keep the PyTorch implementation importable on systems without Triton.
if triton is not None:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        # Program indices
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Offset each pointer with the corresponding batch index
        # multiplied with the batch stride for each tensor
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        mi = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        query_rows = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        key_offsets = tl.arange(0, K_TILE_SIZE)

        for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # Each score tile compares complete query and key feature vectors.
            Sij = tl.dot(Qi, tl.trans(Kj), input_precision="ieee") * scale
            key_rows = j * K_TILE_SIZE + key_offsets
            valid = key_rows[None, :] < N_KEYS
            if IS_CAUSAL:
                valid = valid & (query_rows[:, None] >= key_rows[None, :])
            Sij = tl.where(valid, Sij, float("-inf"))

            mi_new = tl.maximum(mi, tl.max(Sij, axis=1))
            Pij = tl.exp(Sij - mi_new[:, None])
            alpha = tl.exp(mi - mi_new)
            li = alpha * li + tl.sum(Pij, axis=1)
            Oi = tl.dot(
                Pij.to(Vj.dtype), Vj,
                acc=alpha[:, None] * Oi,
                input_precision="ieee",
            )

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
            mi = mi_new

        Oi = Oi / li[:, None]
        Li = mi + tl.log(li)

        tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, Li, boundary_check=(0,))


    @triton.jit
    def flash_bwd_kernel1(
        Q_ptr, K_ptr, V_ptr,
        dO_ptr, L_ptr, D_vec_ptr,
        dQ_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_dob, stride_doq, stride_dod,
        stride_lb, stride_lq,
        stride_Dvec_b, stride_Dvec_q,
        stride_dqb, stride_dqq, stride_dqd,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D),
            strides=(stride_doq, stride_dod),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        D_block_ptr = tl.make_block_ptr(
            D_vec_ptr + batch_index * stride_Dvec_b,
            shape=(N_QUERIES,),
            strides=(stride_Dvec_q,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        dQ_block_ptr = tl.make_block_ptr(
            dQ_ptr + batch_index * stride_dqb,
            shape=(N_QUERIES, D),
            strides=(stride_dqq, stride_dqd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

        dQi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        query_rows = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        key_offsets = tl.arange(0, K_TILE_SIZE)

        for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            Sij = tl.dot(Qi, tl.trans(Kj), input_precision="ieee") * scale
            key_rows = j * K_TILE_SIZE + key_offsets
            valid = (query_rows[:, None] < N_QUERIES) & (key_rows[None, :] < N_KEYS)
            if IS_CAUSAL:
                valid = valid & (query_rows[:, None] >= key_rows[None, :])
            Sij = tl.where(valid, Sij, float("-inf"))

            Pij = tl.exp(Sij - Li[:, None])

            dPij = tl.dot(dOi, tl.trans(Vj), input_precision="ieee")
            dSij = Pij * (dPij - Di[:, None])
            dQi = dQi + tl.dot(dSij.to(Kj.dtype), Kj, input_precision="ieee") * scale

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        tl.store(dQ_block_ptr, dQi.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))

    @triton.jit
    def flash_bwd_kernel2(
        Q_ptr, K_ptr, V_ptr,
        dO_ptr, L_ptr, D_vec_ptr,
        dK_ptr, dV_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_dob, stride_doq, stride_dod,
        stride_lb, stride_lq,
        stride_Dvec_b, stride_Dvec_q,
        stride_dkb, stride_dkk, stride_dkd,
        stride_dvb, stride_dvk, stride_dvd,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        key_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(0, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D),
            strides=(stride_doq, stride_dod),
            offsets=(0, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(0,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        D_block_ptr = tl.make_block_ptr(
            D_vec_ptr + batch_index * stride_Dvec_b,
            shape=(N_QUERIES,),
            strides=(stride_Dvec_q,),
            offsets=(0,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        dK_block_ptr = tl.make_block_ptr(
            dK_ptr + batch_index * stride_dkb,
            shape=(N_KEYS, D),
            strides=(stride_dkk, stride_dkd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        dV_block_ptr = tl.make_block_ptr(
            dV_ptr + batch_index * stride_dvb,
            shape=(N_KEYS, D),
            strides=(stride_dvk, stride_dvd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        dKj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        dVj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

        key_rows = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        query_offsets = tl.arange(0, Q_TILE_SIZE)

        for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
            Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
            Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
            Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

            Sij = tl.dot(Qi, tl.trans(Kj), input_precision="ieee") * scale
            query_rows = i * Q_TILE_SIZE + query_offsets
            valid = (query_rows[:, None] < N_QUERIES) & (key_rows[None, :] < N_KEYS)
            if IS_CAUSAL:
                valid = valid & (query_rows[:, None] >= key_rows[None, :])
            Sij = tl.where(valid, Sij, float("-inf"))

            Pij = tl.exp(Sij - Li[:, None])

            dVj = dVj + tl.dot(tl.trans(Pij).to(dOi.dtype), dOi, input_precision="ieee")
            dPij = tl.dot(dOi, tl.trans(Vj), input_precision="ieee")

            dSij = Pij * (dPij - Di[:, None])

            dKj = dKj + tl.dot(tl.trans(dSij).to(Qi.dtype), Qi, input_precision="ieee") * scale

            Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
            dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
            L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
            D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

        tl.store(dK_block_ptr, dKj.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(dV_block_ptr, dVj.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, q_tile_size=16, k_tile_size=16):
        if triton is None:
            raise RuntimeError("Triton is required for FlashAttentionTriton.")
        if not (Q.is_cuda and K.is_cuda and V.is_cuda):
            raise ValueError("FlashAttentionTriton requires CUDA tensors.")
        if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
            raise ValueError("Expected tensors shaped (batch, sequence, features).")
        B, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[-2]
        if K.shape != V.shape or K.shape[0] != B or K.shape[-1] != D:
            raise ValueError("Q, K and V must have matching batch and feature dimensions; K and V must have matching shapes.")
        if Q.device != K.device or Q.device != V.device or Q.dtype != K.dtype or Q.dtype != V.dtype:
            raise ValueError("Q, K and V must have the same device and dtype.")
        if D < 16 or D & (D - 1) or N_QUERIES < 16 or N_KEYS < 16:
            raise ValueError("Expected a power-of-two feature dimension >= 16 and sequence lengths >= 16.")

        for size in (q_tile_size, k_tile_size):
            if not isinstance(size, int) or size < 16 or size & (size - 1):
                raise ValueError("Triton tile sizes must be powers of two >= 16.")
        Q_TILE_SIZE = q_tile_size
        K_TILE_SIZE = k_tile_size
        ctx.tile_sizes = (q_tile_size, k_tile_size)
        O = torch.empty_like(Q)
        L = torch.empty((B, N_QUERIES), device=Q.device, dtype=torch.float32)

        ctx.is_causal = is_causal

        # Batch offsets and strides address the original tensors directly.
        with torch.cuda.device(Q.device):
            flash_fwd_kernel[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), B)](
                Q, K, V, O, L,
                *Q.stride(), *K.stride(), *V.stride(),
                *O.stride(), *L.stride(),
                N_QUERIES, N_KEYS,
                scale=D ** -0.5,
                D=D,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                IS_CAUSAL=is_causal,
            )

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        B, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[-2]

        Q_TILE_SIZE, K_TILE_SIZE = ctx.tile_sizes
        D_vec = (O.float() * dO.float()).sum(-1)
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        with torch.cuda.device(Q.device):
            flash_bwd_kernel1[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), B)](
                Q, K, V, dO, L, D_vec, dQ,
                *Q.stride(), *K.stride(), *V.stride(),
                *dO.stride(), *L.stride(), *D_vec.stride(),
                *dQ.stride(),
                N_QUERIES, N_KEYS,
                scale=D ** -0.5,
                D=D,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                IS_CAUSAL=ctx.is_causal,
            )
            flash_bwd_kernel2[(triton.cdiv(N_KEYS, K_TILE_SIZE), B)](
                Q, K, V, dO, L, D_vec, dK, dV,
                *Q.stride(), *K.stride(), *V.stride(),
                *dO.stride(), *L.stride(), *D_vec.stride(),
                *dK.stride(), *dV.stride(),
                N_QUERIES, N_KEYS,
                scale=D ** -0.5,
                D=D,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                IS_CAUSAL=ctx.is_causal,
            )

        return (dQ, dK, dV, None, None, None)[:len(ctx.needs_input_grad)]
