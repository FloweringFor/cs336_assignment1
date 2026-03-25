import math
from typing import Optional

import torch
import torch.nn as nn

import basic_building_blocks


class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,  # Hidden dimension of the model
            eps: float = 1e-5,  # Epsilon value for numerical stability
            device: Optional[torch.device] = None,  # Device to store the parameters on
            dtype: Optional[torch.dtype] = None  # Data type of the parameters
    ):
        """
        Construct the RMSNorm module.
        """
        super(RMSNorm, self).__init__()
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape
        """
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        rms_norm = x * self.gain / rms
        return rms_norm.to(in_type)


class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ):
        super(SwiGLU, self).__init__()
        if d_ff is None:
            # 1. 计算 8/3 * d_model
            raw_d_ff = int(8 / 3 * d_model)
            # 2. 向上取整到 64 的倍数 (Hardware alignment)
            # 公式：(n + alignment - 1) // alignment * alignment
            d_ff = (raw_d_ff + 63) // 64 * 64
        self.w1 = basic_building_blocks.Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = basic_building_blocks.Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = basic_building_blocks.Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x):
        w1x = self.w1(x)
        w3x = self.w3(x)
        return self.w2(silu(w1x) * w3x)


class RoPE(nn.Module):
    def __init__(
            self,
            theta: float,  # Θ value for the RoPE
            d_k: int,  # dimension of query and key vectors
            max_seq_len: int,  # Maximum sequence length that will be inputted
            device: Optional[torch.device] = None  # Device to store the buffer on
    ):
        """
        Construct the RoPE module and create buffers.
        """
        super(RoPE, self).__init__()
        self.theta = theta
        self.d_k = d_k
        inv_freq = 1 / theta ** ((torch.arange(0, d_k, 2, device=device)) / d_k)  # d_k // 2
        t = torch.arange(0, max_seq_len, 1, device=device).float()  # max_seq_len
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, d_k // 2)
        emb = freqs.repeat_interleave(2, dim=-1)  # 相邻配对 (max_seq_len, d_k)
        # emb = torch.cat((freqs, freqs), dim=-1)  # 跨半区配对 (max_seq_len, d_k)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor):
        # 跨半区配对
        # x1 = x[..., :self.d_k // 2]
        # x2 = x[..., self.d_k // 2:]
        # return torch.cat((-x2, x1), dim=-1)

        # 相邻配对
        x_even = x[..., 0::2]  # 偶数位
        x_odd = x[..., 1::2]  # 奇数位
        x_rotated = torch.stack((-x_odd, x_even), dim=-1)
        return x_rotated.flatten(-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Heads, Seq_Len, d_k)
        # token_positions: (Batch, Seq_Len)
        cos = self.cos_cached[token_positions]  # (Batch, Seq_Len, d_k)
        sin = self.sin_cached[token_positions]  # (Batch, Seq_Len, d_k)
        cos = cos.unsqueeze(-3)  # (Batch, 1, Seq_Len, d_k)
        sin = sin.unsqueeze(-3)  # (Batch, 1, Seq_Len, d_k)
        return x * cos + self._rotate_half(x) * sin


def silu(x):
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True)[0]  # torch.max 会返回两个值：最大值、最大值所在的索引
    stable_x = x - max_val
    exp_x = torch.exp(stable_x)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(
        query: torch.Tensor,  # (batch_size, ..., seq_len, d_k)
        key: torch.Tensor,  # (batch_size, ..., seq_len, d_k)
        value: torch.Tensor,  # (batch_size, ..., seq_len, d_v)
        mask: Optional[torch.Tensor] = None,  # (seq_len, seq_len)
) -> torch.Tensor:
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    scores = softmax(scores, dim=-1)
    return torch.matmul(scores, value)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,  # Dimensionality of the Transformer block inputs
            num_heads: int,  # Number of heads to use in multi-head self-attention
            max_seq_len: int = 2024,
            theta: float = 10000.0
    ):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Register buffer
        self.register_buffer("casual_mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).bool(), persistent=False)

        # ROPE
        self.rope = RoPE(theta, self.d_k, max_seq_len)

        # Linear layers for Q, K, and V
        self.W_q = basic_building_blocks.Linear(d_model, d_model)
        self.W_k = basic_building_blocks.Linear(d_model, d_model)
        self.W_v = basic_building_blocks.Linear(d_model, d_model)

        # Output projection
        self.W_o = basic_building_blocks.Linear(d_model, d_model)

    def forward(self, x, token_positions: Optional[torch.Tensor] = None):
        _, seq_len, d_model = x.shape

        # 1. Linear projections and split into heads
        # self.W_q(x) (batch_size, seq_len, d_model)
        # view (batch_size, seq_len, num_heads, d_k)
        # transpose (batch_size, num_heads, seq_len, d_k)
        q = self.W_q(x).view(-1, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(-1, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(-1, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2. ROPE
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 3. Scaled Dot-Product Attention: (batch_size, num_heads, seq_len, seq_len)
        attn_output = scaled_dot_product_attention(q, k, v, mask=self.casual_mask[:seq_len, :seq_len])

        # 4. Merge output
        out = attn_output.transpose(1, 2).contiguous().view(-1, seq_len, self.d_model)
        out = self.W_o(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,  # Dimensionality of the Transformer block inputs
            num_heads: int,  # Number of heads to use in multi-head self-attention
            d_ff: int,  # Dimensionality of the position-wise feed-forward inner layer.
            max_seq_len: int,
            theta: float,
    ):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.multi_head_self_attention = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.feed_forward = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions: Optional[torch.Tensor] = None):
        # x (batch sequence_length d_model)
        batch_size, seq_len, d_model = x.shape
        norm = self.attn_norm(x)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        attn = self.multi_head_self_attention(norm, token_positions)
        x = x + attn
        norm = self.ffn_norm(x)
        ffn = self.feed_forward(norm)
        return x + ffn


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,  # The size of the vocabulary
            context_length: int,  # The maximum context length
            d_model: int,
            num_layers: int,  # The number of Transformer blocks to use
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            token_positions: Optional[torch.Tensor] = None
    ):
        super(TransformerLM, self).__init__()

        # 1. Token Embedding: (vocab_size, d_model)
        self.token_embedding = basic_building_blocks.Embedding(vocab_size, d_model)

        # 2. Transformer Blocks: 使用 ModuleList 方便循环传递参数
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)])

        # 3. 最后的 Norm
        self.final_norm = RMSNorm(d_model)

        # 4. Output Embedding
        self.output_embedding = basic_building_blocks.Linear(d_model, vocab_size)

        self.token_positions = token_positions

    def forward(self, x):
        x = self.token_embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, self.token_positions)
        x = self.final_norm(x)
        output = self.output_embedding(x)
        # probabilities = softmax(output, dim=-1)
        return output
