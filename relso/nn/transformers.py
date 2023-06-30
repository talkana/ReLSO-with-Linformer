import math

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


# ---------------------
# Tranformer Block
# ---------------------

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def scaled_dot_product_attention(q, k_short, v_short, mask=None):
    d_k = k_short.size()[-1]  ##d_k == d_q (== d_v as in original Transformer)
    attn_logits = torch.matmul(q, k_short.transpose(-2, -1))
    attn_logits /= math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)  # ??: czy to tak zostaje?
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v_short)
    return values, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # regiaster_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


####################################################################
#### SUPER-DIRTY CODE FOR LINFORMER MULTI-HEAD ATTENTION ##########

class MultiHeadLinformerAttention(nn.Module):
    def __init__(self, input_dim, input_seq_len, h_times_embed_dim, num_heads, k):
        """
            input_dim - embedding vect dim * num_heads
            input_seq_len - length of input sequences (must be constant among sequences)
            h_times_embed_dim - d_v * num_heads (d_v = d_k = d_q; dim of single query/key/val vector)
            num_heads - nuber of heads in attention block
            k - length of sequence after E/F projections
        """
        super().__init__()
        assert h_times_embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."  ## as in original relso implementation

        self.num_heads = num_heads
        self.d_v = h_times_embed_dim // num_heads  # (d_v = d_k = d_q)

        # Projecting input to Q,K,V matrices for all heads (stacked for efficiency)
        self.qkv_proj = nn.Linear(input_dim, h_times_embed_dim * 3, bias=False)

        # Projecting sequence of keys -- from length=input_seq_len to length=k
        self.k_projections = nn.ModuleList([nn.Linear(input_seq_len, k, bias=False) for i in
                                            range(num_heads)])  # projection matrices Ei, i=1,..,num_heads

        # Projecting sequence of values -- from length=input_seq_len to length=k
        self.v_projections = nn.ModuleList([nn.Linear(input_seq_len, k, bias=False) for i in
                                            range(num_heads)])  # projection matrices Fi, i=1,..,num_heads

        self.o_proj = nn.Linear(h_times_embed_dim, h_times_embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_attention=False, mask=None):

        batch_size, seq_length, embed_dim = x.size()  # ????? embed? nie input (to pewnie będzie h x input == h x embed)
        # x.size() = [Batch, SeqLen, InputDim]

        # for each input vec: obtaining corresponding q,k,v, concatenated within each head and for all heads (so dimensionality is now is (d_q + d_k + d_v)*num_heads )
        qkv = self.qkv_proj(x)  # [Batch, SeqLen, 3* SingleEmbedDim* NumHeads]

        # Separate Q, K, V from linear output
        qkv = qkv.view(batch_size, seq_length, self.num_heads,
                       3 * self.d_v)  # [Batch, SeqLen, NumHeads, 3*SingleEmbedDim]
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, NumHeads, SeqLen, 3*SingleEmbedDim]
        q, k, v = qkv.chunk(3, dim=-1)  # [Batch, SeqLen, NumHeads 1*SingleEmbedDim]

        # Project K and V on sequences of length k (separate for each head)
        # Ei projections -- reducing number of key vectors to k
        k = k.permute(0, 2, 3, 1)  # [Batch, SeqLen, EmbedDim, NumHeads]
        ks = k.chunk(self.num_heads, dim=-1)  # [t], t.shape=[Batch, SeqLen, EmbedDim, 1]
        ks = [torch.squeeze(k, dim=-1) for k in ks]  # el: [Batch, SeqLen, EmbedDim]

        ks = [k.permute(0, 2, 1) for k in ks]
        k_shorts = [k_proj_i(k) for k_proj_i, k in zip(self.k_projections, ks)]  # el: [Batch, EmbedDim, SeqLen']

        # concatenate back to shape: [Batch, NumHeads, SeqLen', EmbedDim]
        k_shorts = [torch.unsqueeze(k, dim=-1) for k in k_shorts]  # [Batch, EmbedDim, SeqLen', 1]
        k_short = torch.cat(k_shorts, dim=-1)  # [Batch, EmbedDim, SeqLen', NumHeads]
        k_short = k_short.permute(0, 3, 2, 1)  # [Batch, NumHeads, SeqLen', EmbedDim]

        # Fi projections -- reducing number of value vectors to k
        v = v.permute(0, 2, 3, 1)  # [Batch, SeqLen, EmbedDim, NumHeads]

        vs = v.chunk(self.num_heads, dim=-1)  # [t], t.shape=[Batch, SeqLen, EmbedDim, 1]
        vs = [torch.squeeze(v, dim=-1) for v in vs]
        vs = [v.permute(0, 2, 1) for v in vs]
        v_shorts = [v_proj_i(v) for v_proj_i, v in zip(self.v_projections, vs)]

        # concatenate back to shape: [Batch, NumHeads, SeqLen', EmbedDim]
        v_shorts = [torch.unsqueeze(v, dim=-1) for v in v_shorts]  # [Batch, EmbedDim, SeqLen', 1]
        v_short = torch.cat(v_shorts, dim=-1)  # [Batch, EmbedDim, SeqLen', NumHeads]
        v_short = v_short.permute(0, 3, 2, 1)

        # Calculate attention weights and value outputs
        values, attention = scaled_dot_product_attention(q, k_short, v_short)  # no mask
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


####################################################################
####################################################################


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        """
            input_dim - ???
            embed_dim - ???
            num_heads - number of heads in attention block
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):

        batch_size, seq_length, embed_dim = x.size()  # [64, 20, 100] --> [Batch, SeqLen, h_times_embed_size]
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product_attention(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, seq_len, use_linformer, dropout=0.0,
                 k=20): 
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        if use_linformer:
            self.self_attn = MultiHeadLinformerAttention(input_dim, seq_len, input_dim, num_heads, k)
        else:
            self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps
