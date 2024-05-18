import math
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from utils import clone, attention


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> float:
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Aims to inject information in the input embeddings to make use of the order of sequence."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("positional_encoding", pe)

    def forward(self, x: torch.Tensor) -> float:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class SubLayerConnections(nn.Module):
    def __init__(self, size: int, dropout: float):
        super(SubLayerConnections, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sub_layer: any) -> float:
        return x + self.dropout(sub_layer(self.norm(x)))


class LayerNorm(nn.Module):
    """Standard Normalization Layer"""
    def __init__(self, features: any, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> float:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attention: any, feed_forward: any, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sub_layer = clone(SubLayerConnections(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: any) -> SubLayerConnections:
        x = self.sub_layer[0](x, lambda x: self.self_attention(x, x, x, mask))
        return self.sub_layer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """Sub Layers: 1.) Multi-head self-attention, 2.) Feed Forward Network"""
    def __init__(self, layer: EncoderLayer, N: int = 6):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: any, mask: any) -> float:
        """All layers should have the same input (and mark) each iteration"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attention: any, src_attention: any, feed_forward: any, dropout: float):
        super(DecoderLayer).__init__()
        self.size = size
        self.self_attention = self_attention
        self.src_attention = src_attention
        self.feed_forward = feed_forward
        self.sub_layer = clone(SubLayerConnections(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sub_layer[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.sub_layer[1](x, lambda x: self.src_attention(x, m, m, src_mask))
        return self.sub_layer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """Sub Layers: 1.) Masked Multi-head self-attention, 2.) Multi-head self-attention, 3.) Feed Forward Network"""
    def __init__(self, layer: any, N: int = 6):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory: float, src_mask: any, tgt_mask: any) -> float:
        """Maskings are here to prevent attending to subsequent positions in the output."""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadedAttention).__init__()
        assert d_model % h == 0  # We assume d_v always equals d_k

        self.d_k = d_model // h
        self.h = h
        self.linear_layers = clone(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) \
            -> float:
        if mask is not None:
            mask = mask.unsqueeze(1) # Same mask applied to all h heads.

        batches = query.size(0)

        # Linear Projections
        query, key, value = [
            lin(x).view(batches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linear_layers, (query, key, value))
        ]

        # Apply attention to each projection
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concat
        x = (x.transpose(1, 2).contiguous().view(batches, -1, self.h * self.d_k))

        del query
        del key
        del value

        # Linear Project concatenated data
        return self.linear_layers[-1](x)


class FeedForwardNetwork(nn.Module):
    """Standard FFN equation"""
    def __init__(self, d_model: int, d_ff: int , dropout: float = 0.1):
        super(FeedForwardNetwork, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> float:
        return self.w_2(self.dropout(self.w_1(x).relu()))


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Sequential, tgt_embed: nn.Sequential,
                 generator: Generator):
        super(EncoderDecoder).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) \
            -> torch.Tensor:
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) \
            -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) \
            -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
