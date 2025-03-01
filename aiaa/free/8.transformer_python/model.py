import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_length, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_length, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # 每个头的维度

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, seq_length, d_model]
        # mask: [batch_size, seq_length, seq_length]
        batch_size = Q.size(0)

        # 线性变换并切分多头
        Q = self.W_Q(Q).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # [batch_size, nhead, seq_length, d_k]
        K = self.W_K(K).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch_size, nhead, seq_length, seq_length]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 用 -1e9 替换 0
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)  # [batch_size, nhead, seq_length, d_k]

        # 多头拼接
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_length, d_model]

        # 线性变换
        output = self.W_O(context)  # [batch_size, seq_length, d_model]

        return output

# 前馈网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, dim_feedforward)
        self.W_2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_length, d_model]
        return self.W_2(self.dropout(F.relu(self.W_1(x))))

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_length, d_model]
        # mask: [batch_size, seq_length, seq_length]
        # Self-Attention
        residual = x
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        # Feed Forward
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x

# 编码器
class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.self_attn.d_model)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_length, d_model]
        # mask: [batch_size, seq_length, seq_length]
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        # x: [batch_size, seq_length, d_model]
        # encoder_output: [batch_size, seq_length, d_model]
        # source_mask: [batch_size, seq_length, seq_length]
        # target_mask: [batch_size, seq_length, seq_length]
        # Self-Attention
        residual = x
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, target_mask)))
        # Cross-Attention
        x = self.norm2(x + self.dropout(self.cross_attn(x, encoder_output, encoder_output, source_mask)))
        # Feed Forward
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x

# 解码器
class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(decoder_layer.self_attn.d_model)

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        # x: [batch_size, seq_length, d_model]
        # encoder_output: [batch_size, seq_length, d_model]
        # source_mask: [batch_size, seq_length, seq_length]
        # target_mask: [batch_size, seq_length, seq_length]
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)

# Transformer 模型
class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        encoder_layer = EncoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout)
        self.encoder = Encoder(encoder_layer, config.num_encoder_layers)
        decoder_layer = DecoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout)
        self.decoder = Decoder(decoder_layer, config.num_decoder_layers)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, source, target, source_mask=None, target_mask=None):
        # source: [batch_size, seq_length]
        # target: [batch_size, seq_length]
        # source_mask: [batch_size, seq_length, seq_length]
        # target_mask: [batch_size, seq_length, seq_length]

        # 词嵌入和位置编码
        source_embedded = self.embedding(source)  # [batch_size, seq_length, d_model]
        target_embedded = self.embedding(target)
        source_embedded = self.positional_encoding(source_embedded)
        target_embedded = self.positional_encoding(target_embedded)

        # 编码器和解码器
        encoder_output = self.encoder(source_embedded, source_mask)  # [batch_size, seq_length, d_model]
        decoder_output = self.decoder(target_embedded, encoder_output, source_mask, target_mask)  # [batch_size, seq_length, d_model]

        # 输出层
        output = self.output_layer(decoder_output)  # [batch_size, seq_length, vocab_size]
        return output 