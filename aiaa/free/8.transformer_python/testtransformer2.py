# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 导入torch的数据集操作类
from torch.utils.data import Dataset, DataLoader
# 导入数学计算库
import math
# transformer库，只用了其中的tokenizer
import transformers
# 从魔搭下载模型
from modelscope import snapshot_download


# 位置编码
class PositionalEncoding(nn.Module):
    """
    位置编码器，给输入序列添加位置信息，位置信息是直接用正弦和余弦函数计算直接加到输入序列上
    位置信息虽然会影响模型效果，但是训练样本多了，位置信息对效果的影响会逐渐不敏感
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        初始化位置编码器
        
        Args:
            d_model: 词嵌入维度
            dropout: 丢弃率
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        # 丢弃率
        self.dropout = nn.Dropout(p=dropout)
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        # 位置
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 除数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数位置
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)
        # 位置编码
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 注册位置编码
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列
            
        Returns:
            输出序列
        """
        # 输入序列加上位置编码
        x = x + self.pe[:x.size(0), :]
        # 丢弃率
        return self.dropout(x)

# 单头注意力机制
class ScaledDotProductAttention(nn.Module):
    """
    单头注意力机制，计算注意力权重
    """
    def __init__(self, d_k: int):
        super(ScaledDotProductAttention, self).__init__()
        # 词嵌入维度
        self.d_k = d_k

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            Q: 查询
            K: 键
            V: 值
            attn_mask: 注意力掩码
            
        Returns:
            注意力权重和上下文向量
        """
        # 计算注意力权重
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        # 如果注意力掩码存在，则将注意力权重设置为-1e9
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        # 计算注意力权重
        attn = F.softmax(scores, dim=-1)
        # 计算上下文向量
        context = torch.matmul(attn, V)
        # 返回上下文向量和注意力权重
        return context, attn

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制，用于将输入的查询、键和值转换为上下文向量和注意力权重。
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int):
        """
        初始化多头注意力机制
        
        Args:
            d_model: 词嵌入维度
            n_heads: 多头注意力机制的头部数量
            d_k: 键的维度
            d_v: 值的维度
        """
        super(MultiHeadAttention, self).__init__()
        # 多头注意力机制的头部数量
        self.n_heads = n_heads
        # 键的维度
        self.d_k = d_k
        # 值的维度
        self.d_v = d_v
        # 查询的权重矩阵
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        # 键的权重矩阵
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        # 值的权重矩阵
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # 全连接层
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q: torch.Tensor, input_K: torch.Tensor, input_V: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_Q: 查询
            input_K: 键
            input_V: 值
            attn_mask: 注意力掩码
            
        Returns:
            注意力权重和上下文向量
        """
        # 残差
        residual, batch_size = input_Q, input_Q.size(0)
        # 查询
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # 键
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # 值
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # 如果注意力掩码存在，则将注意力掩码扩展到多头维度
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # 计算注意力权重和上下文向量
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        # 将上下文向量展平
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # 全连接层
        output = self.fc(context)
        # 归一化
        return nn.LayerNorm(output.size(-1)).to(output.device)(output + residual), attn

# 位置全连接前馈网络
class PoswiseFeedForwardNet(nn.Module):
    """
    位置全连接前馈网络,用于将上下文向量展平
    """
    def __init__(self, d_model: int, d_ff: int):
        """
        初始化位置全连接前馈网络
        
        Args:
            d_model: 词嵌入维度
            d_ff: 前馈神经网络的维度
        """
        super(PoswiseFeedForwardNet, self).__init__()
        # 前馈神经网络
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 输入
            
        Returns:
            输出
        """
        # 残差
        residual = inputs
        # 前馈神经网络
        output = self.fc(inputs)
        # 归一化
        return nn.LayerNorm(output.size(-1)).to(output.device)(output + residual)

# 编码器层
class EncoderLayer(nn.Module):
    """
    编码器层，用于将上下文向量展平
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        """
        初始化编码器层
        
        Args:
            d_model: 词嵌入维度
            n_heads: 多头注意力机制的头部数量
            d_k: 键的维度
            d_v: 值的维度
            d_ff: 前馈神经网络的维度
        """
        super(EncoderLayer, self).__init__()
        # 多头注意力机制
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        # 前馈神经网络
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs: torch.Tensor, enc_self_attn_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            enc_inputs: 输入
            enc_self_attn_mask: 注意力掩码
            
        Returns:
            输出和注意力权重
        """
        # 多头注意力机制
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # 前馈神经网络
        enc_outputs = self.pos_ffn(enc_outputs)
        # 返回输出和注意力权重
        return enc_outputs, attn

# 编码器
class Encoder(nn.Module):
    """
    编码器，用于将上下文向量展平
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int, n_layers: int):
        """
        初始化编码器
        
        Args:
            d_model: 词嵌入维度
            n_heads: 多头注意力机制的头部数量
            d_k: 键的维度
            d_v: 值的维度
            d_ff: 前馈神经网络的维度
            n_layers: 编码器层的数量
        """
        super(Encoder, self).__init__()
        # 编码器层
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs: torch.Tensor, enc_self_attn_mask: torch.Tensor = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        前向传播
        
        Args:
            enc_inputs: 输入
            enc_self_attn_mask: 注意力掩码
            
        Returns:
            输出和注意力权重列表
        """
        # 编码器层
        enc_outputs = enc_inputs
        # 注意力权重列表
        enc_self_attns = []
        # 编码器层
        for layer in self.layers:
            # 编码器层
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            # 注意力权重列表
            enc_self_attns.append(enc_self_attn)
        # 返回输出和注意力权重列表
        return enc_outputs, enc_self_attns

# 解码器层
class DecoderLayer(nn.Module):
    """
    解码器层，用于将上下文向量展平
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        """
        初始化解码器层
        
        Args:
            d_model: 词嵌入维度
            n_heads: 多头注意力机制的头部数量
            d_k: 键的维度
            d_v: 值的维度
            d_ff: 前馈神经网络的维度
        """
        super(DecoderLayer, self).__init__()
        # 多头注意力机制
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        # 多头注意力机制
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        # 前馈神经网络
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs: torch.Tensor, enc_outputs: torch.Tensor, dec_self_attn_mask: torch.Tensor = None, dec_enc_attn_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            dec_inputs: 解码器输入
            enc_outputs: 编码器输出
            dec_self_attn_mask: 解码器自注意力掩码
            dec_enc_attn_mask: 解码器编码器注意力掩码
            
        Returns:
            输出和注意力权重
        """
        # 多头注意力机制
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # 多头注意力机制
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # 前馈神经网络
        dec_outputs = self.pos_ffn(dec_outputs)
        # 返回输出和注意力权重
        return dec_outputs, dec_self_attn, dec_enc_attn

# 解码器
class Decoder(nn.Module):
    """
    解码器，用于将上下文向量展平
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int, n_layers: int):
        """
        初始化解码器
        
        Args:
            d_model: 词嵌入维度
            n_heads: 多头注意力机制的头部数量
            d_k: 键的维度
            d_v: 值的维度
            d_ff: 前馈神经网络的维度
            n_layers: 解码器层的数量
        """
        super(Decoder, self).__init__()
        # 解码器层
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])

    def forward(self, dec_inputs: torch.Tensor, enc_outputs: torch.Tensor, dec_self_attn_mask: torch.Tensor = None, dec_enc_attn_mask: torch.Tensor = None) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        前向传播
        
        Args:
            dec_inputs: 解码器输入
            enc_outputs: 编码器输出
            dec_self_attn_mask: 解码器自注意力掩码
            dec_enc_attn_mask: 解码器编码器注意力掩码
            
        Returns:
            输出和注意力权重列表
        """
        # 解码器层
        dec_outputs = dec_inputs
        # 注意力权重列表
        dec_self_attns, dec_enc_attns = [], []
        # 解码器层
        for layer in self.layers:
            # 解码器层
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            # 注意力权重列表
            dec_self_attns.append(dec_self_attn)
            # 注意力权重列表
            dec_enc_attns.append(dec_enc_attn)
        # 返回输出和注意力权重列表
        return dec_outputs, dec_self_attns, dec_enc_attns

# 归一化层
class LayerNormalization(nn.Module):
    """
    归一化层，用于将上下文向量展平
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        初始化归一化层
        
        Args:
            d_model: 词嵌入维度
            eps: 归一化层的epsilon
        """
        super(LayerNormalization, self).__init__()
        # 归一化层的参数
        self.a_2 = nn.Parameter(torch.ones(d_model))
        # 归一化层的偏置
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        # 归一化层的epsilon
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入
            
        Returns:
            输出
        """
        # 计算均值
        mean = x.mean(-1, keepdim=True)
        # 计算标准差
        std = x.std(-1, keepdim=True)
        # 返回输出
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 自定义数据集类
class TranslationDataset(Dataset):
    """
    自定义翻译数据集
    """
    def __init__(self, data: list[tuple[str, str]], tokenizer: transformers.AutoTokenizer, max_len: int = 512):
        """
        自定义翻译数据集
        
        Args:
            data: 数据列表，每个元素为源语言和目标语言的元组
            tokenizer: 分词器
            max_len: 最大序列长度
        """
        super(TranslationDataset, self).__init__()
        # 数据列表
        self.data = data
        # 分词器
        self.tokenizer = tokenizer
        # 最大序列长度
        self.max_len = max_len
    
    def __len__(self) -> int:
        """
        获取数据集长度
        
        Returns:
            int: 数据集长度
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取数据集中的一个样本
        
        Args:
            idx: 样本索引
        Returns:
            源语言文本、目标语言文本和目标语言文本的编码
        """
        # 编码源文本
        encoder_inputs = self.tokenizer(
            self.data[idx][0], 
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        # 获取源文本的编码
        encoder_ids = encoder_inputs['input_ids'].squeeze()

        # 对outputs进行两次编码，一次用于解码器的输入，一次用于验证最终的输出。
        decoder_inputs = self.tokenizer(
            self.data[idx][1], 
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        # 获取目标文本的编码
        decoder_outputs = decoder_inputs.copy();
        
        # 输入(decoder input): [BOS, token1, token2, ..., EOS, PAD, ...]
        # 输出(target): [token1, token2, ..., EOS, PAD, ...]
        decoder_inputs_ids = decoder_inputs['input_ids'].squeeze()
        # 获取目标文本的编码
        decoder_outputs_ids = decoder_outputs['input_ids'].squeeze()
        
        # 返回源语言文本、目标语言文本和目标语言文本的编码
        return encoder_ids, decoder_inputs_ids, decoder_outputs_ids

# Transformer 模型
class Transformer(nn.Module):
    """
    Transformer 模型
    """
    def __init__(self, d_model: int = 512, n_heads: int = 8, d_k: int = 64, d_v: int = 64, d_ff: int = 2048, n_layers: int = 6, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), max_len: int = 512):
        """
        初始化Transformer模型
        
        Args:
            d_model: 词嵌入维度
            n_heads: 多头注意力机制的头部数量
            d_k: 键的维度
            d_v: 值的维度
            d_ff: 前馈神经网络的维度
            n_layers: 编码器和解码器层的数量
            device: 设备
            max_len: 最大序列长度
        """
        super(Transformer, self).__init__()
        # 字符串转token模型，先从魔搭下载模型
        tokenizer_path = snapshot_download("lyramilk/deepseek_v3_tokenizer")
        # 再使用transformers.AutoTokenizer加载模型
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 输入字典大小
        self.src_vocab_size = self.tokenizer.vocab_size
        # 输出字典大小
        self.tgt_vocab_size = self.tokenizer.vocab_size
        # 源语言填充token的索引
        self.src_pad_idx = self.tokenizer.pad_token_id
        # 目标语言填充token的索引
        self.tgt_pad_idx = self.tokenizer.pad_token_id
        # 词嵌入维度
        self.d_model = d_model
        # 多头注意力机制的头部数量
        self.n_heads = n_heads
        # 键的维度
        self.d_k = d_k
        # 值的维度
        self.d_v = d_v
        # 前馈神经网络的维度
        self.d_ff = d_ff
        # 编码器和解码器层的数量
        self.n_layers = n_layers
        # 设备
        self.device = device
        # 最大序列长度
        self.max_len = max_len

        # 源语言嵌入层
        self.src_emb = nn.Embedding(self.src_vocab_size, self.d_model)
        # 目标语言嵌入层
        self.tgt_emb = nn.Embedding(self.tgt_vocab_size, self.d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=self.max_len)
        # 编码器
        self.encoder = Encoder(self.d_model, self.n_heads, self.d_k, self.d_v, self.d_ff, self.n_layers)
        # 解码器
        self.decoder = Decoder(self.d_model, self.n_heads, self.d_k, self.d_v, self.d_ff, self.n_layers)
        # 投影层
        self.projection = nn.Linear(self.d_model, self.tgt_vocab_size, bias=False)

    # 前向传播
    def forward(self, enc_inputs: torch.Tensor, dec_inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        前向传播
        
        Args:
            enc_inputs: 源语言文本
            dec_inputs: 目标语言文本
            
        Returns:
            输出和注意力权重列表
        """
        # 创建掩码
        enc_self_attn_mask = self.get_attn_pad_mask(enc_inputs, self.src_pad_idx)
        dec_self_attn_mask = self.get_attn_subsequence_mask(dec_inputs)
        dec_enc_attn_mask = self.get_attn_pad_mask(enc_inputs, self.src_pad_idx)

        # 将输入转为嵌入
        enc_inputs = self.src_emb(enc_inputs)
        dec_inputs = self.tgt_emb(dec_inputs)
        
        # 添加位置编码
        enc_inputs = self.pos_encoding(enc_inputs.transpose(0, 1)).transpose(0, 1)
        dec_inputs = self.pos_encoding(dec_inputs.transpose(0, 1)).transpose(0, 1)

        # 编码器和解码器前向传播
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, enc_self_attn_mask)
        # 解码器前向传播
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        # 投影层
        dec_logits = self.projection(dec_outputs)
        # 返回输出和注意力权重列表
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

    # 创建填充掩码
    def get_attn_pad_mask(self, seq_q: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """
        创建填充掩码
        
        Args:
            seq_q: 查询序列
            pad_idx: 填充token的索引
            
        Returns:
            填充掩码
        """
        batch_size, len_q = seq_q.size()
        # 创建一个布尔掩码，标记填充位置为True
        pad_attn_mask = seq_q.data.eq(pad_idx).unsqueeze(1)  # [batch_size, 1, len_q]
        return pad_attn_mask.expand(batch_size, len_q, len_q)  # [batch_size, len_q, len_q]

    # 创建后续序列掩码
    def get_attn_subsequence_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        创建后续序列掩码（用于解码器中防止看到未来信息）
        
        Args:
            seq: 输入序列
            
        Returns:
            后续序列掩码
        """
        batch_size, seq_len = seq.size()
        # 创建上三角矩阵（对角线以上为1，其余为0）
        subsequence_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
        # 扩展到批次维度
        subsequence_mask = subsequence_mask.unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        # 返回后续序列掩码
        return subsequence_mask

    # 训练函数
    def train_model(self, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, n_epochs: int):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器
            n_epochs: 训练轮数
        """
        self.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for batch_idx, (enc_inputs, dec_inputs, dec_outputs) in enumerate(train_loader):
                # 将数据移至设备
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(self.device), dec_inputs.to(self.device), dec_outputs.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs, _, _, _ = self(enc_inputs, dec_inputs)
                
                # 计算损失（忽略填充token）
                loss = criterion(outputs, dec_outputs.view(-1))
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 打印批次进度
                if (batch_idx + 1) % 10 == 0:
                    print(f"轮次: {epoch+1}/{n_epochs}, 批次: {batch_idx+1}/{len(train_loader)}, 损失: {loss.item():.4f}")
                
            avg_loss = total_loss / len(train_loader)
            print(f"轮次: {epoch+1}/{n_epochs}, 平均损失: {avg_loss:.4f}")
            
            # 添加终止条件：如果平均loss低于0.1则停止训练
            if avg_loss < 0.1:
                print(f"提前终止训练: 第{epoch+1}轮平均损失{avg_loss:.4f}已低于阈值0.05")
                break

    # 预测函数
    def predict(self, input_sentence: str, max_len: int = 512) -> str:
        """
        使用模型进行预测
        
        Args:
            input_sentence: 输入句子
            max_len: 最大生成长度
            
        Returns:
            预测的句子
        """
        self.eval()
        with torch.no_grad():
            # 将输入句子转换为token
            input_tokens = self.tokenizer(input_sentence, return_tensors="pt")["input_ids"].to(self.device)
            
            # 初始化解码器输入（以BOS token开始）
            dec_inputs = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)
            
            # 确保输入序列的批次大小与解码器输入匹配
            if input_tokens.size(0) != dec_inputs.size(0):
                input_tokens = input_tokens.expand(dec_inputs.size(0), -1)
            
            # 逐个生成token
            for _ in range(max_len):
                # 前向传播
                enc_inputs = self.src_emb(input_tokens)
                dec_inputs_emb = self.tgt_emb(dec_inputs)
                
                # 添加位置编码
                enc_inputs = self.pos_encoding(enc_inputs.transpose(0, 1)).transpose(0, 1)
                dec_inputs_emb = self.pos_encoding(dec_inputs_emb.transpose(0, 1)).transpose(0, 1)
                
                # 创建正确维度的掩码
                enc_self_attn_mask = self.get_attn_pad_mask(input_tokens, self.src_pad_idx)
                dec_self_attn_mask = self.get_attn_subsequence_mask(dec_inputs)
                
                # 创建解码器-编码器注意力掩码
                batch_size, tgt_len = dec_inputs.size()
                src_len = input_tokens.size(1)
                dec_enc_attn_mask = input_tokens.data.eq(self.src_pad_idx).unsqueeze(1).expand(batch_size, tgt_len, src_len)
                
                # 编码器前向传播
                enc_outputs, _ = self.encoder(enc_inputs, enc_self_attn_mask)
                
                # 解码器前向传播
                dec_outputs, _, _ = self.decoder(dec_inputs_emb, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
                dec_logits = self.projection(dec_outputs)
                
                # 获取下一个token
                next_token = dec_logits[:, -1].argmax(dim=-1).unsqueeze(1)
                dec_inputs = torch.cat([dec_inputs, next_token], dim=1)
                
                # 如果生成了EOS token，停止生成
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            # 解码生成的token序列
            predicted_tokens = dec_inputs.squeeze(0).tolist()[1:]  # 去掉BOS token
            predicted_sentence = self.tokenizer.decode(predicted_tokens, skip_special_tokens=True)
            
            return predicted_sentence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 主函数
def main():
    # 设置设备
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    tokenizer_path = snapshot_download("lyramilk/deepseek_v3_tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 准备训练数据
    data = [
        ("你好上海", "Hello Shanghai"),
        ("你好广州", "Hello Guangzhou"),
        ("你好深圳", "Hello Shenzhen"),
        ("你好西安", "Hello Xi'an"),
        
        ("北京很棒", "Beijing is great"),
        ("上海很棒", "Shanghai is great"),
        ("广州很棒", "Guangzhou is great"),
        ("深圳很棒", "Shenzhen is great"),
        ("西安很棒", "Xi'an is great"),
    ]
    
    # 创建数据集
    train_dataset = TranslationDataset(data, tokenizer)
    
    # 创建数据加载器
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    transformer = Transformer()
    transformer = transformer.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 训练模型
    n_epochs = 100
    print("开始训练...")
    transformer.train_model(train_loader, criterion, optimizer, n_epochs)
    
    # 测试模型
    print("\n测试模型预测...")
    test_sentences = [
        "你好北京",
    ]
    
    for sentence in test_sentences:
        predicted = transformer.predict(sentence)
        print(f"输入: {sentence}")
        print(f"预测: {predicted}\n")
    
    # 保存模型
    torch.save(transformer.state_dict(), "transformer_model.pth")
    print("模型已保存为 transformer_model.pth")

# 如果直接运行这个脚本，则执行main函数
if __name__ == "__main__":
    main()
