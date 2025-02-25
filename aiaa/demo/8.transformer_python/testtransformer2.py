# 导入必要的库
import torch
import torch.nn as nn
import math
import transformers
from modelscope import snapshot_download
import torch.nn.functional as F
import copy
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 单头注意力机制
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return nn.LayerNorm(output.size(-1)).to(output.device)(output + residual), attn

# 前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(output.size(-1)).to(output.device)(output + residual)

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask=None):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

# 编码器
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, n_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs, enc_self_attn_mask=None):
        enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

# 解码器
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, n_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        dec_outputs = dec_inputs
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

# 归一化层
class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 自定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len=512):
        """
        自定义翻译数据集
        
        Args:
            src_texts: 源语言文本列表
            tgt_texts: 目标语言文本列表
            tokenizer: 分词器
            max_len: 最大序列长度
        """
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        # 编码源文本
        src_encoded = self.tokenizer(
            self.src_texts[idx], 
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        src_ids = src_encoded['input_ids'].squeeze()
        
        # 编码目标文本 - 两次编码，一次用于输入，一次用于标签
        tgt_encoded_input = self.tokenizer(
            self.tgt_texts[idx], 
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        tgt_encoded_output = self.tokenizer(
            self.tgt_texts[idx], 
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        # 输入(decoder input): [BOS, token1, token2, ..., EOS, PAD, ...]
        # 输出(target): [token1, token2, ..., EOS, PAD, ...]
        tgt_input_ids = tgt_encoded_input['input_ids'].squeeze()
        tgt_output_ids = tgt_encoded_output['input_ids'].squeeze()
        
        return src_ids, tgt_input_ids, tgt_output_ids

# Transformer 模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, d_k, d_v, d_ff, n_layers, src_pad_idx, tgt_pad_idx, device, max_len=512):
        super(Transformer, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        self.max_len = max_len

        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.encoder = Encoder(d_model, n_heads, d_k, d_v, d_ff, n_layers)
        self.decoder = Decoder(d_model, n_heads, d_k, d_v, d_ff, n_layers)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # 字符串转token模型
        tokenizer_path = snapshot_download("lyramilk/deepseek_v3_tokenizer")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    def forward(self, enc_inputs, dec_inputs):
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
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        dec_logits = self.projection(dec_outputs)
        
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

    def get_attn_pad_mask(self, seq_q, pad_idx):
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

    def get_attn_subsequence_mask(self, seq):
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
        return subsequence_mask

    # 训练函数
    def train_model(self, train_loader, criterion, optimizer, n_epochs):
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
                    print(f"Epoch: {epoch+1}/{n_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch: {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
            
            # 添加终止条件：如果平均loss低于0.1则停止训练
            if avg_loss < 0.1:
                print(f"提前终止训练: 第{epoch+1}轮平均损失{avg_loss:.4f}已低于阈值0.05")
                break

    # 预测函数
    def predict(self, input_sentence, max_len=512):
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

# 创建Transformer模型
src_vocab_size = 10000  # 源语言词汇表大小
tgt_vocab_size = 10000  # 目标语言词汇表大小
d_model = 512  # 词嵌入维度
n_heads = 8  # 多头注意力机制的头数
d_k = 64  # 键向量维度
d_v = 64  # 值向量维度
d_ff = 2048  # 前馈神经网络的维度
n_layers = 6  # 编码器和解码器的层数
src_pad_idx = 0  # 源语言填充token的索引
tgt_pad_idx = 0  # 目标语言填充token的索引
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 主函数
def main():
    # 设置设备
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    tokenizer_path = snapshot_download("lyramilk/deepseek_v3_tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 准备训练数据
    src_texts = [
        "你好北京",
        "北京天气很好",
        "我喜欢上海",
        "深圳是科技城市",
        "广州有很多美食",
        "我爱北京",
        "东京是日本的首都",
        "西安发展很快"
    ]
    
    tgt_texts = [
        "Hello Beijing",
        "Beijing weather is good",
        "I like Shanghai",
        "Shenzhen is a tech city",
        "Guangzhou has a lot of food",
        "I love Beijing",
        "Tokyo is the capital of Japan",
        "Xi'an is developing quickly"
    ]
    
    # 创建数据集
    train_dataset = TranslationDataset(src_texts, tgt_texts, tokenizer)
    
    # 创建数据加载器
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 更新模型参数
    src_vocab_size = len(tokenizer)
    tgt_vocab_size = len(tokenizer)
    src_pad_idx = tokenizer.pad_token_id
    tgt_pad_idx = tokenizer.pad_token_id
    
    # 创建模型
    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, n_heads, d_k, d_v, 
        d_ff, n_layers, src_pad_idx, tgt_pad_idx, device
    )
    transformer = transformer.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 训练模型
    n_epochs = 100
    print("开始训练...")
    transformer.train_model(train_loader, criterion, optimizer, n_epochs)
    
    # 测试模型
    print("\n测试模型预测...")
    test_sentences = [
        "北京是中国的首都",
        "我爱中国",
        "深圳发展很快"
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
