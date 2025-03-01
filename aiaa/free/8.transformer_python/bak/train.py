import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer  # 导入 Transformer 模型
#from torchtext.datasets import Multi30k  # 导入数据集
#from torchtext.data import Field, BucketIterator

# 假设您已经有了以下数据预处理步骤：
# 1. 使用 spacy 进行分词
# 2. 构建词汇表 (src_vocab, tgt_vocab)
# 3. 将文本转换为索引序列 (train_data, valid_data, test_data)
# 4. 定义填充索引 (src_pad_idx, tgt_pad_idx)

# 这里提供一个简化的示例，您需要根据实际情况进行修改
# 假设您已经有了以下变量：
src_vocab_size = 10000  # 源语言词汇表大小
tgt_vocab_size = 10000  # 目标语言词汇表大小
src_pad_idx = 1  # 源语言填充索引
tgt_pad_idx = 1  # 目标语言填充索引

# 模型参数
num_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1

# 创建模型
model = Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, num_layers, d_model, num_heads, d_ff, dropout)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)  # 忽略填充索引

# 训练循环
def train(model, iterator, optimizer, criterion):
    model.train()  # 设置为训练模式
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        tgt = batch.tgt

        optimizer.zero_grad()  # 梯度清零

        output = model(src, tgt[:, :-1])  # 前向传播

        # 将输出和目标转换为二维张量，以适应交叉熵损失函数
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(output, tgt)  # 计算损失
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()  # 更新参数

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 训练过程
# 假设您已经有了训练数据迭代器 (train_iterator)
# num_epochs = 10
# for epoch in range(num_epochs):
#     train_loss = train(model, train_iterator, optimizer, criterion)
#     print(f'Epoch: {epoch+1:02}')
#     print(f'\tTrain Loss: {train_loss:.3f}')

# 保存模型
# torch.save(model.state_dict(), 'transformer_model.pth') 