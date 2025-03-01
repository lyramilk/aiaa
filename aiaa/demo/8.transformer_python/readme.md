# Transformer Python 实现

本目录包含了Transformer模型的Python实现，主要用于自然语言处理任务，特别是机器翻译。

## 项目文件

- `testtransformer.py` - Transformer模型的基础实现
- `testtransformer2.py` - 改进版Transformer模型实现，包含完整的训练和预测功能，用于中英文翻译任务

## 技术细节

`testtransformer2.py` 实现了完整的Transformer架构，包括：

1. 位置编码 (PositionalEncoding)
2. 多头注意力机制 (MultiHeadAttention)
3. 前馈神经网络 (PoswiseFeedForwardNet)
4. 编码器和解码器 (Encoder, Decoder)
5. 完整的训练和预测流程

该实现使用了DeepSeek的tokenizer进行分词，并包含了自定义的数据集类用于处理翻译数据。模型训练过程中，当平均损失低于0.05时会自动停止训练。

### 注意事项

- 在预测过程中，需要确保编码器和解码器输入的批次大小匹配
- 注意力掩码的维度需要与输入序列维度匹配，否则会出现维度不匹配错误
