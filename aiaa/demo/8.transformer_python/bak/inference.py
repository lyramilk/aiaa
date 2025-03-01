import torch
from transformer import Transformer  # 导入 Transformer 模型

# 假设您已经有了以下数据预处理步骤：
# 1. 使用 spacy 进行分词
# 2. 构建词汇表 (src_vocab, tgt_vocab)
# 3. 将文本转换为索引序列
# 4. 定义填充索引 (src_pad_idx, tgt_pad_idx)
# 5. 加载训练好的模型

# 这里提供一个简化的示例，您需要根据实际情况进行修改
# 假设您已经有了以下变量：
# src_vocab_size = 10000  # 源语言词汇表大小
# tgt_vocab_size = 10000  # 目标语言词汇表大小
# src_pad_idx = 1  # 源语言填充索引
# tgt_pad_idx = 1  # 目标语言填充索引
# 模型参数
# num_layers = 6
# d_model = 512
# num_heads = 8
# d_ff = 2048
# dropout = 0.1

# 加载模型
# model = Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, num_layers, d_model, num_heads, d_ff, dropout)
# model.load_state_dict(torch.load('transformer_model.pth'))
# model.eval()  # 设置为评估模式

def translate_sentence(model, sentence, src_vocab, tgt_vocab, src_pad_idx, max_len=50):
    """
    翻译句子
    """
    model.eval()

    # 分词并转换为索引
    #tokens = [token.text.lower() for token in spacy_en.tokenizer(sentence)]
    tokens = sentence.lower().split() # 简化示例
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [src_vocab.stoi[token] for token in tokens] # 假设有 .stoi 属性

    # 转换为张量并添加批次维度
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0)

    # 创建源掩码
    src_mask = (src_tensor != src_pad_idx).unsqueeze(1).unsqueeze(2)

    # 使用编码器获取编码器输出
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)

    # 创建初始目标序列（仅包含 <sos> 标记）
    tgt_indexes = [tgt_vocab.stoi['<sos>']]

    for i in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indexes).unsqueeze(0)

        # 创建目标掩码
        tgt_mask = model.generate_mask(src_tensor, tgt_tensor)[1]

        # 使用解码器获取输出
        with torch.no_grad():
            output = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
            output = model.fc(output)

        # 获取预测的下一个标记
        pred_token = output.argmax(2)[:, -1].item()
        tgt_indexes.append(pred_token)

        # 如果预测到 <eos> 标记，则停止
        if pred_token == tgt_vocab.stoi['<eos>']:
            break

    # 将索引转换回文本
    translated_tokens = [tgt_vocab.itos[i] for i in tgt_indexes] # 假设有 .itos 属性

    return translated_tokens[1:]  # 去除 <sos> 标记

# 示例
# sentence = "This is a test sentence."
# translated_sentence = translate_sentence(model, sentence, src_vocab, tgt_vocab, src_pad_idx)
# print(f'Translated sentence: {" ".join(translated_sentence)}') 