import torch
from model import Transformer
from config import Config
import transformers
from modelscope import snapshot_download

# 加载 tokenizer
tokenizer_path = snapshot_download("lyramilk/deepseek_v3_tokenizer")
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

# 预测函数
def predict(model, config, source_text, device):
    model.to(device)
    model.eval()

    # 将输入文本转换为 token
    source = tokenizer.encode(source_text, add_special_tokens=False)
    source = torch.tensor(source, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_length]

    # 创建 source_mask
    source_mask = torch.ones((1, source.size(1), source.size(1)), dtype=torch.long).to(device)

    # 初始 target 为 <pad> token
    target = torch.zeros((1, 1), dtype=torch.long).to(device)  # [1, 1]

    with torch.no_grad():
        for _ in range(config.max_seq_length):
            # 创建 target_mask
            target_mask = torch.ones((1, target.size(1), target.size(1)), dtype=torch.long).to(device)
            for i in range(1, target.size(1)):
                target_mask[0, i, :i] = 1

            # 预测
            output = model(source, target, source_mask, target_mask)  # [1, seq_length, vocab_size]

            # 获取下一个 token 的预测
            next_token_probs = output[0, -1, :]  # [vocab_size]
            next_token = torch.argmax(next_token_probs).item()

            # 将预测的 token 添加到 target
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
            target = torch.cat((target, next_token_tensor), dim=1)  # [1, seq_length + 1]

            # 如果预测到 <eos> token，则停止预测
            if next_token == tokenizer.eos_token_id:
                break

    # 将预测的 token 转换回文本
    predicted_tokens = target[0, 1:].tolist()  # 去掉开头的 <pad> token
    predicted_text = tokenizer.decode(predicted_tokens)

    return predicted_text

# 主函数
if __name__ == "__main__":
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(config)

    # 加载训练好的模型权重（这里假设你已经训练好了模型）
    # model.load_state_dict(torch.load("transformer_model.pth"))

    # 预测
    source_text = "This is a test sentence."
    predicted_text = predict(model, config, source_text, device)
    print(f"Source: {source_text}")
    print(f"Predicted: {predicted_text}") 