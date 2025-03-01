import torch
import unittest
from transformer import Transformer
from encoding import PositionalEncoding
from train import train

class TestTransformer(unittest.TestCase):
    def setUp(self):
        # 设置一些测试参数
        self.src_vocab_size = 100
        self.tgt_vocab_size = 100
        self.src_pad_idx = 0
        self.tgt_pad_idx = 0
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 6
        self.d_ff = 2048
        self.dropout = 0.1
        self.max_len = 50

        # 创建模型实例
        self.model = Transformer(self.src_vocab_size, self.tgt_vocab_size, self.src_pad_idx, self.tgt_pad_idx,
                                 self.num_layers, self.d_model, self.num_heads, self.d_ff, self.dropout)
        
    def test1(self):
        train_data = [
            "小米音箱是小米公司的产品<|endoftext|>",
            "小米手机是小米公司的产品<|endoftext|>",
            "土豆音箱是土豆公司的产品<|endoftext|>",
            "土豆手表是土豆公司的产品<|endoftext|>",
        ]
        train(self.model, train_data)
        self.model.eval("s")

    def test_transformer_output_shape(self):
        # 测试 Transformer 模型的输出维度
        batch_size = 4
        src_seq_len = 20
        tgt_seq_len = 15

        # 创建随机输入数据
        src = torch.randint(1, self.src_vocab_size, (batch_size, src_seq_len))
        tgt = torch.randint(1, self.tgt_vocab_size, (batch_size, tgt_seq_len))

        # 前向传播
        output = self.model(src, tgt)

        # 检查输出维度
        self.assertEqual(output.shape, (batch_size, tgt_seq_len, self.tgt_vocab_size))

    def test_positional_encoding_output_shape(self):
        # 测试位置编码的输出维度
        batch_size = 4
        seq_len = 20

        # 创建位置编码实例
        pos_encoding = PositionalEncoding(self.d_model, self.max_len)

        # 创建随机输入数据
        x = torch.randn(batch_size, seq_len, self.d_model)

        # 应用位置编码
        output = pos_encoding(x)

        # 检查输出维度
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_positional_encoding_values(self):
        # 测试位置编码的值是否在预期范围内
        pos_encoding = PositionalEncoding(self.d_model, self.max_len)
        x = torch.zeros(1, 1, self.d_model)  # 使用零张量来测试
        output = pos_encoding(x)

        # 检查输出是否在 [-1, 1] 之间（因为使用了 sin 和 cos 函数）
        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))

if __name__ == '__main__':
    unittest.main() 