# DeepTransformer 实现

本目录包含了高级Transformer模型的实现，专注于大规模语言模型的核心技术。

## 项目文件

- `testtransformer_ds3.py` - 高级Transformer模型实现，包含混合专家模型(MoE)和旋转位置编码等现代技术
- `kernel.py` - 基础计算核心，提供模型所需的辅助函数

## 技术细节

`testtransformer_ds3.py` 实现了现代大型语言模型中使用的先进技术，包括：

1. 旋转位置编码 (RoPE) - 相对位置编码的一种实现，在长序列处理中表现优异
2. 混合专家模型 (MoE) - 通过专家路由机制提高模型容量和效率
3. RMSNorm 归一化 - 比传统LayerNorm更高效的归一化方法
4. 多头线性注意力 (MLA) - 优化的注意力机制实现

注意：该文件原本包含分布式计算组件和量化代码，现已移除，仅保留单机版本和基本功能。同时，ModelArgs类中未使用的属性也已被移除，使代码更加简洁。

`kernel.py` 提供了基础计算支持，包括：

1. 块大小配置 - 为模型提供默认的计算块大小

## 使用方法

该模型实现主要用于学习和研究现代大型语言模型的架构。可以通过实例化`Transformer`类并提供适当的`ModelArgs`来创建模型：

```python
args = ModelArgs(
    vocab_size=32000,
    dim=2048,
    n_layers=24,
    n_heads=16,
    max_seq_len=4096
)
model = Transformer(args)
```

## 注意事项

- 模型默认使用bfloat16精度
- 混合专家模型需要较大的内存，请确保有足够的计算资源 