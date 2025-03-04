import torch
import torch.nn as nn
import math


class RopePositionEmbedding:
    def __init__(self, dim: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16):
        # 初始化旋转位置编码
        # dim: 嵌入维度
        # max_seq_len: 最大序列长度
        # dtype: 张量数据类型
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        
        # 确保dim是偶数，因为我们需要成对处理维度
        assert dim % 2 == 0, "维度必须是偶数"
        
        # 创建位置索引
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        
        # 创建维度索引
        dim_indices = torch.arange(0, dim // 2, dtype=torch.float)
        
        # 计算频率因子
        freq = 1.0 / (10000 ** (2 * dim_indices / dim))
        
        # 计算位置和频率的外积
        # [max_seq_len, dim/2]
        angles = torch.outer(position, freq)
        
        # 创建复数旋转因子
        # 使用欧拉公式 e^(i*theta) = cos(theta) + i*sin(theta)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # 保存余弦和正弦值以便在forward中使用
        self.register_buffer = False  # 非nn.Module类
        self.cos = cos.to(dtype=dtype)  # [max_seq_len, dim/2]
        self.sin = sin.to(dtype=dtype)  # [max_seq_len, dim/2]
        
        # 添加缓存字典，用于存储不同序列长度的旋转矩阵
        self.cache = {}
        
        # 预计算常用序列长度的旋转矩阵
        self._precompute_rotations()

    def _precompute_rotations(self):
        # 预计算常用序列长度的旋转矩阵
        # 这里我们预计算一些常见的序列长度，如32、64、128、256、512、1024等
        common_lengths = [32, 64, 128, 256, 512, 1024]
        common_lengths = [l for l in common_lengths if l <= self.max_seq_len]
        
        for seq_len in common_lengths:
            # 获取当前序列长度的旋转因子
            cos = self.cos[:seq_len]  # [seq_len, dim/2]
            sin = self.sin[:seq_len]  # [seq_len, dim/2]
            
            # 将旋转因子存入缓存
            self.cache[seq_len] = (cos, sin)
    
    def _get_rotation_matrices(self, seq_len):
        # 从缓存中获取旋转矩阵，如果不存在则计算并缓存
        if seq_len not in self.cache:
            # 获取当前序列长度的旋转因子
            cos = self.cos[:seq_len]  # [seq_len, dim/2]
            sin = self.sin[:seq_len]  # [seq_len, dim/2]
            
            # 将旋转因子存入缓存
            self.cache[seq_len] = (cos, sin)
        
        return self.cache[seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用旋转位置编码
        # x: 输入张量，形状为 [batch_size, seq_len, dim]
        batch_size, seq_len, dim = x.shape
        
        # 确保序列长度不超过最大长度
        assert seq_len <= self.max_seq_len, f"输入序列长度 {seq_len} 超过了最大长度 {self.max_seq_len}"
        
        # 确保维度匹配
        assert dim == self.dim, f"输入维度 {dim} 与初始化维度 {self.dim} 不匹配"
        
        # 从缓存中获取旋转矩阵
        cos, sin = self._get_rotation_matrices(seq_len)
        
        # 将输入张量重塑为便于旋转操作的形式
        # 将维度分成两半，一半用于cos旋转，一半用于sin旋转
        x_reshaped = x.view(batch_size, seq_len, dim // 2, 2)
        
        # 分离出奇偶维度
        x_even = x_reshaped[..., 0]  # [batch_size, seq_len, dim/2]
        x_odd = x_reshaped[..., 1]   # [batch_size, seq_len, dim/2]
        
        # 应用旋转变换
        # 对于位置i和维度2j: 
        # [x_even_i,j, x_odd_i,j] -> [x_even_i,j*cos_i,j - x_odd_i,j*sin_i,j, x_even_i,j*sin_i,j + x_odd_i,j*cos_i,j]
        rotated_x_even = x_even * cos.unsqueeze(0) - x_odd * sin.unsqueeze(0)
        rotated_x_odd = x_even * sin.unsqueeze(0) + x_odd * cos.unsqueeze(0)
        
        # 重新组合旋转后的向量
        rotated_x = torch.stack([rotated_x_even, rotated_x_odd], dim=-1)
        
        # 恢复原始形状
        rotated_x = rotated_x.view(batch_size, seq_len, dim)
        
        return rotated_x

# 测试代码
rope = RopePositionEmbedding(dim=128, max_seq_len=1024, dtype=torch.bfloat16)
x = torch.randn(1, 1024, 128)
y = rope(x)
print(y.shape)

# 测试缓存功能
x2 = torch.randn(1, 512, 128)
y2 = rope(x2)
print(y2.shape)
print(f"缓存的序列长度: {list(rope.cache.keys())}")

