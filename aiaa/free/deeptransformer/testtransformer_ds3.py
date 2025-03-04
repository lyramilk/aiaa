import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F

block_size = 128
gemm_impl: Literal["bf16"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"



"""
{
    "vocab_size": 129280,
    "dim": 7168,
    "inter_dim": 18432,
    "moe_inter_dim": 2048,
    "n_layers": 61,
    "n_dense_layers": 3,
    "n_heads": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "n_activated_experts": 8,
    "n_expert_groups": 8,
    "n_limited_groups": 4,
    "route_scale": 2.5,
    "score_func": "sigmoid",
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "dtype": "fp8"
}
"""


@dataclass
class ModelArgs:
    """
    用于定义模型Args和超Args的数据类。

    Attributes:
        max_seq_len (int): 最大序列长度。
        dtype (Literal["bf16"]): 计算的数据类型。
        vocab_size (int): 词汇表大小。
        dim (int): 模型维度。
        inter_dim (int): MLP层的中间维度。
        moe_inter_dim (int): MoE层的中间维度。
        n_layers (int): Transformer层的数量。
        n_dense_layers (int): 模型中的密集层数量。
        n_heads (int): 注意力头的数量。
        n_routed_experts (int): MoE层的路由专家数量。
        n_shared_experts (int): MoE层的共享专家数量。
        n_activated_experts (int): MoE层中激活的专家数量。
        score_func (Literal["softmax", "sigmoid"]): MoE路由的评分函数。
        route_scale (float): 路由分数的缩放因子。
        qk_nope_head_dim (int): 无位置嵌入的查询-键投影的维度。
        qk_rope_head_dim (int): 带旋转嵌入的查询-键投影的维度。
        v_head_dim (int): 值投影的维度。
        original_seq_len (int): 原始序列长度。
        rope_theta (float): 旋转位置编码的基数。
        rope_factor (float): 扩展序列长度的缩放因子。
        beta_fast (int): 快速beta校正因子。
        beta_slow (int): 慢速beta校正因子。
    """
    # 最大序列长度
    max_seq_len: int = 4096 * 4
    # 数据类型
    dtype: Literal["bf16"] = "bf16"
    # 词汇表大小
    vocab_size: int = 102400
    # 模型维度
    dim: int = 2048
    # MLP层的中间维度
    inter_dim: int = 10944
    # MoE层的中间维度
    moe_inter_dim: int = 1408
    # Transformer层的数量
    n_layers: int = 27
    # 密集层数量
    n_dense_layers: int = 1
    # 注意力头数量
    n_heads: int = 16
    # MoE层的路由专家数量
    n_routed_experts: int = 64
    # MoE层的共享专家数量
    n_shared_experts: int = 2
    # MoE层中激活的专家数量
    n_activated_experts: int = 6
    # MoE路由的评分函数
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    # 路由分数的缩放因子
    route_scale: float = 1.
    # 无位置嵌入的查询-键投影的维度
    qk_nope_head_dim: int = 128
    # 带旋转嵌入的查询-键投影的维度
    qk_rope_head_dim: int = 64
    # 值投影的维度
    v_head_dim: int = 128
    # 原始序列长度
    original_seq_len: int = 4096
    # 旋转位置编码的基数
    rope_theta: float = 10000.0
    # 扩展序列长度的缩放因子
    rope_factor: float = 40
    # 快速beta校正因子
    beta_fast: int = 32
    # 慢速beta校正因子
    beta_slow: int = 1


class Embedding(nn.Module):
    """
    嵌入层。
    """
    def __init__(self, vocab_size: int, dim: int):
        """
        初始化嵌入层。

        Args:
            vocab_size (int): 词汇表大小。
            dim (int): 嵌入维度。
        """
        # 词汇表大小
        self.vocab_size = vocab_size
        # 嵌入维度
        self.dim = dim
        # 权重参数
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        嵌入层的前向传播。

        Args:
            x (torch.Tensor): 包含词元索引的输入张量。

        Returns:
            torch.Tensor: 嵌入表示。
        """
        # 使用F.embedding函数计算嵌入
        y = F.embedding(x, self.weight)
        # 返回嵌入结果
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    线性变换函数。

    Args:
        x (torch.Tensor): 输入张量。
        weight (torch.Tensor): 权重张量。
        bias (Optional[torch.Tensor]): 偏置张量。

    Returns:
        torch.Tensor: 线性变换后的张量。
    """
    # 使用F.linear函数计算线性变换
    y = F.linear(x, weight, bias)
    # 返回结果
    return y


class Linear(nn.Module):
    """
    线性层。
    """
    # 默认数据类型
    dtype = torch.bfloat16
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        """
        初始化线性层。

        Args:
            in_features (int): 输入特征数量。
            out_features (int): 输出特征数量。
            bias (bool): 是否包含偏置项。默认为False。
            dtype (optional): 层的数据类型。默认为`torch.bfloat16`。
        """
        # 使用父类初始化
        super().__init__()
        # 输入特征数量
        self.in_features = in_features
        # 输出特征数量
        self.out_features = out_features
        # 权重参数
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype or self.dtype))
        # 如果包含偏置项
        if bias:
            # 偏置参数
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype or self.dtype))
        # 否则
        else:
            # 注册偏置为None
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        线性层的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 线性变换后的张量。
        """
        # 使用linear函数计算线性变换
        return linear(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    """
    均方根归一化层。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化均方根归一化层。

        Args:
            dim (int): 输入特征维度。
            eps (float): 用于数值稳定性的小常数。默认为1e-6。
        """
        # 使用父类初始化
        super().__init__()
        # 权重参数
        self.weight = nn.Parameter(torch.ones(dim))
        # 小常数
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        均方根归一化层的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 归一化后的张量。
        """
        # 计算均方根
        x_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 归一化
        x = x * x_rms
        # 缩放
        x = x * self.weight
        # 返回结果
        return x


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    预计算基于频率的复指数值，用于旋转位置嵌入。

    Args:
        args (ModelArgs): 包含位置嵌入Args的模型Args。

    Returns:
        torch.Tensor: 位置嵌入的预计算复指数值。
    """
    # 嵌入维度
    dim = args.qk_rope_head_dim
    # 最大序列长度
    seqlen = args.max_seq_len
    # 快速衰减因子
    beta_fast = args.beta_fast
    # 慢速衰减因子
    beta_slow = args.beta_slow
    # 指数计算的基值
    base = args.rope_theta
    # 校正因子
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        计算旋转位置嵌入中给定旋转数的校正维度。

        Args:
            num_rotations (float): 要计算校正的旋转数。
            dim (int): 嵌入空间的维度。
            base (float): 指数计算的基值。
            max_seq_len (int): 最大序列长度。

        Returns:
            float: 基于输入Args的校正维度。
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        计算旋转位置嵌入的校正维度范围。

        Args:
            low_rot (float): 旋转数的下界。
            high_rot (float): 旋转数的上界。
            dim (int): 嵌入空间的维度。
            base (float): 指数计算的基值。
            max_seq_len (int): 最大序列长度。

        Returns:
            Tuple[int, int]: 校正维度的范围（低，高），限制在有效索引内。
        """
        # 计算低旋转数的校正维度
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        # 计算高旋转数的校正维度
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        # 返回校正维度的范围
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        计算线性斜坡函数，用于平滑最小值和最大值范围之间的值。

        Args:
            min (float): 斜坡函数的最小值。
            max (float): 斜坡函数的最大值。
            dim (int): 斜坡张量的维度。

        Returns:
            torch.Tensor: 形状为(dim,)的张量，值在0和1之间线性插值，
                限制在范围[0, 1]内。
        """
        # 如果最小值和最大值相等
        if min == max:
            # 增加一个很小的值
            max += 0.001
        # 计算线性函数
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        # 限制在范围[0, 1]内
        ramp_func = torch.clamp(linear_func, 0, 1)
        # 返回结果
        return ramp_func

    # 计算频率
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # 如果序列长度大于原始序列长度
    if seqlen > args.original_seq_len:
        # 计算校正范围
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        # 计算平滑因子
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        # 计算频率
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # 计算时间序列
    t = torch.arange(seqlen)
    # 计算频率
    freqs = torch.outer(t, freqs)
    # 计算复指数值
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # 返回结果
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    将旋转位置嵌入应用于输入张量。

    Args:
        x (torch.Tensor): 要应用位置嵌入的输入张量。
        freqs_cis (torch.Tensor): 位置嵌入的预计算复指数值。

    Returns:
        torch.Tensor: 应用了旋转嵌入的张量。
    """
    # 数据类型
    dtype = x.dtype
    # 将输入张量转换为复数
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # 将频率嵌入转换为适当形状
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # 将输入张量与频率嵌入相乘
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    # 返回结果
    return y.to(dtype)


class MLA(nn.Module):
    """
    多头线性注意力层。
    """
    def __init__(self, args: ModelArgs):
        """
        初始化多头线性注意力层。

        Args:
            args (ModelArgs): 模型参数。
        """
        # 使用父类初始化
        super().__init__()
        # 模型维度
        self.dim = args.dim
        # 注意力头数量
        self.n_heads = args.n_heads
        # 本地头数量
        self.n_local_heads = args.n_heads
        # 无位置嵌入的头维度
        self.qk_nope_head_dim = args.qk_nope_head_dim
        # 带旋转嵌入的头维度
        self.qk_rope_head_dim = args.qk_rope_head_dim
        # 值头维度
        self.v_head_dim = args.v_head_dim
        # 头维度
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # 查询投影
        self.wq = Linear(args.dim, args.n_heads * self.head_dim)
        # 键投影
        self.wk = Linear(args.dim, args.n_heads * self.head_dim)
        # 值投影
        self.wv = Linear(args.dim, args.n_heads * self.v_head_dim)
        # 输出投影
        self.wo = Linear(args.n_heads * self.v_head_dim, args.dim)
        # 注意力实现
        self.attn_impl = attn_impl

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        多头线性注意力层的前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            start_pos (int): 开始位置。
            freqs_cis (torch.Tensor): 旋转位置编码。
            mask (Optional[torch.Tensor]): 注意力掩码。

        Returns:
            torch.Tensor: 注意力层的输出。
        """
        # 获取批次大小和序列长度
        bsz, seqlen, _ = x.shape
        # 计算结束位置
        end_pos = start_pos + seqlen
        # 计算查询
        q = self.wq(x)
        # 将查询张量变形为(batch_size, seq_len, n_local_heads, head_dim)
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # 将查询张量拆分为非位置查询和旋转位置查询
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # 应用旋转位置嵌入
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        # 计算键
        k = self.wk(x)
        # 将键张量变形为(batch_size, seq_len, n_local_heads, head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # 将键张量拆分为非位置键和旋转位置键
        k_nope, k_pe = torch.split(k, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # 应用旋转位置嵌入
        k_pe = apply_rotary_emb(k_pe, freqs_cis)
        # 计算值
        v = self.wv(x)
        # 将值张量变形为(batch_size, seq_len, n_local_heads, v_head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.v_head_dim)
        # 计算注意力分数
        scores = (torch.einsum("bshd,bthd->bsht", q_nope, k_nope) +
                  torch.einsum("bshd,bthd->bsht", q_pe, k_pe)) * self.head_dim ** -0.5
        # 如果掩码不为None
        if mask is not None:
            # 应用掩码
            scores = scores + mask
        # 计算注意力分数
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        # 计算输出
        x = torch.einsum("bsht,bthd->bshd", scores, v)
        # 计算输出
        x = self.wo(x.flatten(2))
        # 返回结果
        return x


class MLP(nn.Module):
    """
    多层感知机。
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        初始化多层感知机。

        Args:
            dim (int): 输入维度。
            inter_dim (int): 中间维度。
        """
        # 使用父类初始化
        super().__init__()
        # 向上投影
        self.w1 = Linear(dim, inter_dim)
        # 向下投影
        self.w2 = Linear(inter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多层感知机的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 多层感知机的输出。
        """
        # 计算输出
        return self.w2(F.silu(self.w1(x)))


class Gate(nn.Module):
    """
    门控层，用于MoE路由。
    """
    def __init__(self, args: ModelArgs):
        """
        初始化门控层。

        Args:
            args (ModelArgs): 模型参数。
        """
        # 使用父类初始化
        super().__init__()
        # 模型维度
        self.dim = args.dim
        # 路由专家数量
        self.n_routed_experts = args.n_routed_experts
        # 激活专家数量
        self.n_activated_experts = args.n_activated_experts
        # 评分函数
        self.score_func = args.score_func
        # 路由缩放
        self.route_scale = args.route_scale
        # 路由器
        self.router = Linear(args.dim, args.n_routed_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        门控层的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 路由和分数。
        """
        # 计算得分
        scores = self.router(x)
        # 原始得分
        original_scores = scores
        # 计算得分
        scores = scores.softmax(dim=-1, dtype=torch.float32)
        # 计算得分
        indices = torch.topk(scores, self.n_activated_experts, dim=-1)[1]
        # 计算权重
        weights = original_scores.gather(1, indices)
        # 缩放权重
        weights *= self.route_scale
        # 返回结果
        return indices, weights


class Expert(nn.Module):
    """
    专家层，用于MoE。
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        专家层，用于MoE。

        Args:
            dim (int): 输入维度。
            inter_dim (int): 中间维度。
        """
        # 使用父类初始化
        super().__init__()
        # 多层感知机
        self.mlp = MLP(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expert层的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 专家计算后的输出张量。
        """
        # 计算输出
        return self.mlp(x)


class MoE(nn.Module):
    """
    混合专家模型。
    """
    def __init__(self, args: ModelArgs):
        """
        初始化混合专家模型。

        Args:
            args (ModelArgs): 模型参数。
        """
        # 使用父类初始化
        super().__init__()
        # 模型维度
        self.dim = args.dim
        # 路由专家数量
        self.n_routed_experts = args.n_routed_experts
        # 本地专家数量
        self.n_local_experts = args.n_routed_experts
        # 共享专家数量
        self.n_shared_experts = args.n_shared_experts
        # 激活专家数量
        self.n_activated_experts = args.n_activated_experts
        # 门控
        self.gate = Gate(args)
        # 路由专家列表
        self.routed_experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) for _ in range(args.n_routed_experts)])
        # 共享专家列表
        self.shared_experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) for _ in range(args.n_shared_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        混合专家模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 混合专家模型的输出。
        """
        # 获取形状
        B, L, D = x.shape
        # 重塑输入
        x_reshaped = x.view(-1, D)
        # 获取路由和分数
        routes, scores = self.gate(x_reshaped)
        # 初始化输出
        y = torch.zeros_like(x_reshaped)
        # 遍历路由
        for i, (route, score) in enumerate(zip(routes.t(), scores.t())):
            # 获取专家
            expert = self.routed_experts[i]
            # 计算输出
            y[route] += score * expert(x_reshaped[route])
        # 重塑输出
        y = y.view(B, L, D)
        # 返回结果
        return y


class Block(nn.Module):
    """
    初始化Transformer块。
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        初始化Transformer块。

        Args:
            layer_id (int): 层ID。
            args (ModelArgs): 模型参数。
        """
        # 使用父类初始化
        super().__init__()
        # 层ID
        self.layer_id = layer_id
        # 注意力层
        self.attn = MLA(args)
        # 注意力层归一化
        self.attn_norm = RMSNorm(args.dim)
        # 如果层ID小于密集层数量
        if layer_id < args.n_dense_layers:
            # 前馈网络
            self.ffn = MLP(args.dim, args.inter_dim)
        # 否则
        else:
            # 混合专家模型
            self.ffn = MoE(args)
        # 前馈网络归一化
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Transformer块的前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            start_pos (int): 序列中的起始位置。
            freqs_cis (torch.Tensor): 旋转嵌入的预计算复指数值。
            mask (Optional[torch.Tensor]): 用于从注意力中排除某些位置的掩码张量。

        Returns:
            torch.Tensor: 块计算后的输出张量。
        """
        # 注意力
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        # 前馈网络
        x = x + self.ffn(self.ffn_norm(x))
        # 返回输出
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        """
        初始化Transformer模型。

        Args:
            args (ModelArgs): 包含TransformerArgs的模型Args。
            
        Attributes:
            max_seq_len (int): Transformer的最大序列长度。
            embed (nn.Module): 输入词元的嵌入层。
            layers (torch.nn.ModuleList): Transformer块列表。
            norm (nn.Module): 在所有块之后应用的层归一化。
            head (nn.Module): 映射到词汇表大小的输出投影层。
            freqs_cis (torch.Tensor): 旋转嵌入的预计算复指数值。
        """
        # 线性层数据类型
        Linear.dtype = torch.bfloat16
        # 初始化
        super().__init__()
        # 最大序列长度
        self.max_seq_len = args.max_seq_len
        # 输入词元嵌入层
        self.embed = Embedding(args.vocab_size, args.dim)
        # 层列表
        self.layers = torch.nn.ModuleList()
        # 遍历层
        for layer_id in range(args.n_layers):
            # 添加块
            self.layers.append(Block(layer_id, args))
        # 层归一化
        self.norm = RMSNorm(args.dim)
        # 输出投影层
        self.head = Linear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        # 注册缓存
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Transformer模型的前向传播。

        Args:
            tokens (torch.Tensor): 形状为(batch_size, seq_len)的词元ID输入张量。
            start_pos (int, optional): 旋转嵌入序列中的起始位置。默认为0。

        Returns:
            torch.Tensor: 形状为(batch_size, vocab_size)的logits张量。
        """
        # 获取批次大小和序列长度
        B, L = tokens.shape
        # 嵌入词元
        h = self.embed(tokens)
        # 获取复指数频率
        freqs_cis = self.freqs_cis[start_pos:start_pos + L]
        # 创建掩码
        mask = None
        if L > 1:
            # 创建因果掩码
            mask = torch.full((L, L), float("-inf"), device=tokens.device)
            # 设置上三角部分为0
            mask = torch.triu(mask, diagonal=1)
        # 遍历层
        for layer in self.layers:
            # 通过层
            h = layer(h, start_pos, freqs_cis, mask)
        # 归一化
        h = self.norm(h)
        # 获取最后一个词元的输出
        output = self.head(h[:, -1, :])
        # 返回输出
        return output


if __name__ == "__main__":
    # 设置默认数据类型
    torch.set_default_dtype(torch.bfloat16)
    # 设置默认设备
    torch.set_default_device("cuda")
    # 设置随机种子
    torch.manual_seed(0)
    # 创建模型参数
    args = ModelArgs()
    # 创建输入
    x = torch.randint(0, args.vocab_size, (2, 128))
    # 创建模型
    model = Transformer(args)
    # 打印输出大小
    print(model(x).size())
