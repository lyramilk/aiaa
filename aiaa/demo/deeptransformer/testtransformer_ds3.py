import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    """
    用于定义模型Args和超Args的数据类。

    Attributes:
        max_batch_size (int): 最大批处理大小。
        max_seq_len (int): 最大序列长度。
        dtype (Literal["bf16", "fp8"]): 计算的数据类型。
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
        n_expert_groups (int): 专家组的数量。
        n_limited_groups (int): MoE路由的限制组数量。
        score_func (Literal["softmax", "sigmoid"]): MoE路由的评分函数。
        route_scale (float): 路由分数的缩放因子。
        q_lora_rank (int): 查询投影的LoRA秩。
        kv_lora_rank (int): 键值投影的LoRA秩。
        qk_nope_head_dim (int): 无位置嵌入的查询-键投影的维度。
        qk_rope_head_dim (int): 带旋转嵌入的查询-键投影的维度。
        v_head_dim (int): 值投影的维度。
        original_seq_len (int): 原始序列长度。
        rope_theta (float): 旋转位置编码的基数。
        rope_factor (float): 扩展序列长度的缩放因子。
        beta_fast (int): 快速beta校正因子。
        beta_slow (int): 慢速beta校正因子。
        mscale (float): 扩展注意力的缩放因子。
    """
    # 最大批处理大小
    max_batch_size: int = 8
    # 最大序列长度
    max_seq_len: int = 4096 * 4
    # 数据类型
    dtype: Literal["bf16", "fp8"] = "bf16"
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
    # 专家组的数量
    n_expert_groups: int = 1
    # MoE路由的限制组数量
    n_limited_groups: int = 1
    # MoE路由的评分函数
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    # 路由分数的缩放因子
    route_scale: float = 1.
    # 查询投影的LoRA秩
    q_lora_rank: int = 0
    # 键值投影的LoRA秩
    kv_lora_rank: int = 512
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
    # 扩展注意力的缩放因子
    mscale: float = 1.


class ParallelEmbedding(nn.Module):
    """
    支持跨分布式进程并行化的嵌入层。

    Args:
        vocab_size (int): 词汇表大小。
        dim (int): 嵌入维度。
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        # 词汇表大小
        self.vocab_size = vocab_size
        # 嵌入维度
        self.dim = dim
        # 断言词汇表大小必须能被世界大小整除
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        # 词汇表大小除以世界大小
        self.part_vocab_size = (vocab_size // world_size)
        # 词汇表起始索引
        self.vocab_start_idx = rank * self.part_vocab_size
        # 词汇表结束索引
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        # 权重参数
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        并行嵌入层的前向传播。

        Args:
            x (torch.Tensor): 包含词元索引的输入张量。

        Returns:
            torch.Tensor: 嵌入表示。

        异常:
            ValueError: 如果未定义`world_size`。
        """
        # 如果世界大小大于1
        if world_size > 1:
            # 创建一个掩码，用于屏蔽不在当前进程词汇表范围内的词元
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            # 将不在当前进程词汇表范围内的词元索引减去词汇表起始索引
            x = x - self.vocab_start_idx
            # 将不在当前进程词汇表范围内的词元设置为0
            x[mask] = 0
        # 使用F.embedding函数计算嵌入
        y = F.embedding(x, self.weight)
        # 如果世界大小大于1
        if world_size > 1:
            # 将不在当前进程词汇表范围内的词元设置为0
            y[mask] = 0
            # 使用all_reduce函数将不在当前进程词汇表范围内的词元进行广播
            dist.all_reduce(y)
        # 返回嵌入结果
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    对输入数据应用线性变换: y = xA^T + b。
    此函数支持基于量化和张量格式的专门实现。

    Args:
        x (torch.Tensor): 输入张量。
        weight (torch.Tensor): 权重张量。它可能被量化，
            在某些情况下需要反量化。
        bias (Optional[torch.Tensor]): 要添加的偏置张量。默认为None。

    Returns:
        torch.Tensor: 线性变换的结果，根据输入Args可能涉及量化感知计算。

    Notes:
        - 如果`weight`被量化（例如，`element_size() == 1`），则使用反量化版本进行计算。
        - 如果`gemm_impl == "bf16"`，则应用反量化和`bf16` GEMM操作。
        - 对于其他情况，函数对`x`应用量化并使用`fp8_gemm`进行计算。
    """
    # 如果权重元素大小大于1
    if weight.element_size() > 1:
        # 使用F.linear函数计算线性变换
        return F.linear(x, weight, bias)
    # 如果gemm_impl为"bf16"
    elif gemm_impl == "bf16":
        # 使用weight_dequant函数反量化权重
        weight = weight_dequant(weight, weight.scale)
        # 使用F.linear函数计算线性变换
        return F.linear(x, weight, bias)
    else:
        # 使用act_quant函数量化输入
        x, scale = act_quant(x, block_size)
        # 使用fp8_gemm函数计算线性变换
        y = fp8_gemm(x, scale, weight, weight.scale)
        # 如果偏置不为None
        if bias is not None:
            # 将偏置加到结果上
            y += bias
        # 返回结果
        return y


class Linear(nn.Module):
    # 默认数据类型
    dtype = torch.bfloat16
    """
    支持量化权重和可选偏置的自定义线性层。

    Args:
        in_features (int): 输入特征数量。
        out_features (int): 输出特征数量。
        bias (bool): 是否包含偏置项。默认为False。
        dtype (optional): 层的数据类型。默认为`torch.bfloat16`。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        # 输入特征数量
        self.in_features = in_features
        # 输出特征数量
        self.out_features = out_features
        # 权重参数
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        # 如果权重元素大小为1
        if self.weight.element_size() == 1:
            # 计算输出特征数量除以block_size
            scale_out_features = (out_features + block_size - 1) // block_size
            # 计算输入特征数量除以block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            # 创建一个可训练的缩放参数
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            # 注册一个可训练的缩放参数
            self.register_parameter("scale", None)
        # 如果包含偏置项
        if bias:
            # 创建一个可训练的偏置参数
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            # 注册一个可训练的偏置参数
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        自定义线性层的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 线性计算后的变换张量。
        """
        # 使用linear函数计算线性变换
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    具有列并行性的线性层，在分布式进程间拆分输出特征。

    Args:
        in_features (int): 输入特征数量。
        out_features (int): 输出特征总数。
        bias (bool): 是否包含偏置项。默认为False。
        dtype (optional): 层的数据类型。默认为`torch.bfloat16`。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # 断言输出特征数量必须能被世界大小整除
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        # 计算输出特征数量除以世界大小
        self.part_out_features = out_features // world_size
        # 使用父类初始化
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        列并行线性层的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 经过列并行计算的变换张量。
        """
        # 使用linear函数计算线性变换
        y = linear(x, self.weight, self.bias)
        # 返回结果
        return y


class RowParallelLinear(Linear):
    """
    具有行并行性的线性层，在分布式进程间拆分输入特征。

    Args:
        in_features (int): 输入特征总数。
        out_features (int): 输出特征数量。
        bias (bool): 是否包含偏置项。默认为False。
        dtype (optional): 层的数据类型。默认为`torch.bfloat16`。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # 断言输入特征数量必须能被世界大小整除
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        # 计算输入特征数量除以世界大小
        self.part_in_features = in_features // world_size
        # 使用父类初始化
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        行并行线性层的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 经过行并行计算的变换张量。
        """
        # 使用linear函数计算线性变换
        y = linear(x, self.weight)
        # 如果世界大小大于1
        if world_size > 1:
            # 使用all_reduce函数将结果进行广播
            dist.all_reduce(y)
        # 如果包含偏置项
        if self.bias is not None:
            # 将偏置加到结果上
            y += self.bias
        # 返回结果
        return y


class RMSNorm(nn.Module):
    """
    均方根层归一化（RMSNorm）。

    Args:
        dim (int): 输入张量的维度。
        eps (float): 数值稳定性的epsilon值。默认为1e-6。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # 输入维度
        self.dim = dim
        # 数值稳定性的epsilon值
        self.eps = eps
        # 权重参数
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        RMSNorm的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 与输入形状相同的归一化张量。
        """
        # 使用F.rms_norm函数计算归一化
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


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
    多头注意力层（MLA）。

    Attributes:
        dim (int): 输入特征的维度。
        n_heads (int): 注意力头的数量。
        n_local_heads (int): 分布式系统的本地注意力头数量。
        q_lora_rank (int): 低秩查询投影的秩。
        kv_lora_rank (int): 低秩键/值投影的秩。
        qk_nope_head_dim (int): 非位置查询/键投影的维度。
        qk_rope_head_dim (int): 旋转位置查询/键投影的维度。
        qk_head_dim (int): 查询/键投影的总维度。
        v_head_dim (int): 值投影的维度。
        softmax_scale (float): 注意力计算中softmax的缩放因子。
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 输入特征维度
        self.dim = args.dim
        # 注意力头数量
        self.n_heads = args.n_heads
        # 本地注意力头数量
        self.n_local_heads = args.n_heads // world_size
        # 低秩查询投影的秩
        self.q_lora_rank = args.q_lora_rank
        # 低秩键/值投影的秩
        self.kv_lora_rank = args.kv_lora_rank
        # 非位置查询/键投影的维度
        self.qk_nope_head_dim = args.qk_nope_head_dim
        # 旋转位置查询/键投影的维度
        self.qk_rope_head_dim = args.qk_rope_head_dim
        # 查询/键投影的总维度
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        # 值投影的维度
        self.v_head_dim = args.v_head_dim

        # 如果低秩查询投影的秩为0
        if self.q_lora_rank == 0:
            # 创建列并行线性层
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 创建线性层
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            # 创建RMSNorm层
            self.q_norm = RMSNorm(self.q_lora_rank)
            # 创建列并行线性层
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        # 创建线性层
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        # 创建RMSNorm层
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # 创建列并行线性层
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        # 创建行并行线性层
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        # 计算softmax缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5
        # 如果序列长度大于原始序列长度
        if args.max_seq_len > args.original_seq_len:
            # 计算mscale
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            # 更新softmax缩放因子
            self.softmax_scale = self.softmax_scale * mscale * mscale
        # 如果attn_impl为naive
        if attn_impl == "naive":
            # 注册缓存
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        多头注意力层（MLA）的前向传播。

        Args:
            x (torch.Tensor): 形状为(batch_size, seq_len, dim)的输入张量。
            start_pos (int): 用于缓存的序列起始位置。
            freqs_cis (torch.Tensor): 旋转嵌入的预计算复指数值。
            mask (Optional[torch.Tensor]): 用于从注意力中排除某些位置的掩码张量。

        Returns:
            torch.Tensor: 与输入形状相同的输出张量。
        """
        # 获取输入张量的形状
        bsz, seqlen, _ = x.size()
        # 计算结束位置
        end_pos = start_pos + seqlen
        # 如果低秩查询投影的秩为0
        if self.q_lora_rank == 0:
            # 计算查询
            q = self.wq(x)
        else:
            # 计算查询
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        # 将查询张量变形为(batch_size, seq_len, n_local_heads, qk_head_dim)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        # 将查询张量拆分为非位置查询和旋转位置查询
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # 应用旋转位置嵌入
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        # 计算键/值
        kv = self.wkv_a(x)
        # 将键/值张量变形为(batch_size, seq_len, n_local_heads, kv_lora_rank + qk_rope_head_dim)
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.kv_lora_rank + self.qk_rope_head_dim)
        # 将键/值张量拆分为低秩键/值和旋转位置键/值
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # 应用旋转位置嵌入
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        # 如果attn_impl为naive
        if attn_impl == "naive":
            # 将非位置查询和旋转位置查询连接起来
            q = torch.cat([q_nope, q_pe], dim=-1)
            # 计算键/值
            kv = self.wkv_b(self.kv_norm(kv))
            # 将键/值张量变形为(batch_size, seq_len, n_local_heads, qk_nope_head_dim + v_head_dim)
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            # 将键/值张量拆分为低秩键/值和旋转位置键/值
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # 将低秩键/值和旋转位置键/值连接起来
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            # 更新缓存
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            # 计算注意力分数
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # 计算键/值
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            # 将键/值张量变形为(n_local_heads, -1, kv_lora_rank)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            # 计算非位置查询
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            # 更新缓存
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            # 更新缓存
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            # 计算注意力分数
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        # 如果掩码不为None
        if mask is not None:
            # 将掩码变形为(batch_size, 1, seq_len)
            mask = mask.unsqueeze(1)
            # 计算注意力分数
            scores += mask
        # 计算注意力分数
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        # 如果attn_impl为naive
        if attn_impl == "naive":
            # 计算输出
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # 计算输出
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            # 计算输出
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        # 计算输出
        x = self.wo(x.flatten(2))
        # 返回结果
        return x


class MLP(nn.Module):
    """
    多层感知器（MLP），用作前馈层。

    Attributes:
        w1 (nn.Module): 输入到隐藏层变换的线性层。
        w2 (nn.Module): 隐藏层到输出变换的线性层。
        w3 (nn.Module): 特征变换的额外线性层。
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        初始化MLP层。

        Args:
            dim (int): 输入和输出维度。
            inter_dim (int): 隐藏层维度。
        """
        super().__init__()
        # 创建列并行线性层
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        # 创建行并行线性层
        self.w2 = RowParallelLinear(inter_dim, dim)
        # 创建列并行线性层
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MLP层的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: MLP计算后的输出张量。
        """
        # 计算输出
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    混合专家模型（MoE）中用于路由输入的门控机制。

    Attributes:
        dim (int): 输入特征的维度。
        topk (int): 为每个输入激活的顶级专家数量。
        n_groups (int): 路由的组数。
        topk_groups (int): 将输入路由到的组数。
        score_func (str): 评分函数（'softmax'或'sigmoid'）。
        route_scale (float): 路由权重的缩放因子。
        weight (torch.nn.Parameter): 门控的可学习权重。
        bias (Optional[torch.nn.Parameter]): 门控的可选偏置项。
    """
    def __init__(self, args: ModelArgs):
        """
        初始化Gate模块。

        Args:
            args (ModelArgs): 包含门控Args的模型Args。
        """
        super().__init__()
        # 输入特征维度
        self.dim = args.dim
        # 为每个输入激活的顶级专家数量
        self.topk = args.n_activated_experts
        # 路由的组数
        self.n_groups = args.n_expert_groups
        # 将输入路由到的组数
        self.topk_groups = args.n_limited_groups
        # 评分函数
        self.score_func = args.score_func
        # 路由权重的缩放因子
        self.route_scale = args.route_scale
        # 路由权重
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # 偏置
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        门控机制的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 路由权重和选定的专家索引。
        """
        # 计算得分
        scores = linear(x, self.weight)
        # 如果评分函数为softmax
        if self.score_func == "softmax":
            # 计算得分
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            # 计算得分
            scores = scores.sigmoid()
        # 原始得分
        original_scores = scores
        # 如果偏置不为None
        if self.bias is not None:
            # 计算得分
            scores = scores + self.bias
        # 如果路由的组数大于1
        if self.n_groups > 1:
            # 将得分变形为(batch_size, n_groups, -1)
            scores = scores.view(x.size(0), self.n_groups, -1)
            # 如果偏置为None
            if self.bias is None:
                # 计算得分
                group_scores = scores.amax(dim=-1)
            else:
                # 计算得分
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            # 计算得分
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            # 计算掩码
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            # 计算得分
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        # 计算得分
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        # 计算权重
        weights = original_scores.gather(1, indices)
        # 如果评分函数为sigmoid
        if self.score_func == "sigmoid":
            # 计算权重
            weights /= weights.sum(dim=-1, keepdim=True)
        # 缩放权重
        weights *= self.route_scale
        # 返回权重和索引
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    混合专家模型（MoE）的专家层。

    Attributes:
        w1 (nn.Module): 输入到隐藏层变换的线性层。
        w2 (nn.Module): 隐藏层到输出变换的线性层。
        w3 (nn.Module): 特征变换的额外线性层。
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        初始化Expert层。

        Args:
            dim (int): 输入和输出维度。
            inter_dim (int): 隐藏层维度。
        """
        super().__init__()
        # 输入到隐藏层变换的线性层
        self.w1 = Linear(dim, inter_dim)
        # 隐藏层到输出变换的线性层
        self.w2 = Linear(inter_dim, dim)
        # 特征变换的额外线性层
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expert层的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 专家计算后的输出张量。
        """
        # 计算输出
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    混合专家（MoE）模块。

    Attributes:
        dim (int): 输入特征的维度。
        n_routed_experts (int): 模型中专家的总数。
        n_local_experts (int): 分布式系统中本地处理的专家数量。
        n_activated_experts (int): 为每个输入激活的专家数量。
        gate (nn.Module): 将输入路由到专家的门控机制。
        experts (nn.ModuleList): 专家模块列表。
        shared_experts (nn.Module): 应用于所有输入的共享专家。
    """
    def __init__(self, args: ModelArgs):
        """
        初始化MoE模块。

        Args:
            args (ModelArgs): 包含MoEArgs的模型Args。
        """
        super().__init__()
        # 输入特征维度
        self.dim = args.dim
        # 断言专家数量必须能被世界大小整除
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        # 专家总数
        self.n_routed_experts = args.n_routed_experts
        # 本地专家数量
        self.n_local_experts = args.n_routed_experts // world_size
        # 为每个输入激活的专家数量
        self.n_activated_experts = args.n_activated_experts
        # 专家开始索引
        self.experts_start_idx = rank * self.n_local_experts
        # 专家结束索引
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        # 门控机制
        self.gate = Gate(args)
        # 专家列表
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        # 共享专家
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE模块的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 专家路由和计算后的输出张量。
        """
        # 输入形状
        shape = x.size()
        # 将输入变形为(batch_size, -1, dim)
        x = x.view(-1, self.dim)
        # 计算权重和索引
        weights, indices = self.gate(x)
        # 初始化输出
        y = torch.zeros_like(x)
        # 计算每个专家的计数
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        # 遍历专家
        for i in range(self.experts_start_idx, self.experts_end_idx):
            # 如果计数为0
            if counts[i] == 0:
                # 跳过
                continue
            # 获取专家
            expert = self.experts[i]
            # 获取索引
            idx, top = torch.where(indices == i)
            # 计算输出
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        # 计算共享专家
        z = self.shared_experts(x)
        # 如果世界大小大于1
        if world_size > 1:
            # 计算输出
            dist.all_reduce(y)
        # 返回输出
        return (y + z).view(shape)


class Block(nn.Module):
    """
    结合注意力和前馈层的Transformer块。

    Attributes:
        attn (nn.Module): 注意力层（MLA）。
        ffn (nn.Module): 前馈网络（MLP或MoE）。
        attn_norm (nn.Module): 注意力的层归一化。
        ffn_norm (nn.Module): 前馈网络的层归一化。
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        初始化Transformer块。

        Args:
            layer_id (int): Transformer中的层索引。
            args (ModelArgs): 包含块Args的模型Args。
        """
        super().__init__()
        # 注意力层
        self.attn = MLA(args)
        # 前馈网络
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        # 注意力层归一化
        self.attn_norm = RMSNorm(args.dim)
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
    """
    具有位置嵌入、多层和输出投影的Transformer模型。

    Attributes:
        max_seq_len (int): Transformer的最大序列长度。
        embed (nn.Module): 输入词元的嵌入层。
        layers (torch.nn.ModuleList): Transformer块列表。
        norm (nn.Module): 在所有块之后应用的层归一化。
        head (nn.Module): 映射到词汇表大小的输出投影层。
        freqs_cis (torch.Tensor): 旋转嵌入的预计算复指数值。
    """
    def __init__(self, args: ModelArgs):
        """
        初始化Transformer模型。

        Args:
            args (ModelArgs): 包含TransformerArgs的模型Args。
        """
        # 全局变量
        global world_size, rank
        # 世界大小
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # 排名
        rank = dist.get_rank() if dist.is_initialized() else 0
        # 线性层数据类型
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        # 初始化
        super().__init__()
        # 最大序列长度
        self.max_seq_len = args.max_seq_len
        # 输入词元嵌入层
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        # 层列表
        self.layers = torch.nn.ModuleList()
        # 遍历层
        for layer_id in range(args.n_layers):
            # 添加块
            self.layers.append(Block(layer_id, args))
        # 层归一化
        self.norm = RMSNorm(args.dim)
        # 输出投影层
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
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
        # 序列长度
        seqlen = tokens.size(1)
        # 嵌入
        h = self.embed(tokens)
        # 频率
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        # 掩码
        mask = None
        # 如果序列长度大于1
        if seqlen > 1:
            # 创建掩码
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        # 遍历层
        for layer in self.layers:
            # 计算
            h = layer(h, start_pos, freqs_cis, mask)
        # 层归一化
        h = self.norm(h)[:, -1]
        # 计算logits
        logits = self.head(h)
        # 如果世界大小大于1
        if world_size > 1:
            # 创建logits列表
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            # 收集logits
            dist.all_gather(all_logits, logits)
            # 连接logits
            logits = torch.cat(all_logits, dim=-1)
        # 返回logits
        return logits


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
