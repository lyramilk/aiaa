import torch  # 导入torch库
import torch.nn as nn  # 导入torch.nn模块
import math  # 导入math库
import transformers  # 导入transformers库
from modelscope import snapshot_download  # 从modelscope导入snapshot_download函数
import torch.nn.functional as F  # 导入torch.nn.functional模块
import copy  # 导入copy库
import torch.optim as optim  # 导入torch.optim模块


class Embeddings(nn.Module):  # 定义Embeddings类，继承自nn.Module
	"""
	Embeddings类是文本嵌入层，用于将词汇表中的词转换为向量
	"""
	def __init__(self, d_model: int, vocab_size: int):  # 定义构造函数，接收d_model和vocab_size两个参数
		"""
		初始化Embeddings类
		Args:
			d_model: 模型的维度
			vocab_size: 词汇表的大小
		"""
		super().__init__()  # 调用父类nn.Module的构造函数
		self.d_model = d_model  # 将d_model赋值给self.d_model
		self.embedding = nn.Embedding(vocab_size, d_model)  # 创建一个Embedding层，接收vocab_size和d_model两个参数

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，接收一个tensor类型的参数x
		"""
		文本嵌入
		Args:
			x: 输入
		Returns:
			输出
		"""
		return self.embedding(x) * math.sqrt(self.d_model)  # 返回Embedding层的输出乘以self.d_model的平方根


class PositionalEncoding(nn.Module):  # 定义PositionalEncoding类，继承自nn.Module
	"""
	PositionalEncoding类是位置编码层，用于将位置信息添加到输入中
	"""
	def __init__(self, d_model: int, dropout: float, max_length: int = 5000):  # 定义构造函数，接收d_model, dropout和max_length三个参数
		"""
		初始化PositionalEncoding类
		Args:
			d_model: 模型的维度
			dropout: dropout比率
			max_length: 最大长度
		"""
		super().__init__()  # 调用父类nn.Module的构造函数
		self.dropout = nn.Dropout(p=dropout)  # 创建一个Dropout层，接收dropout参数

		pe = torch.zeros(max_length, d_model)  # 创建一个形状为(max_length, d_model)的0矩阵
		position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)  # 创建一个形状为(max_length, 1)的tensor，值为0到max_length-1
		div_term = torch.exp(
			torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
		)  # 创建一个形状为(d_model/2,)的tensor
		pe[:, 0::2] = torch.sin(position * div_term)  # 将pe矩阵的偶数列赋值为sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)  # 将pe矩阵的奇数列赋值为cos(position * div_term)
		pe = pe.unsqueeze(0)  # 将pe矩阵的形状变为(1, max_length, d_model)
		self.register_buffer("pe", pe)  # 将pe注册为buffer

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，接收一个tensor类型的参数x
		"""
		位置编码
		Args:
			x: 输入
		Returns:
			输出
		"""
		x = x + self.pe[:, : x.size(1)]  # 将x与pe矩阵相加
		return self.dropout(x)  # 返回Dropout层的输出


class MultiHeadedAttention(nn.Module):  # 定义MultiHeadedAttention类，继承自nn.Module
	"""
	MultiHeadedAttention类是多头注意力机制，用于将输入的query, key和value转换为多头注意力
	"""
	def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):  # 定义构造函数，接收d_model, n_heads和dropout三个参数
		"""
		初始化MultiHeadedAttention类
		Args:
			d_model: 模型的维度
			n_heads: 多头注意力的头数
			dropout: dropout比率
		"""
		super(MultiHeadedAttention, self).__init__()  # 调用父类nn.Module的构造函数
		assert d_model % n_heads == 0  # 断言d_model可以被n_heads整除
		self.d_k = d_model // n_heads  # 计算每个头的维度
		self.n_heads = n_heads  # 将n_heads赋值给self.n_heads
		self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])  # 创建一个包含4个Linear层的ModuleList
		self.attn = None  # 将self.attn初始化为None
		self.dropout = nn.Dropout(p=dropout)  # 创建一个Dropout层，接收dropout参数

	def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:  # 定义attention函数，接收query, key, value和mask四个参数
		"""
		注意力机制
		Args:
			query: query
			key: key
			value: value
			mask: 掩码
		Returns:
			输出和注意力
		"""
		d_k = query.size(-1)  # 获取query的最后一维的维度
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 计算query和key的转置的点积，并除以d_k的平方根
		if mask is not None:  # 如果mask不为None
			scores = scores.masked_fill(mask == 0, -1e9)  # 将scores中mask为0的位置填充为-1e9
		p_attn = F.softmax(scores, dim=-1)  # 对scores的最后一维进行softmax操作
		if self.dropout is not None:  # 如果self.dropout不为None
			p_attn = self.dropout(p_attn)  # 对p_attn进行dropout操作
		return torch.matmul(p_attn, value), p_attn  # 返回p_attn和value的点积，以及p_attn

	def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:  # 定义前向传播函数，接收query, key, value和mask四个参数
		"""
		多头注意力机制
		Args:
			query: query
			key: key
			value: value
			mask: 掩码
		Returns:
			输出
		"""
		if mask is not None:  # 如果mask不为None
			mask = mask.unsqueeze(1)  # 将mask的形状变为(batch_size, 1, seq_len)
		nbatches = query.size(0)  # 获取query的第一维的维度

		# 将query, key, value分别通过对应的Linear层，并将形状变为(nbatches, n_heads, seq_len, d_k)
		query, key, value = [l(x).view(nbatches, -1, self.n_heads, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

		# 计算x和self.attn
		x, self.attn = self.attention(query, key, value, mask=mask) 

		# 将x的形状变为(nbatches, seq_len, d_model)
		x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_heads * self.d_k)  # 将x的形状变为(nbatches, seq_len, d_model)
		return self.linears[-1](x)  # 返回x通过最后一个Linear层的输出


class LayerNorm(nn.Module):  # 定义LayerNorm类，继承自nn.Module
	"""
	LayerNorm类是层归一化，用于将输入的x归一化
	"""
	def __init__(self, features: int, eps: float = 1e-6):  # 定义构造函数，接收features和eps两个参数
		"""
		初始化LayerNorm
		Args:
			features: 模型的维度
			eps: 一个很小的数，防止除以0
		"""
		super(LayerNorm, self).__init__()  # 调用父类nn.Module的构造函数
		self.a_2 = nn.Parameter(torch.ones(features))  # 创建一个形状为(features,)的tensor，值为1
		self.b_2 = nn.Parameter(torch.zeros(features))  # 创建一个形状为(features,)的tensor，值为0
		self.eps = eps  # 将eps赋值给self.eps

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，接收一个tensor类型的参数x
		"""
		层归一化
		Args:
			x: 输入
		Returns:
			输出
		"""
		mean = x.mean(-1, keepdim=True)  # 计算x的最后一维的均值
		std = x.std(-1, keepdim=True)  # 计算x的最后一维的标准差
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2  # 返回LayerNorm的结果


class SublayerConnection(nn.Module):  # 定义SublayerConnection类，继承自nn.Module
	"""
	SublayerConnection类是子层连接，用于将输入的x加上子层的输出
	"""
	def __init__(self, size: int, dropout: float):  # 定义构造函数，接收size和dropout两个参数
		"""
		初始化SublayerConnection
		Args:
			size: 模型的维度
			dropout: dropout比率
		"""
		super(SublayerConnection, self).__init__()  # 调用父类nn.Module的构造函数
		self.norm = LayerNorm(size)  # 创建一个LayerNorm层
		self.dropout = nn.Dropout(dropout)  # 创建一个Dropout层

	def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:  # 定义前向传播函数，接收x和sublayer两个参数
		"""
		子层连接
		Args:
			x: 输入
			sublayer: 子层
		Returns:
			输出
		"""
		return x + self.dropout(sublayer(self.norm(x)))  # 返回x加上sublayer的输出，其中sublayer的输入为x经过LayerNorm的结果


class PositionwiseFeedForward(nn.Module):  # 定义PositionwiseFeedForward类，继承自nn.Module
	"""
	PositionwiseFeedForward类是位置前馈神经网络，用于将输入的x转换为前馈神经网络的输出
	"""
	def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):  # 定义构造函数，接收d_model, d_ff和dropout三个参数
		"""
		初始化PositionwiseFeedForward
		Args:
			d_model: 模型的维度
			d_ff: 前馈神经网络的维度
			dropout: dropout比率
		"""
		super(PositionwiseFeedForward, self).__init__()  # 调用父类nn.Module的构造函数
		self.w_1 = nn.Linear(d_model, d_ff)  # 创建一个Linear层，输入维度为d_model，输出维度为d_ff
		self.w_2 = nn.Linear(d_ff, d_model)  # 创建一个Linear层，输入维度为d_ff，输出维度为d_model
		self.dropout = nn.Dropout(dropout)  # 创建一个Dropout层

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，接收一个tensor类型的参数x
		"""
		前馈神经网络
		Args:
			x: 输入
		Returns:
			输出
		"""
		return self.w_2(self.dropout(F.relu(self.w_1(x))))  # 返回x经过w_1, relu, dropout和w_2的结果


class EncoderLayer(nn.Module):  # 定义EncoderLayer类，继承自nn.Module
	"""
	EncoderLayer是编码器层，由自注意力机制和前馈神经网络组成
	"""
	def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float):  # 定义构造函数，接收size, self_attn, feed_forward和dropout四个参数
		"""
		初始化EncoderLayer
		Args:
			size: 模型的维度
			self_attn: 多头注意力
			feed_forward: 前馈神经网络
			dropout: dropout比率
		"""
		super(EncoderLayer, self).__init__()  # 调用父类nn.Module的构造函数
		self.self_attn = self_attn  # 将self_attn赋值给self.self_attn
		self.feed_forward = feed_forward  # 将feed_forward赋值给self.feed_forward
		self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])  # 创建一个包含2个SublayerConnection层的ModuleList
		self.size = size  # 将size赋值给self.size

	def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，接收x和mask两个参数
		"""
		编码器层
		Args:
			x: 输入
			mask: 掩码
		Returns:
			输出
		"""
		"Follow Figure 1 (left) for connections."
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 将x通过第一个SublayerConnection层，其中sublayer为self_attn
		return self.sublayer[1](x, self.feed_forward)  # 将x通过第二个SublayerConnection层，其中sublayer为feed_forward


class Encoder(nn.Module):  # 定义Encoder类，继承自nn.Module
	"""
	Encoder类是编码器，由多个EncoderLayer组成
	"""
	def __init__(self, layer: nn.Module, N: int):  # 定义构造函数，接收layer和N两个参数
		"""
		初始化Encoder
		Args:
			layer: 编码器层
			N: 编码器层数
		"""
		super(Encoder, self).__init__()  # 调用父类nn.Module的构造函数
		self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])  # 创建一个包含N个layer的ModuleList
		self.norm = LayerNorm(layer.size)  # 创建一个LayerNorm层

	def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，接收x和mask两个参数
		"""
		编码器
		Args:
			x: 输入
			mask: 掩码
		Returns:
			输出
		"""
		"Pass the input (and mask) through each layer in turn."
		for layer in self.layers:  # 遍历self.layers中的每个layer
			x = layer(x, mask)  # 将x通过layer
		return self.norm(x)  # 返回x经过LayerNorm的结果


class DecoderLayer(nn.Module):  # 定义DecoderLayer类，继承自nn.Module
	"""
	DecoderLayer是解码器层，由自注意力机制、编码器-解码器注意力和前馈神经网络组成
	"""
	def __init__(self, size: int, self_attn: nn.Module, src_attn: nn.Module, feed_forward: nn.Module, dropout: float):  # 定义构造函数，接收size, self_attn, src_attn, feed_forward和dropout五个参数
		"""
		初始化DecoderLayer
		Args:
			size: 模型的维度
			self_attn: 多头注意力（自注意力）
			src_attn: 多头注意力（编码器-解码器注意力）
			feed_forward: 前馈神经网络
			dropout: dropout比率
		"""
		super(DecoderLayer, self).__init__()  # 调用父类nn.Module的构造函数
		self.size = size  # 将size赋值给self.size
		self.self_attn = self_attn  # 将self_attn赋值给self.self_attn
		self.src_attn = src_attn  # 将src_attn赋值给self.src_attn
		self.feed_forward = feed_forward  # 将feed_forward赋值给self.feed_forward
		self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(3)])  # 创建一个包含3个SublayerConnection层的ModuleList

	def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，接收x, memory, src_mask和tgt_mask四个参数
		"""
		解码器层
		Args:
			x: 输入
			memory: 编码器的输出
			src_mask: 源语言掩码
			tgt_mask: 目标语言掩码
		Returns:
			输出
		"""
		"Follow Figure 1 (right) for connections."
		m = memory  # 将memory赋值给m
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # 将x通过第一个SublayerConnection层，其中sublayer为self_attn
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # 将x通过第二个SublayerConnection层，其中sublayer为src_attn
		return self.sublayer[2](x, self.feed_forward)  # 将x通过第三个SublayerConnection层，其中sublayer为feed_forward


class Decoder(nn.Module):  # 定义Decoder类，继承自nn.Module
	"""
	Decoder类是解码器，由多个DecoderLayer组成
	"""
	def __init__(self, layer: nn.Module, N: int):  # 定义构造函数，接收layer和N两个参数
		"""
		初始化Decoder
		Args:
			layer: 解码器层
			N: 解码器层数
		"""
		super(Decoder, self).__init__()  # 调用父类nn.Module的构造函数
		self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])  # 创建一个包含N个layer的ModuleList
		self.norm = LayerNorm(layer.size)  # 创建一个LayerNorm层

	def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，接收x, memory, src_mask和tgt_mask四个参数
		"""
		解码器
		Args:
			x: 输入
			memory: 编码器的输出
			src_mask: 源语言掩码
			tgt_mask: 目标语言掩码
		Returns:
			输出
		"""
		for layer in self.layers:  # 遍历self.layers中的每个layer
			x = layer(x, memory, src_mask, tgt_mask)  # 将x通过layer
		return self.norm(x)  # 返回x经过LayerNorm的结果


class Generator(nn.Module):  # 定义Generator类，继承自nn.Module
	"""
	Generator类是生成器，用于将输入的x转换为生成器的输出
	"""
	def __init__(self, d_model: int, vocab: int):  # 定义构造函数，接收d_model和vocab两个参数
		"""
		初始化生成器
		Args:
			d_model: 模型的维度
			vocab: 词汇表大小
		"""
		super(Generator, self).__init__()  # 调用父类nn.Module的构造函数
		self.proj = nn.Linear(d_model, vocab)  # 创建一个Linear层，输入维度为d_model，输出维度为vocab

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，接收一个tensor类型的参数x
		"""
		生成器
		Args:
			x: 输入
		Returns:
			输出
		"""
		return F.log_softmax(self.proj(x), dim=-1)  # 返回x通过Linear层和log_softmax的结果


class EncoderDecoder(nn.Module):  # 定义EncoderDecoder类，继承自nn.Module
	"""
	EncoderDecoder类是编码解码器，用于将输入的src, tgt, src_mask和tgt_mask转换为生成器的输出
	"""

	def __init__(self, encoder: nn.Module, decoder: nn.Module, src_embed: nn.Module, tgt_embed: nn.Module, generator: nn.Module):  # 定义构造函数，接收encoder, decoder, src_embed, tgt_embed和generator五个参数
		"""
		初始化编码解码器
		Args:
			encoder: 编码器
			decoder: 解码器
			src_embed: 源语言嵌入
			tgt_embed: 目标语言嵌入
			generator: 生成器
		"""
		super(EncoderDecoder, self).__init__()  # 调用父类nn.Module的构造函数
		self.encoder = encoder  # 将encoder赋值给self.encoder
		self.decoder = decoder  # 将decoder赋值给self.decoder
		self.src_embed = src_embed  # 将src_embed赋值给self.src_embed
		self.tgt_embed = tgt_embed  # 将tgt_embed赋值给self.tgt_embed
		self.generator = generator  # 将generator赋值给self.generator

	def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，接收src, tgt, src_mask和tgt_mask四个参数
		"""
		编码解码器
		Args:
			src: 源语言
			tgt: 目标语言
			src_mask: 源语言掩码
			tgt_mask: 目标语言掩码
		Returns:
			输出
		"""
		"Take in and process masked src and target sequences."
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)  # 返回解码器的输出，其中解码器的输入为编码器的输出，src_mask, tgt和tgt_mask

	def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:  # 定义encode函数，接收src和src_mask两个参数
		"""
		编码
		Args:
			src: 源语言
			src_mask: 源语言掩码
		Returns:
			输出
		"""
		return self.encoder(self.src_embed(src), src_mask)  # 返回编码器的输出，其中编码器的输入为src经过src_embed的结果和src_mask

	def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:  # 定义decode函数，接收memory, src_mask, tgt和tgt_mask四个参数
		"""
		解码
		Args:
			memory: 编码器的输出
			src_mask: 源语言掩码
			tgt: 目标语言
			tgt_mask: 目标语言掩码
		Returns:
			输出
		"""
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)  # 返回解码器的输出，其中解码器的输入为tgt经过tgt_embed的结果, memory, src_mask和tgt_mask


class Transformer(nn.Module):  # 定义Transformer类
	"""
	Transformer类是Transformer模型，用于将输入的src, tgt, src_mask和tgt_mask转换为生成器的输出
	"""
	def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1,max_length: int = 200):  # 定义构造函数
		"""
		初始化Transformer
		Args:
			d_model: 模型的维度
			n_heads: 多头注意力的头数
			d_ff: 前馈神经网络的维度
			n_layers: 编码器和解码器的层数
			dropout: dropout比率
			max_length: 上下文长度
		"""
		super(Transformer, self).__init__()  # 调用父类的构造函数
		self.max_length = max_length # 上下文长度
		c = copy.deepcopy  # 定义别名
		attn = MultiHeadedAttention(d_model, n_heads)  # 实例化MultiHeadedAttention
		ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 实例化PositionwiseFeedForward
		position = PositionalEncoding(d_model, dropout,max_length)  # 实例化PositionalEncoding
		self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers)  # 实例化Encoder
		self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers)  # 实例化Decoder

		tokenizer_path = snapshot_download("lyramilk/deepseek_v3_tokenizer")  # 下载分词器

		self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)  # 加载分词器

		self.generator = Generator(d_model, self.tokenizer.vocab_size)  # 实例化Generator

		self.src_embed = nn.Sequential(Embeddings(d_model, self.tokenizer.vocab_size), c(position))  # 定义源语言嵌入
		self.tgt_embed = nn.Sequential(Embeddings(d_model, self.tokenizer.vocab_size), c(position))  # 定义目标语言嵌入
		self.embeddings = nn.Sequential(Embeddings(d_model, self.tokenizer.vocab_size), c(position))  # 定义嵌入

		# 初始化模型参数
		for p in self.parameters():  # 遍历模型参数
			if p.dim() > 1:  # 如果参数的维度大于1
				nn.init.xavier_uniform_(p)  # 使用Xavier均匀分布初始化参数

	def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:  # 定义encode函数，接收src和src_mask两个参数
		"""
		编码
		Args:
			src: 源语言
			src_mask: 源语言掩码
		Returns:
			输出
		"""
		return self.encoder(self.src_embed(src), src_mask)  # 返回编码器的输出，其中编码器的输入为src经过src_embed的结果和src_mask

	def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:  # 定义decode函数，接收memory, src_mask, tgt和tgt_mask四个参数
		"""
		解码
		Args:
			memory: 编码器的输出
			src_mask: 源语言掩码
			tgt: 目标语言
			tgt_mask: 目标语言掩码
		Returns:
			输出
		"""
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)  # 返回解码器的输出，其中解码器的输入为tgt经过tgt_embed的结果, memory, src_mask和tgt_mask

	def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，接收src, tgt, src_mask和tgt_mask四个参数
		"""
		编码解码器
		Args:
			src: 源语言
			tgt: 目标语言
			src_mask: 源语言掩码
			tgt_mask: 目标语言掩码
		Returns:
			输出
		"""
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)  # 返回解码器的输出，其中解码器的输入为编码器的输出，src_mask, tgt和tgt_mask

	def generate(self, src: str, max_len: int = 100) -> str:  # 定义生成函数
		"""
		生成
		Args:
			src: 源语言
		Returns:
			目标语言
		"""
		self.eval()  # 设置为评估模式
		input_ids = torch.tensor(self.tokenizer.encode(src), dtype=torch.long).unsqueeze(0)  # 将输入编码为长整型张量并增加一个维度
		src_mask = torch.ones(input_ids.size(1), input_ids.size(1), dtype=torch.bool)  # 创建源语言掩码，确保形状与输入匹配
		memory = self.encode(input_ids, src_mask)  # 编码
		# 使用 <bos> token 初始化 tgt
		tgt = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long)  # 初始化目标语言

		for _ in range(max_len):  # 循环生成
			# 第N次循环时，tgt.shape=(1, N)

			# 创建 tgt_mask: 三角形遮罩
			tgt_len = tgt.size(1)  # 获取当前tgt的长度
			tgt_mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)  # 创建目标语言掩码
			tgt_mask = (tgt_mask.float().masked_fill(tgt_mask == 0, float("-inf")).masked_fill(tgt_mask == 1, float(0.0)))  # 将目标语言掩码中的0填充为负无穷大，1填充为0
			tgt_mask = tgt_mask.unsqueeze(0)  # 增加一个维度

			# 解码时只使用当前生成的序列
			out = self.decode(memory, src_mask, tgt, tgt_mask)  # 解码
			prob = self.generator(out[:, -1])  # 生成
			_, next_word = torch.max(prob, dim=1)  # 获取概率最大的词
			next_word = next_word.item()  # 获取概率最大的词的id

			# 将新生成的 token 添加到 tgt
			tgt = torch.cat([tgt, torch.tensor([[next_word]], dtype=torch.long)], dim=1)  # 将生成的词添加到目标语言中

			if next_word == self.tokenizer.eos_token_id:  # 如果生成了 <eos> token，则停止
				break

		return self.tokenizer.decode(tgt[0].tolist())  # 解码为目标语言

	def learn(self, src: str, tgt: str,learning_rate: float = 1e-3):
		self.train()
		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		src_ids = torch.tensor(self.tokenizer.encode(src), dtype=torch.long).unsqueeze(0)
		tgt_ids = torch.tensor(self.tokenizer.encode(tgt), dtype=torch.long).unsqueeze(0)

		# 修改src_ids和tgt_ids的形状
		src_ids = torch.cat([src_ids, torch.zeros(1, self.max_length - src_ids.size(1), dtype=torch.long)], dim=1)
		tgt_ids = torch.cat([tgt_ids, torch.zeros(1, self.max_length - tgt_ids.size(1), dtype=torch.long)], dim=1)


		src_mask = torch.ones(src_ids.size(1), src_ids.size(1), dtype=torch.bool)
		tgt_mask = torch.ones(tgt_ids.size(1), tgt_ids.size(1), dtype=torch.bool)
		src = self.embeddings(src_ids)
		memory = self.encoder.forward(src_ids.float(), src_mask)
		print("memory.shape=",memory.shape)
		print("src_mask.shape=",src_mask.shape)
		print("src_ids.shape=",src_ids.shape)
		print("tgt_ids.shape=",tgt_ids.shape)
		print("src.shape=",src.shape)
		print("tgt_mask.shape=",tgt_mask.shape)
		output = self.decoder.forward(self.embeddings(tgt_ids), memory, src_mask, tgt_mask)
		print("output.shape=",output.shape)
		output = self.generator(output[:, -1])
		print("output.shape=",output.shape)
		loss = F.nll_loss(output, tgt_ids[:, -1])
		print("loss=",loss)
		loss.backward()
		self.optimizer.step()
		return loss.item()

tt = Transformer(d_model=512, n_heads=8, d_ff=2048, n_layers=6, dropout=0.1)  # 实例化Transformer


for i in range(3):
	tt.learn("你好", "北京",learning_rate=1e-3)
	tt.learn("北京", "天津",learning_rate=1e-3)


print(tt.generate("你好"))  # 生成目标语言
print(tt.generate("北京"))  # 生成目标语言
 
 