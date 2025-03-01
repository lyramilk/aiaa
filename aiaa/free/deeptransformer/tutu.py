import torch
import torch.nn as nn

torch.set_default_device("cuda")

# 定义一个简单的模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

from torch.utils.tensorboard import SummaryWriter
# 创建 SummaryWriter
writer = SummaryWriter()
# 添加计算图
writer.add_graph(model, torch.randn(1, 10))
# 关闭 writer
writer.close()

# 生成随机输入
x = torch.randn(1, 10)

# 使用 profiler 记录性能
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    y = model(x)

# 打印性能分析结果
print(prof.key_averages().table(sort_by="cuda_time_total"))