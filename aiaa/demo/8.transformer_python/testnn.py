import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

class LoRA:
    def __init__(self, shapeA, shapeB, rank, lr=0.001):
        self.rank = rank
        tensorA = torch.randn(shapeA);
        tensorB = torch.randn(shapeB);
        self.A1, self.B1, self.original_shape1 = self.lora_decomposition(tensorA)
        self.A2, self.B2, self.original_shape2 = self.lora_decomposition(tensorB)
        self.train(lr)

    def lora_decomposition(self, tensor):
        """
        将二维张量进行LoRA分解（低秩适应）
        
        :param tensor: 输入的二维张量
        :return: 两个低秩矩阵A和B，原始形状信息
        """
        # 保存原始形状
        original_shape = tensor.shape
        
        # 确保输入是二维张量
        if len(original_shape) != 2:
            raise ValueError("输入必须是二维张量")
        
        rows, columns = original_shape
        
        # 初始化低秩矩阵 - 使用SVD初始化
        with torch.no_grad():
            U, S, V = torch.svd(tensor)
            A = U[:, :self.rank] @ torch.diag(torch.sqrt(S[:self.rank]))
            B = torch.diag(torch.sqrt(S[:self.rank])) @ V[:, :self.rank].T
            A.requires_grad = True
            B.requires_grad = True
        
        return A, B, original_shape

    def train(self, lr):
        # 使用相同形状的随机向量进行训练
        random_tensor1 = torch.randn(self.original_shape1)
        random_tensor2 = torch.randn(self.original_shape2)

        # 使用优化器进行训练
        optimizer = optim.AdamW([self.A1, self.B1, self.A2, self.B2], lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50)
        
        last_loss = 1000

        for i in range(0,2000,1):
            optimizer.zero_grad()
            
            # 计算重建矩阵
            reconstruction1 = self.A1 @ self.B1
            reconstruction2 = self.A2 @ self.B2
            
            # 计算重建误差
            loss1 = F.mse_loss(reconstruction1, random_tensor1)
            loss2 = F.mse_loss(reconstruction2, random_tensor2)
            loss = loss1 + loss2
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # 打印损失
            if (i + 1) % 100 == 0:
                print(f"迭代 {i + 1}, 损失: {loss.item():.6f}")
                if last_loss <= loss.item():
                    break
                if loss.item() < 0.1:
                    break
                last_loss = loss.item()

    def lora_matrix_multiplication(self, tensor1, tensor2):
        """
        利用LoRA分解的特性高效计算两个二维张量的乘法
        
        :param tensor1: 可选的新输入张量1
        :param tensor2: 可选的新输入张量2
        :return: 乘法结果
        """
        A1, B1, original_shape1 = self.lora_decomposition(tensor1)
        A2, B2, original_shape2 = self.lora_decomposition(tensor2)

        # 验证形状兼容性
        if B1.shape[1] != A2.shape[0]:
            raise ValueError(f"形状不兼容: B1 形状 {B1.shape}, A2 形状 {A2.shape}")
        
        # 计算中间结果 (更高效，因为rank通常远小于原始维度)
        middle_product = B1 @ A2
        
        # 计算最终结果
        result = A1 @ middle_product @ B2
        
        # 验证结果形状
        expected_shape = (original_shape1[0], original_shape2[1])
        if result.shape != expected_shape:
            result = result.reshape(expected_shape)
        
        return result

    def multiply(self,tensor1,tensor2):
        return self.lora_matrix_multiplication(tensor1,tensor2)

def cosin_similarity(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0)

# 使用示例

print("cuda是否可用：", torch.cuda.is_available())
torch.set_default_device('cuda')
torch.manual_seed(1234)
t1 = torch.randn(10000, 1000)
t2 = torch.randn(1000, 10000)
t1p = t1.clone()
t2p = t2.clone()

print(t1p.device)

# 创建LoRA实例并训练
lora_instance = LoRA(t1p.shape, t2p.shape, rank=80, lr=0.0001)


time1 = time.time()
test1 = t1 @ t2  # 计算真实乘法
time2 = time.time()
# 计算乘法
test2 = lora_instance.multiply(t1p,t2p)
time3 = time.time()
print("相似度：", cosin_similarity(test1, test2))

print("普通乘法", time2 - time1)
print("LoRA乘法", time3 - time2)
