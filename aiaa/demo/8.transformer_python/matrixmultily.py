import torch
import torch.nn.functional as F
import torch.optim as optim
import time

torch.set_default_device('cuda')
torch.manual_seed(1)

class MatrixMultiplication:
    def __init__(self, W):
        self.W = W

    def multiply(self, input):
        return input @ self.W


class SVGMatrixMultiplication:
    def __init__(self, W, r):
        U, S, V = torch.svd(W)
        self.A = U[:, :r] @ torch.diag(torch.sqrt(S[:r]))
        self.B = torch.diag(torch.sqrt(S[:r])) @ V[:, :r].T
        print("SVG. A.shape",self.A.shape)
        print("SVG. B.shape",self.B.shape)


    def multiply(self, input):
        bx =  input @ self.A
        return bx @ self.B


class LoRAMatrixMultiplication:
    def __init__(self, W, r):
        self.n_input = W.shape[0]
        self.n_output = W.shape[1]
        self.W = W
        self.A = torch.rand(self.n_input, r, requires_grad=True)
        self.B = torch.rand(r, self.n_output, requires_grad=True)
        print("LoRA. A.shape",self.A.shape)
        print("LoRA. B.shape",self.B.shape)
        self.train(0.001,10000,0.1);

    @torch.no_grad()
    def multiply(self, input):
        bx =  input @ self.A
        return bx @ self.B
    
    def forward(self, input):
        bx =  input @ self.A
        return bx @ self.B

    def train(self, learning_rate=0.1, max_epoch=10000, stop_loss=0.01):
        optimizer = optim.AdamW([self.A, self.B], lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50)
        last_loss = float('inf')
        for i in range(max_epoch):
            input = torch.rand(1,self.n_input)
            optimizer.zero_grad()
            y_lora = self.forward(input)
            output = input @ self.W
            mse_loss = F.mse_loss(y_lora, output)
            if i % 100 == 0:
                #print(f"第{i}次训练，损失为：{mse_loss.item()}")
                #if last_loss <= mse_loss.item():
                #    break
                if mse_loss.item() < stop_loss:
                    break
                last_loss = mse_loss.item()
            mse_loss.backward()
            optimizer.step()
            scheduler.step(mse_loss.item())


def cos_similarity(a,b):
    return F.cosine_similarity(a,b,dim=1).mean()

# 定义矩阵和向量的维度
n_input = 10000
n_output = 2000
r = 10  # 低秩矩阵的秩
# 初始化原始矩阵 W
W = torch.rand(n_input, n_output)


new_input = torch.rand(1,n_input)
print("W.shape",W.shape)
print("new_input.shape",new_input.shape)


# 直接矩阵乘法
w1 = MatrixMultiplication(W)
start_time = time.time()
y_direct = w1.multiply(new_input)
end_time = time.time()
direct_time = end_time - start_time
print(f"直接矩阵乘法耗时: {direct_time} 秒")
print(f"直接矩阵乘法结果: {y_direct.mean()}")
print(f"输出的形状: {y_direct.shape}")


# SVG 分解后的矩阵乘法
w2 = SVGMatrixMultiplication(W, r)
start_time = time.time()
y_svg = w2.multiply(new_input)
end_time = time.time()
svg_time = end_time - start_time
print(f"SVG 分解后的矩阵乘法耗时: {svg_time} 秒")
print(f"SVG 分解后的矩阵乘法结果: {y_svg.mean()}")
print(f"结果是否接近: {torch.allclose(y_direct, y_svg,rtol=0.01)}") # 允许1%的误差
print(f"余弦相似度: {cos_similarity(y_direct, y_svg)}")


# LoRA 分解后的矩阵乘法
w3 = LoRAMatrixMultiplication(W, r)
start_time = time.time()
y_lora = w3.multiply(new_input)
end_time = time.time()
lora_time = end_time - start_time
print(f"LoRA 分解后的矩阵乘法耗时: {lora_time} 秒")
print(f"LoRA 分解后的矩阵乘法结果: {y_lora.mean()}")
print(f"结果是否接近: {torch.allclose(y_direct, y_lora,rtol=0.01)}") # 允许1%的误差
print(f"余弦相似度: {cos_similarity(y_direct, y_lora)}")
