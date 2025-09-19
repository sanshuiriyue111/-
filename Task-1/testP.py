import torch
import torch.nn as nn
import time
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 数据集准备（不变）
def prepare_data(noise=0.1):
    X, y = make_moons(n_samples=1000, noise=noise, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long))

# 模型定义（增大隐藏层到 1000）
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(2, 1000),  # 隐藏层改为 1000 神经元
            nn.ReLU(),
            nn.Linear(1000, 2)
        )
    def forward(self, x):
        return self.layer_stack(x)

# 训练函数（支持 GPU，增加轮次到 10000）
def train_model(model, criterion, optimizer, X_train, y_train, epochs=10000):
    train_losses = []
    train_accuracies = []
    
    # 记录训练总时间
    start_time = time.time()
    
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播 + 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率（每 1000 轮打印一次，避免日志过多）
        if (epoch + 1) % 1000 == 0:
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_train).sum().item() / y_train.size(0)
            train_losses.append(loss.item())
            train_accuracies.append(accuracy)
            print(f"第 {epoch+1:5d} 轮 | 损失: {loss.item():.4f} | 准确率: {accuracy:.4f}")
    
    # 计算训练耗时
    total_time = time.time() - start_time
    avg_epoch_time = total_time / epochs
    print(f"\nPyTorch训练总耗时: {total_time:.4f}秒 | 平均每轮耗时: {avg_epoch_time:.6f}秒")
    
    return train_losses, train_accuracies, total_time, avg_epoch_time

# 推理速度测试（支持 GPU）
def test_inference_speed(model, X_sample, num_runs=1000):
    model.eval()
    # 数据同步到 GPU（如果模型在 GPU）
    if next(model.parameters()).is_cuda:
        X_sample = X_sample.to('cuda')
    
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_runs):
            model(X_sample)
        total_time = time.time() - start_time
    
    avg_inference_time = total_time / num_runs
    print(f"PyTorch单样本推理平均耗时: {avg_inference_time:.8f}秒（重复{num_runs}次）")
    return avg_inference_time

# 主流程（支持 GPU 加速）
if __name__ == '__main__':
    noise = 0.1
    X_train, y_train, X_test, y_test = prepare_data(noise)
    
    # 初始化模型并尝试 GPU 加速
    model = SimpleNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 模型移到 GPU/CPU
    X_train = X_train.to(device)  # 数据移到对应设备
    y_train = y_train.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练（10000 轮）
    train_losses, train_accuracies, total_train_time, avg_epoch_time = train_model(
        model, criterion, optimizer, X_train, y_train, epochs=10000
    )
    
    # 测试推理速度（用一个样本）
    X_sample = X_train[:1]
    test_inference_speed(model, X_sample, num_runs=1000)