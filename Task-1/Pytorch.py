import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# ========================1. 数据集准备 ========================
def prepare_data(noise=0.1):
    """
    生成并预处理 make_moons 数据集
    :param noise: 控制数据噪声（值越大，数据越分散）
    :return: 训练集（X_train, y_train）、测试集（X_test, y_test）（PyTorch 张量格式）
    """
    # 生成半月亮形二分类数据，n_samples=1000 表示样本量
    X, y = make_moons(n_samples=1000, noise=noise, random_state=42)  
    # 标准化：让特征均值为0、方差为1，加速模型收敛
    X = StandardScaler().fit_transform(X)  
    # 划分训练集（80%）和测试集（20%）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    
    # 转换为 PyTorch 张量（float32 节省内存，long 匹配分类标签）
    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long))

# ======================== 2. 神经网络定义 ========================
class SimpleNet(nn.Module):
    """
    简单神经网络类：输入层→隐藏层（ReLU激活）→输出层
    输入：2维特征（来自 make_moons）
    隐藏层：16个神经元（可调整，平衡效果与速度）
    输出：2维（对应二分类的两个类别概率）
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 网络层按顺序堆叠：线性层→激活函数→线性层
        self.layer_stack = nn.Sequential(
            nn.Linear(2, 16),  # 输入层（2特征）→隐藏层（16神经元）
            nn.ReLU(),         # ReLU激活：增加非线性表达能力
            nn.Linear(16, 2)   # 隐藏层→输出层（2类别）
        )
    
    def forward(self, x):
        """
        前向传播：定义数据如何流经网络
        :param x: 输入数据（PyTorch 张量）
        :return: 输出层原始分数（未经过 softmax）
        """
        return self.layer_stack(x)

# ======================== 3. 训练流程 ========================
def train_model(model, criterion, optimizer, X_train, y_train, epochs=100):
    """
    训练模型核心流程
    :param model: 定义好的神经网络（SimpleNet 实例）
    :param criterion: 损失函数（如 CrossEntropyLoss）
    :param optimizer: 优化器（如 Adam）
    :param X_train: 训练集特征（张量）
    :param y_train: 训练集标签（张量）
    :param epochs: 训练轮数（默认100）
    :return: 训练损失历史、训练准确率历史（用于绘图）
    """
    train_losses = []  # 记录每轮损失
    train_accuracies = []  # 记录每轮准确率
    
    for epoch in range(epochs):
        # 1. 前向传播：用模型预测训练集
        outputs = model(X_train)
        # 计算损失：衡量预测与真实标签的差距
        loss = criterion(outputs, y_train)  
        
        # 2. 反向传播 + 参数更新
        optimizer.zero_grad()  # 清空历史梯度（否则会累加）
        loss.backward()        # 自动计算梯度（PyTorch 核心：自动求导）
        optimizer.step()       # 更新网络权重（如 W、b）
        
        # 3. 计算当前轮准确率
        # torch.max(outputs, 1) 取输出层最大值的索引（即预测类别）
        _, predicted = torch.max(outputs, 1)  
        # 对比预测类别与真实标签，计算正确比例
        accuracy = (predicted == y_train).sum().item() / y_train.size(0)  
        
        # 记录训练过程
        train_losses.append(loss.item())
        train_accuracies.append(accuracy)
        
        # 每10轮打印进度（方便观察训练趋势）
        if (epoch + 1) % 10 == 0:
            print(f"第 {epoch+1:3d} 轮训练 | 损失: {loss.item():.4f} | 准确率: {accuracy:.4f}")
    
    return train_losses, train_accuracies

# ==================== 4. 测试集评估 ========================
def evaluate_model(model, criterion, X_test, y_test):
    """
    在测试集上评估模型性能
    :param model: 训练好的模型
    :param criterion: 损失函数
    :param X_test: 测试集特征（张量）
    :param y_test: 测试集标签（张量）
    :return: 测试集损失、测试集准确率
    """
    # 关闭梯度计算（测试阶段不需要更新参数）
    with torch.no_grad():  
        outputs = model(X_test)
        # 计算测试集损失
        test_loss = criterion(outputs, y_test)  
        # 计算测试集准确率
        _, predicted = torch.max(outputs, 1)  
        test_accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    
    print(f"\n测试集表现 | 损失: {test_loss:.4f} | 准确率: {test_accuracy:.4f}")
    return test_loss, test_accuracy

# ======================== 5. 可视化工具（保持简洁，补充功能说明） ========================
def plot_dataset(X, y, title_suffix=''):
    """
    绘制数据集分布散点图
    :param X: 特征数据（numpy 或张量）
    :param y: 标签数据（0/1 二分类）
    :param title_suffix: 标题后缀（如 noise=0.1）
    """
    plt.figure(figsize=(6, 4))
    # 转换为 numpy 格式（matplotlib 不支持直接用张量）
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X  
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap=plt.cm.Spectral)
    plt.title(f"数据集分布 {title_suffix}")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.show()

def plot_training_curves(train_losses, train_accuracies, title_suffix=''):
    """
    绘制训练损失曲线 + 准确率曲线（双子图）
    :param train_losses: 训练损失历史（列表）
    :param train_accuracies: 训练准确率历史（列表）
    :param title_suffix: 标题后缀（如 noise=0.1）
    """
    plt.figure(figsize=(12, 4))
    # 子图1：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失")
    plt.title(f"训练损失 {title_suffix}")
    plt.xlabel("轮数")
    plt.ylabel("损失值")
    plt.legend()
    
    # 子图2：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="训练准确率")
    plt.title(f"训练准确率 {title_suffix}")
    plt.xlabel("轮数")
    plt.ylabel("准确率")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_heatmap(model, X, y, title_suffix=''):
    """
    绘制分类热力图（展示模型决策边界）
    :param model: 训练好的模型
    :param X: 特征数据（用于确定绘图范围）
    :param y: 标签数据（可视化真实分布）
    :param title_suffix: 标题后缀（如 noise=0.1）
    """
    # 转换为 numpy 格式
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X  
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y
    
    # 生成网格点，覆盖数据集范围（用于预测概率）
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]  # 拼接成特征矩阵
    X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32)  # 转张量
    
    # 模型预测（关闭梯度，加快速度）
    with torch.no_grad():
        outputs = model(X_grid_tensor)
        # softmax 转概率，取第二类的概率（可视化用）
        probs = torch.softmax(outputs, dim=1)[:, 1].numpy().reshape(xx.shape)  
    
    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, probs, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title(f"分类热力图 {title_suffix}")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.colorbar()
    plt.show()

# ==================== 6. 主流程 ========================
if __name__ == '__main__':
    # 测试不同噪声对模型的影响（可自行调整 noise 值）
    noise_list = [0.01, 0.1, 0.3]
    
    for noise in noise_list:
        print(f"\n================ 开始训练（noise={noise}） ================")
        
        # ① 准备数据（训练集+测试集）
        X_train, y_train, X_test, y_test = prepare_data(noise)
        
        # ② 初始化模型、损失函数、优化器
        model = SimpleNet()  # 实例化神经网络
        # 交叉熵损失：适合二分类/多分类任务
        criterion = nn.CrossEntropyLoss()  
        # Adam 优化器：学习率 0.01（可调整）
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
        
        # ③ 训练模型（记录损失和准确率）
        train_losses, train_accuracies = train_model(model, criterion, optimizer, X_train, y_train)
        
        # ④ 测试集评估
        test_loss, test_accuracy = evaluate_model(model, criterion, X_test, y_test)
        
        # ⑤ 可视化结果（数据集+训练曲线+热力图）
        plot_dataset(X_train, y_train, title_suffix=f"(noise={noise})")
        plot_training_curves(train_losses, train_accuracies, title_suffix=f"(noise={noise})")
        plot_heatmap(model, X_train, y_train, title_suffix=f"(noise={noise})")
        
        # ⑥ 验证概率归一化（选 1 个样本打印）
        with torch.no_grad():
            sample_output = model(X_train[:1])  # 取第一个样本预测
            probs = torch.softmax(sample_output, dim=1)  # 转概率
            print(f"样本概率（noise={noise}）: {probs.numpy()} → 总和: {probs.sum().item():.2f}（应为1）\n")
