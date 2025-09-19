import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
# 导入 train_test_split 函数
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#1. 数据集准备
def prepare_data(noise=0.1):
    """生成并预处理 make_moons 数据集"""
    X, y = make_moons(n_samples=1000, noise=noise, random_state=42)
    # 标准化：让特征均值为0、方差为1，加速模型收敛
    X = StandardScaler().fit_transform(X)
    # 划分训练集（80%）和测试集（20%）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test


#2. 手动实现神经网络核心逻辑
class NumpyNet:
    """
    手动实现的简单神经网络（NumPy 版）
    结构：输入层(2) → 隐藏层(16, ReLU) → 输出层(2)
    包含：参数初始化、前向传播、损失计算、反向传播、参数更新全手动实现
    """
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=2):
        # 1. 参数初始化（用小随机数，避免梯度消失）
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01  # 输入→隐藏层权重
        self.b1 = np.zeros((1, hidden_dim))  # 隐藏层偏置
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01  # 隐藏→输出层权重
        self.b2 = np.zeros((1, output_dim))  # 输出层偏置

    def forward(self, X):
        """2. 前向传播（手动计算矩阵运算）"""
        # 隐藏层：线性变换 + ReLU
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU 激活

        # 输出层：线性变换 + Softmax（转概率）
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        # Softmax 计算（避免指数爆炸：减去行最大值）
        exp_scores = np.exp(self.Z2 - np.max(self.Z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def compute_loss(self, X, y):
        """3. 计算交叉熵损失（手动实现）"""
        num_examples = X.shape[0]
        # 取真实标签对应的概率（避免 log(0)，加小epsilon）
        correct_log_probs = -np.log(self.probs[range(num_examples), y] + 1e-8)
        loss = np.sum(correct_log_probs) / num_examples  # 平均损失
        return loss

    def backward(self, X, y):
        """4. 反向传播（手动推导梯度公式）"""
        num_examples = X.shape[0]

        # 计算输出层梯度（dZ2 = 预测概率 - 真实标签）
        dZ2 = self.probs.copy()
        dZ2[range(num_examples), y] -= 1
        dZ2 /= num_examples  # 平均梯度

        # 计算隐藏层→输出层的梯度（dW2, db2）
        self.dW2 = np.dot(self.A1.T, dZ2)
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)

        # 计算隐藏层梯度（dA1 = dZ2 · W2^T，再乘以 ReLU 的导数）
        dA1 = np.dot(dZ2, self.W2.T)
        dA1[self.Z1 <= 0] = 0  # ReLU 导数：Z>0 时为1，否则为0

        # 计算输入层→隐藏层的梯度（dW1, db1）
        self.dW1 = np.dot(X.T, dA1)
        self.db1 = np.sum(dA1, axis=0, keepdims=True)

    def update_params(self, learning_rate=0.01):
        """5. 参数更新（梯度下降）"""
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2


#3. 训练与评估流程
def train_numpy_net(model, X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.01):
    """手动训练流程：前向→计算损失→反向→更新参数"""
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # 1. 前向传播
        probs = model.forward(X_train)

        # 2. 计算损失
        loss = model.compute_loss(X_train, y_train)
        train_losses.append(loss)

        # 3. 计算训练集准确率
        y_pred_train = np.argmax(probs, axis=1)
        train_acc = np.mean(y_pred_train == y_train)
        train_accuracies.append(train_acc)

        # 4. 反向传播 + 参数更新
        model.backward(X_train, y_train)
        model.update_params(learning_rate)

        # 5. 计算测试集准确率（验证泛化能力）
        probs_test = model.forward(X_test)
        y_pred_test = np.argmax(probs_test, axis=1)
        test_acc = np.mean(y_pred_test == y_test)
        test_accuracies.append(test_acc)

        # 每10轮打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | 训练损失: {loss:.4f} | 训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")

    return train_losses, train_accuracies, test_accuracies


#4. 可视化工具
def plot_dataset(X, y, title_suffix=''):
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title(f"Dataset Distribution {title_suffix}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def plot_training_curves(train_losses, train_accuracies, test_accuracies, title_suffix=''):
    plt.figure(figsize=(12, 4))
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.title(f"Training Loss {title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 训练准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.title(f"Training Accuracy {title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # 测试准确率曲线
    plt.subplot(1, 3, 3)
    plt.plot(test_accuracies, label="Test Accuracy", color='orange')
    plt.title(f"Test Accuracy {title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_heatmap(model, X, y, title_suffix=''):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # 前向传播预测概率（复用模型的 forward 方法）
    probs = model.forward(X_grid)
    probs_class1 = probs[:, 1].reshape(xx.shape)  # 取第二类概率

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, probs_class1, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title(f"Classification Heatmap {title_suffix}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.show()


#5. 主流程
if __name__ == '__main__':
    noise_list = [0.01, 0.1, 0.3]

    for noise in noise_list:
        print(f"\n================ Training with noise={noise} (NumPy Net) ================")

        # ① 准备数据
        X_train, y_train, X_test, y_test = prepare_data(noise)

        # ② 初始化模型（手动实现的 NumpyNet）
        model = NumpyNet(input_dim=2, hidden_dim=16, output_dim=2)

        # ③ 训练模型（全手动流程）
        train_losses, train_accuracies, test_accuracies = train_numpy_net(
            model, X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.01
        )

        # ④ 可视化结果
        plot_dataset(X_train, y_train, title_suffix=f"(noise={noise})")
        plot_training_curves(train_losses, train_accuracies, test_accuracies, title_suffix=f"(noise={noise})")
        plot_heatmap(model, X_train, y_train, title_suffix=f"(noise={noise})")

        # ⑤ 验证概率归一化
        sample_probs = model.forward(X_train[:1])
        print(f"Example Probabilities (noise={noise}): {sample_probs} → Sum: {np.sum(sample_probs):.2f} (should be 1)\n")
