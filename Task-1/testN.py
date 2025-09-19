import numpy as np
import time
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 数据集准备（不变）
def prepare_data(noise=0.1):
    X, y = make_moons(n_samples=1000, noise=noise, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

# 模型定义（增大隐藏层到 1000）
class NumpyNet:
    def __init__(self, input_dim=2, hidden_dim=1000, output_dim=2):  # 隐藏层 1000
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        exp_scores = np.exp(self.Z2 - np.max(self.Z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs
    def compute_loss(self, X, y):
        num_examples = X.shape[0]
        correct_log_probs = -np.log(self.probs[range(num_examples), y] + 1e-8)
        return np.sum(correct_log_probs) / num_examples
    def backward(self, X, y):
        num_examples = X.shape[0]
        dZ2 = self.probs.copy()
        dZ2[range(num_examples), y] -= 1
        dZ2 /= num_examples
        self.dW2 = np.dot(self.A1.T, dZ2)
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dA1[self.Z1 <= 0] = 0
        self.dW1 = np.dot(X.T, dA1)
        self.db1 = np.sum(dA1, axis=0, keepdims=True)
    def update_params(self, learning_rate=0.01):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.b1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.b2

# 训练函数（增加轮次到 10000）
def train_numpy_net(model, X_train, y_train, epochs=10000, learning_rate=0.01):
    train_losses = []
    train_accuracies = []
    
    # 记录训练总时间
    start_time = time.time()
    
    for epoch in range(epochs):
        probs = model.forward(X_train)
        loss = model.compute_loss(X_train, y_train)
        
        y_pred_train = np.argmax(probs, axis=1)
        train_acc = np.mean(y_pred_train == y_train)
        
        model.backward(X_train, y_train)
        model.update_params(learning_rate)
        
        # 每 1000 轮打印一次
        if (epoch + 1) % 1000 == 0:
            train_losses.append(loss)
            train_accuracies.append(train_acc)
            print(f"Epoch {epoch+1:5d} | 训练损失: {loss:.4f} | 训练准确率: {train_acc:.4f}")
    
    # 计算训练耗时
    total_time = time.time() - start_time
    avg_epoch_time = total_time / epochs
    print(f"\nNumPy训练总耗时: {total_time:.4f}秒 | 平均每轮耗时: {avg_epoch_time:.6f}秒")
    
    return train_losses, train_accuracies, total_time, avg_epoch_time

# 推理速度测试（保持 CPU，公平对比）
def test_inference_speed(model, X_sample, num_runs=1000):
    start_time = time.time()
    for _ in range(num_runs):
        model.forward(X_sample)
    total_time = time.time() - start_time
    
    avg_inference_time = total_time / num_runs
    print(f"NumPy单样本推理平均耗时: {avg_inference_time:.8f}秒（重复{num_runs}次）")
    return avg_inference_time

# 主流程（保持 CPU，与 PyTorch 公平对比）
if __name__ == '__main__':
    noise = 0.1
    X_train, y_train, X_test, y_test = prepare_data(noise)
    
    model = NumpyNet()
    
    # 训练（10000 轮）
    train_losses, train_accuracies, total_train_time, avg_epoch_time = train_numpy_net(
        model, X_train, y_train, epochs=10000, learning_rate=0.01
    )
    
    # 测试推理速度（用一个样本）
    X_sample = X_train[:1]
    test_inference_speed(model, X_sample, num_runs=1000)