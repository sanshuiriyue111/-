import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取 C 代码生成的文件
def load_matrix(filename):
    with open(filename, 'r') as f:
        rows, cols = map(int, f.readline().split())
        matrix = []
        for _ in range(rows):
            row = list(map(float, f.readline().split()))
            matrix.append(row)
    return np.array(matrix)

def load_train_log(filename):
    log = np.loadtxt(filename)
    epochs = log[:, 0]
    losses = log[:, 1]
    accs = log[:, 2]
    return epochs, losses, accs

# 2. 绘制训练曲线（损失+准确率）
def plot_train_curves(epochs, losses, accs):
    plt.figure(figsize=(12, 4))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, color='red')
    plt.title('Train_losses_curve')
    plt.xlabel('turns')
    plt.ylabel('losses')
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accs, color='blue')
    plt.title('Train_accs_curve')
    plt.xlabel('turns')
    plt.ylabel('accs')
    plt.tight_layout()
    plt.show()

# 3. 绘制权重热力图
def plot_weight_heatmaps(W1, W2):
    plt.figure(figsize=(12, 6))
    # W1 热力图（输入→隐藏层）
    plt.subplot(1, 2, 1)
    sns.heatmap(W1, cmap='coolwarm', annot=False)
    plt.title('Input-Hidden (W1)')
    plt.xlabel('Hidden')
    plt.ylabel('Output_x (x1, x2)')
    # W2 热力图（隐藏→输出层）
    plt.subplot(1, 2, 2)
    sns.heatmap(W2, cmap='coolwarm', annot=False)
    plt.title('Hidden-Output (W2)')
    plt.xlabel('Output (0, 1)')
    plt.ylabel('Hidden')
    plt.tight_layout()
    plt.show()

# 主函数：执行可视化
if __name__ == '__main__':
    W1 = load_matrix("W1.txt")
    W2 = load_matrix("W2.txt")
    epochs, losses, accs = load_train_log("train_log.txt")
    
    plot_train_curves(epochs, losses, accs)
    plot_weight_heatmaps(W1, W2)