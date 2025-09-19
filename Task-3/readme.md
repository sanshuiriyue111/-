# 文档类：
图像恢复任务实验报告.pdf
Task_3学习笔记.pdf

# 代码类：
data_augmentation.py #数据增强（为训练提供多样本）工具代码
evaluate_noise.py  # 噪声评估（分析图像噪声分布、强度）脚本
training_data.py # 训练数据预处理（加载、清洗）代码

在training文件夹下：
dataset.py # 数据集加载与预处理类
hinet.py # 模型结构定义
train.py # 模型训练主程序
inference.py # 模型推理脚本
hinet_checkpoints # 模型权重文件
your_result # 推理结果（去噪后图像、定量指标）存储目录

#结果与分析类
noise_analysis_summary.txt  # 噪声分析核心结论总结
noise_evaluation_results/   # 噪声评估可视化结果（含大量截图，大文件）

