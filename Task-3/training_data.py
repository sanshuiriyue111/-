#!/usr/bin/env python
import cv2
import numpy as np
import os
import random

# 
# 原始清晰图像目录（已下载的original路径）
original_dir = "/Users/duqiu/Desktop/焦点计划/JotangRecrument-main/ML/task_3/image_pairs/original"
# 训练集保存路径（新建目录，用于存放生成的原图和退化图）
train_root = "/Users/duqiu/Desktop/焦点计划/train_dataset"
train_original_dir = os.path.join(train_root, "original")  # 训练集原图（复用original的图）
train_degraded_dir = os.path.join(train_root, "degraded")  # 训练集退化图（生成的图）

# 创建保存目录
os.makedirs(train_original_dir, exist_ok=True)
os.makedirs(train_degraded_dir, exist_ok=True)

# 2. 定义退化函数（模拟模糊+噪声，贴近测试集的退化类型）
def add_degradation(clear_img):
    """
    对清晰图像添加退化（模糊+噪声），模拟真实场景的图像退化
    """
    # 随机选择模糊类型（模拟不同原因的模糊）
    blur_type = random.choice(["gaussian", "motion"])
    
    # 步骤1：添加模糊
    if blur_type == "gaussian":
        # 高斯模糊（模拟失焦模糊）
        ksize = random.choice([3, 5, 7, 9, 11])  # 模糊核大小随机，增加多样性
        blurred = cv2.GaussianBlur(clear_img, (ksize, ksize), 0)
    else:
        # 运动模糊（模拟拍摄时抖动）
        ksize = random.choice([5, 7, 9, 11])
        # 生成运动模糊核（随机方向和长度）
        kernel = np.zeros((ksize, ksize))
        angle = random.uniform(0, 180)  # 随机运动角度
        if angle < 45 or angle > 135:
            kernel[int((ksize-1)/2), :] = 1  # 水平方向运动
        else:
            kernel[:, int((ksize-1)/2)] = 1  # 垂直方向运动
        kernel = kernel / ksize  # 归一化
        blurred = cv2.filter2D(clear_img, -1, kernel)
    
    # 步骤2：添加噪声（模拟传感器噪声或传输干扰）
    noise_type = random.choice(["gaussian", "salt_pepper"])
    if noise_type == "gaussian":
        # 高斯噪声
        mean = 0
        var = random.uniform(1, 10)  # 噪声强度随机
        sigma = var **0.5
        gauss = np.random.normal(mean, sigma, blurred.shape).astype(np.int16)
        degraded = np.clip(blurred.astype(np.int16) + gauss, 0, 255).astype(np.uint8)#clip起到限制在0-255的作用
    else:
        # 椒盐噪声
        s_vs_p = 0.5  # 盐噪声与椒噪声的比例
        amount = random.uniform(0.001, 0.01)  # 噪声比例随机
        degraded = np.copy(blurred)
        # 盐噪声（白色点）
        num_salt = np.ceil(amount * blurred.size * s_vs_p)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in blurred.shape[:2]]#blurred.shape【：2】取前两个维度，在行和列分别遍历随机
        degraded[coords[0], coords[1], :] = 255#盐噪声就是255
        # 椒噪声（黑色点）
        num_pepper = np.ceil(amount * blurred.size * (1 - s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in blurred.shape[:2]]
        degraded[coords[0], coords[1], :] = 0#椒噪声就是0
    
    return degraded

# 3. 批量生成训练数据对
# 遍历original目录中的所有图像
for img_name in os.listdir(original_dir):
    # 只处理png图像（根据实际文件格式调整）
    if img_name.endswith(".png"):
        # 读取清晰原图
        clear_path = os.path.join(original_dir, img_name)
        clear_img = cv2.imread(clear_path)
        if clear_img is None:
            print(f"跳过无法读取的文件：{img_name}")
            continue
        
        # 生成退化图像（每个原图生成3个不同退化程度的样本，增加数据量）
        for i in range(3):
            degraded_img = add_degradation(clear_img)
            
            # 保存原图（重命名，避免重复）
            base_name = os.path.splitext(img_name)[0]  # 去掉扩展名
            save_original_name = f"{base_name}_sample_{i}.png"
            cv2.imwrite(os.path.join(train_original_dir, save_original_name), clear_img)
            
            # 保存对应的退化图（与原图同名，确保配对）
            save_degraded_name = f"{base_name}_sample_{i}.png"
            cv2.imwrite(os.path.join(train_degraded_dir, save_degraded_name), degraded_img)

print(f"训练集生成完成！\n原图保存路径：{train_original_dir}\n退化图保存路径：{train_degraded_dir}")

