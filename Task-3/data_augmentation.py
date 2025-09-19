#!/usr/bin/env python
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance

# 配置路径
original_dir = "/Users/duqiu/Desktop/焦点计划/Task-3/train_dataset/original"
degraded_dir = "/Users/duqiu/Desktop/焦点计划/Task-3/train_dataset/degraded"

# 增强后保存路径（需提前创建）
aug_original_dir = "/Users/duqiu/Desktop/焦点计划/Task-3/train_dataset/aug_original"
aug_degraded_dir = "/Users/duqiu/Desktop/焦点计划/Task-3/train_dataset/aug_degraded"
os.makedirs(aug_original_dir, exist_ok=True)
os.makedirs(aug_degraded_dir, exist_ok=True)

# 每个原始样本生成的增强样本数量
augment_times = 3

def random_geometric_transform(original_img, degraded_img):
    """
    对清晰图和退化图进行同步几何变换
    包含：随机旋转（±5°）、随机缩放（0.9-1.1倍）、随机水平翻转
    """
    h, w = original_img.shape[:2]
    #取前两个维度，高度和宽度
    transformed_original = original_img.copy()
    transformed_degraded = degraded_img.copy()
    
    # 随机水平翻转（50%概率）
    if np.random.random() > 0.5:
        transformed_original = cv2.flip(transformed_original, 1)
        transformed_degraded = cv2.flip(transformed_degraded, 1)
    #后面那个参数取1代表水平翻转
    # 随机旋转（±5°）
    angle = np.random.uniform(-5, 5)
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)#生成旋转矩阵
    # 旋转后用白色填充背景（适合文本图像）
    transformed_original = cv2.warpAffine(
        transformed_original, M_rot, (w, h), borderValue=(255, 255, 255)
    )
    transformed_degraded = cv2.warpAffine(
        transformed_degraded, M_rot, (w, h), borderValue=(255, 255, 255)
    )
    
    # 随机缩放（0.9-1.1倍）
    scale = np.random.uniform(0.9, 1.1)
    # 计算缩放后的尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    # 缩放
    transformed_original = cv2.resize(transformed_original, (new_w, new_h))
    transformed_degraded = cv2.resize(transformed_degraded, (new_w, new_h))
    # 若缩放后尺寸小于原尺寸，用白色填充回原尺寸（保持批次图像尺寸一致）
    if scale < 1.0:
        pad_w = (w - new_w) // 2
        pad_h = (h - new_h) // 2
        transformed_original = cv2.copyMakeBorder(
            transformed_original, pad_h, h - new_h - pad_h,
            pad_w, w - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )#先顶部填充，通过相减得到底部填充，先填左后填右
        transformed_degraded = cv2.copyMakeBorder(
            transformed_degraded, pad_h, h - new_h - pad_h,
            pad_w, w - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    # 若缩放后尺寸大于原尺寸，裁剪回原尺寸
    elif scale > 1.0:
        start_w = (new_w - w) // 2
        start_h = (new_h - h) // 2
        transformed_original = transformed_original[start_h:start_h+h, start_w:start_w+w]
        transformed_degraded = transformed_degraded[start_h:start_h+h, start_w:start_w+w]
    
    return transformed_original, transformed_degraded

def random_illumination_adjust(original_img, degraded_img):
    """
    对清晰图和退化图进行同步光照调整
    包含：随机亮度调整（±20%）、随机对比度调整（0.8-1.2倍）
    """
    # 转换为PIL Image以使用ImageEnhance
    pil_original = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    pil_degraded = Image.fromarray(cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB))
    
    # 随机亮度调整（±20%）
    brightness_factor = np.random.uniform(0.8, 1.2)
    enhancer_bright = ImageEnhance.Brightness(pil_original)
    pil_original = enhancer_bright.enhance(brightness_factor)
    enhancer_bright = ImageEnhance.Brightness(pil_degraded)
    pil_degraded = enhancer_bright.enhance(brightness_factor)
    
    # 随机对比度调整（0.8-1.2倍）
    contrast_factor = np.random.uniform(0.8, 1.2)
    enhancer_contrast = ImageEnhance.Contrast(pil_original)
    pil_original = enhancer_contrast.enhance(contrast_factor)
    enhancer_contrast = ImageEnhance.Contrast(pil_degraded)
    pil_degraded = enhancer_contrast.enhance(contrast_factor)
    
    # 转回OpenCV格式（RGB→BGR）
    adjusted_original = cv2.cvtColor(np.array(pil_original), cv2.COLOR_RGB2BGR)
    adjusted_degraded = cv2.cvtColor(np.array(pil_degraded), cv2.COLOR_RGB2BGR)
    
    return adjusted_original, adjusted_degraded

# 批量处理图像
image_names = [f for f in os.listdir(original_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for img_name in tqdm(image_names, desc="数据增强进度"):
    # 读取原图和退化图
    original_path = os.path.join(original_dir, img_name)
    degraded_path = os.path.join(degraded_dir, img_name)
    
    original_img = cv2.imread(original_path)
    degraded_img = cv2.imread(degraded_path)
    
    if original_img is None or degraded_img is None:
        print(f"警告：无法读取图像 {img_name}，已跳过")
        continue
    
    # 生成多个增强样本
    for i in range(augment_times):
        # 步骤1：几何变换（旋转、缩放、翻转）
        aug_geo_original, aug_geo_degraded = random_geometric_transform(original_img, degraded_img)
        
        # 步骤2：光照调整（亮度、对比度）
        aug_final_original, aug_final_degraded = random_illumination_adjust(aug_geo_original, aug_geo_degraded)
        
        # 保存增强后的图像
        base_name, ext = os.path.splitext(img_name)
        aug_name = f"{base_name}_aug{i}{ext}"
        
        cv2.imwrite(os.path.join(aug_original_dir, aug_name), aug_final_original)
        cv2.imwrite(os.path.join(aug_degraded_dir, aug_name), aug_final_degraded)

print(f"数据增强完成！共生成 {len(image_names) * augment_times} 个增强样本")
print(f"增强后的清晰图保存至：{aug_original_dir}")
print(f"增强后的退化图保存至：{aug_degraded_dir}")

