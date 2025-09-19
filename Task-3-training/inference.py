#!/usr/bin/env python
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import argparse
import logging
from tqdm import tqdm

from hinet import HINet

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 固定测试集路径（无需命令行输入）
    input_dir = "/Users/duqiu/Desktop/焦点计划/JotangRecrument-main/ML/task_3/image_pairs/blurred"
    # 结果保存目录（固定为your_result）
    output_dir = "your_result"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"结果将保存到: {output_dir}")
    
    # 检查测试集路径是否存在
    if not os.path.exists(input_dir):
        logger.error(f"测试集路径不存在，请检查: {input_dir}")
        return
    
    # 加载模型
    model = HINet(
        base_channels=args.base_channels,
        num_iterations=args.num_iterations,
        num_blocks_per_iter=args.num_blocks_per_iter
    ).to(device)
    
    # 加载模型权重
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        return
        
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    logger.info(f"已加载模型权重: {args.model_path}")
    
    # 图像转换 - 与训练时保持一致
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    
    # 逆转换 - 将Tensor转换为图像
    to_pil = transforms.ToPILImage()
    
    # 获取所有测试图像
    test_images = [f for f in os.listdir(input_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not test_images:
        logger.error(f"在测试集路径 {input_dir} 中未找到图像文件")
        return
    
    logger.info(f"找到 {len(test_images)} 个测试图像，开始处理...")
    
    # 批量处理图像
    with torch.no_grad():  # 禁用梯度计算，加速推理
        for filename in tqdm(test_images, desc="处理图像"):
            try:
                # 读取图像
                image_path = os.path.join(input_dir, filename)
                img = Image.open(image_path).convert('RGB')
                original_size = img.size  # 保存原始尺寸
                
                # 预处理
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                # 模型推理
                output_tensor = model(input_tensor)
                
                # 后处理
                output_tensor = output_tensor.squeeze(0).cpu()
                output_img = to_pil(output_tensor.clamp(0, 1))  # 确保像素值在有效范围内
                
                # 恢复到原始尺寸
                output_img = output_img.resize(original_size, Image.BILINEAR)
                
                # 保存结果
                output_path = os.path.join(output_dir, filename)
                output_img.save(output_path)
                
            except Exception as e:
                logger.error(f"处理 {filename} 时出错: {str(e)}")
    
    logger.info(f"所有图像处理完成，结果保存在: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HINet图像恢复模型推理（固定测试集路径）')
    
    # 仅保留必要的可配置参数（模型路径和网络参数）
    parser.add_argument('--model_path', type=str, default='hinet_checkpoints/best_model.pth',
                      help='模型权重路径，默认: hinet_checkpoints/best_model.pth')
    parser.add_argument('--image_size', type=int, default=256,
                      help='图像尺寸，需与训练时一致，默认: 256')
    parser.add_argument('--base_channels', type=int, default=32,
                      help='基础通道数，需与训练时一致，默认: 32')
    parser.add_argument('--num_iterations', type=int, default=4,
                      help='迭代次数，需与训练时一致，默认: 4')
    parser.add_argument('--num_blocks_per_iter', type=int, default=3,
                      help='每次迭代的残差块数量，需与训练时一致，默认: 3')
    
    args = parser.parse_args()
    main(args)
