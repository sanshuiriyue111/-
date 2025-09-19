#!/usr/bin/env python
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#日志级别为INFO，日志输出格式包含：日志记录时间，日志级别，日志具体内容
logger = logging.getLogger(__name__)
#创建一个日志实例
class ImageRestorationDataset(Dataset):
    def __init__(self, degraded_dir, original_dir, image_size=256):
        """
        初始化数据集
        :param degraded_dir: 退化图像目录
        :param original_dir: 原始清晰图像目录
        :param image_size: 图像尺寸（已增强数据仅做尺寸调整）
        """
        self.degraded_dir = degraded_dir
        self.original_dir = original_dir
        self.image_size = image_size
        
        # 获取并匹配图像对
        self.image_pairs = self._get_matched_pairs()
        
        # 仅进行必要的转换（不重复增强）
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), transforms.InterpolationMode.BILINEAR),#利用双线性插值的方法进行缩放
            transforms.ToTensor(),#转化为Pytorch可识别的张量形式
        ])
        
        logger.info(f"数据集加载完成，共找到 {len(self.image_pairs)} 对有效图像")

    def _get_matched_pairs(self):
        """匹配退化图像和原始图像，确保文件名一一对应"""
        degraded_files = {f for f in os.listdir(self.degraded_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
                         #集合推导式的形式
        original_files = {f for f in os.listdir(self.original_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        
        # 找到共同的文件名（确保一一对应）
        common_files = degraded_files & original_files
        if not common_files:
            raise ValueError("未找到匹配的退化图像和原始图像，请检查文件名是否一致")
            
        # 过滤不匹配的文件并记录
        missing_in_degraded = original_files - common_files
        missing_in_original = degraded_files - common_files
        
        if missing_in_degraded:
            logger.warning(f"原始图像目录中有 {len(missing_in_degraded)} 个文件在退化图像目录中未找到")
        if missing_in_original:
            logger.warning(f"退化图像目录中有 {len(missing_in_original)} 个文件在原始图像目录中未找到")
            
        # 构建图像对路径
        pairs = []
        for filename in common_files:
            degraded_path = os.path.join(self.degraded_dir, filename)
            original_path = os.path.join(self.original_dir, filename)
            pairs.append((degraded_path, original_path))
            
        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        try:
            degraded_path, original_path = self.image_pairs[idx]
            
            # 读取图像
            degraded_img = Image.open(degraded_path).convert('RGB')
            original_img = Image.open(original_path).convert('RGB')
            
            # 应用转换
            degraded_tensor = self.transform(degraded_img)
            original_tensor = self.transform(original_img)
            
            return degraded_tensor, original_tensor
            
        except Exception as e:
            logger.error(f"加载图像对 {self.image_pairs[idx]} 时出错: {str(e)}")
            # 返回一个空张量，训练时可跳过
            return torch.zeros(3, self.image_size, self.image_size), torch.zeros(3, self.image_size, self.image_size)
