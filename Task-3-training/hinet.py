#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """基础残差块"""
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class HybridIterativeModule(nn.Module):
    """混合迭代模块，HINet核心组件"""
    def __init__(self, channels, num_blocks=3):
        super(HybridIterativeModule, self).__init__()
        self.blocks = nn.Sequential(*[BasicBlock(channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0)

    def forward(self, x, prev_feat):
        # 融合当前特征和前一次迭代特征
        fused = torch.cat([x, prev_feat], dim=1)
        fused = self.conv(fused)
        
        # 通过残差块处理
        out = self.blocks(fused)
        
        # 残差连接
        return out + x

class HINet(nn.Module):
    """HINet模型整体结构"""
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 base_channels=32,
                 num_iterations=4,  # 迭代次数，影响效果和速度
                 num_blocks_per_iter=3):  # 每次迭代的残差块数量
        super(HINet, self).__init__()
        
        # 输入特征提取
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 混合迭代模块
        self.iterative_modules = nn.ModuleList([
            HybridIterativeModule(base_channels, num_blocks_per_iter)
            for _ in range(num_iterations)
        ])
        
        # 输出重建
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始特征
        feat = self.initial_conv(x)
        prev_feat = torch.zeros_like(feat)  # 初始前序特征为0
        
        # 多轮迭代
        for module in self.iterative_modules:
            feat = module(feat, prev_feat)
            prev_feat = feat  # 更新前序特征
        
        # 输出重建
        out = self.final_conv(feat)
        return out + x  # 残差连接，直接学习退化残差

