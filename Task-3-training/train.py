#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import logging
from tqdm import tqdm
import time

from dataset import ImageRestorationDataset
from hinet import HINet

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 加载数据集
    train_dataset = ImageRestorationDataset(
        degraded_dir=args.degraded_dir,
        original_dir=args.original_dir,
        image_size=args.image_size
    )
    
    # 数据加载器（优化配置）
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    
    # 初始化模型
    model = HINet(
        base_channels=args.base_channels,
        num_iterations=args.num_iterations,
        num_blocks_per_iter=args.num_blocks_per_iter
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.L1Loss()  # L1损失更适合图像恢复，收敛更快
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # 加载检查点（如果需要继续训练）
    start_epoch = 0
    best_loss = float('inf')
    if args.resume and os.path.exists(args.checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) if f.startswith('checkpoint_')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            checkpoint_path = os.path.join(args.checkpoint_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            
            logger.info(f"从检查点 {checkpoint_path} 恢复训练，起始 epoch: {start_epoch}")
    
    # 混合精度训练（加速并减少显存占用）
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # 训练循环
    logger.info("开始训练...")
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        # 使用tqdm显示进度
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for degraded, original in progress_bar:
            # 数据移至设备
            degraded = degraded.to(device)
            original = original.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(degraded)
                    loss = criterion(outputs, original)
            else:
                outputs = model(degraded)
                loss = criterion(outputs, original)
            
            # 反向传播和优化
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # 累计损失
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (len(progress_bar))
            progress_bar.set_postfix({'平均损失': f'{avg_loss:.6f}'})
        
        # 计算 epoch 耗时
        epoch_time = time.time() - start_time
        logger.info(f'Epoch {epoch+1}/{args.num_epochs} - 平均损失: {avg_loss:.6f} - 耗时: {epoch_time:.2f}秒')
        
        # 学习率调度
        scheduler.step(avg_loss)
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss
        }
        
        # 保存当前epoch检查点
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_{epoch+1}.pth'))
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            logger.info(f"保存新的最佳模型，损失: {best_loss:.6f}")
        
        # 每10个epoch保存一次模型副本（防止意外中断）
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_{epoch+1}_backup.pth'))
    
    logger.info(f"训练完成！最佳损失: {best_loss:.6f}")
    logger.info(f"模型权重保存在: {args.checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HINet图像恢复模型训练')
    
    # 数据路径（默认使用你的路径）
    parser.add_argument('--degraded_dir', type=str,
                      default='/Users/duqiu/Desktop/焦点计划/Task-3/train_dataset/aug_degraded',
                      help='退化图像目录')
    parser.add_argument('--original_dir', type=str,
                      default='/Users/duqiu/Desktop/焦点计划/Task-3/train_dataset/aug_original',
                      help='原始清晰图像目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    
    # 模型参数（可调整以平衡速度和效果）
    parser.add_argument('--base_channels', type=int, default=32, help='基础通道数')
    parser.add_argument('--num_iterations', type=int, default=4, help='迭代次数')
    parser.add_argument('--num_blocks_per_iter', type=int, default=3, help='每次迭代的残差块数量')
    
    # 其他设置
    parser.add_argument('--checkpoint_dir', type=str, default='hinet_checkpoints', help='模型保存目录')
    parser.add_argument('--resume', action='store_true', help='是否从最近的检查点继续训练')
    
    args = parser.parse_args()
    main(args)
