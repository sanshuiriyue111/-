#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from pacman import GameState
from typing import Tuple,List
import numpy as np


#状态预处理
def preprocess_state(state:GameState) -> torch.Tensor:
  # 1、从GameState提取核心信息（复用GameState的访问方法）
    pac_pos = state.getPacmanPosition()          # Pacman位置 (x,y)
    ghost_pos = state.getGhostPositions()        # 幽灵位置列表 [(x1,y1), (x2,y2), ...]
    food_grid = state.getFood()                  # 食物网格（Grid类型）
    ghost_states = state.getGhostStates()        # 幽灵状态（含scaredTimer）
    width, height = food_grid.width, food_grid.height  # 网格大小（从食物网格获取）

    # 2、处理Pacman位置：归一化到[0,1]（避免网格大小影响）
    pac_x, pac_y = pac_pos
    pac_tensor = torch.tensor([pac_x / width, pac_y / height], dtype=torch.float32)

    # 3、处理幽灵位置：最多保留4个幽灵（项目默认最大幽灵数），不足补0
    max_ghosts = 4
    ghost_tensor = []
    for i in range(max_ghosts):
        if i < len(ghost_pos):
            gx, gy = ghost_pos[i]
            ghost_tensor.extend([gx / width, gy / height])  # 归一化
        else:
            ghost_tensor.extend([0.0, 0.0])  # 不足补0
    ghost_tensor = torch.tensor(ghost_tensor, dtype=torch.float32)

    # 4、处理食物网格：扁平化为一维张量（有食物为1，无为0）
    food_flat = torch.zeros(width * height, dtype=torch.float32)
    for (x, y) in food_grid.asList():  # 复用Grid.asList()获取食物位置
        idx = x + y * width  # 扁平索引（x为宽方向，y为高方向）
        food_flat[idx] = 1.0

    # 5、处理幽灵恐惧时间：归一化到[0,1]（复用game.py中的SCARED_TIME=40）
    scared_tensor = []
    for i in range(max_ghosts):
        if i < len(ghost_states):
            scared_timer = ghost_states[i].scaredTimer  # 从幽灵状态获取恐惧时间
            scared_tensor.append(scared_timer / 40.0)  # 归一化（40为最大恐惧时间）
        else:
            scared_tensor.append(0.0)
    scared_tensor = torch.tensor(scared_tensor, dtype=torch.float32)

    # 6、拼接所有特征（总维度：2 + 8 + (width*height) + 4）
    state_tensor = torch.cat([pac_tensor, ghost_tensor, food_flat, scared_tensor], dim=0)
    return state_tensor
