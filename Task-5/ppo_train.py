import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List

# 导入项目已有模块
from game import Directions, Actions, Agent
from pacman import GameState, ClassicGameRules  # 避免命名冲突
from ghostAgents import RandomGhost
from common import preprocess_state
import layout
import textDisplay  # 训练时用无图形显示
from util import manhattanDistance

class PolicyNetwork(nn.Module):
    """策略网络：输入状态，输出动作概率分布（4个动作：上下左右）"""
    def __init__(self, input_dim: int, action_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 隐藏层1
        self.fc2 = nn.Linear(64, 64)         # 隐藏层2
        self.fc3 = nn.Linear(64, action_dim) # 输出层（4个动作）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：返回动作概率（softmax归一化）"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)  # 概率分布
        return x


class ValueNetwork(nn.Module):
    """价值网络：输入状态，输出状态价值（标量）"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # 输出标量价值

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：返回状态价值"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 线性输出（无激活，价值可正可负）
        return x


class PacmanEnv:
    """Pacman环境封装：提供reset()和step()接口，复用项目Game逻辑"""
    def __init__(self, layout_name: str = 'mediumClassic', num_ghosts: int = 4):
        # 加载布局（复用layout.getLayout()）
        self.layout = layout.getLayout(layout_name)
        if self.layout is None:
            raise ValueError(f"Layout {layout_name} not found (check layouts/ folder)")
        
        # 初始化游戏规则和无图形显示（训练时加速）
        self.rules = ClassicGameRules(timeout=30)
        self.display = textDisplay.NullGraphics()
        self.num_ghosts = num_ghosts

        # 重置环境
        self.reset()

    def reset(self) -> GameState:
        """重置环境：返回初始GameState"""
        # 创建幽灵智能体（复用RandomGhost）
        ghost_agents = [RandomGhost(i+1) for i in range(self.num_ghosts)]
        # 创建临时Pacman智能体（动作由PPO Agent选择，不影响）
        dummy_pacman = Agent()  # 空Agent
        
        # 新建游戏（复用ClassicGameRules.newGame()）
        self.game = self.rules.newGame(
            layout=self.layout,
            pacmanAgent=dummy_pacman,
            ghostAgents=ghost_agents,
            display=self.display,
            quiet=True,  # 关闭训练时的控制台输出
            catchExceptions=False
        )
        return self.game.state.deepCopy()

    def step(self, action: Directions) -> Tuple[GameState, float, bool]:
        """
        执行动作：返回（下一个状态，奖励，是否结束）
        需依次执行Pacman和所有幽灵的动作（复用GameState.generateSuccessor()）
        """
        current_state = self.game.state
        # 检查是否已结束（赢/输）
        if current_state.isWin() or current_state.isLose():
            return current_state, 0.0, True

        # 1. 执行Pacman动作（agentIndex=0，Pacman固定为0号智能体）
        legal_actions = current_state.getLegalPacmanActions()
        if action not in legal_actions:
            action = Directions.STOP  # 非法动作替换为STOP
        next_state = current_state.generateSuccessor(0, action)

        # 2. 执行所有幽灵动作（agentIndex=1~num_ghosts）
        for ghost_idx in range(1, self.num_ghosts + 1):
            if next_state.isWin() or next_state.isLose():
                break  # 结束则停止
            # 幽灵选择动作（复用RandomGhost.getAction()）
            ghost_agent = self.game.agents[ghost_idx]
            ghost_action = ghost_agent.getAction(next_state)
            # 生成幽灵动作后的状态
            next_state = next_state.generateSuccessor(ghost_idx, ghost_action)

        # 3. 计算奖励（基于状态变化，复用GameState的分数和位置方法）
        reward = self._calculate_reward(current_state, next_state)

        # 4. 判断是否结束
        done = next_state.isWin() or next_state.isLose()

        # 5. 更新游戏状态
        self.game.state = next_state
        return next_state, reward, done

    def _calculate_reward(self, prev_state: GameState, curr_state: GameState) -> float:
        """奖励函数设计：引导Pacman吃食物、躲幽灵、快速赢"""
        # 1. 基础奖励：分数变化（吃食物+10，吃胶囊+50等）
        reward = curr_state.getScore() - prev_state.getScore()

        # 2. 每步惩罚：从 -1.0 改为 -0.1（大幅降低惩罚力度）
        reward -= 0.1

        # 3. 赢/输额外奖励（保持不变）
        if curr_state.isWin():
           reward += 500.0
        if curr_state.isLose():
           reward -= 500.0

        # 4. 躲避幽灵奖励（保持不变，若后续仍负，可暂时注释这部分，先让模型学“吃食物”）
        pac_prev_pos = prev_state.getPacmanPosition()
        pac_curr_pos = curr_state.getPacmanPosition()
        for ghost_state in curr_state.getGhostStates():
            if not ghost_state.scaredTimer > 0:
                ghost_pos = ghost_state.getPosition()
                prev_dist = manhattanDistance(pac_prev_pos, ghost_pos)
                curr_dist = manhattanDistance(pac_curr_pos, ghost_pos)
                reward += (curr_dist - prev_dist) * 0.5

        return reward


class PPOAgent:
    """PPO智能体：包含轨迹采样、GAE优势计算、网络更新"""
    def __init__(
        self,
        input_dim: int,
        action_dim: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        update_epochs: int = 10,
        batch_size: int = 64
    ):
        # 动作映射：索引→方向（与训练/推理一致）
        self.action_map = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        self.dir_to_idx = {dir: idx for idx, dir in enumerate(self.action_map)}

        # 初始化网络和优化器
        self.policy_net = PolicyNetwork(input_dim, action_dim)
        self.value_net = ValueNetwork(input_dim)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=lr
        )

        # PPO超参数
        self.gamma = gamma          # 折扣因子
        self.gae_lambda = gae_lambda# GAE lambda
        self.clip_epsilon = clip_epsilon  # PPO clip阈值
        self.update_epochs = update_epochs  # 每次采样后的更新轮次
        self.batch_size = batch_size        # 批次大小

    def select_action(self, state_tensor: torch.Tensor, legal_actions: List[Directions]) -> Tuple[Directions, int, torch.Tensor]:
        """
        选择动作：基于当前状态和合法动作，返回（动作，动作索引，对数概率）
        训练时采样动作，推理时选最大概率动作
        """
        self.policy_net.eval()  # 评估模式（无梯度）
        with torch.no_grad():
            probs = self.policy_net(state_tensor)  # 动作概率

        # 过滤合法动作（仅保留上下左右）
        legal_idxs = [self.dir_to_idx[dir] for dir in legal_actions if dir in self.action_map]
        if not legal_idxs:
            return Directions.STOP, -1, torch.tensor(0.0)  # 极端情况返回STOP

        # 合法动作概率归一化（避免非法动作影响）
        legal_probs = probs[legal_idxs]
        legal_probs = legal_probs / legal_probs.sum()  # 重新归一化

        # 训练时采样动作（ multinomial ）
        action_idx = torch.multinomial(legal_probs, num_samples=1).item()
        orig_idx = legal_idxs[action_idx]  # 网络输出中的原始索引
        log_prob = torch.log(probs[orig_idx] + 1e-8)  # 加1e-8避免log(0)

        return self.action_map[orig_idx], orig_idx, log_prob

    def collect_trajectory(self, env: PacmanEnv, max_steps: int = 1000) -> List[Tuple]:
        """收集一条轨迹：[(状态张量, 动作索引, 对数概率, 奖励, 下一个状态, 是否结束)]"""
        trajectory = []
        state = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            step += 1
            # 预处理状态
            state_tensor = preprocess_state(state)
            # 获取合法动作
            legal_actions = state.getLegalPacmanActions()
            # 选择动作
            action, action_idx, log_prob = self.select_action(state_tensor, legal_actions)
            # 执行动作
            next_state, reward, done = env.step(action)
            # 存储轨迹（含下一个状态，用于GAE计算）
            trajectory.append((state_tensor, action_idx, log_prob, reward, next_state, done))
            # 更新状态
            state = next_state

        return trajectory

    def compute_gae(self, rewards: List[float], dones: List[bool], values: List[float], next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算GAE（广义优势估计）：稳定优势函数计算"""
        # 转换为张量
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        # 时序差分误差（TD Error）
        td_errors = rewards + self.gamma * next_value * (1 - dones) - values

        # 从后往前计算GAE
        advantages = torch.zeros_like(rewards)
        advantage = 0.0
        for t in reversed(range(len(rewards))):
            advantage = td_errors[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage

        # 目标价值：V_target = A + V（用于价值网络更新）
        target_values = advantages + values

        # 优势函数标准化（稳定训练）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, target_values

    def update(self, trajectory: List[Tuple]):
        """更新策略网络和价值网络：PPO Clip损失 + 价值网络MSE损失"""
        # 提取轨迹数据
        state_tensors = torch.stack([t[0] for t in trajectory])
        action_idxs = torch.tensor([t[1] for t in trajectory], dtype=torch.long)
        old_log_probs = torch.stack([t[2] for t in trajectory])
        rewards = [t[3] for t in trajectory]
        next_states = [t[4] for t in trajectory]
        dones = [t[5] for t in trajectory]

        # 计算状态价值V(s)和下一个状态价值V(s')
        self.value_net.eval()
        with torch.no_grad():
            # 当前状态价值
            values = self.value_net(state_tensors).squeeze().tolist()
            # 最后一个状态的下一个价值（结束则为0）
            last_done = dones[-1]
            if last_done:
                next_value = 0.0
            else:
                last_next_tensor = preprocess_state(next_states[-1])
                next_value = self.value_net(last_next_tensor).item()

        # 计算GAE优势和目标价值
        advantages, target_values = self.compute_gae(rewards, dones, values, next_value)

        # 多次迭代更新（PPO核心：多次利用轨迹数据）
        self.policy_net.train()
        self.value_net.train()
        num_samples = len(trajectory)

        for _ in range(self.update_epochs):
            # 打乱样本索引（避免时序相关性）
            indices = torch.randperm(num_samples)
            for start in range(0, num_samples, self.batch_size):
                # 批次索引
                end = start + self.batch_size
                batch_idx = indices[start:end]

                # 批次数据
                batch_states = state_tensors[batch_idx]
                batch_actions = action_idxs[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_targets = target_values[batch_idx]

                # 1. 策略网络损失（PPO Clip损失）
                current_probs = self.policy_net(batch_states)
                current_log_probs = torch.log(
                    current_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1) + 1e-8
                )
                # 概率比率：r(θ) = π(θ)/π(θ_old)
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                # PPO Clip损失：min(r*A, clip(r, 1-ε, 1+ε)*A)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()  # 负号：最小化→最大化

                # 2. 价值网络损失（MSE：预测价值 vs 目标价值）
                current_values = self.value_net(batch_states).squeeze()
                value_loss = F.mse_loss(current_values, batch_targets)

                # 3. 总损失（策略损失 + 价值损失）
                total_loss = policy_loss + value_loss

                # 4. 反向传播和优化
                self.optimizer.zero_grad()
                total_loss.backward()
                # 梯度裁剪：防止梯度爆炸（最大范数0.5）
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    max_norm=0.5
                )
                self.optimizer.step()

    def save_model(self, path: str):
        """保存模型参数：策略网络、价值网络、优化器"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """加载模型参数（用于推理）"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {path}")
        
if __name__ == "__main__":
    # 1. 初始化环境（用smallClassic小地图训练，速度快）
    env = PacmanEnv(layout_name="smallClassic", num_ghosts=2)
    
    # 2. 计算状态输入维度（先预处理一次初始状态，获取张量长度）
    init_state = env.reset()
    init_state_tensor = preprocess_state(init_state)
    input_dim = init_state_tensor.shape[0]  # 输入维度 = 状态张量的长度
    
    # 3. 初始化PPO智能体
    ppo_agent = PPOAgent(
        input_dim=input_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        update_epochs=10,
        batch_size=64
    )
    
    # 4. 开始训练（训练100局，可根据需要调整）
    total_episodes = 300
    for episode in range(total_episodes):
        # 收集一条轨迹（最大步数1000，避免无限循环）
        trajectory = ppo_agent.collect_trajectory(env, max_steps=1000)
        
        # 更新PPO模型（用收集的轨迹训练网络）
        ppo_agent.update(trajectory)
        
        # 计算当前局的总奖励（轨迹中所有奖励之和）
        total_reward = sum([t[3] for t in trajectory])
        
        # 打印训练进度
        print(f"Episode [{episode+1}/{total_episodes}] | Total Reward: {total_reward:.1f} | Trajectory Length: {len(trajectory)}")
        
        # 每20局保存一次模型（避免训练中断丢失）
        if (episode + 1) % 20 == 0:
            ppo_agent.save_model(f"ppo_pacman_model_ep{episode+1}.pth")
    
    # 训练结束后保存最终模型
    ppo_agent.save_model("ppo_pacman_final_model.pth")
    print("Training finished! Final model saved as 'ppo_pacman_final_model.pth'")
