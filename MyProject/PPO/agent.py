
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from collections import namedtuple

from model import Actor,Critic

class ReplayBuffer: #单个actor的样本池
    def __init__(self,capacity,batch_size):
        self.Experience = namedtuple('Experience', ['state', 'action', 'reward', 'value', 'done'])
        self.capacity=capacity
        self.buffer=[]
        self.position=0## 当前存储位置的指针
    def push(self, state, action, reward, value, done):
        """将经验添加到经验回放池"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # 如果缓冲区未满，则添加一个占位符
        self.buffer[self.position] = self.Experience(state, action, reward, value, done)
        self.position = (self.position + 1) % self.capacity  # 更新指针位置

    def sample(self, batch_size):
        """从经验回放池中随机采样一批经验"""
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))  # 如果缓冲区中的经验不足一个批次，则返回所有经验
        return random.sample(self.buffer, batch_size)
    def clear_all(self):
        '''清空所有数据'''
        self.buffer.clear()
        self.position=0
        
    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self,**kwargs):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma=kwargs.get('gamma',0.99)
        self.Lambda=kwargs.get('Lambda',0.95)
        self.lr_start=kwargs.get('lr_start',0.001)
        self.lr_decay=kwargs.get('lr_decay',0.995)
        self.lr_end=kwargs.get('lr_end',0.0001)
        self.eps_clip=kwargs.get('eps_clip',0.2)
        self.batch_size=kwargs.get('batch_size',64)
        self.k_epochs=kwargs.get('K_epochs',4)


        self.state_dim=kwargs.get('state_dim')
        self.action_dim=kwargs.get('action_dim')

        self.memory=ReplayBuffer(kwargs.get('memory_size',10000),self.batch_size)
        self.actor=Actor(self.state_dim,self.action_dim).to(self.device)
        self.old_actor=Actor(self.state_dim,self.action_dim).to(self.device)
        self.critic=Critic(self.state_dim).to(self.device)

        self.optimizer_actor=optim.Adam(self.actor.parameters(),lr=self.lr_start)
        self.optimizer_critic=optim.Adam(self.critic.parameters(),lr=self.lr_start)
        self.scheduler_actor=optim.lr_scheduler.ExponentialLR(self.optimizer_actor,gamma=self.lr_decay)
        self.scheduler_critic=optim.lr_scheduler.ExponentialLR(self.optimizer_critic,gamma=self.lr_decay)

    def get_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor.select_action(state)
        value = self.critic(state)
        return action.item(), value.item()
    def compute_gae(self, rewards, values, dones, next_value):
        """计算GAE优势函数"""
        advantages = []
        gae = 0
        
        # 从后往前计算GAE
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_val = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_val * next_non_terminal - values[i]
            gae = delta + self.gamma * self.Lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
            
        # 获取所有经验
        data = self.memory.buffer
        states, actions, rewards, values, dones = zip(*data)
        
        # 计算下一个状态的价值（这里简化为0，实际应该是最后一个状态的下一个状态的价值）
        next_value = 0
        
        # 计算GAE优势函数
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # 保存旧策略
        self.old_actor.load_state_dict(self.actor.state_dict())
        
        # 转换为tensor (优化性能)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs = self.old_actor.get_log_prob(states, actions).detach()
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        for _ in range(self.k_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算当前策略的log概率
                current_log_probs = self.actor.get_log_prob(batch_states, batch_actions)
                
                # 计算比率
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                
                # 计算PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值函数损失
                current_values = self.critic(batch_states).squeeze(-1)
                critic_loss = F.mse_loss(current_values, batch_returns)
                
                # 更新Actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()
                
                # 更新Critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
        
        # 更新学习率
        if self.optimizer_actor.param_groups[0]['lr'] > self.lr_end:
            self.scheduler_actor.step()
        if self.optimizer_critic.param_groups[0]['lr'] > self.lr_end:
            self.scheduler_critic.step()

    def save_model(self, epoch):
        """保存模型"""
        save_dir = './model'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save({
            'epoch': epoch,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
        }, os.path.join(save_dir, f'ppo_model_epoch_{epoch}.pth'))
        
        print(f"Model saved at epoch {epoch}")
    
    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        
        print(f"Model loaded from {model_path}")
        return checkpoint['epoch']