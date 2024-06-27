import torch
import torch.nn as nn
from torch.distributions import Categorical

# 是否使用GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Memory:#记忆
    def __init__(self):
        """初始化"""
        self.actions = []  # 行动(共4种)
        self.states = []  # 状态, 由8个数字组成
        self.logprobs = []  # 概率
        self.rewards = []  #每个状态的奖励
        self.is_terminals = []  # 包含每个状态是否是游戏终止状态

    def clear_memory(self):
        """清除memory"""
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


class ActorCritic(nn.Module):#策略网络
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()#父类初始化

        # 行动
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # 评判
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)#得到做各个动作的概率
        dist = Categorical(action_probs)#按照概率分布进行抽样
        action = dist.sample()#采样一个动作

        memory.states.append(state)#记住一个状态
        memory.actions.append(action)#记住一个动作
        memory.logprobs.append(dist.log_prob(action))#记住动作的log概率

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)
        state_value = torch.squeeze(state_value)

        return action_logprobs, state_value, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr#学习率
        self.betas = betas#动量
        self.gamma = gamma#折扣因子
        self.eps_clip = eps_clip#探索策略
        self.K_epochs = K_epochs#更新次数

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)#策略网络
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)#旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())#旧策略网络参数初始化为策略网络参数

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)#优化器
        self.MseLoss = nn.MSELoss()#均方误差损失函数


    def update(self, memory):
        """更新梯度"""
        # 蒙特卡罗预测状态回报，计算每个状态的折扣奖励
        rewards = []
        discounted_reward = 0    #折扣奖励=r_t+γ*r_t+1+γ^2*r_t+2+...
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)#标准化奖励


        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        for _ in range(self.K_epochs):#更新K_epochs次
            #计算策略网络的评估值
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            #计算策略网络的损失函数公式  参考了MseLoss和dist_entropy辅助
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())