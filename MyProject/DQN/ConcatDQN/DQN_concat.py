
import math
import random
from collections import namedtuple, deque


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):#定义一个ReplayMemory类，用于存储经验

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)#定义一个双向队列，容量为capacity

    def push(self, *args):#定义一个push函数，用于添加经验到memory中
        """Save a transition"""
        self.memory.append(Transition(*args))#添加一个Transition对象到双向队列中

    def sample(self, batch_size):#定义一个sample函数，用于从memory中随机采样batch_size个Transition对象
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        if n_observations <= 0 or n_actions <= 0:
            raise ValueError("n_observations and n_actions must be positive integers.")
        
        
        self.input=nn.Linear(n_observations,1024)
        self.ln1=nn.Linear(512,256)
        self.ln2=nn.Linear(128,64)
        self.ln3=nn.Linear(32,16)

        self.output=nn.Linear(688,n_actions)
    def forward(self, x):
        x=F.relu(self.input(x))
        x1,x=torch.chunk(x,2,dim=1)
        x=F.relu(self.ln1(x))
        x2,x=torch.chunk(x,2,dim=1)
        x=F.relu(self.ln2(x))
        x3,x=torch.chunk(x,2,dim=1)
        x=F.relu(self.ln3(x))
        x=torch.cat((x1,x2,x3,x),dim=1)
        x=self.output(x)
 
        return x



class Agent:#定义Agent类，用于训练DQN网络

    def __init__(self,BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY,TAU,LR,env):
        #env = gym.make("LunarLander-v2")#定义环境
        self.env = env
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START#定义epsilon的起始值
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY#定义epsilon的衰减值
        self.TAU = TAU#定义target网络的更新参数
        self.LR = LR#定义学习率
        self.steps_done = 0#定义steps_done，用于计算epsilon的衰减值


        self.state , self.info = self.env.reset()#初始化环境，获取初始状态
        self.n_actions = env.action_space.n  # 获取动作空间的数量
        self.n_observations = len(self.state)#获取状态的维度


        self.device = torch.device(#定义device，用于指定训练设备
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu")

        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)#定义policy_net，即DQN网络
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)#定义target_net，即target网络
        self.target_net.load_state_dict(self.policy_net.state_dict())#初始化target_net的权重参数为policy_net的权重参数
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)#定义优化器
        self.memory = ReplayMemory(10000)#定义ReplayMemory，用于存储经验

    def select_action(self,state):  # 定义select_action函数，用于选择动作
        self.state=state

        sample = random.random()  # 生成一个随机数sample，用于判断是否使用随机动作
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)  # 计算epsilon的衰减值
        self.steps_done += 1  # 更新steps_done
        if sample > eps_threshold:  # 如果sample大于epsilon的衰减值，则使用随机动作
            with torch.no_grad():  # 不计算梯度
                # t.max(1)返回每行的最大值，第二列是最大值对应的索引，所以我们选择最大值对应的动作
                # 第二列是最大值对应的索引，所以我们选择最大值对应的动作
                # 找到了最大值，所以我们选择最大值对应的动作
                return self.policy_net(state).max(1).indices.view(1, 1)  # 返回policy_net的输出结果的最大值对应的动作
        else:  # 否则使用policy_net选择动作
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)  # 随机选择一个动作,返回一个tensor

    def optimize_model(self):  # 定义optimize_model函数，用于训练DQN网络
        if len(self.memory) < self.BATCH_SIZE:  # 如果memory中没有足够的经验，则不进行训练
            return
        transitions = self.memory.sample(self.BATCH_SIZE)  # 从memory中随机采样BATCH_SIZE个Transition对象,transitions是一个列表

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)  # 定义一个非终止状态的mask
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])  # 定义非终止状态的next_state
        state_batch = torch.cat(batch.state)#数据类型是tensor，维度是(128,8)
        action_batch = torch.cat(batch.action)#数据类型是tensor，维度是(128,1)
        reward_batch = torch.cat(batch.reward)#数据类型是tensor，维度是(128,1)

        #计算Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)#gather函数用于从矩阵中选取元素，返回一个新的矩阵，其第i行第j列元素为原矩阵第i行第j列元素

        #计算V(s_{t+1})
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        #next_state_values维度是(128,)是1维


        with torch.no_grad():#不计算梯度
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        #expected_state_action_values维度是(128,1)  shape=(128,1)


        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)#梯度裁剪,参数：网络参数，裁剪值
        self.optimizer.step()





