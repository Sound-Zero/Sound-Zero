
import math
import random
from collections import namedtuple, deque


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):

        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):#定义DQN网络结构
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)


    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)


class Agent:

    def __init__(self,BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY,TAU,LR,env):

        self.env = env
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU = TAU
        self.LR = LR
        self.steps_done = 0






        self.state , self.info = self.env.reset()#初始化环境，获取初始状态
        self.n_actions = env.action_space.n  # 获取动作空间的数量
        self.n_observations = len(self.state)#获取状态的维度


        self.device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu")

        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def select_action(self,state):
        self.state=state

        sample = random.random()  # 生成一个随机数sample，用于判断是否使用随机动作
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)  # 计算epsilon的衰减值
        self.steps_done += 1  # 更新steps_done
        if sample > eps_threshold:  # 如果sample大于epsilon的衰减值，则使用随机动作
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:  # 如果memory中没有足够的经验，则不进行训练
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)  # 定义一个非终止状态的mask
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])  # 定义非终止状态的next_state
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)#gather函数用于从矩阵中选取元素，返回一个新的矩阵，其第i行第j列元素为原矩阵第i行第j列元素

        #计算V(s_{t+1})
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)



        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # 计算loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()






