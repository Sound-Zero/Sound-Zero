import torch.nn as nn
import torch.nn.functional


class MLP(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, n_actions)  # 输出层

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

from collections import deque, namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):#定义一个ReplayMemory类，用于存储经验

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)#

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):#随机采样batch_size个Transition对象
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



import torch
import torch.optim as optim
import math
import numpy as np
class Agent:


    def __init__(self,BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY,TAU,LR,env):
        #env = gym.make("LunarLander-v2")#定义环境
        self.env = env
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START#定义epsilon的起始值
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.LR = LR
        self.steps_done = 0

        self.state , self.info = self.env.reset()#初始化环境，获取初始状态
        self.n_actions = env.action_space.n  # 获取动作空间的数量
        self.n_observations = len(self.state)#获取状态的维度



        self.device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu")

        self.policy_net = MLP(self.n_observations,self.n_actions,hidden_dim=256).to(self.device)
        self.target_net = MLP(self.n_observations,self.n_actions,hidden_dim=256).to(self.device)

         # 复制参数到目标网络
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)  # 优化器

        self.memory = ReplayMemory(10000)#定义ReplayMemory，用于存储经验

        self.steps_done = 0#定义steps_done，用于计算epsilon的衰减值



    def select_action(self, state):  # 定义select_action函数，用于选择动作
        self.state = state
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)  # 计算epsilon的衰减值
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device,
                                dtype=torch.long)  # 随机选择一个动作,返回一个tensor


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)

        state_batch=transitions[0].state
        action_batch=transitions[0].action
        next_state_batch=transitions[0].next_state
        reward_batch=transitions[0].reward
        done_batch=transitions[0].done
        for i in transitions[1:]:
            state_batch=torch.cat((state_batch,i.state),0)
            action_batch=torch.cat((action_batch,i.action),0)
            next_state_batch=torch.cat((next_state_batch,i.next_state),0)
            reward_batch=torch.cat((reward_batch,i.reward),0)
        state_batch.to(dtype=torch.float32)
        next_state_batch.to(dtype=torch.float32)
        reward_batch.to(dtype=torch.float32)
        reward_batch = reward_batch.unsqueeze(1)




        # 计算实际Q值和期望Q值
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # 实际的Q值
        next_q_value_batch = self.policy_net(next_state_batch) # 下一个状态对应的实际策略网络Q值



        # 找到下一个状态对应的策略网络Q值最大的动作
        #argmax_next_q_value_batch = next_q_value_batch.max(1)[1].unsqueeze(1)
        # 将策略网络Q值最大的动作对应的目标网络Q值作为期望的Q值
        #next_target_q_value_batch = self.target_net(next_state_batch).gather(dim=1, index=argmax_next_q_value_batch) # 下一个状态对应的目标网络Q值




        # 将策略网络Q值最大的动作对应的目标网络Q值作为期望的Q值
        next_target_value_batch = self.target_net(next_state_batch)  # 下一个状态对应的目标网络Q值
        next_target_q_value_batch = next_target_value_batch.gather(1, torch.max(next_q_value_batch, 1)[1].unsqueeze(1))
        expected_q_value_batch = reward_batch + self.GAMMA * next_target_q_value_batch* (1-done_batch) # 期望的Q值
        # 计算损失
        q_value_batch = q_value_batch.float()
        expected_q_value_batch = expected_q_value_batch.float()
        loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)
        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 20)
        self.optimizer.step()#更新模型参数
        # 每隔一定步数更新目标网络
        if self.steps_done % 5 == 0:  # 每5步更新一次目标网络
            target_net_state_dict = self.target_net.state_dict()  # 获取target_net的权重参数
            policy_net_state_dict = self.policy_net.state_dict()  # 获取policy_net的权重参数
            for key in policy_net_state_dict:  # 更新target_net的权重参数
                target_net_state_dict[key] = policy_net_state_dict[key] * 0.005 + target_net_state_dict[key] * (1 - 0.005)
            self.target_net.load_state_dict(target_net_state_dict)
        self.steps_done += 1











import torch
import numpy as np






#########################################
#使用maptlotlib绘制reward曲线
import matplotlib.pyplot as plt
Episodic_count = [] # 记录每个episode的步数 作为x轴
Episodic_reward = [] # 记录每个episode的奖励 作为y轴

def plot_reward_curve(reward_list):
    plt.plot(Episodic_count, Episodic_reward)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Curve')
    plt.show()

def load_model(dqn, model_path):
    if torch.cuda.is_available():
        dqn.policy_net.load_state_dict(torch.load(model_path))
    else:
        dqn.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


import gym
import torch
def main():

    #超参数
    BATCH_SIZE = 128  # 定义batch_size为128，即从ReplayMemory中随机采样128个Transition对象，用于训练DQN网络
    GAMMA = 0.99  # 定义折扣因子为0.99
    EPS_START = 0.9  # 定义epsilon的起始值，即随机动作的概率
    EPS_END = 0.05  # 定义epsilon的终止值，即随机动作的概率
    EPS_DECAY = 1000  # 定义epsilon的衰减速度，即每经过1000次训练，epsilon减小0.001
    TAU = 0.005  # 定义target网络的更新率，即target_net = policy_net * TAU + target_net * (1-TAU)，即target网络的权重更新慢于policy网络的权重更新
    LR = 0.002  # 定义学习率
    env_name = "LunarLander-v2"  # 游戏名字
    env = gym.make(env_name)#, render_mode="human"


    ddqn = Agent(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, env)


    #设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() or torch.backends.mps.is_available():  # 如果有GPU，则使用GPU
        num_episodes = 10000  # 最大训练的轮数
    else:
        num_episodes = 50


    #记录
    total_reward = 0 # 记录每个episode的总奖励
    max_timesteps = 200  # 定义每局游戏的最大步数
    total_length=0  # 记录每个episode的总步数
    timestep = 0  # 每一个episode的步数


    # # 加载模型
    # # 加载一个名为 DDQN_LunarLander-v2_second_round.pth 的模型
    # import os
    # model_path = './DDQN_{}.pth'.format(env_name)
    # if os.path.exists(model_path):
    #     load_model(, model_path)
    #     print("模型加载成功！")


    for i_episode in range(num_episodes):  # 训练的轮数

        # 初始化环境
        state, info = env.reset()  # 初始化环境，获取初始状态
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)#将初始状态转换为tensor格式

        for t in range(max_timesteps):  # 每局游戏的最大步数
            timestep += 1
            action = ddqn.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            total_reward+=reward  # 记录总奖励

            reward = torch.tensor([reward], device=device)  # 将reward从float格式转换为tensor格式
            done = terminated or truncated  #terminated为True表示游戏结束，truncated为True表示游戏截断

            # 可视化
            # env.render()

            if terminated:
                break
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            # 存储经验

            ddqn.memory.push(state, action, next_state, reward, done)

            # 移动到下一个状态
            state = next_state

            ddqn.optimize_model()  # 训练DDQN网络
       # 记录总步数
        total_length+=t

        # 每一轮结束后：
        if i_episode > 2000:
            print("########## Solved! ##########")
            # 保存模型
            torch.save(ddqn.policy_net.state_dict(), './DDQN_{}_second_round.pth'.format('LunarLander-v2'))
            break

        # 显示训练结果
        if i_episode % 20 == 0:  # 每隔20轮显示一次训练过程
            avg_length = int(total_length / 20)
            running_reward = int(total_reward / 20)
            Episodic_count.append(i_episode)  # X轴数据
            Episodic_reward.append(running_reward)  # Y轴数据

            print('Episode {} \t avg length: {} \t average_reward: {}'.format(i_episode, avg_length, running_reward))
            # 清零
            total_reward = 0
            total_length = 0

    print('Complete')

    plot_reward_curve(Episodic_reward)  # 绘制reward曲线


if __name__ == '__main__':
    main()


