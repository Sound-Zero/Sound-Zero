import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        
        # 将所有网络层移动到GPU
        self.shared = nn.Sequential(
            nn.Linear(input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        ).to(device)
        
        # Actor (策略)网络
        self.policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Critic (价值)网络
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
    
    def forward(self, x):
        x = x.to(device)  # 将输入数据移动到GPU
        shared_out = self.shared(x)
        policy = self.policy(shared_out)
        value = self.value(shared_out)
        return policy, value

def worker(name, model, counter, params, optimizer):
    # 创建游戏环境实例
    env = gym.make(params.env_name)
    # 创建本地模型副本并加载共享模型的参数
    local_model = ActorCritic(params.input_shape, env.action_space.n)
    local_model.load_state_dict(model.state_dict())
    local_model.to(device)  # 将本地模型移动到GPU
    
    episode_count = 0
    rewards_window = []  # 用于存储最近10个episode的奖励
    
    while episode_count < params.max_episodes:
        # 重置环境，开始新的回合
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        while not done:
            # 将状态数据移动到GPU
            state_v = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy, value = local_model(state_v)
            # 将策略移回CPU进行采样
            action = policy.cpu().multinomial(1).item()
            
            states.append(state)
            actions.append(action)
            values.append(value)
            log_probs.append(torch.log(policy[0, action]))
            
            next_state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            state = next_state
            
            total_reward += reward
            
            if total_reward >= 100:
                print(f"Worker {name} reached target reward of 100!")
                # 保存模型时将其移动到CPU
                model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(model_cpu, f'ac_model_worker_{name}.pth')
                return
        
        # 保存最近10个episode的奖励
        rewards_window.append(total_reward)
        if len(rewards_window) > 10:
            rewards_window.pop(0)
        
        # 每10个episode打印一次平均奖励
        if episode_count % 10 == 0:
            avg_reward = sum(rewards_window) / len(rewards_window)
            print(f"Worker {name}, Episodes {episode_count}-{episode_count+9}, Average Reward: {avg_reward:.2f}")
        
        # 计算优势和目标价值
        R = torch.zeros(1, 1).to(device)  # 将 R 移动到 GPU
        if not done:
            state_v = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, R = local_model(state_v)
        
        R = R.detach()
        GAE = torch.zeros(1, 1).to(device)  # 将 GAE 移动到 GPU
        returns = []
        for step in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[step]
            advantage = R - values[step]
            GAE = params.gamma * params.tau * GAE + advantage
            returns.insert(0, GAE + values[step])
        
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # 计算损失
        log_probs = torch.stack(log_probs).squeeze()
        advantages = returns - torch.stack(values).squeeze().detach()
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(torch.stack(values).squeeze(), returns)
        entropy_loss = -(log_probs * torch.exp(log_probs)).mean()
        
        loss = actor_loss + params.value_loss_coef * critic_loss - params.entropy_coef * entropy_loss
        
        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), params.max_grad_norm)
        for local_param, shared_param in zip(local_model.parameters(), model.parameters()):
            shared_param._grad = local_param.grad
        optimizer.step()
        
        episode_count += 1
    
    # 保存模型时将其移动到CPU
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(model_cpu, f'ac_model_worker_{name}.pth')

# 修改参数类，添加A3C算法所需的超参数
class Params:
    def __init__(self):
        self.env_name = "CartPole-v1"
        self.input_shape = (4,)  # CartPole的观察空间
        self.max_episodes = 10000  # 最大训练回合数
        self.use_gpu = True  # 添加GPU控制参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 1.0  # GAE参数
        self.lr = 0.001  # 学习率
        self.entropy_coef = 0.01  # 熵正则化系数
        self.value_loss_coef = 0.5  # 价值损失系数
        self.max_grad_norm = 40  # 梯度裁剪阈值

if __name__ == "__main__":
    # 创建参数对象
    params = Params()
    
    # 环境参数设置
    env = gym.make(params.env_name)
    
    # 创建共享模型并移动到GPU
    input_shape = params.input_shape
    n_actions = env.action_space.n
    shared_model = ActorCritic(input_shape, n_actions)
    shared_model.to(device)
    shared_model.share_memory()
    
    # 创建优化器
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=params.lr)
    
    # 创建计数器
    counter = mp.Value('i', 0)
    
    # 创建多个工作进程
    processes = []
    for i in range(mp.cpu_count()):
        p = mp.Process(target=worker, args=(i, shared_model, counter, params, optimizer))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 保存最终模型时将其移动到CPU
    model_cpu = {k: v.cpu() for k, v in shared_model.state_dict().items()}
    torch.save(model_cpu, 'ac_model_final.pth')
