
import DQN
import gym
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
        # 如果有GPU，加载GPU模型
        dqn.policy_net.load_state_dict(torch.load(model_path))
    else:
        # 如果没有GPU，加载CPU模型
        dqn.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))




def main():

    #超参数
    BATCH_SIZE = 128  # 定义batch_size为128，即从ReplayMemory中随机采样128个Transition对象，用于训练DQN网络
    GAMMA = 0.99  # 定义折扣因子为0.99
    EPS_START = 0.9  # 定义epsilon的起始值，即随机动作的概率
    EPS_END = 0.05  # 定义epsilon的终止值，即随机动作的概率
    EPS_DECAY = 1000  # 定义epsilon的衰减速度，即每经过1000次训练，epsilon减小0.001
    TAU = 0.005  # 定义target网络的更新率，即target_net = policy_net * TAU + target_net * (1-TAU)，即target网络的权重更新慢于policy网络的权重更新
    LR = 1e-4
    # env = gym.make('LunarLander-v2')  # 创建环境
    env_name = "LunarLander-v2"  # 游戏名字
    env = gym.make(env_name)#, render_mode="human"


    dqn = DQN.Agent(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, env)


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


    # 加载模型
    # 加载一个名为 DQN_LunarLander-v2_second_round.pth 的模型
    import os
    model_path = './DQN_{}_second_round.pth'.format(env_name)
    if os.path.exists(model_path):
        load_model(dqn, model_path)
        print("模型加载成功！")


    for i_episode in range(4000,num_episodes):  # 训练的轮数

        # 初始化环境
        state, info = env.reset()  # 初始化环境，获取初始状态
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)#将初始状态转换为tensor格式




        for t in range(max_timesteps):  # 每局游戏的最大步数
            timestep += 1
            action = dqn.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())

            total_reward+=reward  # 记录总奖励

            reward = torch.tensor([reward], device=device)  # 将reward从float格式转换为tensor格式
            done = terminated or truncated

            # 可视化
            # env.render()

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # 存储经验
            dqn.memory.push(state, action, next_state, reward)

            # 移动到下一个状态
            state = next_state

            dqn.optimize_model()  # 训练DQN网络

            target_net_state_dict = dqn.target_net.state_dict()  # 获取target_net的权重参数
            policy_net_state_dict = dqn.policy_net.state_dict()  # 获取policy_net的权重参数
            for key in policy_net_state_dict:  # 更新target_net的权重参数
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            dqn.target_net.load_state_dict(target_net_state_dict)

            if done:
                break


        # 记录总步数
        total_length+=t

        # 每一轮结束后：
        if i_episode > 6000:
            print("########## Solved! ##########")
            # 保存模型
            torch.save(dqn.policy_net.state_dict(), './DQN_{}_third_round.pth'.format('LunarLander-v2'))
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