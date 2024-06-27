import gym
import torch
from PPO import Memory, PPO
import os
############## 超参数 ##############
env_name = "LunarLander-v2"  # 游戏名字
env = gym.make(env_name)#, render_mode="human"
state_dim = 8  # 状态维度
action_dim = 4  # 行动维度
render = False  # 可视化
solved_reward = 230  # 停止循环条件 (奖励 > 230)
log_interval = 20  # 每20次迭代输出一次log
max_episodes = 50000  # 最大迭代次数
max_timesteps = 300  # 最大单次游戏步数
n_latent_var = 64  # 全连接隐层维度
update_timestep = 2000  # 每2000步policy更新一次
lr = 0.002  # 学习率
betas = (0.9, 0.999)  # betas
gamma = 0.99  # gamma
K_epochs = 4  # policy迭代更新次数
eps_clip = 0.2  # PPO 限幅

###############绘图函数##############################
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


def load_model(ppo, model_path):
    if torch.cuda.is_available():
        # 如果有GPU，加载GPU模型
        ppo.policy.load_state_dict(torch.load(model_path))
    else:
        # 如果没有GPU，加载CPU模型
        ppo.policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))



####################主函数##########################
def main():
    # 实例化
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    # 加载模型
    # 加载一个名为 PPO_LunarLander-v2_second_round.pth 的模型
    model_path = './PPO_{}_second_round.pth'.format(env_name)
    if os.path.exists(model_path):
        load_model(ppo, model_path)
        print("模型加载成功！")
    #存放
    total_reward = 0  #每一个episode的总奖励
    total_length = 0  #每一个episode的总步数
    timestep = 0      #每一个episode的步数

    # 训练
    for i_episode in range(4000, max_episodes + 1):

        # 环境初始化
        state, _ = env.reset()  # 初始化（重新玩）

        # 迭代
        for t in range(max_timesteps):
            timestep += 1

            # 用旧policy得到行动
            action = ppo.policy_old.act(state, memory)

            # 行动
            state, reward, terminated, truncated, _ = env.step(action)  # 得到（新的状态，奖励，是否终止，是否截断，额外的调试信息）
            done = terminated or truncated

            # 更新memory(奖励/游戏是否结束)
            memory.rewards.append(reward)#添加一个奖励
            memory.is_terminals.append(done)#添加当前状态是否让游戏结束

            # 更新梯度
            if timestep % update_timestep == 0:
                ppo.update(memory)

                # memory清零
                memory.clear_memory()

                # 累计步数清零
                timestep = 0

            # 累加
            total_reward += reward

            # 可视化
            if render:
                env.render()

            # 如果游戏结束, 退出
            if done:
                break

        # 游戏步长
        total_length += t

        # 如果达到要求(6000轮), 退出循环,第三轮训练从4000到6000
        #if total_reward >= (log_interval * solved_reward):
        if i_episode >= 6000:
            print("########## Solved! ##########")

            # 保存模型
            torch.save(ppo.policy.state_dict(), './PPO_{}_third_round.pt'.format(env_name))


            # 绘制reward曲线
            plot_reward_curve(Episodic_reward)

            # 退出循环
            break

        # 输出log, 每20次迭代
        if i_episode % log_interval == 0:
            # 求20次迭代平均时长/收益
            avg_length = int(total_length / log_interval)
            running_reward = int(total_reward / log_interval)

            #获取绘图数据
            Episodic_count.append(i_episode)#X轴数据
            Episodic_reward.append(running_reward)#Y轴数据
            # 调试输出
            print('Episode {} \t avg length: {} \t average_reward: {}'.format(i_episode, avg_length, running_reward))

            # 清零
            total_reward = 0
            total_length = 0

if __name__ == '__main__':
    main()
