import gym
from agent import Agent
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 创建环境
env = gym.make("LunarLander-v2")

# 配置参数
kwargs = {
    "epoch": 1000,           # 训练轮数
    "max_step": 1000,        # 每轮最大步数
    "batch_size": 64,        # 批次大小
    "lr_start": 1e-3,        # 初始学习率
    'lr_end': 1e-6,          # 最终学习率
    'lr_decay': 0.995,       # 学习率衰减
    "gamma": 0.99,           # 折扣因子
    "memory_size": 10000,    # 经验池大小
    "eps_clip": 0.2,         # PPO裁剪参数
    "Lambda": 0.95,          # GAE参数
    "K_epochs": 4,           # 策略更新次数
    
    'state_dim': env.observation_space.shape[0],
    'action_dim': env.action_space.n,
    'model_save_dir': './model'
}

def plot_training_curve(rewards, save_path='./training_curve.png'):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 6))
    
    # 原始奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, color='blue', linewidth=0.8)
    # 移动平均
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最近100轮的奖励分布
    plt.subplot(1, 2, 2)
    recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
    plt.hist(recent_rewards, bins=20, alpha=0.7, color='green')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title(f'Recent {len(recent_rewards)} Episodes Reward Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {save_path}")

def train_ppo():
    print("开始PPO训练...")
    print(f"环境: LunarLander-v2")
    print(f"状态维度: {kwargs['state_dim']}, 动作维度: {kwargs['action_dim']}")
    
    # 检查设备
    device_count = torch.cuda.device_count()
    if device_count > 0:
        device_name = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print(f"使用设备: {device_name[0]}")
    else:
        print("使用设备: CPU")
    
    # 创建智能体
    agent = Agent(**kwargs)
    
    # 训练记录
    rewards_history = []
    best_reward = float('-inf')
    consecutive_success = 0
    
    # 创建保存目录
    os.makedirs('./logs', exist_ok=True)
    
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(kwargs["epoch"]):
        state, _ = env.reset()
        game_reward = 0
        step_count = 0
        
        for step in range(kwargs["max_step"]):
            action, value = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            game_reward += reward
            step_count += 1
            done = done or truncated or (step >= kwargs["max_step"]-1)
            agent.memory.push(state, action, reward, value, done)
            state = next_state

            if done:
                break
        
        # 只有当经验池有足够数据时才进行更新
        if len(agent.memory) >= agent.batch_size:
            agent.update()
        
        # 每个epoch结束后清空经验池
        agent.memory.clear_all()
        
        # 记录奖励
        rewards_history.append(game_reward)
        
        # 更新最佳奖励
        if game_reward > best_reward:
            best_reward = game_reward
            consecutive_success = 0
        
        # 检查是否连续成功
        if game_reward > 250:  # LunarLander-v2的成功阈值
            consecutive_success += 1
        else:
            consecutive_success = 0
        
        # 打印训练信息
        if epoch % 50 == 0 or epoch < 10:
            current_time = time.strftime("%H:%M:%S", time.localtime())
            avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
            print(f"Epoch {epoch:4d} | 步数: {step_count:3d} | 奖励: {game_reward:7.2f} | 平均奖励(50): {avg_reward:7.2f} | 最佳: {best_reward:7.2f} | 时间: {current_time}")
        
        # 保存模型
        if game_reward > best_reward*1.02 or (consecutive_success >= 10 and epoch%10==0) or epoch == kwargs["epoch"]-1:
            agent.save_model(epoch)
            if consecutive_success >= 10:
                print(f"\n🎉 连续{consecutive_success}轮成功！训练完成！")
                break
            elif epoch == kwargs["epoch"]-1:
                print(f"达到最大训练轮数，训练完成！")
                break
        
        # 定期保存训练曲线
        if epoch % 100 == 0 and epoch > 0:
            plot_training_curve(rewards_history, f'./logs/training_curve_epoch_{epoch}.png')
    
    # 训练结束
    total_time = time.time() - start_time
    print(f"\n训练完成！")
    print(f"总训练时间: {total_time/3600:.2f} 小时")
    print(f"最佳奖励: {best_reward:.2f}")
    print(f"最后50轮平均奖励: {np.mean(rewards_history[-50:]):.2f}")
    
    # 保存最终训练曲线
    plot_training_curve(rewards_history, './logs/final_training_curve.png')
    
    # 保存训练数据
    np.save('./logs/rewards_history.npy', rewards_history)
    print("训练数据已保存到 ./logs/rewards_history.npy")
    
    env.close()
    return rewards_history

if __name__ == '__main__':
    try:
        rewards = train_ppo()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()