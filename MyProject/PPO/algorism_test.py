import gym
from agent import Agent
import torch

# 创建环境
env = gym.make("LunarLander-v2")

# 配置参数
kwargs = {
    "epoch": 5,  # 只测试5轮
    "max_step": 100,  # 每轮最大100步
    "batch_size": 32,
    "lr_start": 1e-3,
    'lr_end': 1e-6,
    'lr_decay': 0.995,
    "gamma": 0.99,
    "memory_size": 1000,
    "eps_clip": 0.2,
    "Lambda": 0.95,
    "K_epochs": 2,  # 减少更新次数
    
    'state_dim': env.observation_space.shape[0],
    'action_dim': env.action_space.n,
    'model_save_dir': './model'
}

def test_ppo():
    print("开始测试PPO算法...")
    
    # 检查设备
    device_count = torch.cuda.device_count()
    if device_count > 0:
        device_name = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print(f"使用设备: {device_name}")
    else:
        print("使用设备: CPU")
    
    # 创建智能体
    agent = Agent(**kwargs)
    print(f"智能体创建成功，状态维度: {kwargs['state_dim']}, 动作维度: {kwargs['action_dim']}")
    
    # 开始训练测试
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
        
        # 只有当经验池有足够数据时才更新
        if len(agent.memory) >= agent.batch_size:
            agent.update()
            print(f"轮次 {epoch}: 步数={step_count}, 奖励={game_reward:.2f}, 经验池大小={len(agent.memory)}")
        else:
            print(f"轮次 {epoch}: 步数={step_count}, 奖励={game_reward:.2f}, 经验池大小={len(agent.memory)} (跳过更新)")
    
    env.close()
    print("测试完成!")

if __name__ == '__main__':
    test_ppo()