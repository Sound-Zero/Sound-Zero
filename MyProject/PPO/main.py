import gym
from agent import Agent
import time
import torch
env = gym.make("LunarLander-v2")
kwargs={
    "epoch":3000,
    "max_step":1000,
    "batch_size":64,
    "lr_start":1e-3,
    'lr_end':1e-6,
    'lr_decay':0.995,
    "gamma":0.99,
    "memory_size":10000,
    "eps_clip":0.2,
    "Lambda":0.95,  # 计算优势函数的折扣因子
    "K_epochs":4,   # 更新策略网络的次数
    
    'state_dim':env.observation_space.shape[0],
    'action_dim':env.action_space.n,

    'model_save_dir':'./model'
}
def main():
    agent = Agent(**kwargs)
    for epoch in range(kwargs["epoch"]):
        state, _ = env.reset()  # gym环境reset返回(observation, info)
        game_reward = 0
        for step in range(kwargs["max_step"]):
            action, value = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            game_reward += reward
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
        
        if epoch % 20 == 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"Epoch: {epoch}, Game Reward: {game_reward:.2f}, Time: {current_time}")
        if game_reward > 250 or epoch == kwargs["epoch"]-1:
            agent.save_model(epoch)
    env.close()

if __name__ == '__main__':
    device_count = torch.cuda.device_count()
    if device_count > 0:
        device_name = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print(f"Using device: {device_name}")
    else:
        print("Using device: CPU")
    main()