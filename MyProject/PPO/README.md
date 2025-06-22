# PPO (Proximal Policy Optimization) 算法实现

这是一个完整的PPO算法实现，用于解决强化学习问题。本实现使用PyTorch框架，并在LunarLander-v2环境中进行训练。

## 文件结构

- `main.py`: 主训练脚本
- `agent.py`: PPO智能体实现，包含经验回放池和核心算法
- `model.py`: 神经网络模型定义，包含Actor和Critic网络
- `requirements.txt`: 项目依赖

## 核心特性

### PPO算法特点
- **Clipped Surrogate Objective**: 使用裁剪的替代目标函数防止策略更新过大
- **GAE (Generalized Advantage Estimation)**: 使用广义优势估计减少方差
- **Actor-Critic架构**: 分离策略网络(Actor)和价值网络(Critic)
- **经验回放**: 使用经验池存储和重用经验数据

### 网络架构
- **ConcatNet**: 自定义的残差连接网络，提高特征学习能力
- **Actor网络**: 输出动作概率分布
- **Critic网络**: 估计状态价值函数

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python main.py
```

### 主要参数配置

```python
kwargs = {
    "epoch": 3000,          # 训练轮数
    "max_step": 1000,       # 每轮最大步数
    "batch_size": 64,       # 批次大小
    "lr_start": 1e-3,       # 初始学习率
    "lr_end": 1e-6,         # 最终学习率
    "lr_decay": 0.995,      # 学习率衰减
    "gamma": 0.99,          # 折扣因子
    "eps_clip": 0.2,        # PPO裁剪参数
    "Lambda": 0.95,         # GAE参数
    "K_epochs": 4,          # 策略更新次数
}
```

## 算法详解

### 1. PPO损失函数

```
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```

其中：
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` 是概率比率
- `A_t` 是优势函数
- `ε` 是裁剪参数

### 2. GAE优势函数

```
A_t^GAE = Σ(γλ)^l δ_{t+l}
```

其中：
- `δ_t = r_t + γV(s_{t+1}) - V(s_t)` 是TD误差
- `λ` 是GAE参数

### 3. 训练流程

1. 收集经验数据
2. 计算GAE优势函数
3. 更新策略网络(Actor)
4. 更新价值网络(Critic)
5. 清空经验池
6. 重复上述步骤

## 模型保存和加载

模型会在以下情况自动保存：
- 游戏奖励超过250分
- 训练结束时

保存路径：`./model/ppo_model_epoch_{epoch}.pth`

加载模型：
```python
agent.load_model('path/to/model.pth')
```

## 性能监控

训练过程中会每20轮输出一次训练信息：
- 当前轮数
- 游戏总奖励
- 当前时间

## 注意事项

1. 确保安装了正确版本的依赖包
2. 如果使用GPU，确保CUDA环境配置正确
3. 可以根据具体环境调整超参数
4. 建议在训练前先测试环境是否正常工作

## 扩展性

本实现具有良好的扩展性，可以轻松适配其他Gym环境：
1. 修改`main.py`中的环境名称
2. 调整`state_dim`和`action_dim`参数
3. 根据需要调整网络架构和超参数