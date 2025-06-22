import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MLP(nn.Module):
    def __init__(self,layer_sizes:List[int]):
        super(MLP, self).__init__()
        # layer_sizes 是一个列表，定义了每一层的神经元数量
        # 例如：layer_sizes = [input_dim, hidden_dim1, hidden_dim2, output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 最后一层不加激活函数
                x = F.relu(x)
        return x


class ConcatNet(nn.Module):
    def __init__(self, input_dim, output_dim,output_prob=True):
        super(ConcatNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(64, 64)
        self.fc_predict = nn.Linear(512, output_dim)

        self.output_prob = output_prob#是否输出概率，如果否则经过fc_predict后直接输出值，不经过激活函数
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x1,x=torch.split(x, 256, dim=-1)
        x = F.relu(self.fc2(x))
        x2,x=torch.split(x, 128, dim=-1)
        x = F.relu(self.fc3(x))
        x3,x=torch.split(x, 64, dim=-1)
        x = F.relu(self.fc4(x))


        x = torch.cat([x1,x2,x3,x], dim=-1)
        x = self.fc_predict(x)
        if self.output_prob:
            prob=F.softmax(x, dim=-1)
            return prob
        else:
            return x

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.concat_net = ConcatNet(input_dim, output_dim)

    def forward(self, x):
        prob=self.concat_net(x)
        return prob
    
    def select_action(self, x):
        with torch.no_grad():
            prob = self.forward(x)
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()#按概率采样得出下标
        return action
    
    def get_log_prob(self, x, action):
        prob = self.forward(x)
        dist = torch.distributions.Categorical(prob)
        log_prob = dist.log_prob(action)
        return log_prob

class Critic(nn.Module):#代表状态函数V(state_t)
    def __init__(self, input_dim, output_dim=1):    #默认输出一个value
        super(Critic, self).__init__()
        self.concat_net = ConcatNet(input_dim, output_dim=1,output_prob=False)

    def forward(self, x):
        value = self.concat_net(x)
        return value

