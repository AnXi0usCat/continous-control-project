import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, seed=1, fc1_dim=400, fc2_dim=300):
        super(Actor, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(state_size, fc1_dim)),
            ('bn1', nn.BatchNorm1d(fc1_dim)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(fc1_dim, fc2_dim)),
            ('relu', nn.ReLU()),
            ('fc3', nn.Linear(fc2_dim, action_size)),
            ('tanh', nn.Tanh())])
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.net.fc1.weight.data.uniform_(*hidden_init(self.net.fc1))
        self.net.fc2.weight.data.uniform_(*hidden_init(self.net.fc2))
        self.net.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, x):
        """
        Actor model which converts State -> Action
        """
        return self.net(x)


class Critic(nn.Module):
    
    def __init__(self, state_size, action_size, seed=1, fc1_dim=400, fc2_dim=300): 
        super(Critic, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.input_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(state_size, fc1_dim)),
            ('bn1', nn.BatchNorm1d(fc1_dim)),
            ('relu', nn.ReLU())])
        )
        
        self.out_net = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(fc1_dim + action_size, fc2_dim)),
            ('relu', nn.ReLU()),
            ('fc3', nn.Linear(fc2_dim, 1))
        ])
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.input_net.fc1.weight.data.uniform_(*hidden_init(self.input_net.fc1))
        self.out_net.fc2.weight.data.uniform_(*hidden_init(self.out_net.fc2))
        self.out_net.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, action):
        """
        Critic model which maps (states, actions) -> Q-Value
        """
        out = self.input_net(x)
        return self.out_net(torch.cat((out, action), dim=1))
