import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from distributions import Categorical, DiagGaussian
from utils import AddBias


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action

    def evaluate_actions(self, inputs, actions):
        value, x = self(inputs)
        action_log_probs, dist_entropy = self.dist.evaluate_actions(x, actions)
        return value, action_log_probs, dist_entropy


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4, bias=False)
        self.ab1 = AddBias(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=False)
        self.ab2 = AddBias(64)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1, bias=False)
        self.ab3 = AddBias(32)

        self.linear1 = nn.Linear(32 * 7 * 7, 512, bias=False)
        self.ab_fc1 = AddBias(512)

        self.critic_linear = nn.Linear(512, 1, bias=False)
        self.ab_fc2 = AddBias(1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(512, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(512, num_outputs)
        else:
            raise NotImplementedError

        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if action_space.__class__.__name__ == "Box":
            self.dist.fc_mean.weight.data.mul_(0.01)

        self.train()

    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = self.ab1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.ab2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.ab3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = self.ab_fc1(x)
        x = F.relu(x)

        return self.ab_fc2(self.critic_linear(x)), x
