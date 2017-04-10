import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
# from torch.autograd import Variable
from . import cuda
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class Policy(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_actions):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.affine2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


class Reinforce(object):
    def __init__(self, gamma, model, optimizer=None):
        self.gamma = gamma
        self.model = model
        if not optimizer:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-2)
        else:
            self.optimizer = optimizer

        self.saved_actions = []
        self.rewards = []

    def select_action(self, state, exploration=None):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(cuda.variable(state))
        action = probs.multinomial()
        self.saved_actions.append(action)
        return action.data

    def optimize(self):
        R = 0
        saved_actions = self.saved_actions
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        # logging.info(rewards)
        if rewards.size()[0] > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + numpy.finfo(numpy.float32).eps)
        # logging.info(rewards)
        for action, r in zip(self.saved_actions, rewards):
            action.reinforce(r)
        self.optimizer.zero_grad()
        autograd.backward(self.saved_actions, [None for _ in self.saved_actions])
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
