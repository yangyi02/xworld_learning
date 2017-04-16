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


class Net(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
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
        state = cuda.to_tensor(state).unsqueeze(0)
        probs = self.model(cuda.variable(state))
        action = probs.multinomial()
        self.saved_actions.append(action)
        return action.data

    def optimize(self):
        R = 0
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = cuda.to_tensor(rewards)
        if rewards.size()[0] > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + numpy.finfo(numpy.float32).eps)
        for action, r in zip(self.saved_actions, rewards):
            action.reinforce(r)
        self.optimizer.zero_grad()
        autograd.backward(self.saved_actions, [None for _ in self.saved_actions])
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
