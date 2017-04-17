import numpy
from collections import namedtuple
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
        self.fc_a = nn.Linear(hidden_size, num_actions)
        self.fc_v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_scores = self.fc_a(x)
        state_values = self.fc_v(x)
        return F.softmax(action_scores), state_values


SavedAction = namedtuple('SavedAction', ['action', 'value'])
class ActorCritic(object):
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
        if type(state) is numpy.ndarray:
            state = cuda.from_numpy(state).unsqueeze(0)
        else:
            state = cuda.to_tensor(state).unsqueeze(0)
        probs, state_value = self.model(cuda.variable(state))
        action = probs.multinomial()
        self.saved_actions.append(SavedAction(action, state_value))
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
        value_loss = 0
        for (action, value), r in zip(self.saved_actions, rewards):
            action.reinforce(r - value.data[0, 0])
            value_loss += F.smooth_l1_loss(value, cuda.variable(cuda.to_tensor([r])))
        self.optimizer.zero_grad()
        final_nodes = [value_loss] + list(map(lambda p: p.action, self.saved_actions))
        gradients = [cuda.to_tensor([1])] + [None] * len(self.saved_actions)
        autograd.backward(final_nodes, gradients)
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
