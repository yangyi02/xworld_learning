import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class Policy(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_actions):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.affine2 = nn.Linear(hidden_size, num_actions)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


class Reinforce(object):
    def __init__(self, gamma, model):
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

    def select_action(self, state, exploration=None):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(Variable(state))
        action = probs.multinomial()
        self.model.saved_actions.append(action)
        return action.data

    def optimize(self, batch):
        R = 0
        saved_actions = self.model.saved_actions
        rewards = []
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + numpy.finfo(numpy.float32).eps)
        for action, r in zip(self.model.saved_actions, rewards):
            action.reinforce(r)
        self.optimizer.zero_grad()
        autograd.backward(self.model.saved_actions, [None for _ in self.model.saved_actions])
        self.optimizer.step()
        del self.model.rewards[:]
        del self.model.saved_actions[:]


def main():
    logging.info('testing reinforce functions')
    model = Policy(4, 128, 2)
    reinforce_model = Reinforce(0.99, model)
    state = numpy.random.randn(4)
    action = reinforce_model.select_action(state)
    reinforce_model.model.rewards.append(1.0)
    reinforce_model.optimize()
    logging.info('testing reinforce functions done')

if __name__ == '__main__':
    main()
