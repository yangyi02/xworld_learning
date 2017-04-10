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


class Policy(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_actions):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores), state_values


SavedAction = namedtuple('SavedAction', ['action', 'value'])
class AsyncActorCritic(object):
    def __init__(self, gamma, model, shared_model, optimizer=None):
        self.gamma = gamma
        self.model = model
        self.shared_model = shared_model
        if not optimizer:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-2)
        else:
            self.optimizer = optimizer

        self.saved_actions = []
        self.rewards = []

    def select_action(self, state, exploration=None):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.model(cuda.variable(state))
        action = probs.multinomial()
        self.saved_actions.append(SavedAction(action, state_value))
        return action.data

    def optimize(self):
        R = 0
        saved_actions = self.saved_actions
        value_loss = 0
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards).cuda() if cuda.use_cuda() else torch.Tensor(rewards)
        if rewards.size()[0] > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + numpy.finfo(numpy.float32).eps)
        for (action, value), r in zip(saved_actions, rewards):
            action.reinforce(r - value.data[0,0])
            value_loss += F.smooth_l1_loss(value, cuda.variable(torch.Tensor([r])))
        self.optimizer.zero_grad()
        final_nodes = [value_loss] + list(map(lambda p: p.action, saved_actions))
        # gradients = [torch.ones(1)] + [None] * len(saved_actions)
        gradients = [torch.ones(1).cuda() if cuda.use_cuda() else torch.ones(1)] + [None] * len(saved_actions)
        autograd.backward(final_nodes, gradients)
        self.ensure_shared_grads()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]

    def ensure_shared_grads(self):
        for param, shared_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
