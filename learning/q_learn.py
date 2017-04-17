import numpy
import random
from collections import deque, namedtuple
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
        q_value = self.fc2(x)
        return q_value


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque()
        self.capacity = capacity

    def push(self, transition):
        if len(self.memory) > self.capacity:
            self.memory.popleft()
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QLearn(object):
    def __init__(self, args, num_actions, model, optimizer=None, exploration=False):
        self.gamma = args.gamma
        self.num_actions = num_actions
        self.batch_size = args.batch_size
        self.epsilon = 1.0
        self.epsilon_decay = (1.0 - 0.05) / (2 / 3 * args.num_games)
        self.model = model
        if not optimizer:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-2)
        else:
            self.optimizer = optimizer
        self.memory = ReplayMemory(args.replay_memory_size)
        self.exploration = exploration

    def select_action(self, state):
        if self.exploration:
            # Use epsilon greedy to select action
            if random.random() < self.epsilon:
                action = random.randrange(self.num_actions)
            else:
                # q_values = self.model(state.view(1, -1))
                if type(state) is numpy.ndarray:
                    state = cuda.from_numpy(state).unsqueeze(0)
                else:
                    state = cuda.to_tensor(state).unsqueeze(0)
                q_values = self.model(cuda.variable(state))
                action = q_values.data.max(1)[1][0, 0]
        else:
            # q_values = self.model(state.view(1, -1))
            if type(state) is numpy.ndarray:
                state = cuda.from_numpy(state).unsqueeze(0)
            else:
                state = cuda.to_tensor(state).unsqueeze(0)
            q_values = self.model(cuda.variable(state))
            action = q_values.data.max(1)[1][0, 0]
        return action

    def optimize(self):
        # Sample mini-batch transitions from memory
        batch = self.memory.sample(self.batch_size)
        state_batch = numpy.vstack([trans[0] for trans in batch])
        action_batch = numpy.vstack([trans[1] for trans in batch])
        reward_batch = numpy.vstack([trans[2] for trans in batch])
        next_state_batch = numpy.vstack([trans[3] for trans in batch])

        state_batch = cuda.variable(cuda.to_tensor(state_batch))
        action_batch = cuda.variable(cuda.to_tensor(action_batch))
        reward_batch = cuda.variable(cuda.to_tensor(reward_batch))
        next_state_batch = cuda.variable(cuda.to_tensor(next_state_batch), volatile=True)

        # Forward + Backward + Opimize
        self.optimizer.zero_grad()
        # q_values = self.model(cuda.variable(cuda.to_tensor(state_batch)))
        q_values = self.model(state_batch)
        # next_q_values = self.model(cuda.variable(cuda.to_tensor(next_state_batch), volatile=True))
        next_q_values = self.model(next_state_batch)
        next_q_values.volatile = False

        # td_target = cuda.variable(cuda.to_tensor(reward_batch)) + self.gamma * next_q_values.max(1)[0]
        td_target = reward_batch + self.gamma * next_q_values.max(1)[0]
        # loss = nn.MSELoss(q_values.gather(1, cuda.variable(cuda.to_tensor(action_batch)).long().view(-1, 1)), td_target)
        # loss = nn.MSELoss()(q_values.gather(1, action_batch.long().view(-1, 1)), td_target)
        loss = F.smooth_l1_loss(q_values.gather(1, action_batch.long().view(-1, 1)), td_target)
        loss.backward()
        self.optimizer.step()
        self.epsilon -= self.epsilon_decay
