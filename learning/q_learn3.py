import argparse
import gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch Q learning example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


plt.ion()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN2(nn.Module):
    def __init__(self):
        super(DQN2, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.score = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.score(x)
        return x

model = DQN2()
model = model.cuda()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), lr=1e-2)


EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 20000
steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    # print("random exploration rate: %.2f" % eps_threshold)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state.cuda())).data.max(1)[1].cpu()
    else:
        return torch.LongTensor([[random.randrange(2)]])

last_sync = 0
def optimize_model():
    if len(memory) < args.batch_size:
        return 0.0
    transitions = memory.sample(args.batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state))).cuda()
    # We don't want to backprop through the expected action values and volatile will save us
    # on temporarily changing the model parameters' requires_grad to False!
    non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)
    non_final_next_states = Variable(non_final_next_states_t.cuda(), volatile=True)
    # non_final_next_states = Variable(non_final_next_states_t)
    state_batch = Variable(torch.cat(batch.state).cuda())
    action_batch = Variable(torch.cat(batch.action).cuda())
    reward_batch = Variable(torch.cat(batch.reward).cuda())

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(args.batch_size)).cuda()
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's clear it.
    # After this, we'll just end up with a Variable that has requires_grad=False
    # next_state_values.volatile = False
    non_final_next_states.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    print("loss: %.2f" % loss.data[0])

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.data[0]


running_reward = 10
for i_episode in count(1):
    # Initialize the environment and state
    state = torch.Tensor([env.reset()])
    for t in range(10000):  # Don't infinite loop while learning
        action = select_action(state)
        next_state, reward, done, _ = env.step(action[0, 0])

        action = torch.Tensor([action])
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        optimize_model()
        if done:
            break

    # Perform one step of the optimization (on the target network)
    running_reward = running_reward * 0.99 + t * 0.01
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, t, running_reward))
    if running_reward > 200:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break
