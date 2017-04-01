import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as T

plt.ion()
env = gym.make('CartPole-v0')
env = env.unwrapped


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


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


resize = T.Compose([T.ToPILImage(), T.Scale(40, interpolation=Image.CUBIC), T.ToTensor()])

# This is based on the code from gym.
screen_width = 600
def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0) # MIDDLE OF CART


def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1)) # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width,None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)

env.reset()
plt.imshow(get_screen().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.show()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

model = DQN()
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(model.parameters())

model.type(dtype)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state.type(dtype), volatile=True)).data.max(1)[1].cpu()
    else:
        return torch.LongTensor([[random.randrange(2)]])


episode_durations = []


def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.Tensor(episode_durations)
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.show()
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.pause(0.0001)


last_sync = 0
USE_CUDA = torch.cuda.is_available()

def optimize_model():
    global last_sync
    global USE_CUDA
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    if USE_CUDA:
        non_final_mask = non_final_mask.cuda()
    # We don't want to backprop through the expected action values and volatile will save us
    # on temporarily changing the model parameters' requires_grad to False!
    non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)
    non_final_next_states = Variable(non_final_next_states_t, volatile=True).cuda()
    # non_final_next_states = Variable(non_final_next_states_t).cuda()
    state_batch = Variable(torch.cat(batch.state)).cuda()
    action_batch = Variable(torch.cat(batch.action)).cuda()
    reward_batch = Variable(torch.cat(batch.reward)).cuda()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    try:
        state_action_values = model(state_batch).gather(1, action_batch)
    except TypeError:
        print("Hello World")

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE)).cuda()
    try:
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    except TypeError:
        print("Hello World")
    # Now, we don't want to mess up the loss with a volatile flag, so let's clear it.
    # After this, we'll just end up with a Variable that has requires_grad=False
    # next_state_values.volatile = False
    non_final_next_states.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


for i_episode in count(1):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action[0, 0])
        reward = torch.Tensor([reward])

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
