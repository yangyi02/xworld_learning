import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp


class Network(nn.Module):
    def __init__(self, height, width, channel, hidden_size, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, hidden_size, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, 4, kernel_size=3, stride=1, padding=1)
        self.attention = nn.Softmax2d()
        self.affine1 = nn.Linear(height * width * 4, hidden_size)
        self.affine2 = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.attention(self.conv2(x))
        x = F.relu(self.affine1(x.view(x.size(0), -1)))
        action_scores = self.affine2(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores), state_values

