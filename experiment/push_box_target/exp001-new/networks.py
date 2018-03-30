import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp


class Network(nn.Module):
    def __init__(self, height, width, channel, hidden_size1, hidden_size2, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, hidden_size1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size1, hidden_size1, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_size1, hidden_size1, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(height * width * hidden_size1, hidden_size2)
        self.fc_a = nn.Linear(hidden_size2, num_actions)
        self.fc_v = nn.Linear(hidden_size2, 1)

    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        action_scores = self.fc_a(x)
        state_values = self.fc_v(x)
        return F.softmax(action_scores, dim=1), state_values

