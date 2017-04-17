import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp


class Network(nn.Module):
    def __init__(self, height, width, channel, hidden_size1, hidden_size2, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, hidden_size1, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(height * width * (hidden_size1 + 1), hidden_size2)
        self.fc_a = nn.Linear(hidden_size2, num_actions)
        self.fc_v = nn.Linear(hidden_size2, 1)

    def forward(self, image, cmd_id):
        if image.size()[0] == 1:
            y = image[:, cmd_id.long().data[0, 0], :, :].unsqueeze(0)
        else:
            print('pytorch not working for batch indexing right now')
        x = self.conv1(image)
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        action_scores = self.fc_a(x)
        state_values = self.fc_v(x)
        return F.softmax(action_scores), state_values

