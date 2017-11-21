import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class NetworkOld(nn.Module):
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


class Network(nn.Module):
    def __init__(self, height, width, channel, hidden_size, dict_length, num_actions):
        super().__init__()
        self.embed = nn.Embedding(dict_length, hidden_size)
        self.conv1 = nn.Conv2d(channel, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_obsta = nn.Conv2d(hidden_size, 1, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(height * width * 2, hidden_size)
        self.fc_a = nn.Linear(hidden_size, num_actions)
        self.fc_v = nn.Linear(hidden_size, 1)

    def forward(self, image, cmd_id):
        language_kernel = self.embed(cmd_id.long().view(1, -1))
        language_kernel = language_kernel.view(-1, language_kernel.size()[-1], 1, 1)
        image_feature = self.conv2(F.relu(self.conv1(image)))
        attention_map = F.conv2d(image_feature, language_kernel)
        obstacle_map = self.conv_obsta(image_feature)
        self.visualize_map(attention_map, obstacle_map)
        x = torch.cat([attention_map, obstacle_map], 1)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        action_scores = self.fc_a(x)
        state_values = self.fc_v(x)
        return F.softmax(action_scores), state_values

    @staticmethod
    def visualize_map(attention_map, obstacle_map):
        a = attention_map.squeeze().cpu().data.numpy()
        b = obstacle_map.squeeze().cpu().data.numpy()
        plt.ion()
        plt.figure(2)
        plt.subplot(1, 2, 1)
        plt.pcolor(a, cmap='jet')
        plt.axis('equal')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.pcolor(b, cmap='jet')
        plt.axis('equal')
        plt.axis('off')
        plt.show()
