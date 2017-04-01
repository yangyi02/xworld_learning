import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import xworld_learning_args
import xworld_navi_easy_goal
from learning.reinforce import Reinforce
from learning.actor_critic import ActorCritic
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class Network(nn.Module):
    def __init__(self, height, width, channel, hidden_size, num_actions):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(channel, hidden_size, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, 2, kernel_size=3, stride=1, padding=1)
        self.attention = nn.Softmax2d()
        self.affine1 = nn.Linear(height * width * 2, hidden_size)
        self.affine2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.attention(self.conv2(x))
        x = F.relu(self.affine1(x.view(x.size(0), -1)))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


def main():
    args = xworld_learning_args.parser().parse_args()
    args.map_config = 'empty_ground.json'
    args.learning_rate = 0.001
    logging.info(args)
    xworld = xworld_navi_easy_goal.XWorldNaviEasyGoal(args)
    xworld.seed(args.seed)
    torch.manual_seed(args.seed)

    xworld.reset()
    (height, width, channel) = xworld.state.onehot_state.shape
    num_hidden = 128
    num_actions = xworld.agent.num_actions
    model = Network(height, width, channel, 128, num_actions)

    if args.opt_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.opt_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.opt_method == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)

    if args.method == 'reinforce':
        reinforce_model = Reinforce(args.gamma, model, optimizer)
    elif args.method == 'actor_critic':
        reinforce_model = ActorCritic(args.gamma, model, optimizer)
    elif args.method == 'q_learn':
        reinforce_model = QLearn(args.gamma, model, optimizer)

    if args.train:
        logging.info('training')

        cumulative_rewards = []
        for i_episode in range(args.num_games):
            state, teacher = xworld.reset()
            cumulative_reward = []
            discount = 1.0
            for t in range(args.max_episode_length):  # Don't infinite loop while learning
                state_input = state.onehot_state.swapaxes(0,2).swapaxes(1,2)
                action = reinforce_model.select_action(state_input)
                next_state, teacher, done = xworld.step(action[0, 0])
                reward = teacher.reward
                reinforce_model.rewards.append(reward)
                cumulative_reward.append(reward * discount)
                discount *= args.gamma
                if done:
                    break
            cumulative_rewards.append(numpy.sum(numpy.asarray(cumulative_reward)))
            reinforce_model.optimize()
            if i_episode % args.log_interval == 0:
                logging.info('Episode {}\taverage cumulative reward: {:.2f}'.format(
                    i_episode, numpy.mean(numpy.asarray(cumulative_rewards))))
                cumulative_rewards = []
            if i_episode % args.save_interval == 0:
                model_name = '%.5d' % (i_episode) + '.pth'
                logging.info('Episode {}\tsaving model: {}'.format(
                    i_episode, model_name))
                with open(os.path.join(args.save_dir, model_name), 'wb') as handle:
                    torch.save(model.state_dict(), handle)

        with open(os.path.join(args.save_dir, 'final.pth'), 'wb') as handle:
            torch.save(model.state_dict(), handle)
    else:
        logging.info('testing')
        model.load_state_dict(torch.load(args.init_model))

        cumulative_rewards = []
        for i_episode in range(args.num_games):
            state, teacher = xworld.reset()
            cumulative_reward = []
            discount = 1.0
            for t in range(args.max_episode_length):  # Don't infinite loop while learning
                state_input = state.onehot_state.swapaxes(0,2).swapaxes(1,2)
                action = reinforce_model.select_action(state_input)
                next_state, teacher, done = xworld.step(action[0, 0])
                reward = teacher.reward
                reinforce_model.rewards.append(reward)
                cumulative_reward.append(reward * discount)
                discount *= args.gamma
                if done:
                    break
            cumulative_rewards.append(numpy.sum(numpy.asarray(cumulative_reward)))
            if i_episode % args.log_interval == 0:
                logging.info('Episode {}\taverage cumulative reward: {:.2f}'.format(
                    i_episode, numpy.mean(numpy.asarray(cumulative_rewards))))
                cumulative_rewards = []

if __name__ == '__main__':
    main()

