import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import xworld_learning_args
from . import xworld_navi_easy_goal
from learning.reinforce import Reinforce
from learning.async_actor_critic import AsyncActorCritic
from learning.cuda import *
from model import Network
import time
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def train(rank, args, shared_model, optimizer):
    torch.manual_seed(args.seed + rank)
    if USE_CUDA:
        torch.cuda.manual_seed_all(args.seed + rank)

    xworld = xworld_navi_easy_goal.XWorldNaviEasyGoal(args)
    xworld.seed(args.seed + rank)

    xworld.reset()
    (height, width, channel) = xworld.state.onehot_state.shape
    num_hidden = 128
    num_actions = xworld.agent.num_actions
    model = Network(height, width, channel, 128, num_actions)
    if USE_CUDA:
        model.cuda()

    if args.method == 'reinforce':
        reinforce_model = Reinforce(args.gamma, model, optimizer)
    elif args.method == 'actor_critic':
        reinforce_model = AsyncActorCritic(args.gamma, model, shared_model, optimizer)
    elif args.method == 'q_learn':
        reinforce_model = QLearn(args.gamma, model, optimizer)

    cumulative_rewards = []
    for i_episode in range(args.num_games):
        model.load_state_dict(shared_model.state_dict())
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
                torch.save(shared_model.state_dict(), handle)

    with open(os.path.join(args.save_dir, 'final.pth'), 'wb') as handle:
        torch.save(model.state_dict(), handle)
