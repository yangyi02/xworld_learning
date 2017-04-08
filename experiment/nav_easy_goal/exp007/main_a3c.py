import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')

import xworld_learning_args
from . import xworld_navi_easy_goal
from learning.reinforce import Reinforce
from learning.actor_critic import ActorCritic
from learning.cuda import *
from .train import train, Network
# from test import test
import time
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    args = xworld_learning_args.parser().parse_args()
    args.map_config = 'empty_ground.json'
    args.learning_rate = 0.001
    logging.info(args)
    xworld = xworld_navi_easy_goal.XWorldNaviEasyGoal(args)
    xworld.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(args.seed)

    xworld.reset()
    (height, width, channel) = xworld.state.onehot_state.shape
    num_hidden = 128
    num_actions = xworld.agent.num_actions
    model = Network(height, width, channel, 128, num_actions)
    if USE_CUDA:
        model.cuda()
    model.share_memory()

    if args.opt_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.opt_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.opt_method == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)

    if args.train:
        logging.info('training')
        processes = []
        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, model, optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        logging.info('testing')
        model.load_state_dict(torch.load(args.init_model))

        if args.method == 'reinforce':
            reinforce_model = Reinforce(args.gamma, model, optimizer)
        elif args.method == 'actor_critic':
            reinforce_model = ActorCritic(args.gamma, model, optimizer)
        elif args.method == 'q_learn':
            reinforce_model = QLearn(args.gamma, model, optimizer)

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

