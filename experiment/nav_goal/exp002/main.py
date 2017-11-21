import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import xworld_learning_args
from xworld import xworld_navi_goal
from learning import cuda, actor_critic
import ac_xworld, networks
import time
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def train(args, xworld, model, solver):
    cumulative_rewards = []
    for i_episode in range(args.num_games):
        state, teacher = xworld.reset()
        cumulative_reward = []
        discount = 1.0
        for t in range(args.max_episode_length):  # Don't infinite loop while learning
            state_input = state.onehot_state.swapaxes(0, 2).swapaxes(1, 2)
            command_id = state.xmap.item_class_id[teacher.command.split(' ')[-1]]
            action = solver.select_action(state_input, command_id)
            next_state, teacher, done = xworld.step(action[0, 0])
            reward = teacher.reward
            solver.rewards.append(reward)
            cumulative_reward.append(reward * discount)
            discount *= args.gamma
            if done:
                break
        cumulative_rewards.append(numpy.sum(numpy.asarray(cumulative_reward)))
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
        solver.optimize()

    with open(os.path.join(args.save_dir, 'final.pth'), 'wb') as handle:
        torch.save(model.state_dict(), handle)


def test(args, xworld, model, solver):
    cumulative_rewards = []
    for i_episode in range(args.num_games):
        state, teacher = xworld.reset()
        cumulative_reward = []
        discount = 1.0
        for t in range(args.max_episode_length):  # Don't infinite loop while learning
            state_input = state.onehot_state.swapaxes(0, 2).swapaxes(1, 2)
            command_id = state.xmap.item_class_id[teacher.command.split(' ')[-1]]
            action = solver.select_action(state_input, command_id)
            input("PRESS ANY KEY TO CONTINUE.")
            next_state, teacher, done = xworld.step(action[0, 0])
            reward = teacher.reward
            solver.rewards.append(reward)
            cumulative_reward.append(reward * discount)
            discount *= args.gamma
            if done:
                break
        cumulative_rewards.append(numpy.sum(numpy.asarray(cumulative_reward)))
        if i_episode % args.log_interval == 0:
            logging.info('Episode {}\taverage cumulative reward: {:.2f}'.format(
                i_episode, numpy.mean(numpy.asarray(cumulative_rewards))))
            cumulative_rewards = []


def main():
    args = xworld_learning_args.parser().parse_args()
    args.map_config = 'empty_ground.json'
    args.learning_rate = 0.001
    args.keep_command = True
    logging.info(args)
    xworld = xworld_navi_goal.XWorldNaviGoal(args)
    xworld.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda.use_cuda():
        torch.cuda.manual_seed_all(args.seed)

    xworld.reset()
    (height, width, channel) = xworld.state.onehot_state.shape
    num_hidden = 128
    dict_length = len(xworld.state.xmap.item_class_id)
    num_actions = xworld.agent.num_actions
    model = networks.Network(height, width, channel, num_hidden, dict_length, num_actions)
    model = model.cuda() if cuda.use_cuda() else model

    if args.opt_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.opt_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.opt_method == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)

    solver = ac_xworld.ActorCritic(args.gamma, model, optimizer)

    if args.train:
        logging.info('training')
        train(args, xworld, model, solver)
    else:
        logging.info('testing')
        model.load_state_dict(torch.load(args.init_model_path))
        test(args, xworld, model, solver)

if __name__ == '__main__':
    main()

