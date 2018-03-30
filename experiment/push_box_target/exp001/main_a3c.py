import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')

import xworld_learning_args
from xworld import xworld_push_box_target
from learning import cuda
import a3c_xworld
import networks
import time
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def train(rank, args, shared_model, optimizer):
    torch.manual_seed(args.seed + rank)
    if cuda.use_cuda():
        torch.cuda.manual_seed_all(args.seed + rank)

    env = xworld_push_box_target.XWorldPushBoxTarget(args)
    env.seed(args.seed + rank)

    env.reset()
    (height, width, channel) = env.state.onehot_state.shape
    num_hidden1 = 1
    num_hidden2 = 4
    num_actions = env.agent.num_actions
    model = networks.Network(height, width, channel, num_hidden1, num_hidden2, num_actions)
    model = model.cuda() if cuda.use_cuda() else model

    solver = a3c_xworld.AsyncActorCritic(args.gamma, model, shared_model, optimizer)

    cumulative_rewards = []
    for i_episode in range(args.num_games):
        model.load_state_dict(shared_model.state_dict())
        state, teacher = env.reset()
        cumulative_reward = []
        discount = 1.0
        for t in range(args.max_episode_length):  # Don't infinite loop while learning
            state_input = state.onehot_state.swapaxes(0, 2).swapaxes(1, 2)
            action = solver.select_action(state_input)
            next_state, teacher, done = env.step(action[0, 0])
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
                torch.save(shared_model.state_dict(), handle)
        solver.optimize()

    with open(os.path.join(args.save_dir, 'final.pth'), 'wb') as handle:
        torch.save(model.state_dict(), handle)


def test(rank, args, shared_model, optimizer):
    torch.manual_seed(args.seed + rank)
    if cuda.use_cuda():
        torch.cuda.manual_seed_all(args.seed + rank)

    env = xworld_push_box_target.XWorldPushBoxTarget(args)
    env.seed(args.seed + rank)

    env.reset()
    (height, width, channel) = env.state.onehot_state.shape
    num_hidden1 = 1
    num_hidden2 = 4
    num_actions = env.agent.num_actions
    model = networks.Network(height, width, channel, num_hidden1, num_hidden2, num_actions)
    model = model.cuda() if cuda.use_cuda() else model

    solver = a3c_xworld.AsyncActorCritic(args.gamma, model, shared_model, optimizer)

    cumulative_rewards = []
    for i_episode in range(args.num_games):
        model.load_state_dict(shared_model.state_dict())
        state, teacher = env.reset()
        cumulative_reward = []
        discount = 1.0
        for t in range(args.max_episode_length):  # Don't infinite loop while learning
            state_input = state.onehot_state.swapaxes(0, 2).swapaxes(1, 2)
            action = solver.select_action(state_input)
            next_state, teacher, done = env.step(action[0, 0])
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
    args.learning_rate = 0.0001
    args.init_model_path = os.path.join(args.save_dir, 'final.pth')
    logging.info(args)
    env = xworld_push_box_target.XWorldPushBoxTarget(args)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda.use_cuda():
        torch.cuda.manual_seed_all(args.seed)

    env.reset()
    (height, width, channel) = env.state.onehot_state.shape
    num_hidden1 = 1
    num_hidden2 = 4
    num_actions = env.agent.num_actions
    model = networks.Network(height, width, channel, num_hidden1, num_hidden2, num_actions)
    model = model.cuda() if cuda.use_cuda() else model
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
        model.load_state_dict(torch.load(args.init_model_path))
        p = mp.Process(target=test, args=(0, args, model, optimizer))
        p.start()

if __name__ == '__main__':
    main()

