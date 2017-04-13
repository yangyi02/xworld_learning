import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')

from learning import learning_args, cuda, async_actor_critic
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def train(rank, args, shared_model, optimizer):
    env = gym.make('CartPole-v0')
    logging.getLogger().setLevel(logging.INFO)
    env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if cuda.use_cuda():
        torch.cuda.manual_seed_all(args.seed + rank)

    num_inputs = env.observation_space.shape[0]
    num_hidden = 128
    num_actions = env.action_space.n
    model = async_actor_critic.Policy(num_inputs, num_hidden, num_actions)
    model = model.cuda() if cuda.use_cuda() else model

    reinforcement_model = async_actor_critic.AsyncActorCritic(args.gamma, model, shared_model,
                                                              optimizer)

    running_reward = 10
    for i_episode in range(1000):
        state = env.reset()
        model.load_state_dict(shared_model.state_dict())
        for t in range(10000):  # Don't infinite loop while learning
            action = reinforcement_model.select_action(state)
            state, reward, done, _ = env.step(action[0, 0])
            if args.render:
                env.render()
            reinforcement_model.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        reinforcement_model.optimize()
        if i_episode % args.log_interval == 0:
            logging.info('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > 195:
            logging.info("Solved! Running reward is now {} and "
                         "the last episode runs to {} time steps!".format(running_reward, t))
            break

    with open('async_actor_critic_model.pth', 'wb') as handle:
        torch.save(model.state_dict(), handle)


def test(rank, args, shared_model, optimizer):
    env = gym.make('CartPole-v0')
    logging.getLogger().setLevel(logging.INFO)
    env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if cuda.use_cuda():
        torch.cuda.manual_seed_all(args.seed + rank)

    num_inputs = env.observation_space.shape[0]
    num_hidden = 128
    num_actions = env.action_space.n
    model = async_actor_critic.Policy(num_inputs, num_hidden, num_actions)
    model = model.cuda() if cuda.use_cuda() else model

    reinforcement_model = async_actor_critic.AsyncActorCritic(args.gamma, model, shared_model,
                                                              optimizer)

    for i_episode in range(100):
        state = env.reset()
        model.load_state_dict(shared_model.state_dict())
        for t in range(10000):  # Don't infinite loop while learning
            action = reinforcement_model.select_action(state)
            state, reward, done, _ = env.step(action[0, 0])
            if args.render:
                env.render()
            if done:
                break

        if i_episode % args.log_interval == 0:
            logging.info('Episode {}\tLast length: {:5d}'.format(i_episode, t))


def main():
    args = learning_args.parser().parse_args()
    env = gym.make('CartPole-v0')
    logging.getLogger().setLevel(logging.INFO)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda.use_cuda():
        torch.cuda.manual_seed_all(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_hidden = 128
    num_actions = env.action_space.n
    model = async_actor_critic.Policy(num_inputs, num_hidden, num_actions)
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
        model.load_state_dict(torch.load('async_actor_critic_model.pth'))
        p = mp.Process(target=test, args=(0, args, model, optimizer))
        p.start()
        # p.join()

if __name__ == '__main__':
    main()
