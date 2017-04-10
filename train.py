import os
import numpy
import torch
from . import reinforce, actor_critic, async_actor_critic, cuda
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def train(rank, args, env, model, shared_model, optimizer):
    torch.manual_seed(args.seed + rank)
    if cuda.use_cuda():
        torch.cuda.manual_seed_all(args.seed + rank)

    # env = xworld_navi_easy_goal.XWorldNaviEasyGoal(args)
    env.seed(args.seed + rank)
    # env.reset()
    # (height, width, channel) = env.state.onehot_state.shape
    # num_hidden = 128
    # num_actions = env.agent.num_actions
    # model = Network(height, width, channel, num_hidden, num_actions)
    # model = model.cuda() if cuda.use_cuda() else model

    if args.method == 'reinforce':
        reinforcement_model = reinforce.Reinforce(args.gamma, model, optimizer)
    elif args.method == 'actor_critic':
        reinforcement_model = actor_critic.ActorCritic(args.gamma, model, optimizer)
    elif args.method == 'async_actor_critic':
        reinforcement_model = async_actor_critic.AsyncActorCritic(args.gamma, model, shared_model, optimizer)
    elif args.method == 'q_learn':
        reinforcement_model = q_learn.QLearn(args.gamma, model, optimizer)

    cumulative_rewards = []
    for i_episode in range(args.num_games):
        model.load_state_dict(shared_model.state_dict())
        state, teacher = env.reset()
        cumulative_reward = []
        discount = 1.0
        for t in range(args.max_episode_length):  # Don't infinite loop while learning
            state_input = state.onehot_state.swapaxes(0, 2).swapaxes(1, 2)
            action = reinforcement_model.select_action(state_input)
            next_state, teacher, done = env.step(action[0, 0])
            reward = teacher.reward
            reinforcement_model.rewards.append(reward)
            cumulative_reward.append(reward * discount)
            discount *= args.gamma
            if done:
                break
        cumulative_rewards.append(numpy.sum(numpy.asarray(cumulative_reward)))
        reinforcement_model.optimize()
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
