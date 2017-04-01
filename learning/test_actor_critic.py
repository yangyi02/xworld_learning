import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import learning_args
from actor_critic import ActorCritic, Policy


def main():
    args = learning_args.parser().parse_args()
    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_hidden = 128
    num_actions = env.action_space.n
    model = Policy(num_inputs, 128, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    actor_critic_model = ActorCritic(args.gamma, model, optimizer)

    if args.train:
        print('training')

        running_reward = 10
        for i_episode in range(1000):
            state = env.reset()
            for t in range(10000): # Don't infinite loop while learning
                action = actor_critic_model.select_action(state)
                state, reward, done, _ = env.step(action[0,0])
                if args.render:
                    env.render()
                actor_critic_model.rewards.append(reward)
                if done:
                    break

            running_reward = running_reward * 0.99 + t * 0.01
            actor_critic_model.optimize()
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                    i_episode, t, running_reward))
            if running_reward > 195:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break

        with open('actor_critic_model.pth', 'wb') as handle:
            torch.save(model.state_dict(), handle)
    else:
        print('testing')
        model.load_state_dict(torch.load('actor_critic_model.pth'))

        for i_episode in range(100):
            state = env.reset()
            for t in range(10000): # Don't infinite loop while learning
                action = actor_critic_model.select_action(state)
                state, reward, done, _ = env.step(action[0,0])
                if args.render:
                    env.render()
                if done:
                    break

            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast length: {:5d}'.format(
                    i_episode, t))


if __name__ == '__main__':
    main()

