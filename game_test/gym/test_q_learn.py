import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from learning import learning_args, cuda, q_learn
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    args = learning_args.parser().parse_args()
    env = gym.make('CartPole-v0')
    args.train = True
    logging.getLogger().setLevel(logging.INFO)
    logging.info(args)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda.use_cuda():
        torch.cuda.manual_seed_all(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_hidden = 128
    num_actions = env.action_space.n
    model = q_learn.Net(num_inputs, num_hidden, num_actions)
    model = model.cuda() if cuda.use_cuda() else model

    if args.train:
        logging.info('training')
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        reinforcement_model = q_learn.QLearn(args, num_actions, model, optimizer, exploration=True)

        running_reward = 10
        for i_episode in range(args.num_games):
            state = env.reset()
            for t in range(args.max_episode_length):  # Don't infinite loop while learning
                # Sample action from epsilon greed policy
                action = reinforcement_model.select_action(state)
                next_state, reward, done, _ = env.step(action)
                if args.render:
                    env.render()
                # Restore transition in memory
                reinforcement_model.memory.push([state, action, reward, next_state])
                if done:
                    break
                state = next_state

            running_reward = running_reward * 0.99 + t * 0.01
            if i_episode % args.log_interval == 0:
                logging.info('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                    i_episode, t, running_reward))
            if running_reward > 195:
                logging.info("Solved! Running reward is now {} and "
                             "the last episode runs to {} time steps!".format(running_reward, t))
                break

            if len(reinforcement_model.memory) >= args.batch_size:
                # Sample mini-batch transitions from memory
                reinforcement_model.optimize()

        with open('actor_critic_model.pth', 'wb') as handle:
            torch.save(model.state_dict(), handle)
    else:
        logging.info('testing')
        model.load_state_dict(torch.load('actor_critic_model.pth'))
        reinforcement_model = q_learn.QLearn(args, num_actions, model)

        for i_episode in range(100):
            state = env.reset()
            for t in range(10000):  # Don't infinite loop while learning
                action = reinforcement_model.select_action(state)
                state, reward, done, _ = env.step(action[0,0])
                if args.render:
                    env.render()
                if done:
                    break

            if i_episode % args.log_interval == 0:
                logging.info('Episode {}\tLast length: {:5d}'.format(i_episode, t))

if __name__ == '__main__':
    main()

