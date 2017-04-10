import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import xworld_learning_args
from xworld import xworld_navi_goal
from learning.reinforce import Reinforce, Policy
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info("test xworld learning functions")
    args = xworld_learning_args.parser().parse_args()
    args.map_config = 'xworld/map_examples/example1.json'
    env = xworld_navi_goal.XWorldNaviGoal(args)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    env.reset()
    num_inputs = env.state.inner_state.size
    num_hidden = 128
    num_actions = env.agent.num_actions
    model = Policy(num_inputs, 128, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    reinforce_model = Reinforce(args.gamma, model, optimizer)

    for i_episode in range(10):
        state, teacher = env.reset()
        for t in range(10): # Don't infinite loop while learning
            action = reinforce_model.select_action(state.inner_state.flatten())
            next_state, teacher, done = env.step(action[0,0])
            reward = teacher.reward
            reinforce_model.rewards.append(reward)
            if done:
                break
        reinforce_model.optimize()
    logging.info("test xworld learning functions done")

if __name__ == '__main__':
    main()
