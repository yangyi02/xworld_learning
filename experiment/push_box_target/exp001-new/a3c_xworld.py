import numpy
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
# from torch.autograd import Variable
from learning import cuda, async_actor_critic
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


SavedAction = namedtuple('SavedAction', ['action', 'value'])
class AsyncActorCritic(async_actor_critic.AsyncActorCritic):
    def __init__(self, gamma, model, shared_model, optimizer=None):
        super().__init__(gamma, model, shared_model, optimizer)

    def select_action(self, state, exploration=None):
        state = cuda.from_numpy(state).unsqueeze(0)
        probs, state_value = self.model(cuda.variable(state))
        action = probs.multinomial()
        self.saved_actions.append(SavedAction(action, state_value))
        return action.data
