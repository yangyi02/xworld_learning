import numpy
from learning import actor_critic, cuda
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info('testing actor-critic functions')
    model = actor_critic.Net(4, 128, 2)
    model = model.cuda() if cuda.use_cuda() else model
    reinforcement_model = actor_critic.ActorCritic(0.99, model)
    state = numpy.random.randn(4)
    reinforcement_model.select_action(state)
    reinforcement_model.rewards.append(1.0)
    reinforcement_model.optimize()
    logging.info('testing actor-critic functions done')

if __name__ == '__main__':
    main()
