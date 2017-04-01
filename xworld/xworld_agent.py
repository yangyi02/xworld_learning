import numpy
import random
import xworld_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorldAgent(object):
    """
    XWorld agent for xworld robot learning
    The agent is able to navigate or speak
    """
    def __init__(self, args):
        """
        In the 2d xworld, navigation is a discrete action with up, down, left, right.
        speak is a continuous action that generate words.
        """
        self.name = 'robot_0'
        self.action_type = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.action_type)
        self.velocity = self.get_velocity()
        # TODO: add random speak action

    def seed(self, seed=None):
        self.seed = seed
        random.seed(seed)

    def random_action(self):
        """
        Randomly sample an agent action
        """
        action = random.randint(0, self.num_actions-1)
        # TODO: add random speak action
        return action

    @staticmethod
    def get_velocity():
        velocity = {}
        velocity['up'] = numpy.array([0, -1])
        velocity['down'] = numpy.array([0, 1])
        velocity['left'] = numpy.array([-1, 0])
        velocity['right'] = numpy.array([1, 0])
        return velocity


def main():
    logging.info("test xworld agent functions")
    args = xworld_args.parser().parse_args()
    agent = XWorldAgent(args)
    for i in range(10):
        action = agent.random_action()
        logging.info(action)
    logging.info("test xworld agent functions done")

if __name__ == '__main__':
    main()
