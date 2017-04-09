import numpy
import copy
import matplotlib.pyplot as plt
import cv2
from xworld import xworld
from xworld import xworld_args
from xworld import xworld_agent
from xworld import xworld_state
from xworld import xworld_teacher
import time
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info("test xworld functions")
    map_config_files = ['map_examples/example1.json', 'map_examples/example2.json',
                        'map_examples/example3.json', 'map_examples/example4.json',
                        'map_examples/example5.json']
    ego_centrics = [False, True]
    visible_radius_units = [0, 1]
    for map_config_file in map_config_files:
        for ego_centric in ego_centrics:
            for visible_radius_unit in visible_radius_units:
                args = xworld_args.parser().parse_args()
                args.map_config = map_config_file
                args.ego_centric = ego_centric
                args.visible_radius_unit = visible_radius_unit
                env = xworld.XWorld(args)
                for i in range(2):
                    env.reset()
                    # running_time = []
                    for j in range(10):
                        # start = time.time()
                        action = env.agent.random_action()
                        next_state, teacher, done = env.step(action)
                        # end = time.time()
                        # running_time.append(end-start)
                        # env.display()
                        # end2 = time.time()
                        # print("render time: %.3f" % (end2 - end))
                        if done:
                            break
                    # logging.info('average simulation time: %.3f' % numpy.mean(numpy.asarray(running_time)))
    logging.info("test world functions done")

if __name__ == '__main__':
    main()
