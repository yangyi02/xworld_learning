import numpy
from xworld import xworld, xworld_args
import time
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info("test xworld functions")
    map_config_files = ['../../xworld/map_examples/example1.json',
                        '../../xworld/map_examples/example2.json',
                        '../../xworld/map_examples/example3.json',
                        '../../xworld/map_examples/example4.json',
                        '../../xworld/map_examples/example5.json']
    ego_centrics = [False, True]
    visible_radius_units = [0, 1]
    running_time = []
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
                    for j in range(10):
                        action = env.agent.random_action()
                        start_time = time.time()
                        next_state, teacher, done = env.step(action)
                        end_time = time.time()
                        running_time.append(end_time-start_time)
                        if done:
                            break
    logging.info('average simulation time per step: %.3f' % numpy.mean(numpy.asarray(running_time)))
    logging.info("test world functions done")

if __name__ == '__main__':
    main()
