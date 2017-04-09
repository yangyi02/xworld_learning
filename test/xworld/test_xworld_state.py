from xworld import xworld_args, xworld_state, xworld_agent
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info("test xworld state functions")
    args = xworld_args.parser().parse_args()
    agent = xworld_agent.XWorldAgent(args)
    agent.name = 'robot_0'
    map_config_files = ['../../xworld/map_examples/example1.json',
                        '../../xworld/map_examples/example2.json',
                        '../../xworld/map_examples/example3.json',
                        '../../xworld/map_examples/example4.json',
                        '../../xworld/map_examples/example5.json']
    for map_config_file in map_config_files:
        args.map_config = map_config_file
        state = xworld_state.XWorldState(args)
        state.reset(agent)
        logging.info(state.xmap)
        logging.info(state.inner_state)
        logging.info(state.image)
    logging.info("test xworld state functions done")

if __name__ == '__main__':
    main()
