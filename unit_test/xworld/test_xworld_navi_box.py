from xworld import xworld_navi_box, xworld_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info("test xworld navigation box functions")
    map_config_files = ['../../xworld/map_examples/example6.json']
    ego_centrics = [False, True]
    for map_config_file in map_config_files:
        for ego_centric in ego_centrics:
            args = xworld_args.parser().parse_args()
            args.ego_centric = ego_centric
            args.map_config = map_config_file
            env = xworld_navi_box.XWorldNaviBox(args)
            for i in range(2):
                env.reset()
                for j in range(10):
                    action = env.agent.random_action()
                    next_state, teacher, done = env.step(action)
                    env.display()
                    if done:
                        break
    logging.info("test world navigation box functions done")

if __name__ == '__main__':
    main()
