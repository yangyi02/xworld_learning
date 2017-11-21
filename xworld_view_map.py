from xworld import xworld, xworld_args
import logging

logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    args = xworld_args.parser().parse_args()
    args.view_map = True
    logging.info(args)
    env = xworld.XWorld(args)
    if args.view_map:
        env.reset()
        env.view_map()


if __name__ == '__main__':
    main()
