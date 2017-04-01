import argparse
from xworld import xworld_args
from learning import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parser():
    xworld_parser = xworld_args.parser()
    learning_parser = learning_args.parser()
    parser = argparse.ArgumentParser(parents=[xworld_parser, learning_parser])
    return parser


def main():
    args = parser().parse_args()
    # xworld args
    logging.info(args.teacher_type)
    logging.info(args.single_word)
    logging.info(args.show_frame)
    logging.info(args.map_config)
    logging.info(args.keep_command)
    logging.info(args.ego_centric)
    logging.info(args.visible_radius_unit)
    logging.info(args.image_block_size)
    logging.info(args.pause_screen)
    # learning args
    logging.info(args.method)
    logging.info(args.gamma)
    logging.info(args.seed)
    logging.info(args.render)
    logging.info(args.log_interval)

if __name__ == '__main__':
    main()
