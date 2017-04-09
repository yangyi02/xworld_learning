import argparse
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parser():
    parser = argparse.ArgumentParser(description='XWorld simulator', add_help=False)
    parser.add_argument('--teacher_type', default='NAVI_GOAL',
                        help='NAVI_GOAL, NAVI_NEAR, NAVI_BOX, PUSH_BOX, ...')
    parser.add_argument('--single_word', action='store_true',
                        help='whether the command is a single word or not')
    parser.add_argument('--show_frame', action='store_true',
                        help='whether show frames during running')
    parser.add_argument('--map_config', default='map_examples/example1.json',
                        help='map config file')
    parser.add_argument('--keep_command', action='store_true',
                        help='whether teacher keep providing command at every step')
    parser.add_argument('--ego_centric', action='store_true',
                        help='whether the image is ego centric')
    parser.add_argument('--visible_radius_unit', type=int, default=1,
                        help='robot visible field radius')
    parser.add_argument('--image_block_size', type=int, default=64,
                        help='image block size per object')
    parser.add_argument('--pause_screen', action='store_true',
                        help='whether pause screen at every step')
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='discount factor in game')
    return parser


def main():
    args = parser().parse_args()
    logging.info(args.teacher_type)
    logging.info(args.single_word)
    logging.info(args.show_frame)
    logging.info(args.map_config)
    logging.info(args.keep_command)
    logging.info(args.ego_centric)
    logging.info(args.visible_radius_unit)
    logging.info(args.image_block_size)
    logging.info(args.pause_screen)

if __name__ == '__main__':
    main()
