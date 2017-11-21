from xworld import xworld_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info("test xworld argument functions")
    args = xworld_args.parser().parse_args()
    logging.info(args.teacher_type)
    logging.info(args.single_word)
    logging.info(args.show_frame)
    logging.info(args.map_config)
    logging.info(args.keep_command)
    logging.info(args.ego_centric)
    logging.info(args.visible_radius_unit)
    logging.info(args.image_block_size)
    logging.info(args.pause_screen)
    logging.info("test xworld argument functions done")

if __name__ == '__main__':
    main()
