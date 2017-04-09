from xworld import xworld_args, xworld_teacher_navi_box
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info("test xworld teacher functions")
    args = xworld_args.parser().parse_args()
    teacher = xworld_teacher_navi_box.XWorldTeacherNaviBox(args)
    logging.info("test world teacher functions done")

if __name__ == '__main__':
    main()
