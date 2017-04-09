import numpy
from xworld import xworld_utils
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    xworld_utils.direction_type()
    xworld_utils.parse_item_list()
    xworld_utils.load_item_images()
    xworld_utils.is_out_border(numpy.array([1, 2]), 3, 4)

if __name__ == '__main__':
    main()
