import numpy
from xworld import xworld_item
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info("test xworld item functions")
    item = xworld_item.XWorldItem()
    item.location = numpy.array([1, 0])
    logging.info(item.get_next_location(numpy.array([1, 0])))
    item.is_movable()
    item.to_be_removed()
    logging.info("test xworld item functions done")

if __name__ == '__main__':
    main()
