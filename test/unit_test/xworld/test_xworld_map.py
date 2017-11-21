from xworld import xworld_map
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info("test xworld map functions")
    map_config_files = ['../../xworld/map_examples/example1.json',
                        '../../xworld/map_examples/example2.json',
                        '../../xworld/map_examples/example3.json',
                        '../../xworld/map_examples/example4.json',
                        '../../xworld/map_examples/example5.json']
    for map_config_file in map_config_files:
        xmap = xworld_map.XWorldMap(map_config_file, 8)
        logging.info(xmap.item_list)
        logging.info(xmap.dim)
        logging.info(xmap.init_items)
        logging.info(xmap.items)
        xmap.reset()
        logging.info(xmap.init_items)
        logging.info(xmap.items)
        logging.info(xmap.item_location_map)
        logging.info(xmap.item_name_map)
        xmap.clean()
    logging.info("test xworld map functions done")

if __name__ == '__main__':
    main()
