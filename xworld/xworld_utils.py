import os
import json
import numpy
from skimage import io
from skimage import transform
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def direction_type():
    direction_list = ['up', 'down', 'left', 'right']
    move_distance = [numpy.array([0,-1]), numpy.array([0,1]), numpy.array([-1,0]), numpy.array([1,0])]
    return direction_list, move_distance


def parse_item_list(item_list_config='confs/item_list.json'):
    logging.info("loading item list")
    script_dir = os.path.dirname(__file__)  # absolute dir the script is in
    item_list = json.load(open(os.path.join(script_dir, item_list_config)))
    item_class_id, cnt = {}, 1
    # for item_type, items in item_list.iteritems():
    for item_type, items in item_list.items():
        for item_class_name in items.keys():
            item_class_id[item_class_name] = cnt
            cnt += 1
    logging.info("finish loading item list")
    return item_list, item_class_id


def load_item_images(image_block_size=64, item_image_dir='images'):
    logging.info("loading item images")
    script_dir = os.path.dirname(__file__)  # absolute dir the script is in
    image_files = os.listdir(os.path.join(script_dir, item_image_dir))
    item_images = {}
    for image_file in image_files:
        if image_file.endswith('.jpg'):
            image = io.imread(os.path.join(script_dir, item_image_dir, image_file))
            image = transform.resize(image, (image_block_size - 2, image_block_size - 2), mode='constant')
            block_image = numpy.full([image_block_size, image_block_size, 3], 0, dtype=float)
            block_image[1:image_block_size - 1, 1:image_block_size - 1, :] = image
            item_images[image_file] = block_image
    logging.info("finish loading item images")
    return item_images


def is_out_border(location, width, height):
    return location[0] < 0 or location[0] >= width or location[1] < 0 or location[1] >= height
