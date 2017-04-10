import copy
import numpy
import json
import random
from . import xworld_item, xworld_utils
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorldMap(object):
    def __init__(self, map_config_file, image_block_size):
        # global xworld properties
        self.item_list, self.item_class_id = xworld_utils.parse_item_list()
        self.item_images = xworld_utils.load_item_images(image_block_size)
        # local xworld map properties
        self.dim = []
        self.init_items = []
        self.items = []
        self.item_location_map = {}
        self.item_name_map = {}
        self.init_map_config(map_config_file)

    def seed(self, seed=None):
        self.seed = seed
        random.seed(seed)

    def init_map_config(self, map_config_file):
        logging.info("loading %s", map_config_file)
        map_config = json.load(open(map_config_file))
        self.dim = map_config['dimensions']  # xworld map height and width
        for item_type, item_conf in map_config['items'].items():
            for i in range(item_conf['number']):
                item = xworld_item.XWorldItem(item_type, i)
                if 'instances' in item_conf:
                    if i < len(item_conf['instances']):
                        # item_name = item_conf['instances'].keys()[i]
                        item_name = list(item_conf['instances'].keys())[i]
                        item_class_name = item_name.split('_')[0]
                        assert (item_class_name in self.item_list[item_type].keys())
                        item_class_id = self.item_class_id[item_class_name]
                        item_location = numpy.array(item_conf['instances'][item_name])
                        item = xworld_item.XWorldItem(item_type, i, item_name, item_class_name, item_class_id, item_location)
                self.init_items.append(item)
        logging.info("finish loading %s", map_config_file)

    def reset(self):
        """
        Reset xworld map
        """
        self.items = copy.deepcopy(self.init_items)
        # Replace items with random names to randomly sampled instances
        self.reset_item()
        # Initialize item corresponding image name
        self.reset_image()
        # Initialize location, make sure it's a valid map
        self.reset_location()
        # Build a mapping from location to item id
        self.build_item_map()

    def reset_item(self):
        """
        Reset xworld map items
        """
        for i, item in enumerate(self.items):
            if item.name == '' or item.class_name == '':
                item_type = item.item_type
                item_class_name = random.choice(list(self.item_list[item_type].keys()))
                item_name = str(item_class_name) + '_' + str(item.index)
                self.items[i].name = item_name
                self.items[i].class_name = item_class_name
                self.items[i].class_id = self.item_class_id[item_class_name]

    def reset_image(self):
        """
        Reset xworld map item images
        """
        for i, item in enumerate(self.items):
            item_type = item.item_type
            item_class_name = item.class_name
            image_id = random.randint(1, int(self.item_list[item_type][item_class_name]))
            image_name = str(item_type) + '_' + str(item_class_name) + '_' + str(image_id) + '.jpg'
            self.items[i].image_name = image_name
            self.items[i].image = self.item_images[image_name]

    def reset_location(self):
        """
        Reset xworld map item locations
        """
        width, height = self.dim['width'], self.dim['height']
        # Collect current specified locations
        location_exist = {}
        for i, item in enumerate(self.items):
            if len(item.location) > 0:
                location = item.location
                # Check locations to make sure they are inside map boundary
                assert not xworld_utils.is_out_border(location, width, height), 'location: ' + str(location) + ' outside of map'
                # Check locations to make sure they are not overlapping
                assert tuple(location) not in location_exist, 'multiple items locate on the same location'
                location_exist[tuple(location)] = True
        # Assign random locations to unspecified items
        # TODO: Make the location sampling more efficient
        # TODO: make the map always a valid map: existing at least on route to goal
        for i, item in enumerate(self.items):
            if len(item.location) == 0:
                location = numpy.array([random.randint(0, width-1), random.randint(0, height-1)])
                while tuple(location) in location_exist:
                    location = numpy.array([random.randint(0, width-1), random.randint(0, height-1)])
                self.items[i].location = location
                location_exist[tuple(location)] = True

    def build_item_map(self):
        self.item_location_map = {}
        for i, item in enumerate(self.items):
            if not item.is_removed:
                if not tuple(item.location) in self.item_location_map:
                    self.item_location_map[tuple(item.location)] = [i]
                else:
                    self.item_location_map[tuple(item.location)].append(i)
        self.item_name_map = {}
        for i, item in enumerate(self.items):
            assert item.name not in self.item_name_map, "error, duplicate item names for multiple objects"
            if not item.is_removed:
                self.item_name_map[item.name] = i

    def step(self, agent, action):
        agent_id = self.item_name_map[agent.name]
        velocity = agent.velocity[action]
        self.update_location(agent_id, velocity)
        self.build_item_map()
        self.clean()
        self.build_item_map()

    def update_location(self, item_id, velocity):
        next_location = self.items[item_id].get_next_location(velocity)
        if xworld_utils.is_out_border(next_location, self.dim['width'], self.dim['height']):
            return False
        if tuple(next_location) in self.item_location_map:
            for next_item_id in self.item_location_map[tuple(next_location)]:
                next_item_type = self.items[next_item_id].item_type
                if not self.items[item_id].is_movable(next_item_type):
                    return False
                if self.items[next_item_id].is_movable():
                    if not self.update_location(next_item_id, velocity):
                        return False
        self.items[item_id].location = next_location
        return True

    def clean(self):
        """
        Clean items if they are on the same location
        Rebuild item_map after cleaning
        """
        for x in range(self.dim['width']):
            for y in range(self.dim['height']):
                location = numpy.array([x, y])
                if tuple(location) in self.item_location_map:
                    self.clean_items(location)

    def clean_items(self, location):
        """
        Clean items if they are on the same location
        """
        item_ids = self.item_location_map[tuple(location)]
        if len(item_ids) == 1:
            return False
        for item_id in item_ids:
            for next_item_id in item_ids:
                if item_id == next_item_id:
                    continue
                next_item_type = self.items[next_item_id].item_type
                if self.items[item_id].to_be_removed(next_item_type):
                    self.items[item_id].is_removed = True
                    break
        return True
