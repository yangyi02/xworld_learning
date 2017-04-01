import numpy
import xworld_map
import xworld_args
import xworld_agent
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorldState(object):
    """
    XWorld state and reward for xworld robot learning
    """
    def __init__(self, args):
        """
        The xworld state contains all the information from the environment, including image and teacher's language as well as feedback reward from teacher
        """
        self.args = args
        self.image_block_size = args.image_block_size
        self.xmap = xworld_map.XWorldMap(args.map_config, args.image_block_size)
        self.image = []
        self.inner_state = []
        self.onehot_state = []
        self.origin_image = []
        self.origin_inner_state = []
        self.origin_onehot_state = []
        self.plain_ground_image = []
        self.init_plain_ground_image()

    def seed(self, seed=None):
        self.xmap.seed(seed)

    def init_plain_ground_image(self):
        width, height = self.xmap.dim['width'], self.xmap.dim['height']
        block_size = self.image_block_size
        self.plain_ground_image = numpy.full((height * block_size, width * block_size, 3), 1, dtype=float)
        # create dark line for visualizing grid
        for i in range(0, height * block_size, block_size):
            self.plain_ground_image[i, 0:width * block_size, :] = 0
        for i in range(block_size-1, height * block_size, block_size):
            self.plain_ground_image[i, 0:width * block_size, :] = 0
        for j in range(0, width * block_size, block_size):
            self.plain_ground_image[0:height * block_size, j, :] = 0
        for j in range(block_size-1, width * block_size, block_size):
            self.plain_ground_image[0:height * block_size, j, :] = 0

    def reset(self, agent):
        self.xmap.reset()
        self.construct_state(agent)

    def step(self, agent, action):
        self.xmap.step(agent, action)
        self.construct_state(agent)

    def construct_state(self, agent):
        """
        Construct state output from xmap
        """
        self.construct_inner_state(agent)
        self.construct_image(agent)

    def construct_inner_state(self, agent):
        """
        Construct simple inner state image
        One can interpret as the abstract of full state image
        """
        width, height = self.xmap.dim['width'], self.xmap.dim['height']
        num_classes = len(self.xmap.item_class_id)
        self.origin_inner_state = numpy.zeros((height, width), dtype=int)
        self.origin_onehot_state = numpy.zeros((height, width, num_classes+1), dtype=int)
        self.origin_onehot_state[:, :, 0] = 1
        for item in self.xmap.items:
            if not item.is_removed:
                location = item.location
                self.origin_inner_state[location[1], location[0]] = item.class_id
                self.origin_onehot_state[location[1], location[0], item.class_id] = 1
                self.origin_onehot_state[location[1], location[0], 0] = 0
        if self.args.ego_centric:
            self.inner_state = numpy.full((2*height-1, 2*width-1), -1, dtype=int)
            self.onehot_state = numpy.full((2*height-1, 2*width-1, num_classes+1), 0, dtype=int)
            agent_id = self.xmap.item_name_map[agent.name]
            agent_location = self.xmap.items[agent_id].location
            start_x = 0 - agent_location[0] + width - 1
            start_y = 0 - agent_location[1] + height - 1
            end_x = start_x + width
            end_y = start_y + height
            self.inner_state[start_y:end_y, start_x:end_x] = self.origin_inner_state
            self.onehot_state[start_y:end_y, start_x:end_x, :] = self.origin_onehot_state
            if self.args.visible_radius_unit > 0:
                # partially observed
                w_radius = min(self.args.visible_radius_unit, width - 1)
                h_radius = min(self.args.visible_radius_unit, height - 1)
                start_x = width - w_radius - 1
                start_y = height - h_radius - 1
                end_x = width + w_radius
                end_y = height + h_radius
                self.inner_state = self.inner_state[start_y:end_y, start_x:end_x]
                self.onehot_state = self.onehot_state[start_y:end_y, start_x:end_x, :]
        else:
            self.inner_state = self.origin_inner_state
            self.onehot_state = self.origin_onehot_state

    def construct_image(self, agent):
        """
        Construct the full state image
        """
        width, height = self.xmap.dim['width'], self.xmap.dim['height']
        block_size = self.image_block_size
        self.origin_image = numpy.copy(self.plain_ground_image)
        for item in self.xmap.items:
            if not item.is_removed:
                location = item.location
                start_x = location[0] * block_size
                start_y = location[1] * block_size
                end_x = start_x + block_size
                end_y = start_y + block_size
                self.origin_image[start_y:end_y, start_x:end_x, :] = item.image
        if self.args.ego_centric:
            self.image = numpy.full(((2 * height - 1) * block_size,
                                     (2 * width - 1) * block_size, 3), 0.5)
            agent_id = self.xmap.item_name_map[agent.name]
            agent_location = self.xmap.items[agent_id].location
            start_x = (0 - agent_location[0] + width - 1) * block_size
            start_y = (0 - agent_location[1] + height - 1) * block_size
            end_x = start_x + width * block_size
            end_y = start_y + height * block_size
            self.image[start_y:end_y, start_x:end_x, :] = self.origin_image
            if self.args.visible_radius_unit > 0:
                # partially observed
                w_radius = min(self.args.visible_radius_unit, width - 1)
                h_radius = min(self.args.visible_radius_unit, height - 1)
                start_x = (width - 1 - w_radius) * block_size
                start_y = (height - 1 - h_radius) * block_size
                end_x = (width + w_radius) * block_size
                end_y = (height + h_radius) * block_size
                self.image = self.image[start_y:end_y, start_x:end_x, :]
        else:
            self.image = self.origin_image


def main():
    logging.info("test xworld state functions")
    args = xworld_args.parser().parse_args()
    agent = xworld_agent.XWorldAgent(args)
    agent.name = 'robot_0'
    map_config_files = ['map_examples/example1.json', 'map_examples/example2.json',
                        'map_examples/example3.json', 'map_examples/example4.json',
                        'map_examples/example5.json']
    for map_config_file in map_config_files:
        args.map_config = map_config_file
        xworld_state = XWorldState(args)
        xworld_state.reset(agent)
        logging.info(xworld_state.xmap)
        logging.info(xworld_state.inner_state)
        logging.info(xworld_state.image)
    logging.info("test xworld state functions done")

if __name__ == '__main__':
    main()
