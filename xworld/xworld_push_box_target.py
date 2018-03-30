import random
from .xworld import XWorld
from .xworld_teacher import XWorldTeacher
from . import xworld_utils
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorldPushBoxTarget(XWorld):
    """
    XWorld interface for xworld robot learning
    """
    def __init__(self, args):
        super().__init__(args)
        self.teacher = XWorldTeacherPushBoxTarget(args)


class XWorldTeacherPushBoxTarget(XWorldTeacher):
    """
    XWorld reward for push box to target task
    """
    def __init__(self, args):
        super().__init__(args)
        self.box_target_locations = []
        self.rewards['push_box_target'] = 0.0

    def reset_command(self, state):
        """
        The command contain nothing but set box goal locations as the target location
        It is possible there are multiple boxes and multiple targets in the map
        The function will return all the goal locations
        """
        self.box_target_locations = []
        for item in state.xmap.items:
            if item.class_name == 'target':
                self.box_target_locations.append(item.location)

    def update_reward(self, agent, state, action, next_state, num_step):
        self.update_push_box_reward(agent, state, action, next_state, num_step)
        self.update_step_reward(agent, state, action, next_state, num_step)
        self.update_out_border_reward(agent, state, action, next_state, num_step)
        self.update_knock_block_reward(agent, state, action, next_state, num_step)

    def update_push_box_reward(self, agent, state, action, next_state, num_step):
        """
        The agent get positive reward when push box to a goal location
        """
        self.rewards['push_box_target'] = 0.0
        box_locations = []
        for i, item in enumerate(next_state.xmap.items):
            if item.item_type == 'box':
                box_locations.append(item.location)
        for box_location in box_locations:
            for box_target_location in self.box_target_locations:
                if (box_location == box_target_location).all():
                    self.rewards['push_box_target'] = 1.0
                    self.done = True
                    return
