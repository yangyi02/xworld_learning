import numpy
import random
import xworld_utils
import xworld_args
from xworld_teacher import XWorldTeacher


class XWorldTeacherPushBox(XWorldTeacher):
    """
    XWorld reward for push box task
    """
    def __init__(self, args):
        XWorldTeacher.__init__(self, args)
        self.goal_box_locations = []
        self.rewards['push_box'] = 0.0

    def reset_command(self, state):
        """
        The command should contain a goal direction to push the box
        It is possible there are multiple boxes in the map
        The function will return all the goal locations
        """
        direction_list, move_distance = xworld_utils.direction_type()
        direction_id = random.randint(0, len(direction_list)-1)
        direction = direction_list[direction_id]
        offset = move_distance[direction_id]
        if self.args.single_word:
            self.command = direction
        else:
            self.command = 'push box ' + direction
        self.goal_box_locations = []
        for item in state.xmap.items:
            if item.class_name == 'box':
                goal_box_location = item.location + offset
                self.goal_box_locations.append(goal_box_location)

    def update_reward(self, agent, state, action, next_state, num_step):
        self.update_push_box_reward(agent, state, action, next_state, num_step)
        self.update_step_reward(agent, state, action, next_state, num_step)
        self.update_out_border_reward(agent, state, action, next_state, num_step)
        self.update_knock_block_reward(agent, state, action, next_state, num_step)

    def update_push_box_reward(self, agent, state, action, next_state, num_step):
        """
        The agent get positive reward when push box to a goal location
        """
        self.rewards['push_box'] = 0.0
        box_locations = []
        for i, item in enumerate(next_state.xmap.items):
            if item.item_type == 'box':
                box_locations.append(item.location)
        for box_location in box_locations:
            for goal_box_location in self.goal_box_locations:
                if (box_location == goal_box_location).all():
                    self.rewards['push_box'] = 1.0
                    self.done = True
                    return

def main():
    print "test xworld teacher functions"
    args = xworld_args.parser().parse_args()
    xworld_teacher_push_box = XWorldTeacherPushBox(args)
    print "test world teacher functions done"


if __name__ == '__main__':
    main()
