import numpy
import random
import xworld_utils
import xworld_args
from xworld_teacher import XWorldTeacher


class XWorldTeacherNaviBox(XWorldTeacher):
    """
    XWorld reward for navigation box task
    """
    def __init__(self, args):
        XWorldTeacher.__init__(self, args)
        self.goal_locations = []
        self.rewards['navi_box'] = 0.0

    def reset_command(self, state):
        """
        The command should contain a goal direction around the box
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
            self.command = 'go to box ' + direction
        self.goal_locations = []
        for item in state.xmap.items:
            if item.class_name == 'box':
                goal_location = item.location + offset
                self.goal_locations.append(goal_location)

    def update_reward(self, agent, state, action, next_state, num_step):
        self.update_navi_box_reward(agent, state, action, next_state, num_step)
        self.update_step_reward(agent, state, action, next_state, num_step)
        self.update_out_border_reward(agent, state, action, next_state, num_step)
        self.update_knock_block_reward(agent, state, action, next_state, num_step)

    def update_navi_box_reward(self, agent, state, action, next_state, num_step):
        """
        The agent get positive reward when navigation reach box close-by location
        """
        self.rewards['navi_box'] = 0.0
        agent_id = next_state.xmap.item_name_map[agent.name]
        agent_location = next_state.xmap.items[agent_id].location
        for goal_location in self.goal_locations:
            if (agent_location == goal_location).all():
                self.rewards['navi_box'] = 1.0
                self.done = True
                return


def main():
    print "test xworld teacher functions"
    args = xworld_args.parser().parse_args()
    xworld_teacher_navi_box = XWorldTeacherNaviBox(args)
    print "test world teacher functions done"

if __name__ == '__main__':
    main()
