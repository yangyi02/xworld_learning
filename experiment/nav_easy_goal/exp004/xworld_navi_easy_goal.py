from xworld.xworld_navi_goal import XWorldNaviGoal
from xworld.xworld_teacher_navi_goal import XWorldTeacherNaviGoal


class XWorldNaviEasyGoal(XWorldNaviGoal):
    """
    XWorld interface for xworld robot learning
    """
    def __init__(self, args):
        XWorldNaviGoal.__init__(self, args)
        self.teacher = XWorldTeacherNaviEasyGoal(args)

class XWorldTeacherNaviEasyGoal(XWorldTeacherNaviGoal):
    """
    XWorld reward for navigation easy goal task
    Only navigate a fixed object class
    """
    def __init__(self, args):
        XWorldTeacherNaviGoal.__init__(self, args)

    def reset_command(self, state):
        """
        The command should contain goal name so the agent knows where to go
        It is possible there are multiple same goals in the map
        The function will return all the goal locations
        """
        goal_list = []
        for location, item_id in state.xmap.item_location_map.iteritems():
            if state.xmap.items[item_id[0]].item_type == 'goal':
                goal_list.append(item_id[0])
        assert len(goal_list) > 0, "error: at least one goal is needed for this task"
        goal_class_name = 'apple'
        if self.args.single_word:
            self.command = goal_class_name
        else:
            self.command = 'go to ' + goal_class_name
        self.goal_locations = []
        self.wrong_goal_locations = []
        for goal_id in goal_list:
            item = state.xmap.items[goal_id]
            if item.class_name == goal_class_name:
                self.goal_locations.append(item.location)
            else:
                self.wrong_goal_locations.append(item.location)
