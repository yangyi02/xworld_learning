from .xworld import XWorld
from . import xworld_args
from . import xworld_agent
from . import xworld_state
from . import xworld_teacher_navi_goal
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorldNaviGoal(XWorld):
    """
    XWorld interface for xworld robot learning
    """
    def __init__(self, args):
        self.args = args
        self.state = xworld_state.XWorldState(args)
        self.agent = xworld_agent.XWorldAgent(args)
        self.teacher = xworld_teacher_navi_goal.XWorldTeacherNaviGoal(args)
        self.num_step = 0


def main():
    logging.info("test xworld navigation goal functions")
    map_config_files = ['map_examples/example6.json']
    ego_centrics = [False, True]
    for map_config_file in map_config_files:
        for ego_centric in ego_centrics:
            args = xworld_args.parser().parse_args()
            args.ego_centric = ego_centric
            args.map_config = map_config_file
            xworld_navi_goal = XWorldNaviGoal(args)
            for i in range(2):
                xworld_navi_goal.reset()
                for j in range(20):
                    action = xworld_navi_goal.agent.random_action()
                    next_state, teacher, done = xworld_navi_goal.step(action)
                    xworld_navi_goal.display()
                    if done:
                        break
    logging.info("test world navigation goal functions done")

if __name__ == '__main__':
    main()
