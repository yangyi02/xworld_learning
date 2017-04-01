from xworld import XWorld
import xworld_args
import xworld_agent
import xworld_state
import xworld_teacher_push_box
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorldPushBox(XWorld):
    """
    XWorld interface for xworld robot learning
    """
    def __init__(self, args):
        self.args = args
        self.state = xworld_state.XWorldState(args)
        self.agent = xworld_agent.XWorldAgent(args)
        self.teacher = xworld_teacher_push_box.XWorldTeacherPushBox(args)
        self.num_step = 0


def main():
    logging.info("test xworld push box functions")
    map_config_files = ['map_examples/example6.json']
    ego_centrics = [False, True]
    for map_config_file in map_config_files:
        for ego_centric in ego_centrics:
            args = xworld_args.parser().parse_args()
            args.ego_centric = ego_centric
            args.map_config = map_config_file
            xworld_push_box = XWorldPushBox(args)
            for i in range(2):
                xworld_push_box.reset()
                for j in range(10):
                    action = xworld_push_box.agent.random_action()
                    next_state, teacher, done = xworld_push_box.step(action)
                    xworld_push_box.display()
                    if done:
                        break
    logging.info("test world push box functions done")

if __name__ == '__main__':
    main()
