import numpy
import random
import xworld_utils
import xworld_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorldTeacher(object):
    """
    XWorld teacher for xworld robot learning
    The teacher is in charge of giving command, teaching language and assign reward
    """
    def __init__(self, args):
        self.args = args
        self.command = ''
        self.language = ''
        self.reward = 0.0
        self.rewards = {'step': 0.0, 'out_border': 0.0, 'knock_block': 0.0}
        self.cumulative_command = []
        self.cumulative_language = []
        self.cumulative_reward = 0.0
        self.cumulative_discount = 1.0
        self.done = False

    def seed(self, seed=None):
        self.seed = seed
        random.seed(seed)

    def reset(self, state):
        self.reset_command(state)
        self.language = ''
        self.reward = 0.0
        self.reset_rewards()
        self.cumulative_command = []
        self.cumulative_language = []
        self.cumulative_reward = 0.0
        self.cumulative_discount = 1.0
        self.done = False

    def reset_command(self, state):
        """
        Construct teacher command, this is task dependent
        """
        self.command = 'random explore'

    def reset_rewards(self):
        """
        Construct a set of teacher rewards, this is task dependent
        :return: self.rewards is a dictionary
        """
        self.rewards = {'step': 0.0, 'out_border': 0.0, 'knock_block': 0.0}

    def step(self, agent, state, action, next_state, num_step):
        self.update_command(agent, state, action, next_state, num_step)
        self.update_language(agent, state, action, next_state, num_step)
        self.assign_reward(agent, state, action, next_state, num_step)
        self.accumulate()
        return self.done

    def update_command(self, agent, state, action, next_state, num_step):
        """
        Construct teacher command
        The teacher may update command once the agent finish a task
        """
        if num_step > 0 and not self.args.keep_command:
            self.command = ''

    def update_language(self, agent, state, action, next_state, num_step):
        """
        Construct teacher language
        The teacher may teach agent what is nearby the agent
        """
        width, height = state.xmap.dim['width'], state.xmap.dim['height']
        agent_id = next_state.xmap.item_name_map[agent.name]
        agent_location = next_state.xmap.items[agent_id].location
        direction_list, move_distance = xworld_utils.direction_type()
        direction_id = random.randint(0, len(direction_list)-1)
        adjacent_location = agent_location + move_distance[direction_id]
        self.language = direction_list[direction_id]
        if tuple(adjacent_location) in next_state.xmap.item_location_map:
            item_id = next_state.xmap.item_location_map[tuple(adjacent_location)][0]
            self.language += ' ' + next_state.xmap.items[item_id].class_name
        elif xworld_utils.is_out_border(adjacent_location, width, height):
            self.language += ' ' + 'border'
        else:
            self.language += ' ' + 'empty'

    def assign_reward(self, agent, state, action, next_state, num_step):
        self.update_reward(agent, state, action, next_state, num_step)
        self.reward = 0.0
        for reward_name, reward in self.rewards.iteritems():
            self.reward += reward

    def update_reward(self, agent, state, action, next_state, num_step):
        """
        Update teacher reward to agent, this is task dependent
        """
        self.update_step_reward(agent, state, action, next_state, num_step)
        self.update_out_border_reward(agent, state, action, next_state, num_step)
        self.update_knock_block_reward(agent, state, action, next_state, num_step)

    def update_step_reward(self, agent, state, action, next_state, num_step):
        """
        The agent pays negative reward at every step
        """
        self.rewards['step'] = -0.1

    def update_out_border_reward(self, agent, state, action, next_state, num_step):
        self.rewards['out_border'] = 0.0
        agent_id = state.xmap.item_name_map[agent.name]
        agent_location = state.xmap.items[agent_id].location
        width, height = state.xmap.dim['width'], state.xmap.dim['height']
        if agent_location[0] == 1 and agent.velocity[action][0] == -1:
            self.rewards['out_border'] = -0.5
        elif agent_location[1] == 1 and agent.velocity[action][1] == -1:
            self.rewards['out_border'] = -0.5
        elif agent_location[0] == width and agent.velocity[action][0] == 1:
            self.rewards['out_border'] = -0.5
        elif agent_location[1] == height and agent.velocity[action][1] == 1:
            self.rewards['out_border'] = -0.5

    def update_knock_block_reward(self, agent, state, action, next_state, num_step):
        self.rewards['knock_block'] = 0.0
        agent_id = state.xmap.item_name_map[agent.name]
        agent_location = state.xmap.items[agent_id].get_next_location(agent.velocity[action])
        if tuple(agent_location) in state.xmap.item_location_map:
            item_id = state.xmap.item_location_map[tuple(agent_location)][0]
            if state.xmap.items[item_id].item_type == 'block':
                self.rewards['knock_block'] = -0.5

    def accumulate(self):
        if len(self.command) > 0:
            if len(self.cumulative_command) == 0:
                self.cumulative_command.append(self.command)
            elif not self.command == self.cumulative_command[-1]:
                self.cumulative_command.append(self.command)
        self.cumulative_language.append(self.language)
        self.cumulative_reward += self.cumulative_discount * self.reward
        self.cumulative_discount *= self.args.discount_factor


def main():
    logging.info("test xworld teacher functions")
    args = xworld_args.parser().parse_args()
    xworld_teacher = XWorldTeacher(args)
    logging.info("test world teacher functions done")

if __name__ == '__main__':
    main()
