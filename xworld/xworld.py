import numpy
import copy
import matplotlib.pyplot as plt
import cv2
from . import xworld_agent, xworld_state, xworld_teacher
import time
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorld(object):
    """
    XWorld interface for xworld robot learning
    """
    def __init__(self, args):
        self.args = args
        self.state = xworld_state.XWorldState(args)
        self.agent = xworld_agent.XWorldAgent(args)
        self.teacher = xworld_teacher.XWorldTeacher(args)
        self.num_step = 0

    def seed(self, seed=None):
        self.state.seed(seed)
        self.agent.seed(seed)
        self.teacher.seed(seed)

    def reset(self):
        self.state.reset(self.agent)
        self.teacher.reset(self.state)
        self.num_step = 0
        if self.args.show_frame:
            self.display()
        return self.state, self.teacher

    def step(self, action):
        current_state = copy.deepcopy(self.state)
        action = self.agent.action_type[action]
        self.state.step(self.agent, action)
        self.teacher.step(self.agent, current_state, action, self.state, self.num_step)
        self.num_step += 1
        if self.args.show_frame:
            self.display()
        return self.state, self.teacher, self.teacher.done

    def display_cv(self):
        """
        Display xworld state as well as teacher's command, language and rewards
        """
        # cv2.imshow('image', self.state.image)
        # cv2.resize(self.state.image, (50,50))
        pass

    def display(self):
        """
        Display xworld state as well as teacher's command, language and rewards
        """
        plt.ion()
        plt.figure(1)
        start1 = time.time()
        plt.clf()
        start2 = time.time()
        # print('clf time: %.3f' % (start2 - start1))
        # plot image
        plt.subplot(2, 3, 1)
        plt.imshow(self.state.image)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plot a big green circle if game done
        if self.teacher.done:
            center_x = self.state.image.shape[1] / 2
            center_y = self.state.image.shape[0] / 2
            radius = min(center_x, center_y) / 2
            line_width = radius/10
            circle = plt.Circle((center_y, center_x), radius=radius, color='g', fill=False,
                                linewidth=line_width)
            plt.gca().add_patch(circle)
        plt.axis('scaled')
        start3 = time.time()
        # print('subplot 1 time: %.3f' % (start3 - start2))
        # plot inner state
        plt.subplot(2, 3, 4)
        plt.table(cellText=self.state.inner_state, bbox=[0, 0, 1, 1])
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.axis('off')
        start4 = time.time()
        # print('subplot inner state time: %.3f' % (start4 - start3))
        # plot complete image
        plt.subplot(2, 3, 2)
        plt.imshow(self.state.origin_image)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        start5 = time.time()
        # print('subplot complete image time: %.3f' % (start5 - start4))
        # plot complete inner state
        plt.subplot(2, 3, 5)
        plt.table(cellText=self.state.origin_inner_state, bbox=[0, 0, 1, 1])
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.axis('off')
        start6 = time.time()
        # print('subplot complete inner state time: %.3f' % (start6 - start5))
        # plot teacher's command, language and reward
        plt.subplot(2, 3, 3)
        teacher_command = 'Teacher command:\n    %s\n' % self.teacher.command
        teacher_language = 'Teacher sentence:\n    %s\n' % self.teacher.language
        teacher_reward = 'Teacher reward: %.2f' % self.teacher.reward
        plt.text(0, 0, teacher_command + teacher_language + teacher_reward)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.axis('off')
        start7 = time.time()
        # print('subplot teacher command time: %.3f' % (start7 - start6))
        # plot historical command, language, reward
        plt.subplot(2, 3, 6)
        cumulative_command = 'Cumulative commands:\n'
        start_idx = max(0, len(self.teacher.cumulative_command) - 3)  # only show 3 history commands
        for i in range(start_idx, len(self.teacher.cumulative_command)):
            cumulative_command += '    ' + self.teacher.cumulative_command[i] + '\n'
        cumulative_language = 'Last 5 sentences:\n'
        start_idx = max(0, len(self.teacher.cumulative_language) - 5)  # only show 5 history language
        for i in range(start_idx, len(self.teacher.cumulative_language)):
            cumulative_language += '    ' + self.teacher.cumulative_language[i] + '\n'
        cumulative_reward = 'Cumulative Reward: %.2f' % self.teacher.cumulative_reward
        plt.text(0, 0, cumulative_command + cumulative_language + cumulative_reward)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.axis('off')
        start8 = time.time()
        # print('subplot history command time: %.3f' % (start8 - start7))
        # plot history language
        plt.show()
        plt.pause(0.01)
        if self.args.pause_screen:
            # plt.ioff()
            input("PRESS ANY KEY TO CONTINUE.")
        start9 = time.time()
        # print('show time: %.3f' % (start9 - start8))
