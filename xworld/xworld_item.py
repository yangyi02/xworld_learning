import numpy
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class XWorldItem(object):
    """
    XWorld item for xworld robot learning
    """
    def __init__(self, item_type='', index=-1, name='', class_name='', class_id=-1, location=numpy.empty([0]), image_name='', image=numpy.empty([0])):
        """
        self.item_type: goal, block, agent, dummy, box, river, ...
        self.index: item index, starting from 0, unique per item for each item_type
        self.name: item name, unique per item
        self.class_name: apple, avocado, banana, ..., brick, robot, dummy, ...
        self.class_id: unique per class
        self.location: numpy.array([1,2])
        self.image_name: images/agent_robot_1.jpg, ...
        self.image: numpy.array, image read from self.image_name
        self.is_removed: item may be removed during the game
        """
        self.item_type = item_type
        self.index = index
        self.name = name
        self.class_name = class_name
        self.class_id = class_id
        self.location = location
        self.image_name = image_name
        self.image = image
        self.is_removed = False

    def get_next_location(self, velocity):
        """
        Get item's next potential location with its velocity
        If an item cannot move, then it's next location stays the same
        However, the item may or may not update its location due to the environment
        """
        if self.is_movable():
            next_location = self.location + velocity
        else:
            next_location = self.location
        return next_location

    def is_movable(self, next_item_type='empty'):
        """
        Check whether item can move to a new place where there is another item
        """
        if self.item_type == 'agent' and next_item_type == 'empty':
            return True
        elif self.item_type == 'agent' and next_item_type == 'goal':
            return True
        elif self.item_type == 'agent' and next_item_type == 'box':
            return True
        elif self.item_type == 'box' and next_item_type == 'empty':
            return True
        elif self.item_type == 'box' and next_item_type == 'river':
            return True
        elif self.item_type == 'box' and next_item_type == 'target':
            return True
        else:
            return False

    def to_be_removed(self, next_item_type='empty'):
        """
        When two items are on the same location, they may get removed
        """
        if self.item_type == 'goal' and next_item_type == 'agent':
            return True
        elif self.item_type == 'box' and next_item_type == 'river':
            return True
        elif self.item_type == 'river' and next_item_type == 'box':
            return True
        else:
            return False

    def to_be_merged(self, next_item_type='empty'):
        """
        When two items are on the same location, they may get merged
        """
        if self.item_type == 'box' and next_item_type == 'target':
            return True
        else:
            return False
