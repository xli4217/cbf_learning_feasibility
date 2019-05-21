import os
import rospy
import numpy as np
import sys
import time
from future.utils import viewitems

from std_msgs.msg import String
from robot_cooking.msg import DictArray

default_config = {
    'init_node': True,
    'robot': 'jaco',
    'RobotCooking': {
        'type': None,
        'config': None
    }
}


class RunRobot(object):

    def __init__(self, config={}):
        self.RunRobot_config = default_config
        self.RunRobot_config.update(config)

        if self.RunRobot_config['init_node']:
            rospy.init_node(self.RunRobot_config['robot']+"_run_robot", anonymous=True)


    def create_publishers_and_subscribers(self):
        pass

    def run(self):
        pass


if __name__ == "__main__":
    from execution.execution_config import ExecutionConfig

    robot_name = str(sys.argv[1])

    exe_config = ExecutionConfig({'robot': robot_name, 'init_node': False})
    automata_cls_type, automata_cls_config = exe_config.low_level_tl_skill_config()

    #### init class ####
    cls_config = default_config
    cls_config['Automata']['type'] = automata_cls_type
    cls_config['Automata']['config'] = automata_cls_config

    cls = RunAutomata(config=cls_config)

    #### run ####
    cls.run()