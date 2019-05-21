import os
import rospy
import numpy as np
import sys
import time
from future.utils import viewitems

from std_msgs.msg import String

default_config = {
    'init_node': True,
    'Automata':{
        'type': None,
        'config': None
    }
}


class RunAutomata(object):
    '''
    The run automata node (currently only support one automata, LowLevelTLSkills supports multiple)
    '''
    def __init__(self, config={}):
        self.RunAutomata_config = default_config
        self.RunAutomata_config.update(config)

        if self.RunAutomata_config['init_node']:
            rospy.init_node("automata", anonymous=True)

        #### create pub and sub ####
        self.create_publishers_and_subscribers()
            
    def create_publishers_and_subscribers(self):
        #### publishers ####

        ## this is in the form "eegoal_gripperaction_otheraction"
        self.skill_pub = rospy.Publisher('automata_output', String, queue_size=10)
        
        #### subscribers ####
        
    def run(self):
        while not rospy.is_shutdown():
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