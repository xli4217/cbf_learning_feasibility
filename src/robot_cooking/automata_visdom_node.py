#!/usr/bin/env python

import os
import rospy
import numpy as np
import sys
import time
import ast
from future.utils import viewitems
from tl_utils.plot_dynamic_automata import PlotDynamicAutomata
from lomap.classes import Fsa
from std_msgs.msg import String
import ast

default_config = {
    'spec': ""
}

class AutomataVisdomNode(object):

    def __init__(self, config={}):
        self.AutomataVisdomNode_config = default_config
        self.AutomataVisdomNode_config.update(config)

        rospy.init_node('automata_visdom_node', anonymous=True)
        
        self.fsa = Fsa()
        self.fsa.from_formula(self.AutomataVisdomNode_config['spec'])
        self.fsa.add_trap_state()

        self.plot_aut = PlotDynamicAutomata(self.fsa)
        self.Q = None
        self.edge = None
        
        self.create_publishers_and_subscribers()
        
    def create_publishers_and_subscribers(self):
        #### pub ####

        #### sub ####
        rospy.Subscriber('automata_node_edge', String, self.cb_node_edge)
        

    def cb_node_edge(self, msg):
        processed_msg = msg.data.split('-')

        self.Q = processed_msg[0]
        self.edge = ast.literal_eval(processed_msg[1])
        
    def run(self):
        while not rospy.is_shutdown():
            if self.Q is not None and self.edge is not None:
                self.plot_aut.update(current_state=self.Q, src_and_dest=self.edge)
            
            
if __name__ == "__main__":
    from execution.execution_config import ExecutionConfig

    robot_name = str(sys.argv[1])
    
    exe_config = ExecutionConfig({'robot': robot_name})

    _, spec_config = exe_config.low_level_tl_skill_config()
    
    config = default_config
    config['spec'] = spec_config['make_hotdog']['formula']

    cls = AutomataVisdomNode(config=config)

    cls.run()