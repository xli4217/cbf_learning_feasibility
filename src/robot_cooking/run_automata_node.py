#!/usr/bin/env python

import os
import rospy
import numpy as np
import sys
import time
from future.utils import viewitems
import copy

from std_msgs.msg import String
import ast
from robot_cooking_msgs.msg import DictArray

default_config = {
    'init_node': True,
    'Automata':{
        'type': None,
        'config': None
    }
}


class RunAutomataNode(object):
    '''
    The run automata node (currently only support one automata, LowLevelTLSkills supports multiple)
    '''
    def __init__(self, config={}):
        self.RunAutomataNode_config = default_config
        self.RunAutomataNode_config.update(config)

        if self.RunAutomataNode_config['init_node']:
            rospy.init_node("automata", anonymous=True)

        #### create pub and sub ####
        self.create_publishers_and_subscribers()

        #### initialize field var ####
        self.robot_skill_args = {}

        #### initialize automata ####
        self.aut = self.RunAutomataNode_config['Automata']['type'](self.RunAutomataNode_config['Automata']['config'])
        
    def create_publishers_and_subscribers(self):
        #### publishers ####

        ## this is in the form "eegoal-gripperaction-otheraction"
        self.skill_pub = rospy.Publisher('automata_output', String, queue_size=10)
        
        #### subscribers ####
        # self.skill_arg_subs = {}
        # for robot in ['jaco', 'baxter']:
        #     self.skill_arg_subs[robot] = rospy.Subscriber(robot+"_skill_arg", DictArray, self.cb_skill_arg_subs, (robot))
        rospy.Subscriber("jaco_skill_arg", DictArray, self.cb_skill_arg_subs, queue_size=1)
        
    def cb_skill_arg_subs(self, msg):
        converted_dict = {}
        for kv in msg.data:
            key = kv.key
            # if key == 'curr_pose':
            #     print(kv.value)
            v = ast.literal_eval(kv.value)
            converted_dict[key] = v
        self.robot_skill_args['jaco'] = converted_dict

        skill_args = copy.copy(self.robot_skill_args['jaco'])
        action_n_constraint, done = self.aut.step(skill_args)

        ee_goal = action_n_constraint['make_hotdog']['node_action']['ee_goal']
        gripper_action = action_n_constraint['make_hotdog']['node_action']['gripper_action']
        other_action = action_n_constraint['make_hotdog']['node_action']['other_action']

        automata_output = str(ee_goal) + "-" + str(gripper_action) + "-" + str(other_action) + "-" + str(done)
        # automata_output = 'moveto_condiment_condimentpre-opengripper-None-False'
        self.skill_pub.publish(automata_output)
        
    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == "__main__":
    from execution.execution_config import ExecutionConfig

    exe_config = ExecutionConfig({'robot': None, 'init_node': False})
    automata_cls_type, automata_cls_config = exe_config.low_level_tl_skill_config()

    #### init class ####
    cls_config = default_config
    cls_config['Automata']['type'] = automata_cls_type
    cls_config['Automata']['config'] = automata_cls_config

    cls = RunAutomataNode(config=cls_config)

    #### run ####
    cls.run()