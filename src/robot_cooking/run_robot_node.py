#!/usr/bin/env python

import os
import rospy
import numpy as np
import sys
import time
import ast
from future.utils import viewitems

from std_msgs.msg import String
from robot_cooking_msgs.msg import DictArray
from diagnostic_msgs.msg import KeyValue

default_config = {
    'init_node': True,
    'robot': 'jaco',
    'dry_run': True,
    'RobotCooking': {
        'type': None,
        'config': None
    }
}


class RunRobotNode(object):

    def __init__(self, config={}):
        self.RunRobotNode_config = default_config
        self.RunRobotNode_config.update(config)

        self.robot = self.RunRobotNode_config['robot']

        if self.RunRobotNode_config['init_node']:
            rospy.init_node(self.RunRobotNode_config['robot']+"_run_robot", anonymous=True)

        self.robot_cooking = self.RunRobotNode_config['RobotCooking']['type'](self.RunRobotNode_config['RobotCooking']['config'])

        self.create_publishers_and_subscribers()
        
    def create_publishers_and_subscribers(self):
        #### publishers ####
        self.skill_arg_pub = rospy.Publisher(self.robot+"_skill_arg", DictArray, queue_size=10)

        #### subscribers ####
        self.skill_to_execute_sub = rospy.Subscriber("automata_output", String, self.cb_skill_to_execute_sub, queue_size=1)

        
    def cb_skill_to_execute_sub(self, msg):
        ee_goal, gripper_action, other_action, done = msg.data.split("-")
        # print("ee_goal:{}".format(ee_goal))
        # print("gripper_action:{}".format(gripper_action))
        # print("other_action:{}".format(other_action))
        # print("----------")
        if ee_goal == "None":
            ee_goal = None
        if gripper_action == "None":
            gripper_action = None
        if other_action == "None":
            other_action = None

        t1 = time.time()
        if done != 'True':    
            self.robot_cooking.execute_automata_output(ee_goal=ee_goal,
                                                       gripper_action=gripper_action,
                                                       other_action=other_action,
                                                       dry_run=self.RunRobotNode_config['dry_run'])
        t2 = time.time()
        # print("dt:{}".format(str(t2-t1)))
        
    def update_and_publish_skill_arg(self):
        self.robot_cooking.update_skill_arg()
        skill_arg_dict_array = DictArray()
        for k, v in viewitems(self.robot_cooking.skill_arg):
            kv = KeyValue()
            kv.key = k

            if isinstance(v, np.ndarray):
                v_processed = list(v)
            elif isinstance(v, dict):
                v_processed = {}
                for k_, v_ in viewitems(v):
                    if isinstance(v_, np.ndarray):
                        v_ = list(v_)
                    v_processed[k_] = v_
            else:
                v_processed = v

            kv.value = str(v_processed)
            skill_arg_dict_array.data.append(kv)
            
        self.skill_arg_pub.publish(skill_arg_dict_array)
            
    def run(self):
        i = 0
        while not rospy.is_shutdown():
            self.update_and_publish_skill_arg()

if __name__ == "__main__":
    from execution.execution_config import ExecutionConfig
    from execution.run_robot_cooking import RunRobotCooking
    
    robot_name = str(sys.argv[1])
    dry_run = str(sys.argv[2])
    if dry_run == 'true':
        dry_run = 'True'
    else:
        dry_run = 'False'
    
    exe_config = ExecutionConfig({'robot': robot_name})
    run_robot_cooking_config = exe_config.run_robot_cooking_config(mode='real', robot=robot_name, init_node=False, with_ll_tl_skill=False)

    #### init class ####
    cls_config = default_config
    cls_config['dry_run'] = dry_run
    cls_config['RobotCooking']['type'] = RunRobotCooking
    cls_config['RobotCooking']['config'] = run_robot_cooking_config

    cls = RunRobotNode(config=cls_config)

    #### run ####
    cls.run()