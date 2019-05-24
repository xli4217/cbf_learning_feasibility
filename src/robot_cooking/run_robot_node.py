#!/usr/bin/env python

import os
import rospy
import numpy as np
import sys
import time
import ast
from future.utils import viewitems
import zmq

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
    },
    'ZMQ':{
        'pub': {
            'skill_args': None
        },
        'sub': {
            'aut_output': None
        }
    }
}


class RunRobotNode(object):

    def __init__(self, config={}):
        self.RunRobotNode_config = default_config
        self.RunRobotNode_config.update(config)

        self.robot = self.RunRobotNode_config['robot']
        
        if self.RunRobotNode_config['init_node']:
            rospy.init_node(self.RunRobotNode_config['robot']+"_run_robot", anonymous=False)

        self.robot_cooking = self.RunRobotNode_config['RobotCooking']['type'](self.RunRobotNode_config['RobotCooking']['config'])

        self.create_publishers_and_subscribers()

        self.ros_ee_goal = None
        self.ros_gripper_action = None
        self.ros_other_action = None
        self.ros_done = 'False'

        self.zmq_ee_goal = None
        self.zmq_gripper_action = None
        self.zmq_other_action = None
        self.zmq_done = 'False'
      
        
    def create_publishers_and_subscribers(self):
        #### publishers ####
        self.skill_arg_pub = rospy.Publisher(self.robot+"_skill_arg", DictArray, queue_size=1)

        #### subscribers ####
        self.skill_to_execute_sub = rospy.Subscriber("automata_output", String, self.cb_skill_to_execute_sub, queue_size=1)

        self.zmq_pub = None
        self.zmq_pub = None
        if self.RunRobotNode_config['ZMQ'] is not None:
            self.context = zmq.Context()
            #### zmq pubs ####
            self.zmq_pub = self.context.socket(zmq.PUB)
            self.zmq_pub.bind("tcp://*:%s" %self.RunRobotNode_config['ZMQ']['pub']['skill_args'])
            
            #### zmp subs ####
            self.zmq_sub = self.context.socket(zmq.SUB)
            self.zmq_sub.connect("tcp://burobotics:%s" %self.RunRobotNode_config['ZMQ']['sub']['aut_output'])
            self.zmq_sub.setsockopt(zmq.SUBSCRIBE, "")
            
    def cb_skill_to_execute_sub(self, msg):
        self.ros_ee_goal, self.ros_gripper_action, self.ros_other_action, self.ros_done = msg.data.split("-")
        # print("ee_goal:{}".format(ee_goal))
        # print("gripper_action:{}".format(gripper_action))
        # print("other_action:{}".format(other_action))
        # print("----------")

        if self.ros_ee_goal == "None":
            self.ros_ee_goal = None
        if self.ros_gripper_action == "None":
            self.ros_gripper_action = None
        if self.ros_other_action == "None":
            self.ros_other_action = None

    def update_and_publish_skill_arg(self):
        self.robot_cooking.update_skill_arg()
        skill_arg_dict_array = DictArray()
        skill_arg_dict_processed = {} # transform np.ndarray to list
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

            skill_arg_dict_processed[k] = v_processed
            
        if self.zmq_pub is not None:
            self.zmq_pub.send_json((self.robot, skill_arg_dict_processed))
            
        self.skill_arg_pub.publish(skill_arg_dict_array)
            
    def run(self):
        while not rospy.is_shutdown():
            for _ in range(200):
                self.update_and_publish_skill_arg()
                ## aut skill output through zmq
                if self.zmq_sub is not None:
                    topic, data = self.zmq_sub.recv_string().split()
                    if topic == 'automata_output':
                        aut_output = data
                        self.zmq_ee_goal, self.zmq_gripper_action, self.zmq_other_action, self.zmq_done = aut_output.split('-')
                        if self.zmq_ee_goal == "None":
                            self.zmq_ee_goal = None
                        if self.zmq_gripper_action == "None":
                            self.zmq_gripper_action = None
                        if self.zmq_other_action == "None":
                            self.zmq_other_action = None


                if self.RunRobotNode_config['msg_system'] == 'ros':
                    ee_goal = self.ros_ee_goal
                    gripper_action = self.ros_gripper_action
                    other_action = self.ros_other_action
                    done = self.ros_done
                elif self.RunRobotNode_config['msg_system'] == 'zmq':
                    ee_goal = self.zmq_ee_goal
                    gripper_action = self.zmq_gripper_action
                    other_action = self.zmq_other_action
                    done = self.zmq_done
                else:
                    raise ValueError('msg system unsupported')
                    
                    
            # if done != 'True':    
            #     self.robot_cooking.execute_automata_output(ee_goal=ee_goal,
            #                                                gripper_action=gripper_action,
            #                                                other_action=other_action,
            #                                                dry_run=self.RunRobotNode_config['dry_run'])
       
if __name__ == "__main__":
    from execution.execution_config import ExecutionConfig
    from execution.run_robot_cooking import RunRobotCooking
    
    robot_name = str(sys.argv[1])
    dry_run = str(sys.argv[2])
    zmq_pub_port = str(sys.argv[3])
    aut_sub_port = str(sys.argv[4])
    msg_system = str(sys.argv[5])
    
    if dry_run == 'true' or dry_run == 'True':
        dry_run = True
    elif dry_run == 'false' or dry_run == 'False':
        dry_run = False
    else:
        raise ValueError()

    exe_config = ExecutionConfig({'robot': robot_name})
    run_robot_cooking_config = exe_config.run_robot_cooking_config(mode='real', robot=robot_name, init_node=False, with_ll_tl_skill=False)

    #### init class ####
    cls_config = default_config
    cls_config['robot'] = robot_name
    cls_config['dry_run'] = dry_run
    cls_config['RobotCooking']['type'] = RunRobotCooking
    cls_config['RobotCooking']['config'] = run_robot_cooking_config
    cls_config['msg_system'] = msg_system
    
    cls_config['ZMQ'] = {
        'pub': {
            'skill_args': zmq_pub_port
        },
        'sub': {
            'aut_output': aut_sub_port
        }
    }
    
    cls = RunRobotNode(config=cls_config)

    #### run ####
    cls.run()