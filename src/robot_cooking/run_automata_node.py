#!/usr/bin/env python

import os
import rospy
import numpy as np
import sys
import time
from future.utils import viewitems
import copy
import zmq

from std_msgs.msg import String
import ast
from robot_cooking_msgs.msg import DictArray

default_config = {
    'init_node': True,
    'msg_system': "ros",
    'nb_robot': 1,
    'Automata':{
        'type': None,
        'config': None
    },
    'ZMQ': {
        'pub': {
            'aut_output': None
        },
        'sub': {
            'baxter_skill_arg': None,
            'jaco_skill_arg': None
        }
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
            rospy.init_node("aut_node", anonymous=False)

        #### create pub and sub ####
        self.create_publishers_and_subscribers()

        #### initialize field var ####
        self.robot_skill_args = {}
        self.zmq_robot_skill_args = {}
        
        #### initialize automata ####
        self.aut = self.RunAutomataNode_config['Automata']['type'](self.RunAutomataNode_config['Automata']['config'])
        
    def create_publishers_and_subscribers(self):
        #### publishers ####

        ## this is in the form "eegoal-gripperaction-otheraction"
        self.skill_pub = rospy.Publisher('automata_output', String, queue_size=1)
        self.aut_node_and_edge_pub = rospy.Publisher('automata_node_edge', String, queue_size=1)
        
        #### subscribers ####
        self.skill_arg_subs = {}
        for robot in ['jaco', 'baxter']:
            self.skill_arg_subs[robot] = rospy.Subscriber(robot+"_skill_arg", DictArray, self.cb_skill_arg_subs, (robot))

        #### ZMQ ####
        self.zmq_pub = None
        self.zmq_sub = None
        if self.RunAutomataNode_config['ZMQ'] is not None:
            self.context = zmq.Context()
            #### zmq pubs ####
            self.zmq_pub = self.context.socket(zmq.PUB)
            self.zmq_pub.bind("tcp://*:%s" %self.RunAutomataNode_config['ZMQ']['pub']['aut_output'])
            
            #### zmp subs ####
            self.zmq_sub = self.context.socket(zmq.SUB)
            self.zmq_sub.connect("tcp://burobotics:%s" %self.RunAutomataNode_config['ZMQ']['sub']['baxter_skill_arg'])
            addr = "tcp://burobotics:{}".format(self.RunAutomataNode_config['ZMQ']['sub']['jaco_skill_arg'])
            self.zmq_sub.connect(addr)
            self.zmq_sub.setsockopt(zmq.SUBSCRIBE, "")


    def construct_combined_skill_args(self):
        combined_skill_args = {}
            
    def cb_skill_arg_subs(self, msg, robot):
        converted_dict = {}
        for kv in msg.data:
            key = kv.key
            # if key == 'curr_pose':
            #     print(kv.value)
            v = ast.literal_eval(kv.value)
            converted_dict[key] = v
        self.robot_skill_args[robot] = converted_dict

        
        #### ZMQ ####
        if self.zmq_sub is not None:
            robot_name, robot_skill_args = self.zmq_sub.recv_json()
            self.zmq_robot_skill_args[robot_name] = robot_skill_args
            
    def run(self):
        while not rospy.is_shutdown():
            if self.RunAutomataNode_config['msg_system'] == 'ros':
                robot_skill_args = self.robot_skill_args
            elif self.RunAutomataNode_config['msg_system'] == 'zmq':
                robot_skill_args = self.zmq_robot_skill_args
            else:
                raise ValueError("unsupported msg_system")
                
            if len(self.robot_skill_args.keys()) > 0:
                skill_args = copy.copy(self.robot_skill_args['jaco'])
                action_n_constraint, done = self.aut.step(skill_args)

                if not done:
                    #### pub skills output by automata ####
                    ee_goal = action_n_constraint['make_hotdog']['node_action']['ee_goal']
                    gripper_action = action_n_constraint['make_hotdog']['node_action']['gripper_action']
                    other_action = action_n_constraint['make_hotdog']['node_action']['other_action']
                
                    automata_output = str(ee_goal) + "-" + str(gripper_action) + "-" + str(other_action) + "-" + str(done)            
                else:
                    automata_output = "None-None-None-True"
            else:
                automata_output = "None-None-None-True"
            # automata_output = 'moveto_condiment_condimentpre-opengripper-None-False'
            self.skill_pub.publish(automata_output)

            #### pub node and edges ####
            curr_node_edge = self.aut.curr_node_edge
            curr_node = curr_node_edge['make_hotdog']['node']
            curr_edge = curr_node_edge['make_hotdog']['edge']
            
            node_edge_msg = curr_node + "-" + str(curr_edge)
            
            self.aut_node_and_edge_pub.publish(node_edge_msg)

            #### ZMQ ####
            if self.zmq_pub is not None:
                self.zmq_pub.send_string("%s %s" %("automata_output", automata_output))

            
if __name__ == "__main__":
    from execution.execution_config import ExecutionConfig


    aut_out_pub_port = str(sys.argv[1])
    jaco_sub_port = str(sys.argv[2])
    baxter_sub_port = str(sys.argv[3])
    msg_system = str(sys.argv[4])
    nb_robot = int(sys.argv[5])
    
    exe_config = ExecutionConfig({'robot': None, 'init_node': False})
    automata_cls_type, automata_cls_config = exe_config.low_level_tl_skill_config()

    #### init class ####
    cls_config = default_config
    cls_config['nb_robots'] = nb_robot
    cls_config['msg_system'] = msg_system
    cls_config['Automata']['type'] = automata_cls_type
    cls_config['Automata']['config'] = automata_cls_config
    cls_config['ZMQ'] = {
        'pub': {
            'aut_output': aut_out_pub_port
        },
        'sub': {
            'baxter_skill_arg': baxter_sub_port,
            'jaco_skill_arg': jaco_sub_port
        }
    }

    cls = RunAutomataNode(config=cls_config)

    #### run ####
    cls.run()