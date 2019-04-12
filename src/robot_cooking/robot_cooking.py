import sys
import rospy
import numpy as np
import copy
import os
import time

from kinova_api.kinova_driver_utils import KinovaDriverUtils
from kinova_api.kinova_moveit_utils import KinovaMoveitUtils


default_config = {
    'rate': 10,    
    'DriverUtils': {
        'type': KinovaDriverUtils,
        'config':  {
            'init_node': False,
            'cmd_publish_rate': 100,
            "gripper_move_group": 'gripper',
            "reference_link": "j2s7s300_link_base",
            'joint_vel_bound': {
                'upper': 1.5 * np.ones(7),
                'lower': -1.5 * np.ones(7),    
            },
            'safe_workspace': {
                # safe zone defined here takes precedence
                'env_json_path': os.path.join(os.path.dirname(os.path.abspath(__file__))
, 'env', 'config', 'env_config.json'),
                'x': [-0.1, 0.6],
                'y': [-0.25, 0.16],
                'z': [0.08 ,0.8]
            }
        }
    },
    'MoveitUtils': {
        'type': KinovaMoveitUtils,
        'config':  {
            "init_node": False,
            "arm_move_group": 'arm',
            "gripper_move_group": 'gripper',
            "reference_link": "j2s7s300_link_base",
            "end_effector_link": "j2s7s300_end_effector",
            "neutral_joint_positions": [4.935252420414662, 2.485132499632967, -0.6278464188357148, 0.763206574666036, 4.204106310531545, 4.064730307514466, 4.778262802580017, 1.1112613559570534, 1.108797399316512, 1.108797399316512],
            "joint_names": ["j2s7s300_joint_1", "j2s7s300_joint_2", "j2s7s300_joint_3", "j2s7s300_joint_4", "j2s7s300_joint_5", "j2s7s300_joint_6", "j2s7s300_joint_7"],
            "env_json_path": "",
        }
    }
}

class RobotCooking(object):

    def __init__(self, config={}):
        self.RobotCooking_config = default_config
        self.RobotCooking_config.update(config)

        rospy.init_node("robot_cooking", anonymous=False)
        rospy.on_shutdown(self.cleanup)
        
        self.driver_utils = self.RobotCooking_config['DriverUtils']['type'](self.RobotCooking_config['DriverUtils']['config'])
        self.moveit_utils = self.RobotCooking_config['MoveitUtils']['type'](self.RobotCooking_config['MoveitUtils']['config'])

        self.rate = rospy.Rate(self.RobotCooking_config['rate'])
        
    def servo_to_pose_target(self, pt):
        assert len(pt) == 7
        # calculate the joint positions using ik
        ik_sol = self.moveit_utils.ik(position=pt[:-4], orientation=pt[-4:])
        ik_jp = ik_sol[:-3]

        damping = 1.2
        natural_freq = 0.9
        kp = (2 * np.pi * natural_freq) ** 2
        kd = 2 * damping * 2 * np.pi * natural_freq

        joint_angle_diff = ik_jp - self.driver_utils.get_joint_values()
        joint_vel = self.driver_utils.get_joint_velocities()

        jv = kp * joint_angle_diff + kd * joint_vel
        jv = 0.5 * jv


        ee_pos, ee_quat = self.driver_utils.get_ee_pose()
        pose_distance = np.linalg.norm(ee_pos - pt[:3])
        
        if not self.driver_utils.is_tool_in_safe_workspace():
            print('tool not in safe workspace')

        # start servoing
        if pose_distance > 0.01 and self.driver_utils.is_tool_in_safe_workspace():
            self.driver_utils.pub_joint_velocity(jv, duration_sec=1./self.RobotCooking_config['rate'])
            return False
        else:
            return True
            
    def run(self):
        # i = 0
        # for i in range(1000):
        #     ee_pos, ee_quat = cls.driver_utils.get_ee_pose()
        #     pt = np.array(ee_pos) + np.array([0.05 * np.sin(0.1 * i), 0, 0])
        #     pt = np.concatenate([pt, ee_quat])
        #     self.servo_to_pose_target(pt)
        #     self.rate.sleep()

        # ee_pos, ee_quat = cls.driver_utils.get_ee_pose()
        # pt = np.array(ee_pos) + np.array([0.1, 0, 0])
        # pt = np.concatenate([pt, ee_quat])
        # print(ee_pos)
        # print(pt)
        # i = 0
        # for i in range(1000):
        #     if self.servo_to_pose_target(pt):
        #         print('reached goal target')
        #         break
        #     self.rate.sleep()

        for _ in range(1000):
            print(self.driver_utils.is_tool_in_safe_workspace())
            
    def cleanup(self):
        pass
        
        
if __name__ == "__main__":
    cls = RobotCooking()
    time.sleep(0.5)
    
    #### test IK ####
    # curr_ee_pos, curr_ee_quat = cls.driver_utils.get_ee_pose()
    # curr_jp = cls.driver_utils.get_joint_values()
    # ik_sol = cls.moveit_utils.ik(position=curr_ee_pos, orientation=curr_ee_quat)
    # # for kinova
    # ik_jp = ik_sol[:-3]
    # ik_fp = ik_sol[-3:]
    # print(curr_jp)
    # print(ik_jp)

    #### test servo to target pose ####
    cls.run()