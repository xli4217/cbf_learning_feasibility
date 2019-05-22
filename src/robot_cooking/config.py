import os
import numpy as np
import sys
from future.utils import viewitems

from kinova_api.kinova_driver_utils import KinovaDriverUtils
from baxter_api.baxter_driver_utils import BaxterDriverUtils

from kinova_api.moveit_utils import MoveitUtils



default_config = {
    'robot': 'jaco'
}

class RobotCookingInterfaceConfig(object):

    def __init__(self, config={}):
        self.RobotCookingInterfaceConfig_config = default_config
        self.RobotCookingInterfaceConfig_config.update(config)

        self.robot = self.RobotCookingInterfaceConfig_config['robot']


    def get_robot_cooking_interface_config(self):
        if self.robot == 'baxter':
            damping = 0.2
            natural_freq = 0.2
            
            driver_utils = {
                'type': BaxterDriverUtils,
                'config': {
                    "arm": 'right',
                    "cmd_publish_rate": 100,
                    'init_node': False,
                    'joint_vel_bound': {
                        'upper': 0.25 * np.ones(7),
                        'lower': -0.25 * np.ones(7),    
                    },
                    'safe_workspace': {
                        # safe zone defined here takes precedence
                        'env_json_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env', 'config', 'baxter_env_config.json'),
                        'init_pose': [0.72, -0.56, -0.23, 0, 0, 0, 1],
                        'scale': [1.3, 1.22, 0.6]
                    }
                }
            }

            
            moveit_utils = {
                'type': MoveitUtils,
                'config':  {
                    "init_node": False,
                    "arm_move_group": 'right_arm',
                    "gripper_move_group": None,
                    "reference_link": "base",
                    "end_effector_link": "right_gripper",
                    "neutral_joint_positions": [ 0.21859226227367146, -1.2686021115812371, -0.03681553890924993, 1.366009891611961, 0.10507768397015084, 1.2973642513540886, -0.3658544179106712],
                    "joint_names": ["right_s0","right_s1","right_e0","right_e1","right_w0","right_w1","right_w2"],
                    "env_json_path": "",
                }
            }
            
        elif self.robot == 'jaco':
            damping = 1.5
            natural_freq = 2.1


            driver_utils =  {
                'type': KinovaDriverUtils,
                'config':  {
                    'init_node': False,
                    'cmd_publish_rate': 100,
                    "gripper_move_group": 'gripper',
                    "reference_link": "j2s7s300_link_base",
                    'joint_vel_bound': {
                        'upper': 4 * np.array([2.5,2.5,2.5,2.5,5.,5.,5.]),
                        'lower': -4 * np.array([2.5,2.5,2.5,2.5,5.,5.,5.]),    
                    },
                    'safe_workspace': {
                        # safe zone defined here takes precedence
                        'env_json_path': os.path.join(os.path.dirname(os.path.abspath(__file__))
                                                      , 'env', 'config', 'jaco_env_config.json'),
                        'init_pose': [0.72, -0.56, -0.23, 0, 0, 0, 1],
                        'scale': [1.3, 1.22, 0.6]
                    }
                }
            }

            moveit_utils = {
                'type': MoveitUtils,
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
        
        else:
            raise ValueError('robot not supported')
        
        robot_cooking_interface_config = {
            "robot": self.robot,
            "rate": 10,
            "init_node": False,
            "damping": damping,
            "natural_freq": natural_freq,
            'DriverUtils': driver_utils,
            'MoveitUtils': moveit_utils
        }

        return robot_cooking_interface_config