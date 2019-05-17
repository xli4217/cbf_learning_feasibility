import os
import numpy as np
import time
from future.utils import viewitems

from utils.utils import *

default_config = {

}

class ExecutionConfig(object):

    def __init__(self, config={}):
        self.ExecutionConfig_config = default_config
        self.ExecutionConfig_config.update(default_config)

    def simulation_config(self):
        from cooking_env.env.base.ce import CookingEnv
        
        sim_config = {
            # specific to this env
            "suffix": "",
            "particle_test": True,
            "arm": "jaco",
            "control_mode": "velocity"
        }

        return CookingEnv, sim_config

    def kinova_config(self):
        from robot_cooking.robot_cooking import RobotCooking
        from kinova_api.kinova_driver_utils import KinovaDriverUtils
        from kinova_api.kinova_moveit_utils import KinovaMoveitUtils

        
        config = {
            "rate": 10,
            "policy_info": {
                "state_space": None,
                "action_space": None,
                "training_config_restore_path": None,
                "policy_restore_path": None,
                "state_preprocessor_restore_path": None
            },
            "WPGenerator": {
                'type': None,
                'config': {}
            },
            'DriverUtils': {
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
                        'env_json_path': os.path.join(os.environ['ROBOT_COOKING_ROS_PATH'], 'robot_cooking', 'env', 'config', 'env_config.json'),
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
        
        return RobotCooking, config
        
    def motor_skill_config(self):
        from skills.motor_skills import MotorSkills

        #### Set up trajectory generator #####
        from traj_generators.trajectory_generator import TrajectoryGenerator
        skills_config = {}
        skills_config['TrajGenerator'] = {
            'type': TrajectoryGenerator,
            'config': {
                'dmp_config': {
                    # gain on attractor term y dynamics (linear)
                    'ay': 55,
                    # gain on attractor term y dynamics (linear)
                    'by': None,
                    # gain on attractor term y dynamics (angular)
                    'az': 55,
                    # gain on attractor term y dynamics (angular)
                    'bz': None,
                    # timestep
                    'dt': 0.001,
                    # time scaling, increase tau to make the system execute faster
                    'tau': 1.,
                    'use_canonical': False,
                    # for canonical
                    'apx': 1.,
                    'gamma': 0.3,
                    # for faster convergence
                    'app': 0.5,
                    'apr': 0.5,
                    # for integrating goal
                    'ag': 3.0,
                    'ago': 3.0,
                    # if True, then update according to dmp_pose, else update according to current pose
                    'use_dmp_pose': True,
                    'n_linear_dmp': 3,
                    'n_angular_dmp': 3
                    
                },
                'clf_cbf_config': {
                    'k_cbf': 1.1,
                    'epsilon':0.8,
                    'num_states':3,
                    'action_space': {'shape': (3,), 'upper_bound': 0.2 * np.ones(3), 'lower_bound': -0.2 * np.ones(3)},
                    'use_own_pose': True,
                    'dt': 0.015,
                    'log_dir': os.path.join(os.environ['LEARNING_PATH'], 'execution', 'log')
                },
                'translation_gen': 'clf_cbf',
                # 'translation_gen': 'dmp',
                'orientation_gen': 'dmp'            
            }
        }
        
        #### Load learned skills #####
        
        ## flip switch open 
        experiment_root_dir = os.path.join(os.environ['LEARNING_PATH'], 'learning', 'experiments')
        experiment_name = 'test'
        hyperparam_dir = 'seed0'
        itr = 100
        
        learned_skill_config = {
            "state_space": {'type': 'float', 'shape': (3, ), "upper_bound": [], 'lower_bound': []},
            "action_space": {'type': 'float', 'shape': (3, ), "upper_bound": 70*np.ones(3), 'lower_bound': -70*np.ones(3)},
            "training_config_restore_path": os.path.join(experiment_root_dir, experiment_name, 'config', hyperparam_dir, 'config.pkl'),
            "policy_restore_path": os.path.join(experiment_root_dir, experiment_name, 'transitions', hyperparam_dir, 'itr_'+str(itr)),
            "state_preprocessor_restore_path": os.path.join(experiment_root_dir, experiment_name, 'info', hyperparam_dir, 'state_preprocessor_params.pkl')
        }
        skills_config['LearnedSkills'] = {'flipswitchon': learned_skill_config}

        return MotorSkills, skills_config

    def low_level_tl_skill_config(self):
        from skills.low_level_tl_skills import LowLevelTLSkills
        from tl_utils.tl_config import KEY_POSITIONS, OBJECT_RELATIVE_POSE, STATE_IDX_MAP, PREDICATES

        pick_hotdog = "F((moveto_hotdogplate && opengripper) && X F (closegripper))"
        place_bun = "F((moveto_bunplate && closegripper)  && X F (opengripper))"

        pick_sausage_place_grill = "F ((moveto_hotdogplate && opengripper) && X F (closegripper && XF ((moveto_grill && closegripper)  && X F (opengripper))))"

        apply_condiment = "(moveto_condiment_condimentpre && opengripper) && X F " + \
                          "(moveto_condiment_condimentpost && X F " + \
                          "(closegripper && X F "+ \
                          "((moveto_bunplate_relativeplateapplycondimentpre && closegripper) && X F "+\
                          "(opengripper))))"
        
        entire_task = "(moveto_hotdogplate && opengripper) && X F " + \
                      "(closegripper && X F " + \
                      "((moveto_grill && closegripper) && X F " + \
                      "(opengripper && X F "+ \
                      "(closegripper && X F "+\
                      "((moveto_bunplate && closegripper) && X F "+\
                      "(opengripper))))))"

        test_spec = 'flipswitchon'
        
        config = {
            'make_hotdog': {
                'formula':"F (" + test_spec + ")",
                'visdom': False,
                'key_positions': KEY_POSITIONS,
                'object_relative_pose': OBJECT_RELATIVE_POSE,
                'state_idx_map': STATE_IDX_MAP,
                'predicate_robustness': PREDICATES,
                'fsa_save_dir': os.path.join(os.environ['LEARNING_PATH'], 'execution', 'figures'),
                'dot_file_name': 'make_hotdog',
                'svg_file_name': 'make_hotdog',
                'mdp_state_space': {'type': 'float', 'shape': (37, ), 'upper_bound':[], 'lower_bound': []}
            }
        }

        return LowLevelTLSkills, config


