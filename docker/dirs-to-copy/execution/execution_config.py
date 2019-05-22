import os
import numpy as np
import time
from future.utils import viewitems

from utils.utils import *

default_config = {
    'robot': 'jaco',
}

class ExecutionConfig(object):

    def __init__(self, config={}):
        self.ExecutionConfig_config = default_config
        self.ExecutionConfig_config.update(config)

        self.robot = self.ExecutionConfig_config['robot']
        
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

    def robot_config(self, init_node=False):
        from robot_cooking.robot_cooking_interface import RobotCookingInterface
        from robot_cooking.config import RobotCookingInterfaceConfig
        
        config = RobotCookingInterfaceConfig(config={'robot': self.robot}).get_robot_cooking_interface_config()
        config['init_node'] = init_node
        
        return RobotCookingInterface, config
        
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
        experiment_name = 'switchon'
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

        apply_condiment_ = "(moveto_condiment_condimentpre && opengripper) && X F " + \
                           "(moveto_condiment_condimentpost && X F " + \
                           "(closegripper && X F "+ \
                           "((moveto_bunplate_relativeplateapplycondimentpost && closegripper) && X F "+\
                           "(applycondiment && X F" + \
                           "((moveto_world_placecondimentgoal && closegripper) && X F" + \
                           "(opengripper && X F" + \
                           "(moveto_condiment_condimentpre" + \
                           ")))))))"
        
        entire_task = "(moveto_hotdogplate && opengripper) && X F " + \
                      "(closegripper && X F " + \
                      "((moveto_grill && closegripper) && X F " + \
                      "(opengripper && X F "+ \
                      "(closegripper && X F "+\
                      "((moveto_bunplate && closegripper) && X F "+\
                      "(opengripper" + \
                      "))))))"

        test_spec_ = 'moveto_hotdogplate && X F (moveto_grill)'
        
        config = {
            'make_hotdog': {
                'formula':"F (" + test_spec_ + ")",
                'visdom': False,
                'key_positions': KEY_POSITIONS,
                'object_relative_pose': OBJECT_RELATIVE_POSE,
                'state_idx_map': STATE_IDX_MAP,
                'predicate_robustness': PREDICATES,
                'fsa_save_dir': os.path.join(os.environ['LEARNING_PATH'], 'execution', 'figures'),
                'dot_file_name': 'make_hotdog',
                'svg_file_name': 'make_hotdog',
                'mdp_state_space': {'type': 'float', 'shape': (38, ), 'upper_bound':[], 'lower_bound': []}
            }
        }

        return LowLevelTLSkills, config


    def run_robot_cooking_config(self, mode='sim', robot=None, init_node=False, with_ll_tl_skill=True):
        robot = robot
        if robot is None:
            robot = self.robot
        config = {
            #### this can be 'sim' or 'real' ####
            'mode': mode,
            #### this can be 'jaco' or 'baxter' ####
            'robot': robot,
            'init_node': init_node,
            'SimulationEnv':{
                'type': None,
                'config':{}
            },
            'ExperimentEnv': {
                'type': None,
                'config': {}
            },
            'Skills':{
                'construct_skill_state': None,
                'MotorSkills': {
                    'type': None,
                    'config':{}
                },
                'LowLevelTLSkills': {
                    'type': None,
                    'config': {}
                }
            }
        }

        ####################
        # Setup simulation #
        ####################
        cls_type, cls_config = self.simulation_config()
    
        config['SimulationEnv'] = {
            'type': cls_type,
            'config': cls_config 
        }

        ####################
        # Setup experiment #
        ####################
        cls_type, cls_config = self.robot_config()
        config['ExperimentEnv'] = {
            'type': cls_type,
            'config': cls_config
        }
    
        ################
        # Setup Skills #
        ################
        motor_skill_type, motor_skill_config = self.motor_skill_config()
        low_level_tl_skill_type, low_level_tl_skill_config = self.low_level_tl_skill_config()

        if not with_ll_tl_skill:
            low_level_tl_skill_type = None
            low_level_tl_skill_config = {}
                
        config['Skills'] = {
            'MotorSkills':{
                'type': motor_skill_type,
                'config': motor_skill_config
            },
            'LowLevelTLSkills':{
                'type': low_level_tl_skill_type,
                'config': low_level_tl_skill_config
            }
        }

        return config