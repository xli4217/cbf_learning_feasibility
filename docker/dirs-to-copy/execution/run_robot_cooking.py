import os
import numpy as np
import time
from future.utils import viewitems

from utils.utils import *

default_config = {
    #### this can be 'sim' or 'real' ####
    'mode': 'sim',
    'SimulationEnv':{
        'type': None,
        'config':{}
    },
    'ExperimentEnv': {
        'type': None,
        'config': {}
    },
    'Automata': {
        'type': None,
        'config':{}
    },
    'Skills':{
        'type': None,
        'config':{}
    }
}

class RunRobotCooking(object):

    def __init__(self, config={}):
        self.RunRobotCooking_config = default_config
        self.RunRobotCooking_config.update(config)

        if self.RunRobotCooking_config['mode'] == 'sim':
            self.env = self.RunRobotCooking_config['SimulationEnv']['type']( self.RunRobotCooking_config['SimulationEnv']['config'])
        elif self.RunRobotCooking_config['mode'] == 'real':
            self.env = self.RunRobotCooking_config['ExperimentEnv']['type']( self.RunRobotCooking_config['ExperimentEnv']['config'])
        else:
            raise ValueError('unsupported mode')
            
        self.aut = self.RunRobotCooking_config['Automata']['type'](self.RunRobotCooking_config['Automata']['config'])
        
        self.skills = self.RunRobotCooking_config['Skills']['type'](self.RunRobotCooking_config['Skills']['config'])
        
    def run(self):
        pass
        
    def test(self):
        from tl_utils.tl_config import *
        pt = KEY_POSITIONS['neutral']
        self.env.set_goal_pose()
        while True:
            curr_pose = self.env.get_target_pose()
            pos_distance, quat_distance = pose_distance(pt, curr_pose) 

            if pos_distance < 0.01 and quat_distance < 0.15:
                print('reached goal')
                break
                
            skill_arg = {
                'goal': pt,
                'curr_pose': curr_pose,
                'curr_vel': self.env.get_target_velocity(),
                'obs_info': self.env.get_obstacle_info()
            }
            
            action = self.skills.get_action(skill_name='move_to', skill_arg=skill_arg)
            self.env.move_to(y)

            
            
if __name__ == "__main__":
    config = default_config
    config['mode'] = 'sim'

    ####################
    # Setup simulation #
    ####################
    from cooking_env.env.base.ce import CookingEnv

    config['SimulationEnv'] = {
        'type': CookingEnv,
        'config': {
            # specific to this env
            "suffix": "",
            "particle_test": True,
            "arm": "jaco",
            "control_mode": "velocity"
        }
    }

    ####################
    # Setup experiment #
    ####################
    # from robot_cooking.robot_cooking import RobotCooking

    ###################
    # Set up automata #
    ###################
    from tl_utils.generate_automata import GenerateAutomata
    from tl_utils.tl_config import KEY_POSITIONS, OBJECT_RELATIVE_POSE, STATE_IDX_MAP, PREDICATES
    
    config['Automata'] = {
        'type': GenerateAutomata,
        'config': {
            'formula':'F((move_to && open_gripper) && X F (close_gripper))',
            'key_positions': KEY_POSITIONS,
            'object_relative_pose': OBJECT_RELATIVE_POSE,
            'state_idx_map': STATE_IDX_MAP,
            'predicate_robustness': PREDICATES,
            'fsa_save_dir': os.path.join(os.environ['LEARNING_PATH'], 'tl_utils', 'sandbox', 'figures'),
            'dot_file_name': 'fsa',
            'svg_file_name': 'fsa',
            'mdp_state_space': {'type': 'float', 'shape': (22, ), 'upper_bound':[], 'lower_bound': []}
        }
    }
    

    ################
    # Setup Skills #
    ################
    from skills.skills import Skills
    
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
                'action_space': {'shape': (3,), 'upper_bound': [0.1, 0.1, 0.1], 'lower_bound': [-0.1,-0.1,-0.1]},
                'use_own_pose': True,
                'dt': 0.015
            },
            #'translation_gen': 'clf_cbf',
            'translation_gen': 'dmp',
            'orientation_gen': 'dmp'            
        }
    }

    #### Load learned skills #####

    ## flip switch open 
    experiment_root_dir = os.path.join(os.environ['LEARNING_PATH'], 'learning', 'experiments')
    experiment_name = 'test'
    hyperparam_dir = 'seed0'
    itr = 0

    learned_skill_config = {
        "state_space": {'type': 'float', 'shape': (3, ), "upper_bound": [], 'lower_bound': []},
        "action_space": {'type': 'float', 'shape': (3, ), "upper_bound": 70*np.ones(3), 'lower_bound': -70*np.ones(3)},
        "training_config_restore_path": os.path.join(experiment_root_dir, experiment_name, 'config', hyperparam_dir, 'config.pkl'),
        "policy_restore_path": os.path.join(experiment_root_dir, experiment_name, 'transitions', hyperparam_dir, 'itr_'+str(itr)),
        "state_preprocessor_restore_path": os.path.join(experiment_root_dir, experiment_name, 'info', hyperparam_dir, 'state_preprocessor_params.pkl')
    }
    skills_config['LearnedSkills'] = {'flip_switch_on': learned_skill_config}

    ##############################
    # Initialize Execution Class #
    ##############################
    cls = RunRobotCooking(config=skills_config)

    #### Run ####
    # cls.run()

    #### Test ####
    cls.test()