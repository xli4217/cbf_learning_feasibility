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
                    'action_space': {'shape': (3,), 'upper_bound': [0.1, 0.1, 0.1], 'lower_bound': [-0.1,-0.1,-0.1]},
                    'use_own_pose': True,
                    'dt': 0.015,
                    'log_dir': os.path.join(os.environ['LEARNING_PATH'], 'execution', 'log')
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
        skills_config['LearnedSkills'] = {'flipswitchon': learned_skill_config}

        return MotorSkills, skills_config

    def low_level_tl_skill_config(self):
        from skills.low_level_tl_skills import LowLevelTLSkills
        from tl_utils.tl_config import KEY_POSITIONS, OBJECT_RELATIVE_POSE, STATE_IDX_MAP, PREDICATES

        pick_hotdog = "F((moveto_hotdogplate && opengripper) && X F (closegripper))"
        place_bun = "F((moveto_bunplate && closegripper)  && X F (opengripper))"

        pick_hotdog_place_bun = "F ((moveto_hotdogplate && opengripper) && X F (closegripper && XF ((moveto_bunplate && closegripper)  && X F (opengripper))))"

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
        
        config = {
            'make_hotdog': {
                'formula':"F (" + apply_condiment + ")",
                'visdom': True,
                'key_positions': KEY_POSITIONS,
                'object_relative_pose': OBJECT_RELATIVE_POSE,
                'state_idx_map': STATE_IDX_MAP,
                'predicate_robustness': PREDICATES,
                'fsa_save_dir': os.path.join(os.environ['LEARNING_PATH'], 'execution', 'figures'),
                'dot_file_name': 'pick_hotdog',
                'svg_file_name': 'pick_hotdog',
                'mdp_state_space': {'type': 'float', 'shape': (36, ), 'upper_bound':[], 'lower_bound': []}
            }
        }

        return LowLevelTLSkills, config


