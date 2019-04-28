import numpy as np
from future.utils import viewitems
import os
import copy
from rl_pipeline.configuration.configuration import Configuration

import json

# env_cal = json.loads(open(os.path.join(os.environ['RLFPS_PATH'], 'examples', 'iros2019', 'env_config.json'), 'rb').read())

env_cal = {}
default_config = {
    'env_name': "vrep_baxter",
    'headless': False
}

class ExperimentConfig(object):

    def __init__(self, config={}):
        self.ExperimentConfig_config = Configuration(default_config)
        self.ExperimentConfig_config.update(config)

        self.env_name = self.ExperimentConfig_config.get('env_name')
        self.headless = self.ExperimentConfig_config.get('headless')
        
        # used in batch_sampler to post-process rewards
        self.process_rewards = None
        self.Environment = None

        self.construct_mdp_env()
   
    def construct_mdp_env(self):
        self.reset = {
            'type': None,
            'config': {}
        }
        if self.env_name == 'vrep_baxter':            
            from learning.env.learning_env import LearningEnv
            from traj_generators.dmp.dmp import DMP
            from cooking_env.env.base.ce import CookingEnv
            self.get_state, self.get_reward, self.is_done, self.state_space, self.action_space, env_other = self.baxter_env_config()
            self.reset = env_other['reset']
        
         
            self.mdp_env_type = LearningEnv
            self.mdp_env_config = {
                # Common to all envs
                "seed": 10,
                "state_space": self.state_space,
                "action_space": self.action_space,
                "get_state": self.get_state,
                "get_reward": self.get_reward,
                "is_done": self.is_done,
                "get_info": None,
                #### class specific ####
                "WPGenerator": {
                    'type':DMP,
                    'config': {
                        # initial goal
                        'initial_goal': [0.43, -1.85, 0.9, 0, 0, 0 ,1],
                        # gain on attractor term y dynamics (linear)
                        'ay': 50,
                        # gain on attractor term y dynamics (linear)
                        'by': None,
                        # gain on attractor term y dynamics (angular)
                        'az': 50,
                        # gain on attractor term y dynamics (angular)
                        'bz': None,
                        # timestep
                        'dt': 0.02,
                        # time scaling, increase tau to make the system execute faster
                        'tau': 1.0,
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
                        'n_linear_dmp': 3,
                        'n_angular_dmp': 3
                    }
                },
                # for cooking environment
                "BaseEnv":{
                    'type': CookingEnv,
                    'config': {
                        # specific to this env
                        "suffix": "",
                        "particle_test": True,
                        "arm": "jaco",
                        "control_mode": "velocity"
                    }
                }
            }

        else:
            raise ValueError('unsupported environment')

        self.Environment = {
            'type': self.mdp_env_type,
            'config': self.mdp_env_config
        }
            
            
    def baxter_env_config(self):

        #### State ####
        def get_state(all_info):
            mdp_state = np.array(list(all_info['target_pos']) + \
                                 list(all_info['target_quat']) + \
                                 list(all_info['button_rel_angle']))
            return mdp_state

        #### Reward ####
        def get_reward(state=None, action=None, next_state=None, all_info=None):
            return 0
        
        #### Done ####
        def is_done(state=None, action=None, next_state=None, all_info=None):
            motion_range = all_info['motion_range']
            low = np.array([motion_range['x'][0], motion_range['y'][0], motion_range['z'][0]])
            high = np.array([motion_range['x'][1], motion_range['y'][1], motion_range['z'][1]])

            target_pos = state[:3]

            
            if all(target_pos < low) or all(target_pos > high):
                return True
            else:
                return False

            
        state_space = {'type': 'float', 'shape': (8, ), 'upper_bound': [], 'lower_bound': []}

        action_coeff = 2.
        action_space = {'type': 'float', 'shape': (6, ), "upper_bound": np.array([1, 1, 1, 0.01, 0.01, 0.01]) * action_coeff, "lower_bound": np.array([1, 1, 1, 0.01, 0.01, 0.01]) * action_coeff}


        #### Reset ####
        reset_config = {}
        
        from learning.reset.experiment_env_reset import ExperimentEnvVrepReset

        other = {'reset': {'type': None, 'config': reset_config}}

        return get_state, get_reward, is_done, state_space, action_space, other
        
       
if __name__ == "__main__":
    pass