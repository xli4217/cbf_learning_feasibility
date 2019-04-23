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
    'dist_th': 0.05,
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
                        # gain on attractor term y dynamics (linear)
                        'ay': None,
                        # gain on attractor term y dynamics (linear)
                        'by': None,
                        # gain on attractor term y dynamics (angular)
                        'az': 20,
                        # gain on attractor term y dynamics (angular)
                        'bz': None,
                        # timestep
                        'dt': 0.05,
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
                        "particle_test": False,
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
        def get_state(all_info):
            return np.array(list(all_info['target_pos']) + \
                            list(all_info['target_quat']) + \
                            list(all_info['button_rel_angle']))

        def get_reward(state=None, action=None, next_state=None):
            ee_pos = next_state[-3:]
            red_pos = next_state[7:10]
            rel_pos = np.array(red_pos) - np.array(ee_pos)
            
            sqrt_dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
            # th_precise = dist_th**2 - np.exp(-100*dist_th**2)
            # r_precise = -(sqrt_dist**2 - np.exp(-100*sqrt_dist**2))
            
            return -sqrt_dist
        

        def is_done(state=None, action=None, next_state=None):
            ee_pos = state[-3:]
            red_pos = state[7:10]
            rel_pos = np.array(red_pos) - np.array(ee_pos)
            
            sqrt_dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
            if sqrt_dist < self.dist_th:
                return True
            else:
                return False
            
            
        state_space = {'type': 'float', 'shape': (22, ), 'upper_bound': [], 'lower_bound': []}

        if self.env_name == 'vrep_baxter':
            action_coeff = 0.7
        elif self.env_name == 'baxter':
            action_coeff = 0.1
        action_space = {'type': 'float', 'shape': (7, ), "upper_bound":np.array([1.5,1.5,1.5,1.5,3.5,3.5,3.5]) * action_coeff, "lower_bound": np.array([-1.5,-1.5,-1.5,-1.5,-3.5,-3.5,-3.5]) * action_coeff}

        reset_config = {
            'object_poses': {
                'red': {
                    'randomize': True,
                    'nominal': [0.7, -0.45, -0.06],
                    'min': env_cal['red']['sample_range_min'],
                    'max': env_cal['red']['sample_range_max']
                },
                'green': {
                    'randomize': True,
                    'nominal': [0.7, -0.45, -0.046],
                    'min': env_cal['green']['sample_range_min'],
                    'max': env_cal['green']['sample_range_max']
                },
                'blue': {
                    'randomize': True,
                    'nominal': [0.7, -0.45, -0.046],
                    'min': env_cal['blue']['sample_range_min'],
                    'max': env_cal['blue']['sample_range_max']
                },
                'hand': {
                    'randomize': True,
                    'nominal': [0, 0, 0],
                    'min': np.array(env_cal['appear_in_sight']['min']) - 0.1,
                    'max': np.array(env_cal['appear_in_sight']['max']) + 0.1
                }
            },
            'joint_angles': {
                'randomize': True,
                'nominal': [ 0.5564515308054339, -1.1094516048381255, 0.006135923151541655, 0.9990049881103757, 0.15033011721277054, 1.5780827355371194, -0.11888351106111957],
                "min": [0.19538355, -0.95567003, -0.69987873,  0.05867477, -0.13805827, 0.39269908, -1.72227693],
                "max": [0.6259730e+00, -5.29223372e-02,  1.16199045e-01, 1.58268468e+00,  1.10484966e+00,  1.71997596e+00,  1.52631088e-01]
            }
        }

        from examples.rss2019.reset.experiment_env_reset import ExperimentEnvVrepReset

        other = {'reset': {'type': ExperimentEnvVrepReset, 'config': reset_config}}

        return get_state, get_reward, is_done, state_space, action_space, other
        
       
if __name__ == "__main__":
    pass