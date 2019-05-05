import numpy as np
from future.utils import viewitems
import os
import copy
from rl_pipeline.configuration.configuration import Configuration

import json

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
                        'initial_goal': [0.425, -1.85, 0.89, 0, 0, 0 ,1],
                        # gain on attractor term y dynamics (linear)
                        'ay': 35,
                        # gain on attractor term y dynamics (linear)
                        'by': None,
                        # gain on attractor term y dynamics (angular)
                        'az': 35,
                        # gain on attractor term y dynamics (angular)
                        'bz': None,
                        # timestep
                        'dt': 0.005,
                        # time scaling, increase tau to make the system execute faster
                        'tau': 1.0,
                        'translation_front_term': True,
                        'rotation_front_term': False,
                        # for canonical
                        'apx': 1.,
                        'gamma': 0.3,
                        # for faster convergence
                        'app': 0.5,
                        'apr': 0.5,
                        # for integrating goal
                        'ag': 3.0,
                        'ago': 3.0,
                        'use_dmp_pose': True,
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
            # mdp_state = np.array(list(all_info['target_pos']) + \
            #                      list(all_info['target_quat']))
            mdp_state = np.array(all_info['target_pos'])
            return mdp_state

        #### Reward ####
        def get_reward(state=None, action=None, next_state=None, all_info=None):
            button_joint_angle = all_info['button_joint_angle']
            button_rel_pose = all_info['button_rel_pose']
            button_vel = all_info['button_vel']
            toaster_joint_frame_angle = all_info['button_joint_frame_angle']
            
            r = toaster_joint_frame_angle[2] # - np.linalg.norm(button_vel[:3])
            
            return r
            
        
        #### Done ####
        def is_done(state=None, action=None, next_state=None, all_info=None):
            done = False
            motion_range = all_info['motion_range']
            low = np.array([motion_range['x'][0], motion_range['y'][0], motion_range['z'][0]])
            high = np.array([motion_range['x'][1], motion_range['y'][1], motion_range['z'][1]])
            target_pos = state[:3]

            ## done if move outside of motion region
            if any(target_pos < low) or any(target_pos > high):
                print("done: moved outside of motion region")
                done = True

            ## done if reached goal
            goal_dist = np.linalg.norm(target_pos - all_info['goal'][:3])
            if goal_dist < 0.01:
                print("done: reached goal")
                done = True

            ## done if hit button 
            button_vel = all_info['button_vel']
            button_disturbance =  np.linalg.norm(np.concatenate([button_vel[:4], np.array([button_vel[-1]])]))

            if button_disturbance > 0.1:
                print('done: button pushed away from nominal')
                done = True

            ## done if finished task
            toaster_joint_frame_angle = all_info['button_joint_frame_angle'][2]
            if toaster_joint_frame_angle > 0.4:
                print('done: turn on task done')
                done = True
                
            return done
            
        state_space = {'type': 'float', 'shape': (3, ), 'upper_bound': [], 'lower_bound': []}

        action_coeff = 70
        action_space = {'type': 'float', 'shape': (3, ), "upper_bound": np.ones(3) * action_coeff, "lower_bound": -np.ones(3) * action_coeff}


        #### Reset ####
        reset_config = {}
        
        from learning.reset.experiment_env_reset import ExperimentEnvVrepReset

        other = {'reset': {'type': None, 'config': reset_config}}

        return get_state, get_reward, is_done, state_space, action_space, other
        
       
if __name__ == "__main__":
    pass