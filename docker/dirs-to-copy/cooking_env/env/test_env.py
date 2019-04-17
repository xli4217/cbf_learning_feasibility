import os
import numpy as np
import time
from future.utils import viewitems

from cooking_env.env.base.ce import CookingEnv
from cooking_env.env.dmp.dmp import DMP

default_config = {
    # Common to all envs
    "seed": 10,
    "state_space": None,
    "action_space": None,
    "get_state": None,
    "get_reward": None,
    "is_done": None,
    "get_info": None,
    #### class specific ####
    "WPGenerator": {
        'type':DMP,
        'config': {
            # gain on attractor term y dynamics
            'ay': None,
            # gain on attractor term y dynamics
            'by': None,
            # timestep
            'dt': 0.01,
            # time scaling, increase tau to make the system execute faster
            'tau': 1.0,
            'use_canonical': False,
            # for cooking environment
            
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

class TestEnv(object):

    def __init__(self, config={}, seed=None, base_env=None, wp_gen=None):
        self.TestEnv_config = default_config
        self.TestEnv_config.update(config)

        if base_env is None:
            self.base_env = self.TestEnv_config['BaseEnv']['type']( self.TestEnv_config['BaseEnv']['config'])
        else:
            self.base_env = base_env

        if wp_gen is None:
            self.wp_gen = self.TestEnv_config['WPGenerator']['type']( self.TestEnv_config['WPGenerator']['config'])
        else:
            self.wp_gen = wp_gen

            
        self.all_info = {}

        if seed:
            self.set_seed(seed)
        else:
            self.set_seed(self.TestEnv_config.get('seed'))
      
    
    def set_goal_pos(self, goal):
        self.wp_gen.set_goal(goal)
        
    def get_state(self):
        self.update_all_info()
        if self.TestEnv_config.get('get_state'):
            return self.TestEnv_config.get('get_state')(self.all_info)
        else:
            return np.array([0])

        
    def get_reward(self, state=None, action=None, next_state=None):
        if self.TestEnv_config.get('get_reward'):
            return self.TestEnv_config.get('get_reward')(state, action, next_state)
        else:
            return 0
            
    def is_done(self, state=None, action=None, next_state=None):
        if self.TestEnv_config.get('is_done'):
            return self.TestEnv_config.get('is_done')(state, action, next_state)
        else:
            return False

            
    def step(self, action):
        '''
        here action is forcing function output
        '''
        action = np.array(action).flatten()
        assert action.size == self.action_space['shape'][0]
        

        ## currently only support translation
        curr_pose, _ = self.base_env.get_target_pose()
        curr_vel, _ = self.base_env.get_target_velocity()

        ddy, dy, y = self.wp_gen.get_next_wp(action, curr_pose, curr_vel)

        if len(y) < 7:
            y = np.concatenate([y, np.array([0,0,0,1])])
            
        self.base_env.set_target_pose(y)
        
    def set_seed(self, seed):
        np.random.seed(seed)

    @property
    def state_space(self):
        return self.TestEnv_config.get('state_space')

    @property
    def action_space(self):
        return self.TestEnv_config.get('action_space')

    def save(self, save_dir):
        pass
        
    def restore(self, restore_dir):
        pass


if __name__ == "__main__":
    from cooking_env.env.QP_waypoints.QPcontroller import QPcontroller

    dmp_gen = {
        'type':DMP,
        'config': {
            # gain on attractor term y dynamics
            'ay': None,
            # gain on attractor term y dynamics
            'by': None,
            # timestep
            'dt': 0.01,
            # time scaling, increase tau to make the system execute faster
            'tau': 1.0,
            'use_canonical': False,
        }
    }

    qp_gen = {
        'type': QPcontroller,
        'config': {}
    }
    
    config = {
        # Common to all envs
        "seed": 10,
        "state_space": None,
        "action_space": {'type': 'float', 'shape': (3,), 'upper_bound': [1.,1.,1.], 'lower_bound':[-1.,-1.,-1.]},
        "get_state": None,
        "get_reward": None,
        "is_done": None,
        "get_info": None,
        #### class specific ####
        "WPGenerator": qp_gen,
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


    cls = TestEnv(config=config)
    curr_pos, curr_quat = cls.base_env.get_target_pose()
    goal = curr_pos + np.array([0.5, 0.5, 0])
        
    for i in range(1000):
        cls.set_goal_pos(goal)
        cls.step(np.array([0,0,0]))
        
    # print(cls.ce_env.get_target_velocity())