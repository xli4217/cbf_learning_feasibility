import os
import numpy as np
import time
from future.utils import viewitems

from cooking_env.env.base.ce import CookingEnv

default_config = {
    # Common to all envs
    "seed": 10,
    "state_space": None,
    "action_space": None,
    "get_state": None,
    "get_reward": None,
    "is_done": None,
    "get_info": None,
    # class specific

    # for cooking environment
    "ce_config": {
        # specific to this env
        "suffix": "",
        "particle_test": True,
        "arm": "jaco",
        "control_mode": "velocity"
    }
}

class DMPEnv(object):

    def __init__(self, config={}, seed=None):
        self.DMPEnv_config = default_config
        self.DMPEnv_config.update(config)

        self.ce_env = CookingEnv(self.DMPEnv_config['ce_config'])

        self.all_info = {}

        if seed:
            self.set_seed(seed)
        else:
            self.set_seed(self.DMPEnv_config.get('seed'))
        

        
    def get_state(self):
        self.update_all_info()
        if self.DMPEnv_config.get('get_state'):
            return self.DMPEnv_config.get('get_state')(self.all_info)
        else:
            return np.array([0])

        
    def get_reward(self, state=None, action=None, next_state=None):
        if self.DMPEnv_config.get('get_reward'):
            return self.DMPEnv_config.get('get_reward')(state, action, next_state)
        else:
            return 0
            
    def is_done(self, state=None, action=None, next_state=None):
        if self.DMPEnv_config.get('is_done'):
            return self.DMPEnv_config.get('is_done')(state, action, next_state)
        else:
            return False

    def step(self, action):
        pass
            
    def set_seed(self, seed):
        np.random.seed(seed)

    @property
    def state_space(self):
        return self.DMPEnv_config.get('state_space')

    @property
    def action_space(self):
        return self.DMPEnv_config.get('action_space')

    def save(self, save_dir):
        pass
        
    def restore(self, restore_dir):
        pass


if __name__ == "__main__":
    cls = DMPEnv()