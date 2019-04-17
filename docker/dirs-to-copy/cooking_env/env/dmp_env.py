import os
import numpy as np
import time
from future.utils import viewitems

from cooking_env.env.base.ce import CookingEnv
from cooking_env.env.dmp.canonical_system import CanonicalSystem

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

class DMPEnv(object):

    def __init__(self, config={}, seed=None, base_env=None):
        self.DMPEnv_config = default_config
        self.DMPEnv_config.update(config)

        if base_env is None:
            self.base_env = self.DMPEnv_config['BaseEnv']['type']( self.DMPEnv_config['BaseEnv']['config'])
        else:
            self.base_env = base_env
            
        self.all_info = {}

        if seed:
            self.set_seed(seed)
        else:
            self.set_seed(self.DMPEnv_config.get('seed'))

        self.build_graph()
      
    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self.n_dmps):
            if (self.y0[d] == self.goal[d]):
                print("initial position too close to goal, offsetting")
                self.goal[d] += 1e-4

    def set_initial_pos(self, y0):
        self.y0 = y0

    def set_goal_pos(self, goal):
        self.goal = goal


    def reset(self, y=None, dy=None, ddy=None):
        if y is None:
            self.y = copy.copy(self.y0).astype(np.float64)
        else:
            self.y = copy.copy(y).astype(np.float64)
            
        if dy is None:
            self.dy = np.zeros(self.n_dmps).astype(np.float64)
        else:
            self.dy = copy.copy(dy).astype(np.float64)

        if ddy is None:
            self.ddy = np.zeros(self.n_dmps).astype(np.float64)
        else:
            self.ddy = copy.copy(ddy).astype(np.float64)

        self.ddy = np.zeros(self.n_dmps).astype(np.float64)
        self.cs.reset_state()

        if self.goal is not None:
            self.check_offset()

    def reset_cs(self):
        self.cs.reset_state()

    def build_graph(self):
        self.goal = None
        self.y0 = None
        self.nb_get_action_calls = 0
    

        self.n_dmps = self.action_space['shape'][0]
        self.dt = self.DMPEnv_config.get('dt')
        self.tau = self.DMPEnv_config.get('tau')
        
        self.by = self.DMPEnv_config.get('by')

        ay = self.DMPEnv_config.get('ay')
        if ay is None:
            self.ay = np.ones(self.n_dmps) * 25
        elif isinstance(ay, (int, float)):
            self.ay = np.ones(self.n_dmps) * ay
        else:
            self.ay = ay

        
        by = self.DMPEnv_config.get('by')
        if by is None:
            self.by = self.ay / 4
        elif isinstance(by, (int, flot)):
            self.by = np.ones(self.n_dmps) * by
        else:
            self.by = by

        self.cs = CanonicalSystem(dt=self.dt)
        self.timesteps = int(self.cs.run_time / self.dt)

        
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
        '''
        here action is forcing function output
        '''
        action = np.array(action).flatten()
        assert action.size == self.action_space['shape'][0]
        
        self.nb_get_action_calls += 1

        ## currently only support translation
        self.y, _ = self.base_env.get_target_pose()
        self.dy, _ = self.base_env.get_target_velocity()

        point_attractor = self.ay * ( self.by * (self.goal - self.y) - self.dy )

        # optional system feedback
        error = 1.0
        error_coupling = 1. / (1. + error)
    
        if self.DMPEnv_config['use_canonical']:
            cx = self.cs.step(tau=self.tau, error_coupling=error_coupling)
            # generate forcing term
            front_terms = cx * (self.goal - self.y)
            point_attractor += front_terms

        self.ddy = (point_attractor + action) * self.tau
        self.dy += self.ddy * self.dt * error_coupling
        self.y += self.dy * self.dt * error_coupling

        self.base_env.set_target_pose(self.y)
        
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


    cls = DMPEnv(config=config)
    curr_pos = cls.ce_env.get_target_pose()
    goal = curr_pos[:3] + np.array([0.5, 0.5, 0, 0, 0, 0, 1])
        
    for i in range(1000):
        cls.set_goal_pos(goal)
        cls.step(np.array([0,0,0]))
        
    # print(cls.ce_env.get_target_velocity())