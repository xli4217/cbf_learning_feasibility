import os
import numpy as np
import time
from future.utils import viewitems

from cooking_env.env.base.ce import CookingEnv
import cooking_env.vrep as vrep

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
        'type':None,
        'config': {}
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

class LearningEnv(object):

    def __init__(self, config={}, seed=None, base_env=None, wp_gen=None):
        self.LearningEnv_config = default_config
        self.LearningEnv_config.update(config)

        if base_env is None:
            self.base_env = self.LearningEnv_config['BaseEnv']['type']( self.LearningEnv_config['BaseEnv']['config'])
        else:
            self.base_env = base_env

        if wp_gen is None:
            self.wp_gen = self.LearningEnv_config['WPGenerator']['type']( self.LearningEnv_config['WPGenerator']['config'])
        else:
            self.wp_gen = wp_gen

            
        self.all_info = {}

        if seed:
            self.set_seed(seed)
        else:
            self.set_seed(self.LearningEnv_config.get('seed'))
      
    
    def set_goal_pose(self, goal):
        self.wp_gen.set_goal(goal)

        if len(goal) != 7:
            goal = np.concatenate([goal, np.array([0,0,0,1])])
        self.base_env.set_goal_pose(goal)
        
    def get_state(self):
        self.update_all_info()
        if self.LearningEnv_config.get('get_state'):
            return self.LearningEnv_config.get('get_state')(self.all_info)
        else:
            return np.array([0])

        
    def get_reward(self, state=None, action=None, next_state=None):
        if self.LearningEnv_config.get('get_reward'):
            return self.LearningEnv_config.get('get_reward')(state, action, next_state)
        else:
            return 0
            
    def is_done(self, state=None, action=None, next_state=None):
        if self.LearningEnv_config.get('is_done'):
            return self.LearningEnv_config.get('is_done')(state, action, next_state)
        else:
            return False

    def update_all_info(self):
        self.base_env.synchronous_trigger()
        target_pos, target_quat = self.base_env.get_target_pose()

        rc, button_rel_angle = vrep.simxGetObjectOrientation(self.base_env.clientID,
                                                             self.base_env.object_handles['toaster_button'],
                                                             self.base_env.object_handles['hotdog_cooker'],
                                                             vrep.simx_opmode_oneshot)

        self.all_info = {
            'target_pos': target_pos,
            'target_quat': target_quat,
            'button_rel_angle': button_rel_angle
        }
        
    def step(self, action):
        '''
        here action is forcing function output
        '''
        action = np.array(action).flatten()
        assert action.size == self.action_space['shape'][0]
        
        curr_pos, curr_quat = self.base_env.get_target_pose()
        curr_linear_vel, curr_angular_vel = self.base_env.get_target_velocity()
        curr_angular_vel = curr_angular_vel * np.pi / 180

        curr_pose = np.concatenate([curr_pos, curr_quat])
        curr_vel = np.concatenate([curr_linear_vel, curr_angular_vel])
        
        ddy, dy, y = self.wp_gen.get_next_wp(action, curr_pose, curr_vel)

        if len(y) < 7:
            y = np.concatenate([y, np.array([0,0,0,1])])
            
        self.base_env.set_target_pose(y)
        
    def set_seed(self, seed):
        np.random.seed(seed)

    @property
    def state_space(self):
        return self.LearningEnv_config.get('state_space')

    @property
    def action_space(self):
        return self.LearningEnv_config.get('action_space')

    def save(self, save_dir):
        pass
        
    def restore(self, restore_dir):
        pass


if __name__ == "__main__":
    # from cooking_env.env.QP_waypoints.QPcontroller import QPcontroller
    from traj_generators.dmp.dmp import DMP
    
    dmp_gen = {
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
    }

    # qp_gen = {
    #     'type': QPcontroller,
    #     'config': {}
    # }
    
    config = default_config
    config['action_space'] = {'type': 'float', 'shape':(6, ), 'upper_bound': np.ones(3), 'lower_bound': -np.ones(3)}
    config['WPGenerator'] = dmp_gen

    
    cls = LearningEnv(config=config)
    
    curr_pos, curr_quat = cls.base_env.get_target_pose()
    goal_pos, goal_quat = cls.base_env.get_goal_pose()
    goal = np.concatenate([goal_pos, goal_quat])
    cls.set_goal_pose(goal)
    
    for i in range(1000):
        # cls.step(np.array([0,0,0,0,0,0]))
        cls.update_all_info()