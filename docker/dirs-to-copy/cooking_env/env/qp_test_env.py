import os
import numpy as np
import time
from future.utils import viewitems

from cooking_env.env.base.ce import CookingEnv

default_config = {
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
            "particle_test": True,
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

          
    def set_goal_pos(self, goal):
        self.wp_gen.set_goal(goal)

        if len(goal) != 7:
            goal = np.concatenate([goal, np.array([0,0,0,1])])
        self.base_env.set_goal_pose(goal)        
            
    def step(self, action):
        '''
        here action is forcing function output
        '''
        action = np.array(action).flatten()
        
        curr_pos, curr_quat = self.base_env.get_target_pose()
        curr_linear_vel, curr_angular_vel = self.base_env.get_target_velocity()

        curr_pose = np.concatenate([curr_pos, curr_quat])
        curr_vel = np.concatenate([curr_linear_vel, curr_angular_vel])

        obs_info = self.base_env.get_obstacle_info()
        
        ddy, dy, y = self.wp_gen.get_next_wp(action, curr_pose[:3], curr_vel, obs_info)
        
        if len(y) < 7:
            y = np.concatenate([y, np.array([0,0,0,1])])

        self.base_env.set_target_pose(y)
        
  
if __name__ == "__main__":
    from traj_generators.clf_cbf.QPcontroller import QPcontroller


    qp_gen = {
        'type': QPcontroller,
        'config': {
            'k_cbf': 1,
            'epsilon':0.8,
            'num_states':3,
            'action_space': {'shape': (6, ), 'upper_bound': 0.1 * np.zeros(6), 'lower_bound': -0.1 * np.zeros(6)},
            'use_own_pose': False,
            'dt': 0.2
        }
    }
    
    config = default_config
    config['WPGenerator'] = qp_gen

    
    cls = TestEnv(config=config)
    goal_pos, goal_quat = cls.base_env.get_goal_pose()
    goal = np.concatenate([goal_pos, goal_quat])
    cls.set_goal_pos(goal)
    
    for i in range(1000):
        cls.step(np.array([0,0,0,0,0,0]))
       
