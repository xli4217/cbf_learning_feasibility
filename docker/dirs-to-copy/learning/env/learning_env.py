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
        'config': {
            'initial_goal': [0,0,0,0,0,0,1]
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

class LearningEnv(object):

    def __init__(self,
                 config={},
                 seed=None,
                 port_num=19999,
                 suffix="",
                 base_env=None,
                 wp_gen=None,
                 reset=None,
                 logger=None):
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

        if self.wp_gen is not None:
            self.set_goal_pose(self.LearningEnv_config['WPGenerator']['config']['initial_goal'])
            
        self.all_info = {}

        if seed:
            self.set_seed(seed)
        else:
            self.set_seed(self.LearningEnv_config.get('seed'))

        self.sample_range = self.base_env.get_region_info(region='sample_region')
        self.motion_range = self.base_env.get_region_info(region='motion_region')
        
    def reset(self):
        low = [self.sample_range['x'][0], self.sample_range['y'][0], self.sample_range['z'][0]]
        high = [self.sample_range['x'][1], self.sample_range['y'][1], self.sample_range['z'][1]]
        self.target_pos = np.random.uniform(low, high, 3)
        
        self.base_env.set_target_pose(np.concatenate([self.target_pos, np.array([0,0,0,1])]))
        
        self.base_env.synchronous_trigger()
        
    def get_info(self):
        return self.all_info
        
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
            return self.LearningEnv_config.get('get_reward')(state, action, next_state, self.all_info)
        else:
            return 0
            
    def is_done(self, state=None, action=None, next_state=None):
        if self.LearningEnv_config.get('is_done'):
            return self.LearningEnv_config.get('is_done')(state, action, next_state, self.all_info)
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
            'button_rel_angle': [button_rel_angle[1]],
            'sample_range': self.sample_range,
            'motion_range': self.motion_range
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

    def teleop(self, cmd):
        pass

if __name__ == "__main__":
    # from cooking_env.env.QP_waypoints.QPcontroller import QPcontroller
    from traj_generators.dmp.dmp import DMP
    
    dmp_gen = {
        'type':DMP,
        'config': {
            'initial_goal': [0,0,0,0,0,0,1],
            # gain on attractor term y dynamics (linear)
            'ay': 50,
            # gain on attractor term y dynamics (linear)
            'by': None,
            # gain on attractor term y dynamics (angular)
            'az': 50,
            # gain on attractor term y dynamics (angular)
            'bz': None,
            # timestep
            'dt': 0.005,
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
    
    # config = default_config
    # config['action_space'] = {'type': 'float', 'shape':(6, ), 'upper_bound': np.ones(6), 'lower_bound': -np.ones(6)}
    # config['WPGenerator'] = dmp_gen
    # config['BaseEnv']['config']['particle_test'] = True
    
    # cls = LearningEnv(config=config)
    
    # curr_pos, curr_quat = cls.base_env.get_target_pose()
    # goal_pos, goal_quat = cls.base_env.get_goal_pose()
    # goal = np.concatenate([goal_pos, goal_quat])
    # cls.set_goal_pose(goal)


    #### test nn forcing function ####
    from rl_pipeline.algo_devel.ppo.pytorch.policy.mlp_policy import PytorchMlp
    from learning.experiment_config import ExperimentConfig

    exp_config = ExperimentConfig()

    cls = exp_config.Environment['type'](exp_config.Environment['config'])
    
    policy_config = {
        'scope': 'policy',
        'obs_dim': 8,
        'action_dim': 6,
    }
    policy = PytorchMlp(policy_config)

    cls.reset()
    for i in range(1000):
        s = cls.get_state()
        a = policy.get_action(s)
        cls.step(a*10)
        # cls.step(np.zeros(6))
        # if cls.is_done(state=s):
        #     cls.reset()
        cls.update_all_info()

   