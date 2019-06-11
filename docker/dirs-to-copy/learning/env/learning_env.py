import os
import numpy as np
import time
from future.utils import viewitems
from utils import transformations

from cooking_env.env.base.ce import CookingEnv
import cooking_env.vrep as vrep

from utils.utils import *

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
    '''
    Equivalent of robot_cooking_interface.py
    '''
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
            # self.set_goal_pose(self.LearningEnv_config['WPGenerator']['config']['initial_goal'])
            goal_pos, goal_quat = self.base_env.get_goal_pose()
            self.set_goal_pose(np.concatenate([goal_pos, goal_quat]))
            
        self.all_info = {}

        if seed:
            self.set_seed(seed)
        else:
            self.set_seed(self.LearningEnv_config.get('seed'))

        self.sample_range = self.base_env.get_region_info(region='sample_region')
        self.motion_range = self.base_env.get_region_info(region='motion_region')
        
    def reset(self, s=None):
        self.update_all_info()

        if s is None:
            ## sampel target
            low = [self.sample_range['x'][0], self.sample_range['y'][0], self.sample_range['z'][0]]
            high = [self.sample_range['x'][1], self.sample_range['y'][1], self.sample_range['z'][1]]
            self.target_pos = np.random.uniform(low, high, 3)
        else:
            self.target_pos = s
            
        quat = np.array([9.96912479e-01,  5.98265615e-08,   -5.05284581e-04, 7.85199478e-02])
        self.base_env.set_target_pose(np.concatenate([self.target_pos, quat]))
        self.wp_gen.reset(np.concatenate([self.target_pos, quat]), np.zeros(6))

        ## reset button
        rc = vrep.simxSetJointTargetPosition(self.base_env.clientID,
                                             self.base_env.object_handles['toaster_button_joint'],
                                             5.5*np.pi,
                                             vrep.simx_opmode_blocking)
            
        
        self.base_env.synchronous_trigger()
        
        while np.linalg.norm(self.all_info['button_vel']) > 0.01:
            self.update_all_info()
            self.base_env.synchronous_trigger()
        
    def get_info(self):
        return self.all_info
        
        
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
        ee_pos, ee_quat = self.get_ee_pose()
        lv, av = self.get_ee_velocity()
        rc = 1
        while rc != 0:
            rc, button_linear_vel, button_angular_vel = vrep.simxGetObjectVelocity(self.base_env.clientID,
                                                                                   self.base_env.object_handles['toaster_button'],
                                                                                   vrep.simx_opmode_streaming)

        button_vel = np.concatenate([np.array(button_linear_vel), np.array(button_angular_vel)])
        curr_pose = np.concatenate([np.array(ee_pos), np.array(ee_quat)])
        curr_vel = np.concatenate([np.array(lv), np.array(av)])
        
        self.all_info = {
            'goal': self.goal,
            'button_vel': button_vel,
            'sample_range': self.sample_range,
            'motion_range': self.motion_range,
            #### skill args ###
            'curr_pose': curr_pose,
            'curr_vel': curr_vel,
            'switchon': self.get_switch_state(),
            'condimentapplied': -10,
            'gripper_state': self.get_gripper_state(),
            'obs_info': self.get_obstacle_info(),
            'obj_poses': self.get_object_pose()
        }
        
    def step(self, action):
        '''
        here action is forcing function output
        '''
        action = np.array(action).flatten()
        assert action.size == self.action_space['shape'][0]

        action *= 100

        # clip action
        action = np.clip(action, self.action_space['lower_bound'], self.action_space['upper_bound'])
        
        curr_pos, curr_quat = self.get_target_pose()
        curr_linear_vel, curr_angular_vel = self.get_target_velocity()
        curr_angular_vel = curr_angular_vel * np.pi / 180

        curr_pose = np.concatenate([curr_pos, curr_quat])
        curr_vel = np.concatenate([curr_linear_vel, curr_angular_vel])

        #### HACK! remove when done debugging #####
        action = np.zeros(action.shape)
        
        if len(action) == 3:
            action = np.concatenate([action, np.zeros(3)])
        ddy, dy, y = self.wp_gen.get_next_wp(action, curr_pose, curr_vel)
        
        if len(y) < 7:
            y = np.concatenate([y, np.array([0,0,0,1])])

        # time.sleep(0.05)
        self.base_env.set_target_pose(y)


    ####################
    # Common Interface #
    ####################

    def home_robot(self):
        pass

    def get_target_pose(self):
        return self.base_env.get_target_pose()

    def set_target_pose(self,pt):
        return self.base_env.set_target_pose(pt)

    def get_target_velocity(self):
        return self.base_env.get_target_velocity()

    def get_ee_pose(self):
        return self.base_env.get_ee_pose()

    def set_ee_pose(self, pt):
        self.base_env.set_ee_pose(pt)

    def get_ee_velocity(self):
        return self.base_env.get_ee_velocity()

    def pub_ee_frame_velocity(self, direction='z',vel_scale=1.0, duration_sec=0.1):
        target_pos, target_quat = self.get_target_pose()
        T = transformations.quaternion_matrix(target_quat)
        T[:3,3] = target_pos

        dT = np.eye(4)
        if direction == 'x':
            dT[0,3] = vel_scale*0.005
        if direction == 'y':
            dT[1,3] = vel_scale*0.005
        if direction == 'z':
            dT[2,3] = vel_scale*0.005
       
        T_post_vel = T.dot(dT)

        new_target_pos = T_post_vel[:3,3]
        new_target_quat = transformations.quaternion_from_matrix(T_post_vel)

        new_target_pose = np.concatenate([new_target_pos, new_target_quat])
        
        self.set_target_pose(new_target_pose)
        
    def get_gripper_state(self):
        return self.base_env.get_gripper_state()

    def set_gripper_state(self, gs):
        self.base_env.set_gripper_state(gs)
        
    def get_object_pose(self):
        return self.base_env.get_object_pose()
        
    def get_obstacle_info(self):
        return self.base_env.get_obstacle_info()
        
    def set_goal_pose(self, goal):
        self.goal = goal
        self.wp_gen.set_goal(goal)

        if len(goal) != 7:
            goal = np.concatenate([goal, np.array([0,0,0,1])])

        self.base_env.set_goal_pose(goal)

        
    def get_switch_state(self):
        switch_angle_rel_grill = self.base_env.get_switch_state()
        
        if switch_angle_rel_grill < 1.04:
            return 10
        else:
            return -10
        
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

    def close(self):
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

    # cls.reset()
    while True:
        # cls.reset()
        #s = cls.get_state()
        # a = policy.get_action(s)
        # cls.step(a*10)
        #print(cls.is_done(state=s))
        # if cls.is_done(state=s):
        #     cls.reset()
        cls.update_all_info()
        cls.set_gripper_state(1.)
        print(cls.get_gripper_state())
        cls.base_env.synchronous_trigger()
        # cls.get_reward(state=cls.all_info['target_pos'])
        #print(cls.is_done(state=cls.all_info['target_pos']))
        # time.sleep(0.05)
        # target_pos = np.concatenate([cls.all_info['target_pos'], cls.all_info['target_quat']])
        # target_pos += np.array([0.01,0,0,0,0,0,0])
        # cls.base_env.set_target_pose(target_pos)