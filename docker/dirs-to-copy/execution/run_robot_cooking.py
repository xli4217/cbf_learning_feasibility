import os
import numpy as np
import time
from future.utils import viewitems

from utils.utils import *

default_config = {
    #### this can be 'sim' or 'real' ####
    'mode': 'sim',
    'SimulationEnv':{
        'type': None,
        'config':{}
    },
    'ExperimentEnv': {
        'type': None,
        'config': {}
    },
    'Skills':{
        'construct_skill_state': None,
        'MotorSkills': {
            'type': None,
            'config':{}
        },
        'LowLevelTLSkills': {
            'type': None,
            'config': {}
        }
    }
}

class RunRobotCooking(object):

    def __init__(self, config={}):
        self.RunRobotCooking_config = default_config
        self.RunRobotCooking_config.update(config)

        if self.RunRobotCooking_config['mode'] == 'sim':
            self.env = self.RunRobotCooking_config['SimulationEnv']['type']( self.RunRobotCooking_config['SimulationEnv']['config'])
        elif self.RunRobotCooking_config['mode'] == 'real':
            self.env = self.RunRobotCooking_config['ExperimentEnv']['type']( self.RunRobotCooking_config['ExperimentEnv']['config'])
        else:
            raise ValueError('unsupported mode')
        
        self.motor_skills = self.RunRobotCooking_config['Skills']['MotorSkills']['type'](config=self.RunRobotCooking_config['Skills']['MotorSkills']['config'])
        
        self.ll_tl_skills = self.RunRobotCooking_config['Skills']['LowLevelTLSkills']['type'](config=self.RunRobotCooking_config['Skills']['LowLevelTLSkills']['config'])
        
        self.skill_arg = {}
        
    def update_skill_arg(self):
        curr_pos, curr_quat = self.env.get_target_pose()
        curr_pose = np.concatenate([curr_pos, curr_quat])

        linear_vel, angular_vel = self.env.get_target_velocity()
        curr_vel = np.concatenate([linear_vel, angular_vel])

        object_poses = self.env.get_object_pose()
        
        self.skill_arg.update({
            'curr_pose': curr_pose,
            'curr_vel': curr_vel,
            'gripper_state': self.env.get_gripper_state(),
            'obs_info': self.env.get_obstacle_info(),
            'obj_poses': object_poses
        })

        
    def move_to_target_with_motor_skill(self, pt, skill_name):
        self.update_skill_arg()
        
        self.env.set_goal_pose(pt)
        self.skill_arg['goal'] = pt
        
        while True:            
            pos_dist, quat_dist = pose_distance(pt, self.skill_arg['curr_pose']) 
            if pos_dist < 0.01 and quat_dist < 0.15:
                print('reached goal')
                break
                
            action = self.skills.get_action(skill_name=skill_name, skill_arg=self.skill_arg)
            self.env.move_to(action['value'])
            self.update_skill_arg()

    def execute_motor_skill(self, skill_name):
        from tl_utils.tl_config import KEY_POSITIONS, OBJECT_RELATIVE_POSE, STATE_IDX_MAP, PREDICATES

        if skill_name == 'opengripper':
            self.env.set_gripper_state(0.3)
        elif skill_name == 'closegripper':
            self.env.set_gripper_state(0.5)
        elif skill_name == "flipswitchon":
            pt = KEY_POSITIONS['switch_on_goal']
            self.move_to_target_with_motor_skill(pt, skill_name='flipswitchon')
        else:
            raise ValueError('unsupported skill')

    def get_low_level_tl_skill_actions(self):
        self.update_skill_arg()
        s = self.RunRobotCooking_config['Skills']['construct_skill_state'](self.skill_arg)
        ll_skill_action_n_constraint = self.ll_tl_skills.step(s)
        print(ll_skill_action_n_constraint)
        
    def run(self):
        pass
        
    def test(self):
        self.get_low_level_tl_skill_actions()
        # self.execute_skill('flipswitchon')
            
if __name__ == "__main__":
    config = default_config
    config['mode'] = 'sim'

    from execution_config import ExecutionConfig
    exe_config = ExecutionConfig()
    
    ####################
    # Setup simulation #
    ####################
    cls_type, cls_config = exe_config.simulation_config()
    
    config['SimulationEnv'] = {
        'type': cls_type,
        'config': cls_config 
    }

    ####################
    # Setup experiment #
    ####################
    # from robot_cooking.robot_cooking import RobotCooking

    ################
    # Setup Skills #
    ################
    motor_skill_type, motor_skill_config = exe_config.motor_skill_config()
    low_level_tl_skill_type, low_level_tl_skill_config = exe_config.low_level_tl_skill_config()

    from tl_utils.tl_config import construct_skill_state
    
    config['Skills'] = {
        'construct_skill_state': construct_skill_state,
        'MotorSkills':{
            'type': motor_skill_type,
            'config': motor_skill_config
        },
        'LowLevelTLSkills':{
            'type': low_level_tl_skill_type,
            'config': low_level_tl_skill_config
        }
    }
    
    ##############################
    # Initialize Execution Class #
    ##############################
    cls = RunRobotCooking(config=config)

    #### Run ####
    # cls.run()

    #### Test ####
    cls.test()