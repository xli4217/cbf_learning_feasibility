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
    'Automata': {
        'type': None,
        'config':{}
    },
    'Skills':{
        'type': None,
        'config':{}
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

        self.aut = self.RunRobotCooking_config['Automata']['type'](config=self.RunRobotCooking_config['Automata']['config'])
        
        self.skills = self.RunRobotCooking_config['Skills']['type'](config=self.RunRobotCooking_config['Skills']['config'])

        self.skill_arg = {}
        
    def update_skill_arg(self):
        curr_pos, curr_quat = self.env.get_target_pose()
        curr_pose = np.concatenate([curr_pos, curr_quat])

        linear_vel, angular_vel = self.env.get_target_velocity()
        curr_vel = np.concatenate([linear_vel, angular_vel])

        self.skill_arg.update({
            'curr_pose': curr_pose,
            'curr_vel': curr_vel,
            'obs_info': self.env.get_obstacle_info()
        })
        
        
    def move_to_target_with_skill(self, pt, skill_name):
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

    def execute_skill(self, skill_name):
        from tl_utils.tl_config import KEY_POSITIONS, OBJECT_RELATIVE_POSE, STATE_IDX_MAP, PREDICATES

        if skill_name == 'opengripper':
            self.env.set_gripper_state(0.3)
        elif skill_name == 'closegripper':
            self.env.set_gripper_state(0.5)
        elif skill_name == "flipswitchon":
            pt = KEY_POSITIONS['switch_on_goal']
            self.move_to_target_with_skill(pt, skill_name='flipswitchon')
        else:
            raise ValueError('unsupported skill')
            
    def run(self):
        pass
        
    def test(self):
        self.execute_skill('flipswitchon')
            
if __name__ == "__main__":
    config = default_config
    config['mode'] = 'sim'

    from execution_config import ExecutionConfig
    exe_config = ExecutionConfig()
    
    ####################
    # Setup simulation #
    ####################
    from cooking_env.env.base.ce import CookingEnv

    config['SimulationEnv'] = {
        'type': CookingEnv,
        'config': exe_config.simulation_config()
    }

    ####################
    # Setup experiment #
    ####################
    # from robot_cooking.robot_cooking import RobotCooking

    ###################
    # Set up automata #
    ###################
    from tl_utils.generate_automata import GenerateAutomata
    from tl_utils.tl_config import KEY_POSITIONS, OBJECT_RELATIVE_POSE, STATE_IDX_MAP, PREDICATES
    
    config['Automata'] = {
        'type': GenerateAutomata,
        'config': {
            'formula':"F((move_to && open_gripper) && X F (close_gripper))",
            'key_positions': KEY_POSITIONS,
            'object_relative_pose': OBJECT_RELATIVE_POSE,
            'state_idx_map': STATE_IDX_MAP,
            'predicate_robustness': PREDICATES,
            'fsa_save_dir': os.path.join(os.environ['LEARNING_PATH'], 'execution', 'figures'),
            'dot_file_name': 'fsa',
            'svg_file_name': 'fsa',
            'mdp_state_space': {'type': 'float', 'shape': (22, ), 'upper_bound':[], 'lower_bound': []}
        }
    }
    

    ################
    # Setup Skills #
    ################
    from skills.skills import MotorSkills
    config['Skills']['type'] = MotorSkills
    config['Skills']['config'] = exe_config.motor_skill_config()
    
    ##############################
    # Initialize Execution Class #
    ##############################
    cls = RunRobotCooking(config=skills_config)

    #### Run ####
    # cls.run()

    #### Test ####
    cls.test()