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
    '''
    functions required from env:
    - pos, quat = env.get_target_pose()
    - linear_vel, angular_vel = env.get_target_velocity()
    - object_poses = env.get_object_pose()
    - obs_info = env.get_obs_info()
    - switch_state = env.get_switch_state()
    - env.set_goal_pose(pt)
    - env.move_to(wp)
    - env.set_gripper_state(gripper_state) # 0 is open, 1 is close
    - gripper_state = env.get_gripper_state()
    '''
    def __init__(self, config={}):
        self.RunRobotCooking_config = default_config
        self.RunRobotCooking_config.update(config)

        #### Initialize env ####
        if self.RunRobotCooking_config['mode'] == 'sim':
            self.env = self.RunRobotCooking_config['SimulationEnv']['type']( self.RunRobotCooking_config['SimulationEnv']['config'])
        elif self.RunRobotCooking_config['mode'] == 'real':
            self.env = self.RunRobotCooking_config['ExperimentEnv']['type']( self.RunRobotCooking_config['ExperimentEnv']['config'])
        else:
            raise ValueError('unsupported mode')

        #### Initialize motor skills ####
        self.motor_skills = self.RunRobotCooking_config['Skills']['MotorSkills']['type'](config=self.RunRobotCooking_config['Skills']['MotorSkills']['config'])

        #### Initialize low level tl skills ####
        self.ll_tl_skills = self.RunRobotCooking_config['Skills']['LowLevelTLSkills']['type'](config=self.RunRobotCooking_config['Skills']['LowLevelTLSkills']['config'])

        #### Initialize field variables ####
        self.skill_arg = {}
        self.dry_run_target_pose = None
        
        
    def update_skill_arg(self, dry_run=True):
        curr_pos, curr_quat = self.env.get_target_pose()
        curr_pose = np.concatenate([curr_pos, curr_quat])
        if dry_run:
            if self.dry_run_target_pose is None:
                self.dry_run_target_pose = curr_pose
            else:
                curr_pose = self.dry_run_target_pose

        linear_vel, angular_vel = self.env.get_target_velocity()
        curr_vel = np.concatenate([linear_vel, angular_vel])
        
        object_poses = self.env.get_object_pose()
    
        self.skill_arg.update({
            'curr_pose': curr_pose,
            'curr_vel': curr_vel,
            'switchon': self.env.get_switch_state(),
            'gripper_state': self.env.get_gripper_state(),
            'obs_info': self.env.get_obstacle_info(),
            'obj_poses': object_poses
        })

        
    def move_to_target_with_motor_skill(self, pt, skill_name, dry_run=True):
        while True:
            self.update_skill_arg(dry_run)
            self.env.set_goal_pose(pt)
            self.skill_arg['goal'] = pt

            curr_pose = self.skill_arg['curr_pose']
            if self.RunRobotCooking_config['mode'] == 'real' and skill_name == 'flipswitchon':
                curr_pose[0] -= 0.19
                curr_pose[2] -= 0.1

            pos_dist, quat_dist = pose_distance(self.skill_arg['curr_pose'], pt) 
            if pos_dist < 0.01 and quat_dist < 0.15:
                break
        
            action = self.motor_skills.get_action(skill_name=skill_name, skill_arg=self.skill_arg)
            y = action['value']
            if self.RunRobotCooking_config['mode'] == 'real' and skill_name == 'flipswitchon':
                curr_pose[0] += 0.19
                curr_pose[2] += 0.1

            self.dry_run_target_pose = y
            self.env.move_to(y, dry_run=dry_run)

    def execute_motor_skill(self, skill_name, dry_run=True):
        from tl_utils.tl_config import KEY_POSITIONS, OBJECT_RELATIVE_POSE, STATE_IDX_MAP, PREDICATES, get_object_goal_pose

        self.update_skill_arg(dry_run)

        if skill_name == 'opengripper':
            self.env.set_gripper_state(0.3)
        elif skill_name == 'closegripper':
            self.env.set_gripper_state(0.9)
        elif skill_name == "flipswitchon":
            pt = KEY_POSITIONS['switch_on_goal']
            self.env.set_gripper_state(0.9)
            self.move_to_target_with_motor_skill(pt, skill_name='flipswitchon', dry_run=dry_run)
        elif skill_name.split('_')[0] == 'moveto':
            object_name = skill_name.split('_')[1]
            if len(skill_name.split('_')) == 3:
                object_rel_pose_name = skill_name.split('_')[2]
            else:
                object_rel_pose_name = object_name

            pt = get_object_goal_pose(self.skill_arg['obj_poses'][object_name], OBJECT_RELATIVE_POSE[object_rel_pose_name])
            if object_name == 'grill':
                # TODO: rise gripper a bit before going to grill (otherwise it chooses to go underneath)
                pt_rise = get_object_goal_pose(self.skill_arg['curr_pose'], np.array([0,0,-0.12,0,0,0,1]))
                self.move_to_target_with_motor_skill(pt_rise, skill_name='moveto', dry_run=dry_run)

            self.move_to_target_with_motor_skill(pt, skill_name='moveto', dry_run=dry_run)
            self.update_skill_arg(dry_run)
            
        else:
            raise ValueError('unsupported skill')

    def get_low_level_tl_skill_actions(self, dry_run=True):
        self.update_skill_arg(dry_run)
        s = self.RunRobotCooking_config['Skills']['construct_skill_state'](self.skill_arg)
        ll_skill_action_n_constraint, done = self.ll_tl_skills.step(s)
        return ll_skill_action_n_constraint, done
        
    def run(self, dry_run=True):
        done = False
        while not done:
            action_n_constraint, done = self.get_low_level_tl_skill_actions(dry_run)
            if not done:
                node_action = action_n_constraint['make_hotdog']['node_action']
                ee_goal = node_action['ee_goal']
                gripper_action = node_action['gripper_action']
                print("actions:", (ee_goal, gripper_action))
                
                if gripper_action is not None:
                    self.execute_motor_skill(gripper_action, dry_run=dry_run)
         
                if ee_goal is not None:
                    self.execute_motor_skill(ee_goal, dry_run=dry_run)
            else:
                print('done')

        
    def test(self, dry_run=True):
        self.run(dry_run=dry_run)
        #self.execute_motor_skill('moveto_hotdogplate', dry_run=True)
        
##############
# Run script #
##############                
def run(mode='sim', dry_run='True'):
    if dry_run == 'True':
        dry_run = True
    else:
        dry_run = False

    config = default_config
    config['mode'] = mode

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
    cls_type, cls_config = exe_config.kinova_config()
    config['ExperimentEnv'] = {
        'type': cls_type,
        'config': cls_config
    }
    
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
    cls.test(dry_run=dry_run)
                
if __name__ == "__main__":
    import fire

    fire.Fire(run)