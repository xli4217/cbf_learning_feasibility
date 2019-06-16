import os
import numpy as np
import time
from future.utils import viewitems


default_config = {
    'TrajGenerator':{
        'type': None,
        'config': None
    },
    'LearnedSkills':{
        'skill_name': {
            #### state space of the trained policy #####
            'state_space': None,
            #### action space of the trained policy ####
            'action_space': None,
            'training_config_restore_path': None,
            'policy_restore_path': None,
            'state_preprocessor_restore_path': None
        }
    }
}

class MotorSkills(object):

    def __init__(self, config={}):
        self.Skills_config = default_config
        self.Skills_config.update(config)

        self.traj_generator = self.Skills_config['TrajGenerator']['type'](self.Skills_config['TrajGenerator']['config'])

        self.learned_skills_dict = {}
        self.load_learned_skills()
        
    def load_learned_skills(self):
        from utils.utils import load_policy_and_preprocessor

        for skill_name, skill_config in viewitems(self.Skills_config['LearnedSkills']):
            policy, state_preprocessor = load_policy_and_preprocessor(skill_config)
            self.learned_skills_dict[skill_name] = {
                'policy': policy,
                'state_preprocessor': state_preprocessor
            }

            
    def skill_closegripper(self):
        percentage_close = 0.3
        return percentage_close

    def skill_opengripper(self):
        percentage_close = 0.9
        return percentage_close

    def skill_moveto(self,
                     goal,
                     forcing,
                     curr_pose,
                     curr_vel,
                     obs_info,
                     translation_gen=None,
                     orientation_gen=None):
        
        self.traj_generator.set_goal(goal)
        ddy, dy ,y = self.traj_generator.get_next_wp(action=forcing,
                                                     curr_pose=curr_pose,
                                                     curr_vel=curr_vel,
                                                     obs_info=obs_info,
                                                     translation_gen=translation_gen,
                                                     orientation_gen=orientation_gen)
        return ddy, dy, y

    def skill_flipswitchon(self, goal, curr_pose, curr_vel, obs_info):
        self.traj_generator.set_goal(goal)

        policy = self.learned_skills_dict['flipswitchon']['policy']
        state_preprocessor = self.learned_skills_dict['flipswitchon']['state_preprocessor']

        state_dim = self.Skills_config['LearnedSkills']['flipswitchon']['state_space']['shape'][0]
        
        s = curr_pose[:state_dim]

        if state_preprocessor is not None:
            s = state_preprocessor.get_scaled_x(s)
            
        forcing = policy.get_action(s, deterministic=True).flatten()

        forcing *= 100

        forcing = 0
        action_space = self.Skills_config['LearnedSkills']['flipswitchon']['action_space']
        forcing = np.clip(forcing, action_space['lower_bound'], action_space['upper_bound'])
        forcing = np.concatenate([forcing, np.zeros(3)])

        ddy, dy, y = self.traj_generator.get_next_wp(action=forcing,
                                                     curr_pose=curr_pose,
                                                     curr_vel=curr_vel,
                                                     obs_info=obs_info)
        return ddy, dy , y
 
        
    def get_action(self, skill_name=None, skill_arg={}):
        if skill_name == 'closegripper':
            action = {
                'description': 'percentage_gripper_close',
                'value': self.skill_closegripper()
            }
            return action
        elif skill_name == 'opengripper':
            action = {
                'description': 'percentage_gripper_close',
                'value': self.skill_opengripper()
            }
            return action
        elif skill_name == 'moveto':
            curr_pose = skill_arg['curr_pose']
            curr_vel = skill_arg['curr_vel']
            obs_info = skill_arg['obs_info']
            goal = skill_arg['goal']

            ddy, dy, y = self.skill_moveto(goal=goal,
                                            forcing=np.zeros(6),
                                            curr_pose=curr_pose,
                                            curr_vel=curr_vel,
                                            obs_info=obs_info)
            action = {
                'description': 'target position',
                'value': y
            }

            return action
        elif skill_name == 'flipswitchon':
            curr_pose = skill_arg['curr_pose']
            curr_vel = skill_arg['curr_vel']
            obs_info = skill_arg['obs_info']
            goal = skill_arg['goal']

            ddy, dy, y = self.skill_flipswitchon(goal=goal,
                                                 curr_pose=curr_pose,
                                                 curr_vel=curr_vel,
                                                 obs_info=obs_info)
            action = {
                'description': 'target position',
                'value': y
            }

            return action
        else:
            raise ValueError('skill {} not supported'.format(skill_name))


if __name__ == "__main__":
    from execution.execution_config import ExecutionConfig
    exe_config = ExecutionConfig()

    config = exe_config.motor_skill_config()

    ##############
    # Initialize #
    ##############
    cls = MotorSkills(config)


    ########
    # Test #
    ########
    cls.get_action(skill_name='open_gripper', skill_arg={})