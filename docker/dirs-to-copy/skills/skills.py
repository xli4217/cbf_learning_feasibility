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

class Skills(object):

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

            
    def skill_close_gripper(self):
        percentage_close = 0.3
        return percentage_close

    def skill_open_gripper(self):
        percentage_close = 0.9
        return percentage_close

    def skill_move_to(self, goal, forcing, curr_pose, curr_vel, obs_info):
        self.traj_generator.set_goal(goal)
        ddy, dy ,y = self.traj_generator.get_next_wp(action=forcing,
                                                     curr_pose=curr_pose,
                                                     curr_vel=curr_vel,
                                                     obs_info=obs_info)
        return ddy, dy, y

    def skill_flip_switch_on(self, goal, curr_pose, curr_vel, obs_info):
        self.traj_generator.set_goal(goal)

        policy = self.learned_skills_dict['flip_switch_on']['policy']
        state_preprocessor = self.learned_skills_dict['flip_switch_on']['state_preprocessor']

        state_dim = self.Skills_config['LearnedSkills']['flip_switch_on']['state_space']['shape'][0]
        
        s = curr_pose[:state_dim]

        if state_preprocessor is not None:
            s = state_preprocessor.get_scaled_x(s)

        forcing = policy.get_action(s, deterministic=True)
        forcing *= 100

        action_space = self.Skills_config['LearnedSkills']['flip_switch_on']['action_space']
        forcing = np.clip(forcing, action_space['lower_bound'], action_space['upper_bound'])
        forcing = np.concatenate([action, np.zeros(3)])

        ddy, dy, y = self.traj_generator.get_next_wp(action=forcing,
                                                     curr_pose=curr_pose,
                                                     curr_vel=curr_vel,
                                                     obs_info=obs_info)
        return ddy, dy , y

        
    def get_action(self, skill_name=None, skill_arg={}):
        if skill_name == 'close_gripper':
            action = {
                'description': 'percentage_gripper_close',
                'value': self.skill_close_gripper()
            }
            return action
        elif skill_name == 'open_gripper':
            action = {
                'description': 'percentage_gripper_close',
                'value': self.skill_open_gripper()
            }
            return action
        elif skill_name == 'move_to':
            curr_pose = skill_arg['curr_pose']
            curr_vel = skill_arg['curr_vel']
            obs_info = skill_arg['obs_info']
            goal = skill_arg['goal']

            ddy, dy, y = self.skill_move_to(goal=goal,
                                            action=np.zeros(6),
                                            curr_pose=curr_pose,
                                            curr_vel=curr_vel,
                                            obs_info=obs_info)
            action = {
                'description': 'target position',
                'value': y
            }

            return action
        elif skill_name == 'flip_switch_on':
            curr_pose = skill_arg['curr_pose']
            curr_vel = skill_arg['curr_vel']
            obs_info = skill_arg['obs_info']
            goal = skill_arg['goal']

            ddy, dy, y = self.skill_flip_switch_on(goal=goal,
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
    ###############################
    # Set up trajectory generator #
    ###############################
    from traj_generators.trajectory_generator import TrajectoryGenerator
    config = default_config
    config['TrajGenerator'] = {
        'type': TrajectoryGenerator,
        'config': {
            'dmp_config': {
                # gain on attractor term y dynamics (linear)
                'ay': 55,
                # gain on attractor term y dynamics (linear)
                'by': None,
                # gain on attractor term y dynamics (angular)
                'az': 55,
                # gain on attractor term y dynamics (angular)
                'bz': None,
                # timestep
                'dt': 0.001,
                # time scaling, increase tau to make the system execute faster
                'tau': 1.,
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
                # if True, then update according to dmp_pose, else update according to current pose
                'use_dmp_pose': True,
                'n_linear_dmp': 3,
                'n_angular_dmp': 3
                
            },
            'clf_cbf_config': {
                'k_cbf': 1.1,
                'epsilon':0.8,
                'num_states':3,
                'action_space': {'shape': (3,), 'upper_bound': [0.1, 0.1, 0.1], 'lower_bound': [-0.1,-0.1,-0.1]},
                'use_own_pose': True,
                'dt': 0.015
            },
            #'translation_gen': 'clf_cbf',
            'translation_gen': 'dmp',
            'orientation_gen': 'dmp'            
        }
    }

    #######################
    # Load learned skills #
    #######################

    #### flip switch open ####
    experiment_root_dir = os.path.join(os.environ['LEARNING_PATH'], 'learning', 'experiments')
    experiment_name = 'test'
    hyperparam_dir = 'seed0'
    itr = 0

    skill_config = {
        "state_space": {'type': 'float', 'shape': (3, ), "upper_bound": [], 'lower_bound': []},
        "action_space": {'type': 'float', 'shape': (3, ), "upper_bound": 70*np.ones(3), 'lower_bound': -70*np.ones(3)},
        "training_config_restore_path": os.path.join(experiment_root_dir, experiment_name, 'config', hyperparam_dir, 'config.pkl'),
        "policy_restore_path": os.path.join(experiment_root_dir, experiment_name, 'transitions', hyperparam_dir, 'itr_'+str(itr)),
        "state_preprocessor_restore_path": os.path.join(experiment_root_dir, experiment_name, 'info', hyperparam_dir, 'state_preprocessor_params.pkl')
    }
    config['LearnedSkills']['flip_switch_on'] = skill_config


    ##############
    # Initialize #
    ##############
    cls = Skills(config)


    ########
    # Test #
    ########
    cls.get_action(skill_name='open_gripper', skill_arg={})