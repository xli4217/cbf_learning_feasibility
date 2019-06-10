import numpy as np
from future.utils import viewitems
import os
import copy
from rl_pipeline.configuration.configuration import Configuration

import json

env_cal = {}
default_config = {
    'env_name': "vrep",
    'robot': 'jaco',
    'headless': False,
    # this list can contain 'mdp', 'fsa', 'cbf', 'dmp' 
    'components': {'mdp': True, 'fsa': False, 'cbf': False, 'dmp': False},
    # this can be 'makehotdog (for jaco+mdp+fsa)', 'serve (for baxter+mdp+fsa)', 'switchon (for jaco + mdp)'
    'task': 'switchon',
    'fsa_save_dir': os.getcwd(),
    'fsa_name': 'g'
}

class ExperimentConfig(object):

    def __init__(self, config={}):
        self.ExperimentConfig_config = Configuration(default_config)
        self.ExperimentConfig_config.update(config)

        self.env_name = self.ExperimentConfig_config.get('env_name')
        self.headless = self.ExperimentConfig_config.get('headless')
        self.components = self.ExperimentConfig_config.get('components')
        self.robot = self.ExperimentConfig_config.get('robot')
        
        # used in batch_sampler to post-process rewards
        self.process_rewards = None
        self.Environment = None

        self.Environment = self.construct_env()

    def construct_env(self):
        #### traj generator ####
        if not self.components['cbf'] and not self.components['dmp']:
            translation_gen = None
        elif not self.components['dmp'] and self.components['cbf']:
            translation_gen = 'cbf'
        elif self.components['dmp'] and not self.components['cbf']:
            translation_gen = 'clf'
        elif self.components['dmp'] and self.components['cbf']:
            translation_gen = 'clf_cbf'
        else:
            raise ValueError('invalid combination of translation_gen')
            
        traj_gen_type, traj_gen_config = self.construct_traj_generator(translation_gen=translation_gen)

        #### MDP ####
        if self.components['mdp'] and not self.components['fsa']:
            if self.ExperimentConfig_config.get('task') == 'switchon' and self.robot == 'jaco':
                get_state, get_reward, is_done, state_space, action_space, other = self.switch_task_mdp_config(on_or_off='on')
            else:
                raise ValueError('task and robot not match')

            construct_mdp_env_config = {
                'get_state': get_state,
                'get_reward': get_reward,
                'is_done': is_done,
                'state_space': state_space,
                'action_space': action_space,
                'other': other
            }
            mdp_env_type, mdp_env_config = self.construct_mdp_env(mdp_config=construct_mdp_env_config,
                                                                  traj_gen_type=traj_gen_type,
                                                                  traj_gen_config=traj_gen_config)
            Environment = {
                'type': mdp_env_type,
                'config': mdp_env_config
            }

            return Environment

        #### FSA ####
        elif self.components['mdp'] and self.components['fsa']:
            if self.ExperimentConfig_config.get('task') == 'makehotdog' and self.robot == 'jaco':
                task_spec, construct_state, predicate_robustness, obs_dim = self.get_tl_related(task='makehotdog')
                get_state, get_reward, is_done, state_space, action_space, other = self.makehotdog_task_mdp_config(get_state_fn=construct_state, obs_dim=obs_dim)
      
            elif self.ExperimentConfig_config.get('task') == 'serve' and self.robot == 'baxter':
                pass
            else:
                raise ValueError('task and robot not match')


            construct_mdp_env_config = {
                'get_state': get_state,
                'get_reward': get_reward,
                'is_done': is_done,
                'state_space': state_space,
                'action_space': action_space,
                'other': other
            }
            mdp_env_type, mdp_env_config = self.construct_mdp_env(mdp_config=construct_mdp_env_config,
                                                                  traj_gen_type=traj_gen_type,
                                                                  traj_gen_config=traj_gen_config)

            from learning.env.fsa_augmented_env import FsaAugmentedEnv

            fsa_augmented_env_config = {
                'fsa_save_dir': self.ExperimentConfig_config.get('fsa_save_dir'),
                'dot_file_name': self.ExperimentConfig_config.get('fsa_name'),
                'svg_file_name': self.ExperimentConfig_config.get('fsa_name'),
                'base_env': {
                    'type': mdp_env_type,
                    'config': mdp_env_config,
                },
                'spec': {
                    'predicate_form': task_spec,
                    'predicate_robustness': predicate_robustness
                }
            }
            
            Environment = {
                'type': FsaAugmentedEnv,
                'config': fsa_augmented_env_config
            }

            return Environment
        else:
            raise ValueError('can not have fsa without mdp')
            return None
        
                
    def construct_mdp_env(self, mdp_config={}, traj_gen_type=None, traj_gen_config=None):
        self.reset = {
            'type': None,
            'config': {}
        }
        
        if self.env_name == 'vrep':            
            from learning.env.learning_env import LearningEnv
            from cooking_env.env.base.ce import CookingEnv

            self.reset = mdp_config['other']['reset']
        
            self.mdp_env_type = LearningEnv
            self.mdp_env_config = {
                # Common to all envs
                "seed": 10,
                "state_space": mdp_config['state_space'],
                "action_space": mdp_config['action_space'],
                "get_state": mdp_config['get_state'],
                "get_reward": mdp_config['get_reward'],
                "is_done": mdp_config['is_done'],
                "get_info": None,
                #### class specific ####
                "WPGenerator": {
                    'type': traj_gen_type,
                    'config': traj_gen_config
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

        else:
            raise ValueError('unsupported environment')

        return self.mdp_env_type, self.mdp_env_config

    def construct_traj_generator(self, translation_gen="clf_cbf", orientation_gen='dmp'):
        from traj_generators.trajectory_generator import TrajectoryGenerator

        config =  {
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
                'dt': 0.005,
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
                'action_space': {'shape': (3,), 'upper_bound': 0.2 * np.ones(3), 'lower_bound': -0.2 * np.ones(3)},
                'use_own_pose': True,
                'dt': 0.05,
                'log_dir': os.path.join(os.environ['LEARNING_PATH'], 'execution', 'log')
            },
            'translation_gen': translation_gen,
            'orientation_gen': orientation_gen
        }

        return TrajectoryGenerator, config


    def serve_task_mdp_config(self):        
        #### State ####
        def get_state(all_info):
            return mdp_state

        #### Reward ####
        def get_reward(state=None, action=None, next_state=None, all_info=None):
            return r
            
        
        #### Done ####
        def is_done(state=None, action=None, next_state=None, all_info=None):
            return done
            
        state_space = {'type': 'float', 'shape': (3, ), 'upper_bound': [], 'lower_bound': []}

        action_coeff = 120
        action_space = {'type': 'float', 'shape': (3, ), "upper_bound": np.ones(3) * action_coeff, "lower_bound": -np.ones(3) * action_coeff}


        #### Reset ####
        reset_config = {}
        
        from learning.reset.experiment_env_reset import ExperimentEnvVrepReset

        other = {'reset': {'type': None, 'config': reset_config}}

        return get_state, get_reward, is_done, state_space, action_space, other

   
        
    def makehotdog_task_mdp_config(self, get_state_fn=None, obs_dim=None):        
        #### State ####
        def get_state(all_info):
            if not get_state:
                raise ValueError('need to provide get_state')
            else:
                mdp_state = get_state_fn(all_info)
            return mdp_state

        #### Reward ####
        def get_reward(state=None, action=None, next_state=None, all_info=None):
            return 0
            
        #### Done ####
        def is_done(state=None, action=None, next_state=None, all_info=None):
            return False
            
        state_space = {'type': 'float', 'shape': (obs_dim, ), 'upper_bound': [], 'lower_bound': []}

        action_coeff = 120
        action_space = {'type': 'float', 'shape': (3, ), "upper_bound": np.ones(3) * action_coeff, "lower_bound": -np.ones(3) * action_coeff}


        #### Reset ####
        reset_config = {}
        
        from learning.reset.experiment_env_reset import ExperimentEnvVrepReset

        other = {'reset': {'type': None, 'config': reset_config}}

        return get_state, get_reward, is_done, state_space, action_space, other

        
    def switch_task_mdp_config(self, on_or_off='on'):

        #### State ####
        def get_state(all_info):
            # mdp_state = np.array(list(all_info['target_pos']) + \
            #                      list(all_info['target_quat']))
            mdp_state = np.array(all_info['target_pos'])
            return mdp_state

        #### Reward ####
        def get_reward(state=None, action=None, next_state=None, all_info=None):
            button_joint_angle = all_info['button_joint_angle']
            button_rel_pose = all_info['button_rel_pose']
            button_vel = all_info['button_vel']
            button_joint_frame_angle = all_info['button_joint_frame_angle']

            r = -10*(button_joint_frame_angle[2] - 1.15)

            if button_joint_frame_angle[2] < 0.6:
                r += 5.

            motion_range = all_info['motion_range']
            low = np.array([motion_range['x'][0], motion_range['y'][0], motion_range['z'][0]])
            high = np.array([motion_range['x'][1], motion_range['y'][1], motion_range['z'][1]])
            target_pos = state[:3]

            ## done if move outside of motion region
            if any(target_pos < low) or any(target_pos > high):
                r -= 1.5

            button_disturbance =  np.linalg.norm(np.concatenate([button_vel[:3], button_vel[-2:]]))
            if button_disturbance > 0.1:
                r = -2.
            
            return r
            
        
        #### Done ####
        def is_done(state=None, action=None, next_state=None, all_info=None):
            done = False
            motion_range = all_info['motion_range']
            low = np.array([motion_range['x'][0], motion_range['y'][0], motion_range['z'][0]])
            high = np.array([motion_range['x'][1], motion_range['y'][1], motion_range['z'][1]])
            target_pos = state[:3]

            ## done if move outside of motion region
            if any(target_pos < low) or any(target_pos > high):
                print("done: moved outside of motion region")
                done = True

            ## done if reached goal
            goal_dist = np.linalg.norm(target_pos - all_info['goal'][:3])
            if goal_dist < 0.01:
                print("done: reached goal")
                done = True

            ## done if hit button 
            button_vel = all_info['button_vel']
            button_disturbance =  np.linalg.norm(np.concatenate([button_vel[:3], button_vel[-2:]]))
            if button_disturbance > 0.1:
                print('done: button pushed away from nominal')
                done = True

            ## done if finished task
            toaster_joint_frame_angle = all_info['button_joint_frame_angle'][2]
            # print(all_info['button_joint_frame_angle'][2])
            if toaster_joint_frame_angle < 0.57:
                print('done: turn on task done')
                done = True
                
            return done
            
        state_space = {'type': 'float', 'shape': (3, ), 'upper_bound': [], 'lower_bound': []}

        action_coeff = 120
        action_space = {'type': 'float', 'shape': (3, ), "upper_bound": np.ones(3) * action_coeff, "lower_bound": -np.ones(3) * action_coeff}


        #### Reset ####
        reset_config = {}
        
        from learning.reset.experiment_env_reset import ExperimentEnvVrepReset

        other = {'reset': {'type': None, 'config': reset_config}}

        return get_state, get_reward, is_done, state_space, action_space, other

    def get_tl_related(self, task='makehotdog'):
        from tl_utils.tl_config import TLConfig

        tlconfig = TLConfig({'robot': self.robot})

        if task == 'makehotdog' and self.robot == 'jaco':
            task_spec = "moveto_world_jaconeutral && X F" + \
                        "((closegripper) && X F (" + \
                        "(moveto_hotdogplate && opengripper) && X F " + \
                        "(closegripper && X F " + \
                        "((moveto_grill && closegripper) && X F " + \
                        "(opengripper && X F "+ \
                        "(moveto_world_jaconeutral && X F " + \
                        "((moveto_grill && opengripper) && X F " + \
                        "(closegripper && X F "+\
                        "((moveto_bunplate && closegripper) && X F "+\
                        "(opengripper && X F " + \
                        "(moveto_world_jaconeutral" + \
                        ")))))))))))"

            apply_condiment_ = "(moveto_condiment_condimentpre && opengripper) && X F " + \
                               "(moveto_condiment_condimentpost && X F " + \
                               "(closegripper && X F "+ \
                               "((moveto_bunplate_relativeplateapplycondimentpost && closegripper) && X F "+\
                               "((moveto_world_placecondimentgoal && closegripper) && X F" + \
                               "(opengripper && X F" + \
                               "(moveto_condiment_condimentpre" + \
                               "))))))"

            
            task_spec = "F(" + apply_condiment_ + ")"
            
        elif task == 'serve' and self.robot == 'baxter':
            serve = "(moveto_bunplate && opengripper) && X F " + \
                    "(closegripper && X F " + \
                    "((moveto_serveplate && closegripper) && X F " + \
                    "(opengripper  && X F "+ \
                    "(moveto_world_baxterneutral " + \
                    "))))"

            

            #### serve task KG ####
            serve_task_KB = "G (!(moveto_serveplate && moveto_bunplate)) && " + \
                            "G (!(opengripper && closegripper)) && " + \
                            "G (!(moveto_serveplate && moveto_world_baxterneutral)) && " + \
                            "G (!(moveto_bunplate && moveto_world_baxterneutral)) && " + \
                            "G (!(moveto_serveplate && moveto_bunplate && moveto_world_baxterneutral))"
     

            #### FSA version
            # serve_task_ = "(( inservezone_serveplate -> X F (" + serve + ")) && (!inservezone_serveplate -> X F moveto_world_baxterneutral))" + " && " + serve_task_KB

            task_spec = "F(" +  serve + " )&& (! (" + serve + ") U inservezone_serveplate)" + " && " + serve_task_KB
            
            #### Buchi version 
            # serve_task_ = "G (( inservezone_serveplate -> X F (" + serve + ")))" + " && " + serve_task_KB

        else:
            raise ValueError('task and robot do not match')


        return task_spec, tlconfig.construct_skill_state, tlconfig.PREDICATES, tlconfig.obs_dim
            
if __name__ == "__main__":
    pass