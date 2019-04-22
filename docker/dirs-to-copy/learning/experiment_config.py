import numpy as np
from future.utils import viewitems
import os
import copy
from rl_pipeline.configuration.configuration import Configuration

import json

# env_cal = json.loads(open(os.path.join(os.environ['RLFPS_PATH'], 'examples', 'iros2019', 'env_config.json'), 'rb').read())

env_cal = {}
default_config = {
    # this can be 'particle', 'vrep_baxter', 'baxter'
    'env_name': "vrep_baxter",
    'task': 'tl_flat',
    'tl_with_mdp_reward': False,
    'tl_with_mdp_done': False,
    'dist_th': 0.05,
    'fsa_save_dir': os.getcwd(),
    'fsa_name': 'fsa',
    'headless': False,
    'dmp': False,
    'clf': False,
    'softmax': False,
    'beta': 5.0,
    'other': {
        # for particle env
        'channel_width': 0.4,
        'topk':2,
    }
}

class ExperimentConfig(object):

    def __init__(self, config={}):
        self.ExperimentConfig_config = Configuration(default_config)
        self.ExperimentConfig_config.update(config)

        self.env_name = self.ExperimentConfig_config.get('env_name')
        self.task = self.ExperimentConfig_config.get('task')
        self.dist_th = self.ExperimentConfig_config.get('dist_th')
        self.headless = self.ExperimentConfig_config.get('headless')
        
        # used in batch_sampler to post-process rewards
        self.process_rewards = None
        self.Environment = None

        self.construct_mdp_env()
        if 'tl' in self.task:
            self.construct_fsa_env(tl_with_mdp_reward=self.ExperimentConfig_config.get('tl_with_mdp_reward'),
                                   tl_with_mdp_done=self.ExperimentConfig_config.get('tl_with_mdp_done'))

    def construct_mdp_env(self):
        self.reset = {
            'type': None,
            'config': {}
        }
        if self.env_name == 'particle':            
            get_state, get_reward, is_done, self.state_space, self.action_space, other = self.particle_env_config(topk=self.ExperimentConfig_config.get(['other','topk']), goals=None, obstacles=None)

            self.mdp_env_type = other['env_type']
            self.mdp_env_config = other['env_config']
            
        elif self.env_name == 'vrep_baxter':            
            from examples.iros2019.env.experiment_env_vrep import ExperimentEnvVrep

            self.get_state, self.get_reward, self.is_done, self.state_space, self.action_space, env_other = self.baxter_env_config()
            self.reset = env_other['reset']
            
            self.process_rewards = None
         
            self.mdp_env_type = ExperimentEnvVrep
            self.mdp_env_config = {
                # this can be 'bag' which subscribes from rosbag, or 'deploy'
                'mode': 'deploy',
                'seed': 0,
                'suffix': "",
                'bag_path': None,
                'env_config_json': os.path.join(os.environ['RLFPS_PATH'], 'examples', 'rss2019', 'env_config.json'),
                'base_env_config': {
                    "seed": 10,
                    "state_space": self.state_space,
                    "action_space": self.action_space,
                    "get_state": self.get_state,
                    "get_reward": self.get_reward,
                    "is_done": self.is_done,
                    "get_info": None,
                    # specific to this env
                    "suffix": "",
                    "arm": "right",
                    "control_mode": "velocity",
                }
            }

        elif self.env_name == 'baxter':
            from rl_pipeline.env.ros_env.baxter_env import BaxterEnv
            self.mdp_env_type = BaxterEnv
            self.mdp_env_config = {
                'control_mode': 'velocity',
                "control_rate": 20,
                "seed": 10,
                "state_space": self.state_space,
                "action_space": self.action_space,
                "get_state": self.get_state,
                "get_reward": self.get_reward,
                "is_done": self.is_done,
                "get_info": None,
                'baxter_utils_config': {
                    'arm': "right",
                    'env_json_path': os.path.join(os.environ['RLFPS_PATH'],
                                                  'examples',
                                                  'rss2019',
                                                  'env_config.json'),
                }

            }
        else:
            raise ValueError('unsupported environment')

        self.Environment = {
            'type': self.mdp_env_type,
            'config': self.mdp_env_config
        }
            
    def construct_fsa_env(self, tl_with_mdp_reward=False, tl_with_mdp_done=False):
        if 'baxter' in self.env_name:
            self.spec, spec_other = self.baxter_tl_task(task=self.task, dist_th=self.dist_th)
        elif self.env_name == 'particle':
            self.spec, spec_other = self.particle_tl_task(task=self.task, dist_th=self.dist_th)
        else:
            raise ValueError('environment unsupported')
            
        self.process_rewards = spec_other['robustness_reward']


        from rl_pipeline.env.fsa_augmented_env.fsa_augmented_env import FsaAugmentedEnv
            
        if 'base_env_config' in self.mdp_env_config.keys():
            c = self.mdp_env_config['base_env_config']
        else:
            c = self.mdp_env_config
            
        if not tl_with_mdp_reward:
            c['get_reward'] = lambda state=None, action=None, next_state=None: (0., 0.)
        if not tl_with_mdp_done:
            c['is_done'] = lambda state=None, action=None, next_state=None: False

            
        fsa_augmented_env_config = {
            'fsa_save_dir': self.ExperimentConfig_config.get('fsa_save_dir'),
            'dot_file_name': self.ExperimentConfig_config.get('fsa_name'),
            'svg_file_name': self.ExperimentConfig_config.get('fsa_name'),
            'softmax': self.ExperimentConfig_config.get('softmax'),
            'beta': self.ExperimentConfig_config.get('beta'),
            'base_env': {
                'type': self.mdp_env_type,
                'config': self.mdp_env_config
            },
            'spec': {
                'predicate_form': self.spec['predicate_form'],
                'full_form': None,
                'predicate_robustness': self.spec['predicate_robustness'],
                'goal_from_state': self.spec['goal_from_state']
            },
            'visdom': False
        }

        if self.ExperimentConfig_config.get('dmp') or self.ExperimentConfig_config.get('clf'):
            from examples.iros2019.env.fsa_augmented_dmp_env import FsaAugmentedDMPEnv
            
            fsa_augmented_dmp_env_config = {
                'base_env_config': fsa_augmented_env_config
            }

            self.Environment = {
                'type': FsaAugmentedDMPEnv,
                'config': fsa_augmented_dmp_env_config
            }
        else:
            self.Environment = {
                'type': FsaAugmentedEnv,
                'config': fsa_augmented_env_config
            }

            
    def baxter_env_config(self):
        def get_state(all_info):

            obj_of_interest = ['red', 'green', 'blue', 'hand']
            obj_pose = [0] * 12
        
            for k, v in viewitems(all_info['object_pos']):
                if k in obj_of_interest:
                    pos = v[:3]
                    if k == 'red':
                        obj_pose[0:3] = pos
                    if k == 'green':
                        obj_pose[3:6] = pos
                    if k == 'blue':
                        obj_pose[6:9] = pos
                    if k == 'hand':
                        obj_pose[9:12] = pos
                        
            mdp_state = list(all_info['joint_angles']) + list(obj_pose) + list(all_info['ee_pose'][:3])
   
            return np.array(mdp_state)

        def get_reward(state=None, action=None, next_state=None):
            ee_pos = next_state[-3:]
            red_pos = next_state[7:10]
            rel_pos = np.array(red_pos) - np.array(ee_pos)
            
            sqrt_dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
            # th_precise = dist_th**2 - np.exp(-100*dist_th**2)
            # r_precise = -(sqrt_dist**2 - np.exp(-100*sqrt_dist**2))
            
            return -sqrt_dist
        

        def is_done(state=None, action=None, next_state=None):
            ee_pos = state[-3:]
            red_pos = state[7:10]
            rel_pos = np.array(red_pos) - np.array(ee_pos)
            
            sqrt_dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
            if sqrt_dist < self.dist_th:
                return True
            else:
                return False
            
            
        state_space = {'type': 'float', 'shape': (22, ), 'upper_bound': [], 'lower_bound': []}

        if self.env_name == 'vrep_baxter':
            action_coeff = 0.7
        elif self.env_name == 'baxter':
            action_coeff = 0.1
        action_space = {'type': 'float', 'shape': (7, ), "upper_bound":np.array([1.5,1.5,1.5,1.5,3.5,3.5,3.5]) * action_coeff, "lower_bound": np.array([-1.5,-1.5,-1.5,-1.5,-3.5,-3.5,-3.5]) * action_coeff}

        reset_config = {
            'object_poses': {
                'red': {
                    'randomize': True,
                    'nominal': [0.7, -0.45, -0.06],
                    'min': env_cal['red']['sample_range_min'],
                    'max': env_cal['red']['sample_range_max']
                },
                'green': {
                    'randomize': True,
                    'nominal': [0.7, -0.45, -0.046],
                    'min': env_cal['green']['sample_range_min'],
                    'max': env_cal['green']['sample_range_max']
                },
                'blue': {
                    'randomize': True,
                    'nominal': [0.7, -0.45, -0.046],
                    'min': env_cal['blue']['sample_range_min'],
                    'max': env_cal['blue']['sample_range_max']
                },
                'hand': {
                    'randomize': True,
                    'nominal': [0, 0, 0],
                    'min': np.array(env_cal['appear_in_sight']['min']) - 0.1,
                    'max': np.array(env_cal['appear_in_sight']['max']) + 0.1
                }
            },
            'joint_angles': {
                'randomize': True,
                'nominal': [ 0.5564515308054339, -1.1094516048381255, 0.006135923151541655, 0.9990049881103757, 0.15033011721277054, 1.5780827355371194, -0.11888351106111957],
                "min": [0.19538355, -0.95567003, -0.69987873,  0.05867477, -0.13805827, 0.39269908, -1.72227693],
                "max": [0.6259730e+00, -5.29223372e-02,  1.16199045e-01, 1.58268468e+00,  1.10484966e+00,  1.71997596e+00,  1.52631088e-01]
            }
        }

        from examples.rss2019.reset.experiment_env_reset import ExperimentEnvVrepReset

        other = {'reset': {'type': ExperimentEnvVrepReset, 'config': reset_config}}

        return get_state, get_reward, is_done, state_space, action_space, other
        
    def baxter_tl_task(self, task=1.1, dist_th=0.05):
        def in_region(abs_pos, ee_pose):
            rel_pos = np.array(abs_pos) - np.array(ee_pose)
            sqrt_dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
            r_sqrt = dist_th - sqrt_dist
            # r_log = np.log(dist_th) - np.log(sqrt_dist)
            
            # th_precise = dist_th**2 - np.exp(-100*dist_th**2)
            # r_precise = th_precise - (sqrt_dist**2 - np.exp(-100*(sqrt_dist**2)))
            # print(r_precise)
            
            r_shaped = 1. - np.power((sqrt_dist / dist_th),0.4)
            return r_sqrt

        def appear_in_sight(hand_abs_pos):
            bounds = {'min': env_cal['appear_in_sight']['min'],
                      'max': env_cal['appear_in_sight']['max']}
            r = min(bounds['max'][0] - hand_abs_pos[0], hand_abs_pos[0] - bounds['min'][0],
                    bounds['max'][1] - hand_abs_pos[1], hand_abs_pos[1] - bounds['min'][1],
                    bounds['max'][2] - hand_abs_pos[2], hand_abs_pos[2] - bounds['min'][2])
            
            if r < 0:
                r = -10
            return r

            
        #####################
        # Define predicates #
        #####################

        # to be inside a region, gripper frame at most 7cm above the the plate, distance < radius(10 cm) from center of the plate in the XY plane

            
        # r red(gripper)
        r = {
            "predicate": "red",
            "object": "gripper",
            'predicate_eval': lambda s, a, sp: in_region(sp[7:10], sp[-3:]) # these are all mdp states without q pre-appended
        }
    
        # g is green(gripper)
        g = {
            "predicate": "green",
            "object": "gripper",
            "predicate_eval": lambda s, a, sp: in_region(sp[10:13], sp[-3:])   
        }
        
        # b is blue(gripper)
        b = {
            "predicate": "blue",
            "object": "gripper",
            "predicate_eval": lambda s, a, sp: in_region(sp[13:16], sp[-3:])  
        }

        # h is go_to_hand(gripper)
        h = {
            "predicate": "go_to_hand",
            "object": "gripper",
            "predicate_eval": lambda s, a, sp: in_region(sp[16:19], sp[-3:])  
        }

        # ha is appear_in_sight(hand)
        ha = {
            "predicate": "appear_in_sight",
            "object": "hand",
            "predicate_eval": lambda s, a, sp: appear_in_sight(sp[16:19])
        }

        # these are all distances of r, g, b, h wrt robot gripper
        rs = "r < " + str(dist_th)
        gs = "g < " + str(dist_th)
        bs = "b < " + str(dist_th)
        hs = "h < " + str(dist_th)
        # hx, hy, hz are coordinates of human hand
        has = "(hx > " + str(0.72-0.76/3) + ") && (hx < " + str(0.72+0.76/3) + ") && ( hy > " + str(-0.52-1.22/3) + ") && (hy < " + str(-0.52+1.22/3) + ") && ( hz > " + str(-0.06+0.1) + ") && (hz < " + str(-0.06+0.3) + ")"

        first_skill_predicate_from = ""
        second_skill_predicate_form = ""

        if task == 'tl_1.1':
            predicate_form = "F r && F g"
            state_predicate_form = "F " + rs + " && F " + gs
        elif task == 'tl_1.2':
            predicate_form = "F g && F b"
            state_predicate_form = "F " + gs + " && F " + bs
        elif task == 'tl_1.3':
            predicate_form = "(F r && F g) || (F g && F b)"
            state_predicate_form = None
        elif task == 'tl_2.1':
            predicate_form = "F (r && F( g && F b))"
            state_predicate_form = "F (" + rs + " && F( " + gs +" && F " + bs +"))"
            first_skill_predicate_from = None
            second_skill_predicate_form = None
        elif task == 'tl_2.2':
            predicate_form = "(ha -> F h) U b "
            state_predicate_form = "(" + has + " -> F" + hs + ") U " + bs
            first_skill_predicate_from = None
            second_skill_predicate_form = None
        elif task == 'tl_2.3' or task == 'compose':
            predicate_form = "(F (r && F( g && Fb)))" + " && ((ha -> F h) U b)"
            state_predicate_form = "(F (" + rs + " && F( " + gs +" && F " + bs +")))" + "&& ((" + has + " -> F" + hs + ") U " + bs + ")"
            first_skill_predicate_from = "(F (r && F( g && Fb)))"
            second_skill_predicate_form = "((ha -> F h) U b)"
        else:
            raise ValueError("task not supported")

        spec = {
            "predicate_form": predicate_form,
            "skill_predicate_forms": {
                "first_skill_predicate_form": first_skill_predicate_from,
                "second_skill_predicate_form": second_skill_predicate_form
            },
            "tltl": {
                'state_predicate_form': state_predicate_form,
                'varnames': ['r', 'g', 'b', 'h', 'hx', 'hy', 'hz']
            },
            "predicate_robustness": {
                "r": r['predicate_eval'],
                "g": g['predicate_eval'],
                "b": b['predicate_eval'],
                "h": h['predicate_eval'],
                "ha": ha['predicate_eval']
            }
        }

        other = {'robustness_reward': None}
        return spec, other
    
    def particle_env_config(self, topk=2, goals=None, obstacles=None):
        from rl_pipeline.env.multiagent_env.multiagent_env import MultiAgentParticleEnv
        from examples.iros2019.env.hierarchical_scenario import HierarchicalScenario

        #### Make goals ####
        goal_size = 0.12
        goals = [
            {
                'name': "goal_1",
                'size': goal_size,
                'color': np.array([0.95,0.,0.0, 0.5]),
                'shape': 'circle',
                'collide': False,
                'sub_goals':[]
            },
            {
                'name': "goal_2",
                'size': goal_size,
                'color': np.array([0.95, 0., 0., 0.7]),
                'shape': 'circle',
                'collide': False,
                'sub_goals':[]
            },
            {
                'name': "goal_3",
                'size': goal_size,
                'color': np.array([0, 0., 0.95, 0.5]),
                'shape': 'circle',
                'collide': False,
                'sub_goals':[]
            },
            # {
            #     'name': "goal_4",
            #     'size': goal_size,
            #     'color': np.array([0, 0.95, 0., 0.5]),
            #     'shape': 'circle',
            #     'collide': False,
            #     'sub_goals':[]
            # },
    
        ]

        
        #### Make obstacles ####
        channel_width = self.ExperimentConfig_config.get(['other', 'channel_width'])
        def channel_y(x, u_or_l):
            if u_or_l == 'upper':
                return x + channel_width
            elif u_or_l == 'lower':
                res = x - channel_width
                return res
            else:
                raise ValueError()
                
        obstacle_size = 0.3
        obstacles = [
            {
                'name': "obs_1",
                'size': obstacle_size,
                'color': np.array([0.1,0.1,0.1, 0.8]),
                'shape': 'line',
                'vertices': [np.array([-1, channel_y(-1, 'upper')]), np.array([1.,channel_y(1., 'upper')])],
                'center': np.array([0, 0]),
                'collide': False,
            },
            {
                'name': "obs_2",
                'size': obstacle_size,
                'color': np.array([0.1,0.1,0.1, 0.8]),
                'shape': 'line',
                'vertices': [np.array([-1., channel_y(-1, 'lower')]), np.array([1.,channel_y(1., 'lower')])],
                'center': np.array([0, 0]),
                'collide': False,
            }
        ]
            
        

        def get_state(all_info):
            '''
            all_info = {
            'agent': {
            'pos': [x,y],
            'vel': [x,y]
            },
            'scripted_agents':[
            {
            'pos': [x,y],
            'vel': [x,y]
            },
            {
            ...
            }
            ]
            'obstacles': [
            {'name': 'obstacle_0', 'pos': [x,y]},
            {'name': 'obstacle_1', 'pos': [x,y]},
            ],
            'goals': [
            {'name': 'goal_0', 'pos': [x,y]},
            {'name': 'goal_1', 'pos': [x,y]},
            ]
            }
            '''
        
            agent_state = list(all_info['agent']['pos'])# + list(all_info['agent']['vel'])

            # find 'topk' scripted agents closest to the policy agent
            sa_dist_vec = []
            for sa_state in all_info['scripted_agents']:
                sa_dist_vec.append(np.linalg.norm(np.array(all_info['agent']['pos']) - np.array(sa_state['pos'])))
            sa_dist_vec = -np.array(sa_dist_vec)
            topk_closest_sa_idx = sa_dist_vec.argsort()[-topk:][::-1]

            topk_scripted_agents_states = []
            for i in topk_closest_sa_idx:
                topk_scripted_agents_states.append(list(all_info['scripted_agents'][i]['pos']) + (list(all_info['scripted_agents'][i]['vel'])))

            topk_scripted_agents_states = sum(topk_scripted_agents_states, [])
        
                
            goal_state = []
            for goal in all_info['goals']:
                goal_state += list(goal['pos'])

            state = np.array(list(agent_state) + goal_state) #+ topk_scripted_agents_states)
        
            return state


        def get_reward(state=None, action=None, next_state=None):
            agent_pos = next_state[:2]                
            dist_to_sa = []
            for i in range(1, self.ExperimentConfig_config.get(['other', 'topk'])+1):
                sa_pos = next_state[-4*i:-4*i+2]
                dist_to_sa.append(np.linalg.norm(agent_pos - sa_pos) - self.dist_th * 4)

            sa_min_dist = np.min(np.array(dist_to_sa))
            obs_reward = np.minimum(0, sa_min_dist * 10)

            return obs_reward, sa_min_dist

            
        def is_done(state=None, action=None, next_state=None):
            agent_pos = state[:2]
            dist_to_sa = []
            for i in range(1, self.ExperimentConfig_config.get(['other', 'topk'])+1):
                sa_pos = state[-4*i:-4*i+2]
                dist_to_sa.append(np.linalg.norm(agent_pos - sa_pos) - self.dist_th * 2)

            sa_min_dist = np.min(np.array(dist_to_sa))
            
            if sa_min_dist < 0:
                return True
            else:
                return False
            
        state_space = {'type': 'float', 'shape': (2+len(goals)*2+topk*4, ), 'upper_bound': [], 'lower_bound': []}
        action_space = {'type': 'float', 'shape': (2, ), 'upper_bound': 20* np.array([0.2, 0.2]), 'lower_bound': 20 * np.array([-0.2, -0.2])}

        def info_callback(agent, world):
            ''' constructs all_info '''
            all_info = {
                'agent': {},
                'scripted_agents':[],
                'obstacles': [],
                'goals': []
            }
           
            # record policy agent
            if isinstance(agent, list):
                agent = agent[0]
                all_info['agent'] = {
                    'pos': agent.state.p_pos,
                    'vel': agent.state.p_vel
                }
                
            # record scripted agents
            for sa in world.scripted_agents:
                sa_dict = {
                    'pos': sa.state.p_pos,
                    'vel': sa.state.p_vel
                }
                all_info['scripted_agents'].append(sa_dict)
            
            # record landmarks
            for entity in world.landmarks:
                if entity.name[:3] == "obs":
                    all_info['obstacles'].append({'name': entity.name, 'pos': entity.state.p_pos})
                else:
                    all_info['goals'].append({'name': entity.name, 'pos': entity.state.p_pos})

            return all_info


        class RandomMotion(object):
            def __init__(self, x_bound=[-0.2, 0.2], y_bound=[-0.2, 0.2], max_speed=0.4, dt=0.1):
                self.x_bound = x_bound
                self.y_bound = y_bound
                self.max_speed = max_speed
                self.dt = dt
                
                self.s_low = [self.x_bound[0], self.y_bound[0]]
                self.s_high = [self.x_bound[1], self.y_bound[1]]
                
                self.s = np.random.uniform(low=self.s_low, high=self.s_high)
                self.ds = np.zeros(self.s.shape)
                self.dds = np.zeros(self.s.shape)
                self.goal = np.random.uniform(low=self.s_low, high=self.s_high)
                
                self.s0 = copy.copy(self.s)
                
                self.i = 0
                self.a = 5.
                self.b = self.a / 4
                self.tau = 0.2
                    
            def step(self):
                self.i += 1
                if self.i % 20 == 0:
                    self.goal = np.random.uniform(low=self.s_low, high=self.s_high)
                    
                self.dds = self.a * ( self.b * (self.goal - self.s) - self.ds / self.tau )
                self.ds += self.dds * self.tau * self.dt
                self.s += self.ds * self.dt
                
                self.s = np.clip(self.s, self.s_low, self.s_high)
                    
                return np.concatenate([self.s, self.ds])

        scripted_agent1_motion = RandomMotion(x_bound=[-0.2, -0.0], y_bound=[-0.2, 0.5])
        scripted_agent2_motion = RandomMotion(x_bound=[-0.2, 0.4], y_bound=[-0.2, 0.4])
        scripted_agent3_motion = RandomMotion(x_bound=[-0.3, -0.1], y_bound=[-0.3, 0.4])
        scripted_agent4_motion = RandomMotion(x_bound=[-0.3, 0.3], y_bound=[-0.3, 0.3])

            
        scripted_agents = [
            {
                'name': 'sa1',
                'control_mode': 'position',
                'action_callback': scripted_agent1_motion.step,
                'collide': False
            },
            {
                'name': 'sa2',
                'control_mode': 'position',
                'action_callback': scripted_agent2_motion.step,
                'collide': False
            }
        ]


        config = {
            'seed': 10,
            "state_space": state_space,
            "action_space": action_space,
            "get_state": get_state,
            "get_reward": get_reward,
            "is_done": is_done,
            "get_info": None,
            "headless": self.ExperimentConfig_config.get('headless'),
            ####
            'shared_viewer': True,
            'info_callback': info_callback,
            "control_mode": 'position',
            "scenario": {
                'type': HierarchicalScenario,
                'config': {
                    'goals': goals,
                    'obstacles': obstacles,
                    'scripted_agents':scripted_agents,
                    'randomize_reset': True,
                    'reset_config': {
                        'agent_pos': [[-0.3,-0.3], [0.5,0.5], [-0.5,-0.5]],
                        'goal_pos': [[0.5,0.5], [0.3,0.3], [0.1,0.1]],
                        'obstacle_pos': [[0,0], [0,0]],
                    }
                }
            }
        }
        
        other = {'info_callback': info_callback,
                 'goals': goals,
                 'env_type': MultiAgentParticleEnv,
                 'env_config': config
             }
        
    
        return get_state, get_reward, is_done, state_space, action_space, other


    def particle_tl_task(self, task='tl_flat', dist_th=0.01):
        channel_width = self.ExperimentConfig_config.get(['other', 'channel_width'])

        def in_region_robustness(s, region_center, dist_th):
            return dist_th - np.linalg.norm(s - region_center)


        r1 = {
            'predicate': 'r1',
            'goal_from_state': lambda s: s[2:4],
            'predicate_eval': lambda s,a,sp: in_region_robustness(sp[:2], sp[2:4], dist_th)
        }
        r2 = {
            'predicate': 'r2',
            'goal_from_state': lambda s: s[4:6],
            'predicate_eval': lambda s,a,sp: in_region_robustness(sp[:2], sp[4:6], dist_th)
        }
        r3 = {
            'predicate': 'r3',
            'goal_from_state': lambda s: s[6:8],
            'predicate_eval': lambda s,a,sp: in_region_robustness(sp[:2], sp[6:8], dist_th)
        }
        # r4 = {
        #     'predicate': 'r4',
        #     'goal_from_state': lambda s: s[8:10],
        #     'predicate_eval': lambda s,a,sp: in_region_robustness(sp[:2], sp[8:10], dist_th)
        # }

        c1 = {
            'predicate': 'r3',
            'goal_from_state': None,
            'predicate_eval': lambda s,a,sp: -sp[1] + sp[0] + channel_width
        }
        c2 = {
            'predicate': 'r3',
            'goal_from_state': None,
            'predicate_eval': lambda s,a,sp: sp[1] - sp[0] + channel_width
        }
      
        
        self.predicate_robustness = {
            'r1': r1['predicate_eval'],
            'r2': r2['predicate_eval'],
            'r3': r3['predicate_eval'],
            #'r4': r4['predicate_eval'],
            'c1': c1['predicate_eval'],
            'c2': c2['predicate_eval']
        }

        self.goal_from_state = {
            'r1': r1['goal_from_state'],
            'r2': r2['goal_from_state'],
            'r3': r3['goal_from_state'],
            #'r4': r4['goal_from_state'],
        }
        
      
        #### Define TL specifications ####
        # phi_flat = "(F r1 || F r2) && (F r3) && ((! r3 U (r1 || r2))) && (c1 && c2 U r3)"
        #phi_flat = "(F r1 || F r2) && (F r3) && (c1 && c2 U r3)"
        phi_flat = " F (r2 && F r3)"
        
        #### create specs for different tasks ####
        other = {'robustness_reward': None}
        if task == 'tl_flat':
            spec = {
                "predicate_form": phi_flat,
                "predicate_robustness": self.predicate_robustness,
                "goal_from_state": self.goal_from_state
            }
        else:
            raise ValueError('')
    
        other = {'robustness_reward': None}

        return spec, other
       
if __name__ == "__main__":
    pass