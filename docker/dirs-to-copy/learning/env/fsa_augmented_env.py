import numpy as np
import os
from future.utils import viewitems

from tl_utils.fsa_reward import FsaReward
from lomap.classes import Fsa
from utils.configuration import Configuration


#### HACK ####
from tl_utils.tl_config import TLConfig
from utils.utils import get_object_goal_pose
tl_conf = TLConfig(config={'robot':'jaco'})
OBJECT_RELATIVE_POSE = tl_conf.OBJECT_RELATIVE_POSE


default_config = {
    'fsa_save_dir': os.getcwd(),
    'dot_file_name': 'g',
    'svg_file_name': 'svg',
    'base_env': {
        'type': None,
        'config': {}
    },
    'spec': {
        'predicate_form': None,
        'predicate_robustness': None
    }
}

class FsaAugmentedEnv(object):
    def __init__(self, env_params={}, seed=0, base_env=None, suffix="", reset=None, logger=None, port_num=None):
        self.FsaAugmentedEnv_config = Configuration(default_config)
        self.FsaAugmentedEnv_config.update(env_params)

        #### construct base env ####
        self.base_env = base_env
        if self.base_env is None and self.FsaAugmentedEnv_config.get(['base_env', 'type']) is not None:
            # construct base
            self.base_env = self.FsaAugmentedEnv_config.get(['base_env', 'type'])(self.FsaAugmentedEnv_config.get(['base_env', 'config']), port_num=port_num, suffix=suffix, reset=reset, seed=seed, logger=logger)

        if self.base_env is not None:
            if not os.path.isdir(self.FsaAugmentedEnv_config.get('fsa_save_dir')):
                os.makedirs(self.FsaAugmentedEnv_config.get('fsa_save_dir'))
            self.base_env.set_seed(seed)
            self.logger = logger
            # construct fsa reward
            self.spec = self.FsaAugmentedEnv_config.get('spec')
            self.predicate_robustness = self.spec['predicate_robustness']
            self.fsa = Fsa()
            self.fsa.from_formula(self.FsaAugmentedEnv_config.get(['spec', 'predicate_form']))
            self.fsa.add_trap_state()
            self.fsa.visualize(draw='pydot',
                               save_path=self.FsaAugmentedEnv_config.get('fsa_save_dir'),
                               dot_file_name=self.FsaAugmentedEnv_config.get('dot_file_name'),
                               svg_file_name=self.FsaAugmentedEnv_config.get('svg_file_name'))

            self.fsa_reward = FsaReward(self.fsa, self.FsaAugmentedEnv_config.get(['spec', 'predicate_robustness']), logger=self.logger)

            self.qs = [v for k, v in viewitems(self.fsa_reward.aut_states_dict) if v != 1]
            self.q = None # this is the numerical representation of the automata state (we use Q to represent the string version)

            
        # seed the environment
        self.seed(seed)

        self.all_info = {}

        self.FsaAugmentedEnv_reset = reset

        self.state = None

        #### hack ####
        self.load_switchon_policy()
        self.condimentapplied = -10
        self.OBJECT_RELATIVE_POSE = OBJECT_RELATIVE_POSE
        self.get_object_goal_pose = get_object_goal_pose
        
    def get_state(self, **kwargs):
        self.update_all_info()
        return np.concatenate([np.array([self.q]),  self.base_env.get_state()])

    def reset(self, random_q=False, **kwargs):
        self.base_env_state = self.base_env.reset()
        if random_q:
            self.q = int(np.random.choice(self.qs, 1))
        else:
            self.q = 0
        self.fsa_done = False
        self.fsa_r = None
        self.Q_next = None
        self.curr_edge = None
        self.Dq = None
        self.Q = None

        
        self.state = self.get_state()

        
        return self.state
            
    def get_reward(self, state=None, action=None, next_state=None, **kwargs):
        # r = self.base_env.get_reward(state[1:], action, next_state[1:]) + self.fsa_r
        r = np.minimum(self.base_env.get_reward(state[1:], action, next_state[1:]), self.fsa_r)
        return r

    def is_done(self, state=None, action=None, next_state=None, **kwargs):
        base_env_done = False
        if state is not None:
            base_env_done = self.base_env.is_done(state[1:], action)
            if base_env_done:
                print("base env done")

        return any([base_env_done, self.fsa_done])

    def get_info(self):
        return self.all_info
 
    def update_all_info(self, info={}):

        self.base_env.update_all_info(info={'condimentapplied': self.condimentapplied})

        self.all_info.update(self.base_env.get_info())

        self.all_info.update({"Q":self.Q, "Q_next": self.Q_next, 'curr_edge': self.curr_edge, 'Dq': self.Dq})
        self.all_info.update(info)
        
    def set_node_goal(self, best_node_guard):

        ee_goal = None
        gripper_action = None
        other_action = None

        if best_node_guard is not None:
            best_node_guard_pred_list = best_node_guard.strip().split("&")
            for node_guard_pred in best_node_guard_pred_list:
                node_guard_pred = node_guard_pred.strip()
        
                if node_guard_pred == 'opengripper':
                    gripper_action = 'opengripper'
                if node_guard_pred == 'closegripper':
                    gripper_action = 'closegripper'
                     
                if 'moveto' in node_guard_pred and node_guard_pred[0] != "~" and node_guard_pred[0] != "!":
                    ee_goal = node_guard_pred

                if node_guard_pred == 'flipswitchon':
                    other_action = "flipswitchon"
                             
                if node_guard_pred == 'applycondiment':
                    other_action = 'applycondiment'

        if ee_goal is not None:
            object_name = ee_goal.split('_')[1]
            if len(ee_goal.split('_')) == 3:
                object_rel_pose_name = ee_goal.split('_')[2]
            else:
                object_rel_pose_name = object_name

            if object_name == 'world':
                pt = self.OBJECT_RELATIVE_POSE[object_rel_pose_name]
            else:
                pt = self.get_object_goal_pose(self.all_info['obj_poses'][object_name], self.OBJECT_RELATIVE_POSE[object_rel_pose_name])
            self.base_env.set_goal_pose(pt)

        if gripper_action is not None:
            if gripper_action == 'opengripper':
                if self.base_env.get_gripper_state() != 0:
                    self.base_env.set_gripper_state(0)
            if gripper_action == 'closegripper':
                if self.base_env.get_gripper_state() != 1:
                    self.base_env.set_gripper_state(1)

        if other_action is not None:
            if other_action == 'flipswitchon':
                from tl_utils.tl_config import TLConfig
                from utils.utils import get_object_goal_pose
        
                tl_conf = TLConfig(config={'robot':'jaco'})

                OBJECT_RELATIVE_POSE = tl_conf.OBJECT_RELATIVE_POSE

                #### close gripper
                if self.base_env.get_gripper_state() != 1:
                    self.base_env.set_gripper_state(1.)
                
                pt = get_object_goal_pose(self.all_info['obj_poses']['grill'], OBJECT_RELATIVE_POSE['switchon'])
                self.base_env.set_goal_pose(pt)
                
                curr_pos, curr_quat = self.base_env.get_ee_pose()
                curr_linear_vel, curr_angular_vel = self.base_env.get_ee_velocity()
                curr_angular_vel = curr_angular_vel * np.pi / 180

                curr_pose = np.concatenate([curr_pos, curr_quat])
                curr_vel = np.concatenate([curr_linear_vel, curr_angular_vel])

                ddy, dy, y = self.skill_flipswitchon(pt, curr_pose, curr_vel, self.all_info['obs_info'])

                self.base_env.set_target_pose(y)
                
            elif other_action == 'applycondiment':
                for i in range(20):
                    vel_scale = 2. * np.sin(0.3*i)
                    self.base_env.pub_ee_frame_velocity(direction='z',vel_scale=vel_scale, duration_sec=0.1)
                self.condimentapplied = 10
                self.update_all_info()
                    
            else:
                raise ValueError('other_action not supported')
                    
    def step_fsa(self, mdp_state, action, next_mdp_state):
        Q = self.fsa_reward.get_node_name_from_value(self.q)

        Q_next, fsa_r, curr_edge, fsa_done, DQ_nontrap, DQ_trap, best_node_guard, trap_node_guard = self.fsa_reward.step(Q, mdp_state, action, next_mdp_state)
        
        self.q = self.fsa_reward.get_node_value_from_name(Q_next)
        self.fsa_done = fsa_done
        self.fsa_r = fsa_r

        self.set_node_goal(best_node_guard)
        
    def step_base_env(self, actions, **kwargs):
        self.base_env.step(actions)        

    def step(self, actions):
        # TODO: need to modify this for concurrent mode
        self.step_base_env(actions)
        state = self.get_state()
        self.step_fsa(self.state[1:], actions, state[1:])
        self.state = state
        
    def seed(self, seed=0, **kwargs):
        np.random.seed(seed)


    def close(self, **kwargs):
        self.base_env.close()

    @property
    def state_space(self):
        return {"type":"float",
                "shape": (self.base_env.state_space['shape'][0]+1,),
                "upper_bound": [len(self.fsa_reward.aut_states_dict.keys())] + self.base_env.state_space['upper_bound'],
                "lower_bound":[0] + self.base_env.state_space['lower_bound']}

    @property
    def action_space(self):
        return self.base_env.action_space
        
    def save(self, save_dir):
        pass
        
    def restore(self, restore_dir):
        '''
        restores the environment
        '''
        pass

    def teleop(self, cmd):
        pass


    ########
    # Misc #
    ########

    def load_switchon_policy(self):
        from utils.utils import load_policy_and_preprocessor

        ## flip switch open 
        experiment_root_dir = os.path.join(os.environ['LEARNING_PATH'], 'learning', 'experiments')
        experiment_name = 'switchon'
        hyperparam_dir = 'seed0'
        itr = 100
        
        learned_skill_config = {
            "state_space": {'type': 'float', 'shape': (3, ), "upper_bound": [], 'lower_bound': []},
            "action_space": {'type': 'float', 'shape': (3, ), "upper_bound": 70*np.ones(3), 'lower_bound': -70*np.ones(3)},
            "training_config_restore_path": os.path.join(experiment_root_dir, experiment_name, 'config', hyperparam_dir, 'config.pkl'),
            "policy_restore_path": os.path.join(experiment_root_dir, experiment_name, 'transitions', hyperparam_dir, 'itr_'+str(itr)),
            "state_preprocessor_restore_path": os.path.join(experiment_root_dir, experiment_name, 'info', hyperparam_dir, 'state_preprocessor_params.pkl')
        }

        self.switchon_policy, self.switchon_state_preprocessor = load_policy_and_preprocessor(learned_skill_config)

        self.learned_skills_dict = {
            'flipswitchon':{
                'policy': self.switchon_policy,
                'state_preprocessor': self.switchon_state_preprocessor
            }
        }

    def skill_flipswitchon(self, goal, curr_pose, curr_vel, obs_info):
        self.base_env.wp_gen.set_goal(goal)
        
        policy = self.learned_skills_dict['flipswitchon']['policy']
        state_preprocessor = self.learned_skills_dict['flipswitchon']['state_preprocessor']

        state_dim = 3
        
        s = curr_pose[:state_dim]

        if state_preprocessor is not None:
            s = state_preprocessor.get_scaled_x(s)
            
        forcing = policy.get_action(s, deterministic=True).flatten()

        forcing *= 100

        action_space = 3
        forcing = np.clip(forcing, -70, 70)
        forcing = np.concatenate([forcing, np.zeros(3)])

        ddy, dy, y = self.base_env.wp_gen.get_next_wp(action=forcing,
                                                      curr_pose=curr_pose,
                                                      curr_vel=curr_vel,
                                                      obs_info=obs_info)
        return ddy, dy , y
