import numpy as np
import os
from future.utils import viewitems

from tl_utils.fsa_reward import FsaReward
from lomap.classes import Fsa
from utils.configuration import Configuration

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
 
    def update_all_info(self):
        self.base_env.update_all_info()
        self.all_info.update(self.base_env.get_info())
        self.all_info.update({"Q":self.Q, "Q_next": self.Q_next, 'curr_edge': self.curr_edge, 'Dq': self.Dq})
        
    def step_fsa(self, mdp_state, action, next_mdp_state):
        Q = self.fsa_reward.get_node_name_from_value(self.q)

        Q_next, fsa_r, curr_edge, fsa_done, DQ_nontrap, DQ_trap, _, _ = self.fsa_reward.step(Q, mdp_state, action, next_mdp_state)
        
        self.q = self.fsa_reward.get_node_value_from_name(Q_next)
        self.fsa_done = fsa_done
        self.fsa_r = fsa_r
        
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