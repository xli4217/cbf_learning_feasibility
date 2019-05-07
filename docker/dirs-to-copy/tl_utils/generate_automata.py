import os
import sys
import numpy as np
import time

from lomap.classes import Fsa
from tl_tools.fsa_reward import FsaReward

default_config = {
    'key_positions': None,
    'object_relative_pose': None,
    'state_idx_map': None,
    'predicate_robustness': None,
    'fsa_save_dir': None,
    'dot_file_name': 'fsa',
    'svg_file_name': 'fsa',
    'mdp_state_space': {'type': 'float', 'shape': (22, ), 'upper_bound':[], 'lower_bound': []}
}

class GenerateAutomata(object):

    def __init__(self, formula, config={}):
        self.GenerateAutomata_config = default_config
        self.GenerateAutomata_config.update(config)

        self.fsa = Fsa()
        self.fsa.from_formula(formula)
        self.fsa.add_trap_state()
        self.fsa.visualize(draw='pydot',
                           save_path=self.GenerateAutomata_config['fsa_save_dir'],
                           dot_file_name=self.GenerateAutomata_config['dot_file_name'],
                           svg_file_name=self.GenerateAutomata_config['svg_file_name'])


        self.FSA = FsaReward(self.fsa, self.GenerateAutomata_config['predicate_robustness'])

        self.q = 0
        self.Q = 'T0_init'
        
    def get_state(self):
        return self.q, self.Q

    def get_goal(self):
        out_edge_formula_list = self.FSA.get_out_edge_formula_list(self.Q)
        print(out_edge_formula_list)
        
    def get_constraints(self):
        pass

    def step(self, s, g):
        assert isinstance(s, np.ndarray)
        assert isinstance(g, np.ndarray)
        assert s.shape == (self.GenerateAutomata_config['mdp_state_space']['shape'][0],)
        assert g.shape == (7, )

        self.Q, r, edge, done, Dq = self.FSA.get_reward_and_next_state(self.Q, s=s, g=g)
        self.q = self.FSA.get_node_value_from_name(self.Q)
        

if __name__ == "__main__":
    from tl_config import KEY_POSITIONS, OBJECT_RELATIVE_POSE, STATE_IDX_MAP, PREDICATES

    config = {
        'key_positions': KEY_POSITIONS,
        'object_relative_pose': OBJECT_RELATIVE_POSE,
        'state_idx_map': STATE_IDX_MAP,
        'predicate_robustness': PREDICATES,
        'fsa_save_dir': os.path.join(os.environ['LEARNING_PATH'], 'tl_utils', 'sandbox', 'figures'),
        'dot_file_name': 'fsa',
        'svg_file_name': 'fsa',
        'mdp_state_space': {'type': 'float', 'shape': (22, ), 'upper_bound':[], 'lower_bound': []}
    }
    
    cls = GenerateAutomata(formula='F move_to && F close_gripper', config=config)

    s = np.random.rand(22)
    g = np.random.rand(7)

    cls.step(s,g)
    # print(cls.get_state())
    print(cls.get_goal())