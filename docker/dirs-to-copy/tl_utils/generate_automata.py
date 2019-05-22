import os
import sys
import numpy as np
import time

from lomap.classes import Fsa
from tl_utils.fsa_reward import FsaReward
from plot_dynamic_automata import PlotDynamicAutomata

default_config = {
    'formula': None,
    'visdom': False,
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

    def __init__(self, formula=None, config={}):
        self.GenerateAutomata_config = default_config
        self.GenerateAutomata_config.update(config)

        self.formula = self.GenerateAutomata_config['formula']

        if formula is not None:
            self.formula = formula

        self.fsa = Fsa()
        self.fsa.from_formula(self.formula)
        self.fsa.add_trap_state()
        self.fsa.visualize(draw='pydot',
                           save_path=self.GenerateAutomata_config['fsa_save_dir'],
                           dot_file_name=self.GenerateAutomata_config['dot_file_name'],
                           svg_file_name=self.GenerateAutomata_config['svg_file_name'])


        self.FSA = FsaReward(self.fsa, self.GenerateAutomata_config['predicate_robustness'])

        if self.GenerateAutomata_config['visdom']:
            self.plot_aut = PlotDynamicAutomata(self.fsa)

        self.q = 0
        self.Q = 'T0_init'
        
    def get_state(self):
        return self.q, self.Q
        
    def get_constraints(self):
        pass

    def step(self, s):
        '''
        if the outgoing edge contains 'move_to' or 'open/close_gripper', returns the next goal and gripper position, else return zero

        TODO: if current state has a connection to 'trap', get constraints
        '''
        assert isinstance(s, np.ndarray)
        assert s.shape == (self.GenerateAutomata_config['mdp_state_space']['shape'][0],)

        t1 = time.time()
        self.Q, r, edge, done, Dq, best_edge_guard_bin = self.FSA.step(self.Q, s=s)

        self.q = self.FSA.get_node_value_from_name(self.Q)

        if self.GenerateAutomata_config['visdom']:
            self.plot_aut.update(current_state = self.Q, src_and_dest=edge)

      
        #### assign actions ####
        ee_goal = None
        gripper_action = None
        other_action = None
        # print(self.FSA.sorted_props)
        # print(self.Q)
        # print(best_edge_guard_bin)
        # print('-----')
        for i in range(len(best_edge_guard_bin)):
            if self.FSA.sorted_props[i] == 'opengripper' and best_edge_guard_bin[i] == 1:
                gripper_action = 'opengripper'
            if self.FSA.sorted_props[i] == 'closegripper' and best_edge_guard_bin[i] == 1:
                gripper_action = 'closegripper'
                     
            if 'moveto' in self.FSA.sorted_props[i] and best_edge_guard_bin[i] == 1:
                ee_goal = self.FSA.sorted_props[i]

            if self.FSA.sorted_props[i] == 'flipswitchon' and best_edge_guard_bin[i] == 1:
                other_action = "flipswitchon"
                             
            if self.FSA.sorted_props[i] == 'applycondiment' and best_edge_guard_bin[i] == 1:
                other_action = 'applycondiment'
                  
        node_action = dict(ee_goal=ee_goal, gripper_action=gripper_action, other_action=other_action)
                
           
  
        #### Get constraints ####
        node_constraints = None

        t2 = time.time()
        # print("dt:{}".format(str(t2-t1)))
        return node_action, node_constraints, done
      
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
    
    cls = GenerateAutomata(formula='F((moveto_hotdogplate && opengripper) && X F (closegripper))', config=config)

    s = np.random.rand(22)
    s[7] == 0

    print(cls.step(s))
    # print(cls.get_state())
    # print(cls.get_goal())