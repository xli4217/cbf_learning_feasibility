import os
import sys
import numpy as np
import time

from lomap.classes import Fsa
from tl_tools.fsa_reward import FsaReward

default_config = {
    'formula': None,
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

        self.q = 0
        self.Q = 'T0_init'
        
    def get_state(self):
        return self.q, self.Q

    # def get_goal(self):
    #     out_edge_formula_list = self.FSA.get_out_edge_formula_list(self.Q)
    #     print(out_edge_formula_list)
        
    def get_constraints(self):
        pass

    def step(self, s, g):
        '''
        if the outgoing edge contains 'move_to' or 'open/close_gripper', returns the next goal and gripper position, else return zero

        TODO: if current state has a connection to 'trap', get constraints
        '''
        
        assert isinstance(s, np.ndarray)
        assert isinstance(g, np.ndarray)
        assert s.shape == (self.GenerateAutomata_config['mdp_state_space']['shape'][0],)
        assert g.shape == (7, )

        self.Q, r, edge, done, Dq = self.FSA.get_reward_and_next_state(self.Q, s=s, g=g)
        self.q = self.FSA.get_node_value_from_name(self.Q)

        out_edges = self.FSA.g.out_edges(self.Q, data=True)
        for edge in out_edges:
            #### get goal from the closest outgoing edge ####
            edge_pred_rob_list = []
            edge_action_list = []
            if edge[1] != self.Q and edge[1] != 'trap': # for each edge (guarding pred in dnf)
                input_list = edge[2]['input']
                input_pred_rob_list = []
                input_pred_action_list = []
                for input_pred in input_list: # for each conjunction in dnf
                    b = self.FSA.to_binary(input_pred)
                    bin_string = str(b)[::-1]
                    for i in range(len(bin_string)):
                        if self.FSA.sorted_props[i] == 'open_gripper' and int(bin_string[i]) == 1:
                            gripper_pos = 0.2
                        if self.FSA.sorted_props[i] == 'close_gripper' and int(bin_string[i]) == 1:
                            gripper_pos = 0.8

                        if self.FSA.sorted_props[i] == 'move_to' and int(bin_string[i]) == 1:
                            ee_goal = g
                            
                        print(self.FSA.sorted_props[i])
                        print(bin_string[i])
                    print("--")

                    ## this is robustness of each conjunction in the dnf
                    r, r_list = self.FSA.get_r_from_input(input_pred, s, a=None, sp=None, phi_b_truth=None, g=g)
                
                    input_pred_rob_list.append(r)
                    input_pred_action_list.append(dict(ee_goal=ee_goal, gripper_pos=gripper_pos))
                max_input_pred_idx = np.argmax(np.array(input_pred_rob_list))
                best_edge_action = input_pred_action_list[max_input_pred_idx]

        
                edge_pred_rob_list.append(np.max(input_pred_rob_list))
                edge_action_list.append(best_edge_action)

            
            #### get constraints from the connecting trap state ####
            if edge[1] == 'trap':
                trap_input_list = edge[2]['input']
                for input_pred in trap_input_list:
                    b = self.FSA.to_binary(input_pred)
                    bin_string = str(b)[::-1]
                    for i in range(len(bin_string)):
                        pass
                    #     print(self.FSA.sorted_props[i])
                    #     print(bin_string[i])
                    # print("--")

        #### Get best action ####
        if len(edge_pred_rob_list) == 1:
            best_node_action = best_edge_action
        elif len(edge_pred_rob_list) > 0:
            best_node_action = edge_action_list[np.argmax(np.array(edge_pred_rob_list))]
        else:
            best_node_action = None
      
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
    
    cls = GenerateAutomata(formula='F((move_to && open_gripper) && X F (close_gripper))', config=config)

    s = np.random.rand(22)
    s[7] == 0
    g = np.random.rand(7)

    cls.step(s,g)
    # print(cls.get_state())
    # print(cls.get_goal())