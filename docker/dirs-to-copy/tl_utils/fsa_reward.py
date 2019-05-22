import numpy as np
from lomap.classes import Fsa
from future.utils import viewitems
import networkx
import sys

if sys.version_info > (3, 0):
    PYTHON_VERSION = 3
else:
    PYTHON_VERSION = 2

default_config = {
    'softmax': False,
    'beta': 5.
}
    
class FsaReward(object):

    def __init__(self, fsa, predicate_reward_dict=None, softmax=False, logger=None, config={}):
        '''
        keys for predicate_reward_dict must much proposition names
        '''
        
        self.fsa = fsa
        self.fsa_name = fsa.name
        self.fsa_props = fsa.props
        self.alphabet = fsa.alphabet
        self.fsa_init_node = fsa.init
        self.fsa_final_node = fsa.final
        self.g = fsa.g

        self.FsaReward_config = default_config
        self.FsaReward_config.update(config)
        
        self.logger = logger
        
        # fsa_props = {<prop_name>:<prop_value>, ...} where prop_value depends on the binary representation of the proposition
        # for example if 'a' = 01 and 'b' = 10 in binary (here 'a' is true when the rightmost digit is 1, false otherwise,)
        # and likewise for 'b'. Then fsa_props = {'a':1, 'b':2} and sorted_props returns ['a', 'b']. Depending on how
        # fsa_props is generated, sometimes fsa_props = {'b':1, 'a': 2}, then sorted_props = ['b', 'a'], this is random

        print("fsa propositions: ", self.fsa_props)
        if self.logger is not None:
            self.logger.log_text(str(self.fsa_props))
            
        self.sorted_props = sorted(self.fsa_props, key=lambda key: self.fsa_props[key])
        print("sorted fsa propositions: %s" %self.sorted_props)
        if self.logger is not None:
            self.logger.log_text("sorted props: {}".format(self.sorted_props))
       
        
        self.predicate_reward_dict = predicate_reward_dict

        self.aut_states_dict = {}
        self.generate_node_dict()

    def logsumexp(self, seq, max_or_min='max'):
        beta = self.FsaReward_config['beta']
        if max_or_min == 'max':
            beta = np.abs(beta)
        elif max_or_min == 'min':
            beta = -np.abs(beta)
        else:
            raise ValueError('invalid beta')

        return (1 / beta) * np.log(np.sum(np.exp(beta * seq)))

    def get_out_edge_formula_list(self, Q):
        out_edge_formula_list = []
        out_edge = self.g.out_edges(Q, data=True)
        #print("out_edge: ", out_edge)
        out_edge_disjunction_formula = None
        for edge in out_edge:
            if edge[1] != edge[0] and edge[1] != 'trap':
                out_edge_formula_list.append(edge[2]['guard'])
        return out_edge_formula_list
        
    def get_out_edge_disjunction_formula(self, Q):
        out_edge = self.g.out_edges(Q, data=True)
        #print("out_edge: ", out_edge)
        out_edge_disjunction_formula = None
        for edge in out_edge:
            if edge[1] != edge[0] and edge[1] != 'trap':
                if out_edge_disjunction_formula is None:
                    out_edge_disjunction_formula = edge[2]['guard']
                else:
                    out_edge_disjunction_formula += " || " + edge[2]['guard']

        return out_edge_disjunction_formula
                
    def step(self,
             Q,
             s=None,
             a=None,
             sp=None,
             phi_b_truth=None,
             **kwargs):
        '''
        get reward at current step
        q: current automata state (as string, eg. "T0_init")
        s: current system state
        a: current action
        sp: next system state
        phi_b_truth: a dictionary of the truth value of the base formulas (used in hierarchical mode), takes the form {'phi1': -1, 'phi2': 1, ...}
        
        return: next_automata_state as string e.g. "T0_init",
                binary reward: if True, then gives 1 for each transition out of current q state and 0 otherwise. If False, then provides the robustness reward
                current_automata_edge as tuple (src_node, destination_node),
                done: True if next_automata_state="accept_all" or "trap",
                Dq: robustness of disjunction of all predicates guarding outgoing edges
        '''

        next_Q = Q
        reward_list = []
        trap_reward_list = []
        # if isinstance(self.g, networkx.classes.digraph.DiGraph):
        out_edges = self.g.out_edges(Q, data=True)
        # else:
        #out_edges = self.g.out_edges(q, keys=True, data=True)

        next_Q = Q
        edge_guard_bin_list = []
        edge_rob_list = []
        for edge in out_edges:
            if edge[1] != Q:
                # input_list gives a set of predicates in the form {2, 4, 5} which are to be translated to binary, the resulting predicate is the disjunction of the list
                input_list = edge[2]["input"]
                edge_bin_list = []
                for input_pred in input_list:
                    b = self.to_binary(input_pred) # e.g. 10011, 00111
                    bin_string = str(b)[::-1]
                    bin_int = [int(i) for i in bin_string]
                    edge_bin_list.append(bin_int)

                edge_bin_list = np.array(edge_bin_list)

                #### calculate disjunction of edge dnfs
                processed_edge_bin = []
                for j in range(edge_bin_list.shape[1]):
                    if np.all(edge_bin_list[:,j] == 0):
                        processed_edge_bin.append(0)
                    elif np.all(edge_bin_list[:,j] == 1):
                        processed_edge_bin.append(1)
                    else:
                        processed_edge_bin.append(-1)

                # in the form [0,1,1,-1], -1 means doesn't matter
                processed_edge_bin = np.array(processed_edge_bin)
                        
                edge_guard_bin_list.append(processed_edge_bin)
                
                rob_list = []
                for i in range(len(processed_edge_bin)):
                    #### calculate robustness ####
                    edge_dnf_rob = self.predicate_reward_dict[self.sorted_props[i]](s, a=None, sp=None)
                    if processed_edge_bin[i] == 0:
                        if edge_dnf_rob != 0: # TODO: Fix this 0 situation, here assume 0 means violation
                            edge_dnf_rob = -edge_dnf_rob
                        else:
                            edge_dnf_rob = -1.0
                    elif processed_edge_bin[i] == 1:
                        edge_dnf_rob = edge_dnf_rob
                    elif processed_edge_bin[i] == -1:
                        edge_dnf_rob = 100000
                    else:
                        raise ValueError('invalide processed_edge_bin')
                        
                    rob_list.append(edge_dnf_rob)

                #### edge robustness ####
                # print(self.sorted_props)
                # print(processed_edge_bin)
                # print(rob_list)
                edge_rob = min(rob_list)
                edge_rob_list.append(edge_rob)
                
                if edge[1] != 'trap':
                    reward_list.append(edge_rob)
                else:
                    trap_reward_list.append(-edge_rob)
                if edge_rob > 0:
                    next_Q = edge[1]
                    
                
        reward = 0
        # this assumes at starting position, goal is not reached and spec is not violated
        Dq = -1
        Dq_trap = 1
        if np.array(reward_list).size != 0:
            if self.FsaReward_config['softmax']:
                Dq = self.logsumexp(np.array(reward_list), max_or_min='max')
            else:
                Dq = np.max(np.array(reward_list))
            reward = Dq
            if Dq > 0:
                reward += 1.0
        if len(trap_reward_list) != 0:
            Dq_trap = np.min(np.array(trap_reward_list))
            reward = np.minimum(Dq, Dq_trap)
            
        done = False
        if next_Q == "accept_all" or next_Q == ("accept_all", "accept_all"):
            # print('fsa done!')
            reward += 2.0
            done = True
        if next_Q == "trap" or "trap" in next_Q:
            print('entered trap state')
            reward += -2.0
            done = True

        if len(edge_rob_list) > 0:
            best_edge_guard_bin = edge_guard_bin_list[np.argmax(np.array(edge_rob_list))]
        else:
            # this happens at the acceptance node of an FSA
            best_edge_guard_bin = None
            
        return next_Q, reward, (Q, next_Q), done, Dq, best_edge_guard_bin

        
    def to_binary(self, num):
        '''
        returns the binary representation of the decimal input
        '''
        return ("{0:0" + str(len(self.fsa_props)) + "b}").format(num)

    def get_node_name_from_value(self, node_value):
        for key, value in self.aut_states_dict.items():
            if value == node_value:
                return key

    def get_node_value_from_name(self, node_name):
        for key, value in self.aut_states_dict.items():
            if node_name == key:
                return int(value)

    def generate_node_dict(self):
        '''
        returns a dictionary in the form {'T0_init':0, 'accept_all': 1, 'T0_S1':2, ...} or
        {('T0_init', 'T0_init'): 0, ('accept_all', 'accept_all'): 1,  ('T0_init', 'T0_S1'): 2}

        this currently only support product of 2 FSAs
        '''

        for node_name in self.g.nodes():
            if 'accept' in node_name:
                accept_state_name = node_name
        
        if PYTHON_VERSION == 2:
            if isinstance(self.g.nodes(data=True)[0][0], str) or isinstance(self.g.nodes(data=True)[0][0], unicode):
                self.aut_states_dict["T0_init"] = 0
        else:
            if isinstance(self.g.nodes(data=True)[0][0], str): 
                self.aut_states_dict["T0_init"] = 0

        if isinstance(self.g.nodes(data=True)[0][0], tuple):
            self.aut_states_dict[('T0_init', 'T0_init')] = 0

        
        if PYTHON_VERSION == 2:
            if isinstance(self.g.nodes(data=True)[0][0], str) or isinstance(self.g.nodes(data=True)[0][0], unicode):
                self.aut_states_dict[accept_state_name] = 1
        else:
            if isinstance(self.g.nodes(data=True)[0][0], str):
                self.aut_states_dict[accept_state_name] = 1
      
        if isinstance(self.g.nodes(data=True)[0][0], tuple):
            self.aut_states_dict[(accept_state_name, accept_state_name)] = 1
      
        i = 2
        for node in self.g.nodes(data=True):
            if PYTHON_VERSION == 2:
                if isinstance(node[0], str) or isinstance(self.g.nodes(data=True)[0][0], unicode):
                    if node[0] != "T0_init" and node[0] != accept_state_name:
                        self.aut_states_dict[str(node[0])] = int(i)
                        i += 1
            else:
                if isinstance(node[0], str):
                    if node[0] != "T0_init" and node[0] != accept_state_name:
                        self.aut_states_dict[str(node[0])] = int(i)
                        i += 1
                
            if isinstance(node[0], tuple): # in this case, the keys of aut_states_dict are tuples
                if PYTHON_VERSION == 2:
                    if node[0] != (unicode('T0_init'), unicode('T0_init')) and node[0] != (unicode(accept_state_name), unicode(accept_state_name)):
                        self.aut_states_dict[(str(node[0][0]),str(node[0][1]))] = int(i)
                        i += 1
                else:
                    if node[0] != ('T0_init', 'T0_init') and node[0] != (accept_state_name, accept_state_name):
                        self.aut_states_dict[(str(node[0][0]),str(node[0][1]))] = int(i)
                        i += 1
                    
        print("aut_state_dict: ", self.aut_states_dict)
        if self.logger is not None:
            self.logger.log_text("aut_states_dict: {}".format(self.aut_states_dict))
       