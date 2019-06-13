# import numpy as np
from lomap.classes import Fsa
from future.utils import viewitems
import networkx
import sys
from sympy import *
import sympy
import numpy as np

if sys.version_info > (3, 0):
    PYTHON_VERSION = 3
else:
    PYTHON_VERSION = 2

default_config = {
    'debug': False,
    'softmax': False,
    'beta': 5.,
    'repeat': False
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
        self.debug = self.FsaReward_config['debug']
        
        '''
        fsa_props = {<prop_name>:<prop_value>, ...} where prop_value depends on the binary representation of the proposition
        for example if 'a' = 01 and 'b' = 10 in binary (here 'a' is true when the rightmost digit is 1, false otherwise,)
        and likewise for 'b'. Then fsa_props = {'a':1, 'b':2} and sorted_props returns ['a', 'b']. Depending on how
        fsa_props is generated, sometimes fsa_props = {'b':1, 'a': 2}, then sorted_props = ['b', 'a'], this is random
        '''
        
        print("fsa propositions: ", self.fsa_props)
        if self.logger is not None:
            self.logger.log_text(str(self.fsa_props))
            
        self.sorted_props = sorted(self.fsa_props, key=lambda key: self.fsa_props[key])
        print("sorted fsa propositions: %s" %self.sorted_props)

        ## for sympy
        sorted_props_str = " ".join(self.sorted_props)
        self.sorted_props_sympy = list(sympy.symbols(sorted_props_str))

        if self.logger is not None:
            self.logger.log_text("sorted props: {}".format(self.sorted_props))
       
        
        self.predicate_reward_dict = predicate_reward_dict

        self.aut_states_dict = {}
        self.generate_node_dict()

        print("number of node: {}".format(str(len(self.g.nodes()))))
        print("number of edges: {}".format(str(len(self.g.edges()))))
        
        #### for buchi ####
        nb_acceptance_states = 0
        for k in self.aut_states_dict.keys():
            if 'accept' in k:
                nb_acceptance_states += 1

        if nb_acceptance_states > 1:
            self.aut_type = 'Buchi'
        else:
            self.aut_type = 'FSA'

        print("Automata type:{}".format(self.aut_type))
            
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


    def check_done(self, Q):
        done = False
        if Q == "accept_all" or Q == ("accept_all", "accept_all") or 'accept' in Q:
            # buchi is never done
            if self.aut_type == 'FSA':
                print('fsa done!')
                done = True
        if Q == "trap" or "trap" in Q:
            print('entered trap state')
            done = True

        return done
        
    def step(self,
             Q,
             s=None,
             a=None,
             sp=None,
             repeat=False,
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

        done = self.check_done(Q)
        if not done:
            next_Q, DQ_nontrap, DQ_trap, best_node_guard, trap_node_guard = self.get_node_guard_bin_and_node_rob(Q, s, a, sp, debug=False)
            done = self.check_done(next_Q)
            
            while next_Q != Q and not done:
                Q = next_Q
                next_Q, DQ_nontrap, DQ_trap, best_node_guard, trap_node_guard = self.get_node_guard_bin_and_node_rob(Q, s, a, sp, debug=False)
                
                done = self.check_done(next_Q)
                if done:
                    break
        else:
            next_Q = Q
            DQ_nontrap = 0
            DQ_trap = 0
            best_node_guard = None
            trap_node_guard = None
            
        if repeat and done and self.aut_type == 'FSA':
            next_Q = self.init_state_name
   
        reward = 0

        return next_Q, reward, (Q, next_Q), done, DQ_nontrap, DQ_trap, best_node_guard, trap_node_guard

        
    def get_node_guard_bin_and_node_rob(self, Q, s, a, sp, debug=False):
        out_edges = self.g.out_edges(Q, data=True)
        next_Q = Q

        non_accept_node_rob_list = []
        non_accept_node_guard_list = []
        non_accept_node_list = []

        accept_node_rob_list = []
        accept_node_guard_list = []
        accept_node_list = []

        trap_node_rob_list = []
        trap_node_guard_list = []
        trap_node_list = []

        
        if self.debug or debug:
            print("NODE:", Q)
            print("------")
            
        for edge in out_edges:
            edge_total_rob, edge_guard = self.get_edge_guard_bin_and_edge_rob_symbolic(edge, s, a, sp)
            if edge[1] != 'trap':
                process = False
                if 'accept' not in Q:
                    if edge[1] != Q and edge_total_rob > -100:
                        process = True
                else:
                    if edge_total_rob > -100:
                        process = True
                        
                if process:
                    #### edge robustness ####
                    if 'accept' in edge[1]:
                        accept_node_rob_list.append(edge_total_rob)
                        accept_node_guard_list.append(edge_guard)
                        accept_node_list.append(edge[1])
                    else:
                        non_accept_node_rob_list.append(edge_total_rob)
                        non_accept_node_guard_list.append(edge_guard)
                        non_accept_node_list.append(edge[1])
            else:
                trap_node_rob_list.append(edge_total_rob)
                trap_node_guard_list.append(edge_guard)
                trap_node_list.append(edge[1])
             
            
            if self.debug or debug:
                print("next_node:", edge[1])
                print('edge guard:', edge_guard)
                print("edge total rob:",edge_total_rob)
                print('----------')

            # sometimes self edge and outgoing edge can activate at the same time, why?
            # sometimes directly goes into trap    
            #if edge_total_rob > 0 and edge[1] != Q and edge[1] != 'trap':
            if edge_total_rob > 0 and edge[1] != Q and edge[1] != 'trap':     
                next_Q = edge[1]
    
        DQ_nontrap = 0
        DQ_trap = 0        
        #### For non-trap outgoing edges ####
        if len(accept_node_rob_list) > 0:
            node_rob_list = accept_node_rob_list
            node_guard_list = accept_node_guard_list
            node_list = accept_node_list
        else:
            node_rob_list = non_accept_node_rob_list
            node_guard_list = non_accept_node_guard_list
            node_list = non_accept_node_list

        if len(node_rob_list) > 0:
            best_node_guard = node_guard_list[np.argmax(np.array(node_rob_list))]
            DQ_nontrap = np.max(np.array(node_rob_list))
            if self.debug or debug:
                print("*********")
                print("Final Lists")
                print("node_ list:", node_list)
                print("node_rob_list:", node_rob_list)
                
                print("********")
                print("chosen next Q:{}".format(next_Q))
                print("chosen edge: ", (Q, node_list[np.argmax(np.array(node_rob_list))]))
                print("chosen guard: {}".format(str(node_guard_list[np.argmax(np.array(node_rob_list))])))
                print("sorted props:", self.sorted_props)
                print("========================")
         
        else:
            # this happens at the acceptance node of an FSA
            best_node_guard = None
            DQ_nontrap = 0
            
        #### For trap outgoing edge ####
        if len(trap_node_rob_list) > 0: # there is a connection to trap state
            trap_node_guard = trap_node_guard_list[0]
            DQ_trap = trap_node_rob_list[0]
            if self.debug or debug:
                print("trap_node_guard_list:", trap_node_guard_list)
        else:
            trap_node_guard = None
            DQ_trap = 0
            
        return next_Q, DQ_nontrap, DQ_trap, best_node_guard, trap_node_guard
                

    def get_guard_robustness(self, guard, prop_robustness_dict, sym_pkg='lomap'):
        '''
        if sym_pkg='sympy' - guard is a string of form "a & b & ~c" or "(a & b & ~c)" with only conjunction
        if sym_pkg='lomap' - guard is a string of form "((!(c)) && (!(b)) && (a))" or "(!(c))" or "(a)"
        '''

        if sym_pkg == 'sympy':
            stripped_guard = guard.strip()
            if stripped_guard[0] == "(":
                stripped_guard = stripped_guard[1:-1]
            guard_pred_list = stripped_guard.split("&")
        elif sym_pkg == 'lomap':
            stripped_guard = ""
            for c in guard:
                if c != '(' and c != ")" and c != " ":
                    stripped_guard += c
            guard_pred_list = stripped_guard.split("&&")
        else:
            raise ValueError('sym_pkg not supported')
                
            
        guard_pred_rob_list = []
        for pred in guard_pred_list:
            if pred[0] == "~" or pred[0] == '!':
                guard_pred_rob_list.append(-prop_robustness_dict[pred[1:]])
            else:
                guard_pred_rob_list.append(prop_robustness_dict[pred])

        return stripped_guard, min(guard_pred_rob_list)
        
    def get_edge_guard_bin_and_edge_rob_symbolic(self, edge, s, a=None, sp=None):

        if edge[1] == 'trap':
            # print('trap state currently not supported')
            return -1000, 'trap_guard'
            
        #### get robustness of all predicates at current state ####
        prop_robustness = {}
        for prop in self.sorted_props:
            rob, action = self.predicate_reward_dict[prop](s, a=None, sp=None)
            prop_robustness[prop] = rob            
        ####

        #### sympy way currently doesn't scale with large formula ####
        # input_list = list(edge[2]['input'])
        # simplified_pos_dnf_form = str(SOPform(self.sorted_props_sympy[::-1], input_list))
        # simp_pos_dnf_split = simplified_pos_dnf_form.split('|')

        #### directly using the guard that lomap generates ####
        simplified_pos_dnf_form = str(edge[2]['guard'])
        if "||" in simplified_pos_dnf_form:
            simplified_pos_dnf_form = simplified_pos_dnf_form[1:-1]
        simp_pos_dnf_split = simplified_pos_dnf_form.split('||')
        
        edge_dnf_rob_list = []
        processed_edge_guard_list = []
        for edge_guard  in simp_pos_dnf_split:
            edge_guard, edge_dnf_rob = self.get_guard_robustness(edge_guard, prop_robustness, sym_pkg='lomap')
            edge_dnf_rob_list.append(edge_dnf_rob)
            processed_edge_guard_list.append(edge_guard)
            
        edge_total_rob = max(edge_dnf_rob_list)
        edge_guard = processed_edge_guard_list[np.argmax(np.array(edge_dnf_rob_list))]
            
        
        return edge_total_rob, edge_guard
        
        
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

        note: FSA always starts with 'T0_init', buchi can be something else e.g. 'accept_init'

        '''

        for node_name in self.g.nodes():
            if 'accept' in node_name:
                accept_state_name = node_name
            if 'init' in node_name:
                self.init_state_name = node_name
                
        if PYTHON_VERSION == 2:
            if isinstance(self.g.nodes(data=True)[0][0], str) or isinstance(self.g.nodes(data=True)[0][0], unicode):
                self.aut_states_dict[self.init_state_name] = 0
        else:
            if isinstance(self.g.nodes(data=True)[0][0], str): 
                self.aut_states_dict[self.init_state_name] = 0

        if isinstance(self.g.nodes(data=True)[0][0], tuple):
            self.aut_states_dict[('T0_self.init', 'T0_self.init')] = 0

        
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
                    if node[0] != self.init_state_name and node[0] != accept_state_name:
                        self.aut_states_dict[str(node[0])] = int(i)
                        i += 1
            else:
                if isinstance(node[0], str):
                    if node[0] != self.init_state_name and node[0] != accept_state_name:
                        self.aut_states_dict[str(node[0])] = int(i)
                        i += 1
                
            if isinstance(node[0], tuple): # in this case, the keys of aut_states_dict are tuples
                if PYTHON_VERSION == 2:
                    if node[0] != (unicode(self.init_state_name), unicode(self.init_state_name)) and node[0] != (unicode(accept_state_name), unicode(accept_state_name)):
                        self.aut_states_dict[(str(node[0][0]),str(node[0][1]))] = int(i)
                        i += 1
                else:
                    if node[0] != (self.init_state_name, self.init_state_name) and node[0] != (accept_state_name, accept_state_name):
                        self.aut_states_dict[(str(node[0][0]),str(node[0][1]))] = int(i)
                        i += 1
                    
        print("aut_state_dict: ", self.aut_states_dict)
        if self.logger is not None:
            self.logger.log_text("aut_states_dict: {}".format(self.aut_states_dict))
       


