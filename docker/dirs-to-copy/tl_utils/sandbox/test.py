from tl_tools.fsa_reward import FsaReward
#from tl_tools.visualization.plot_dynamic_graph import PlotDynamicAutomata
from lomap.classes import Fsa
import numpy as np
import os


'''
G = Always
F = Eventually
! = not
U = Until
-> = Imply
X = Next
&& = And
|| = Or
'''
get_fsa = True
###########
# Get FSA #
###########
if get_fsa:
    spec_without_serve = "F((grill_on && X F (PP_raw_dog_grill && X F (wait_2 && X F (PP_cooked_dog_bun && X F ( apply_condiment && X F (grill_off) ))))))"

    spec_with_serve = spec_without_serve + "&&" + 'G (customer -> X serve)'

    spec = spec_without_serve

    # spec = 'G ((a -> X b) && (!a -> X !b))'
    #### add task specific conditions and constraints ####
    conditions = ['(! serve U apply_condiment)']
    # for condition in conditions:
    #     spec += "&& " + condition

    ######
    # KB #
    ######    

    #### general info ####
    general_info = {
        'condiment': 'ketchup || mustard || cheese'
    }
        
    #### add constraints ####
        
    #### edge prune KB ####
    edge_prune_KB = [
        'G (! (apply_condiment && serve))',
        'G ( ! (apply_condiment && wait_2))',
        'G ( ! (serve && wait_2))',
        'G ( ! (PP_raw_dog_grill && serve))',
        'G ( ! (PP_cooked_dog_bun && serve))',
    ]
    # for ep in edge_prune_KB:
    #     spec += "&&" + ep
   
    #### feasibility KB ####
    feasibility_KB = ['G(!serve)']    
    # for fb in feasibility_KB:
    #     spec += "&&" + fb
  
        
    # ------- Get FSA ------------
    aut = Fsa()
    aut.from_formula(spec)
    aut.add_trap_state()
    print("number of nodes: ", len(aut.g.nodes()))
    print("number of edges: ", len(aut.g.edges()))
    # print(aut)
    # print(aut.g.nodes())
    aut.visualize(draw='pydot', save_path=os.path.join(os.environ['LEARNING_PATH'], 'tl_utils', 'sandbox', 'figures'), dot_file_name='g', svg_file_name='file')

    # fsa_reward = FsaReward(fsa=aut)
    # Q = "T0_init"
    # # print(aut.g.out_edges(Q, data=True))
    # phi_b_truth = {'phi1': -0.1, 'phi2': -0.1}
    # next_Q, reward, edge, done, Dq = fsa_reward.get_reward_and_next_state(Q, phi_b_truth=phi_b_truth)
    # print('next_Q:', next_Q)
    # print('reward:', reward)
    # print('edge: ', edge)
    # print('done: ', done)
    # print('Dq: ', Dq)
