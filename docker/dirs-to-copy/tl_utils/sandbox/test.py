from tl_utils.fsa_reward import FsaReward
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

    # spec = 'G ((inservezone -> X F moveto) && (!inservezone -> X !moveto))'
    
    serve = "F((moveto_bunplate && opengripper) && X F " + \
                "(closegripper && X F " + \
                "((moveto_serveplate && closegripper) && X F " + \
                "(opengripper && X F "+ \
                "(moveto_world_baxterneutral " + \
                ")))))" #&& (!collide U moveto_world_baxterneutral)"


    test = "F(a && X F b) && G(!c)"
    
    serve_task_KB = "G (!(moveto_serveplate && moveto_bunplate)) && " + \
                        "G (!(moveto_serveplate && moveto_world_baxterneutral)) && " + \
                        "G (!(moveto_bunplate && moveto_world_baxterneutral)) && " + \
                        "G (!(moveto_serveplate && moveto_bunplate && moveto_word_baxterneutral)) && " + \
                        "G (!(opengripper && closegripper))"

    # spec = "G (( inservezone_serveplate -> X F (" + serve + ")) && (!inservezone_serveplate -> X F moveto_world_baxterneutral))" + " && " + serve_task_KB
    #spec = "(((r && c) -> X F pp) && ((! r || ! c) -> X (F pp && (!pp U (r && c))))) && G(!col)"
    # spec = "(F pp && (!pp U (r && c)))"
    spec = serve + "&&" + "G(! (opengripper && closegripper))"
    
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
    print(aut.g.nodes())
    aut.visualize(draw='pydot', save_path=os.path.join(os.environ['LEARNING_PATH'], 'tl_utils', 'sandbox', 'figures'), dot_file_name='g', svg_file_name='file')


    def to_binary(aut,num):
        '''
        returns the binary representation of the decimal input
        '''
        return ("{0:0" + str(len(aut.props)) + "b}").format(num)

    # for n in aut.g.nodes():
    #     out_edges = aut.g.out_edges(n, data=True)
    #     for edge in out_edges:
    #         input_list = edge[2]["input"]
    #         edge_bin_list = []
    #         for input_pred in input_list:
    #             b = to_binary(aut, input_pred) # e.g. 10011, 00111
    #             bin_string = str(b)[::-1]
    #             bin_int = [int(i) for i in bin_string]
    #             edge_bin_list.append(bin_int)
    #         if edge[1] == 'trap':
    #             print(edge_bin_list)
    
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
