import os
import numpy as np
import time
from future.utils import viewitems

from tl_utils.generate_automata import GenerateAutomata
from tl_utils.tl_config import TLConfig
import time

tl_conf = TLConfig(config={'robot':'jaco'})

construct_skill_state = tl_conf.construct_skill_state

default_config = {
}

class LowLevelTLSkills(object):

    def __init__(self, config={}):
        self.LowLevelTLSkills_config = default_config
        self.LowLevelTLSkills_config.update(config)
        
        self.auts = {}
        self.curr_node_edge = {}
        for k, v in viewitems(self.LowLevelTLSkills_config):
            self.auts[k] = GenerateAutomata(config=v)
            self.curr_node_edge[k] = {'node': self.auts[k].Q, 'edge': self.auts[k].edge}
            
    def step(self, skill_arg):
        s = construct_skill_state(skill_arg)
        skill_action_n_constraint = {}

        for skill_name, skill_aut in viewitems(self.auts):
            node_action, node_constraint, done = skill_aut.step(s)
            skill_action_n_constraint[skill_name] = {'node_action': node_action, 'node_constraint': node_constraint}
            self.curr_node_edge[skill_name] = {'node': skill_aut.Q, 'edge': skill_aut.edge}

        return skill_action_n_constraint, done
            
    def test(self):
        pass

if __name__ == "__main__":
    from execution.execution_config import ExecutionConfig
    exe_config = ExecutionConfig()
    
    config = exe_config.low_level_tl_skill_config()
    
    ##############
    # Initialize #
    ##############
    cls = LowLevelTLSkills(config=config)

    #### test ####
    cls.test()