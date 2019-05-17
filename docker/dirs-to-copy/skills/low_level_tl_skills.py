import os
import numpy as np
import time
from future.utils import viewitems

from tl_utils.generate_automata import GenerateAutomata

default_config = {
}

class LowLevelTLSkills(object):

    def __init__(self, config={}):
        self.LowLevelTLSkills_config = default_config
        self.LowLevelTLSkills_config.update(config)
        
        self.auts = {}
        for k, v in viewitems(self.LowLevelTLSkills_config):
            self.auts[k] = GenerateAutomata(config=v)


    def step(self, s):
        skill_action_n_constraint = {}
        for skill_name, skill_aut in viewitems(self.auts):
            node_action, node_constraint, done = skill_aut.step(s)
            skill_action_n_constraint[skill_name] = {'node_action': node_action, 'node_constraint': node_constraint}

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