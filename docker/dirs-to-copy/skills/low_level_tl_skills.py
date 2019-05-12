import os
import numpy as np
import time
from future.utils import viewitems

from motor_skills import MotorSkills
from tl_utils.generate_automata import GenerateAutomata

default_config = {
    'skill_formula': {},
    'motor_skills_config':{}
}

class LowLevelTLSkills(MotorSkills):

    def __init__(self, config={}):
        self.LowLevelTLSkills_config = default_config
        self.LowLevelTLSkills_config.update(config)

        super(LowLevelTLSkills, self).__init__(self.LowLevelTLSkills_config['motor_skills_config'])

        self.auts = {}
        for k, v in viewitems(self.LowLevelTLSkills_config['skill_formula']):
            self.auts[k] = GenerateAutomata(config=v)


    def step(self, s):
        pass

    def test(self):
        pass

if __name__ == "__main__":
    from tl_utils.tl_config import KEY_POSITIONS, OBJECT_RELATIVE_POSE, STATE_IDX_MAP, PREDICATES
    config = default_config

    #################
    # Set up skills #
    #################
    config['skill_formula'] = {
        'pick_hotdog': {
            'formula':"F((moveto_hotdogplate && opengripper) && X F (closegripper))",
            'key_positions': KEY_POSITIONS,
            'object_relative_pose': OBJECT_RELATIVE_POSE,
            'state_idx_map': STATE_IDX_MAP,
            'predicate_robustness': PREDICATES,
            'fsa_save_dir': os.path.join(os.environ['LEARNING_PATH'], 'skills', 'figures'),
            'dot_file_name': 'pick_hotdog',
            'svg_file_name': 'pick_hotdog',
            'mdp_state_space': {'type': 'float', 'shape': (22, ), 'upper_bound':[], 'lower_bound': []}
        }
        
    }

    ####################### 
    # Set up motor skills #
    #######################
    from execution.execution_config import ExecutionConfig
    exe_config = ExecutionConfig()
    
    config['motor_skills_config'] = exe_config.motor_skill_config()
    
    ##############
    # Initialize #
    ##############
    cls = LowLevelTLSkills(config=config)

    #### test ####
    cls.test()