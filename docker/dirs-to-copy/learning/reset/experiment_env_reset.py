from rl_pipeline.configuration.configuration import Configuration
import numpy as np
import os
from future.utils import viewitems
import json

env_cal = json.loads(open(os.path.join(os.environ['RLFPS_PATH'], 'examples', 'rss2019', 'env_config.json'), 'rb').read())

default_config = {
    'object_poses': {
        'red': {
            'randomize': True,
            'nominal': [0.7, -0.45, -0.06],
            'min': env_cal['red']['sample_range_min'],
            'max': env_cal['red']['sample_range_max']
        },
        'green': {
            'randomize': True,
            'nominal': [0.7, -0.45, -0.046],
            'min': env_cal['green']['sample_range_min'],
            'max': env_cal['green']['sample_range_max']
        },
        'blue': {
            'randomize': True,
            'nominal': [0.7, -0.45, -0.046],
            'min': env_cal['blue']['sample_range_min'],
            'max': env_cal['blue']['sample_range_max']
        },
        'hand': {
            'randomize': True,
            'nominal': [0, 0, 0],
            'min': np.array(env_cal['appear_in_sight']['min']) - 0.1,
            'max': np.array(env_cal['appear_in_sight']['max']) + 0.1
        }
    },
    'joint_angles': {
        'randomize': True,
        'nominal': [ 0.5564515308054339, -1.1094516048381255, 0.006135923151541655, 0.9990049881103757, 0.15033011721277054, 1.5780827355371194, -0.11888351106111957],
        "min": [0.39538355, -0.95567003, -0.69987873,  0.05867477, -0.13805827, 0.39269908, -1.72227693],
        "max": [1.61259730e+00, -5.29223372e-02,  1.16199045e-01, 1.58268468e+00,  1.10484966e+00,  1.71997596e+00,  1.52631088e-01]
    },
    'q':{
        'randomize': True,
        'nominal': 0
    }
}

class ExperimentEnvVrepReset(object):
    def __init__(self, config={}):
        self.ExperimentEnvReset_config = Configuration(default_config)
        self.ExperimentEnvReset_config.update(config)

    def reset(self):

        # get new joint_angles
        if self.ExperimentEnvReset_config.get(['joint_angles', 'randomize']):
            new_joint_angles = np.random.uniform(low=self.ExperimentEnvReset_config.get(['joint_angles', 'min']), high=self.ExperimentEnvReset_config.get(['joint_angles', 'max']))
        else:
            new_joint_angles = self.ExperimentEnvReset_config.get(['joint_angles', 'nominal'])
            
        # get new object positions
        new_object_poses = {}
        for obj_name, obj_reset_dict in viewitems(self.ExperimentEnvReset_config.get('object_poses')):
            if obj_reset_dict['randomize']:
                new_object_poses[obj_name] = np.random.uniform(low=obj_reset_dict['min'], high=obj_reset_dict['max'])
            else:
                new_object_poses[obj_name] = obj_reset_dict['nominal']
                
        new_env_reset_state = {'new_joint_angles': new_joint_angles,
                               'new_object_poses': new_object_poses}
        
        return new_env_reset_state        
                
            