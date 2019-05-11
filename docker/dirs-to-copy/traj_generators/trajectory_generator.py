import os
import numpy as np
from future.utils import viewitems

from traj_generators.dmp.dmp import DMP
from traj_generators.clf_cbf.QPcontroller import QPcontroller


default_config = {
    'dmp_config': {
        # gain on attractor term y dynamics (linear)
        'ay': None,
        # gain on attractor term y dynamics (linear)
        'by': None,
        # gain on attractor term y dynamics (angular)
        'az': None,
        # gain on attractor term y dynamics (angular)
        'bz': None,
        # timestep
        'dt': 0.01,
        # time scaling, increase tau to make the system execute faster
        'tau': 1.0,
        'use_canonical': False,
        # for canonical
        'apx': 1.,
        'gamma': 0.3,
        # for faster convergence
        'app': 0.5,
        'apr': 0.5,
        # for integrating goal
        'ag': 1.0,
        'ago': 1.0,
        # if True, then update according to dmp_pose, else update according to current pose
        'use_dmp_pose': True,
        'n_linear_dmp': 3,
        'n_angular_dmp': 4
    },
    'clf_cbf_config': {
        'k_cbf': 1,
        'epsilon':0.8,
        'num_states':3,
        'action_space': {'shape': 3, 'upper_bound': [0.1, 0.1, 0.1], 'lower_bound': [-0.1,-0.1,-0.1]},
        'use_own_pose': False,
        'dt': 0.2
    },
    'translation_gen': 'clf_cbf',
    'orientation_gen': 'dmp'
}

class TrajectoryGenerator(object):

    def __init__(self, config={}):
        self.TrajectoryGenerator_config = default_config
        self.TrajectoryGenerator_config.update(config)

        self.dmp_gen = DMP(self.TrajectoryGenerator_config['dmp_config'])
        self.clf_cbf_gen = QPcontroller(self.TrajectoryGenerator_config['clf_cbf_config'])

    def set_goal(self, goal):
        self.dmp_gen.set_goal(goal)
        self.clf_cbf_gen.set_goal(goal)
        
    def get_next_wp(self, action, curr_pose, curr_vel, obs_info={}):
      
        #### translation generator ####
        if self.TrajectoryGenerator_config['translation_gen'] == 'dmp':
            #### this include translation and orientation
            dmp_ddy, dmp_dy, dmp_y = self.dmp_gen.get_next_wp(action, curr_pose, curr_vel)
            ddy_trans = dmp_ddy[:3]
            dy_trans = dmp_dy[:3]
            y_trans = dmp_y[:3]
        elif self.TrajectoryGenerator_config['translation_gen'] == 'clf_cbf':
            #### this currently only includes translation
            clf_cbf_ddy, clf_cbf_dy, clf_cbf_y = self.clf_cbf_gen.get_next_wp(action, curr_pose, curr_vel, obs_info)

            ddy_trans = clf_cbf_ddy[:3]
            dy_trans = clf_cbf_dy[:3]
            y_trans = clf_cbf_y[:3]
        else:
            raise ValueError('trajectory generator not supported')

        #### orientation generator ####
        if self.TrajectoryGenerator_config['orientation_gen'] == 'dmp':
            #### this include translation and orientation
            dmp_ddy, dmp_dy, dmp_y = self.dmp_gen.get_next_wp(action, curr_pose, curr_vel)
            ddy_ori = dmp_ddy[3:]
            dy_ori = dmp_dy[3:]
            y_ori = dmp_y[3:]
        elif self.TrajectoryGenerator_config['orientation_gen'] == 'clf_cbf':
            #### this currently only includes translation
            clf_cbf_ddy, clf_cbf_dy, clf_cbf_y = self.clf_cbf_gen.get_next_wp(action, curr_pose, curr_vel, obs_info)

            ddy_ori = clf_cbf_ddy[3:]
            dy_ori = clf_cbf_dy[3:]
            y_ori = clf_cbf_y[3:]
        else:
            raise ValueError('trajectory generator not supported')


        return np.concatenate([ddy_trans, ddy_ori]), np.concatenate([dy_trans, dy_ori]), np.concatenate([y_trans, y_ori])