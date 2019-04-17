import os
import numpy as np
import time
from future.utils import viewitems
import copy

from cooking_env.env.dmp.canonical_system import CanonicalSystem

default_config = {
    # gain on attractor term y dynamics
    'ay': None,
    # gain on attractor term y dynamics
    'by': None,
    # timestep
    'dt': 0.01,
    # time scaling, increase tau to make the system execute faster
    'tau': 1.0,
    'use_canonical': False,
    'n_dmp': 3
    # for cooking environment

}

class DMP(object):

    def __init__(self, config={}, seed=None):
        self.DMP_config = default_config
        self.DMP_config.update(config)

        self.build_graph()
        
    def get_next_wp(self, action, curr_pose, curr_vel):

        point_attractor = self.ay * ( self.by * (self.goal - curr_pose) - curr_vel )

        # optional system feedback
        error = 1.0
        error_coupling = 1. / (1. + error)
    
        if self.DMP_config['use_canonical']:
            cx = self.cs.step(tau=self.tau, error_coupling=error_coupling)
            # generate forcing term
            front_terms = cx * (self.goal - self.y)
            point_attractor += front_terms

        ddy = (point_attractor + action) * self.tau
        dy = curr_vel + ddy * self.dt * error_coupling
        y = curr_pose + dy * self.dt * error_coupling

        return ddy, dy, y
        
    def set_seed(self, seed):
        np.random.seed(seed)

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self.n_dmps):
            if (self.y0[d] == self.goal[d]):
                print("initial position too close to goal, offsetting")
                self.goal[d] += 1e-4

    def reset(self, y=None, dy=None, ddy=None):
        if y is None:
            self.y = copy.copy(self.y0).astype(np.float64)
        else:
            self.y = copy.copy(y).astype(np.float64)
            
        if dy is None:
            self.dy = np.zeros(self.n_dmps).astype(np.float64)
        else:
            self.dy = copy.copy(dy).astype(np.float64)

        if ddy is None:
            self.ddy = np.zeros(self.n_dmps).astype(np.float64)
        else:
            self.ddy = copy.copy(ddy).astype(np.float64)

        self.ddy = np.zeros(self.n_dmps).astype(np.float64)
        self.cs.reset_state()

        if self.goal is not None:
            self.check_offset()
            
    def reset_cs(self):
        self.cs.reset_state()


    def set_initial_pos(self, y0):
        self.y0 = y0

    def set_goal(self, goal):
        self.goal = goal

        
    def build_graph(self):
        self.goal = None
        self.y0 = None

        self.n_dmps = self.DMP_config['n_dmp']
        self.dt = self.DMP_config['dt']
        self.tau = self.DMP_config['tau']
        
        self.by = self.DMP_config['by']

        ay = self.DMP_config['ay']
        if ay is None:
            self.ay = np.ones(self.n_dmps) * 25
        elif isinstance(ay, (int, float)):
            self.ay = np.ones(self.n_dmps) * ay
        else:
            self.ay = ay

        
        by = self.DMP_config['by']
        if by is None:
            self.by = self.ay / 4
        elif isinstance(by, (int, flot)):
            self.by = np.ones(self.n_dmps) * by
        else:
            self.by = by

        self.cs = CanonicalSystem(dt=self.dt)
        self.timesteps = int(self.cs.run_time / self.dt)


        
if __name__ == "__main__":
    pass