import os
import numpy as np
import time
from future.utils import viewitems
import copy

from cooking_env.env.dmp.canonical_system import CanonicalSystem
import cooking_env.utils.transformations as t

def quaternion_log(q):
    u = q[:3]
    v = q[3]

    if np.linalg.norm(u) == 0:
        return np.array([0,0,0])
    else:
        return np.arccos(v) * (u / np.linalg.norm(u))

def quaternion_exp(q):

    u = q[:3]
    v = q[3]
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        return np.zeros(len(q))
    else:
        return np.concatenate([np.sin(u_norm) * (u / u_norm), np.array([ np.cos(u_norm)])])
        
default_config = {
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
    'n_linear_dmp': 3,
    'n_angular_dmp': 4
}

class DMP(object):

    def __init__(self, config={}, seed=None):
        self.DMP_config = default_config
        self.DMP_config.update(config)

        self.build_graph()

    def get_next_wp(self, action, curr_pose, curr_vel):
        curr_pos = curr_pose[:3]
        curr_quat = curr_pose[3:] 

        curr_linear_vel = curr_vel[:3]
        curr_angular_vel = curr_vel[3:]

        action_linear = action[:3]
        action_angular = action[3:]

        lddy, ldy, ly = self.get_next_wp_pos(action_linear, curr_pos, curr_linear_vel)

        #### addy is anglar acceleration, ady is angular velocity, ay is quaternion
        add, ad, a= self.get_next_wp_quat(action_angular, curr_quat, curr_angular_vel)

        return np.concatenate([lddy, add]),  np.concatenate([ldy, ad]),  np.concatenate([ly, a])

        
    def get_next_wp_quat(self, action, curr_quat, curr_angular_vel):
        '''
        reference: Orientation in Cartesian Space Dynamic Movement Primitives - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6907291
        '''
        
        goal_quat = self.goal[3:]
        curr_angular_vel = curr_angular_vel * np.pi / 180
        eta = self.tau * curr_angular_vel
        
        # eta_dot
        addy = (1 / self.tau) * (self.az * ( self.bz * 2 * quaternion_log( t.quaternion_multiply( goal_quat, t.quaternion_conjugate(curr_quat) ) ) - eta ) )
        # eta
        ady = eta + addy * self.dt
        # target quat
        ay = t.quaternion_multiply( quaternion_exp((self.dt / 2) * (np.concatenate([ady, np.array([0])]) / self.tau) ), curr_quat )
        ay = ay / np.linalg.norm(ay)
                
        return self.tau * addy, self.tau * ady, ay
        
    def get_next_wp_pos(self, action, curr_pos, curr_linear_vel):

        pos_goal = self.goal[:3]
        point_attractor = self.ay * ( self.by * (pos_goal - curr_pos) - curr_linear_vel )

        # optional system feedback
        error = 1.0
        error_coupling = 1. / (1. + error)
    
        if self.DMP_config['use_canonical']:
            cx = self.cs.step(tau=self.tau, error_coupling=error_coupling)
            # generate forcing term
            front_terms = cx * (pos_goal - curr_pos)
            point_attractor += front_terms

        lddy = (point_attractor + action) * self.tau
        ldy = curr_linear_vel + lddy * self.dt * error_coupling
        ly = curr_pos + ldy * self.dt * error_coupling

        return lddy, ldy, ly
        
    def set_seed(self, seed):
        np.random.seed(seed)

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self.n_dmps):
            if (self.y0[d] == self.goal[d]):
                print("initial position too close to goal, offsetting")
                self.goal[d] += 1e-4

    # def reset(self, y=None, dy=None, ddy=None):
    #     if y is None:
    #         self.y = copy.copy(self.y0).astype(np.float64)
    #     else:
    #         self.y = copy.copy(y).astype(np.float64)
            
    #     if dy is None:
    #         self.dy = np.zeros(self.n_dmps).astype(np.float64)
    #     else:
    #         self.dy = copy.copy(dy).astype(np.float64)

    #     if ddy is None:
    #         self.ddy = np.zeros(self.n_dmps).astype(np.float64)
    #     else:
    #         self.ddy = copy.copy(ddy).astype(np.float64)

    #     self.ddy = np.zeros(self.n_dmps).astype(np.float64)
    #     self.cs.reset_state()

    #     if self.goal is not None:
    #         self.check_offset()
            
    def reset_cs(self):
        self.cs.reset_state()


    def set_initial_pos(self, y0):
        self.y0 = y0

    def set_goal(self, goal):
        self.goal = goal

        
    def build_graph(self):
        self.goal = None
        self.y0 = None

        self.n_linear_dmp = self.DMP_config['n_linear_dmp']
        self.n_angular_dmp = self.DMP_config['n_angular_dmp']
        self.dt = self.DMP_config['dt']
        self.tau = self.DMP_config['tau']
        
        self.by = self.DMP_config['by']

        ay = self.DMP_config['ay']
        if ay is None:
            self.ay = np.ones(self.n_linear_dmp) * 25
        elif isinstance(ay, (int, float)):
            self.ay = np.ones(self.n_linear_dmp) * ay
        else:
            self.ay = ay

        
        by = self.DMP_config['by']
        if by is None:
            self.by = self.ay / 4
        elif isinstance(by, (int, float)):
            self.by = np.ones(self.n_linear_dmp) * by
        else:
            self.by = by

        az = self.DMP_config['az']
        if az is None:
            self.az = np.ones(self.n_angular_dmp) * 5
        elif isinstance(az, (int, float)):
            self.az = np.ones(self.n_angular_dmp) * az
        else:
            self.az = az

        
        bz = self.DMP_config['bz']
        if bz is None:
            self.bz = self.az / 4
        elif isinstance(bz, (int, float)):
            self.bz = np.ones(self.n_angular_dmp) * bz
        else:
            self.bz = bz

            
        self.cs = CanonicalSystem(dt=self.dt)
        self.timesteps = int(self.cs.run_time / self.dt)


        
if __name__ == "__main__":
    pass