import os
import numpy as np
import time
from future.utils import viewitems
import copy

from traj_generators.dmp.canonical_system import CanonicalSystem
import utils.transformations as t

def quaternion_log(q):
    u = q[:3]
    v = q[3]

    if np.linalg.norm(u) == 0:
        return np.array([0,0,0])
    else:
        if v > 0.999:
            v = 0.999
        return np.arccos(v) * (u / np.linalg.norm(u))

def quaternion_exp(q):
    '''
    this only supports unit quaternion
    '''
    u = q[:3]
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        return np.zeros(4)
    else:
        return np.concatenate([np.sin(u_norm) * (u / u_norm), np.array([ np.cos(u_norm)])])

def quaternion_dist(q1, q2):
    conjugate_product = t.quaternion_multiply(q1, t.quaternion_conjugate(q2))

    if all(conjugate_product == np.array([0,0,0,-1])):
        return 2*np.pi
    else:
        return 2 * np.linalg.norm(quaternion_log(conjugate_product))
        
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
    'use_dmp_pose': False,
    'n_linear_dmp': 3,
    'n_angular_dmp': 4
}

class DMP(object):

    def __init__(self, config={}, seed=None):
        self.DMP_config = default_config
        self.DMP_config.update(config)

        self.build_graph()

        self.dmp_pos = None
        self.dmp_quat = None
        self.dmp_linear_vel = None
        self.dmp_angular_vel = None
        
    def get_next_wp(self, action, curr_pose, curr_vel, dt=None, obs_info={}):
        if dt is not None:
            self.dt = dt

        curr_pos = curr_pose[:3]
        curr_quat = curr_pose[3:] 

        curr_linear_vel = curr_vel[:3]
        curr_angular_vel = curr_vel[3:]
        
        action_linear = action[:3]
        action_angular = action[3:]

        if self.DMP_config['use_dmp_pose']:
            if self.dmp_pos is None:
                self.dmp_pos = curr_pos
                self.dmp_quat = curr_quat
                self.dmp_linear_vel = curr_linear_vel
                self.dmp_angular_vel = curr_angular_vel
      
            lddy, ldy, ly = self.get_next_wp_pos(action_linear, self.dmp_pos, self.dmp_linear_vel, self.linear_front_terms)
            #### addy is anglar acceleration, ady is angular velocity, ay is quaternion
            addy, ady, ay= self.get_next_wp_quat(action_angular, self.dmp_quat, self.dmp_angular_vel, self.angular_front_terms)
            self.dmp_pos = ly
            self.dmp_quat = ay
            self.dmp_linear_vel = ldy
            self.dmp_angular_vel = ady
        else:
            lddy, ldy, ly = self.get_next_wp_pos(action_linear, curr_pos, curr_linear_vel, self.linear_front_terms)
            #### addy is anglar acceleration, ady is angular velocity, ay is quaternion
            addy, ady, ay= self.get_next_wp_quat(action_angular, curr_quat, curr_angular_vel, self.angular_front_terms)

        
        if self.DMP_config['use_canonical']:
            #### implement phase stopping ####
            linear_tracking_error = (curr_pos - ly)
            angular_tracking_error = quaternion_dist(curr_quat, ay)
            error_coupling = 1. / (1. + self.apx * (np.linalg.norm(linear_tracking_error) + self.gamma * angular_tracking_error))
            cx = self.cs.step(tau=self.tau, error_coupling=error_coupling)
            # generate forcing term
            self.linear_front_terms = cx * linear_tracking_error
            self.angular_front_terms = cx * angular_tracking_error

        #### continuous goal integrate ####
        # linear_goal = self.goal[:3] + self.ag * (self.new_goal[:3] - self.goal[:3]) / self.tau
        # goal_quat = copy.copy(self.goal[3:])
        # new_goal_quat = copy.copy(self.new_goal[3:])
        # tmp = self.ago * quaternion_log(t.quaternion_multiply(new_goal_quat, t.quaternion_conjugate(goal_quat)))
        # tmp = quaternion_exp(self.dt * tmp / 2 / self.tau)
        
        # quat_goal = t.quaternion_multiply(tmp, goal_quat)

        # self.goal = np.concatenate([linear_goal, quat_goal])
        
        return np.concatenate([lddy, addy]),  np.concatenate([ldy, ady]),  np.concatenate([ly, ay])

        
    def get_next_wp_quat(self, action, curr_quat, curr_angular_vel, front_terms):
        '''
        reference: Orientation in Cartesian Space Dynamic Movement Primitives - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6907291

        angular velocity is in rad/s
        '''

        goal_quat = self.goal[3:]

        eta = self.tau * curr_angular_vel

        log_conjugate_product = quaternion_log(t.quaternion_multiply(goal_quat, t.quaternion_conjugate(curr_quat)))

        # eta_dot
        addy = (1 / self.tau) * (self.az * ( self.bz * 2 * log_conjugate_product - eta ) + action * front_terms)
        
        # eta
        ady = (eta + addy * self.dt) / self.tau
        # ady +=  self.apr * 2 * quaternion_log(t.quaternion_multiply(goal_quat, t.quaternion_conjugate(curr_quat)))
        
        # target quat
        ay = t.quaternion_multiply( quaternion_exp((self.dt / 2) * ( ady / self.tau) ), curr_quat )
        
                
        return self.tau * addy, self.tau * ady, ay
        
    def get_next_wp_pos(self, action, curr_pos, curr_linear_vel, front_terms):

        pos_goal = self.goal[:3]
        point_attractor = self.ay * ( self.by * (pos_goal - curr_pos) - curr_linear_vel ) + action * front_terms

        lddy = ((point_attractor + action) * self.tau) / self.tau
        ldy = (curr_linear_vel + lddy * self.dt) / self.tau

        # ldy += self.app * (pos_goal - curr_pos)
        ly = curr_pos + ldy * self.dt

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
        # self.new_goal = goal
        # if self.goal is None:
        #     self.goal = goal
        
    def build_graph(self):
        self.new_goal = None
        self.goal = None
        self.y0 = None

        self.linear_front_terms = 1.
        self.angular_front_terms = 1.

        
        self.n_linear_dmp = self.DMP_config['n_linear_dmp']
        self.n_angular_dmp = self.DMP_config['n_angular_dmp']
        self.dt = self.DMP_config['dt']
        self.tau = self.DMP_config['tau']
        self.apx = self.DMP_config['apx']
        self.gamma = self.DMP_config['gamma']
        self.app = self.DMP_config['app']
        self.apr = self.DMP_config['apr']
        self.ag = self.DMP_config['ag']
        self.ago = self.DMP_config['ago']
        
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