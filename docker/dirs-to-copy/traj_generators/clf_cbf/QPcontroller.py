#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from gurobipy import *
import math
import numpy as np
import os

default_config = {
    'k_cbf': 1,
    'epsilon':0.8,
    'num_states':3,
    'action_space': {'shape': (3,), 'upper_bound': [0.1, 0.1, 0.1], 'lower_bound': [-0.1,-0.1,-0.1]},
    'use_own_pose': False,
    'dt': 0.2,
    'log_dir': os.getcwd()
}

class QPcontroller:
    def __init__(self, config={}):
        self.QPcontroller_config = default_config
        self.QPcontroller_config.update(config)
        
        self.k_cbf = self.QPcontroller_config['k_cbf'] #CBF coefficient
        self.epsilon = self.QPcontroller_config['epsilon'] #Finite time CLF coefficient
        self.m = Model("CBF_CLF_QP")
        self.num_of_states = self.QPcontroller_config['num_states']
        self.num_of_control_inputs = self.QPcontroller_config['action_space']['shape'][0]
        self.u1_upper_lim = self.QPcontroller_config['action_space']['upper_bound'][0] # From Create Autonomy
        self.u1_lower_lim =  self.QPcontroller_config['action_space']['lower_bound'][0]
        self.u2_upper_lim =  self.QPcontroller_config['action_space']['upper_bound'][1]
        self.u2_lower_lim =  self.QPcontroller_config['action_space']['lower_bound'][1]
        self.u3_upper_lim =  self.QPcontroller_config['action_space']['upper_bound'][2]
        self.u3_lower_lim =  self.QPcontroller_config['action_space']['lower_bound'][2]

        self.dt = self.QPcontroller_config['dt']
        self.goal = None

        # Control Variables
        self.u1 = self.m.addVar(lb=self.u1_lower_lim, ub=self.u1_upper_lim,vtype=GRB.CONTINUOUS, name="x1_input_acceleration")
        self.u2 = self.m.addVar(lb=self.u2_lower_lim, ub=self.u2_upper_lim,vtype=GRB.CONTINUOUS, name="x2_input_acceleration")
        self.u3 = self.m.addVar(lb=self.u3_lower_lim, ub=self.u3_upper_lim,vtype=GRB.CONTINUOUS, name="x3_input_acceleration")
        # Soft Constraint Variable for CLF
        self.delta = self.m.addVar(lb=-30, ub=30,vtype=GRB.CONTINUOUS, name="relaxation_CLF")

        self.own_pose = None

    def reset(self, pose, vel):
        self.own_pose = pose[:3]
        
    def set_goal(self, goal):
        self.goal = goal

    def get_next_wp(self, action=[0,0,0], curr_pose=None, curr_vel=None, obs_info={}, clf=True, cbf=True):
        target_accel, target_vel, target_pose = self.generate_control(action, curr_pose[:3], obs_info, clf, cbf)
        return np.concatenate([target_accel, np.zeros(3)]), np.concatenate([target_vel, np.zeros(3)]), np.concatenate([target_pose, np.array([0,0,0,1])])

    def generate_control(self, action, x_current, obs_info={}, clf=True, cbf=True):
        self.m.remove(self.m.getConstrs())

        if self.QPcontroller_config['use_own_pose']:
            if self.own_pose is None:
                self.own_pose = np.array(x_current)
            else:
                x_current = np.array(self.own_pose)
        
        num_of_obstacles = len(obs_info) #For later use when more obstacles are involved

        # Loop through sphereical obstacles and set constraints
        for i in range(0,num_of_obstacles):
            if obs_info[i]['name'] == 'table':
                table_height = obs_info[i]['position'][2]# + 0.12

                # Table Constraint
                h_table = (x_current[2] - table_height)
                if cbf:
                    self.m.addConstr(self.u3 + self.k_cbf*h_table >= 0, "CBF_Constraint_for_table")
            else:
                pos = obs_info[i]['position']
                rad = obs_info[i]['radius']
                h = (x_current[0]-pos[0])**2 + (x_current[1]-pos[1])**2 + (x_current[2]-pos[2])**2 - rad**2

                if cbf:
                    self.m.addConstr(2*(x_current[0]-pos[0])*(self.u1+action[0]) + 2*(x_current[1]-pos[1])*(self.u2+action[1]) + 2*(x_current[2]-pos[2])*(self.u3+action[2]) + self.k_cbf*h >= 0, "CBF_Constraint_"+obs_info[i]['name'])
      
        # Initialize Cost Function
        self.cost_func = self.u1*self.u1+self.u2*self.u2+self.u3*self.u3 + self.delta*self.delta
        self.m.setObjective(self.cost_func,GRB.MINIMIZE)
      
        # CLF constraint
        V = 0.5*(x_current[0]-self.goal[0])**2 + 0.5*(x_current[1]-self.goal[1])**2 + 0.5*(x_current[2]-self.goal[2])**2

        partial_V_x1 = (x_current[0]-self.goal[0])
        partial_V_x2 = (x_current[1]-self.goal[1])
        partial_V_x3 = (x_current[2]-self.goal[2])

        if clf:
            self.m.addConstr(partial_V_x1*(self.u1+action[0]) + partial_V_x2*(self.u2+action[1]) + partial_V_x3*(self.u3+action[2]) + self.epsilon +self.delta <= -20, "Relaxed_CLF_constraint")


        #Stop optimizer from publsihing results to console - remove if desired
        self.m.Params.LogToConsole = 0

        #Solve the optimization problem
        self.m.optimize()
        self.solution = self.m.getVars()

        # get final decision variables
        self.control_u1 = self.solution[0].x
        self.control_u2 = self.solution[1].x
        self.control_u3 = self.solution[2].x
        
        # For debuging only, save model to view constraints etc.
        self.m.write(os.path.join(self.QPcontroller_config['log_dir'], "qp_model.lp"))

        target_vel = np.array([self.control_u1, self.control_u2, self.control_u3])

        target_pose = x_current + target_vel * self.dt
        
        if self.QPcontroller_config['use_own_pose']:
            self.own_pose = np.array(target_pose)
        
        return np.zeros(3), target_vel, target_pose
