import numpy as np
from lomap.classes import Fsa

from utils.utils import pos_distance, quat_distance, pose_distance, get_object_goal_pose
from utils import transformations
import json
import os
import sys
import time

default_config = {
    'robot': 'jaco'
}

class TLConfig(object):

    def __init__(self, config={}):
        self.TLConfig_config = default_config
        self.TLConfig_config.update(config)

        
        self.baxter_env_config_json_path = os.path.join(os.environ['RC_PATH'],
                                                        'src',
                                                        'robot_cooking',
                                                        'env',
                                                        'config',
                                                        'baxter_env_config.json')

        with open(self.baxter_env_config_json_path) as f:
            self.baxter_env_config = json.loads(f.read())

        ############# 
        # Key Poses #
        #############

        if self.TLConfig_config['robot'] == 'jaco':
            plate_rel = np.array([0.0, 0.005, 0.02, 0.656, 0.754, -0.016, -0.016])
            serve_plate_rel = plate_rel
        elif self.TLConfig_config['robot'] == 'baxter':
            plate_rel = np.array([0.0, 0.005, -0.015, 0.656, 0.754, -0.016, -0.016])
            serve_plate_rel = plate_rel + np.array([0,0,0.03, 0,0,0,0])
        else:
            raise ValueError('robot not supported')
            
        cpre = np.array([0.038, 0.002, 0.2, -0.502, -0.540, -0.295, 0.608])
        Mcpre = transformations.quaternion_matrix(cpre[3:])
        Mrot = transformations.euler_matrix(0, 0, -180 * np.pi/180)
        Mcpost = Mcpre.dot(Mrot)
        cpost = np.concatenate([cpre[:3], transformations.quaternion_from_matrix(Mcpost)])
        
        self.OBJECT_RELATIVE_POSE = {
            'hotdogplate': np.array([0.0, 0.005, 0.02, 0.656, 0.754, -0.016, -0.016]),
            'bunplate': plate_rel,
            'serveplate': serve_plate_rel,
            'grill': np.array([0.007, -0.012, 0.006, 0.710, 0.704, 0.017, 0.027]), # this needs confirmation
            'switchon': np.array([-0.001, -0.247-0.01, 0.076-0.012, 0.993, 0.072, 0.064, 0.073]),
            'condimentpre': np.array([0.022, -0.129, -0.09, -0.594, -0.433, -0.426, 0.528]),
            'condimentpost': np.array([0.022, -0.129+0.11, -0.09, -0.594, -0.433, -0.426, 0.528]),
            'relativeplateapplycondimentpre': cpre,
            'relativeplateapplycondimentpost': cpost,
            'placecondimentgoal': np.array([0.488,-0.0669,0.038,0.6135,0.3485,0.6266,-0.33]),
            'baxterneutral': np.array([0.729, -0.29, 0.19, -0.052, 0.998, 0.031, 0.020]),
            'jaconeutral': np.array([-0.075, -0.316, 0.26, 0.779, -0.621, -0.052, -0.076])
        }


        self.STATE_IDX_MAP = {
            'end_effector_pose': [0, 7],
            'gripper_state': [7],
            'condiment': [8, 15],
            'hotdogplate': [15, 22],
            'bunplate': [22, 29],
            'serveplate': [29, 36],
            'grill': [36, 43],
            'switchon': [43],
            'condimentapplied': [44]
        }

        self.obs_dim = 45

        ##############
        # TL Related #
        ##############
        self.state_idx_map = self.STATE_IDX_MAP


        self.PREDICATES = {
            'moveto_hotdogplate': lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'hotdogplate', 'hotdogplate'),
            'moveto_serveplate': lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'serveplate', 'serveplate'),

            'moveto_bunplate': lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'bunplate', 'bunplate'),
            'moveto_world_baxterneutral':lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'world', 'baxterneutral'),
            'moveto_world_jaconeutral':lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'world', 'jaconeutral'),
            'moveto_grill': lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'grill', 'grill'),
            'moveto_condiment_condimentpre' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'condiment', 'condimentpre'),
            'moveto_condiment_condimentpost' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp, 'condiment', 'condimentpost'),
            'moveto_world_placecondimentgoal' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp, 'world', 'placecondimentgoal'),
            'moveto_bunplate_relativeplateapplycondimentpre' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp, 'bunplate', 'relativeplateapplycondimentpre'),
            'moveto_bunplate_relativeplateapplycondimentpost' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp, 'bunplate', 'relativeplateapplycondimentpost'),
            'applycondiment': lambda s, a=None, sp=None: self.apply_condiment_robustness(s, a, sp),
            'flipswitchon': lambda s, a=None, sp=None: self.switch_robustness(s,a,sp),
            'closegripper': lambda s, a=None, sp=None:  self.gripper_robustness(s,a,sp, 'close'),
            'opengripper': lambda s, a=None, sp=None:  self.gripper_robustness(s,a,sp,'open'),
            'inservezone_serveplate': lambda s, a=None, sp=None:  self.in_serve_zone_robustness(s,a,sp,'serveplate')
        }

        
    def construct_skill_state(self, skill_arg):
        state = np.zeros(self.obs_dim)
        
        state[self.STATE_IDX_MAP['end_effector_pose'][0]:self.STATE_IDX_MAP['end_effector_pose'][1]] = skill_arg['curr_pose']
        state[self.STATE_IDX_MAP['gripper_state'][0]] = skill_arg['gripper_state']
        state[self.STATE_IDX_MAP['condiment'][0]:self.STATE_IDX_MAP['condiment'][1]] = skill_arg['obj_poses']['condiment']
        state[self.STATE_IDX_MAP['hotdogplate'][0]:self.STATE_IDX_MAP['hotdogplate'][1]] = skill_arg['obj_poses']['hotdogplate']
        state[self.STATE_IDX_MAP['bunplate'][0]:self.STATE_IDX_MAP['bunplate'][1]] = skill_arg['obj_poses']['bunplate']
        state[self.STATE_IDX_MAP['serveplate'][0]:self.STATE_IDX_MAP['serveplate'][1]] = skill_arg['obj_poses']['serveplate']
        state[self.STATE_IDX_MAP['grill'][0]:self.STATE_IDX_MAP['grill'][1]] = skill_arg['obj_poses']['grill']
        state[self.STATE_IDX_MAP['switchon']] = skill_arg['switchon']
        state[self.STATE_IDX_MAP['condimentapplied']] = skill_arg['condimentapplied']
    
        return state
            

    def moveto_robustness(self, s=None, a=None, sp=None, object_name=None, rel_pose_name=None):
        '''
        pos_dist < 0.02, quat_dist < 0.25 --> satisfy
        
        robustness mapped to [-1, 1] where 1 is satisfy
        '''

        ee_pose = s[self.state_idx_map['end_effector_pose'][0]:self.state_idx_map['end_effector_pose'][1]]
        if object_name == 'world':
            goal_pose = self.OBJECT_RELATIVE_POSE[rel_pose_name]
        else:
            object_pose = s[self.state_idx_map[object_name][0]:self.state_idx_map[object_name][1]]
            goal_pose = get_object_goal_pose(object_pose, self.OBJECT_RELATIVE_POSE[rel_pose_name])

        pos_dist, quat_dist = pose_distance(ee_pose, goal_pose)

        if pos_dist > 0.04:
            pos_dist = 0.04
        
        mapped_pos_rob = (0.02 - pos_dist) / 0.02

        if quat_dist > 0.5:
            quat_dist = 0.5

        mapped_quat_rob = (0.25 - quat_dist) / 0.25

        rob = np.minimum(mapped_pos_rob, mapped_quat_rob)
        
        return (rob, 'action')
        

    def switch_robustness(self, s=None, a=None, sp=None):
        switch_on = s[self.state_idx_map['switchon']]
        if switch_on > 0:
            return (1., 'nonaction')
        else:
            return (-1., 'nonaction')
  
    def apply_condiment_robustness(self, s, a=None, sp=None):
        '''
        0.1 if true, -0.1 if false
        '''
        rob = s[self.state_idx_map['condimentapplied']]
        return (float(rob), 'action')

    def gripper_robustness(self, s, a=None, sp=None, oc='open'):
        '''
        returns measure of completion in the range [-1,1]
        '''
        v_open = 0.
        v_close = 1.
        if oc == 'open':
            rob = v_open - float(s[self.state_idx_map['gripper_state']]) * 2 + 1
        elif oc == 'close':
            rob = float(s[self.state_idx_map['gripper_state']]) * 2 - 1  
        else:
            raise ValueError()

        return (float(rob), 'action')

    def in_serve_zone_robustness(self, s, a=None, sp=None, object_name='serveplate'):
        serve_zone_center = self.baxter_env_config['serve_zone']['init_pose'][:3]
        serve_zone_size = self.baxter_env_config['serve_zone']['scale']
        object_pose = s[self.state_idx_map[object_name][0]:self.state_idx_map[object_name][1]]

        x_max = serve_zone_center[0] + serve_zone_size[0]/2
        y_max = serve_zone_center[1] + serve_zone_size[1]/2
        z_max = serve_zone_center[2] + serve_zone_size[2]/2
        
        x_min = serve_zone_center[0] - serve_zone_size[0]/2
        y_min = serve_zone_center[1] - serve_zone_size[1]/2
        z_min = serve_zone_center[2] - serve_zone_size[2]/2

        rob = np.min([object_pose[0] - x_min, x_max - object_pose[0],
                      object_pose[1] - y_min, y_max - object_pose[1],
                      object_pose[2] - z_min, z_max - object_pose[2]])

        if rob > 0:
            return (100, 'nonaction')
        else:
            return (-100, 'nonaction')
   



