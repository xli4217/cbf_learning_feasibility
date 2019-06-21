import numpy as np
from lomap.classes import Fsa

from utils.utils import pos_distance, quat_distance, pose_distance, get_object_goal_pose
from utils import transformations
import json
import os
import sys
import time

default_config = {
    'robot': 'jaco',
    'mode': 'real'
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
            bunplate_rel = np.array([0.033, -0.022, 0.098, 0.807, 0.588, -0.018, 0.048])
            serve_plate_rel = bunplate_rel
        elif self.TLConfig_config['robot'] == 'baxter':
            bunplate_rel = np.array([-0.05, -0.05, -0.015, 0.656, 0.754, -0.016, -0.016])
            serve_plate_rel = bunplate_rel + np.array([0,0,0.04, 0,0,0,0])
        else:
            raise ValueError('robot not supported')
            
        # cpost = np.array([0.018, 0.051, 0.142, -0.492, 0.546, 0.416, 0.535])
        cpost = np.array([0.048, 0.043, 0.124, -0.305, 0.589, 0.654, 0.364])

        # Mcpre = transformations.quaternion_matrix(cpre[3:])
        # Mrot = transformations.euler_matrix(0, 0, -180 * np.pi/180)
        # Mcpost = Mcpre.dot(Mrot)
        # cpost = np.concatenate([cpre[:3], transformations.quaternion_from_matrix(Mcpost)])

        cpostpost = np.array([0.020, -0.038, 0.143, -0.468, 0.566, 0.394, 0.552])
        
        self.OBJECT_RELATIVE_POSE = {
            'hotdogplate': np.array([0.0, 0.005, 0.02, 0.656, 0.754, -0.016, -0.016]),
            'bunplate': bunplate_rel,
            'baxterhotdogplate': np.array([0.570, -0.799, -0.231, -0.108, 0.993, -0.042, 0.030]),
            'serveplate': serve_plate_rel,
            'grill': np.array([0.007, -0.032, 0.003, 0.710, 0.704, 0.017, 0.027]), # this needs confirmation
            'switchon': np.array([-0.001, -0.247-0.05, 0.076, 0.993, 0.072, 0.064, 0.073]),
            'switchoff': np.array([-0.117, -0.293, 0.038, 0.990, 0.126, -0.071, 0.002]),
            'condimentpre': np.array([0.006, -0.112, -0.020, 0.608, 0.373, 0.553, -0.430]),
            'condimentpost': np.array([0.023, -0.023, -0.021, 0.604, 0.375, 0.551, -0.436]),
            'relativeplateapplycondimentpre': cpost,
            'relativeplateapplycondimentpost': cpost + np.array([0, -0.14, 0, 0, 0, 0, 0]),
            'applycondimentpre': np.array([0.504, -0.403, 0.132, 0.107, 0.732, 0.165, 0.652]),
            'applycondimentpost': np.array([0.401, -0.399, 0.125, 0.163, 0.732, 0.153, 0.643]),
            'placecondimentgoal': np.array([0.488,-0.0669,0.04,0.6135,0.3485,0.6266,-0.33]),
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
            'condimentapplied': [44],
            'hotdogprob': [45]
        }

        self.obs_dim = 46

        ##############
        # TL Related #
        ##############
        self.state_idx_map = self.STATE_IDX_MAP


        self.PREDICATES = {
            'moveto_hotdogplate': lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'hotdogplate', 'hotdogplate'),
            'moveto_serveplate': lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'serveplate', 'serveplate'),

            'moveto_bunplate': lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'bunplate', 'bunplate'),
            'moveto_world_baxterhotdogplate':lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'world', 'baxterhotdogplate'),
            'moveto_world_baxterneutral':lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'world', 'baxterneutral'),
            'moveto_world_jaconeutral':lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'world', 'jaconeutral'),
            'moveto_grill': lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'grill', 'grill'),
            'moveto_condiment_condimentpre' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp,'condiment', 'condimentpre'),
            'moveto_condiment_condimentpost' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp, 'condiment', 'condimentpost'),
            'moveto_world_placecondimentgoal' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp, 'world', 'placecondimentgoal'),
            'moveto_bunplate_relativeplateapplycondimentpre' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp, 'bunplate', 'relativeplateapplycondimentpre'),
            'moveto_bunplate_relativeplateapplycondimentpost' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp, 'bunplate', 'relativeplateapplycondimentpost'),
            'moveto_world_applycondimentpre' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp, 'world', 'applycondimentpre'),
            'moveto_world_applycondimentpost' : lambda s, a=None, sp=None: self.moveto_robustness(s,a,sp, 'world', 'applycondimentpost'),
            'applycondiment': lambda s, a=None, sp=None: self.apply_condiment_robustness(s, a, sp),
            'flipswitchon': lambda s, a=None, sp=None: self.switch_robustness(s,a,sp, 'on'),
            'flipswitchoff': lambda s, a=None, sp=None: self.switch_robustness(s,a,sp, 'off'),
            'closegripper': lambda s, a=None, sp=None:  self.gripper_robustness(s,a,sp, 'close'),
            'opengripper': lambda s, a=None, sp=None:  self.gripper_robustness(s,a,sp,'open'),
            'squeezegripper': lambda s, a=None, sp=None:  self.gripper_robustness(s,a,sp,'squeeze'),
            'unsqueezegripper': lambda s, a=None, sp=None:  self.gripper_robustness(s,a,sp,'unsqueeze'),
            'inservezone_serveplate': lambda s, a=None, sp=None:  self.in_serve_zone_robustness(s,a,sp,'serveplate'),
            'hotdogready': lambda s, a=None, sp=None:  self.hotdogready_robustness(s,a,sp)
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
        state[self.STATE_IDX_MAP['hotdogprob']] = skill_arg['hotdogprob']

        
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
            pos_dist_m = 0.04
        else:
            pos_dist_m = pos_dist
            
        if self.TLConfig_config['mode'] == 'sim':
            if pos_dist < 0.04:
                pos_dist_m = 0
            else:
                pos_dist_m = pos_dist
                
        mapped_pos_rob = (0.02 - pos_dist_m) / 0.02

        if quat_dist > 0.5:
            quat_dist_m = 0.5
        else:
            quat_dist_m = quat_dist
            
        mapped_quat_rob = (0.25 - quat_dist_m) / 0.25

        
        rob = np.minimum(mapped_pos_rob, mapped_quat_rob)

        # if rel_pose_name == 'relativeplateapplycondimentpost':
        #     print(object_pose)
        #     print(ee_pose)
        #     print(goal_pose)
        #     print(pos_dist, quat_dist)
        #     print(mapped_pos_rob, mapped_quat_rob)
        #     print("---")
            
        return (rob, 'action')

    def hotdogready_robustness(self, s=None, a=None, sp=None):
        hotdogprob = s[self.state_idx_map['hotdogprob']]
        if hotdogprob > 0.2:
            return (100, 'nonaction')
        else:
            return (-100, 'nonaction')
            
    def switch_robustness(self, s=None, a=None, sp=None, on_or_off='on'):
        switch_on = s[self.state_idx_map['switchon']]

        if on_or_off == 'on':
            if switch_on > 0:
                return (1., 'nonaction')
            else:
                return (-1., 'nonaction')
        elif on_or_off == 'off':
            if switch_on > 0:
                return (-1., 'nonaction')
            else:
                return (1., 'nonaction')
   
  
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
        elif oc == 'squeeze':
            # rob = float(s[self.state_idx_map['gripper_state']]) - 0.83
            rob = float(s[self.state_idx_map['gripper_state']]) * 2 - 1
        elif oc == 'unsqueeze':
            rob = 0.9 - float(s[self.state_idx_map['gripper_state']])
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
   



