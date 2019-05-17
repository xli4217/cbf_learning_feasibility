import numpy as np
from lomap.classes import Fsa
from utils.utils import pos_distance, quat_distance, pose_distance
from utils import transformations

def get_object_goal_pose(object_pose, goal_rel_pose):
    assert len(object_pose) == len(goal_rel_pose) == 7
    
    Ms = transformations.quaternion_matrix(object_pose[3:])
    Ms[:3,3] = object_pose[:3]

    M_rel = transformations.quaternion_matrix(goal_rel_pose[3:])
    M_rel[:3,3] = goal_rel_pose[:3]

    M = Ms.dot(M_rel)

    quat_M = transformations.quaternion_from_matrix(M)
    pose_M = np.concatenate([M[:3,3], quat_M])

    return pose_M
    
############# 
# Key Poses #
#############

KEY_POSITIONS = {
    'neutral': np.array([0.178, -0.28, 0.228, 0.991, 0.13, 0.011, -0.016]),
    'condiment_placement_goal': np.array([0.488,-0.0669,0.038,0.6135,0.3485,0.6266,-0.33]),
    'switch_on_goal': np.array([-0.0617,-0.575,0.1565,0.7515,-0.6523,-0.0085,0.098]),
    #'switch_on_goal': np.array([ -0.054, -0.569 + 0.02, 0.144, 0,0,0,1])
}

OBJECT_RELATIVE_POSE = {
    'hotdogplate': np.array([0.0, 0.005, 0.02, 0.656, 0.754, -0.016, -0.016]),
    'bunplate': np.array([0.0, 0.005, 0.02, 0.656, 0.754, -0.016, -0.016]),
    'grill': np.array([0.0, 0.005, 0.02, 0.656, 0.754, -0.016, -0.016]), # this needs confirmation
    'condimentpre': np.array([0.008, -0.105, -0.100, -0.594, -0.407, -0.421, 0.552]),
    'condimentpost': np.array([0.008, -0.02-0.05, -0.100,-0.575, -0.474, -0.372, 0.554]),
    'relativeplateapplycondimentpre': np.array([0.038, 0.002, 0.15, -0.502, -0.540, -0.295, 0.608]),
    'postplacecondiment': np.array([0.488,-0.0669,0.3,0.6135,0.3485,0.6266,-0.33])
}


STATE_IDX_MAP = {
    'end_effector_pose': [0, 7],
    'gripper_state': [7],
    'condiment': [8, 15],
    'hotdogplate': [15, 22],
    'bunplate': [22, 29],
    'grill': [29, 36],
    'switchon': [36]
}

def construct_skill_state(skill_arg):
    state = np.zeros(37)

    state[STATE_IDX_MAP['end_effector_pose'][0]:STATE_IDX_MAP['end_effector_pose'][1]] = skill_arg['curr_pose']
    state[STATE_IDX_MAP['gripper_state'][0]] = skill_arg['gripper_state']
    state[STATE_IDX_MAP['condiment'][0]:STATE_IDX_MAP['condiment'][1]] = skill_arg['obj_poses']['condiment']
    state[STATE_IDX_MAP['hotdogplate'][0]:STATE_IDX_MAP['hotdogplate'][1]] = skill_arg['obj_poses']['hotdogplate']
    state[STATE_IDX_MAP['bunplate'][0]:STATE_IDX_MAP['bunplate'][1]] = skill_arg['obj_poses']['bunplate']
    state[STATE_IDX_MAP['grill'][0]:STATE_IDX_MAP['grill'][1]] = skill_arg['obj_poses']['grill']
    state[STATE_IDX_MAP['switchon']] = skill_arg['switchon']
    
    return state
    
##############
# TL Related #
##############
state_idx_map = STATE_IDX_MAP

def moveto_robustness(s=None, a=None, sp=None, object_name=None, rel_pose_name=None):
    ee_pose = s[state_idx_map['end_effector_pose'][0]:state_idx_map['end_effector_pose'][1]]
    object_pose = s[state_idx_map[object_name][0]:state_idx_map[object_name][1]]
    goal_pose = get_object_goal_pose(object_pose, OBJECT_RELATIVE_POSE[rel_pose_name])

    pos_dist, quat_dist = pose_distance(ee_pose, goal_pose)
    rob = np.minimum(0.01 - pos_dist, 0.15 - quat_dist)
    
    return rob 
        

def switch_robustness(s=None, a=None, sp=None, sim_or_real='sim'):
    switch_on = s[state_idx_map['switchon']]
    return switch_on
    # if sim_or_real == 'sim':
    #     if switch_angle < 1.:
    #         return 0.1
    #     else:
    #         return -0.1
    # elif sim_or_real == 'real':
    #     return -0.1
    # else:
    #     raise ValueError('mode not supported')
    
PREDICATES = {
    'moveto_hotdogplate': lambda s, a=None, sp=None: moveto_robustness(s,a,sp,'hotdogplate', 'hotdogplate'),
    'moveto_bunplate': lambda s, a=None, sp=None: moveto_robustness(s,a,sp,'bunplate', 'bunplate'),
    'moveto_grill': lambda s, a=None, sp=None: moveto_robustness(s,a,sp,'grill', 'grill'),
    'moveto_condiment_condimentpre' : lambda s, a=None, sp=None: moveto_robustness(s,a,sp,'condiment', 'condimentpre'),
    'moveto_condiment_condimentpost' : lambda s, a=None, sp=None: moveto_robustness(s,a,sp, 'condiment', 'condimentpost'),
    'moveto_bunplate_relativeplateapplycondimentpre' : lambda s, a=None, sp=None: moveto_robustness(s,a,sp, 'bunplate', 'relativeplateapplycondimentpre'),
    'flipswitchon': lambda s, a=None, sp=None: switch_robustness(s,a,sp, sim_or_real='sim'),
    'closegripper': lambda s, a=None, sp=None:  s[state_idx_map['gripper_state']] - 0.8,
    'opengripper': lambda s, a=None, sp=None:  0.4 - s[state_idx_map['gripper_state']]
}

