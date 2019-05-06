import numpy as np
from lomap.classes import Fsa

#########
# Utils # 
#########

def pos_distance(p1, p2):
    pos_dist = np.linalg.norm(p1[:3] - p2[:3])
    return pos_dist

def quat_distance(p1, p2):
    quat_dist_arg = 2 * np.inner(p1[3:], p2[3:]) - 1
    quat_dist_arg = np.modf(quat_dist_arg)[0]

    if quat_dist_arg > 0.99:
        quat_distance = 0.
    elif quat_dist_arg < -0.99:
        quat_distance = 0
    else:
        quat_distance = np.arccos(quat_dist_arg)

    return quat_distance
        
############# 
# Key Poses #
#############

KEY_POSITIONS = {
    'neutral': np.array([0.178, -0.28, 0.228, 0.991, 0.13, 0.011, -0.016]),
    'condiment_placement_location': np.array([0.488,-0.0669,0.038,0.6135,0.3485,0.6266,-0.33]),
}

OBJECT_RELATIVE_POSE = {
    'plate': np.array([0.0, 0.005, 0.02, 0.656, 0.754, -0.016, -0.016]),
    'grill': np.array([0.167, -0.585, 0.08, 0.999, -0.011, -0.011, 0.0233]), # this needs confirmation
    'condiment': np.array([0.008, -0.105, -0.100, -0.594, -0.407, -0.421, 0.552]),
}


STATE_IDX_MAP = {
    'end_effector_pose': [0, 7],
    'gripper_position': [7],
    'condiment_pose': [8, 15],
    'plate_pose': [15, 22],
}


##############
# TL Related #
##############

PREDICATES = {
    'MoveTo': lambda s, g, state_idx_map: np.minimum(0.01 - pos_distance(s[state_idx_map['end_effector_pose'][0]:state_idx_map['end_effector_pose'][1]], g), 0.1 - quat_distance(s[state_idx_map['end_effector_pose'][0]:state_idx_map['end_effector_pose'][1]], g)),
    'CloseGripper': lambda s, g, state_idx_map: 0.2 - 0
}

