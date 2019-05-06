import numpy as np

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

PREDICATES = {
    
}