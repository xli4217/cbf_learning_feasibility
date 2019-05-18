import numpy as np

waypoints_dict = {
    'neutral': np.array([0.178, -0.28, 0.228, 0.991, 0.13, 0.011, -0.016]),
    'relative_plate': np.array([0.0, 0.005, 0.02, 0.656, 0.754, -0.016, -0.016]),
    'toaster_waypoint': np.array([0.19668,-0.57735,0.28,0.98233,-0.153,0.0997,0.0404]),
    'toaster_absolute': np.array([0.167, -0.585, 0.08, 0.999, -0.011, -0.011, 0.0233]),
    'switch_pre': np.array([-0.06127, -0.481, 0.159, 0.7445, -0.6598, -0.00588, 0.1]),
    'switch_post': np.array([-0.06127, -0.481, 0.359, 0.7445, -0.6598, -0.00588, 0.1]),
    'switch_on': np.array([-0.0617,-0.575,0.1565,0.7515,-0.6523,-0.0085,0.098]),
    'switch_off': np.array([-0.0615,-0.4614,0.1147,0.7424,-0.6622,-0.00755,0.1]),
    'relative_condiment_pre': np.array([0.008, -0.105, -0.100, -0.594, -0.407, -0.421, 0.552]),
    'relative_condiment_post': np.array([0.008, -0.02, -0.100,-0.575, -0.474, -0.372, 0.554]),
    'relative_plate_apply_condiment_pre': np.array([0.038, 0.002, 0.15, -0.502, -0.540, -0.295, 0.608]),
    # 'relative_plate_apply_condiment_post': np.array([0.038, 0.002, 0.238, 0.632, -0.379, -0.534, -0.414]),
    'place_condiment': np.array([0.488,-0.0669,0.038,0.6135,0.3485,0.6266,-0.33]),
    'post_place_condiment': np.array([0.488,-0.0669,0.3,0.6135,0.3485,0.6266,-0.33])
}

