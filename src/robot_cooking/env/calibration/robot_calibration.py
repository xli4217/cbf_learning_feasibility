import numpy as np
import tf

########
# Jaco #
########

'''
Jaco Frame
^ z         (Window)
|   ^ -y
|  /
| /
|/
---------->-x
'''

robot_frame = {
    "x": [1,0,0],
    'y': [0,1,0],
    'z': [0,0,1]
}



#### Z up configuration ####
'''
Motive Frame
^ z         (Window)
|   ^ x
|  /
| /
|/
---------->-y
'''
motive_frame = {
    'x': [0,-1,0],
    'y': [1,0,0],
    'z': [0,0,1]
}

p_in_motive_frame = np.array([-1.11, 2.325, 0.77])
R = np.array([motive_frame['x'], motive_frame['y'], motive_frame['z']]).T

p_in_robot_frame = np.array([0.42667, -0.311, -0.0145])
p_in_aligned_motive_frame = p_in_robot_frame - np.dot(R, p_in_motive_frame)

T_motive2jaco = np.eye(4)
T_motive2jaco[:3,:3] = R
T_motive2jaco[:3,3] =  np.array(p_in_aligned_motive_frame)


##########
# Baxter #
##########

'''
Baxter Frame
^ z         (Window)
|   ^ y
|  /
| /
|/
---------->x
'''

robot_frame = {
    "x": [1,0,0],
    'y': [0,1,0],
    'z': [0,0,1]
}



#### Z up configuration ####
'''
Motive Frame
^ z         (Window)
|   ^ x
|  /
| /
|/
---------->-y
'''
motive_frame = {
    'x': [0,1,0],
    'y': [-1,0,0],
    'z': [0,0,1]
}

p_in_motive_frame = np.array([-0.495, 2.123, 0.767])
R = np.array([motive_frame['x'], motive_frame['y'], motive_frame['z']]).T

p_in_robot_frame = np.array([0.77, -0.17, -0.22])
p_in_aligned_motive_frame = p_in_robot_frame - np.dot(R, p_in_motive_frame)

T_motive2baxter = np.eye(4)
T_motive2baxter[:3,:3] = R
T_motive2baxter[:3,3] =  np.array(p_in_aligned_motive_frame)

