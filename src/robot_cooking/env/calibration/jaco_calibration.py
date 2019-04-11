import numpy as np
import tf

'''
Jaco Frame
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



#### Y up configuration ####
'''
Motive Frame
^ y         (Window)
|   ^ x
|  /
| /
|/
---------->z
'''
motive_frame = {
    'x': [0,1,0],
    'y': [0,0,1],
    'z': [1,0,0]
}

p_in_motive_frame = np.array([-1.08164, 0.9665, -2.357])
R = np.array([[0,1,0], [0,0,1], [1,0,0]]).T

#### Z up configuration
# '''
# Motive Frame
# (Window)   ^ z         
#            |   ^ x
#            |  /
#            | /
# y          |/
# <-----------
# '''

# motive_frame = {
#     'x': [0,1,0],
#     'y': [-1,0,0],
#     'z': [0,0,1]
# }
# R = np.array([[0,1,0], [-1,0,0], [0,0,1]]).T
# p_in_motive_frame = np.array([-1.08164, 2.357, 0.9665])

p_in_robot_frame = np.array([0.869, -0.078, -0.028])
p_in_aligned_motive_frame = p_in_robot_frame - np.dot(R, p_in_motive_frame)

#####################
# Create Transforms #
#####################

T_motive2baxter = np.eye(4)
T_motive2baxter[:3,:3] = R
T_motive2baxter[:3,3] =  np.array(p_in_aligned_motive_frame)

