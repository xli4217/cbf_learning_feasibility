from future.utils import viewitems

ur5_joint_handles = ["UR5_joint"+str(i+1) for i in range(6)]

jaco_joint_handles = ["Jaco_joint"+str(i+1) for i in range(6)]

    
    
robot_handles = {
    'UR5':{
        'joint_handles': ur5_joint_handles,
        'gripper_handles': {
            'toggle_handle': 'robotiqClosing',
            'attachpoint_handle': 'ROBOTIQ_85_attachPoint',
            'prox_sensor_handle': 'ROBOTIQ_85_attachProxSensor'
        },
        'goal_handle': 'UR5_goal_dummy',
        'target_handle': 'Jaco_target',
        'particle_handle': 'UR5_goal_test'
    },
    'Jaco': {
        'joint_handles': jaco_joint_handles,
        'gripper_handles': {
            'toggle_handle': '',
            'attachpoint_handle': 'ROBOTIQ_85_attachPoint',
            'prox_sensor_handle': 'ROBOTIQ_85_attachProxSensor'
        },
        'goal_handle': 'Jaco_goal',
        'target_handle': 'Jaco_target',
        'particle_handle': 'Jaco_goal'
    }
}

obstacle_handles = [
    {'name': 'sphere_left', 'handle': 'sphere_left'},
    {'name': 'sphere_low', 'handle': 'sphere_low'},
    {'name': 'sphere_right', 'handle': 'sphere_right'},
]

object_handles = [
    {
        'name': 'obj',
        'handle': 'obj'
    }
]

