from future.utils import viewitems

ur5_joint_handles = ["UR5_joint"+str(i+1) for i in range(6)]

jaco_joint_handles = ["Jaco_joint"+str(i+1) for i in range(6)]

world_frame_handle = "Jaco_World_Frame"    
    
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
        'particle_handle': 'UR5_goal_test',
        'ee_sample_region_handle': "ee_sample_region",
        'ee_motion_region_handle': "ee_motion_region"
    },
    'Jaco': {
        'joint_handles': jaco_joint_handles,
        'gripper_handles': {
            'toggle_handle': '',
            'attachpoint_handle': 'ROBOTIQ_85_attachPoint',
            'prox_sensor_handle': 'ROBOTIQ_85_attachProxSensor'
        },
        'goal_handle': 'Jaco_goal',
        'target_handle': 'Jaco_target_dummy',
        'particle_handle': 'Jaco_goal_test',
        'ee_sample_region_handle': "ee_sample_region",
        'ee_motion_region_handle': "ee_motion_region"
    }
}

obstacle_handles = [
    {'name': 'sphere_left', 'handle': 'sphere_left'},
    {'name': 'sphere_low', 'handle': 'sphere_low'},
    {'name': 'sphere_right', 'handle': 'sphere_right'}
]

object_handles = [
    {'name': 'hotdog_cooker', 'handle': 'hotdog_cooker'},
    {'name': 'toaster_button', 'handle': 'toaster_joint_frame'},
    {'name': 'toaster_button_joint', 'handle': 'button_revolute_joint'}
]

