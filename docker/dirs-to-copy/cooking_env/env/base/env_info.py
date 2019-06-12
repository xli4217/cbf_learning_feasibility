from future.utils import viewitems

baxter_joint_handles = ["Baxter_rightArm_joint"+str(i+1) for i in range(6)]

jaco_joint_handles = ["Jaco_joint"+str(i+1) for i in range(6)]

    
robot_handles = {
    'Baxter':{
        'world_frame_handle': 'Baxter_World_Frame',
        'joint_handles': baxter_joint_handles,
        'gripper_handles': {
            'toggle_handle': 'baxterClosing',
            'attachpoint_handle': '',
            'prox_sensor_handle': ''
        },
        'goal_handle': 'Baxter_goal',
        'particle_target_handle': 'Baxter_particle_target',
        'particle_handle': 'Baxter_particle',
        'ee_target_handle': 'Baxter_target_dummy',
        'ee_handle': 'Baxter_ee_frame',
        'ee_sample_region_handle': "Baxter_ee_sample_region",
        'ee_motion_region_handle': "Baxter_ee_motion_region"
    },
    'Jaco': {
        'world_frame_handle': 'Jaco_World_Frame',
        'joint_handles': jaco_joint_handles,
        'gripper_handles': {
            'toggle_handle': 'jacoClosing',
            'attachpoint_handle': '',
            'prox_sensor_handle': ''
        },
        'goal_handle': 'Jaco_goal',
        'particle_target_handle': 'Jaco_particle_target',
        'particle_handle': 'Jaco_particle',
        'ee_target_handle': 'Jaco_target_dummy',
        'ee_handle': 'Jaco_ee_frame',
        'ee_sample_region_handle': "Jaco_ee_sample_region",
        'ee_motion_region_handle': "Jaco_ee_motion_region"
    }
}

obstacle_handles = [
    {'name': 'sphere_left', 'handle': 'sphere_left'},
    {'name': 'sphere_low', 'handle': 'sphere_low'},
    {'name': 'sphere_right', 'handle': 'sphere_right'}
]

object_handles = [
    {'name': 'grill', 'handle': 'grill_mapped'},
    {'name': 'toaster_button', 'handle': 'toaster_joint_frame'},
    {'name': 'toaster_button_joint', 'handle': 'button_revolute_joint'},
    {'name': 'hotdogplate', 'handle': 'hotdogplate'},
    {'name': 'bunplate', 'handle': 'bunplate'},
    {'name': 'condiment', 'handle': 'condiment_frame'},
    {'name': 'serveplate', 'handle': 'serveplate'},
]

