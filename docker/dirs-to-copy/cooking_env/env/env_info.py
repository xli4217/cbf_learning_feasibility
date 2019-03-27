from future.utils import viewitems

ur5_joint_handles = ["UR5_joint"+str(i+1) for i in range(6)]

jaco_joint_handles = ["Jaco_joint"+str(i+1) for i in range(6)]

    
    
robot_handles = {
    'UR5':{
        'joint_handles': ur5_joint_handles,
        'gripper_handles': {
            'toggle_handle': '',
            'connector_handle': 'ROBOTIQ_85_attachPoint',
            'prox_sensor_handle': 'ROBOTIQ_85_attachProxSensor'
        },
        'goal_handle': 'Jaco_goal'
    },
    'Jaco': {
        'joint_handles': jaco_joint_handles,
        'jaco_hand_handles': {
            'toggle_handle': '',
            'connector_handle': 'ROBOTIQ_85_attachPoint',
            'prox_sensor_handle': 'ROBOTIQ_85_attachProxSensor'
        },
        'goal_handle': 'Jaco_goal'
    }
}

object_handles = []