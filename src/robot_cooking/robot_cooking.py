import sys
import rospy
import numpy as np
import copy
import os
import time

from kinova_api.kinova_driver_utils import KinovaDriverUtils
from kinova_api.kinova_moveit_utils import KinovaMoveitUtils

import tf2_ros
import tf
from visualization_msgs.msg import *
from  geometry_msgs.msg import *

default_config = {
    'rate': 10,
    "WPGenerator": {
        'type': None,
        'config': {}
    },
    'DriverUtils': {
        'type': KinovaDriverUtils,
        'config':  {
            'init_node': False,
            'cmd_publish_rate': 100,
            "gripper_move_group": 'gripper',
            "reference_link": "j2s7s300_link_base",
            'joint_vel_bound': {
                'upper': 4 * np.array([1.5,1.5,1.5,1.5,2.,2.,5.]),
                'lower': -4 * np.array([1.5,1.5,1.5,1.5,2.,2.,5.]),    
            },
            'safe_workspace': {
                # safe zone defined here takes precedence
                'env_json_path': os.path.join(os.path.dirname(os.path.abspath(__file__))
, 'env', 'config', 'env_config.json'),
                'x': [-0.1, 0.6],
                'y': [-0.25, 0.16],
                'z': [0.08 ,0.8]
            }
        }
    },
    'MoveitUtils': {
        'type': KinovaMoveitUtils,
        'config':  {
            "init_node": False,
            "arm_move_group": 'arm',
            "gripper_move_group": 'gripper',
            "reference_link": "j2s7s300_link_base",
            "end_effector_link": "j2s7s300_end_effector",
            "neutral_joint_positions": [4.935252420414662, 2.485132499632967, -0.6278464188357148, 0.763206574666036, 4.204106310531545, 4.064730307514466, 4.778262802580017, 1.1112613559570534, 1.108797399316512, 1.108797399316512],
            "joint_names": ["j2s7s300_joint_1", "j2s7s300_joint_2", "j2s7s300_joint_3", "j2s7s300_joint_4", "j2s7s300_joint_5", "j2s7s300_joint_6", "j2s7s300_joint_7"],
            "env_json_path": "",
        }
    }
}

class RobotCooking(object):

    def __init__(self, config={}):
        self.RobotCooking_config = default_config
        self.RobotCooking_config.update(config)

        rospy.init_node("robot_cooking", anonymous=False)
        rospy.on_shutdown(self.cleanup)
        
        self.driver_utils = self.RobotCooking_config['DriverUtils']['type'](self.RobotCooking_config['DriverUtils']['config'])
        self.moveit_utils = self.RobotCooking_config['MoveitUtils']['type'](self.RobotCooking_config['MoveitUtils']['config'])

        self.rate = rospy.Rate(self.RobotCooking_config['rate'])

        self.wp_gen = self.RobotCooking_config['WPGenerator']['type'](self.RobotCooking_config['WPGenerator']['config'])
        #### Initialize tf ####
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def update_goal_tf(self, pt):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = "goal"

        t.transform.translation.x = pt[0]
        t.transform.translation.y = pt[1]
        t.transform.translation.z = pt[2]
        
        t.transform.rotation.x = pt[3]
        t.transform.rotation.y = pt[4]
        t.transform.rotation.z = pt[5]
        t.transform.rotation.w = pt[6]

        self.tf_broadcaster.sendTransform(t)
       
        
    def update_pose_target_tf(self, pt):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = "pose_target"

        t.transform.translation.x = pt[0]
        t.transform.translation.y = pt[1]
        t.transform.translation.z = pt[2]
        
        t.transform.rotation.x = pt[3]
        t.transform.rotation.y = pt[4]
        t.transform.rotation.z = pt[5]
        t.transform.rotation.w = pt[6]

        self.tf_broadcaster.sendTransform(t)
        
    def servo_to_pose_target(self, pt):
        assert len(pt) == 7

        # update tf
        self.update_pose_target_tf(pt)
        
        # calculate the joint positions using ik
        ik_sol = self.moveit_utils.ik(position=pt[:-4], orientation=pt[-4:])
        ik_jp = ik_sol[:-3]

        damping = 0.5
        natural_freq = 1.0
        kp = (2 * np.pi * natural_freq) ** 2
        kd = 2 * damping * 2 * np.pi * natural_freq

        joint_angle_diff = ik_jp - self.driver_utils.get_joint_values()
        joint_vel = self.driver_utils.get_joint_velocities()

        jv = kp * joint_angle_diff + kd * joint_vel

        ee_pos, ee_quat = self.driver_utils.get_ee_pose()
        pose_distance = np.linalg.norm(ee_pos - pt[:3])
        quat_distance = np.arccos(2 * np.inner(ee_quat, pt[3:]) - 1)
        
        if not self.driver_utils.is_tool_in_safe_workspace():
            print('tool not in safe workspace')

        # start servoing
        if (pose_distance > 0.005 or quat_distance > 0.08) and self.driver_utils.is_tool_in_safe_workspace():
            self.driver_utils.pub_joint_velocity(jv, duration_sec=1./self.RobotCooking_config['rate'])
            self.rate.sleep()
            return False
        else:
            return True
            
    def run(self):
        from waypoints import waypoints_dict
        ## go to neutral
        goal = waypoints_dict['neutral']
        self.wp_gen.set_goal(goal[:3])
        dist = 5
        
        while dist > 0.01:
            ee_pos, ee_quat = self.driver_utils.get_ee_pose()
            # ee_linear_vel, ee_angular_vel = self.driver_utils.get_ee_velocity()
            ee_linear_vel = np.array([0,0,0])
            _, dwp, wp = self.wp_gen.get_next_wp(np.array([0,0,0]), ee_pos, ee_linear_vel)
            # pt = np.concatenate([ee_pos+dwp, ee_quat])
            pt = np.concatenate([wp, ee_quat])

            self.servo_to_pose_target(pt)
            
            dist = np.linalg.norm(ee_pos-goal[:3])
            print(dist)
            
            self.update_goal_tf(goal)

        print("finished")
            
    def get_tf_pose(self, src_frame, target_frame):
        try:
            pt_tf = self.tf_buffer.lookup_transform(src_frame, target_frame, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

        pt = pt_tf.transform

        return pt_tf, np.array([pt.translation.x,
                                pt.translation.y,
                                pt.translation.z,
                                -pt.rotation.x,
                                -pt.rotation.y,
                                -pt.rotation.z,
                                -pt.rotation.w])

    def get_target_frame(self, obj_name):
        from waypoints import waypoints_dict
        if obj_name == 'blue_mapped' or obj_name == 'green_mapped':
            relative_pose = waypoints_dict['relative_plate']
        elif obj_name == 'toaster':
            relative_pose = waypoints_dict['toaster']
        elif obj_name == 'mustard':
            relative_pose = waypoints_dict['mustard']
        else:
            raise ValueError('relative frame not supported')

        obj_tf, obj_pose = self.get_tf_pose("world", obj_name)

        obj_M = tf.transformations.quaternion_matrix(obj_pose[3:])
        obj_M[0,3] = obj_pose[0]
        obj_M[1,3] = obj_pose[1]
        obj_M[2,3] = obj_pose[2]

        rel_M = tf.transformations.quaternion_matrix(relative_pose[3:])
        rel_M[0,3] = relative_pose[0]
        rel_M[1,3] = relative_pose[1]
        rel_M[2,3] = relative_pose[2]

        target_M = obj_M.dot(rel_M)

        target_quat = tf.transformations.quaternion_from_matrix(target_M)
        target_pos = np.array([target_M[0,3], target_M[1,3], target_M[2,3]])

        return np.concatenate([target_pos, target_quat])

        
    def waypoint_cooking(self):
        from waypoints import waypoints_dict

        # open gripper 
        self.driver_utils.set_finger_positions([0.1, 0.1, 0.1])
        
        # go to neutral
        pt = waypoints_dict['neutral']
        while not self.servo_to_pose_target(pt):
            pass

        # # blue plate 
        # pt = self.get_target_frame('blue_mapped')
        # while not self.servo_to_pose_target(pt):
        #     pass

        # ## close gripper
        # self.driver_utils.set_finger_positions([0.9, 0.9, 0.9])

        # ## go to neutral
        # pt = waypoints_dict['neutral']
        # while not self.servo_to_pose_target(pt):
        #     pass

        # # ## go to toaster waypoint
        # pt = waypoints_dict['toaster_waypoint']
        # while not self.servo_to_pose_target(pt):
        #     pass

        # # ## go to toaster
        # pt = waypoints_dict['toaster_absolute']
        # while not self.servo_to_pose_target(pt):
        #     pass

        # # ## open gripper
        # self.driver_utils.set_finger_positions([0.1, 0.1, 0.1])


        # ## go to switch pre
        # pt = waypoints_dict['switch_pre']
        # while not self.servo_to_pose_target(pt):
        #     pass

        
        # ## move to toast switch turn on pre-config
        # pt = waypoints_dict['switch_on_wp']
        # while not self.servo_to_pose_target(pt):
        #     pass

        # ## close first two figures
        # self.driver_utils.set_finger_positions([1., 1., 0.1])

            
        # ## turn on switch
        # pt = waypoints_dict['switch_on']
        # while not self.servo_to_pose_target(pt):
        #     pass

        # ## move to toast switch turn on pre-config
        # pt = waypoints_dict['switch_on_wp']
        # while not self.servo_to_pose_target(pt):
        #     pass

        # ## open gripper
        # self.driver_utils.set_finger_positions([0.0, 0.0, 0.0])


        # ## go to switch pre
        # pt = waypoints_dict['switch_pre']
        # while not self.servo_to_pose_target(pt):
        #     pass

        
        # ## go to toaster waypoint
        # pt = waypoints_dict['toaster_waypoint']
        # while not self.servo_to_pose_target(pt):
        #     pass

        # ## go to toaster
        # pt = waypoints_dict['toaster_absolute']
        # while not self.servo_to_pose_target(pt):
        #     pass


        # ## close gripper
        # self.driver_utils.set_finger_positions([0.9, 0.9, 0.9])


        # # ## go to toaster waypoint
        # pt = waypoints_dict['toaster_waypoint']
        # while not self.servo_to_pose_target(pt):
        #     pass


        # ## go to neutral
        # pt = waypoints_dict['neutral']
        # while not self.servo_to_pose_target(pt):
        #     pass

        
        # # blue plate 
        # pt = self.get_target_frame('blue_mapped')
        # while not self.servo_to_pose_target(pt):
        #     pass


        # # open gripper
        # self.driver_utils.set_finger_positions([0., 0., 0.])


        # ## go to neutral
        # pt = waypoints_dict['neutral']
        # while not self.servo_to_pose_target(pt):
        #     pass

        
        # # go to toaster waypoint
        # pt = waypoints_dict['toaster_waypoint']
        # while not self.servo_to_pose_target(pt):
        #     pass

        
        # # ## move to toast switch turn on pre-config
        # pt = waypoints_dict['switch_pre']
        # while not self.servo_to_pose_target(pt):
        #     pass
        
        # ## move to toast switch turn off waypoint
        # pt = waypoints_dict['switch_off_wp']
        # while not self.servo_to_pose_target(pt):
        #     pass

        # ## close first two figures
        # self.driver_utils.set_finger_positions([1., 1., 0.1])
            
        # ## turn off switch
        # pt = waypoints_dict['switch_off']
        # while not self.servo_to_pose_target(pt):
        #     pass

        # ## move to toast switch turn off pre-config
        # pt = waypoints_dict['switch_pre']
        # while not self.servo_to_pose_target(pt):
        #     pass

        # ## open gripper
        # self.driver_utils.set_finger_positions([0., 0., 0.])
      
        # ## go to toaster waypoint
        # pt = waypoints_dict['toaster_waypoint']
        # while not self.servo_to_pose_target(pt):
        #     pass

        
    def cleanup(self):
        pass


        
if __name__ == "__main__":
    #### CLF-CBF ####

    # from cooking_env.env.QP_waypoints.QPcontroller import QPcontroller

    # config = default_config
    # config['WPGenerator'] = {
    #     'type': QPcontroller,
    #     'config': {}
    # }

    #### DMP ####
    from cooking_env.env.dmp.dmp import DMP
    config = default_config
    config['WPGenerator'] = {
        'type': DMP,
        'config': {
            # gain on attractor term y dynamics
            'ay': 30,
            # gain on attractor term y dynamics
            'by': None,
            # timestep
            'dt': 0.1,
            # time scaling, increase tau to make the system execute faster
            'tau': 1.0,
            'use_canonical': False,
            'n_dmp': 3
        }
    }

    
    #### Initialize ####
    cls = RobotCooking(config)
    time.sleep(.5)
    
    #### test pose target following ####
    
    #### test IK ####
    # curr_ee_pos, curr_ee_quat = cls.driver_utils.get_ee_pose()
    # curr_jp = cls.driver_utils.get_joint_values()
    # ik_sol = cls.moveit_utils.ik(position=curr_ee_pos, orientation=curr_ee_quat)
    # # for kinova
    # ik_jp = ik_sol[:-3]
    # ik_fp = ik_sol[-3:]
    # print(curr_jp)
    # print(ik_jp)

    #### test servo to target pose ####
    cls.run()
    
    #### waypoint cooking ####
    # cls.waypoint_cooking()