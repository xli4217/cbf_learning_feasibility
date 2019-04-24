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

def quaternion_log(q):
    u = q[:3]
    v = q[3]

    if np.linalg.norm(u) == 0:
        return np.array([0,0,0])
    else:
        if v > 0.999:
            v = 0.999
        return np.arccos(v) * (u / np.linalg.norm(u))


def quaternion_dist(q1, q2):
    conjugate_product = tf.transformations.quaternion_multiply(q1, tf.transformations.quaternion_conjugate(q2))

    if all(conjugate_product == np.array([0,0,0,-1])):
        return 2*np.pi
    else:
        return 2 * np.linalg.norm(quaternion_log(conjugate_product))


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
                'upper': 4 * np.array([1.5,1.5,1.5,1.5,5.,5.,5.]),
                'lower': -4 * np.array([1.5,1.5,1.5,1.5,5.,5.,5.]),    
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

        self.last_quat_distance = None
        
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
        
    def servo_to_pose_target(self, pt, pos_th=0.01, quat_th=0.1):
        assert len(pt) == 7

        # update tf
        self.update_pose_target_tf(pt)
        
        # calculate the joint positions using ik
        ik_sol = self.moveit_utils.ik(position=pt[:-4], orientation=pt[-4:])
        ik_jp = ik_sol[:-3]

        damping = 1.5
        natural_freq = 2.0
        kp = (2 * np.pi * natural_freq) ** 2
        kd = 2 * damping * 2 * np.pi * natural_freq

        joint_angle_diff = ik_jp - self.driver_utils.get_joint_values()
        joint_vel = self.driver_utils.get_joint_velocities()

        jv = kp * joint_angle_diff + kd * joint_vel

        ee_pos, ee_quat = self.driver_utils.get_ee_pose()

        pose_distance = np.linalg.norm(ee_pos - pt[:3])
        quat_dist_arg = 2 * np.inner(ee_quat, pt[3:]) - 1
        quat_dist_arg = np.modf(quat_dist_arg)[0]

        if quat_dist_arg > 0.99:
            quat_distance = 0.
        elif quat_dist_arg < -0.99:
            quat_distance = 0
        else:
            quat_distance = np.arccos(quat_dist_arg)

        if not self.driver_utils.is_tool_in_safe_workspace():
            print('tool not in safe workspace')
            return True

        # start servoing
        if (pose_distance > pos_th or quat_distance > quat_th) and self.driver_utils.is_tool_in_safe_workspace():
            self.driver_utils.pub_joint_velocity(jv, duration_sec=1./self.RobotCooking_config['rate'])
            self.rate.sleep()
            return False
        else:
            return("servo reached goal")
            return True

    def plan_to_pose_target(self, pt):
        self.wp_gen.set_goal(pt)
        self.update_goal_tf(pt)
        ee_pos, ee_quat = self.driver_utils.get_ee_pose()
        ee_linear_vel, ee_angular_vel = self.driver_utils.get_ee_velocity()

        curr_pos = np.concatenate([ee_pos, ee_quat])
        curr_vel = np.concatenate([ee_linear_vel, ee_angular_vel])

        pose_distance = np.linalg.norm(ee_pos - pt[:3])

        quat_dist_arg = 2 * np.inner(ee_quat, pt[3:]) - 1
        quat_dist_arg = np.modf(quat_dist_arg)[0]
        
        if quat_dist_arg > 0.99:
            quat_distance = 0.
        elif quat_dist_arg < -0.99:
            quat_distance = 0
        else:
            quat_distance = np.arccos(quat_dist_arg)

        quat_distance = np.arccos(quat_dist_arg)
            
        
        if not self.driver_utils.is_tool_in_safe_workspace():
            print('tool not in safe workspace')
            return True

        if (pose_distance > 0.01 or quat_distance > 0.15) and self.driver_utils.is_tool_in_safe_workspace():
        
            # dt = 0.003 * np.exp(1./(pose_distance + quat_distance))
            # dt = np.clip(dt, 0.015, 0.03)
            
            action = np.array([0,0,0,0,0,0])
            ddy, dy, y = self.wp_gen.get_next_wp(action, curr_pos, curr_vel, dt=None)
            self.servo_to_pose_target(y, pos_th=0.005, quat_th=0.1)
            return False
        else:
            print("plan reached goal")
            return True
            
    def run(self):
        from waypoints import waypoints_dict
        ## go to neutral
        goal = waypoints_dict['neutral']
        while not self.plan_to_pose_target(goal):
            pass
        
        
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

    def get_target_frame(self, obj_name, rel_pose_name):
        from waypoints import waypoints_dict

        relative_pose = waypoints_dict[rel_pose_name]
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

        
    def waypoint_cooking(self, mode="servo"):
        from waypoints import waypoints_dict

        servo_script = [
            'open',
            'neutral',
            #'blue_mapped',
            'close',
            'neutral',
            'toaster_waypoint',
            'toaster_absolute',
            'open',
            'toaster_waypoint',
            'switch_pre',
            'close',
            'switch_on',
            'switch_pre',
            'switch_post',
            'open',
            'toaster_waypoint',
            'toaster_absolute',
            'close',
            'toaster_waypoint',
            # 'blue_mapped',
            'open',
            'toaster_waypoint',
            'switch_pre',
            'close',
            'switch_off',
            'switch_pre',
            'switch_post',
            'neutral'
        ]

        plan_script = [
            'open',
            'neutral',
            #### pick raw dog ####
            # 'blue_mapped',
            # 'close',
            # 'toaster_waypoint',
            # 'toaster_absolute',
            # 'open',
            # 'toaster_waypoint',
            #### close switch ####
            # 'switch_pre',
            # 'close',
            # 'switch_on',
            # 'switch_pre',
            # 'switch_post',
            #### pick cooked dog ####
            # 'open',
            # 'toaster_waypoint',
            # 'toaster_absolute',
            # 'close',
            # 'toaster_waypoint',
            # 'neutral',
            #### place cooked dog ####
            # 'blue_mapped',
            # 'open',
            # 'neutral',
            #### apply condiment ####
            'relative_condiment_pre',
            'relative_condiment_post',
            'close',
            'relative_plate_apply_condiment_pre',
            'flip_condiment',
            'apply_condiment',
            'flip_condiment_back',
            'place_condiment',
            'open',
            'relative_condiment_pre',
            'post_place_condiment',
            #### serve ####
            #### close switch ####
            # 'switch_pre',
            # 'close',
            # 'switch_off',
            # 'switch_pre',
            # 'switch_post',
            # 'neutral'
        ]

        if mode == "servo":
            script = servo_script
            action_fn = self.servo_to_pose_target
        elif mode == 'plan':
            script = plan_script
            action_fn = self.plan_to_pose_target
        else:
            raise ValueError("mode not supported")
        
        for pt_name in script:
            if pt_name == 'open':
                # open gripper 
                self.driver_utils.set_finger_positions([0.3, 0.3, 0.3])
            elif pt_name == 'close':
                # close gripper 
                self.driver_utils.set_finger_positions([0.9, 0.9, 0.9])
            elif pt_name == 'relative_condiment_pre' or pt_name == 'relative_condiment_post':
                done = False
                while not done:
                    pt = self.get_target_frame('condiment_mapped', pt_name)
                    done = action_fn(pt)
            elif pt_name == 'relative_plate_apply_condiment_pre':
                done = False
                while not done:
                    pt = self.get_target_frame('blue_mapped', pt_name)
                    done = action_fn(pt)
            elif pt_name == 'apply_condiment':
                for i in range(70):
                    vel_scale = 2. * np.sin(0.08*i)
                    cls.driver_utils.pub_ee_frame_velocity(direction='z',vel_scale=vel_scale, duration_sec=0.1)
                    time.sleep(0.1)
            elif pt_name == 'flip_condiment':
                self.driver_utils.pub_joint_velocity([0,0,0,0,0,0,20], duration_sec=10)
            elif pt_name == 'flip_condiment_back':
                self.driver_utils.pub_joint_velocity([0,0,0,0,0,0,-20], duration_sec=10)
            elif pt_name == 'blue_mapped':
                done = False
                while not done:
                    pt = self.get_target_frame("blue_mapped", "relative_plate")
                    done = action_fn(pt)
            else:
                pt = waypoints_dict[pt_name]
                while not action_fn(pt):
                    pass
        
                
        
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
    from traj_generators.dmp.dmp import DMP
    config = default_config
    config['rate'] = 30
    config['WPGenerator'] = {
        'type': DMP,
        'config': {
            # gain on attractor term y dynamics (linear)
            'ay': 55,
            # gain on attractor term y dynamics (linear)
            'by': None,
            # gain on attractor term y dynamics (angular)
            'az': 55,
            # gain on attractor term y dynamics (angular)
            'bz': None,
            # timestep
            'dt': 0.001,
            # time scaling, increase tau to make the system execute faster
            'tau': 1.,
            'use_canonical': False,
            # for canonical
            'apx': 1.,
            'gamma': 0.3,
            # for faster convergence
            'app': 0.5,
            'apr': 0.5,
            # for integrating goal
            'ag': 3.0,
            'ago': 3.0,
            # if True, then update according to dmp_pose, else update according to current pose
            'use_dmp_pose': True,
            'n_linear_dmp': 3,
            'n_angular_dmp': 3
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

    #### test plan to target pose ####
    # cls.run()
    
    #### waypoint cooking ####
    cls.waypoint_cooking(mode='plan')

