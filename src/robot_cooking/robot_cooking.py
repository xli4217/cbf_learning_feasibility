import sys
import rospy
import numpy as np
import copy
import os
import time
import json
from future.utils import viewitems
import cloudpickle

from kinova_api.kinova_driver_utils import KinovaDriverUtils
from kinova_api.kinova_moveit_utils import KinovaMoveitUtils

import tf2_ros
import tf
from visualization_msgs.msg import *
from  geometry_msgs.msg import *

from utils.utils import *
        
default_config = {
    "rate": 10,
    "policy_info": {
        "state_space": None,
        "action_space": None,
        "training_config_restore_path": None,
        "policy_restore_path": None,
        "state_preprocessor_restore_path": None
    },
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
                'upper': 4 * np.array([2.5,2.5,2.5,2.5,5.,5.,5.]),
                'lower': -4 * np.array([2.5,2.5,2.5,2.5,5.,5.,5.]),    
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

        # self.M_finger2ee_offset = np.eye(4)
        # self.M_finger2ee_offset[0,3] = -0.05
        # self.M_finger2ee_offset[2,3] = -0.1

        # self.M_ee2finger_offset = np.linalg.inv(self.M_finger2ee_offset)
        
        #### Initialize tf ####
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.last_quat_distance = None

        #### load json config ####
        with open(os.path.join(os.environ['RC_PATH'], 'src', 'robot_cooking', 'env', 'config', 'env_config.json')) as f:
            self.config_json = json.loads(f.read())

        #### all info ####
        self.all_info = None

        self.goal = None
        self.target = None
        self.obs_info = None
        
        #### load ####
        self.policy, self.state_preprocessor = load_policy_and_preprocessor(slef.RobotCooking_config['policy_info'])


    def finger2ee_pose_hack(self, finger_pose=None):
        assert isinstance(finger_pose, np.ndarray)
        assert len(finger_pose) == 7

        M_finger_pose = tf.transformations.quaternion_matrix(finger_pose[3:])
        M_finger_pose[:3,3] = finger_pose[:3]

    
        M_ee = M_finger_pose.dot(self.M_finger2ee_offset)

        ## transform to vector
        ee_quat = tf.transformations.quaternion_from_matrix(M_ee)

        ee_pose = np.concatenate([M_ee[:3,3], ee_quat])

        return ee_pose

        
    def ee2finger_pose(self, ee_pose=None):
        
        assert isinstance(ee_pose, np.ndarray)
        assert len(ee_pose) == 7

        M_ee_pose = tf.transformations.quaternion_matrix(ee_pose[3:])
        M_ee_pose[:3,3] = ee_pose[:3]

        ee2finger_pose_tf_stamped = self.tf_buffer.lookup_transform('j2s7s300_end_effector', 'j2s7s300_link_finger_tip_2', rospy.Time())
        ee2finger_pose = [
            ee2finger_pose_tf_stamped.transform.translation.x,
            ee2finger_pose_tf_stamped.transform.translation.y,
            ee2finger_pose_tf_stamped.transform.translation.z,
            ee2finger_pose_tf_stamped.transform.rotation.x,
            ee2finger_pose_tf_stamped.transform.rotation.y,
            ee2finger_pose_tf_stamped.transform.rotation.z,
            ee2finger_pose_tf_stamped.transform.rotation.w,
        ]

        M_ee2finger = tf.transformations.quaternion_matrix(ee2finger_pose[3:])
        M_ee2finger[:3,3] = ee2finger_pose[:3]

        ## this is the finger_tip link pose in tf
        M_finger_tip_link = M_ee_pose.dot(M_ee2finger)

        ## this is the desired finger pose (more towards the tip)
        M_tiplink2finger_offset = np.eye(4)
        M_tiplink2finger_offset[0,3] = 0.1

        M_finger = M_finger_tip_link.dot(M_tiplink2finger_offset)

        ## transform to vector
        finger_quat = tf.transformations.quaternion_from_matrix(M_finger)

        finger_pose = np.concatenate([M_finger[:3,3], finger_quat])

        return finger_pose
        
    def get_policy_output(self):
        s = self.all_info['target_pose'][:3]
        if self.state_preprocessor is not None:
            s = self.state_preprocessor.get_scaled_x(s)
            
        if self.policy is not None:
            action = self.policy.get_action(s, deterministic=True)
            action *= 100
            action = np.clip(action,
                             self.RobotCooking_config['policy_info']['action_space']['lower_bound'],
                             self.RobotCooking_config['policy_info']['action_space']['upper_bound'])
        
        else:
            action = np.zeros(self.RobotCooking_config['policy_info']['action_space']['shape'][0])

        action = np.concatenate([action.flatten(), np.zeros(3)])

        # print(action)
        # action = np.zeros(6)
        
        return action
        

    def update_all_info(self):
        if self.target is None:
            target_pos, target_quat = self.driver_utils.get_ee_pose()
            self.target = np.concatenate([target_pos, target_quat])

        self.all_info = {
            'obs_info': self.obs_info,
            'target_pose': self.target,
            'goal_pose': self.goal
        }
        
    def update_goal_tf(self, pt):
        self.goal = pt
        self.update_all_info()
        
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
        self.target = pt
        self.update_all_info()
        
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
        
    def servo_to_pose_target(self, pt, pos_th=0.01, quat_th=0.1, dry_run=True):
        assert len(pt) == 7

        # update tf
        self.update_pose_target_tf(pt)
        
        # calculate the joint positions using ik
        ik_sol = self.moveit_utils.ik(position=pt[:-4], orientation=pt[-4:])
        ik_jp = ik_sol[:-3]

        damping = 1.5
        natural_freq = 2.1

        kp = (2 * np.pi * natural_freq) ** 2
        kd = 2 * damping * 2 * np.pi * natural_freq

        joint_angle_diff = ik_jp - self.driver_utils.get_joint_values()
        joint_vel = self.driver_utils.get_joint_velocities()

        jv = kp * joint_angle_diff + kd * joint_vel
        
        ee_pos, ee_quat = self.driver_utils.get_ee_pose()

        ee_pose = np.concatenate([ee_pos, ee_quat])
        pos_distance, quat_distance = pose_distance(ee_pose, pt)
            
        if not self.driver_utils.is_tool_in_safe_workspace():
            print('tool not in safe workspace')
            return True

        # start servoing
        if (pos_distance > pos_th or quat_distance > quat_th) and self.driver_utils.is_tool_in_safe_workspace():
            if not dry_run:
                self.driver_utils.pub_joint_velocity(jv, duration_sec=1./self.RobotCooking_config['rate'])
            self.rate.sleep()
            return False
        else:
            return("servo reached goal")
            return True

    
    def plan_to_pose_target(self, pt, control_point='ee', dry_run=True):
        '''
        control_point can be 'ee' or 'finger_tip'
        '''
        
        self.wp_gen.set_goal(pt)
        self.update_goal_tf(pt)

        self.get_obstacle_info()
        
        ee_pos, ee_quat = self.driver_utils.get_ee_pose()
        ee_linear_vel, ee_angular_vel = self.driver_utils.get_ee_velocity()

        curr_pose = np.concatenate([ee_pos, ee_quat])
        curr_vel = np.concatenate([ee_linear_vel, ee_angular_vel])

        if control_point == 'finger_tip':
            curr_pose[0] -= 0.19
            curr_pose[2] -= 0.1

        pos_distance, quat_distance = pose_distance(curr_pose, pt)
            
        if not self.driver_utils.is_tool_in_safe_workspace():
            print('tool not in safe workspace')
            return True

        if (pos_distance > 0.01 or quat_distance > 0.15) and self.driver_utils.is_tool_in_safe_workspace():
        
            action = self.get_policy_output()
            ddy, dy, y = self.wp_gen.get_next_wp(action, curr_pose, curr_vel, obs_info=self.get_obstacle_info())
            
            if control_point == 'finger_tip':
                # y = get_p2(y, self.M_finger2ee_offset)
                y[0] += 0.19
                y[2] += 0.1
                
            if not dry_run:
                self.servo_to_pose_target(y, pos_th=0.005, quat_th=0.1, dry_run=dry_run)  
            else:
                self.update_pose_target_tf(y)
                self.rate.sleep()
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

        
    def waypoint_cooking(self, mode="servo", dry_run=True):
        from waypoints import waypoints_dict

        servo_script = []

        plan_script = [
            'open',
            'neutral',
            #### pick raw dog ####
            'blue_mapped',
            'close',
            'toaster_waypoint',
            'toaster_absolute',
            'open',
            'toaster_waypoint',
            #### close switch ####
            'switch_pre',
            'close',
            'switch_on',
            'switch_pre',
            'switch_post',
            #### pick cooked dog ####
            'open',
            'toaster_waypoint',
            'toaster_absolute',
            'close',
            'toaster_waypoint',
            'neutral',
            #### place cooked dog ####
            'blue_mapped',
            'open',
            'neutral',
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
            'neutral',
            'blue_mapped',
            'close',
            'neutral',
            'green_mapped',
            'open',
            #### close switch ####
            'neutral',
            'switch_pre',
            'close',
            'switch_off',
            'switch_pre',
            'switch_post',
            'neutral',
            'open'
        ]

        test_script = [
            # 'toaster_absolute'
            'close',
            'switch_on'
        ]
        
        if mode == "servo":
            script = servo_script
            action_fn = self.servo_to_pose_target
        elif mode == 'plan':
            script = plan_script
            action_fn = self.plan_to_pose_target
        elif mode == 'test':
            script = test_script
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
                    done = action_fn(pt, dry_run=dry_run)
            elif pt_name == 'relative_plate_apply_condiment_pre':
                done = False
                while not done:
                    pt = self.get_target_frame('blue_mapped', pt_name)
                    done = action_fn(pt, dry_run=dry_run)
            elif pt_name == 'apply_condiment':
                for i in range(25):
                    vel_scale = 2. * np.sin(0.3*i)
                    cls.driver_utils.pub_ee_frame_velocity(direction='z',vel_scale=vel_scale, duration_sec=0.1)
                    time.sleep(0.1)
            elif pt_name == 'flip_condiment':
                self.driver_utils.pub_joint_velocity([0,0,0,0,0,0,20], duration_sec=9)
            elif pt_name == 'flip_condiment_back':
                self.driver_utils.pub_joint_velocity([0,0,0,0,0,0,-20], duration_sec=9)
            elif pt_name == 'blue_mapped' or pt_name == 'green_mapped':
                done = False
                while not done:
                    pt = self.get_target_frame(pt_name, "relative_plate")
                    done = action_fn(pt, dry_run=dry_run)
            elif pt_name == 'switch_on':
                pt = waypoints_dict[pt_name]
                while not action_fn(pt, dry_run=dry_run, control_point='finger_tip'):
                    pass
            else:
                pt = waypoints_dict[pt_name]
                while not action_fn(pt, dry_run=dry_run):
                    pass
        
                
        
    def cleanup(self):
        pass

    ##############################
    # Common interface functions #
    ##############################
    def get_target_pose(self):
        pos, quat = self.driver_utils.get_ee_pose()
        return pos, quat

    def get_target_velocity(self):
        lv, av = self.driver_utils.get_ee_velocity()
        return lv, av

    def set_gripper_state(self, gs):
        self.driver_utils.set_finger_positions(gs * np.ones(3))

    def get_switch_state(self):
        pass

    def move_to(self, pt):
        pass

    def set_goal_pose(self, pt):
        self.wp_gen.set_goal(pt)
        self.update_goal_tf(pt)

        
    def get_obstacle_info(self):
        fitted_obstacles = self.config_json['fitted_elliptical_obstacles']['fitted_obstacles']
        obs_info = []
        for k, v in viewitems(fitted_obstacles):
            obs_pose_tf_stamped = self.tf_buffer.lookup_transform(v['parent_frame_id'][1:], v['child_frame_id'][1:], rospy.Time())
            obs_pos = [
                obs_pose_tf_stamped.transform.translation.x,
                obs_pose_tf_stamped.transform.translation.y,
                obs_pose_tf_stamped.transform.translation.z
            ]
            obs_info.append({ 'name': k, 'position': obs_pos, 'radius':v['scale'][0]/2})


        table_pose_tf_stamped  = self.tf_buffer.lookup_transform('world', 'table_mapped', rospy.Time())
        table_pos = [
            table_pose_tf_stamped.transform.translation.x,
            table_pose_tf_stamped.transform.translation.y,
            table_pose_tf_stamped.transform.translation.z
        ]
          
        table_info = {'name': 'table', 'position': table_pos}
        obs_info.append(table_info)

        self.obs_info = obs_info

        self.update_all_info()
        
        return self.obs_info


    def get_object_pose(self):
        fitted_obstacles = self.config_json['fitted_elliptical_obstacles']['fitted_obstacles']
        object_poses = []
        for k, v in viewitems(self.config_json):
            obs_pose_tf_stamped = self.tf_buffer.lookup_transform(v['parent_frame_id'][1:], v['child_frame_id'][1:], rospy.Time())
            obs_pos = [
                obs_pose_tf_stamped.transform.translation.x,
                obs_pose_tf_stamped.transform.translation.y,
                obs_pose_tf_stamped.transform.translation.z
            ]
            obs_info.append({ 'name': k, 'position': obs_pos, 'radius':v['scale'][0]/2})


        table_pose_tf_stamped  = self.tf_buffer.lookup_transform('world', 'table_mapped', rospy.Time())
        table_pos = [
            table_pose_tf_stamped.transform.translation.x,
            table_pose_tf_stamped.transform.translation.y,
            table_pose_tf_stamped.transform.translation.z
        ]
          
        table_info = {'name': 'table', 'position': table_pos}
        obs_info.append(table_info)

        self.object_poses = object_poses

        self.update_all_info()
        
        return self.object_poses

        
        
if __name__ == "__main__":
    #### hybrid traj generator ####
    from traj_generators.trajectory_generator import TrajectoryGenerator
    config = default_config
    config['rate'] = 30
    config['WPGenerator'] = {
        'type': TrajectoryGenerator,
        'config': {
            'dmp_config': {
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
                
            },
            'clf_cbf_config': {
                'k_cbf': 1.1,
                'epsilon':0.8,
                'num_states':3,
                'action_space': {'shape': (3,), 'upper_bound': [0.1, 0.1, 0.1], 'lower_bound': [-0.1,-0.1,-0.1]},
                'use_own_pose': True,
                'dt': 0.015
            },
            #'translation_gen': 'clf_cbf',
            'translation_gen': 'dmp',
            'orientation_gen': 'dmp'            
        }
    }


    #### Learned policy restoration ####
    experiment_root_dir = os.path.join(os.environ['LEARNING_PATH'], 'learning', 'experiments')
    experiment_name = 'test'
    hyperparam_dir = 'seed0'
    itr = 100

    config['policy_info'] = {
        "state_space": {'type': 'float', 'shape': (3, ), "upper_bound": [], 'lower_bound': []},
        "action_space": {'type': 'float', 'shape': (3, ), "upper_bound": 70*np.ones(3), 'lower_bound': -70*np.ones(3)},
        "training_config_restore_path": os.path.join(experiment_root_dir, experiment_name, 'config', hyperparam_dir, 'config.pkl'),
        "policy_restore_path": os.path.join(experiment_root_dir, experiment_name, 'transitions', hyperparam_dir, 'itr_'+str(itr)),
        "state_preprocessor_restore_path": os.path.join(experiment_root_dir, experiment_name, 'info', hyperparam_dir, 'state_preprocessor_params.pkl')
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
    # cls.waypoint_cooking(mode='plan')
    # cls.waypoint_cooking(mode='test', dry_run=False)

    