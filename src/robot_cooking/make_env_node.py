#!/usr/bin/env python
import rospy
from robot_cooking.env.visualization.pub_goal_marker import MarkerPublisher
import numpy as np
import json
import argparse
import tf2_ros
import tf
from  geometry_msgs.msg import *
from future.utils import viewitems
from visualization_msgs.msg import *
from robot_cooking.env.calibration.robot_calibration import T_motive2jaco, T_motive2baxter


class MakeEnv(object):
    '''
    makes the environment (send transforms) according to config given by json_config_path (create pubs and subs as necessary), and publish rviz markers if specified    
    '''
    
    def __init__(self, json_config_path, optitrack=False, robot_name='jaco'):

        with open(json_config_path) as f:
            self.json_config = json.loads(f.read())

        self.marker_pub = MarkerPublisher()

        #### motive to robot ####
        if optitrack:
            if robot_name == 'jaco':
                self.T_motive2robot = T_motive2jaco
            elif robot_name == 'baxter':
                self.T_motive2robot = T_motive2baxter
            else:
                raise ValueError('robot not supported')
        
        # add rviz markers
        for key, value in viewitems(self.json_config):
            if 'marker_type' in value.keys():
                if value['marker_type'] == "interactive":
                    self.marker_pub.make6DofMarker(False, InteractiveMarkerControl.MOVE_ROTATE_3D, Point(value['init_pose'][0], value['init_pose'][1], value['init_pose'][2]), True)
                else:
                    if key != "fitted_elliptical_obstacles":
            
                        if optitrack and value['require_motive2robot_transform'] == "True":
                            # this is Z up motive configuration
                            scale = [value['scale'][1], value['scale'][0], value['scale'][2]]
                        else:
                            scale = [value['scale'][0], value['scale'][1], value['scale'][2]]
                        self.marker_pub.add_marker(
                            key,
                            value['init_pose'],
                            scale,
                            value['rgba'],
                            value['marker_type'])
                        
                    if key == "fitted_elliptical_obstacles":
                        for obs_k, obs_v in viewitems(value['fitted_obstacles']):
                            if optitrack and value['require_motive2robot_transform'] == "True":
                                # this is Z up motive configuration
                                scale = [obs_v['scale'][1], obs_v['scale'][0], obs_v['scale'][2]]
                            else:
                                scale = [obs_v['scale'][0], obs_v['scale'][1], obs_v['scale'][2]]
                            self.marker_pub.add_marker(
                                obs_k,
                                obs_v['init_pose'],
                                scale,
                                obs_v['rgba'],
                                value['marker_type']
                            )
               

        self.marker_pub.publish_marker_array()

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.frames = []
        self.object_pose_dict = {}

        obj_names = []
        topic_names = []
        motive_offsets = []
        self.marker_offsets = {}
        require_motive2robot_transform = []
        parent_frame_id = []
        child_frame_id = []
        for object_name, value in viewitems(self.json_config):
            if object_name != "fitted_elliptical_obstacles":
                if 'vrpn_topic' in value.keys():
                    obj_names.append(object_name)
                    topic_names.append(value['vrpn_topic'])
                    motive_offsets.append(value['motive_offset'])
                    self.marker_offsets[object_name] = value['marker_offset']
                    require_motive2robot_transform.append(value['require_motive2robot_transform'])
                    parent_frame_id.append(value['parent_frame_id'])
                    child_frame_id.append(value['child_frame_id'])
            #### for fitted_elliptical_obstacles ####
            if object_name == "fitted_elliptical_obstacles":
                for k, v in viewitems(value['fitted_obstacles']):
                    obj_names.append(k)
                    topic_names.append(v['vrpn_topic'])
                    motive_offsets.append(v['motive_offset'])
                    self.marker_offsets[k] = v['marker_offset']
                    require_motive2robot_transform.append(value['require_motive2robot_transform'])
                    parent_frame_id.append(v['parent_frame_id'])
                    child_frame_id.append(v['child_frame_id'])
                    
        self.object_pose_dict = {}
        
        if optitrack: # use optitrack to get environment data
            [rospy.Subscriber(topic_name, geometry_msgs.msg.PoseStamped, self.optitrack_callback, (obj_name, motive_offset, require_transform, pfid, cfid), queue_size=1) for topic_name, obj_name, motive_offset, require_transform, pfid, cfid in zip(topic_names, obj_names, motive_offsets, require_motive2robot_transform, parent_frame_id, child_frame_id)]
            while not rospy.is_shutdown():
                pass
        else:
            while not rospy.is_shutdown():
              for obj_name, value in viewitems(self.json_config):
                if 'marker_type' in value.keys():
                    if value['marker_type'] == "interactive":
                        pose = [self.marker_pub.int_marker_pose.position.x,
                                self.marker_pub.int_marker_pose.position.y,
                                self.marker_pub.int_marker_pose.position.z,
                                0,0,0,1]
                    else:
                        pose = value['init_pose']
                    self.pub_transforms(obj_name, value['parent_frame_id'], value['child_frame_id'], pose)
            
    def optitrack_callback(self, msg, arg):
        obj_name, offset, require_transform, parent_frame_id, child_frame_id = arg
        if require_transform == "True":
            quaternion =  [msg.pose.orientation.x,
                           msg.pose.orientation.y,
                           msg.pose.orientation.z,
                           msg.pose.orientation.w]
            M = tf.transformations.quaternion_matrix(quaternion)
            
            M[0,3] = msg.pose.position.x
            M[1,3] = msg.pose.position.y
            M[2,3] = msg.pose.position.z

            transformed_M = np.dot(self.T_motive2robot, M)
            transformed_q = tf.transformations.quaternion_from_matrix(transformed_M)
            pose = [
                transformed_M[0,3],
                transformed_M[1,3],
                transformed_M[2,3],
                transformed_q[0],
                transformed_q[1],
                transformed_q[2],
                transformed_q[3],
            ]

            pose[:3] = np.array(pose)[:3] + np.array(offset)[:3]
            
        else:
            pose = [msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w]
            
        self.pub_transforms(obj_name, parent_frame_id, child_frame_id, pose)
        
        
            
    def update_rviz_markers(self):
        for key, value in viewitems(self.object_pose_dict):
            pose = np.array(value)
            pose[:3] = pose[:3] + np.array(self.marker_offsets[key])[:3]
            self.marker_pub.update_marker_pose(name=key, pose=pose)
        self.marker_pub.publish_marker_array()

    def pub_transforms(self, obj_name, parent_frame_id, child_frame_id, pose):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent_frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = pose[0]
        t.transform.translation.y = pose[1]
        t.transform.translation.z = pose[2]
        # q = tf_conversions.transformations.quaternion_from_euler(0, 0, msg.theta)
        t.transform.rotation.x = pose[3]
        t.transform.rotation.y = pose[4]
        t.transform.rotation.z = pose[5]
        t.transform.rotation.w = pose[6]
        self.tf_broadcaster.sendTransform(t)
        self.object_pose_dict[obj_name] = pose
        self.update_rviz_markers()

        
if __name__ == "__main__":
    import sys
    import os
    
    optitrack = str(sys.argv[1])
    if optitrack == "false":
        optitrack = False
    else:
        optitrack = True

    robot_name = str(sys.argv[2])

    rospy.init_node(robot_name + '_make_env', anonymous=True)

    env_json_path = os.path.join(os.environ['RC_PATH'],
                                 'src', 'robot_cooking', 'env',
                                 'config', robot_name+'_env_config.json')
    
    env = MakeEnv(env_json_path, optitrack, robot_name)