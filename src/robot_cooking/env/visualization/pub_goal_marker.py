#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Pose, Point
from interactive_markers.menu_handler import *

class MarkerPublisher(object):
    def __init__(self, config={}):
        self.marker_array = MarkerArray()
        self.marker_array_pub =  rospy.Publisher('/robot_cooking/marker_array', MarkerArray, queue_size=10)
        self.int_marker_pub = rospy.Publisher('/robot_cooking/int_marker_pose', Pose, queue_size=10)

        # rospy.init_node('marker_publisher')

        self.int_marker_server = InteractiveMarkerServer("/simple_marker")
        self.menu_handler = MenuHandler()
        
        self.int_marker_pose = Pose()
        self.current_marker_num = 0
        

    # add regular marker to MarkerArray for publishing
    def add_marker(self,
                   name,
                   pose, # 7 tuple with x,y,z and quaternion
                   scale, # 3 tuple defining the bounding box
                   rgba, # 4 tuple: r,g,b values and a=transparency (0 being invisible)
                   marker_type # currently supports sphere and cube
               ):

        goal_marker = Marker()
        goal_marker.id = self.current_marker_num
        self.current_marker_num += 1

        goal_marker.text = name
        
        goal_marker.header.frame_id = "/world"
        if marker_type == "sphere":
            goal_marker.type = goal_marker.SPHERE
        elif marker_type == "cube":
            goal_marker.type = goal_marker.CUBE
        else:
            rospy.logerror("invalid marker type")

        goal_marker.action = goal_marker.ADD
        goal_marker.scale.x = scale[0]
        goal_marker.scale.y = scale[1]
        goal_marker.scale.z = scale[2]
        
        goal_marker.color.r = rgba[0]
        goal_marker.color.g = rgba[1]
        goal_marker.color.b = rgba[2]
        goal_marker.color.a = rgba[3]
        
        goal_marker.pose.orientation.x = pose[3]
        goal_marker.pose.orientation.y = pose[4]
        goal_marker.pose.orientation.z = pose[5]
        goal_marker.pose.orientation.w = pose[6]
        goal_marker.pose.position.x = pose[0]
        goal_marker.pose.position.y = pose[1]
        goal_marker.pose.position.z = pose[2]

        self.marker_array.markers.append(goal_marker)

    def update_marker_pose(self, name, pose):
        '''
        Update the pose of marker with the given name 
        '''
        for marker in self.marker_array.markers:
            if marker.text == name:
                marker.pose.position.x = pose[0]
                marker.pose.position.y = pose[1]
                marker.pose.position.z = pose[2]

                marker.pose.orientation.x = pose[3]
                marker.pose.orientation.y = pose[4]
                marker.pose.orientation.z = pose[5]
                marker.pose.orientation.w = pose[6]
      
                
    def publish_marker_array(self):
        self.int_marker_server.applyChanges()
        #rospy.spin()
        #while not rospy.is_shutdown():
        #print("publishing marker array!")
        self.int_marker_pub.publish(self.int_marker_pose)
        self.marker_array_pub.publish(self.marker_array)
           
    def process_int_marker_feedback(self, feedback):
        '''
        process feedback from interactive markers
        '''
        s = "Feedback from marker '" + feedback.marker_name
        s += "' / control '" + feedback.control_name + "'"

        mp = ""
        if feedback.mouse_point_valid:
            mp = " at " + str(feedback.mouse_point.x)
            mp += ", " + str(feedback.mouse_point.y)
            mp += ", " + str(feedback.mouse_point.z)
            mp += " in frame " + feedback.header.frame_id

        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            rospy.loginfo( s + ": button click" + mp + "." )
        elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            rospy.loginfo( s + ": menu item " + str(feedback.menu_entry_id) + " clicked" + mp + "." )
        elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            rospy.loginfo( s + ": pose changed")
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            rospy.loginfo( s + ": mouse down" + mp + "." )
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            rospy.loginfo( s + ": mouse up" + mp + "." )
        self.int_marker_server.applyChanges()

        self.int_marker_pose = feedback.pose
        
    def makeBox(self, msg ):
        marker = Marker()
        
        marker.type = Marker.CUBE
        marker.scale.x = msg.scale * 0.2
        marker.scale.y = msg.scale * 0.2
        marker.scale.z = msg.scale * 0.2
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0

        return marker

        
    def makeBoxControl(self, msg ):
        control =  InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append( self.makeBox(msg) )
        msg.controls.append( control )
        return control

        
    def make6DofMarker(self, fixed, interaction_mode, position, show_6dof = False):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "/world"
        int_marker.pose.position = position
        int_marker.scale = 0.1

        int_marker.name = "simple_6dof"
        int_marker.description = "Simple 6-DOF Control"

        # insert a box
        self.makeBoxControl(int_marker)
        int_marker.controls[0].interaction_mode = interaction_mode

        if fixed:
            int_marker.name += "_fixed"
            int_marker.description += "\n(fixed orientation)"

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = { 
                InteractiveMarkerControl.MOVE_3D : "MOVE_3D",
                InteractiveMarkerControl.ROTATE_3D : "ROTATE_3D",
                InteractiveMarkerControl.MOVE_ROTATE_3D : "MOVE_ROTATE_3D" }
            int_marker.name += "_" + control_modes_dict[interaction_mode]
            int_marker.description = "3D Control"
            if show_6dof: 
                int_marker.description += " + 6-DOF controls"
            int_marker.description += "\n" + control_modes_dict[interaction_mode]
    
        if show_6dof: 
            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            control.name = "rotate_x"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            control.name = "move_x"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            control.name = "rotate_z"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            control.name = "move_z"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.name = "rotate_y"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.name = "move_y"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            self.int_marker_server.insert(int_marker, self.process_int_marker_feedback)


        
if __name__=="__main__":


    marker_pub = MarkerPublisher()

    table_pose = [1,0,0,0,0,0,1]
    table_scale = [1,1.5,0.05]
    table_rgba = [0.1,0.1,0.1,1]
    table_marker_type = "cube"
    
    marker_pub.add_marker(
        "table",
        table_pose, # 7 tuple with x,y,z and quaternion
        table_scale, # 3 tuple defining the bounding box
        table_rgba, # 4 tuple: r,g,b values and a=transparency (0 being invisible)
        table_marker_type # currently supports sphere and cube
    )

    
    goal1_pose = [0.8,-0.4,0.05,0,0,0,1]
    goal1_scale = [0.5,0.5,0.1]
    goal1_rgba = [1,0,0,1]
    goal1_marker_type = "sphere"
    
    marker_pub.add_marker(
        "goal1",
        goal1_pose, # 7 tuple with x,y,z and quaternion
        goal1_scale, # 3 tuple defining the bounding box
        goal1_rgba, # 4 tuple: r,g,b values and a=transparency (0 being invisible)
        goal1_marker_type # currently supports sphere and cube
    )


    goal2_pose = [0.8,0.4,0.05,0,0,0,1]
    goal2_scale = [0.5,0.5,0.1]
    goal2_rgba = [0,1,0,1]
    goal2_marker_type = "sphere"
    
    marker_pub.add_marker(
        "goal2",
        goal2_pose, # 7 tuple with x,y,z and quaternion
        goal2_scale, # 3 tuple defining the bounding box
        goal2_rgba, # 4 tuple: r,g,b values and a=transparency (0 being invisible)
        goal2_marker_type # currently supports sphere and cube
    )


    event_region_pose = [0.8,1.0,0.3,0,0,0,1]
    event_region_scale = [0.5,0.5,0.5]
    event_region_rgba = [0,0,1,0.4]
    event_region_marker_type = "sphere"
    
    marker_pub.add_marker(
        "event_region",
        event_region_pose, # 7 tuple with x,y,z and quaternion
        event_region_scale, # 3 tuple defining the bounding box
        event_region_rgba, # 4 tuple: r,g,b values and a=transparency (0 being invisible)
        event_region_marker_type # currently supports sphere and cube
    )
   

    #marker_pub.create_interactive_marker("int_marker")
    position = Point( 0, 0, 0)
    marker_pub.make6DofMarker( False, InteractiveMarkerControl.MOVE_ROTATE_3D, position, True )

    marker_pub.publish_marker_array()