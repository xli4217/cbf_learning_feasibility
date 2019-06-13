import cv2
import numpy as np
import time
import rospy
import roslib
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('image_publisher')
cap = cv2.VideoCapture(0) #Select 1 for external USB cam if computer has a built-in cam
cap.set(3,640) #width of frame in video stream
cap.set(4,480) #height of frame in video stream
image_pub = rospy.Publisher("camera/rgb/image_raw",Image,queue_size=10)
bridge = CvBridge()

while(True):
    ret, frame = cap.read()
    if ret:
        try:
          image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
        except CvBridgeError as e:
          print(e)
