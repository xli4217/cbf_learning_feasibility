import rospy
from sensor_msgs.msg import Image

class TopicMapper(object):

    def __init__(self):
        rospy.init_node('image_topic_mapper', anonymous=True)
        rospy.Subscriber('/camera/color/image_raw', Image, self.cb)

        self.pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=10)
        
    def cb(self, msg):
        self.pub.publish(msg)
        
if __name__ == "__main__":
    cls = TopicMapper()
    while not rospy.is_shutdown():
        pass
