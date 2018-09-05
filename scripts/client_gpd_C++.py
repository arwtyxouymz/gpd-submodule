#!/usr/bin/env python

import sys
import rospy
from hayate_planning_interface.srv import *
from geometry_msgs.msg import Pose
from gpd.srv import *

def handler():
    rospy.wait_for_service('CPPgpd')
    try:
        gpd_cpp = rospy.ServiceProxy('CPPgpd', GPD_CPP)
        resp = gpd_cpp()
        return resp.object_pose
    except rospy.ServiceException, e:
        print("Service call failed: %s"%e)

if __name__ == "__main__":
    object_pose = handler()
    print(object_pose)
