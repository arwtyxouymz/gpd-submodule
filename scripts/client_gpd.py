#!/usr/bin/env python

import sys
import rospy
from hayate_planning_interface.srv import *

def handler():
    rospy.wait_for_service('/hayate/fine_localization', timeout=20)
    try:
        gpd = rospy.ServiceProxy('/hayate/fine_localization', FineLocalization)
        resp1 = gpd()
        print(resp1.object_pose)
        print(resp1.is_success)
        # return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
    handler()
