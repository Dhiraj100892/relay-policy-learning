#!/usr/bin/env python3
from IPython import embed

import adept_envs
import gym


def try_cv2_import():
    """
    In order to import cv2 in python3 we need to remove
    the python2.7 path from sys.path. To use the Habitat-PyRobot integration the user
    needs to export environment variable ROS_PATH which will look something like:
    /opt/ros/kinetic/lib/python2.7/dist-packages
    """
    import sys
    import os

    ros_path_kinetic = "/opt/ros/kinetic/lib/python2.7/dist-packages"
    ros_path_melodic = "/opt/ros/melodic/lib/python2.7/dist-packages"

    if ros_path_kinetic in sys.path:
        sys.path.remove(ros_path_kinetic)
        import cv2

        sys.path.append(ros_path_kinetic)
    elif ros_path_melodic in sys.path:
        sys.path.remove(ros_path_melodic)
        import cv2

        sys.path.append(ros_path_melodic)
    else:
        import cv2

    return cv2
cv2 = try_cv2_import()


env = gym.make('kitchen_relax-v1')

env.reset()
embed()
for i in range(1000):
    out = env.step(env.action_space.sample())
    img = out[3]["images"]
    cv2.imshow("test", cv2.resize(img, (int(img.shape[1]/4),
                                        int(img.shape[0]/4))))
    cv2.waitKey(0)
    print("step number = {}".format(i))

embed()