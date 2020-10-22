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

