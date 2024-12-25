import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/ulstu/ros2_ws/src/install/webots_ros2_suv'
