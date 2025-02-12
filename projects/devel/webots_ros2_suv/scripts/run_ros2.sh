cd ~/ros2_ws
colcon build
source ~/ros2_ws/install/setup.bash
ros2 launch webots_ros2_suv robot_launch.py params_file:=src/webots_ros2_suv/config/nav2_params.yaml