#!/bin/bash



# Load the Bash configuration
source ~/.bashrc

# Run commands in parallel
bash ~/ros2_ws/src/webots_ros2_suv/scripts/run_nav2.sh &
bash ~/ros2_ws/src/webots_ros2_suv/scripts/run_ros2.sh &
bash ~/ros2_ws/src/webots_ros2_suv/scripts/run_rviz2.sh &

# Wait for all background processes to finish
wait
