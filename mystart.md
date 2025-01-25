## docker preparation
#### go to project folder
```bash
cd ~/Desktop/robocross.virtual
```
#### run the docker
```bash
bash start.sh
```
if error occurs try to close docker
```bash
docker stop ulstu-devel && docker rm ulstu-devel
```
#### open VSCode in browser (if needed)
```bash
xdg-open http://localhost:31415
```
#### get access to dockers terminal
```bash
docker exec -it ulstu-devel bash
```
## start project
```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
ros2 launch webots_ros2_suv robot_launch.py
```
note: commands must be inputed in docker terminal