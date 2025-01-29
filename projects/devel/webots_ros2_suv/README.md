# webots_ros2_suv
## [node_ego_controller.py](https://github.com/ssskiv/robocross.virtual/blob/navigation/projects/devel/webots_ros2_suv/webots_ros2_suv/node_ego_controller.py):  
### def __init_:  
    содержит подписки на топики и издателей этих топиков 
### def __on_cmd_message:  
    обработка Twist в необходимый вид для движения вид и отправка в self.drive   
### def drive:  
    отправка обработанного в __on_cmd_message twist в __ackermann_publisher  
### def __on_image_message:
	callback топика `/vehicle/camera/image_color`  
## [nav2_params.yaml](https://github.com/ssskiv/robocross.virtual/blob/navigation/projects/devel/webots_ros2_suv/config/nav2_params.yaml)  
файл, в котором содержатся настройки `nav2 stack`, т.е. `controller_server, local_costmap, global_costmap, planner_server`  
