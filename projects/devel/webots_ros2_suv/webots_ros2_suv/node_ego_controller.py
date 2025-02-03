import rclpy
import numpy as np
import traceback
import cv2
import os
import math
import time
import yaml
import threading
import matplotlib.pyplot as plt
import sensor_msgs.msg
from std_msgs.msg import Header
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from std_msgs.msg import Float32, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, TransformStamped, Quaternion, Twist
from tf2_ros import TransformBroadcaster
from ackermann_msgs.msg import AckermannDrive
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.utils import controller_url_prefix
from PIL import Image
from .lib.timeit import timeit
from .lib.orientation import euler_from_quaternion
from .lib.map_server import start_web_server, MapWebServer
from .lib.world_model import WorldModel
from robot_interfaces.srv import PoseService
from robot_interfaces.msg import EgoPose
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Int32


SENSOR_DEPTH = 40
SPEED_SENS = 1
TURN_SENS = 1
TURN_MAX = 30

class NodeEgoController(Node):
    def __init__(self):
        try:
            super().__init__("node_ego_controller")
            self._logger.info(f"Node Ego Started")
            qos = qos_profile_sensor_data
            qos.reliability = QoSReliabilityPolicy.RELIABLE 

            self.__world_model = WorldModel()
            self.__ws = None

            package_dir = get_package_share_directory("webots_ros2_suv")

            self.create_subscription(Odometry, "/odom", self.__on_odom_message, qos)
            self.create_subscription(
                sensor_msgs.msg.Image,
                "/vehicle/camera/image_color",
                self.__on_image_message,
                qos,
            )
            self.create_subscription(
                sensor_msgs.msg.PointCloud2, "/lidar", self.__on_lidar_message, qos
            )

            self.create_subscription(Twist, '/cmd_vel', self.__on_cmd_message, qos)

            self.obstacle_publisher = self.create_publisher(
                sensor_msgs.msg.PointCloud2, "detected_obstacles", qos
            )
            self.closest_publisher = self.create_publisher(
                sensor_msgs.msg.PointCloud2, "closest_obstacle", qos
            )

            self.__ackermann_publisher = self.create_publisher(
                AckermannDrive, "cmd_ackermann", 1
            )
            self.__trafficlight_publisher = self.create_publisher(Int32, 'traffic_light', 1)
            self.start_web_server()


            self.traffic_light_state = 0  
            self.flag = False
            # Создаем подписчика на топик /traffic_light
            self.traffic_light_subscriber = self.create_subscription(
                Int32,  # Тип сообщения
                '/traffic_light',  # Имя топика
                self.__on_traffic_light_message,  # Функция-обработчик
                10  # Очередь сообщений (QoS)
            )

        except Exception as err:
            self._logger.error(
                "".join(traceback.TracebackException.from_exception(err).format())
            )

    def start_web_server(self):
        self.__ws = MapWebServer(log=self._logger.info)
        threading.Thread(target=start_web_server, args=[self.__ws]).start()

    def __on_lidar_message(self, data):
        pc_data = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
        points = list(pc_data)
        self.process_point_cloud(np.array(points))

    def process_point_cloud(self, points):
        #self._logger.info(f"Processing {len(points)} points")

        obstacles = self.detect_obstacles(points)
        self.publish_obstacles(obstacles)
        if obstacles.size > 0:#if any obstacles found
            closest_obstacle, distance = self.detect_closest_obstacle(obstacles)

            #self._logger.info(f"{len(obstacles)} obstacles detected!")
            #setting self.drive() parameters 
            #self.steering_angle = (1 / (closest_obstacle[0]**0.25*closest_obstacle[1])) * TURN_SENS
            #self.steering_angle = np.clip(self.steering_angle, -TURN_MAX, TURN_MAX)

            # if (distance > 15) and (closest_obstacle[0] > 10):
            #     self.speed = SPEED_SENS
            # else:
            #     self.speed = distance * 2

            #self._logger.info(
            #    f"Closest object in [{closest_obstacle[0]},{closest_obstacle[1]}, {closest_obstacle[2]}]"
            #)
            #self._logger.info(f"Distance = {distance}")
            self.publish_closest(
                obstacles[
                    (obstacles[:, 0] == closest_obstacle[0])
                    & (obstacles[:, 1] == closest_obstacle[1])
                    & (obstacles[:, 2] == closest_obstacle[2])
                ]
            )
        else:
            pass
            #self.steering_angle = 0.0
            #self.speed = SPEED_SENS * 2.0
        #self.drive()
        pass

    def detect_obstacles(
        self,
        points,
        min_distance=0,
        max_distance=30.0,
        min_height=-2.2,
        max_height=0,
        min_y=-5.0,
        max_y=5.0):
        xyz = np.stack([points["x"], points["y"], points["z"]], axis=-1)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        obstacles = xyz[
            (xyz[:, 0] >= min_distance)
            & (x <= max_distance)
            & (y >= min_y)
            & (y <= max_y)
            & (z >= min_height)
            & (z <= max_height)
        ]
        return obstacles

    def detect_closest_obstacle(self, obstacles):
        distances = np.sqrt(obstacles[:, 0] ** 2 + obstacles[:, 1] ** 2)
        closest_index = np.argmin(distances)
        return obstacles[closest_index], distances[closest_index]

    def publish_obstacles(self, obstacles):#method for publishing found obstacles PointCloud2(cropbox of /lidar topic)
        if len(obstacles) == 0:
            #self._logger.info("No obstacles to publish.")
            return
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = (
            "base_link"
        )
        points = [(obstacle[0], obstacle[1], obstacle[2]) for obstacle in obstacles]
        pointcloud_msg = pc2.create_cloud_xyz32(header, points)
        self.obstacle_publisher.publish(pointcloud_msg)
        #self._logger.info(f"Published {len(points)} obstacles.")

    def publish_closest(self, obstacles):#method for publishing PointCloud2 with ONE point(closest to sensor)(cropbox of detected_obstacles topic)
        if len(obstacles) == 0:
            #self._logger.info("No obstacles to publish.")
            return

        # Create a PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = (
            "base_link"  # Set the frame of reference (e.g., the robot's base)
        )

        # Convert structured array to list of tuples
        points = [(obstacle[0], obstacle[1], obstacle[2]) for obstacle in obstacles]

        # Create the PointCloud2 message
        pointcloud_msg = pc2.create_cloud_xyz32(header, points)

        # Publish the message
        self.closest_publisher.publish(pointcloud_msg)
        #self._logger.info(f"Published {len(points)} obstacles.")

    def __on_cmd_message(self, data):
        linear_velocity = data.linear.x
        angular_velocity = -1.25*data.angular.z
        
        # Compute steering angle based on Ackermann kinematics
        if abs(angular_velocity) == 0:  # Handle straight motion
            steering_angle = 0.0
        else:
            if linear_velocity!=0:
                wheelbase = 4.3#rear to front axles
                steering_angle = math.atan(wheelbase * angular_velocity / linear_velocity)
            else:
                steering_angle = 0.0
        
        # Limit steering angle within allowed bounds
        self.steering_angle = max(-np.pi/4, min(np.pi/4, steering_angle))
        self.speed = SPEED_SENS * linear_velocity

        #self.steering_angle=data.angular.z

        self.drive()

    def __on_range_image_message(self, data):
        image = np.frombuffer(data.data, dtype="float32").reshape(
            (data.height, data.width, 1)
        )
        image[image == np.inf] = SENSOR_DEPTH
        image[image == -np.inf] = 0

        range_image = image / SENSOR_DEPTH

       
    def __on_traffic_light_message(self, msg):
        """Функция обработки сообщений из /traffic_light."""
        self.traffic_light_state = msg.data  # Сохраняем текущее значение светофора
        #self._logger.info(f"Получено значение светофора: {self.traffic_light_state}")


    def drive(self):
        
        if self.traffic_light_state == 1:
            self.flag = True
        if self.flag:    
            self.__world_model.command_message.speed = float(self.speed)
            self.__world_model.command_message.steering_angle = float(
                self.steering_angle 
            )
            #self._logger.info(f"Driving to {self.speed} speed, {self.steering_angle} angle")
            self.__ackermann_publisher.publish(self.__world_model.command_message)

    # @timeit
    def __on_image_message(self, data: sensor_msgs.msg.Image):
        image = data.data
        image = np.frombuffer(image, dtype=np.uint8).reshape((data.height, data.width, 4))
        analyze_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))

        self.__world_model.rgb_image = np.asarray(analyze_image)

        t1 = time.time()
        # TODO: put your code here
        #---------------------------------------------- ANTITIMOFEY ----------------------------------------
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (99,23,0), (121,67,36))
        #cv2.imshow("mask", mask1)
        cv2.waitKey(1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        contours,hierarchy = cv2.findContours(mask1, 1, 2)
        #print("Number of contours detected:", len(contours))

        msg = Int32()
        msg.data = 0
        if contours is not None:
            msg.data = -1

        img = image
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if not (600 < w*h < 900):
                continue
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            #cv2.putText(img, f'hight is {h}', (x-100, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            #cv2.putText(img, f'width is {w}', (x-100, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            #cv2.putText(img, f'area is {w*h}', (x-100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            crop_hsv = hsv[y:y+h, x:x+w]
            crop_mask = cv2.inRange(crop_hsv, (67,24,0), (80,245,255))
            #cv2.imshow("crop_mask", crop_mask)

            green_cnt = 0
            for width in range(w):
                for hight in range(h):
                    if crop_mask[hight][width] == 255:
                        green_cnt += 1
            #cv2.putText(img, f'green_cnt {green_cnt}', (x-100, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if green_cnt > 30:
                cv2.putText(img, f'GREEN LIGHT!', (x-100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                msg.data = 1
        self.__trafficlight_publisher.publish(msg)


        #cv2.imshow("Camera", img)
        cv2.waitKey(0)
        #---------------------------------------------- ANTITIMOFEY ----------------------------------------
        t2 = time.time()

        delta = t2 - t1
        fps = 1 / delta if delta > 0 else 100
        #self._logger.info(f"Current FPS: {fps}")

        pos = self.__world_model.get_current_position()

        #self.drive()

        if self.__ws is not None:
            self.__ws.update_model(self.__world_model)

    def __on_odom_message(self, data):
        roll, pitch, yaw = euler_from_quaternion(
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w,
        )
        lat, lon, orientation = self.__world_model.coords_transformer.get_global_coords(
            data.pose.pose.position.x, data.pose.pose.position.y, yaw
        )
        self.__world_model.update_car_pos(lat, lon, orientation)
        if self.__ws is not None:
            self.__ws.update_model(self.__world_model)


def main(args=None):
    try:
        rclpy.init(args=args)
        path_controller = NodeEgoController()
        rclpy.spin(path_controller)
        rclpy.shutdown()
    except KeyboardInterrupt:
        print("server stopped cleanly")
    except Exception as err:
        print("".join(traceback.TracebackException.from_exception(err).format()))
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
