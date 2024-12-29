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
from geometry_msgs.msg  import PointStamped, TransformStamped, Quaternion
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


SENSOR_DEPTH = 40

class NodeEgoController(Node):
    """Drive controller

    Args:
        Node (_type_): _description_
    """
    def __init__(self):
        try:
            super().__init__('node_ego_controller')
            self._logger.info(f'Node Ego Started')
            qos = qos_profile_sensor_data
            qos.reliability = QoSReliabilityPolicy.RELIABLE

            self.t2 = time.time()
            self.min_distance = 5
            self.steering_angle=0.0
            self.speed=0.0
            self.turn_sensitivity = 80.0
            self.speed_senitivity = 20.0
            self.__world_model = WorldModel()
            self.__ws = None
            

            package_dir = get_package_share_directory("webots_ros2_suv")

            self.create_subscription(Odometry, '/odom', self.__on_odom_message, qos)
            self.create_subscription(sensor_msgs.msg.Image, '/vehicle/camera/image_color', self.__on_image_message, qos)
            self.create_subscription(sensor_msgs.msg.PointCloud2, "/lidar", self.__on_lidar_message, qos)

            self.obstacle_publisher = self.create_publisher(sensor_msgs.msg.PointCloud2, 'detected_obstacles', qos)
            self.closest_publisher = self.create_publisher(sensor_msgs.msg.PointCloud2, 'closest_obstacle', qos)

            self.__ackermann_publisher = self.create_publisher(AckermannDrive, 'cmd_ackermann', 1)
            
            self.start_web_server()

        except  Exception as err:
            self._logger.error(''.join(traceback.TracebackException.from_exception(err).format()))

    def start_web_server(self):
        self.__ws = MapWebServer(log=self._logger.info)
        threading.Thread(target=start_web_server, args=[self.__ws]).start()

    def __on_lidar_message(self, data):
        """Lidar callback function

        Args:
            data (sensor_msgs.msg.PointCloud2)
        """
        pc_data = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
        points = list(pc_data)  # List of (x, y, z) tuples
        self.process_point_cloud(np.array(points))

    def process_point_cloud(self, points):
        """Main function to handle lidar

        Args:
            points (np.array): array of points
        """
        # Implement the logic to process the point cloud data
        self._logger.info(f'Processing {len(points)} points')

        # Example: simple obstacle detection (basic check for obstacles in front)
        obstacles = self.detect_obstacles(points)

        self.publish_obstacles(obstacles)

        #obstacles = self.filter_obstacles(points)
        if obstacles.size>0:
            closest_obstacle, distance  = self.detect_closest_obstacle(obstacles)
            self._logger.info(f"{len(obstacles)} obstacles detected!")
            #if np.median(obstacles['x'])
            '''
            mean_y = np.mean(obstacles[:,1])
            min_x = np.amin(obstacles[:,0])
            min_z = np.amin(obstacles[:,2])
            self.steering_angle = (1/mean_y) * self.turn_sensitivity
            self.speed =  min_x * self.speed_senitivity

            self.speed = np.clip(self.speed, 0, 40)
            delta = 0.05
            threshold = -0.00
            self.steering_angle = np.clip(self.steering_angle, self.buf_steer-delta, self.buf_steer+delta)+threshold
            self.steering_angle = np.clip(self.steering_angle, -0.25, 0.25)*np.pi
            '''
            '''
            if closest_obstacle[1]<0:
                self.steering_angle = -(1/closest_obstacle[1])
            else:
                self.steering_angle = 30
''' 
            self.steering_angle = (1/closest_obstacle[1])*self.turn_sensitivity
            self.steering_angle = np.clip(self.steering_angle, -30, 30)
            if ((distance  >  15) and (closest_obstacle[0]>10)):
                self.speed = self.speed_senitivity
            else:
                self.speed = distance * 2

            self._logger.info(f'Closest object in [{closest_obstacle[0]},{closest_obstacle[1]}, {closest_obstacle[2]}]')
            self._logger.info(f'Distance = {distance}')
            self.publish_closest(obstacles[(obstacles[:,0]==closest_obstacle[0])&
            (obstacles[:,1]==closest_obstacle[1])&
            (obstacles[:,2]==closest_obstacle[2])])
            #self._logger.info(f'Min z is {min_z}')
            
                    #self._logger.info(f'Detected obstacle in [{obstacle[0]},{obstacle[1]},{obstacle[2]}]')   
            # You can trigger path planning or stopping logic here
        else:
            self.steering_angle = 0.0
            self.speed = self.speed_senitivity * 2.0
        #self.buf_steer = self.steering_angle/np.pi
        self.drive()

    def detect_obstacles(self, points, min_distance=0, max_distance=30.0, min_height=-2.2, max_height=0, min_y=-5.0, max_y=5.0):
        """_summary_

        Args:
            points (np.array): array of points
            min_distance (int, optional): _description_. Defaults to 0.
            max_distance (float, optional): _description_. Defaults to 30.0.
            min_height (float, optional): _description_. Defaults to -2.2.
            max_height (int, optional): _description_. Defaults to 0.
            min_y (float, optional): _description_. Defaults to -5.0.
            max_y (float, optional): _description_. Defaults to 5.0.
        """
            # Check for points that are closer than the threshold
            # Extract 'x', 'y', and 'z' fields into a regular NumPy array
        xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)
        
        # Calculate the Euclidean distance for each point
        x = xyz[:,0]
        y = xyz[:,1]
        z= xyz[:,2]
        
        # Filter points based on the distance threshold
        obstacles = xyz[(xyz[:,0] >= min_distance) & (x <= max_distance) &  # Distance filtering
            (y>= min_y) & (y <= max_y) &  # Height filtering
            (z>= min_height) & (z <= max_height)  # Y-coordinate filtering
        ]
        return obstacles

    def detect_closest_obstacle(self, obstacles):
        """Detecting closest point from obstacles array

        Args:
            obstacles (np.array): _description_

        Returns:
            np.array: closest point
        """
    # Compute distance to each obstacle
        distances = np.sqrt(obstacles[:, 0]**2 + obstacles[:, 1]**2)
        closest_index = np.argmin(distances)
        return obstacles[closest_index], distances[closest_index]

    def publish_obstacles(self, obstacles):
        """Method for publishing found obstacles

        Args:
            obstacles (np.array): _description_
        """
        if len(obstacles) == 0:
            self._logger.info("No obstacles to publish.")
            return

        # Create a PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"  # Set the frame of reference (e.g., the robot's base)

        # Convert structured array to list of tuples
        points = [(obstacle[0], obstacle[1], obstacle[2]) for obstacle in obstacles]

        # Create the PointCloud2 message
        pointcloud_msg = pc2.create_cloud_xyz32(header, points)

        # Publish the message
        self.obstacle_publisher.publish(pointcloud_msg)
        self._logger.info(f"Published {len(points)} obstacles.")
    def publish_closest(self, obstacles):
        """Method for publishing ONE closest point

        Args:
            obstacles (np.array): _description_
        """
        if len(obstacles) == 0:
            self._logger.info("No obstacles to publish.")
            return

        # Create a PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"  # Set the frame of reference (e.g., the robot's base)

        # Convert structured array to list of tuples
        points = [(obstacle[0], obstacle[1], obstacle[2]) for obstacle in obstacles]

        # Create the PointCloud2 message
        pointcloud_msg = pc2.create_cloud_xyz32(header, points)

        # Publish the message
        self.closest_publisher.publish(pointcloud_msg)
        self._logger.info(f"Published {len(points)} obstacles.")


    def __on_range_image_message(self, data):
        """RangeFinder callback function

        Args:
            data (): _description_
        """
        image = np.frombuffer(data.data, dtype="float32").reshape((data.height, data.width, 1))
        image[image == np.inf] = SENSOR_DEPTH
        image[image == -np.inf] = 0
        
        range_image = image / SENSOR_DEPTH

    def drive(self):
        """Method for drive. Getting speed, steering angle(degrees) from self properties
        """
        self.__world_model.command_message.speed = float(self.speed)
        self.__world_model.command_message.steering_angle = float(self.steering_angle/180*np.pi)
        self._logger.info(f'Driving to {self.speed} speed, {self.steering_angle} angle')
        self.__ackermann_publisher.publish(self.__world_model.command_message)

    #@timeit
    def __on_image_message(self, data):
        """Camera callback function

        Args:
            data (sensor_msgs.msg.Image): image from camera
        """
        image = data.data
        image = np.frombuffer(image, dtype=np.uint8).reshape((data.height, data.width, 4))
        analyze_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))

        self.__world_model.rgb_image = np.asarray(analyze_image)

        t1 = time.time()
        # TODO: put your code here
        t2 = self.t2
                
        
        delta = t2 - t1
        self.t2 = t1
        fps = 1 / delta if delta > 0 else 100
        self._logger.info(f"Current image FPS: {fps}")

        pos = self.__world_model.get_current_position()

        #self.drive()

        if self.__ws is not None:
            self.__ws.update_model(self.__world_model)

    def __on_odom_message(self, data):
        """Odometry callback

        Args:
            data (nav_msgs.msg.Odometry): odometry data
        """
        roll, pitch, yaw = euler_from_quaternion(data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        lat, lon, orientation = self.__world_model.coords_transformer.get_global_coords(data.pose.pose.position.x, data.pose.pose.position.y, yaw)
        self.__world_model.update_car_pos(lat, lon, orientation)
        if self.__ws is not None:
            self.__ws.update_model(self.__world_model)

def main(args=None):
    """Main controller spin ROS2 function

    Args:
        args (_type_, optional): _description_. Defaults to None.
    """
    try:
        rclpy.init(args=args)
        path_controller = NodeEgoController()
        rclpy.spin(path_controller)
        rclpy.shutdown()
    except KeyboardInterrupt:
        print('server stopped cleanly')
    except  Exception as err:
        print(''.join(traceback.TracebackException.from_exception(err).format()))
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
