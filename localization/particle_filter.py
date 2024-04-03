from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel
from sensor_msgs.msg import LaserScan

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy

import tf_transformations

import numpy as np

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.particles = None  # TODO: initialization procedure for particles

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self.particle_positions = np.array([])
        self.laser_ranges = np.array([])
        self.particle_samples_indices = np.array([])

    def publish_avg_pose(self):
        particle_samples = self.particle_positions[self.particle_samples_indices, :] if len(self.particle_samples_indices) != 0 else self.particle_positions
        positions = particle_samples[:, :2]
        angles = particle_samples[:, 2]

        average_position = np.mean(positions, axis=0)
        average_angle = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
        
        average_pose = np.hstack((average_position, average_angle))

        msg = Odometry()
        
        msg.pose.pose.position.x = average_pose[0]
        msg.pose.pose.position.y = average_pose[1]

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 1.0
        msg.pose.pose.orientation.w = average_angle

        msg.child_frame_id = '/base_link'
        self.odom_pub.publish(msg)

    def odom_callback(self, msg):
        if len(self.particle_positions) == 0: return

        x = msg.twist.twist.linear.x
        y = msg.twist.twist.linear.y
        angle = msg.twist.twist.angular.z

        odometry = np.array([x, y, angle])

        # print(self.particle_positions)
        self.motion_model.evaluate(self.particle_positions, odometry)

        self.publish_avg_pose()

    def laser_callback(self, msg):
        if len(self.particle_positions) == 0: 
            self.get_logger().info(f"self.particle_positions is empty.")
            return
        laser_ranges = np.random.choice(np.array(msg.ranges), 100)
        weights = self.sensor_model.evaluate(self.particle_positions, laser_ranges)
        if weights is None: 
            self.get_logger().info(f"self.sensor_model.evaluate() returned empty weights.")
            return
        
        self.get_logger().info(f"np.shape(weights): {np.shape(weights)}")
        M = len(weights)
        self.particle_samples_indices = np.random.choice(M, size=M, p=weights)

        self.publish_avg_pose()

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        angle = msg.pose.pose.orientation.w

        pose = np.array([[x, y , angle]])
        self.particle_positions = np.vstack((self.particle_positions, pose)) if len(self.particle_positions) != 0 else pose


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
