from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseArray

from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy

import numpy as np

assert rclpy

import time

import tf2_ros
import tf_transformations
import geometry_msgs
from geometry_msgs.msg import TransformStamped

import threading


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.get_logger().info('symlink check')

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

        self.get_logger().info("topics")
        self.get_logger().info(scan_topic)
        self.get_logger().info(odom_topic)

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

        self.particles_pub = self.create_publisher(PoseArray, "/particles", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY 2+=============")

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

        self.previous_time = time.perf_counter()
        # self.previous_time = rclpy.time.Time().nanoseconds * 10e-9

        self.declare_parameter('num_particles', "default")
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value

        self.br = tf2_ros.TransformBroadcaster(self)

        self.num_beams_per_particle = self.get_parameter("num_beams_per_particle").get_parameter_value().integer_value

        self.prev_speed = 0.0

        self.lock = threading.Lock()

        
        self.scan_counter = 0


    def publish_avg_pose(self):
        self.publish_particle_poses()
        particle_samples = self.particle_positions

        positions = particle_samples[:, :2]
        angles = particle_samples[:, 2]

        average_position = np.mean(positions, axis=0)
        average_angle = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
        
        average_pose = np.hstack((average_position, average_angle))

        msg = Odometry()

        msg.header.frame_id = '/map'
        
        msg.pose.pose.position.x = average_pose[0]
        msg.pose.pose.position.y = average_pose[1]

        # rotation is around z-axis
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = np.sin(average_angle / 2)
        msg.pose.pose.orientation.w = np.cos(average_angle / 2)

        msg.child_frame_id = self.particle_filter_frame
        # self.get_logger().info("%s" % average_pose)
        # self.get_logger().info("%s" % average_angle)
        self.odom_pub.publish(msg)

        
        # Publish Transform

        obj = geometry_msgs.msg.TransformStamped()

        # current time
        # obj.header.stamp = time.to_msg()

        # frame names
        obj.header.frame_id = '/map'
        obj.child_frame_id = self.particle_filter_frame

        # translation component
        obj.transform.translation.x = average_pose[0]
        obj.transform.translation.y = average_pose[1]
        obj.transform.translation.z = 0.0

        # rotation (quaternion)
        obj.transform.rotation.x = 0.0
        obj.transform.rotation.y = 0.0
        obj.transform.rotation.z = np.sin(average_angle / 2)
        obj.transform.rotation.w = np.cos(average_angle / 2)

        self.br.sendTransform(obj)

    def odom_callback(self, msg): # 50 hz
        # self.get_logger().info("odom callback")
        if len(self.particle_positions) == 0: return
        # manually time dt
        current_time = time.perf_counter()  # maybe rclpy.time.Time()?
        dt = current_time-self.previous_time
        # current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 10e-9
        # dt = current_time - self.previous_time

        self.previous_time = current_time


        # relative to robot frame
        x_vel = msg.twist.twist.linear.x
        y_vel = msg.twist.twist.linear.y
        angle_vel = msg.twist.twist.angular.z

        self.prev_speed = np.sqrt(x_vel**2 + y_vel ** 2)

        angle_fix = 0.08
        if x_vel < 0: angle_vel -= angle_fix
        else: angle_vel += angle_fix
        
        odometry = - dt * np.array([x_vel, y_vel, angle_vel])

        # print(self.particle_positions)
        # self.particle_positions = self.motion_model.evaluate(self.particle_positions, odometry)
        with self.lock:
            self.particle_positions = self.motion_model.evaluate(self.particle_positions, odometry)

            self.publish_avg_pose()


    def laser_callback(self, msg): # 90 hz
        # self.get_logger().info("laser callback")   
        if len(self.particle_positions) == 0: return
        with self.lock:
            if self.prev_speed == 0.0: return

            self.scan_counter += 1
            if self.scan_counter % 10 != 0: return

            downsampled_indices = np.linspace(0, len(msg.ranges)-1, self.num_beams_per_particle).astype(int)
            downsampled_laser_ranges = np.array(msg.ranges)[downsampled_indices]
            # downsampled_laser_ranges = np.random.choice(np.array(msg.ranges), self.num_beams_per_particle)
            weights = self.sensor_model.evaluate(self.particle_positions, downsampled_laser_ranges)
            if weights is None:
                return

            if np.sum(weights) == 0: return

            weights /= np.sum(weights) # normalize so they add to 1

            # number of particles to keep
            keep = int(self.num_particles * 0.95)

            # prevent error
            if np.count_nonzero(weights) < keep: return

            # sample with replacement
            particle_samples_indices = np.random.choice(self.num_particles, size=self.num_particles, p=weights, replace=True)


            self.particle_positions = self.particle_positions[particle_samples_indices,:] 
            self.particle_positions[:, 0] += np.random.normal(0.05, self.prev_speed*0.05, self.num_particles)
            self.particle_positions[:, 1] += np.random.normal(0, self.prev_speed*0.05, self.num_particles)
            self.particle_positions[:, 2] += np.random.normal(0, np.pi/30, self.num_particles)

            self.publish_avg_pose()



    def pose_callback(self, msg):
        self.get_logger().info("initial pose")
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        angle = 2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        if self.num_particles == 1:
            self.particle_positions = np.array([[x, y, angle]])
            return

        self.get_logger().info(f"initial pose set: {x}, {y}, {angle}")

        # Initialize particles with normal distribution around the user-specified pose
        def normalize_angle(angle):
            # Normalize the angle to be within the range [-π, π]
            normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
            return normalized_angle
        
        x = np.random.normal(loc=x, scale=0.5, size=(self.num_particles,1))
        y = np.random.normal(loc=y, scale=0.5, size=(self.num_particles,1))
        theta = np.random.normal(loc=angle, scale=np.pi/4, size=(self.num_particles,1))

        # Normalize angles
        theta = np.apply_along_axis(normalize_angle, axis=0, arr=theta)

        self.particle_positions = np.hstack((x, y, theta))
        # self.get_logger().info("self.particle_positions: %s" % self.particle_positions)
            
        self.publish_particle_poses()

    def publish_particle_poses(self):
        # return
        poses = []
        for i in range(len(self.particle_positions)):
            particle = self.particle_positions[i, :]
            msg = Pose()
            
            msg.position.x = particle[0]
            msg.position.y = particle[1]

            # rotation is around z-axis
            angle = particle[2]
            msg.orientation.x = 0.0
            msg.orientation.y = 0.0
            msg.orientation.z = np.sin(angle / 2)
            msg.orientation.w = np.cos(angle / 2)

            poses.append(msg)

        msg = PoseArray()
        msg.header.frame_id = '/map'

        msg.poses = poses

        self.particles_pub.publish(msg)


        


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
