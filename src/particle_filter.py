#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

class ParticleFilter:

    def __init__(self):

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Implement the MCL algorithm
        # using the sensor model and the motion model

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
