import numpy as np


class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        pass

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        def rotation_matrix(angle):
            return np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
        
        # convert [x y theta] to 3x3 transform matrix
        def transform_matrix(pose):
            X = np.eye(3)
            X[:2, :2] = rotation_matrix(pose[2])
            X[:2, -1] = np.array(pose[:2])
            return X
        

        if odometry[0] == odometry[1] == 0:
            return particles
        
        # add noise
        # odometry_random_noise = np.random.normal(0, 0.02, odometry.shape)
        # odometry += odometry_random_noise
        
        # 3N x 3 Matrix (every particles row converted to transform matrix)
        all_transforms = np.apply_along_axis(transform_matrix, axis=1, arr=particles)
        all_transforms = np.vstack(all_transforms)
        
        # apply transform from the odometry
        transform_delta = transform_matrix(odometry)
        all_transforms = all_transforms @ transform_delta
        
        # convert back to original form (N x 3 matrix)
        poses = np.zeros(particles.shape)

        # each row is the first column of the rotation matrices for each particle
        rotations = all_transforms[:, 0].reshape(particles.shape[0], 3) 
        poses[:, 2] = np.arctan2(rotations[:, 1], rotations[:, 0])

        # each row is the last column of the transformation matrix for each particle
        positions = all_transforms[:, 2].reshape((particles.shape[0], 3))
        poses[:, :2] = positions[:, :2]
        
        return poses

        ####################################
