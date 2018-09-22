"""
File: reconstruction.py
Author: Nrupatunga
Email: nrupatunga.tunga@gmail.com
Github: https://github.com/nrupatunga
Description: 3D reconstruction for scene using sequence of images

This class expects the data input directory 'input_dir' from SLAM to have this structure

        |-input_dir
          |- images
            |- *.png # rbg frames
            |- *.txt # keypoint text files for each rgb image
          |- mappoints.txt
          |- KeyFrameTrajectory.txt

"""

import glob
import os
# import config
# import cv2
import tensorflow as tf
from pointcloud import PointCloud3d


class SceneReconstruction(object):

    """Class which provides utilities to reconstruct 3d from set of
    images and information from ORB-slam """

    def __init__(self, sess, input_dir):
        """Initialize the input parameters

        Args:
            sess: tensorflow session
        """
        self.objPC = PointCloud3d(sess=session)
        self.objD = self.objPC.objD
        self.process_data(input_dir)

    def process_data(self, input_dir):
        """Process the data in the input directory and set the required
        parameters

        Args:
            input_dir: input data directory containing SLAM data

        """
        self.parent_dir = input_dir
        self.mappoint_file = os.path.join(input_dir, 'mappoints.txt')
        self.gt_file = os.path.join(input_dir, 'KeyFrameTrajectory.txt')
        self.rgb_folder = os.path.join(input_dir, 'images')

        self.img_files = sorted(glob.glob(os.path.join(self.rgb_folder, '*.png')))
        self.kp_files = sorted(glob.glob(os.path.join(self.rgb_folder, '*.txt')))


if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    parent_dir = '/media/nrupatunga/Data/2018/september/3D-reconstuction/SLAM_3DPoints_Trajectory/'
    objScene3D = SceneReconstruction(session, parent_dir)