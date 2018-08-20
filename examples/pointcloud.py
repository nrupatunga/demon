"""
File: pointcloud.py
Author: Nrupatunga
Email: nrupatunga.tunga@gmail.com
Github: https://github.com/nrupatunga
Description: Utility for pointcloud generation and processing
"""

import tensorflow as tf
from demon import DemonNet
from PIL import Image as Im
import numpy as np
import cv2


class PointCloud(object):

    """Utility class for point cloud generation and processing"""

    def __init__(self, sess=None):
        """initialization

        Args:
            sess: tensorflow session
        """

        if sess is None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

        self.sess = sess
        self.objD = DemonNet(sess)
        self.pointclouds = {}
        self.count = 0

    def generate_point_cloud(self, img1, img2, label=None):
        """point cloud generation

        Args:
            img1: image 1
            img2: image 2
            label: label to store the point cloud
            match: do the 3d matching, currently used sift correspondences
        """
        _, out_256_192 = self.objD.run(img1, img2)

        depth = np.squeeze(out_256_192['depth'])
        image = np.squeeze(out_256_192['image'])

        # store width and height
        self.H = out_256_192['depth'].shape[0]
        self.W = out_256_192['depth'].shape[1]

        # point cloud
        xyz, color = self.objD.get_point_cloud(depth, image=image)

        # store point cloud into single structure
        pointcloud = {}
        pointcloud['points'] = xyz
        pointcloud['colors'] = color

        if label is None:
            self.count = self.count + 1
            self.pointclouds[str(self.count)] = pointcloud
        else:
            self.pointclouds[label] = pointcloud

    def find_correspondence(self, pc1, pc2):
        """find the matching 3d points between two point clouds

        Args:
            pc1: point cloud 1
            pc2: point cloud 2

        """
        img1 = pc1['colors'].reshape((self.H, self.W, 3))
        img2 = pc2['colors'].reshape((self.H, self.W, 3))
        gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # sift correspondences
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_1, None)
        kp2, des2 = sift.detectAndCompute(gray_2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m_, n_ in matches:
            if m_.distance < 0.75 * n_.distance:
                good.append([m_])

        pc1_match = []
        pc2_match = []
        mx = 0
        for i, dmatch in enumerate(good):
            x1, y1 = kp1[dmatch[0].queryIdx].pt
            mx = max(mx, x1)
            x2, y2 = kp2[dmatch[0].trainIdx].pt

            pc1_match.append(pc1['points'][round(y1) * self.W + round(x1)].tolist())
            pc2_match.append(pc2['points'][round(y2) * self.W + round(x2)].tolist())

        return pc1_match, pc2_match

    def find_transform(self, pc1, pc2):
        """Find the R and t between two point clouds

        Args:
            pc1: point cloud 1
            pc2: point cloud 2
        """
        # reg = affine_registration(X, Y, maxIterations=200)
        pass


if __name__ == "__main__":

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    objPC = PointCloud(sess=session)

    with open('./input.txt', 'r') as f:
        for line in f:
            img_path_1, img_path_2 = line.strip().split()
            img1 = Im.open(img_path_1)
            img2 = Im.open(img_path_2)
            objPC.generate_point_cloud(img1, img2)

    __import__('pdb').set_trace()
    objPC.find_correspondence(objPC.pointclouds['1'], objPC.pointclouds['2'])
