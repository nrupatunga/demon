"""
File: pointcloud.py
Author: Nrupatunga
Email: nrupatunga.tunga@gmail.com
Github: https://github.com/nrupatunga
Description: Utility for pointcloud generation and processing
"""

import tensorflow as tf
from demon.demon import DemonNet
from PIL import Image as Im
import numpy as np
import cv2
# from affine_registration import affine_registration
from pycpd.pycpd.rigid_registration import rigid_registration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from functools import partial
from open3d import PointCloud, Vector3dVector, draw_geometries


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.0000001)


def dummy_visualize(iteration, error, X, Y, ax):
    pass


class PointCloud3d(object):

    """Utility class for point cloud generation and processing"""

    def __init__(self, sess=None, depth_network_type='demon'):
        """initialization

        Args:
            sess: tensorflow session
        """

        if sess is None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

        self.sess = sess
        self.pointclouds = {}
        self.count = 0

        if depth_network_type is 'demon':
            self.objD = DemonNet(sess)
        else:
            pass

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
        self.H = depth.shape[0]
        self.W = depth.shape[1]

        # point cloud
        image = np.array(image).astype(np.float32) / 255 - 0.5  # get_point_cloud works on image in range 0 - 1
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

        # images are stores as list in point cloud structure; using the
        # same to avoid redundancy
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

            # 3d points
            pc1_match.append(pc1['points'][round(y1) * self.W + round(x1)].tolist())
            pc2_match.append(pc2['points'][round(y2) * self.W + round(x2)].tolist())

        img3 = np.copy(img2)
        img3 = cv2.drawMatchesKnn(gray_1, kp1, gray_2, kp2, good, img3, flags=2)
        plt.imshow(img3)
        plt.show()
        return pc1_match, pc2_match,

    def find_transform(self, pc1, pc2, vis=False):
        """Find the R and t between two point clouds

        Args:
            pc1: point cloud 1
            pc2: point cloud 2
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if vis:
            callback = partial(visualize, ax=ax)
        else:
            callback = partial(dummy_visualize, ax=ax)

        reg = rigid_registration(**{'X': np.asarray(pc1), 'Y': np.asarray(pc2)})
        reg.register(callback)

        if vis:
            plt.show()

        return reg

    def transform_point_cloud(self, pc, reg):
        """

        Args:
            pc: point cloud
            reg: registration handle
        """
        s, R, t = reg.get_registration_parameters()
        pc_t = {}
        pc_t['points'] = s * np.dot(pc['points'], np.transpose(R)) + np.tile(t, (pc['points'].shape[0], 1))
        pc_t['colors'] = pc['colors']

        return pc_t

    def merge_visualise(self, pc1, pc2):
        """Concatenate all 3d points in pc1 and pc2

        Args:
            pc1: point cloud 1
            pc2: point cloud 2
        """
        pcd = PointCloud()
        points = np.concatenate((pc1['points'], pc2['points']), axis=0)
        colors = np.concatenate((pc1['colors'], pc2['colors']), axis=0)

        pcd.points = Vector3dVector(np.asarray(points))
        # pcd.colors = Vector3dVector(np.asarray(colors) / 255.)
        pcd.colors = Vector3dVector(np.asarray(colors))
        draw_geometries([pcd])

        return pcd


if __name__ == "__main__":

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    objPC = PointCloud3d(sess=session)

    with open('input.txt', 'r') as f:
        for line in f:
            img_path_1, img_path_2 = line.strip().split()
            img1 = Im.open(img_path_1)
            img2 = Im.open(img_path_2)
            objPC.generate_point_cloud(img1, img2)

    pc1, pc2 = objPC.find_correspondence(objPC.pointclouds['1'], objPC.pointclouds['2'])
    reg = objPC.find_transform(pc1, pc2)
    transformed_pc = objPC.transform_point_cloud(objPC.pointclouds['2'], reg)
    objPC.merge_visualise(objPC.pointclouds['1'], transformed_pc)
    objPC.merge_visualise(objPC.pointclouds['1'], objPC.pointclouds['2'])
