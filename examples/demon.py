'''
File: demon.py
Author: Nrupatunga
Email: nrupatunga.tunga@gmail.com
Github: https://github.com/nrupatunga
Description: Script to run the DeMon net on mulitple batch of images

Available Modules:
------------------
    run:
        - This module outputs depth, normal map of dimensions 64x48 and 256x92

    write2pcl:
        - write the point cloud to pcl format

    gtk3dVis:
        - visualize point cloud using gtk library

    open3dVis:
        - Open3d visualization of point cloud

    __compute_local_planes:
        - takes point cloud (x, y, z) and estimates the normal for each point in the point cloud

    get_point_cloud:
        - get the point cloud from depth map

    __compute_point_cloud_from_depthmap:
        - helper function for get_point_cloud
'''

import tensorflow as tf
import numpy as np
from PIL import Image as Im
import os
import cv2
import sys
import pathmagic
from helper import indices as find
from depthmotionnet.networks_original import BootstrapNet, IterativeNet, RefinementNet
from open3d import PointCloud, Vector3dVector, draw_geometries
from camera_params import (fx, fy, cx, cy, relDepthThresh)
import pyximport
pyximport.install()

examples_dir = pathmagic.examples_dir
weights_dir = os.path.join(examples_dir, '..', 'weights')
sys.path.insert(0, '/home/nrupatunga/2018/demon/python/depthmotionnet/')


def prepare_input_data(img1, img2, data_format):
    """Creates the arrays used as input from the two images."""
    # scale images if necessary
    if img1.size[0] != 256 or img1.size[1] != 192:
        img1 = img1.resize((256, 192))
    if img2.size[0] != 256 or img2.size[1] != 192:
        img2 = img2.resize((256, 192))
    img2_2 = img2.resize((64, 48))

    # transform range from [0,255] to [-0.5,0.5]
    img1_arr = np.array(img1).astype(np.float32) / 255 - 0.5
    img2_arr = np.array(img2).astype(np.float32) / 255 - 0.5
    img2_2_arr = np.array(img2_2).astype(np.float32) / 255 - 0.5

    if data_format == 'channels_first':
        img1_arr = img1_arr.transpose([2, 0, 1])
        img2_arr = img2_arr.transpose([2, 0, 1])
        img2_2_arr = img2_2_arr.transpose([2, 0, 1])
        image_pair = np.concatenate((img1_arr, img2_arr), axis=0)
    else:
        image_pair = np.concatenate((img1_arr, img2_arr), axis=-1)

    result = {'image_pair': image_pair[np.newaxis, :],
              'image1': img1_arr[np.newaxis, :],  # first image
              'image2_2': img2_2_arr[np.newaxis, :],  # second image with (w=64,h=48)
              }

    return result


class DemonNet(object):
    """class for Demon network"""

    def __init__(self, session):
        """initialize the network here """

        if tf.test.is_gpu_available(True):
            data_format = 'channels_first'
        else:  # running on cpu requires channels_last data format
            data_format = 'channels_last'

        self.data_format = data_format

        # TODO: somehow this is not working, so figure it out
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

        self.session = session

        # init networks
        self.bootstrap_net = BootstrapNet(session, data_format)
        self.iterative_net = IterativeNet(session, data_format)
        self.refine_net = RefinementNet(session, data_format)
        self.session.run(tf.global_variables_initializer())

        # load weights
        saver = tf.train.Saver()
        saver.restore(session, os.path.join(weights_dir, 'demon_original'))

        # camera params
        scale = 2.5
        self.fx = fx / scale
        self.fy = fy / scale
        self.cx = cx / scale
        self.cy = cy / scale

    def run(self, img1, img2):
        """Compute depth, surface normals, optical flow, flow confidence

        Args:
            img1: input image 1
            img2; input image 2
        Returns:
            Depth, surface normals, optical flow
        """
        # run the network
        input_data = prepare_input_data(img1, img2, self.data_format)
        self.input_data = input_data
        result = self.bootstrap_net.eval(input_data['image_pair'], input_data['image2_2'])
        for i in range(3):
            result = self.iterative_net.eval(input_data['image_pair'],
                                             input_data['image2_2'],
                                             result['predict_depth2'],
                                             result['predict_normal2'],
                                             result['predict_rotation'],
                                             result['predict_translation'])

        # show image
        # __import__('pdb').set_trace()
        # plt.subplot(2, 2, 1)
        # plt.imshow(result['predict_conf2'][0, 0, :, :], cmap='jet')
        # plt.subplot(2, 2, 2)
        # plt.imshow(result['predict_conf2'][0, 1, :, :], cmap='jet')
        # plt.subplot(2, 2, 3)
        # plt.imshow(result['predict_depth2'][0, 0, :, :], cmap='jet')
        # plt.show()

        rotation = result['predict_rotation']
        translation = result['predict_translation']

        # get the image
        image = input_data['image1']
        img = ((image + 0.5) * 255).astype(np.uint8)
        img = img[0].transpose([1, 2, 0])
        img = cv2.resize(img, (64, 48))
        img = np.transpose(img, [2, 0, 1])

        # store the outputs
        out_64_48 = {}
        out_64_48['image'] = img[np.newaxis, ...]
        out_64_48['rotation'] = rotation
        out_64_48['translation'] = translation
        out_64_48['depth'] = result['predict_depth2']
        out_64_48['normal'] = result['predict_normal2']
        out_64_48['conf'] = result['predict_conf2']

        result = self.refine_net.eval(input_data['image1'], result['predict_depth2'])
        out_256_192 = {}
        image = input_data['image1']
        img = ((image + 0.5) * 255).astype(np.uint8)
        out_256_192['image'] = img
        out_256_192['depth'] = result['predict_depth0']

        self.out_64_48 = out_64_48
        self.out_256_192 = out_256_192

        return out_64_48, out_256_192

    def __compute_local_planes(self, X, Y, Z):
        """compute the local planes, normals, normals confidence

        Args:
            X: x coordinates
            Y: y coordinates
            Z: z coordinates

        Returns:
            imgPlanes: image planes
            imgNormals: normals for each 3d-point
            imgConfs: confidence maps
        """

        H = 192
        W = 256

        N = H * W

        # store 3d points and convert to homogeneous coordinates
        pts = np.zeros((N, 4))
        pts[:, 0] = X.ravel()
        pts[:, 1] = Y.ravel()
        pts[:, 2] = Z.ravel()
        pts[:, 3] = np.ones(N)

        u, v = np.meshgrid(np.arange(0, W), np.arange(0, H))
        u, v = u.flatten('F'), v.flatten('F')
        blockWidths = [-1, -3, -6, -9, 0, 1, 3, 6, 9]
        nu, nv = np.meshgrid(blockWidths, blockWidths)

        nx = np.zeros((H, W)).flatten()
        ny = np.zeros((H, W)).flatten()
        nz = np.zeros((H, W)).flatten()
        nd = np.zeros((H, W)).flatten()
        imgConfs = np.zeros((H, W)).flatten()

        ind_all = find(Z, lambda x: x != 0)
        for k in ind_all:
            u2 = u[k] + nu
            v2 = v[k] + nv

            # check that u2 and v2 are in the image
            valid = (u2 >= 0) & (v2 >= 0) & (u2 < W) & (v2 < H)
            u2 = u2[valid]
            v2 = v2[valid]
            ind2 = u2 * H + v2
            ind2 = sorted(ind2)

            # check that depth difference is not too large
            valid = abs(Z[ind2] - Z[k]) < Z[k] * relDepthThresh
            u2 = u2[valid]
            v2 = v2[valid]
            ind2 = u2 * H + v2
            ind2 = sorted(ind2)

            if len(u2) < 3:
                continue

            A = pts[ind2]
            [eigvalues, eigvectors] = np.linalg.eig(np.matmul(A.transpose(), A))
            idx = eigvalues.argsort()
            eigvalues = eigvalues[idx]
            eigvectors = eigvectors[:, idx]

            nx[k] = eigvectors[0, 0]
            ny[k] = eigvectors[1, 0]
            nz[k] = eigvectors[2, 0]
            nd[k] = eigvectors[3, 0]
            imgConfs[k] = 1 - (np.sqrt(eigvalues[0] / eigvalues[1]))

        nx = -1 * np.reshape(nx, (H, W, 1), order='F')
        ny = -1 * np.reshape(ny, (H, W, 1), order='F')
        nz = -1 * np.reshape(nz, (H, W, 1), order='F')
        nd = -1 * np.reshape(nd, (H, W, 1), order='F')
        imgConfs = -1 * np.reshape(imgConfs, (H, W), order='F')

        imgPlanes = np.concatenate((nx, ny, nz, nd), axis=2)
        length = np.sqrt(np.square(nx) + np.square(ny) + np.square(nz))
        eps = 2.2204e-16

        imgPlanes = np.divide(imgPlanes, np.repeat(length + eps, 4, axis=2))
        imgNormals = imgPlanes[:, :, 0:3]
        return imgPlanes, imgNormals, imgConfs

    def gtk3dVis(self):
        """visualize point cloud
        """

        data_format = self.data_format
        input_data = self.input_data
        rotation = self.out_64_48['rotation']
        translation = self.out_64_48['translation']

        # try to visualize the point cloud
        try:
            from depthmotionnet.vis import visualize_prediction
            visualize_prediction(inverse_depth=self.out_256_192['depth'],
                                 image=input_data['image_pair'][0, 0:3] if data_format == 'channels_first' else input_data['image_pair'].transpose([0, 3, 1, 2])[0, 0:3],
                                 rotation=rotation,
                                 translation=translation)
        except ImportError as err:
            print("Cannot visualize as pointcloud.", err)

    def __compute_point_cloud_from_depthmap(self, depth, K, R, t, normals=None, colors=None):
        """Creates a point cloud numpy array and optional normals and colors arrays

        depth: numpy.ndarray
            2d array with depth values

        K: numpy.ndarray
            3x3 matrix with internal camera parameters

        R: numpy.ndarray
            3x3 rotation matrix

        t: numpy.ndarray
            3d translation vector

        normals: numpy.ndarray
            optional array with normal vectors

        colors: numpy.ndarray
            optional RGB image with the same dimensions as the depth map.
            The shape is (3,h,w) with type uint8

        """
        from vis_cython import compute_point_cloud_from_depthmap as _compute_point_cloud_from_depthmap
        return _compute_point_cloud_from_depthmap(depth, K, R, t, normals, colors)

    def get_point_cloud(self, inverse_depth, image=None):
        """Get tht point cloud

        Args:
            depth: input depth to calculate the 3d-points

        Returns:
            points: 3d points

        """
        depth = 1 / inverse_depth
        (h, w) = depth.shape

        intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5])  # sun3d intrinsics
        K = np.eye(3)
        K[0, 0] = intrinsics[0] * w
        K[1, 1] = intrinsics[1] * h
        K[0, 2] = intrinsics[2] * w
        K[1, 2] = intrinsics[3] * h

        R1 = np.eye(3)
        t1 = np.zeros((3,))

        if image is not None:
            img = ((image + 0.5) * 255).astype(np.uint8)
        else:
            img = None

        pointcloud = self.__compute_point_cloud_from_depthmap(depth, K, R1, t1, None, img)

        return pointcloud['points'], pointcloud['colors']

    def write2pcl(self, data):
        """write to pcl file

        Args:
            data: dict containing, input, depth, normal

        TODO: Debug writing header at the beginning to the file
        """

        depth = data['depth'][0, 0]
        xyz = self.get_point_cloud(depth)
        color = data['image'].transpose().reshape((-1, 3))

        color_pack = []
        for rgb in color:
            r, g, b = rgb.tolist()
            pack_rgb = (r << 16) + (g << 8) + b
            color_pack.append(pack_rgb)

        pcl_pc = np.zeros((depth.shape[-1] * depth.shape[-2], 4))
        pcl_pc[:, :-1] = xyz
        pcl_pc[:, -1] = color_pack
        pcl_file = 'pcd.ply'
        np.savetxt(pcl_file, pcl_pc,  fmt='%.7f')

        with open(pcl_file, 'r+') as dest, open('./pcl_header.txt', 'r') as src:
            content = dest.read()
            dest.seek(0)
            dest.truncate()
            for line in src:
                dest.write(line)
            dest.write(content)

    def open3dVis(self, data):
        """Visualize through open3d

        Args:
            data: dict containing, input, depth, normal

        """
        pcd = PointCloud()
        depth = data['depth'][0, 0]
        xyz, _ = self.get_point_cloud(depth, data['image'][0])
        if 'normal' in data.keys():
            normals = np.transpose(data['normal'][0],  (1, 2, 0))
            normals = normals.reshape((-1, 3))
        else:
            _, normals, _ = self.__compute_local_planes(xyz[:, 0], xyz[:, 1], xyz[:, 2])
            normals = normals.reshape((-1, 3))

        color = np.transpose(data['image'][0], [1, 2, 0]).reshape((-1, 3))

        pcd.points = Vector3dVector(xyz)
        pcd.colors = Vector3dVector(color / 255.)
        pcd.normals = Vector3dVector(normals)
        draw_geometries([pcd])


if __name__ == "__main__":

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    objD = DemonNet(session)
    with open('./input.txt', 'r') as f:
        for line in f:
            img_path_1, img_path_2 = line.strip().split()
            img1 = Im.open(img_path_1)
            img2 = Im.open(img_path_2)
            out_64_48, out_256_192 = objD.run(img1, img2)
            # objD.gtk3dVis()
            objD.open3dVis(out_64_48)
            # objD.open3dVis(out_256_192)
            # objD.write2pcl(out_64_48)
