""" File: reconstruction.py
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
import cv2
import copy
from tqdm import tqdm
import numpy as np
from PIL import Image as Im
import tensorflow as tf
from pointcloud.pointcloud import PointCloud3d
from collections import OrderedDict
# from open3d import PointCloud, Vector3dVector, copy, draw_geometries
from open3d import *  # noqa


class SceneReconstruction(object):

    """Class which provides utilities to reconstruct 3d from set of
    images and information from ORB-slam """

    def __init__(self, sess, input_dir, slam_w, slam_h, depth_w=64, depth_h=48):
        """Initialize the input parameters

        Args:
            sess: tensorflow session
            input_dir: input to the folder containing images and SLAM
            data
            width: width of the depth map
            height: height of the depth map
        """
        self.objPC = PointCloud3d(sess=session)
        self.objD = self.objPC.objD

        # data structures to store different intermediate data
        self.out = OrderedDict()
        self.map_struct = OrderedDict()
        self.R_t_s = OrderedDict()

        self._process_data(input_dir)

        # In order to set these, please check the width and height of
        # the depth map
        self.d_W = depth_w
        self.d_H = depth_h

        self.scale_factor = slam_w * 1. / depth_w
        self.scale_factor = slam_h * 1. / depth_h

    def _process_data(self, input_dir):
        """Process the data in the input directory and set the required
        parameters

        Args:
            input_dir: input data directory containing SLAM data

        """
        self.parent_dir = input_dir
        self.mappoint_file = os.path.join(input_dir, 'mappoints.txt')
        self.gt_file = os.path.join(input_dir, 'KeyFrameTrajectory.txt')
        self.rgb_folder = os.path.join(input_dir, 'images')

        mappoints_ids = np.loadtxt(self.mappoint_file)
        self.mappoints = mappoints_ids[:, 1:]
        ids = mappoints_ids[:, 0].astype(int)

        # storing id -> (x, y, z)
        self.dict_mappoints = dict((i, j.tolist()) for i, j in zip(ids, self.mappoints))

        self._process_orb_slam_data(self.rgb_folder)

    def construct_3d(self):
        """construct the 3d structure of the given sequence of images

        """

        # step-1 construct the point cloud for all the images
        self._get_pointclouds(self.rgb_folder)
        self._find_pointcloud_transforms(self.dict_mappoints, self.out,
                                         scale_factor=self.scale_factor)
        self._merge_point_clouds(self.R_t_s)

    def _repeating_elements(self, arr):
        """Identify the repeating elements in an array

        Args:
            arr: input array

        Returns:
            returns the repeating array items
        """
        from collections import Counter

        import sys
        if sys.version_info[0] == 3:
            return [item for item, count in Counter(arr).items() if count > 1]
        else:
            return [item for item, count in Counter(arr).iteritems() if count > 1]

    def _draw_keypoints(self, img1, kp1, img2, kp2):
        """Draw keppoint matches

        Args:
            img1: image 1
            kp1: keypoints on image 1
            img2: image 2
            kp2: keypoints on image 2

        """
        if len(img1.shape) == 3:
            new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
        elif len(img1.shape) == 2:
            new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
        new_img = np.zeros(new_shape, type(img1.flat[0]))
        # Place images onto the new image.
        new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
        new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

        kp_len = len(kp1)
        thickness = 2
        r = 5
        for m in range(kp_len):
            # Generate random color for RGB/BGR and grayscale images as needed.
            # c = np.random.randint(0, 256, 3) if len(img1.shape) == 3 else np.random.randint(0, 256)
            c = (0, 255, 0)
            # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
            # wants locs as a tuple of ints.
            end1 = tuple(kp1[m])
            end2 = tuple(kp2[m] + np.array([img1.shape[1], 0]))
            cv2.line(new_img, end1, end2, c, thickness)
            cv2.circle(new_img, end1, r, c, thickness)
            cv2.circle(new_img, end2, r, c, thickness)

        cv2.imshow('output', new_img)
        cv2.waitKey(0)

    def _find_matching_kp(self, mp1, mp2, vis=False):
        """find the matching keypoints between two mappoint struct

        Args:
            mp1: mappoint 1
            mp2: mappoint 2
        """
        if vis:
            cv2.imshow('input', np.concatenate((mp1['image'], mp2['image']), axis=1))
            cv2.imwrite('input.jpg', np.concatenate((mp1['image'], mp2['image']), axis=1))
            cv2.waitKey(0)

        id_1 = mp1['id']
        id_2 = mp2['id']

        id_match = np.array(list(set(id_1) & set(id_2)))
        idx_1 = np.nonzero(np.isin(id_1, id_match))
        idx_2 = np.nonzero(np.isin(id_2, id_match))

        kp_1 = mp1['kp'][idx_1]
        kp_2 = mp2['kp'][idx_2]

        if vis:
            self._draw_keypoints(mp1['image'], kp_1, mp2['image'], kp_2)

        return kp_1, kp_2

    def _process_orb_slam_data(self, input_dir, img_ext='png'):
        """ process the orb slam data
        """

        # currently expecting png images
        self.img_files = sorted(glob.glob(os.path.join(input_dir, '*.' + img_ext)))
        # currently we assume the format of file is '.txt'
        self.kp_files = sorted(glob.glob(os.path.join(self.rgb_folder, '*.txt')))

        # rgb files and keypoint files
        img_files = self.img_files[:-1]  # last file does not have a point cloud so we do not use it
        kp_files = self.kp_files[:-1]

        for i, (img_file, kp_file) in tqdm(enumerate(zip(img_files, kp_files)),
                                           desc='Processing slam data', total=len(img_files)):
            img = cv2.imread(img_file)
            kp_id = np.loadtxt(kp_file)
            kp_id = kp_id[kp_id[:, 2].argsort()]

            kp = np.round(kp_id[:, 0:2]).astype(int)
            id_ = kp_id[:, 2].astype(int)
            repeats = self._repeating_elements(id_)
            for e in repeats:
                idx_0 = np.where(id_ == e)[0][0]  # index 0
                id_ = np.delete(id_, idx_0, axis=0)
                kp = np.delete(kp, idx_0, axis=0)

            xyz = np.asarray([self.dict_mappoints[i] for i in id_])

            d = {}
            d['image'] = img  # image data
            d['kp'] = kp  # (x, y)
            d['id'] = id_  # id for mappoints
            d['xyz'] = xyz  # xyz mappoints

            key = os.path.basename(img_file)
            self.map_struct[key] = d

    def _get_pointclouds(self, input_dir, file_ext='png'):
        """Calculate the depth of images

        Args:
            input_dir: input directory containing rgb images
        """
        # currently expecting png images
        self.img_files = sorted(glob.glob(os.path.join(input_dir, '*.' + file_ext)))

        for i, (f1, f2) in tqdm(enumerate(zip(self.img_files[:-1],
                                          self.img_files[1:])),
                                desc='Generating point clouds', total=len(self.img_files) - 1):

            img1 = Im.open(f1)
            img2 = Im.open(f2)

            # key
            key = os.path.basename(f1)

            # calculate point cloud
            out_64_48, _ = self.objD.run(img1, img2)
            pcd = self.objD.open3dPcd(out_64_48)

            # keys present in out_64_48 : ['translation', 'pointcloud', 'image', 'depth', 'rotation', 'conf', 'normal'])
            out_64_48['pointcloud'] = pcd
            self.out[key] = out_64_48

    def _find_pointcloud_transforms(self, slam_mappoints, mp, scale_factor=1):
        """find the transformation between consecutive point clouds

        Args:
            mp: map point struct
        """
        keys = self.map_struct.keys()
        keys = [k for k in keys]

        for i, key in tqdm(enumerate(keys), desc='Transforming point clouds'):
            mp = self.map_struct[key]

            # get the mappoints in current frame
            pc_slam = np.array([slam_mappoints[i] for i in mp['id'].tolist()])

            kp = mp['kp']
            kp = np.round(kp / scale_factor).astype(int)
            pc_dl = np.array(self.out[key]['pointcloud'].points)
            pc_dl = pc_dl[kp[:, 1] * self.d_W + kp[:, 0]]

            self.R_t_s[key] = self.objPC.find_transform(pc_slam.tolist(),
                                                        pc_dl.tolist(),
                                                        vis=False)

    def convert_transform_4x4(self, reg):
        """Convert R, t, s to 4x4 matrix

        Args:
            reg: registration handle

        Returns:
            sRt: 4x4 transformation

        """
        sRt = np.eye(4)
        sR = reg.s * reg.R
        sRt[0:3, 0:3] = sR
        sRt[0:3, 3] = reg.t

        return sRt

    def _convert_to_open3d_format(self, pointcloud):
        """convert the point cloud containing color and points
        information to open3d format

        Args:
            pointcloud: numpy dict containing point cloud and colors
        """
        pcd = PointCloud()
        pcd.points = Vector3dVector(pointcloud['points'])
        pcd.colors = Vector3dVector(pointcloud['colors'])

        return pcd

    def _run_icp(self, pc1, pc2):
        """run icp

        :pc1: TODO
        :pc2: TODO
        :returns: TODO

        """

    def draw_registration_result_original_color(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        source_temp.transform(transformation)
        draw_geometries([source_temp, target])

    def _merge_point_clouds(self, R_t_s, step=10):
        """Merge all the point clouds

        Args:
            R_t_s: rigid transformation between each point clouds

        """
        keys = self.R_t_s.keys()
        keys = [k for k in keys]
        keys = keys[0:10]

        # slam point clouds
        pcd_base = {}
        pcd_base['points'] = np.asarray(self.mappoints)
        pcd_base['colors'] = np.zeros(self.mappoints.shape)

        # first_merge = True
        transformed_pc = []
        for key in tqdm(keys, desc='Merging point clouds'):

            # registration parameters
            reg = R_t_s[key]

            # point cloud to be transformed
            pcd = {}
            pcd['points'] = np.asarray(self.out[key]['pointcloud'].points)
            pcd['colors'] = np.asarray(self.out[key]['pointcloud'].colors)

            transformed_pc.append(self._convert_to_open3d_format(self.objPC.transform_point_cloud(pcd, reg)))
            # if first_merge:
            # pcd_final = self.objPC.merge_visualise(transformed_pc, transformed_pc)
            # first_merge = False
            # else:
            # pcd_final = self.objPC.merge_visualise(pcd_base, transformed_pc)

            # pcd_base = {}
            # pcd_base['points'] = np.asarray(pcd_final.points)
            # pcd_base['colors'] = np.asarray(pcd_final.colors)
        draw_geometries(transformed_pc)

        source = transformed_pc[0]
        target = transformed_pc[1]

        current_transformation = np.identity(4)
        # self.draw_registration_result_original_color(source, target, current_transformation)

        # point to plane ICP
        current_transformation = np.identity(4)
        print("2. Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. Distance threshold 0.02.")
        result_icp = registration_icp(source, target, 0.5, current_transformation, TransformationEstimationPointToPoint())
        print(result_icp)
        self.draw_registration_result_original_color(source, target, result_icp.transformation)


if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    parent_dir = '/media/nrupatunga/data/datasets/3D-reconstuction/2/'
    objScene3D = SceneReconstruction(session, parent_dir, slam_w=640,
                                     slam_h=480)
    objScene3D.construct_3d()
