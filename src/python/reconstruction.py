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
import cv2
import numpy as np
from PIL import Image as Im
import tensorflow as tf
from pointcloud.pointcloud import PointCloud3d
from collections import OrderedDict


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

        # data structures to store different intermediate data
        self.out = OrderedDict()

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

        mappoints_ids = np.loadtxt(self.mappoint_file)
        mappoints = mappoints_ids[:, 1:]
        ids = mappoints_ids[:, 0].astype(int)

        # storing id -> (x, y, z)
        self.dict_mappoints = dict((i, j.tolist()) for i, j in zip(ids, mappoints))

        self._process_orb_slam_data(self.rgb_folder)

    def construct_3d(self):
        """construct the 3d structure of the given sequence of images

        """

        # step-1 construct the point cloud for all the images
        self.get_pointcloud_in_batch(self.rgb_folder)
        self.find_pointcloud_transform(self.out)

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

    def _find_matching_kp(self, mp1, mp2, debug=False):
        """find the matching keypoints between two mappoint struct

        Args:
            mp1: mappoint 1
            mp2: mappoint 2
        """
        if debug:
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

        # draw_keypoints(mp1['image'], kp_1, mp2['image'], kp_2)

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

        map_struct = OrderedDict()
        for i, (img_file, kp_file) in enumerate(zip(img_files, kp_files)):
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
            map_struct[key] = d

        self.map_struct = map_struct

    def get_pointcloud_in_batch(self, input_dir, file_ext='png'):
        """Calculate the depth of images

        Args:
            input_dir: input directory containing rgb images
        """
        # currently expecting png images
        self.img_files = sorted(glob.glob(os.path.join(input_dir, '*.' + file_ext)))

        for i, (f1, f2) in enumerate(zip(self.img_files[:-1], self.img_files[1:])):
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

    def find_pointcloud_transform(self, mp, scale_factor=1):
        """find the transformation between consecutive point clouds

        Args:
            mp: map point struct
        """

        __import__('pdb').set_trace()
        keys = self.map_struct.keys()
        for i, (k1, k2) in enumerate(zip(keys[:-1], keys[1:])):

            mp1 = self.map_struct[k1]
            mp2 = self.map_struct[k2]

            kp_1, kp_2 = self._find_matching_kp(mp1, mp2)

            kp_1 = np.round(kp_1 / scale_factor).astype(int)
            kp_2 = np.round(kp_2 / scale_factor).astype(int)


if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    parent_dir = '/media/nrupatunga/Data/2018/september/3D-reconstuction/SLAM_3DPoints_Trajectory/'
    objScene3D = SceneReconstruction(session, parent_dir)
    objScene3D.construct_3d()
