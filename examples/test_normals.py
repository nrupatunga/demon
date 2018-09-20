"""
File: test_normals.py
author: hmishra2250
pytest module for normal and write2pcl
Command: py.test test_normals.py -v -s
"""

import pytest
import pickle
from demon import DemonNet
import numpy as np
import tensorflow as tf
import filecmp
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
objD = DemonNet(session)


class TestNormal(object):

    """Test modules for nyu toolset"""

    @pytest.mark.normal
    def test_normal(self):
        """Test get_surface_3d module

        Args:
            norms3d: normal map
        """

        out = pickle.load(open('test_data/out_64_48.pkl', 'rb'))
        gt_normals = np.load('test_data/normals.npy')
        normals = objD.filter_norm_room(out)['normal']
        assert np.sqrt(np.mean((normals.flatten() - gt_normals.flatten())**2)) <= 1e-6

    @pytest.mark.write2pcl
    def test_write2pcl(self):
        """Test get_room_directions
        :returns: TODO

        """
        out = pickle.load(open('test_data/out_64_48.pkl', 'rb'))
        objD.write2pcl(out)
        assert filecmp.cmp('pcl.pcd', 'test_data/test_pcl.pcd') == True

