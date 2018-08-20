"""
File: main.py
Author: Nrupatunga
Email: nrupatunga.tunga@gmail.com
Github: https://github.com/nrupatunga
Description: test file to run demon and eigen networks
"""

from depth_normal_vgg import vgg_depth_normal
from PIL import Image as Im
from demon import DemonNet
import tensorflow as tf
import numpy as np
import config
import time

# dimensions of DL depth
WIDTH = config.WIDTH
HEIGHT = config.HEIGHT


# Class structure for holding weights and bias of each layer
class Weights_bias(object):

    """Docstring for Weights_bias. """

    def __init__(self, W, b):
        """TODO: to be defined1. """
        self.W = W
        self.b = b


if __name__ == "__main__":

    # tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    # Demon net
    objDemonDepth = DemonNet(session)

    # Eigen depth and normal
    imgs = tf.placeholder(tf.float32, [1, HEIGHT, WIDTH, 3])
    objVggDepth = vgg_depth_normal(imgs, session, './depth_normals/depth_normal.npy')

    # run
    with open('./input.txt', 'r') as f:
        for line in f:
            img_path_1, img_path_2 = line.strip().split()
            img1 = Im.open(img_path_1)
            img2 = Im.open(img_path_2)

            # run demon and visualise
            start = time.time()
            out_64_48, out_256_192 = objDemonDepth.run(img1, img2)
            print('Total Demon Time = {}'.format(time.time() - start))
            objDemonDepth.open3dVis(out_64_48)

            # run eigen and visualise
            img1 = img1.resize((WIDTH, HEIGHT), Im.BICUBIC)
            img1 = np.asarray(img1)
            start = time.time()
            output = session.run([objVggDepth.normal, objVggDepth.depth], feed_dict={objVggDepth.imgs: [img1]})
            print('Total Eigen Time = {}'.format(time.time() - start))

            # store the output data
            img1 = Im.open(img_path_1)
            img1 = img1.resize((output[0].shape[-2], output[1].shape[-3]), Im.BICUBIC)
            out_140_109 = {}
            img1 = np.asarray(img1)
            img1 = img1[np.newaxis, ...]
            # out_140_109['image'] = img1
            # out_140_109['depth'] = output[0]
            # out_140_109['normal'] = output[1]
            # objVggDepth.open3dVis(np.squeeze(output[1]), np.squeeze(output[0]), np.squeeze(img1[0]))
            objVggDepth.open3dVis(np.squeeze(output[1]), None, np.squeeze(img1[0]))
