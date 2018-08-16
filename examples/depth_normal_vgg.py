# depth_normal_vgg.py
# !/usr/bin/env python2
# Date: Monday 25 June 2018
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Depth and normal estimation

'''
    defines a class and apis to get the depth and normal of the input
Available classes:
    Weights_bias: class structure to pickle weights and bias from .npy
    file
    vgg_depth_normal: class which defines VGG depth and normal
    architecture
Available class modules:
    vgg_depth_normal.build_net: builds the VGG network
'''

import tensorflow as tf
import numpy as np
import glob
import time
from PIL import Image
import config
from open3d import PointCloud, Vector3dVector, draw_geometries
from camera_params import relDepthThresh
from helper import indices as find

# Keynames of different stacks
imagenet_stack = config.imagenet_stack
scale2_stack = config.scale2_stack


# Class structure for holding weights and bias of each layer
class Weights_bias(object):

    """Weight bias class"""

    def __init__(self, W, b):
        """TODO: to be defined1. """
        self.W = W
        self.b = b


def get3dpoints(depthmap, image):
    """TODO: Docstring for function.
    """

    mappoints = []
    height = depthmap.shape[0]
    width = depthmap.shape[1]
    with open('points.txt', 'w') as f:
        for i in range(height):
            for j in range(width):
                z = depthmap[i, j]
                x = (j * config.fxi + config.cxi) * z
                y = (i * config.fyi + config.cyi) * z

                point = [x, y, z]
                pixel_c = image[i, j, :]
                f.write('{}, {}, {}, {}, {}, {}\n'.format(x, y, z, pixel_c[2], pixel_c[1], pixel_c[0]))
                mappoints.append(point)

    return mappoints


class vgg_depth_normal(object):

    """defines the vgg depth and normal architecture with apis for
    testing these networks"""

    def __init__(self, imgs, sess, weights_path=None):
        """Initialization of network input parameters

        Args:
            imgs: input batch of images, can be single or more than one
            sess: tensorflow session
            weights_path: path to weights of the network
        """

        self.imgs = imgs

        # input dims to the network
        self.input_h = 228
        self.input_w = 304

        # scale 2 size
        self.scale2_size = (55, 74)

        # full2 feature size
        self.fc2_feature_size = (1, 64, 14, 19)

        # calculate the crop indices
        [_, H, W, C] = imgs.shape.as_list()
        dh = H - self.input_h
        dw = W - self.input_w

        # indices to crop
        (self.i0, self.i1) = (dh / 2, H - dh / 2)
        (self.j0, self.j1) = (dw / 2, W - dw / 2)

        # Pretrained model
        if weights_path:
            self.model_params = np.load(weights_path, encoding='latin1')
        else:
            self.model_params = np.load('./depth_normal.npy')
            print(self.model_params)

        # visualize
        self.vis = True

        # logdepths_std
        self.logdepths_std = 0.45723134

        # build the network
        self.build_net()

    def compute_local_planes(self, X, Y, Z):
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

        H = 109
        W = 147

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

    def open3dVis(self, depth, normal=None, color=None):
        """Visualize 3d map points from eigen depth

        Args:
            depth: depth map
            normal: normal map
            color: rgb image

        """
        points = get3dpoints(depth, color)
        points = np.asarray(points)
        pcd = PointCloud()
        pcd.points = Vector3dVector(points)

        if color is not None:
            color = np.reshape(color, [-1, 3])
            pcd.colors = Vector3dVector(color / 255.)

        if normal is not None:
            normal = np.reshape(normal, [-1, 3])
        else:
            _, normal, _ = self.compute_local_planes(points[:, 0],
                                                     points[:, 1],
                                                     points[:, 2])
            normal = np.reshape(normal, [-1, 3])

        pcd.normals = Vector3dVector(normal)
        draw_geometries([pcd])

    def write_summaries(self):
        """ writes the tensorflow summaries
        """
        # save graph
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(logdir='./logs', graph=sess.graph)

    def get_conv_filter(self, name):
        """get the convolution filter weights

        Args:
            name: name of the convolution layer
        Returns:
            returns conv weights
        """

        W = self.model_params.item().get(name).W
        W = np.transpose(W, [2, 3, 1, 0])
        return W

    def get_fc_weights(self, name):
        """get the FC layer weights

        Args:
            name: name of the FC layer
        Returns:
            returns FC weights
        """

        return self.model_params.item().get(name).W

    def get_bias(self, name):
        """get the biases

        Args:
            name: name of the layer
        Returns:
            returns bias
        """

        return self.model_params.item().get(name).b

    def conv_layer(self, bottom, name, relu_flag=True, padding='SAME', stride=1):
        """ defines the convolution layer

        Args:
            bottom: input to the layer
            name: name of the conv layer
            relu_flag: if true relu applied after convolution
            padding: type of padding to the input
            stride: stride for filtering

        Returns:
            returns TF conv op
        """

        with tf.variable_scope(name):

            # get model parameters
            W = self.get_conv_filter(name)
            b = self.get_bias(name)

            # calculate the padding, if the stride != 1
            if stride != 1:
                if padding != 'VALID':
                    raise NotImplementedError()

                (_, in_H, in_W, _) = bottom.shape.as_list()
                (k_H, k_W, _, _) = W.shape
                old_H = np.ceil((in_H - k_H) / float(stride)) * stride + k_H
                old_W = np.ceil((in_W - k_W) / float(stride)) * stride + k_W
                pad = (int(np.ceil((old_H - in_H) / 2.)), int(np.ceil((old_W - in_W) / 2.)))
                bottom = tf.pad(bottom, [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], 'CONSTANT')

            # out = relu(Wx + b)
            conv = tf.nn.conv2d(bottom, W, [1, stride, stride, 1], padding=padding)
            out_conv = tf.nn.bias_add(conv, b)
            if relu_flag:
                out_relu = tf.nn.relu(out_conv)
            else:
                out_relu = out_conv

            return out_relu

    def conv_T_layer(self, x, name):
        """ defines the transpose convolution layer

        Args:
            x: input to the layer
            name: name of the conv layer
        Returns:
            returns transpose of convolution TF op
        """

        (batch, h, w, channels) = x.shape.as_list()

        ker = self.get_conv_filter(name)
        output_shape = (batch, h, w, ker.shape[2])
        conv_t = tf.nn.conv2d_transpose(x, ker, output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME')

        b = self.get_bias(name)
        out_conv = tf.nn.bias_add(conv_t, b)

        return out_conv

    def max_pool(self, bottom, name, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1]):
        """ defines the max pooling layer

        Args:
            bottom: input to the layer
            name: name of the conv layer
            ksize: kernel size
            stride: stride of filtering

        Returns:
            returns max-pool TF op
        """

        return tf.nn.max_pool(bottom, ksize=ksize, strides=stride, padding='VALID', name=name)

    def fc_layer(self, bottom, name, extra=''):
        """ defines the FC layer

        Args:
            bottom: input to the layer
            name: name of the conv layer
            extra: name + extra

        Returns:
            returns fc TF op
        """

        with tf.variable_scope(name + extra):
            W = self.get_fc_weights(name)
            b = self.get_bias(name)
            fc = tf.nn.bias_add(tf.matmul(bottom, W), b)

        return fc

    def inverse_depth_transform(self, depth):
        """ inverse depth transformation

        Args:
            depth: input depth
        Returns:
            transformed depth
        """

        return tf.exp(depth * self.logdepths_std, name='inverse_depth_transform')

    def define_scale2_onestack(self, stack_type, p_1_drop, p_1_mean):
        """ defines scale2 specific to depth and normal

        Args:
            stack_typ: 'normals' or 'depths'
            p_1_drop: drop-out input
            p_1_mean: mean input
        Returns:
            None
        """

        if stack_type == 'normals':
            self.conv_s2_2_drop_n = self.conv_layer(p_1_drop, '{}_conv_s2_2'.format(stack_type))
            self.conv_s2_2_mean_n = self.conv_layer(p_1_mean, '{}_conv_s2_2'.format(stack_type))

            self.conv_s2_3_drop_n = self.conv_layer(self.conv_s2_2_drop_n, '{}_conv_s2_3'.format(stack_type))
            self.conv_s2_3_mean_n = self.conv_layer(self.conv_s2_2_mean_n, '{}_conv_s2_3'.format(stack_type))

            self.conv_s2_4_drop_n = self.conv_layer(self.conv_s2_3_drop_n, '{}_conv_s2_4'.format(stack_type))
            self.conv_s2_4_mean_n = self.conv_layer(self.conv_s2_3_mean_n, '{}_conv_s2_4'.format(stack_type))

            self.conv_s2_5_drop_n = self.conv_T_layer(self.conv_s2_4_drop_n, '{}_conv_s2_5'.format(stack_type))
            self.conv_s2_5_mean_n = self.conv_T_layer(self.conv_s2_4_mean_n, '{}_conv_s2_5'.format(stack_type))

            pred_drop = self.conv_s2_5_drop_n
            pred_mean = self.conv_s2_5_mean_n

            pred_drop_sum = tf.sqrt(tf.reduce_sum(pred_drop ** 2, axis=3) + 1e-4)
            pred_drop_sum = tf.expand_dims(pred_drop_sum, axis=3)
            pred_mean_sum = tf.sqrt(tf.reduce_sum(pred_mean ** 2, axis=3) + 1e-4)
            pred_mean_sum = tf.expand_dims(pred_mean_sum, axis=3)

            self.pred_drop_n = pred_drop / pred_drop_sum
            self.pred_mean_n = pred_mean / pred_mean_sum
            self.pred_drop_n_up = tf.image.resize_nearest_neighbor(self.pred_drop_n, (self.scale2_size[0] * 2, self.scale2_size[1] * 2))
            self.pred_mean_n_up = tf.image.resize_nearest_neighbor(self.pred_mean_n, (self.scale2_size[0] * 2, self.scale2_size[1] * 2))

            (n_b, n_h, n_w, n_c) = self.pred_drop_n_up.shape.as_list()
            self.pred_drop_n_up_crop = tf.slice(self.pred_drop_n_up, [0, 0, 0, 0], [n_b, n_h - 1, n_w - 1, n_c])
            self.pred_mean_n_up_crop = tf.slice(self.pred_mean_n_up, [0, 0, 0, 0], [n_b, n_h - 1, n_w - 1, n_c])

        elif stack_type == 'depths':
            self.conv_s2_2_drop_d = self.conv_layer(p_1_drop, '{}_conv_s2_2'.format(stack_type))
            self.conv_s2_2_mean_d = self.conv_layer(p_1_mean, '{}_conv_s2_2'.format(stack_type))

            self.conv_s2_3_drop_d = self.conv_layer(self.conv_s2_2_drop_d, '{}_conv_s2_3'.format(stack_type))
            self.conv_s2_3_mean_d = self.conv_layer(self.conv_s2_2_mean_d, '{}_conv_s2_3'.format(stack_type))

            self.conv_s2_4_drop_d = self.conv_layer(self.conv_s2_3_drop_d, '{}_conv_s2_4'.format(stack_type))
            self.conv_s2_4_mean_d = self.conv_layer(self.conv_s2_3_mean_d, '{}_conv_s2_4'.format(stack_type))

            self.conv_s2_5_drop_d = self.conv_T_layer(self.conv_s2_4_drop_d, '{}_conv_s2_5'.format(stack_type))
            self.conv_s2_5_mean_d = self.conv_T_layer(self.conv_s2_4_mean_d, '{}_conv_s2_5'.format(stack_type))

            pred_drop = self.conv_s2_5_drop_d
            pred_mean = self.conv_s2_5_mean_d

            bias = self.get_bias('depths_bias')
            bias_drop = tf.reshape(bias, shape=pred_drop.shape)
            bias_mean = tf.reshape(bias, shape=pred_drop.shape)

            self.pred_drop_d = pred_drop + bias_drop
            self.pred_mean_d = pred_mean + bias_mean
            self.pred_drop_d_up = tf.image.resize_nearest_neighbor(self.pred_drop_d, (self.scale2_size[0] * 2, self.scale2_size[1] * 2))
            self.pred_mean_d_up = tf.image.resize_nearest_neighbor(self.pred_mean_d, (self.scale2_size[0] * 2, self.scale2_size[1] * 2))

            (n_b, n_h, n_w, n_c) = self.pred_drop_d_up.shape.as_list()
            self.pred_drop_d_up_crop = tf.slice(self.pred_drop_d_up, [0, 0, 0, 0], [n_b, n_h - 1, n_w - 1, n_c])
            self.pred_mean_d_up_crop = tf.slice(self.pred_mean_d_up, [0, 0, 0, 0], [n_b, n_h - 1, n_w - 1, n_c])

    def define_scale3_onestack(self, stack_type, p_2_drop, p_2_mean, crop_size):
        """ defines scale3 specific to depth and normal

        Args:
            stack_typ: 'normals' or 'depths'
            p2_drop: drop-out input
            p2_mean: mean input
            crop_size: crop size
        Returns:
            None
        """

        if stack_type == 'normals':
            self.conv_s3_2_drop_n = self.conv_layer(p_2_drop, '{}_conv_s3_2'.format(stack_type))
            self.conv_s3_2_mean_n = self.conv_layer(p_2_mean, '{}_conv_s3_2'.format(stack_type))

            self.conv_s3_3_drop_n = self.conv_layer(self.conv_s3_2_drop_n, '{}_conv_s3_3'.format(stack_type))
            self.conv_s3_3_mean_n = self.conv_layer(self.conv_s3_2_mean_n, '{}_conv_s3_3'.format(stack_type))

            self.conv_s3_4_drop_n = self.conv_T_layer(self.conv_s3_3_drop_n, '{}_conv_s3_4'.format(stack_type))
            self.conv_s3_4_mean_n = self.conv_T_layer(self.conv_s3_3_mean_n, '{}_conv_s3_4'.format(stack_type))

            pred_drop = self.conv_s3_4_drop_n
            pred_mean = self.conv_s3_4_mean_n

            pred_drop_sum = tf.sqrt(tf.reduce_sum(pred_drop ** 2, axis=3) + 1e-4)
            pred_drop_sum = tf.expand_dims(pred_drop_sum, axis=3)
            pred_mean_sum = tf.sqrt(tf.reduce_sum(pred_mean ** 2, axis=3) + 1e-4)
            pred_mean_sum = tf.expand_dims(pred_mean_sum, axis=3)

            self.pred_drop_normal = pred_drop / pred_drop_sum
            self.pred_mean_normal = pred_mean / pred_mean_sum

            self.normal = self.pred_mean_normal
            self.normal_r = tf.image.resize_bicubic(self.normal, (config.HEIGHT, config.WIDTH))

        elif stack_type == 'depths':
            self.conv_s3_2_drop_d = self.conv_layer(p_2_drop, '{}_conv_s3_2'.format(stack_type))
            self.conv_s3_2_mean_d = self.conv_layer(p_2_mean, '{}_conv_s3_2'.format(stack_type))

            self.conv_s3_3_drop_d = self.conv_layer(self.conv_s3_2_drop_d, '{}_conv_s3_3'.format(stack_type))
            self.conv_s3_3_mean_d = self.conv_layer(self.conv_s3_2_mean_d, '{}_conv_s3_3'.format(stack_type))

            self.pred_drop_depth = self.conv_T_layer(self.conv_s3_3_drop_d, '{}_conv_s3_4'.format(stack_type))
            self.pred_mean_depth = self.conv_T_layer(self.conv_s3_3_mean_d, '{}_conv_s3_4'.format(stack_type))

            self.depth = self.inverse_depth_transform(self.pred_mean_depth)
            self.depth_r = tf.image.resize_bicubic(self.depth, (config.HEIGHT, config.WIDTH))

    # Works for batch with size = 1, TODO: make it general
    def upsample_bilinear(self, x, scale):
        """ defines bilinear upsize using tf ops only

        Args:
            x: input
            scale: scale

        Returns:
            upsampled output
        """

        (batch, h, w, channels) = x.shape.as_list()
        x = tf.transpose(x, [3, 1, 2, 0])

        kx = np.linspace(0, 1, scale + 1)[1:-1]
        kx = np.concatenate((kx, [1], kx[::-1]))
        ker = kx[None, :] * kx[:, None]
        ker = ker[:, :, None, None]
        ker = tf.convert_to_tensor(ker, dtype=tf.float32)

        (k_h, k_w, _, _) = ker.shape.as_list()
        output_shape = (channels * batch, (h - 1) * scale + k_h, (w - 1) * scale + k_w, 1)

        upsample = tf.nn.conv2d_transpose(x, ker,
                                          output_shape=output_shape,
                                          strides=[1, scale, scale, 1],
                                          padding='VALID')

        upsample = tf.transpose(upsample, [3, 1, 2, 0])
        return upsample

    def define_scale3_training_crop(self):
        """ defines scale3 arch (refer paper for understanding)

        Args:
            None required
        """

        (ch, cw) = self.scale2_size
        (oh, ow) = (2 * ch - 1, 2 * cw - 1)

        rh = np.floor(np.random.uniform() * (oh - ch)).astype(np.int32)
        rw = np.floor(np.random.uniform() * (ow - cw)).astype(np.int32)
        rh = np.floor(0.2 * (oh - ch)).astype(np.int32)
        rw = np.floor(0.2 * (ow - cw)).astype(np.int32)

        x0 = self.x0
        x0_crop = tf.image.crop_to_bounding_box(x0, 2 * rh, 2 * rw, 2 * (ch + 1) + 8, 2 * (cw + 1) + 8)

        return rh, rw, x0_crop

    def build_net(self):
        """ builds the VGG depth and normal network architecture

        Args:
            None required
        """

        self.parameters = []
        with tf.name_scope('preprocess_1'):
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.x0 = tf.image.crop_to_bounding_box(self.imgs, int(self.i0), int(self.j0), int(self.i1 - self.i0), int(self.j1 - self.j0))
            images = self.imgs - mean
            images = tf.image.crop_to_bounding_box(images, int(self.i0), int(self.j0), int(self.i1 - self.i0), int(self.j1 - self.j0))
            self.images_p = images

        with tf.name_scope('preprocess_2'):
            mean = tf.constant([109.31410628, 109.31410628, 109.31410628], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            std = 0.013126239125505505
            images = self.imgs - mean
            images = images * std
            images = tf.image.crop_to_bounding_box(images, int(self.i0), int(self.j0), int(self.i1 - self.i0), int(self.j1 - self.j0))
            self.images_p1 = images

        with tf.name_scope('preprocess_3'):
            mean = tf.constant([109.31410628, 109.31410628, 109.31410628], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            std = 0.013126239125505505
            self.rh, self.rw, images_p2 = self.define_scale3_training_crop()
            images_p2 = images_p2 - mean
            self.images_p2 = images_p2 * std

        with tf.name_scope('vgg'):
            with tf.name_scope('imagenet'):
                self.conv1_1 = self.conv_layer(self.images_p, imagenet_stack[0], padding='VALID')
                self.conv1_2 = self.conv_layer(self.conv1_1, imagenet_stack[1], relu_flag=False, padding='VALID')
                self.pool1 = self.max_pool(self.conv1_2, imagenet_stack[2])
                self.pool1 = tf.nn.relu(self.pool1)

                self.conv2_1 = self.conv_layer(self.pool1, imagenet_stack[3])
                self.conv2_2 = self.conv_layer(self.conv2_1, imagenet_stack[4], relu_flag=False)
                self.pool2 = self.max_pool(self.conv2_2, imagenet_stack[5])
                self.pool2 = tf.nn.relu(self.pool2)

                self.conv3_1 = self.conv_layer(self.pool2, imagenet_stack[6])
                self.conv3_2 = self.conv_layer(self.conv3_1, imagenet_stack[7])
                self.conv3_3 = self.conv_layer(self.conv3_2, imagenet_stack[8])
                self.pool3 = self.max_pool(self.conv3_3, imagenet_stack[9])
                self.pool3 = tf.nn.relu(self.pool3)

                self.conv4_1 = self.conv_layer(self.pool3, imagenet_stack[10])
                self.conv4_2 = self.conv_layer(self.conv4_1, imagenet_stack[11])
                self.conv4_3 = self.conv_layer(self.conv4_2, imagenet_stack[12])
                self.pool4 = self.max_pool(self.conv4_3, imagenet_stack[13])
                self.pool4 = tf.nn.relu(self.pool4)

                self.conv5_1 = self.conv_layer(self.pool4, imagenet_stack[14])
                self.conv5_2 = self.conv_layer(self.conv5_1, imagenet_stack[15])
                self.conv5_3 = self.conv_layer(self.conv5_2, imagenet_stack[16])
                self.pool5 = self.max_pool(self.conv5_3, imagenet_stack[17])
                self.pool5 = tf.nn.relu(self.pool5)

                # Pretrained weights are rather large, rescale down to nicer range
                self.pool5_sc = 0.01 * self.pool5
                self.pool5_sc = tf.transpose(self.pool5_sc, [0, 3, 1, 2])  # Damn bug
                self.imnet_feats = tf.reshape(self.pool5_sc, [-1, int(np.prod(self.pool5_sc.get_shape()[1:]))])

            with tf.name_scope('scale1'):
                self.fc1 = self.fc_layer(self.imnet_feats, 'full1')
                self.fc1 = tf.nn.relu(self.fc1)
                self.fc1_drop = tf.nn.dropout(self.fc1, 1)
                # self.fc1_drop = self.fc1
                self.fc1_mean = 0.5 * self.fc1_drop

                self.fc2_drop = self.fc_layer(self.fc1_drop, 'full2', extra='drop')
                self.fc2_mean = self.fc_layer(self.fc1_mean, 'full2', extra='mean')

                self.fc2_drop = tf.nn.relu(self.fc2_drop)
                self.fc2_drop = tf.reshape(self.fc2_drop, self.fc2_feature_size)
                self.fc2_drop = tf.transpose(self.fc2_drop, [0, 2, 3, 1])
                self.fc2_mean = tf.nn.relu(self.fc2_mean)
                self.fc2_mean = tf.reshape(self.fc2_mean, self.fc2_feature_size)
                self.fc2_mean = tf.transpose(self.fc2_mean, [0, 2, 3, 1])

                (fh, fw) = self.scale2_size
                self.fc2_drop_up = self.upsample_bilinear(self.fc2_drop, 4)
                self.fc2_drop_up = tf.image.crop_to_bounding_box(self.fc2_drop_up, 2, 2, self.scale2_size[0], self.scale2_size[1])
                self.fc2_mean_up = self.upsample_bilinear(self.fc2_mean, 4)
                self.fc2_mean_up = tf.image.crop_to_bounding_box(self.fc2_mean_up, 2, 2, self.scale2_size[0], self.scale2_size[1])

            with tf.name_scope('scale2'):
                self.conv_s2_1 = self.conv_layer(self.images_p1, scale2_stack[0], padding='VALID', stride=2)
                self.pool_s2_1 = self.max_pool(self.conv_s2_1, scale2_stack[1], ksize=[1, 3, 3, 1])

                self.p_1_drop = tf.concat([self.fc2_drop_up, self.pool_s2_1], 3, name='concat_scale2')
                self.p_1_mean = tf.concat([self.fc2_mean_up, self.pool_s2_1], 3, name='concat_scale2')

                self.define_scale2_onestack('normals', self.p_1_drop, self.p_1_mean)
                self.define_scale2_onestack('depths', self.p_1_drop, self.p_1_mean)

            with tf.name_scope('scale3'):
                self.conv_s3_1 = self.conv_layer(self.images_p1, 'conv_s3_1', padding='VALID', stride=2)
                self.pool_s3_1 = self.max_pool(self.conv_s3_1, 'pool_s3_1', ksize=[1, 3, 3, 1], stride=[1, 1, 1, 1])

                self.conv_s3_1_crop = self.conv_layer(self.images_p2, 'conv_s3_1', padding='VALID', stride=2)
                self.pool_s3_1_crop = self.max_pool(self.conv_s3_1_crop, 'pool_s3_1', ksize=[1, 3, 3, 1], stride=[1, 1, 1, 1])

                (n_b, _, _, n_c) = self.pred_drop_d_up_crop.shape.as_list()
                self.pred_drop_d_cat = tf.slice(self.pred_drop_d_up_crop, [0, self.rh, self.rw, 0], [n_b, self.scale2_size[0], self.scale2_size[1], n_c])
                (n_b, _, _, n_c) = self.pred_drop_n_up_crop.shape.as_list()
                self.pred_drop_n_cat = tf.slice(self.pred_drop_n_up_crop, [0, self.rh, self.rw, 0], [n_b, self.scale2_size[0], self.scale2_size[1], n_c])
                (n_b, n_h, n_w, n_c) = self.pool_s3_1_crop.shape.as_list()
                self.pool_s3_1_crop_cat = tf.slice(self.pool_s3_1_crop, [0, 0, 0, 4], [n_b, n_h, n_w, n_c - 4])
                (n_b, n_h, n_w, n_c) = self.pool_s3_1.shape.as_list()
                self.pool_s3_1_cat = tf.slice(self.pool_s3_1, [0, 0, 0, 4], [n_b, n_h, n_w, n_c - 4])

                self.p_2_drop = tf.concat([self.pred_drop_d_cat, self.pred_drop_n_cat, self.pool_s3_1_crop_cat], 3, name='concat_scale3')
                self.p_2_mean = tf.concat([self.pred_mean_d_up_crop, self.pred_mean_n_up_crop, self.pool_s3_1_cat], 3, name='concat_scale3')

                self.define_scale3_onestack('normals', self.p_2_drop, self.p_2_mean, self.scale2_size)
                self.define_scale3_onestack('depths', self.p_2_drop, self.p_2_mean, self.scale2_size)


if __name__ == '__main__':
    sess = tf.Session()
    width = config.WIDTH
    height = config.HEIGHT
    imgs = tf.placeholder(tf.float32, [1, height, width, 3])
    sc1 = vgg_depth_normal(imgs, sess, '../dl/depth_normals/depth_normal.npy')

    images = glob.glob('/home/nrupatunga/Pictures/*.jpeg')
    for i, img_path in enumerate(images):
        img1 = Image.open(img_path)
        img1 = img1.resize((width, height), Image.BICUBIC)
        img1 = np.asarray(img1)
        start_time = time.time()
        output = sess.run([sc1.normal_r, sc1.depth_r], feed_dict={sc1.imgs: [img1]})
        __import__('pdb').set_trace()
        s = 'Total time taken = {} ms'.format((time.time() - start_time) * 1000)
        print(s)
