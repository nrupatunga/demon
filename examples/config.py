# config.py
# Date: Tuesday 19 June 2018
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Global configuration file

import sys
import os

CAFFE_PATH = '/usr/local/caffe-segnet-cudnn5/python/'
DEPTH_DIR = '../dl/depth_normals/'
SEGNET_DIR = '../dl/segmentation/'

sys.path.insert(0, CAFFE_PATH)
sys.path.append(DEPTH_DIR)
weights_dnl_tf = os.path.join(DEPTH_DIR, 'depth_normal.npy')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'

# Enable/ Disable debug codes
DEBUG = False

# Number of DL processes
NUMBER_OF_PROCESS = 3

# ports for visualization
PORT = 5000
PORT_VIS = 5001

# dimensions of DL depth, segmentation output
WIDTH = 320
HEIGHT = 240

# Colors for printing
W = '\033[0m'  # white (normal)
G = '\033[0m'  # white (normal)
R = '\033[31m'  # red
Y = '\33[33m'

# Thresholds
SCENE_TYPE_NONE_TH = 0.83
SCENE_IN_OUT_TH = 0.018

# number of clusters
NUM_KMEAN_CLUSTERS = 5

# Threshold Confidence for table
TABLE_CONFIDENCE = 0.5

# Depth map compression, 0 = off, 1 = on
COMPRESSION_MODE = 0

# time out
TIMEOUT = 100

# refined median
REFINED_MEDIAN = True

# Slam camera parameter for dims 147x109
fx = 245.803755089 / 2.1769
fy = 245.801752003 / 2.2018
cx = 156.861570037 / 2.1769
cy = 121.329457123 / 2.2018

fxi = 1 / fx
fyi = 1 / fy
cxi = -cx * fxi
cyi = -cy * fyi

# Tensorflow depth/normal model stack
imagenet_stack = ['imnet_conv1_1',
                  'imnet_conv1_2',
                  'imnet_pool1',
                  'imnet_conv2_1',
                  'imnet_conv2_2',
                  'imnet_pool2',
                  'imnet_conv3_1',
                  'imnet_conv3_2',
                  'imnet_conv3_3',
                  'imnet_pool3',
                  'imnet_conv4_1',
                  'imnet_conv4_2',
                  'imnet_conv4_3',
                  'imnet_pool4',
                  'imnet_conv5_1',
                  'imnet_conv5_2',
                  'imnet_conv5_3',
                  'imnet_pool5'
                  ]

scale2_stack = ['conv_s2_1',
                'pool_s2_1'
                ]

# Few fixes added for samsung demo
SAMSUNG_DEMO_FIX = False
# image sent by app taken in portrait/ landscape mode
PORTRAIT_MODE = False

RADIAN_TH = 0.5
