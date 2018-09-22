# config.py
# Date: Tuesday 19 June 2018
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Global configuration file

import sys
import os

DEPTH_DIR = '../dl/depth_normals/'
sys.path.append(DEPTH_DIR)
weights_dnl_tf = os.path.join(DEPTH_DIR, 'depth_normal.npy')

# Enable/ Disable debug codes
DEBUG = False

# dimensions of DL depth, segmentation output
WIDTH = 320
HEIGHT = 240

# RGB intrinsic parameters for 640x480
fx = 5.1885790117450188e+02
fy = 5.1946961112127485e+02
cx = 3.2558244941119034e+02
cy = 2.5373616633400465e+02

fxi = 1 / fx
fyi = 1 / fy
cxi = -cx * fxi
cyi = -cy * fyi
