# 'asdasdasd'

"""Random variable transformation classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import helper 
import pdb 
import numpy as np
import math 
import transforms
import distributions
import time
import os
from pathlib import Path
import platform
import subprocess

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import spatial_transformer
from skimage.transform import rescale, resize, downscale_local_mean

plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16

my_dpi = 350
grid_on, ticks_on, axis_on = False, False, True
quality = 0.2
marker_size = 10/3
marker_line = 0.3/10

scale_xx = 1
range_1_min = -scale_xx
range_1_max = scale_xx

def set_axis_prop(ax, grid_on, ticks_on, axis_on):
    ax.grid(grid_on)
    if not ticks_on:
        if hasattr(ax, 'set_xticks'): ax.set_xticks([])
        if hasattr(ax, 'set_yticks'): ax.set_yticks([])
        if hasattr(ax, 'set_zticks'): ax.set_zticks([])
    if not axis_on: ax.axis('off')

def get_full_grid_samples(resolution=100, range_min=-1, range_max=1):
    x0_range = np.linspace(range_min, range_max, 2*resolution+1)
    x1_range = np.linspace(range_min, range_max, 2*resolution+1)
    x0v, x1v = np.meshgrid(x0_range, x1_range)
    grid_flat = np.concatenate([x0v.flatten()[:, np.newaxis], x1v.flatten()[:, np.newaxis]], axis=1)
    grid = np.concatenate([x0v[:,:,np.newaxis], x1v[:,:,np.newaxis]], axis=2)
    return grid_flat, grid, x0_range, x1_range

def get_sparse_grid_samples(resolution=100, subsample_rate=10, range_min=-1, range_max=1):
    full_grid_flat, full_grid, _, _ = get_full_grid_samples(resolution=resolution, range_min=range_min, range_max=range_max)
    index_x0v, index_x1v = np.meshgrid(np.arange(full_grid.shape[0]), np.arange(full_grid.shape[1]))
    index_grid_flat = np.concatenate([index_x0v.flatten()[:, np.newaxis], index_x1v.flatten()[:, np.newaxis]], axis=1)
    index_grid = np.concatenate([index_x0v[:,:,np.newaxis], index_x1v[:,:,np.newaxis]], axis=2)
    subsample_mask = ((index_grid_flat[:, 0]%subsample_rate == 0)+ (index_grid_flat[:, 1]%subsample_rate == 0))>0
    return full_grid_flat[subsample_mask,:]

filter_size = 5
im_size = 50
n_layers = 10

regular_mask_l1_np = np.ones([filter_size, filter_size], np.float32)
regular_mask_l1_np[math.floor(filter_size/2)+1:, :] = 0
regular_mask_l1_np[math.floor(filter_size/2), math.floor(filter_size/2):] = 0
regular_mask_l1 = tf.constant(regular_mask_l1_np, tf.float32)

regular_mask_np = np.ones([filter_size, filter_size], np.float32)
regular_mask_np[math.floor(filter_size/2)+1:, :] = 0
regular_mask_np[math.floor(filter_size/2), math.floor(filter_size/2)+1:] = 0
regular_mask = tf.constant(regular_mask_np, tf.float32)

horizontal_mask_l1 = regular_mask_l1 
horizontal_mask = regular_mask 

vertical_mask_l1_np = np.ones([filter_size, filter_size], np.float32)
vertical_mask_l1_np[math.floor(filter_size/2):, :] = 0
vertical_mask_l1 = tf.constant(vertical_mask_l1_np, tf.float32)

vertical_mask_np = np.ones([filter_size, filter_size], np.float32)
vertical_mask_np[math.floor(filter_size/2)+1:, :] = 0
vertical_mask = tf.constant(vertical_mask_np, tf.float32)

input_im_np = np.zeros([im_size*im_size, im_size, im_size, 1], np.float32)
for i in range(input_im_np.shape[1]):
    for j in range(input_im_np.shape[2]):
        input_im_np[i*input_im_np.shape[2]+j, i, j, :] = 1
input_im = tf.constant(input_im_np, tf.float32)

filter_weights = tf.ones([5, 5, 1, 1], dtype = tf.float32)

filter_regular_mask_l1 = regular_mask_l1[:, :, np.newaxis, np.newaxis]*filter_weights
filter_regular_mask = regular_mask[:, :, np.newaxis, np.newaxis]*filter_weights

filter_horizontal_mask_l1 = horizontal_mask_l1[:, :, np.newaxis, np.newaxis]*filter_weights
filter_horizontal_mask = horizontal_mask[:, :, np.newaxis, np.newaxis]*filter_weights

filter_vertical_mask_l1 = vertical_mask_l1[:, :, np.newaxis, np.newaxis]*filter_weights
filter_vertical_mask = vertical_mask[:, :, np.newaxis, np.newaxis]*filter_weights

for curr_n_layers in range(1,n_layers+1):
    print('Current n layers: ', curr_n_layers)

    conv_out = input_im
    conv_out_vertical = input_im
    conv_out_horizontal = input_im
    for i in range(curr_n_layers):
        if i == 0: 
            conv_out_vertical = tf.math.sign(tf.nn.conv2d(conv_out_vertical, filter_vertical_mask_l1, strides=[1,1,1,1], padding='SAME'))
            conv_out_horizontal = tf.math.sign(conv_out_vertical+tf.nn.conv2d(conv_out_horizontal, filter_horizontal_mask_l1, strides=[1,1,1,1], padding='SAME'))
            # conv_out_regular = tf.math.sign(tf.nn.conv2d(conv_out, filter_regular_mask_l1, strides=[1,1,1,1], padding='SAME'))
            # conv_out = conv_out_regular
            conv_out = conv_out_horizontal
        else:
            conv_out_vertical = tf.math.sign(tf.nn.conv2d(conv_out_vertical, filter_vertical_mask, strides=[1,1,1,1], padding='SAME'))
            conv_out_horizontal = tf.math.sign(conv_out_vertical+tf.nn.conv2d(conv_out_horizontal, filter_horizontal_mask, strides=[1,1,1,1], padding='SAME'))
            # conv_out_regular = tf.math.sign(tf.nn.conv2d(conv_out, filter_regular_mask, strides=[1,1,1,1], padding='SAME'))
            # conv_out = conv_out_regular
            conv_out = conv_out_horizontal

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()  
    sess.run(init)

    input_im_np, conv_out_np = sess.run([input_im, conv_out])

    feature = 0 
    query_pixel = [math.floor(im_size/2), math.floor(im_size/2)]

    actives_masked = conv_out_np[:, query_pixel[0], query_pixel[1], 0]
    querys_effect = conv_out_np[query_pixel[0]*input_im_np.shape[2]+query_pixel[1], :, :, 0]

    query_pixel_image = np.zeros((im_size, im_size))
    query_pixel_image[query_pixel[0], query_pixel[1]] = 1
    cone_masked = np.zeros((im_size, im_size))
    for i in range(input_im_np.shape[1]):
        for j in range(input_im_np.shape[2]):
            cone_masked[i, j] = actives_masked[i*input_im_np.shape[2]+j]

    cone_masked_im = np.concatenate([query_pixel_image[:,:,np.newaxis], cone_masked[:,:,np.newaxis], cone_masked[:,:,np.newaxis]], axis=2)
    plt.imsave(str(Path.home())+'/Pixel_CNN_ims/cone_masked_'+str(curr_n_layers)+'.png', cone_masked_im)

    querys_effect_im = np.concatenate([query_pixel_image[:,:,np.newaxis], querys_effect[:,:,np.newaxis], querys_effect[:,:,np.newaxis]], axis=2)
    plt.imsave(str(Path.home())+'/Pixel_CNN_ims/querys_effect_'+str(curr_n_layers)+'.png', querys_effect_im)
pdb.set_trace()






