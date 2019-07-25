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
import scipy

from tensorflow.contrib.distributions import RelaxedOneHotCategorical

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


unwarped_path_1 = '/Users/mevlana.gemici/unwarped_small_1.png'
unwarped_path_2 = '/Users/mevlana.gemici/unwarped_small_2.png'
warped_path_1 = '/Users/mevlana.gemici/warped_small_1.png'
warped_path_2 = '/Users/mevlana.gemici/warped_small_2.png'
warped_rot_path_1 = '/Users/mevlana.gemici/warped_rot_small_1.png'
warped_rot_path_2 = '/Users/mevlana.gemici/warped_rot_small_2.png'
unwarped_white_path = '/Users/mevlana.gemici/unwarped_white.png'
warped_white_path = '/Users/mevlana.gemici/warped_white.png'

unwarped_1_np = plt.imread(unwarped_path_1)[:,:,:3]
unwarped_2_np = plt.imread(unwarped_path_2)[:,:,:3]
warped_1_np = plt.imread(warped_path_1)[:,:,:3]
warped_2_np = plt.imread(warped_path_2)[:,:,:3]
warped_rot_1_np = plt.imread(warped_rot_path_1)[:,:,:3]
warped_rot_2_np = plt.imread(warped_rot_path_2)[:,:,:3]
unwarped_white_np = plt.imread(unwarped_white_path)[:,:,:3]
warped_white_np = plt.imread(warped_white_path)[:,:,:3]

def augment_im(input_im):
    aug_im = np.zeros((input_im.shape[0]*2, input_im.shape[1]*2, input_im.shape[2]))
    aug_im[int(input_im.shape[0]/2):int(input_im.shape[0]/2)+input_im.shape[0], int(input_im.shape[1]/2):int(input_im.shape[1]/2)+input_im.shape[1],:] = input_im

    radius = 5
    for i in range(aug_im.shape[0]):
        for j in range(aug_im.shape[1]):
            if (i < int(input_im.shape[0]/2) or i >= int(input_im.shape[0]/2)+input_im.shape[0]) or (j < int(input_im.shape[1]/2) or j >= int(input_im.shape[1]/2)+input_im.shape[1]):
                a_float = float(j)/float((input_im.shape[0]/2))-2
                b_float = -(float(i)/float((input_im.shape[1]/2))-2)
                # print(i,j,a_float,b_float)
                if np.abs(a_float)>=np.abs(b_float):
                    pixel_x = np.sign(a_float)
                    pixel_y = b_float/np.abs(a_float)
                    pixel_j_adj = int(round((1+pixel_x)*(int(input_im.shape[1]/2)-1)))
                    pixel_i_adj = int(round((1-pixel_y)*(int(input_im.shape[1]/2)-1)))
                    min_pixel_i_adj = min(max(0, pixel_i_adj-radius), input_im.shape[0]-1)
                    max_pixel_i_adj = min(max(0, pixel_i_adj+radius), input_im.shape[0]-1)
                    min_pixel_j_adj = min(max(0, pixel_j_adj-radius), input_im.shape[1]-1)
                    max_pixel_j_adj = min(max(0, pixel_j_adj+radius), input_im.shape[1]-1)
                    aug_im[i,j,:] = (1/np.abs(a_float))*np.mean(np.mean(input_im[min_pixel_i_adj:max_pixel_i_adj, min_pixel_j_adj:max_pixel_j_adj,:], axis=0), axis=0)
                else:
                    pixel_x = a_float/np.abs(b_float)
                    pixel_y = np.sign(b_float)
                    pixel_j_adj = int(round((1+pixel_x)*(int(input_im.shape[1]/2)-1)))
                    pixel_i_adj = int(round((1-pixel_y)*(int(input_im.shape[1]/2)-1)))
                    min_pixel_i_adj = min(max(0, pixel_i_adj-radius), input_im.shape[0]-1)
                    max_pixel_i_adj = min(max(0, pixel_i_adj+radius), input_im.shape[0]-1)
                    min_pixel_j_adj = min(max(0, pixel_j_adj-radius), input_im.shape[1]-1)
                    max_pixel_j_adj = min(max(0, pixel_j_adj+radius), input_im.shape[1]-1)
                    aug_im[i,j,:] = (1/np.abs(b_float))*np.mean(np.mean(input_im[min_pixel_i_adj:max_pixel_i_adj, min_pixel_j_adj:max_pixel_j_adj,:], axis=0), axis=0)

    return aug_im

# aug_im = augment_im(unwarped_1_np)
# plt.imshow(aug_im)
# plt.show()
# pdb.set_trace()
assert (unwarped_1_np.shape == warped_1_np.shape)
assert (unwarped_2_np.shape == warped_2_np.shape)
assert (unwarped_1_np.shape == warped_rot_1_np.shape)
assert (unwarped_2_np.shape == warped_rot_2_np.shape)


resolution = 1000
loc_batch_size = 10000
loc_vis_epoch_rate = 15
grid_samples = get_sparse_grid_samples(resolution=resolution, subsample_rate=10, range_min=10*range_1_min, range_max=10*range_1_max)

use_gpu = False 
if platform.dist()[0] == 'Ubuntu': 
    print('On Collab!!!!!')
    use_gpu = True

if use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("os.environ['CUDA_VISIBLE_DEVICES'], ", os.environ['CUDA_VISIBLE_DEVICES'])

exp_dir = str(Path.home())+'/ExperimentalResults/Align_EXP/'
if not os.path.exists(exp_dir): os.makedirs(exp_dir)
if not os.path.exists(exp_dir+'all/'): os.makedirs(exp_di+'all/')
if not os.path.exists(exp_dir+'im/'): os.makedirs(exp_dir+'im/')
if not os.path.exists(exp_dir+'gt/'): os.makedirs(exp_dir+'gt/')
if not os.path.exists(exp_dir+'weig/'): os.makedirs(exp_dir+'weig/')
if not os.path.exists(exp_dir+'inp_ims/'): os.makedirs(exp_dir+'inp_ims/')

n_epochs = 1000
n_updates_per_epoch = 1000
vis_epoch_rate = 1
n_location_samples = 1000 # works well non-amortized


cost_mode = 'regular'
# cost_mode = 'ignore_background_cost'
# cost_mode = 'regularize_background_cost'
# cost_mode = 'ssim multiscale'

# beta1=0.99
# beta2=0.999
# init_learning_rate = 0.000005
# min_learning_rate = 0.000005
# cost_mode = 'regular'
# # cost_mode = 'ignore_background_cost'
# # cost_mode = 'regularize_background_cost'

# im_target_np = np.concatenate([unwarped_1_np[np.newaxis, :, :, :], unwarped_2_np[np.newaxis, :, :, :]], axis=0)
# im_input_np = np.concatenate([warped_rot_1_np[np.newaxis, :, :, :], warped_rot_2_np[np.newaxis, :, :, :]], axis=0)
# im_auxiliary_np = np.concatenate([warped_1_np[np.newaxis, :, :, :], warped_2_np[np.newaxis, :, :, :]], axis=0)

# im_target_np = np.concatenate([unwarped_1_np[np.newaxis, :, :, :], unwarped_2_np[np.newaxis, :, :, :]], axis=0)
# im_input_np = np.concatenate([warped_1_np[np.newaxis, :, :, :], warped_2_np[np.newaxis, :, :, :]], axis=0)
# im_auxiliary_np = np.concatenate([warped_1_np[np.newaxis, :, :, :], warped_2_np[np.newaxis, :, :, :]], axis=0)
# # # im_target_np = np.concatenate([unwarped_1_np[np.newaxis, :, :, :], unwarped_1_np[np.newaxis, :, :, :]], axis=0)
# # # im_input_np = np.concatenate([warped_rot_1_np[np.newaxis, :, :, :], warped_rot_1_np[np.newaxis, :, :, :]], axis=0)
# # # im_auxiliary_np = np.concatenate([warped_1_np[np.newaxis, :, :, :], warped_1_np[np.newaxis, :, :, :]], axis=0)

im_target_np = unwarped_2_np[np.newaxis, :, :, :]
im_input_np = warped_rot_2_np[np.newaxis, :, :, :]
im_auxiliary_np = warped_2_np[np.newaxis, :, :, :]

# im_target_np = unwarped_1_np[np.newaxis, :, :, :]
# im_input_np = warped_rot_1_np[np.newaxis, :, :, :]
# im_auxiliary_np = warped_1_np[np.newaxis, :, :, :]

# im_target_np = unwarped_2_np[np.newaxis, :, :, :]
# im_input_np = warped_2_np[np.newaxis, :, :, :]
# im_auxiliary_np = warped_2_np[np.newaxis, :, :, :]

# im_target_np = unwarped_1_np[np.newaxis, :, :, :]
# im_input_np = warped_1_np[np.newaxis, :, :, :]
# im_auxiliary_np = warped_1_np[np.newaxis, :, :, :]


# im_target_np = unwarped_1_np[np.newaxis, :, :, :]
# im_input_np = augment_im(warped_1_np)[np.newaxis, :, :, :]
# im_auxiliary_np = warped_1_np[np.newaxis, :, :, :]



# im_target_np = unwarped_white_np[np.newaxis, :50, :50, :]
# im_input_np = warped_white_np[np.newaxis, :50, :50, :]
# im_auxiliary_np = warped_white_np[np.newaxis, :50, :50, :]

temperature = tf.placeholder(tf.float32, [])


# im_target = tf.placeholder(tf.float32, [None, None, None, 3])
# im_input = tf.placeholder(tf.float32, [None, None, None, 3])
# location_input_tf = tf.placeholder(tf.float32, [None, 2])
im_target = tf.placeholder(tf.float32, im_target_np.shape)
im_input = tf.placeholder(tf.float32, im_input_np.shape)
location_input_tf = tf.placeholder(tf.float32, [None, 2])

batch_size_np = im_input_np.shape[0]
batch_size_tf = tf.shape(im_input)[0]

n_filters = 64
# nonlinearity = tf.nn.relu
nonlinearity = tf.nn.tanh

if batch_size_np == 1:
    beta1=0.9
    beta2=0.99
    init_learning_rate = 0.00025
    min_learning_rate = 0.00025

    # beta1=0.9
    # beta2=0.5
    # init_learning_rate = 0.0001
    # min_learning_rate = 0.0001
    flow_param_input = tf.ones(shape=(batch_size_np, 1)) # single map

else:
    beta1=0.9
    beta2=0.5
    init_learning_rate = 0.0001
    min_learning_rate = 0.0001

    # beta1=0.9
    # beta2=0.5
    # init_learning_rate = 0.0001
    # min_learning_rate = 0.0001

    # im_both = tf.concat([im_input, im_target], axis=-1)
    # with tf.variable_scope("eft_third", reuse=False):
    #     lay1_image = tf.layers.conv2d(inputs=im_both, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=nonlinearity)
    #     lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=nonlinearity)
    #     lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=nonlinearity)
    #     lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=nonlinearity)
    #     lay5_image = tf.layers.conv2d(inputs=lay4_image, filters=4*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=nonlinearity)
    #     lay6_image = tf.layers.conv2d(inputs=lay5_image, filters=4*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=nonlinearity)
    # flow_param_input = tf.reshape(lay6_image, [-1, lay6_image.get_shape()[1].value*lay6_image.get_shape()[2].value*4*n_filters])

    im_input_small = helper.tf_resize_image(im_input, resize_ratios=[1/8,1/8], mode='bicubic')
    im_target_small = helper.tf_resize_image(im_target, resize_ratios=[1/8,1/8], mode='bicubic')
    im_input_small_90 = tf.image.rot90(im_input_small, k=1)
    im_input_small_180 = tf.image.rot90(im_input_small, k=2)
    im_input_small_270 = tf.image.rot90(im_input_small, k=3)

    im_input_diff = (im_target_small-im_input_small)
    im_input_diff_90 = (im_target_small-im_input_small_90)
    im_input_diff_180 = (im_target_small-im_input_small_180)
    im_input_diff_270 = (im_target_small-im_input_small_270)

    im_input_diff_sq = im_input_diff**2
    im_input_diff_sq_90 = im_input_diff_90**2
    im_input_diff_sq_180 = im_input_diff_180**2
    im_input_diff_sq_270 = im_input_diff_270**2

    im_input_diff_mean_mat = tf.concat([tf.reduce_mean(im_input_diff_sq,axis=[1,2,3])[:,np.newaxis], 
                                        tf.reduce_mean(im_input_diff_sq_90,axis=[1,2,3])[:,np.newaxis], 
                                        tf.reduce_mean(im_input_diff_sq_180,axis=[1,2,3])[:,np.newaxis], 
                                        tf.reduce_mean(im_input_diff_sq_270,axis=[1,2,3])[:,np.newaxis]], axis=-1)

    im_param_input = tf.concat([im_input_small, 
                                im_target_small,
                                im_input_diff,
                                im_input_diff_90,
                                im_input_diff_180,
                                im_input_diff_270,
                                im_input_diff_sq,
                                im_input_diff_sq_90,
                                im_input_diff_sq_180,
                                im_input_diff_sq_270,
                                ], axis=-1)

    with tf.variable_scope("eft_third", reuse=False):
        lay1_image = tf.layers.conv2d(inputs=im_param_input, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=nonlinearity)
        lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=nonlinearity)
        lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=4*n_filters, kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=nonlinearity)
        lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=4*n_filters, kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=nonlinearity)
        lay5_image = tf.layers.conv2d(inputs=lay4_image, filters=500, kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=nonlinearity)

    flow_param_input = tf.reshape(lay5_image, [-1, 500])
    # flow_param_input = tf.reshape(lay3_image, [-1, lay3_image.get_shape()[1].value*lay3_image.get_shape()[2].value*4*n_filters])

with tf.variable_scope("eft_fourth", reuse=False):
    if batch_size_np == 1:
        rot_weights_logits = tf.layers.dense(inputs = flow_param_input, units = 4, use_bias = False, activation = None)
    else:
        rot_weights_logits = tf.layers.dense(inputs = flow_param_input, units = 4, use_bias = True, activation = None)

    # rot_weights_logits = 2*tf.layers.dense(inputs = flow_param_input, units = 4, use_bias = False, activation = tf.nn.sigmoid)
    # rot_weights_logits = tf.layers.dense(inputs = 1/im_input_diff_mean_mat, units = 4, use_bias = False, activation = None)
    # pdb.set_trace()

# rot_weights_logits = 1/im_input_diff_mean_mat
    
    # rot_weights_logits = tf.layers.dense(inputs = flow_param_input, units = 4, use_bias = True, activation = None)

# # im_both = tf.concat([helper.tf_resize_image(im_input, resize_ratios=[1/16,1/16]), helper.tf_resize_image(im_target, resize_ratios=[1/16,1/16])], axis=-1)
# # with tf.variable_scope("eft_third", reuse=False):
# #     lay1_image = tf.layers.conv2d(inputs=im_both, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=nonlinearity)
# #     lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", use_bias=True, activation=nonlinearity)
# #     lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=4*n_filters, kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=nonlinearity)
# # flow_param_input_2 = tf.reshape(lay3_image, [-1, lay3_image.get_shape()[1].value*lay3_image.get_shape()[2].value*4*n_filters])

rot_weights_temperature = temperature 
dist = RelaxedOneHotCategorical(rot_weights_temperature, logits=rot_weights_logits)
rot_weights = dist.sample()

im_input_90 = tf.image.rot90(im_input, k=1)
im_input_180 = tf.image.rot90(im_input, k=2)
im_input_270 = tf.image.rot90(im_input, k=3)
im_input_all_rotations = tf.concat([im_input[:,:,:,:,np.newaxis], im_input_90[:,:,:,:,np.newaxis], im_input_180[:,:,:,:,np.newaxis], im_input_270[:,:,:,:,np.newaxis]], axis=-1)
im_input_attended = tf.reduce_sum(rot_weights[:, np.newaxis, np.newaxis, np.newaxis, :]*im_input_all_rotations, axis=-1) 

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# fd = {im_input: im_input_np, im_target: im_target_np, temperature:0.2}
# rot_weights_np, im_input_attended_np = sess.run([rot_weights, im_input_attended], feed_dict=fd)
# rot_weights_np, rot_weights_logits_np = sess.run([rot_weights, rot_weights_logits], feed_dict=fd)
# np.concatenate([rot_weights_np[0, :, np.newaxis],rot_weights_logits_np[0, :, np.newaxis], scipy.special.softmax(rot_weights_logits_np[0, :, np.newaxis])], axis=1)
# pdb.set_trace()



################################################################################################## WORKS WELL FOR SINGLE IMAGES ROTATED
# n_dim = 2
# n_flows = 5
# flow_class_1 = transforms.ProperIsometricFlow
# # flow_class_1 = transforms.NotManyReflectionsRotationFlow
# # flow_class_2 = transforms.NonLinearIARFlow 
# flow_class_2 = transforms.ProperIsometricFlow

# flow_class_1_parameters_list, flow_class_2_parameters_list = [], []
# for i in range(n_flows):
#     if flow_class_1.required_num_parameters(n_dim) > 0: 
#         with tf.variable_scope("eft_first"+str(i), reuse=False):
#             flow_class_1_parameters = 1*tf.layers.dense(inputs = flow_param_input, units = flow_class_1.required_num_parameters(n_dim), use_bias = False, activation = None)
#             flow_class_1_parameters_list.append(flow_class_1_parameters)

#     if flow_class_2.required_num_parameters(n_dim) > 0:
#         with tf.variable_scope("eft_second"+str(i), reuse=False):
#             flow_class_2_parameters = 1*tf.layers.dense(inputs = flow_param_input, units = flow_class_2.required_num_parameters(n_dim), use_bias = False, activation = None)
#             flow_class_2_parameters_list.append(flow_class_2_parameters)

# serial_flow_list = []
# for j in range(batch_size_np):
#     normalizing_flow_list = []
#     for i in range(n_flows):
#         normalizing_flow_list.append(flow_class_1(input_dim=n_dim, parameters=flow_class_1_parameters_list[i][j, np.newaxis, :]))
#         nf_param = flow_class_2_parameters_list[i][j, np.newaxis, :]
#         normalizing_flow_list.append(flow_class_2(input_dim=n_dim, parameters=flow_class_2_parameters_list[i][j, np.newaxis, :]))
#     serial_flow_list.append(transforms.SerialFlow(normalizing_flow_list))

#########################################################################################################

n_dim = 2
flow_class_list = [*[transforms.ProperIsometricFlow]*2,
                   # *[transforms.InverseOrderDimensionFlow, transforms.NonLinearIARFlow]*6, # make sure it is even
                   # *[transforms.NotManyReflectionsRotationFlow, transforms.NonLinearIARFlow]*10, # make sure it is even
                   *[transforms.ProperIsometricFlow]*2,
]              
flow_class_list = [transforms.Affine2DFlow]              

flow_parameters_list = [None]*len(flow_class_list)

for i in range(len(flow_class_list)):
    curr_flow_class = flow_class_list[i]
    if curr_flow_class.required_num_parameters(n_dim) > 0: 

        if curr_flow_class == transforms.ProperIsometricFlow or curr_flow_class == transforms.Affine2DFlow:
            scope_name = "eft_first"+str(i)
        elif curr_flow_class == transforms.NonLinearIARFlow or curr_flow_class == transforms.NotManyReflectionsRotationFlow:
            scope_name = "eft_second"+str(i)
        else: pdb.set_trace()

        with tf.variable_scope(scope_name, reuse=False):
            if batch_size_np == 1:
                flow_parameters_list[i] = 1*tf.layers.dense(inputs = flow_param_input, units = curr_flow_class.required_num_parameters(n_dim), use_bias = False, activation = None)
            else:
                flow_parameters_list[i] = 1*tf.layers.dense(inputs = flow_param_input, units = curr_flow_class.required_num_parameters(n_dim), use_bias = True, activation = None)

serial_flow_list = []
for j in range(batch_size_np):
    normalizing_flow_list = []
    for i in range(len(flow_class_list)):
        curr_param = None
        if flow_parameters_list[i] is not None:
            curr_param = flow_parameters_list[i][j, np.newaxis, :]
        
        curr_flow_class = flow_class_list[i]
        normalizing_flow_list.append(curr_flow_class(input_dim=n_dim, parameters=curr_param))
        if curr_flow_class == transforms.NonLinearIARFlow:
            normalizing_flow_list.append(transforms.GeneralInverseFlow(transform=normalizing_flow_list[-2]))
    serial_flow_list.append(transforms.SerialFlow(normalizing_flow_list))

################################################################################################################################################################


def nonlinear_pixel_transformation_clousure(batch_input_pixels, b_inverse=False):
    # input = [tf.shape(input_pixels)[0] (corresponding the individual pixels in a grid of output image), 2]
    # output = [batch_size_tf, 2, tf.shape(input_pixels)[0]]
    # serial_flow does not allow different transformations for each image in the batch
    # See the note in linear_pixel_transformation_clousure for understanding the transformation.
    
    output_pixels_list = []
    for j in range(batch_size_np):
        if b_inverse: curr_output_pixels, _ = serial_flow_list[j].inverse_transform(batch_input_pixels[j,:,:], None)
        else: curr_output_pixels, _ = serial_flow_list[j].transform(batch_input_pixels[j,:,:], None)
        output_pixels_list.append(curr_output_pixels[np.newaxis,:,:])
    batch_output_pixels = tf.concat(output_pixels_list, axis=0)
    return batch_output_pixels

location_input_tiled = tf.tile(location_input_tf[np.newaxis, :, :], [batch_size_np, 1, 1])
location_transformed_tf = nonlinear_pixel_transformation_clousure(location_input_tiled, b_inverse=True)

################################################################################################################################################################

im_transformed, vis_im_transformed, im_target_gathered, invalid_map, _ = spatial_transformer.transformer(input_im=im_input_attended, pixel_transformation_clousure=nonlinear_pixel_transformation_clousure, 
                                                                              out_size=[tf.shape(im_input)[1], tf.shape(im_input)[2]], 
                                                                              n_location_samples=None, out_comparison_im=im_target)


im_transformed_sampled, vis_im_transformed_sampled, im_target_gathered_sampled, invalid_map, location_mask = spatial_transformer.transformer(input_im=im_input, pixel_transformation_clousure=nonlinear_pixel_transformation_clousure, 
                                                                                                              out_size=[tf.shape(im_input)[1], tf.shape(im_input)[2]], 
                                                                                                              n_location_samples=n_location_samples, out_comparison_im=im_target)

im_transformed_sampled_90, vis_im_transformed_sampled_90, _, _, _ = spatial_transformer.transformer(input_im=im_input_90, pixel_transformation_clousure=nonlinear_pixel_transformation_clousure, 
                                                                                                              out_size=[tf.shape(im_input)[1], tf.shape(im_input)[2]], 
                                                                                                              n_location_samples=n_location_samples, out_comparison_im=im_target)

im_transformed_sampled_180, vis_im_transformed_sampled_180, _, _, _ = spatial_transformer.transformer(input_im=im_input_180, pixel_transformation_clousure=nonlinear_pixel_transformation_clousure, 
                                                                                                              out_size=[tf.shape(im_input)[1], tf.shape(im_input)[2]], 
                                                                                                              n_location_samples=n_location_samples, out_comparison_im=im_target)

im_transformed_sampled_270, vis_im_transformed_sampled_270, _, _, _ = spatial_transformer.transformer(input_im=im_input_270, pixel_transformation_clousure=nonlinear_pixel_transformation_clousure, 
                                                                                                              out_size=[tf.shape(im_input)[1], tf.shape(im_input)[2]], 
                                                                                                              n_location_samples=n_location_samples, out_comparison_im=im_target)

im_error_raw = (im_transformed-im_target)
im_error_norm = tf.reduce_sum(im_error_raw**2, axis=-1, keep_dims=True)
im_error_min = tf.reduce_min(im_error_norm, axis=[1,2], keep_dims=True)
im_error_max = tf.reduce_max(im_error_norm, axis=[1,2], keep_dims=True)
im_error_norm_01 = (im_error_norm-im_error_min)/(im_error_max-im_error_min)
im_error = tf.concat([im_error_norm_01, 1-im_error_norm_01, 1-im_error_norm_01], axis=-1)

if cost_mode == 'regular':
    # cost = tf.reduce_mean((im_transformed_sampled-im_target_gathered_sampled)**2)
    
    # im_transformed_sampled_attended = tf.reduce_sum(rot_weights[:, np.newaxis, np.newaxis, :]*tf.concat([im_transformed_sampled[:, :, :, np.newaxis], im_transformed_sampled_90[:, :, :, np.newaxis], im_transformed_sampled_180[:, :, :, np.newaxis], im_transformed_sampled_270[:, :, :, np.newaxis]], axis=-1), axis=-1) 
    # cost = tf.reduce_mean((im_transformed_sampled_attended-im_target_gathered_sampled)**2)
    
    diffe = tf.concat([((im_transformed_sampled-im_target_gathered_sampled)**2)[:, :, :, np.newaxis], 
                       ((im_transformed_sampled_90-im_target_gathered_sampled)**2)[:, :, :, np.newaxis],
                       ((im_transformed_sampled_180-im_target_gathered_sampled)**2)[:, :, :, np.newaxis],
                       ((im_transformed_sampled_270-im_target_gathered_sampled)**2)[:, :, :, np.newaxis]], axis=-1)
    
    diffe_mean = tf.reduce_mean(diffe, axis=[1,2])
    cost =  tf.reduce_mean(tf.reduce_sum(rot_weights*diffe_mean,axis=-1))
elif cost_mode == 'ignore_background_cost':
    cost = tf.reduce_mean(((im_transformed_sampled-im_target_gathered_sampled)*tf.stop_gradient(1-invalid_map))**2)
elif cost_mode == 'regularize_background_cost':
    cost = tf.reduce_mean((im_transformed_sampled-im_target_gathered_sampled)**2)+100*tf.reduce_mean(invalid_map)
elif cost_mode == 'ssim multiscale':
    cost =  tf.reduce_mean(tf.image.ssim_multiscale(im_transformed, im_target, max_val=255, power_factors=(0.1,0.2,0.4,0.2,0.1), filter_size=3))

first_vars = [v for v in tf.trainable_variables() if 'first' in v.name]
second_vars = [v for v in tf.trainable_variables() if 'second' in v.name] 
eft_vars = [v for v in tf.trainable_variables() if 'eft' in v.name] 

global_step = tf.Variable(0.0, name='global_step', trainable=False)
# opt_step_first = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(
#                  learning_rate=init_learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08), 
#                  loss=cost, var_list=first_vars, global_step=global_step, clip_param=5)
# opt_step_second = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(
#                  learning_rate=init_learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08), 
#                  loss=cost, var_list=second_vars, global_step=global_step, clip_param=5)
opt_step_eft = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(
                 learning_rate=init_learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08), 
                 loss=cost, var_list=eft_vars, global_step=global_step, clip_param=2)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()  
sess.run(init)

plt.imsave(exp_dir+'im_input_np'+'.png', im_input_np[0, :, :, :])
plt.imsave(exp_dir+'im_target_np'+'.png', im_target_np[0, :, :, :])

switch = 0

print('Start Timer: ')
start = time.time();
temperature_raw_np = 0.5
temperature_np = min(1, max(0, temperature_raw_np))
for epoch in range(1, n_epochs+1): 
    if temperature_np > 0.3:
        if epoch % 1 == 0:
            temperature_raw_np = temperature_raw_np - 0.05
            temperature_np = min(1, max(1e-05, temperature_raw_np))
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! temperature_np: ', temperature_np)
    else:
        if epoch % 5 == 4:
            temperature_raw_np = temperature_raw_np - 0.025
            temperature_np = min(1, max(1e-05, temperature_raw_np))
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! temperature_np: ', temperature_np)

    if epoch < switch: opt_step = opt_step_first
    else: opt_step = opt_step_eft

    learning_rate = init_learning_rate
    print('Current learning rate: ', learning_rate)
    # if epoch == 1 or epoch == n_epochs or epoch % vis_epoch_rate == 0: 
    if epoch == n_epochs or epoch % vis_epoch_rate == 0: 
        print('Eval and Visualize: Epoch, Time: {:d} {:.3f}'.format(epoch, time.time()-start))
        fd = {im_input: im_input_np, im_target: im_target_np, temperature:temperature_np}
        im_transformed_np, im_error_np, location_mask_np, rot_weights_np, rot_weights_logits_np = sess.run([vis_im_transformed, im_error, location_mask, rot_weights, rot_weights_logits], feed_dict=fd)
        location_mask_np_tiled = np.tile(location_mask_np.astype(np.float32)[np.newaxis,:,:,np.newaxis], [im_transformed_np.shape[0], 1, 1, 3])
        rot_weights_probs_np = scipy.special.softmax(rot_weights_logits_np, axis=-1)

        # im_input_small_np, im_input_small_90_np, im_input_small_180_np, im_input_small_270_np, im_target_small_np, im_input_diff_mean_mat_np, diffe_mean_np = sess.run([im_input_small, im_input_small_90, im_input_small_180, im_input_small_270, im_target_small, im_input_diff_mean_mat, diffe_mean], feed_dict=fd)

        # im_input_small_all = np.concatenate(np.split(im_input_small_np, im_input_small_np.shape[0], axis=0), axis=1)[0] 
        # im_input_small_90_all = np.concatenate(np.split(im_input_small_90_np, im_input_small_90_np.shape[0], axis=0), axis=1)[0] 
        # im_input_small_180_all = np.concatenate(np.split(im_input_small_180_np, im_input_small_180_np.shape[0], axis=0), axis=1)[0] 
        # im_input_small_270_all = np.concatenate(np.split(im_input_small_270_np, im_input_small_270_np.shape[0], axis=0), axis=1)[0] 
        # im_target_small_all = np.concatenate(np.split(im_target_small_np, im_target_small_np.shape[0], axis=0), axis=1)[0] 
        # im_input_small_all_all = np.concatenate([im_input_small_all, im_input_small_90_all, im_input_small_180_all, im_input_small_270_all, im_target_small_all], axis=1)
        # plt.imsave(exp_dir+'inp_ims/inp_ims_'+str(epoch)+'.png', im_input_small_all_all)

        im_transformed_all_list, in_out_target_list, weig_all_list = [], [], []
        for i in range(batch_size_np):
            im_transformed_all_list.append(im_transformed_np[i, :, :, :])
            in_out_target_list.append(np.concatenate([im_input_np[i, :, :, :], im_transformed_np[i, :, :, :], im_target_np[i, :, :, :], im_error_np[i, :, :, :], location_mask_np_tiled[0, :, :, :]], axis=1)) 

        im_transformed_all = np.concatenate(im_transformed_all_list, axis=0)
        in_out_target = np.concatenate(in_out_target_list, axis=0)
        plt.imsave(exp_dir+'im/im_transformed_np_'+str(epoch)+'.png', im_transformed_all)
        plt.imsave(exp_dir+'all/all_im_transformed_np_'+str(epoch)+'.png', in_out_target)


        fig, ax = plt.subplots(figsize=(7*3, 7*batch_size_np))
        plt.clf()
        for i in range(batch_size_np):
            ax_curr = fig.add_subplot(batch_size_np, 3, 3*i+1)
            ax_curr.bar(np.arange(4), rot_weights_probs_np[i,:])
            ax_curr.set_ylim(0, 1)
            ax_curr = fig.add_subplot(batch_size_np, 3, 3*i+2)
            ax_curr.bar(np.arange(4), rot_weights_np[i,:])
            ax_curr.set_ylim(0, 1)
            ax_curr = fig.add_subplot(batch_size_np, 3, 3*i+3)
            # ax_curr.bar(np.arange(4), im_input_diff_mean_mat_np[i,:])
            ax_curr.bar(np.arange(4), rot_weights_np[i,:])

            print('weights:', i)
            print('rot_weights_probs_np: ', rot_weights_probs_np[i,:])
            print('rot_weights_np: ', rot_weights_np[i,:])
            # print('diffe_mean_np: ', diffe_mean_np[i,:])
            print('\n\n')
        plt.savefig(exp_dir+'weig/weig_'+str(epoch)+'.png', bbox_inches='tight', format='png', dpi=int(quality*my_dpi), transparent=False)


    # if epoch == 1 or epoch == n_epochs or epoch % loc_vis_epoch_rate == 0: 
    if epoch == n_epochs or epoch % loc_vis_epoch_rate == 0: 
        all_transformed_grid_samples_np = np.zeros([batch_size_np, *grid_samples.shape])
        for i in range(math.ceil(grid_samples.shape[0]/float(loc_batch_size))):
            curr_batch_np = grid_samples[i*loc_batch_size:min((i+1)*loc_batch_size, grid_samples.shape[0]), :]
            fd = {im_input: im_input_np, im_target: im_target_np, location_input_tf: curr_batch_np, temperature:temperature_np}
            location_transformed_np = sess.run(location_transformed_tf, feed_dict=fd)
            all_transformed_grid_samples_np[:, i*loc_batch_size:min((i+1)*loc_batch_size, grid_samples.shape[0]), :] = location_transformed_np
            
        gt_list = []
        for i in range(batch_size_np):
            fig, ax = plt.subplots(figsize=(7*5, 7*1))
            plt.clf()
            
            rot_weights_probs_np[i,:], rot_weights_np[i,:] 
            ax_1 = fig.add_subplot(1, 5, 1)
            ax_1.scatter(grid_samples[:, 1], -grid_samples[:, 0], s=marker_size, lw = marker_line, edgecolors='k')
            ax_2 = fig.add_subplot(1, 5, 2)
            ax_2.scatter(all_transformed_grid_samples_np[i, :, 1], -all_transformed_grid_samples_np[i, :, 0], s=marker_size, lw = marker_line, edgecolors='k')
            ax_3 = fig.add_subplot(1, 5, 3)
            ax_3.scatter(grid_samples[:, 1], -grid_samples[:, 0], s=marker_size, lw = marker_line, edgecolors='k')
            ax_4 = fig.add_subplot(1, 5, 4)
            ax_4.bar(np.arange(4), rot_weights_probs_np[i,:]) # softmax of logits
            ax_4.set_ylim(0, 1)
            ax_5 = fig.add_subplot(1, 5, 5)
            ax_5.bar(np.arange(4), rot_weights_np[i,:]) # actual sample = softmax of logits+gumbel noise
            ax_5.set_ylim(0, 1)

            for ax in [ax_1, ax_2, ax_3]:
                ax.set_xlim(range_1_min, range_1_max)
                ax.set_ylim(range_1_min, range_1_max)
                set_axis_prop(ax, grid_on, ticks_on, axis_on)

            for ax in [ax_4, ax_5]:
                set_axis_prop(ax, grid_on, ticks_on, axis_on)

            plt.subplots_adjust(wspace=0.02, hspace=0.02)
            plt.draw()
            plt.savefig(exp_dir+'gt/gt_temp.png', bbox_inches='tight', format='png', dpi=int(quality*my_dpi), transparent=False)
            tmp_image = plt.imread(exp_dir+'gt/gt_temp.png')[:,:,:3]
            scaled_tmp_image = rescale(tmp_image, (in_out_target.shape[0]/(batch_size_np*tmp_image.shape[0]), in_out_target.shape[1]/tmp_image.shape[1]), anti_aliasing=False)
            gt_list.append(np.concatenate([scaled_tmp_image, in_out_target[int(i*in_out_target.shape[0]/batch_size_np):int((i+1)*in_out_target.shape[0]/batch_size_np), ...]], axis=0))
        plt.imsave(exp_dir+'gt/gt_'+str(epoch)+'.png',  np.concatenate(gt_list, axis=0))

    for i in range(1, n_updates_per_epoch+1):
        fd = {im_input: im_input_np, im_target: im_target_np, temperature:temperature_np}
        _, cost_np = sess.run([opt_step, cost], feed_dict=fd)
        # if epoch > switch and i == 1:
        #     nf_param_np = sess.run(nf_param, feed_dict=fd)
        #     print('min: ', nf_param_np.min(), 'max: ', nf_param_np.max(), 'mean: ', nf_param_np.mean(), 'std: ', nf_param_np.std())
    print('Epoch: '+str(epoch)+' Update: '+str(i)+ ' Cost: '+str(cost_np))










# n_dim = 2
# n_flows = 10
# normalizing_flow_list = []
# flow_class_1 = transforms.ProperIsometricFlow
# flow_class_2 = transforms.NonLinearIARFlow 
# for i in range(n_flows):
#     flow_class_1_parameters = None
#     if flow_class_1.required_num_parameters(n_dim) > 0: 
#         with tf.variable_scope("eft_first"+str(i), reuse=False):
#             flow_class_1_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_1.required_num_parameters(n_dim), use_bias = False, activation = None)
#     normalizing_flow_list.append(flow_class_1(input_dim=n_dim, parameters=flow_class_1_parameters))

#     flow_class_2_parameters = None
#     if flow_class_2.required_num_parameters(n_dim) > 0:
#         with tf.variable_scope("eft_second"+str(i), reuse=False):
#             flow_class_2_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_2.required_num_parameters(n_dim), use_bias = False, activation = None)
#     normalizing_flow_list.append(flow_class_2(input_dim=n_dim, parameters=flow_class_2_parameters))
# serial_flow = transforms.SerialFlow(normalizing_flow_list)

# def nonlinear_pixel_transformation_clousure(batch_input_pixels, b_inverse=False):
#     # input = [tf.shape(input_pixels)[0] (corresponding the individual pixels in a grid of output image), 2]
#     # output = [batch_size_tf, 2, tf.shape(input_pixels)[0]]
#     # serial_flow does not allow different transformations for each image in the batch
#     # See the note in linear_pixel_transformation_clousure for understanding the transformation.
    
#     input_pixels_flat = tf.reshape(batch_input_pixels, [-1, batch_input_pixels.get_shape()[-1].value])
#     if b_inverse: output_pixels_flat, _ = serial_flow.inverse_transform(input_pixels_flat, None)
#     else: output_pixels_flat, _ = serial_flow.transform(input_pixels_flat, None)
#     batch_output_pixels = tf.reshape(output_pixels_flat, tf.shape(batch_input_pixels))
#     return batch_output_pixels

# location_input_tiled = tf.tile(location_input_tf[np.newaxis, :, :], [im_input_np.shape[0], 1, 1])
# location_transformed_tf = nonlinear_pixel_transformation_clousure(location_input_tiled, b_inverse=True)



# weighted_scaled_costs = []
# # for (downsample_rate, weight) in [(0.5, 1.0), (0.25, 1.0), (0.10, 1.0)]:
# for (downsample_rate, weight) in [(0.10, 1.0)]:
#     new_height = tf.cast(tf.cast(tf.shape(im_input)[1], tf.float32)*downsample_rate, tf.int32)
#     new_width = tf.cast(tf.cast(tf.shape(im_input)[2], tf.float32)*downsample_rate, tf.int32)
#     im_input_rescaled = tf.image.resize_images(im_input, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False, preserve_aspect_ratio=False)
#     im_target_rescaled = tf.image.resize_images(im_target, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False, preserve_aspect_ratio=False)

#     im_scaled_transformed_sampled, _, im_scaled_target_gathered_sampled, _ = spatial_transformer.transformer(input_im=im_input_rescaled, pixel_transformation_clousure=nonlinear_pixel_transformation_clousure, 
#                                                                              out_size=[new_height, new_width], n_location_samples=n_location_samples, out_comparison_im=im_target_rescaled)
#     weighted_scaled_costs.append(weight*tf.reduce_mean((im_scaled_transformed_sampled-im_scaled_target_gathered_sampled)**2))






# if ignore_background_cost:
#     cost = tf.reduce_mean(((im_transformed_sampled-im_target_gathered_sampled)*tf.stop_gradient(1-invalid_map))**2) #+tf.add_n(weighted_scaled_costs)
# else:
#     cost = tf.reduce_mean((im_transformed_sampled-im_target_gathered_sampled)**2) #+tf.add_n(weighted_scaled_costs)




# def linear_pixel_transformation_clousure(batch_input_pixels, b_inverse=False):
#     # input = [tf.shape(input_pixels)[0] (corresponding the individual pixels in a grid of output image), 2]
#     # output = [batch_size_tf, 2, tf.shape(input_pixels)[0]]
#     # affine_matrix = [batch_size_tf, 2, 3] allows different transformations for each image in the batch

#     zoom = 1.0
#     angle = 10 

#     # NOTE: Since the transformation is transforming the coordinates of output grid --> input grid, input grid --> output grid is the inverse
#     # of this transformation. Therefore, in order to increase the scale of the input image and zoom, we must scale with 1/zoom and in order to rotate the 
#     # input image by 10 degrees counterclockwise, we must rotate the output grid by -10 degrees counterclockwise. 

#     z_sin = tf.zeros((batch_size_tf, 1), tf.float32)+np.sin(-angle*np.pi/180)
#     z_cos = tf.zeros((batch_size_tf, 1), tf.float32)+np.cos(-angle*np.pi/180)
#     dummy_zero = tf.zeros([batch_size_tf, 1], tf.float32) # translation can replace it
#     affine_matrix = tf.concat([tf.concat([(1/zoom)*z_cos[:,:,np.newaxis], -(1/zoom)*z_sin[:,:,np.newaxis], dummy_zero[:,:,np.newaxis]], axis=2), 
#                                tf.concat([(1/zoom)*z_sin[:,:,np.newaxis], (1/zoom)*z_cos[:,:,np.newaxis], dummy_zero[:,:,np.newaxis]], axis=2)], axis=1)

#     input_pixels_aug = tf.concat([input_pixels, tf.ones([tf.shape(input_pixels)[0], 1])], axis=1)
#     input_pixels_tiled_transposed = tf.transpose(tf.tile(input_pixels_aug[np.newaxis,:,:], [batch_size_tf, 1, 1]), perm=[0, 2, 1])
#     output_pixels = tf.matmul(affine_matrix, input_pixels_tiled_transposed)
#     return output_pixels



