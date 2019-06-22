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

scale_xx = 3
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


# unwarped_max = -1000
# warped_max = -1000
# for i in range(location_np.shape[0]):
#     for j in range(location_np.shape[1]):
        
#         index_loc = -1 + unwarped_withlocation_np[i,j,3:5].astype(int)
#         curr_max = (np.abs(unwarped_withlocation_np[i,j,:3]-unwarped_np[index_loc[0], index_loc[1],:]).max())
#         if curr_max > unwarped_max: unwarped_max = curr_max
        
#         index_loc = -1 + warped_withlocation_np[i,j,3:5].astype(int)
#         curr_max = (np.abs(warped_withlocation_np[i,j,:3]-warped_np[index_loc[0], index_loc[1],:]).max())
#         if curr_max > warped_max: warped_max = curr_max
# print(unwarped_max, warped_max)

# plt.imshow(unwarped_np)
# plt.show()

unwarped_path = '/Users/mevlana.gemici/unwarped.png'
warped_path = '/Users/mevlana.gemici/warped.png'
unwarped_np = plt.imread(unwarped_path)[:,:,:3]
warped_np = plt.imread(warped_path)[:,:,:3]
assert (warped_np.shape == unwarped_np.shape)

# x_range = np.linspace(0, unwarped_np.shape[1]-1, unwarped_np.shape[1])
# y_range = np.linspace(0, unwarped_np.shape[0]-1, unwarped_np.shape[0])
# xv, yv = np.meshgrid(x_range, y_range)
# location_np = np.concatenate([yv[:, :, np.newaxis], xv[:, :, np.newaxis]], axis=2)+1
# unwarped_withlocation_np = np.concatenate([unwarped_np, location_np], axis=2)
# warped_withlocation_np = np.concatenate([warped_np, location_np], axis=2)

# unwarped_withlocation_flat_np = unwarped_withlocation_np.reshape(-1, unwarped_withlocation_np.shape[2])
# warped_withlocation_flat_np = warped_withlocation_np.reshape(-1, warped_withlocation_np.shape[2])


use_gpu = False 
if platform.dist()[0] == 'Ubuntu': 
    print('On Collab!!!!!')
    use_gpu = True

if use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("os.environ['CUDA_VISIBLE_DEVICES'], ", os.environ['CUDA_VISIBLE_DEVICES'])

exp_dir = str(Path.home())+'/ExperimentalResults/Align_EXP/'
if not os.path.exists(exp_dir): os.makedirs(exp_dir)

n_epochs = 10
n_updates_per_epoch = 3
vis_epoch_rate = 1
n_location_samples = 100

beta1=0.99
beta2=0.999
init_learning_rate = 0.00025
min_learning_rate = 0.00025

im_input_np = np.tile(warped_np[np.newaxis, :, :, :], [1, 1, 1, 1])
im_target_np = np.tile(unwarped_np[np.newaxis, :, :, :], [1, 1, 1, 1])

im_input = tf.placeholder(tf.float32, [None, None, None, 3])
im_target = tf.placeholder(tf.float32, [None, None, None, 3])
batch_size_tf = tf.shape(im_input)[0]

def linear_pixel_transformation_clousure(input_pixels): 
    # input = [tf.shape(input_pixels)[0] (corresponding the individual pixels in a grid of output image), 2]
    # output = [batch_size_tf, 2, tf.shape(input_pixels)[0]]
    # affine_matrix = [batch_size_tf, 2, 3] allows different transformations for each image in the batch

    zoom = 1.0
    angle = 90
    z_cos = tf.zeros((batch_size_tf, 1), tf.float32)+np.cos(angle*np.pi/180)
    z_sin = helper.safe_tf_sqrt(1-z_cos**2)
    dummy_zero = tf.zeros([batch_size_tf, 1], tf.float32)
    affine_matrix = tf.concat([(1/zoom)*z_cos, -(1/zoom)*z_sin, dummy_zero, (1/zoom)*z_sin, (1/zoom)*z_cos, dummy_zero], axis=1)
    affine_matrix = tf.cast(tf.reshape(affine_matrix, (-1, 2, 3)), 'float32')

    input_pixels_aug = tf.concat([input_pixels, tf.ones([tf.shape(input_pixels)[0], 1])], axis=1)
    input_pixels_tiled_transposed = tf.transpose(tf.tile(input_pixels_aug[np.newaxis,:,:], [batch_size_tf, 1, 1]), perm=[0, 2, 1])
    output_pixels = tf.matmul(affine_matrix, input_pixels_tiled_transposed)
    return output_pixels

def nonlinear_pixel_transformation_clousure(input_pixels):
    # input = [tf.shape(input_pixels)[0] (corresponding the individual pixels in a grid of output image), 2]
    # output = [batch_size_tf, 2, tf.shape(input_pixels)[0]]
    # serial_flow does not allow different transformations for each image in the batch

    n_dim = 2
    n_flows = 5
    normalizing_flow_list = []
    flow_class_1 = transforms.NonLinearIARFlow 
    flow_class_2 = transforms.NotManyReflectionsRotationFlow

    for i in range(n_flows):
        flow_class_1_parameters = None
        if flow_class_1.required_num_parameters(n_dim) > 0: flow_class_1_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_1.required_num_parameters(n_dim), use_bias = False, activation = None)
        normalizing_flow_list.append(flow_class_1(input_dim=n_dim, parameters=flow_class_1_parameters))

        flow_class_2_parameters = None
        if flow_class_2.required_num_parameters(n_dim) > 0: flow_class_2_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_2.required_num_parameters(n_dim), use_bias = False, activation = None)
        normalizing_flow_list.append(flow_class_2(input_dim=n_dim, parameters=flow_class_2_parameters))
    serial_flow = transforms.SerialFlow(normalizing_flow_list)

    output_pixels_const, _ = serial_flow.transform(input_pixels, None)
    output_pixels = tf.transpose(tf.tile(output_pixels_const[np.newaxis, :, :], [batch_size_tf, 1, 1]), [0, 2, 1])
    return output_pixels

# im_transformed = spatial_transformer.transformer(im_input, linear_pixel_transformation_clousure, [tf.shape(im_input)[1], tf.shape(im_input)[2]])
# im_transformed = spatial_transformer.transformer(im_input, nonlinear_pixel_transformation_clousure, [tf.shape(im_input)[1], tf.shape(im_input)[2]])
# cost = tf.reduce_mean((im_target[0,:,:,:]-im_transformed[0,:,:,:])**2)

im_transformed_sampled, location_mask = spatial_transformer.transformer_sampled(im_input, nonlinear_pixel_transformation_clousure, [tf.shape(im_input)[1], tf.shape(im_input)[2]], n_location_samples)
im_im_target_sampled = (im_target, location_mask)
pdb.set_trace()

optimizer = tf.train.AdamOptimizer(learning_rate=init_learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08)
opt_step = optimizer.minimize(cost)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()  
sess.run(init)

plt.imsave(exp_dir+'im_input_np'+'.png', im_input_np[0, :, :, :])
plt.imsave(exp_dir+'im_target_np'+'.png', im_target_np[0, :, :, :])

print('Start Timer: ')
start = time.time();
for epoch in range(1, n_epochs+1): 
    learning_rate = init_learning_rate
    print('Current learning rate: ', learning_rate)

    for i in range(1, n_updates_per_epoch+1):
        fd = {im_input: im_input_np, im_target: im_target_np}
        _, cost_np = sess.run([opt_step, cost], feed_dict=fd)
        print('Epoch: '+str(epoch)+' Update: '+str(i)+ ' Cost: '+str(cost_np))

    if epoch == n_epochs or epoch % vis_epoch_rate == 0: 
        print('Eval and Visualize: Epoch, Time: {:d} {:.3f}'.format(epoch, time.time()-start))
    
        fd = {im_input: im_input_np, im_target: im_target_np}
        im_transformed_np = sess.run(im_transformed, feed_dict=fd)
        plt.imsave(exp_dir+'im_transformed_np_'+str(epoch)+'.png', im_transformed_np[0, :, :, :])






















