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


unwarped_path_1 = '/Users/mevlana.gemici/unwarped_small_1.png'
unwarped_path_2 = '/Users/mevlana.gemici/unwarped_small_2.png'
warped_path_1 = '/Users/mevlana.gemici/warped_small_1.png'
warped_path_2 = '/Users/mevlana.gemici/warped_small_2.png'
warped_rot_path_1 = '/Users/mevlana.gemici/warped_rot_small_1.png'
warped_rot_path_2 = '/Users/mevlana.gemici/warped_rot_small_2.png'

unwarped_1_np = plt.imread(unwarped_path_1)[:,:,:3]
unwarped_2_np = plt.imread(unwarped_path_2)[:,:,:3]
warped_1_np = plt.imread(warped_path_1)[:,:,:3]
warped_2_np = plt.imread(warped_path_2)[:,:,:3]
warped_rot_1_np = plt.imread(warped_rot_path_1)[:,:,:3]
warped_rot_2_np = plt.imread(warped_rot_path_2)[:,:,:3]

assert (unwarped_1_np.shape == warped_1_np.shape)
assert (unwarped_2_np.shape == warped_2_np.shape)
assert (unwarped_1_np.shape == warped_rot_1_np.shape)
assert (unwarped_2_np.shape == warped_rot_2_np.shape)


resolution = 1000
loc_batch_size = 100
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

n_epochs = 1000
n_updates_per_epoch = 1000
vis_epoch_rate = 1
n_location_samples = 400

beta1=0.9
beta2=0.99
init_learning_rate = 0.00025
min_learning_rate = 0.00025
ignore_background_cost = False

# im_target_np = np.concatenate([unwarped_1_np[np.newaxis, :, :, :], unwarped_2_np[np.newaxis, :, :, :]], axis=0)
# im_input_np = np.concatenate([warped_rot_1_np[np.newaxis, :, :, :], warped_rot_2_np[np.newaxis, :, :, :]], axis=0)
# im_auxiliary_np = np.concatenate([warped_1_np[np.newaxis, :, :, :], warped_2_np[np.newaxis, :, :, :]], axis=0)
im_target_np = np.concatenate([unwarped_1_np[np.newaxis, :, :, :], unwarped_1_np[np.newaxis, :, :, :]], axis=0)
im_input_np = np.concatenate([warped_rot_1_np[np.newaxis, :, :, :], warped_rot_1_np[np.newaxis, :, :, :]], axis=0)
im_auxiliary_np = np.concatenate([warped_1_np[np.newaxis, :, :, :], warped_1_np[np.newaxis, :, :, :]], axis=0)

# im_target = tf.placeholder(tf.float32, [None, None, None, 3])
# im_input = tf.placeholder(tf.float32, [None, None, None, 3])
# location_input_tf = tf.placeholder(tf.float32, [None, 2])
im_target = tf.placeholder(tf.float32, im_target_np.shape)
im_input = tf.placeholder(tf.float32, im_input_np.shape)
location_input_tf = tf.placeholder(tf.float32, [None, 2])

batch_size_tf = tf.shape(im_input)[0]
################################################################################################################################################################

def linear_pixel_transformation_clousure(input_pixels): 
    # input = [tf.shape(input_pixels)[0] (corresponding the individual pixels in a grid of output image), 2]
    # output = [batch_size_tf, 2, tf.shape(input_pixels)[0]]
    # affine_matrix = [batch_size_tf, 2, 3] allows different transformations for each image in the batch

    zoom = 1.0
    angle = 10 

    # NOTE: Since the transformation is transforming the coordinates of output grid --> input grid, input grid --> output grid is the inverse
    # of this transformation. Therefore, in order to increase the scale of the input image and zoom, we must scale with 1/zoom and in order to rotate the 
    # input image by 10 degrees counterclockwise, we must rotate the output grid by -10 degrees counterclockwise. 

    z_sin = tf.zeros((batch_size_tf, 1), tf.float32)+np.sin(-angle*np.pi/180)
    z_cos = tf.zeros((batch_size_tf, 1), tf.float32)+np.cos(-angle*np.pi/180)
    dummy_zero = tf.zeros([batch_size_tf, 1], tf.float32) # translation can replace it
    affine_matrix = tf.concat([tf.concat([(1/zoom)*z_cos[:,:,np.newaxis], -(1/zoom)*z_sin[:,:,np.newaxis], dummy_zero[:,:,np.newaxis]], axis=2), 
                               tf.concat([(1/zoom)*z_sin[:,:,np.newaxis], (1/zoom)*z_cos[:,:,np.newaxis], dummy_zero[:,:,np.newaxis]], axis=2)], axis=1)

    input_pixels_aug = tf.concat([input_pixels, tf.ones([tf.shape(input_pixels)[0], 1])], axis=1)
    input_pixels_tiled_transposed = tf.transpose(tf.tile(input_pixels_aug[np.newaxis,:,:], [batch_size_tf, 1, 1]), perm=[0, 2, 1])
    output_pixels = tf.matmul(affine_matrix, input_pixels_tiled_transposed)
    return output_pixels

################################################################################################################################################################

n_dim = 2
n_flows = 10
normalizing_flow_list = []
flow_class_1 = transforms.ProperIsometricFlow
flow_class_2 = transforms.NonLinearIARFlow 
for i in range(n_flows):
    flow_class_1_parameters = None
    if flow_class_1.required_num_parameters(n_dim) > 0: 
        with tf.variable_scope("eft_first"+str(i), reuse=False):
            flow_class_1_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_1.required_num_parameters(n_dim), use_bias = False, activation = None)
    normalizing_flow_list.append(flow_class_1(input_dim=n_dim, parameters=flow_class_1_parameters))

    flow_class_2_parameters = None
    if flow_class_2.required_num_parameters(n_dim) > 0:
        with tf.variable_scope("eft_second"+str(i), reuse=False):
            flow_class_2_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_2.required_num_parameters(n_dim), use_bias = False, activation = None)
    normalizing_flow_list.append(flow_class_2(input_dim=n_dim, parameters=flow_class_2_parameters))
serial_flow = transforms.SerialFlow(normalizing_flow_list)
location_transformed_tf, _ = serial_flow.inverse_transform(location_input_tf, None)

def nonlinear_pixel_transformation_clousure(batch_input_pixels):
    # input = [tf.shape(input_pixels)[0] (corresponding the individual pixels in a grid of output image), 2]
    # output = [batch_size_tf, 2, tf.shape(input_pixels)[0]]
    # serial_flow does not allow different transformations for each image in the batch
    # See the note in linear_pixel_transformation_clousure for understanding the transformation.
    
    input_pixels_flat = tf.reshape(batch_input_pixels, [-1, batch_input_pixels.get_shape()[-1].value])
    output_pixels_flat, _ = serial_flow.transform(input_pixels_flat, None)
    batch_output_pixels = tf.reshape(output_pixels_flat, tf.shape(batch_input_pixels))
    return batch_output_pixels

################################################################################################################################################################

im_transformed, vis_im_transformed, im_target_gathered, invalid_map = spatial_transformer.transformer(input_im=im_input, pixel_transformation_clousure=nonlinear_pixel_transformation_clousure, 
                                                                      out_size=[tf.shape(im_input)[1], tf.shape(im_input)[2]], 
                                                                      n_location_samples=None, out_comparison_im=im_target)

im_transformed_sampled, vis_im_transformed_sampled, im_target_gathered_sampled, invalid_map = spatial_transformer.transformer(input_im=im_input, pixel_transformation_clousure=nonlinear_pixel_transformation_clousure, 
                                                                                              out_size=[tf.shape(im_input)[1], tf.shape(im_input)[2]], 
                                                                                              n_location_samples=n_location_samples, out_comparison_im=im_target)

if ignore_background_cost:
    cost = tf.reduce_mean(((im_transformed_sampled-im_target_gathered_sampled)*tf.stop_gradient(1-invalid_map))**2)
else:
    cost = tf.reduce_mean((im_transformed_sampled-im_target_gathered_sampled)**2)

first_vars = [v for v in tf.trainable_variables() if 'first' in v.name]
second_vars = [v for v in tf.trainable_variables() if 'second' in v.name] 
eft_vars = [v for v in tf.trainable_variables() if 'eft' in v.name] 

global_step = tf.Variable(0.0, name='global_step', trainable=False)
opt_step_first = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(
                 learning_rate=init_learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08), 
                 loss=cost, var_list=first_vars, global_step=global_step, clip_param=5)
opt_step_second = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(
                 learning_rate=init_learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08), 
                 loss=cost, var_list=second_vars, global_step=global_step, clip_param=5)
opt_step_eft = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(
                 learning_rate=init_learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08), 
                 loss=cost, var_list=eft_vars, global_step=global_step, clip_param=5)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()  
sess.run(init)

plt.imsave(exp_dir+'im_input_np'+'.png', im_input_np[0, :, :, :])
plt.imsave(exp_dir+'im_target_np'+'.png', im_target_np[0, :, :, :])

print('Start Timer: ')
start = time.time();
for epoch in range(1, n_epochs+1): 
    if epoch < 20: opt_step = opt_step_first
    else: opt_step = opt_step_eft

    learning_rate = init_learning_rate
    print('Current learning rate: ', learning_rate)
    if epoch == 1 or epoch == n_epochs or epoch % vis_epoch_rate == 0: 
        print('Eval and Visualize: Epoch, Time: {:d} {:.3f}'.format(epoch, time.time()-start))
        fd = {im_input: im_input_np, im_target: im_target_np}
        im_transformed_np = sess.run(vis_im_transformed, feed_dict=fd)
        in_out_target = np.concatenate([im_input_np[0, :, :, :], im_transformed_np[0, :, :, :], im_target_np[0, :, :, :]], axis=1) 
        plt.imsave(exp_dir+'im/im_transformed_np_'+str(epoch)+'.png', im_transformed_np[0, :, :, :])
        plt.imsave(exp_dir+'all/all_im_transformed_np_'+str(epoch)+'.png', in_out_target)

    # if epoch == 1 or epoch == n_epochs or epoch % loc_vis_epoch_rate == 0: 
    if epoch == n_epochs or epoch % loc_vis_epoch_rate == 0: 
        all_transformed_grid_samples_np = np.zeros(grid_samples.shape)
        for i in range(math.ceil(grid_samples.shape[0]/float(loc_batch_size))):
            curr_batch_np = grid_samples[i*loc_batch_size:min((i+1)*loc_batch_size, grid_samples.shape[0]), :]
            fd = {location_input_tf: curr_batch_np}
            fd = {im_input: im_input_np, im_target: im_target_np, location_input_tf: curr_batch_np}
            location_transformed_np = sess.run(location_transformed_tf, feed_dict=fd)
            all_transformed_grid_samples_np[i*loc_batch_size:min((i+1)*loc_batch_size, grid_samples.shape[0]), :] = location_transformed_np

        fig, ax = plt.subplots(figsize=(7*3, 7*1))
        plt.clf()
        ax_1 = fig.add_subplot(1, 3, 1)
        ax_1.scatter(grid_samples[:, 1], -grid_samples[:, 0], s=marker_size, lw = marker_line, edgecolors='k')
        ax_2 = fig.add_subplot(1, 3, 2)
        ax_2.scatter(all_transformed_grid_samples_np[:, 1], -all_transformed_grid_samples_np[:, 0], s=marker_size, lw = marker_line, edgecolors='k')
        ax_3 = fig.add_subplot(1, 3, 3)
        ax_3.scatter(grid_samples[:, 1], -grid_samples[:, 0], s=marker_size, lw = marker_line, edgecolors='k')

        for ax in [ax_1, ax_2, ax_3]:
            ax.set_xlim(range_1_min, range_1_max)
            ax.set_ylim(range_1_min, range_1_max)
            set_axis_prop(ax, grid_on, ticks_on, axis_on)
        
        # plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.draw()
        plt.savefig(exp_dir+'gt/gt_'+str(epoch)+'.png', bbox_inches='tight', format='png', dpi=int(quality*my_dpi), transparent=False)
        tmp_image = plt.imread(exp_dir+'gt/gt_'+str(epoch)+'.png')[:,:,:3]
        scaled_tmp_image = rescale(tmp_image, (in_out_target.shape[0]/tmp_image.shape[0], in_out_target.shape[1]/tmp_image.shape[1]), anti_aliasing=False)
        plt.imsave(exp_dir+'gt/gt_'+str(epoch)+'.png', np.concatenate([scaled_tmp_image, in_out_target], axis=0))

    for i in range(1, n_updates_per_epoch+1):
        fd = {im_input: im_input_np, im_target: im_target_np}
        _, cost_np = sess.run([opt_step, cost], feed_dict=fd)
    
    print('Epoch: '+str(epoch)+' Update: '+str(i)+ ' Cost: '+str(cost_np))












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






