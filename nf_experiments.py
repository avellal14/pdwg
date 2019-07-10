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

use_gpu = False 
if platform.dist()[0] == 'Ubuntu': 
    print('On Collab!!!!!')
    use_gpu = True

if use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("os.environ['CUDA_VISIBLE_DEVICES'], ", os.environ['CUDA_VISIBLE_DEVICES'])

exp_dir = str(Path.home())+'/ExperimentalResults/NF_EXP/'
if not os.path.exists(exp_dir): os.makedirs(exp_dir)

resolution = 1000
n_samples = 20000
batch_size = 100
n_dim = 2

uniform_samples = np.random.uniform(0, 1, (n_samples, n_dim))*(range_1_max-range_1_min)+range_1_min
normal_samples = np.random.randn(n_samples, n_dim)
normal_samples_log_pdf = (multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]]).logpdf(normal_samples))[:, np.newaxis]
grid_samples = get_sparse_grid_samples(resolution=resolution, subsample_rate=10, range_min=10*range_1_min, range_max=10*range_1_max)

n_epochs = 1000
vis_epoch_rate = 25

beta1=0.99
beta2=0.999
init_learning_rate = 0.00025
min_learning_rate = 0.00025

################################# TF model ##################################################################
learning_rate_tf = tf.placeholder(tf.float32, [], name='learning_rate')
x_input_tf = tf.placeholder(tf.float32, [None, n_dim])
batch_size_tf = tf.shape(x_input_tf)[0]

n_flows = 20
normalizing_flow_list = []
# flow_class_1 = transforms.SpecificRotationFlow
# flow_class_1 = transforms.NotManyReflectionsRotationFlow
# flow_class_1 = transforms.HouseholdRotationFlow 
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

start_mean = tf.zeros(shape=(batch_size_tf, n_dim))
start_log_var = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*1)
start_dist = distributions.DiagonalGaussianDistribution(params = tf.concat([start_mean, start_log_var], axis=1))
x_log_pdf = start_dist.log_pdf(x_input_tf)

x_transformed, x_transformed_log_pdf = serial_flow.transform(x_input_tf, x_log_pdf)

# weights = [0.33, 0.33, 1-0.33-0.33]
# mean_1 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([-0.5, -0.5])[np.newaxis, :]
# mean_2 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([-0.5, +0.5])[np.newaxis, :]
# mean_3 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([+0.5, -0.5])[np.newaxis, :]
# log_var_1 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*0.3)
# log_var_2 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*0.3)
# log_var_3 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*0.3)
# dist_1 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_1, log_var_1], axis=1))
# dist_2 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_2, log_var_2], axis=1))
# dist_3 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_3, log_var_3], axis=1))
# dists = [dist_1, dist_2, dist_3]

weights = [(1-3/40)/7, (1-3/40)/7, (1-3/40)/7, (1-3/40)/7, (1-3/40)/7, (1-3/40)/7, (1-3/40)/7, 1/40, 1/40, 1/40]
mean_1 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([-scale_xx*0.5, +scale_xx*0.5])[np.newaxis, :]
mean_2 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([+scale_xx*0.0, +scale_xx*0.5])[np.newaxis, :]
mean_3 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([+scale_xx*0.5, +scale_xx*0.5])[np.newaxis, :]
mean_4 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([-scale_xx*0.5, +scale_xx*0.0])[np.newaxis, :]
mean_5 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([-scale_xx*0.5, -scale_xx*0.5])[np.newaxis, :]
mean_6 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([+scale_xx*0.0, -scale_xx*0.5])[np.newaxis, :]
mean_7 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([+scale_xx*0.5, -scale_xx*0.5])[np.newaxis, :]
mean_8 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([-scale_xx*0.2, +scale_xx*0.0])[np.newaxis, :]
mean_9 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([-scale_xx*0.0, +scale_xx*0.0])[np.newaxis, :]
mean_10 = tf.zeros(shape=(batch_size_tf, n_dim))+np.asarray([+3*0.2, +3*0.0])[np.newaxis, :]
log_var_1 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*scale_xx*0.2)
log_var_2 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*scale_xx*0.2)
log_var_3 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*scale_xx*0.2)
log_var_4 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*scale_xx*0.2)
log_var_5 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*scale_xx*0.2)
log_var_6 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*scale_xx*0.2)
log_var_7 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*scale_xx*0.2)
log_var_8 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*scale_xx*0.1)
log_var_9 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*scale_xx*0.1)
log_var_10 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*scale_xx*0.1)
dist_1 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_1, log_var_1], axis=1))
dist_2 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_2, log_var_2], axis=1))
dist_3 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_3, log_var_3], axis=1))
dist_4 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_4, log_var_4], axis=1))
dist_5 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_5, log_var_5], axis=1))
dist_6 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_6, log_var_6], axis=1))
dist_7 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_7, log_var_7], axis=1))
dist_8 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_8, log_var_8], axis=1))
dist_9 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_9, log_var_9], axis=1))
dist_10 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_10, log_var_10], axis=1))
dists = [dist_1, dist_2, dist_3, dist_4, dist_5, dist_6, dist_7, dist_8, dist_9, dist_10]

mix_dist = distributions.MixtureDistribution(dists=dists, weights=weights)
samples_mix = mix_dist.sample()
samples_mix_log_pdf = mix_dist.log_pdf(samples_mix)
x_transformed_mix_log_pdf = mix_dist.log_pdf(x_transformed)

kl_div = tf.reduce_mean(x_transformed_log_pdf)-tf.reduce_mean(x_transformed_mix_log_pdf)
cost = kl_div

optimizer = tf.train.AdamOptimizer(learning_rate=init_learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08)
opt_step = optimizer.minimize(cost)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()  
sess.run(init)

print('Start Timer: ')
start = time.time();
for epoch in range(1, n_epochs+1): 
    learning_rate = init_learning_rate
    # learning_rate = max(min_learning_rate, init_learning_rate*1/(2**(epoch-1)))
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.5, epsilon=1e-08)
    # opt_step = optimizer.minimize(cost)
    # adam_initializers = [var.initializer for var in tf.global_variables() if ('Adam' in var.name or 'beta' in var.name)]
    # sess.run(adam_initializers)

    for i in range(math.ceil(normal_samples.shape[0]/float(batch_size))):
        curr_batch_np = normal_samples[i*batch_size:min((i+1)*batch_size, normal_samples.shape[0]), :]
        fd = {x_input_tf: curr_batch_np, learning_rate_tf:learning_rate}
        _, cost_np = sess.run([opt_step, cost], feed_dict=fd)
    
    print('Epoch: '+str(epoch)+' Current learning rate: '+str(learning_rate)+' Cost_np: '+str(cost_np))

    if epoch == n_epochs or epoch % vis_epoch_rate == 0: 
        print('Eval and Visualize: Epoch, Time: {:d} {:.3f}'.format(epoch, time.time()-start))

        all_samples_mix_np = None
        all_samples_mix_log_pdf_np = None
        all_x_transformed_np = None
        all_x_transformed_log_pdf_np = None
        for i in range(math.ceil(normal_samples.shape[0]/float(batch_size))):
            curr_batch_np = normal_samples[i*batch_size:min((i+1)*batch_size, normal_samples.shape[0]), :]
            fd = {x_input_tf: curr_batch_np, learning_rate_tf:learning_rate}
            samples_mix_np, samples_mix_log_pdf_np, x_transformed_np, x_transformed_log_pdf_np = sess.run([samples_mix, samples_mix_log_pdf, x_transformed, x_transformed_log_pdf], feed_dict=fd)

            if all_samples_mix_np is None: all_samples_mix_np = samples_mix_np
            else: all_samples_mix_np = np.concatenate([all_samples_mix_np, samples_mix_np], axis=0)
            if all_samples_mix_log_pdf_np is None: all_samples_mix_log_pdf_np = samples_mix_log_pdf_np
            else: all_samples_mix_log_pdf_np = np.concatenate([all_samples_mix_log_pdf_np, samples_mix_log_pdf_np], axis=0)

            if all_x_transformed_np is None: all_x_transformed_np = x_transformed_np
            else: all_x_transformed_np = np.concatenate([all_x_transformed_np, x_transformed_np], axis=0)
            if all_x_transformed_log_pdf_np is None: all_x_transformed_log_pdf_np = x_transformed_log_pdf_np
            else: all_x_transformed_log_pdf_np = np.concatenate([all_x_transformed_log_pdf_np, x_transformed_log_pdf_np], axis=0)

        # all_transformed_grid_samples_np = None
        all_transformed_grid_samples_np = np.zeros(grid_samples.shape)
        for i in range(math.ceil(grid_samples.shape[0]/float(batch_size))):
            curr_batch_np = grid_samples[i*batch_size:min((i+1)*batch_size, grid_samples.shape[0]), :]
            fd = {x_input_tf: curr_batch_np, learning_rate_tf:learning_rate}
            x_transformed_np = sess.run(x_transformed, feed_dict=fd)
            all_transformed_grid_samples_np[i*loc_batch_size:min((i+1)*loc_batch_size, grid_samples.shape[0]), :] = x_transformed_np

            # if all_transformed_grid_samples_np is None: all_transformed_grid_samples_np = x_transformed_np
            # else: all_transformed_grid_samples_np = np.concatenate([all_transformed_grid_samples_np, x_transformed_np], axis=0)

        fig, ax = plt.subplots(figsize=(7*3, 7*2))
        plt.clf()
        ax_1 = fig.add_subplot(2, 3, 1)
        ax_1.scatter(normal_samples[:, 0], normal_samples[:, 1], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("PiYG")(Normalize()(np.exp(normal_samples_log_pdf[:, 0]))))
        ax_2 = fig.add_subplot(2, 3, 2)
        ax_2.scatter(all_x_transformed_np[:, 0], all_x_transformed_np[:, 1], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("PiYG")(Normalize()(np.exp(all_x_transformed_log_pdf_np[:, 0]))))
        ax_3 = fig.add_subplot(2, 3, 3)
        ax_3.scatter(all_samples_mix_np[:, 0], all_samples_mix_np[:, 1], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("PiYG")(Normalize()(np.exp(all_samples_mix_log_pdf_np[:, 0]))))
        ax_4 = fig.add_subplot(2, 3, 4)
        ax_4.scatter(grid_samples[:, 0], grid_samples[:, 1], s=marker_size, lw = marker_line, edgecolors='k')
        ax_5 = fig.add_subplot(2, 3, 5)
        ax_5.scatter(all_transformed_grid_samples_np[:, 0], all_transformed_grid_samples_np[:, 1], s=marker_size, lw = marker_line, edgecolors='k')
        ax_6 = fig.add_subplot(2, 3, 6)
        ax_6.scatter(all_samples_mix_np[:, 0], all_samples_mix_np[:, 1], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("PiYG")(Normalize()(np.exp(all_samples_mix_log_pdf_np[:, 0]))))

        for ax in [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6]:
            ax.set_xlim(range_1_min, range_1_max)
            ax.set_ylim(range_1_min, range_1_max)
            set_axis_prop(ax, grid_on, ticks_on, axis_on)
            
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.draw()
        if epoch == n_epochs: plt.savefig(exp_dir+'comp_'+str(epoch)+'.png', bbox_inches='tight', format='png', dpi=int(my_dpi), transparent=False)
        else: plt.savefig(exp_dir+'comp_'+str(epoch)+'.png', bbox_inches='tight', format='png', dpi=int(quality*my_dpi), transparent=False)
 
































