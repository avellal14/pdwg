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
grid_on, ticks_on, axis_on = False, True, True
quality = 0.2
marker_size = 10/2
marker_line = 0.3/10

range_1_min = -1
range_1_max = 1

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
grid_samples = get_sparse_grid_samples(resolution=resolution, subsample_rate=10, range_min=10*range_1_min, range_max=10*range_1_max)

n_epochs = 30
vis_epoch_rate = 1
init_learning_rate = 0.01 
min_learning_rate = 0.0001 

################################# TF model ##################################################################
learning_rate_tf = tf.placeholder(tf.float32, [], name='learning_rate')
x_input_tf = tf.placeholder(tf.float32, [None, n_dim])
batch_size_tf = tf.shape(x_input_tf)[0]

normalizing_flow_list = []
flow_class_1 = transforms.NonLinearIARFlow
flow_class_1_parameters_1, flow_class_1_parameters_2, flow_class_1_parameters_3, flow_class_1_parameters_4 = None, None, None, None
if flow_class_1.required_num_parameters(n_dim) > 0:flow_class_1_parameters_1 = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_1.required_num_parameters(n_dim), use_bias = False, activation = None)
if flow_class_1.required_num_parameters(n_dim) > 0:flow_class_1_parameters_2 = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_1.required_num_parameters(n_dim), use_bias = False, activation = None)
if flow_class_1.required_num_parameters(n_dim) > 0:flow_class_1_parameters_3 = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_1.required_num_parameters(n_dim), use_bias = False, activation = None)
if flow_class_1.required_num_parameters(n_dim) > 0:flow_class_1_parameters_4 = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_1.required_num_parameters(n_dim), use_bias = False, activation = None)

# flow_class_2 = transforms.SpecificRotationFlow
# flow_class_2 = transforms.NotManyReflectionsRotationFlow
flow_class_2 = transforms.HouseholdRotationFlow
flow_class_2_parameters_1, flow_class_2_parameters_2, flow_class_2_parameters_3, flow_class_2_parameters_4 = None, None, None, None
if flow_class_2.required_num_parameters(n_dim) > 0: flow_class_2_parameters_1 = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_2.required_num_parameters(n_dim), use_bias = False, activation = None)
if flow_class_2.required_num_parameters(n_dim) > 0: flow_class_2_parameters_2 = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_2.required_num_parameters(n_dim), use_bias = False, activation = None)
if flow_class_2.required_num_parameters(n_dim) > 0: flow_class_2_parameters_3 = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_2.required_num_parameters(n_dim), use_bias = False, activation = None)
if flow_class_2.required_num_parameters(n_dim) > 0: flow_class_2_parameters_4 = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class_2.required_num_parameters(n_dim), use_bias = False, activation = None)

normalizing_flow_list.append(flow_class_1(input_dim=n_dim, parameters=flow_class_1_parameters_1))
normalizing_flow_list.append(flow_class_2(input_dim=n_dim, parameters=flow_class_2_parameters_1))
normalizing_flow_list.append(flow_class_1(input_dim=n_dim, parameters=flow_class_1_parameters_2))
normalizing_flow_list.append(flow_class_2(input_dim=n_dim, parameters=flow_class_2_parameters_2))
normalizing_flow_list.append(flow_class_1(input_dim=n_dim, parameters=flow_class_1_parameters_3))
normalizing_flow_list.append(flow_class_2(input_dim=n_dim, parameters=flow_class_2_parameters_3))
normalizing_flow_list.append(flow_class_1(input_dim=n_dim, parameters=flow_class_1_parameters_4))
normalizing_flow_list.append(flow_class_2(input_dim=n_dim, parameters=flow_class_2_parameters_4))
serial_flow = transforms.SerialFlow(normalizing_flow_list)

start_mean = tf.zeros(shape=(batch_size_tf, n_dim))
start_log_var = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*1)
start_dist = distributions.DiagonalGaussianDistribution(params = tf.concat([start_mean, start_log_var], axis=1))
x_log_pdf = start_dist.log_pdf(x_input_tf)

x_transformed, x_transformed_log_pdf = serial_flow.transform(x_input_tf, x_log_pdf)

mean_1 = tf.zeros(shape=(batch_size_tf, n_dim))-0.5
mean_2 = tf.zeros(shape=(batch_size_tf, n_dim))+0.1
mean_3 = tf.zeros(shape=(batch_size_tf, n_dim))+0.7
log_var_1 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*0.1)
log_var_2 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*0.1)
log_var_3 = tf.log(tf.ones(shape=(batch_size_tf, n_dim))*0.1)

dist_1 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_1, log_var_1], axis=1))
dist_2 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_2, log_var_2], axis=1))
dist_3 = distributions.DiagonalGaussianDistribution(params = tf.concat([mean_3, log_var_3], axis=1))
mix_dist = distributions.MixtureDistribution(dists = [dist_1, dist_2, dist_3], weights=[0.33, 0.33, 1-0.33-0.33])
samples_mix = mix_dist.sample()
samples_mix_log_pdf = mix_dist.log_pdf(samples_mix)
x_transformed_mix_log_pdf = mix_dist.log_pdf(x_transformed)

kl_div = tf.reduce_mean(x_transformed_log_pdf)-tf.reduce_mean(x_transformed_mix_log_pdf)
cost = kl_div

# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_tf, beta1=0.9, beta2=0.9, epsilon=1e-08)
# opt_step = optimizer.minimize(cost)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()  
sess.run(init)

print('Start Timer: ')
start = time.time();
for epoch in range(1, n_epochs+1): 
    learning_rate = max(min_learning_rate, init_learning_rate*1/(2**(epoch-1)))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.5, epsilon=1e-08)
    opt_step = optimizer.minimize(cost)

    adam_initializers = [var.initializer for var in tf.global_variables() if ('Adam' in var.name or 'beta' in var.name)]
    sess.run(adam_initializers)

    print('Current learning rate: ', learning_rate)

    for i in range(math.ceil(normal_samples.shape[0]/float(batch_size))):
        curr_batch_np = normal_samples[i*batch_size:min((i+1)*batch_size, normal_samples.shape[0]), :]
        fd = {x_input_tf: curr_batch_np, learning_rate_tf:learning_rate}
        _, cost_np = sess.run([opt_step, cost], feed_dict=fd)
        print('cost_np', cost_np)

    if epoch % vis_epoch_rate == 0: 
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

        fig, ax = plt.subplots(figsize=(14, 7))
        plt.clf()
        ax_1 = fig.add_subplot(1, 2, 1)
        ax_1.scatter(all_samples_mix_np[:, 0], all_samples_mix_np[:, 1], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("PiYG")(Normalize()(np.exp(all_samples_mix_log_pdf_np[:, 0]))))
        ax_2 = fig.add_subplot(1, 2, 2)
        ax_2.scatter(all_x_transformed_np[:, 0], all_x_transformed_np[:, 1], s=marker_size, lw = marker_line, edgecolors='k', facecolors=cm.get_cmap("PiYG")(Normalize()(np.exp(all_x_transformed_log_pdf_np[:, 0]))))

        ax_1.set_xlim(range_1_min, range_1_max)
        ax_1.set_ylim(range_1_min, range_1_max)
        ax_2.set_xlim(range_1_min, range_1_max)
        ax_2.set_ylim(range_1_min, range_1_max)

        set_axis_prop(ax_1, grid_on, ticks_on, axis_on)
        set_axis_prop(ax_2, grid_on, ticks_on, axis_on)
        plt.draw()
        plt.savefig(exp_dir+'comp_'+str(epoch)+'.png', bbox_inches='tight', format='png', dpi=int(quality*my_dpi), transparent=False)

pdb.set_trace()

################################# TF training ##################################################################

# flow_class = transforms.RadialFlow
# n_nfs = 1
# normalizing_flow_parameters_list = []
# for i in range(n_nfs):
#     normalizing_flow_parameters_list.append(1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_class.required_num_parameters(n_latent), use_bias = False, activation = None))
# normalizing_flow_list = []
# for i in range(n_nfs):
#     normalizing_flow_list.append(flow_class(input_dim=n_latent, parameters=normalizing_flow_parameters_list[i]))


normalizing_flow_list = []
normalizing_flow_parameters_1 = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = transforms.RadialFlow.required_num_parameters(n_latent), use_bias = False, activation = None)
normalizing_flow_parameters_2 = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = transforms.RadialFlow2.required_num_parameters(n_latent), use_bias = False, activation = None)
normalizing_flow_list.append(transforms.RadialFlow(input_dim=n_latent, parameters=normalizing_flow_parameters_1))
normalizing_flow_list.append(transforms.RadialFlow2(input_dim=n_latent, parameters=normalizing_flow_parameters_2))
serial_flow = transforms.SerialFlow(normalizing_flow_list)

prior_param = tf.zeros((batch_size_tf, 2*n_latent), tf.float32)
prior_dist = distributions.DiagonalGaussianDistribution(params=prior_param)

x_rec, _ = serial_flow.transform(x_input, None)

# optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.1, beta2=0.1, epsilon=1e-08)
# cost_step = optimizer.minimize(rec_cost)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()  
sess.run(init)

print('Start Timer: ')
start = time.time();

data_manifold = data_manifold_xy
grid_manifold = full_grid_xy_samples

rec_data_manifold = np.zeros(data_manifold.shape)
rec_grid_manifold = np.zeros(grid_manifold.shape)

rec_data_manifolds = None
rec_grid_manifolds = None
for epoch in range(1, n_epochs+1):    
    if epoch % vis_epoch_rate == 0: 
        print('Eval and Visualize: Epoch, Time: {:d} {:.3f}'.format(epoch, time.time()-start))

        for i in range(math.ceil(data_manifold.shape[0]/float(batch_size))):
            curr_batch_np = data_manifold[i*batch_size:min((i+1)*batch_size, data_manifold.shape[0]), :]
            fd = {x_input: curr_batch_np,}
            x_rec_np = sess.run(x_rec, feed_dict=fd)
            rec_data_manifold[i*batch_size:min((i+1)*batch_size, data_manifold.shape[0]), :] = x_rec_np

        for i in range(math.ceil(grid_manifold.shape[0]/float(batch_size))):
            curr_batch_np = grid_manifold[i*batch_size:min((i+1)*batch_size, grid_manifold.shape[0]), :]
            fd = {x_input: curr_batch_np,}
            x_rec_np = sess.run(x_rec, feed_dict=fd)
            rec_grid_manifold[i*batch_size:min((i+1)*batch_size, grid_manifold.shape[0]), :] = x_rec_np

        if rec_data_manifolds is None: rec_data_manifolds = rec_data_manifold[np.newaxis, ...]
        else: rec_data_manifolds = np.concatenate([rec_data_manifolds, rec_data_manifold[np.newaxis, ...]], axis=0)

        if rec_grid_manifolds is None: rec_grid_manifolds = rec_grid_manifold[np.newaxis, ...]
        else: rec_grid_manifolds = np.concatenate([rec_grid_manifolds, rec_grid_manifold[np.newaxis, ...]], axis=0)

fig, ax = plt.subplots(figsize=(14, 7))
plt.clf()
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(grid_manifold[:, 0], grid_manifold[:, 1], s=marker_size, linewidths = marker_line, edgecolors='k')
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(rec_grid_manifold[:, 0], rec_grid_manifold[:, 1], s=marker_size, linewidths = marker_line, edgecolors='k')

# ax1.set_xlim(range_1_min, range_1_max)
# ax1.set_ylim(range_1_min, range_1_max)
# ax2.set_xlim(range_1_min, range_1_max)
# ax2.set_ylim(range_1_min, range_1_max)

ax1.set_xlim(range_1_min/5, range_1_max/5)
ax1.set_ylim(range_1_min/5, range_1_max/5)
ax2.set_xlim(range_1_min/5, range_1_max/5)
ax2.set_ylim(range_1_min/5, range_1_max/5)

set_axis_prop(ax1, grid_on, ticks_on, axis_on )
set_axis_prop(ax2, grid_on, ticks_on, axis_on )
plt.draw()
plt.show()
pdb.set_trace()


# end = time.time()
# print('Overall Time: {:.3f}\n'.format((end - start)))

# np.save(exp_dir+'data_manifold.npy', data_manifold, allow_pickle=True, fix_imports=True)
# np.save(exp_dir+'grid_manifold.npy', grid_manifold, allow_pickle=True, fix_imports=True)
# np.save(exp_dir+'rec_data_manifolds.npy', rec_data_manifolds, allow_pickle=True, fix_imports=True)
# np.save(exp_dir+'rec_grid_manifolds.npy', rec_grid_manifolds, allow_pickle=True, fix_imports=True)































# """Random variable transformation classes."""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import tensorflow as tf
# import helper 
# import pdb 
# import numpy as np
# import math 
# import transforms
# import distributions
# import time
# import os
# from pathlib import Path
# import platform
# import subprocess

# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# from matplotlib import cm
# from sklearn.datasets import make_blobs, make_circles, make_moons
# from sklearn.preprocessing import StandardScaler

# def get_full_grid_samples(resolution=100, range_min=-1, range_max=1):
#     x0_range = np.linspace(range_min, range_max, 2*resolution+1)
#     x1_range = np.linspace(range_min, range_max, 2*resolution+1)
#     x0v, x1v = np.meshgrid(x0_range, x1_range)
#     grid_flat = np.concatenate([x0v.flatten()[:, np.newaxis], x1v.flatten()[:, np.newaxis]], axis=1)
#     grid = np.concatenate([x0v[:,:,np.newaxis], x1v[:,:,np.newaxis]], axis=2)
#     return grid_flat, grid, x0_range, x1_range

# def get_sparse_grid_samples(resolution=100, subsample_rate=10, range_min=-1, range_max=1):
#     full_grid_flat, full_grid, _, _ = get_full_grid_samples(resolution=resolution, range_min=range_min, range_max=range_max)
#     index_x0v, index_x1v = np.meshgrid(np.arange(full_grid.shape[0]), np.arange(full_grid.shape[1]))
#     index_grid_flat = np.concatenate([index_x0v.flatten()[:, np.newaxis], index_x1v.flatten()[:, np.newaxis]], axis=1)
#     index_grid = np.concatenate([index_x0v[:,:,np.newaxis], index_x1v[:,:,np.newaxis]], axis=2)
#     subsample_mask = ((index_grid_flat[:, 0]%subsample_rate == 0)+ (index_grid_flat[:, 1]%subsample_rate == 0))>0
#     return full_grid_flat[subsample_mask,:]

# def obj_fun(X):
#     func_scale_1 = 0.8
#     func_scale_2 = 0.2
#     vals = np.zeros((X.shape[0],))
#     for i in range(X.shape[0]):
#         vals[i] = func_scale_1*(func_scale_2*X[i,0]**2-0.5*np.cos(2*np.pi*X[i,0])+X[i,1]**2-0.5*np.cos(2*np.pi*X[i,1]))
#     return vals

# use_gpu = False 
# if platform.dist()[0] == 'Ubuntu': 
#     print('On Collab!!!!!')
#     use_gpu = True

# exp_dir = str(Path.home())+'/ExperimentalResults/RNF_EXP/'

# range_1_min = -1
# range_1_max = 1

# resolution = 200
# n_samples = 20000
# n_training_samples = 20000

# n_epochs = 20
# vis_epoch_rate = 1

# batch_size = 500
# train_batch_size = 50
# n_latent = 2
# n_out = 3
# n_input_CPO, n_output_CPO = 15, 15

# data_manifold_xy = np.random.uniform(0, 1, (n_samples, 2))*(range_1_max-range_1_min)+range_1_min
# data_manifold_z = obj_fun(data_manifold_xy)[:, np.newaxis]
# data_manifold = np.concatenate([data_manifold_xy, data_manifold_z], axis=1)

# full_grid_xy_samples, full_grid_xy, full_x0_range, full_x1_range = get_full_grid_samples(resolution=resolution, range_min=range_1_min, range_max=range_1_max)
# full_grid_z_samples = obj_fun(full_grid_xy_samples)[:, np.newaxis]
# grid_manifold = np.concatenate([full_grid_xy_samples, full_grid_z_samples], axis=1)

# rec_data_manifold = np.zeros(data_manifold.shape)
# rec_grid_manifold = np.zeros(grid_manifold.shape)

# ################################# TF training ##################################################################

# if use_gpu:
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     print("os.environ['CUDA_VISIBLE_DEVICES'], ", os.environ['CUDA_VISIBLE_DEVICES'])

# x_input = tf.placeholder(tf.float32, [None, n_out])
# batch_size_tf = tf.shape(x_input)[0]

# n_parameter = transforms.RiemannianFlow.required_num_parameters(n_latent, n_out, n_input_CPO, n_output_CPO)
# flow_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
# riemannian_flow = transforms.RiemannianFlow(input_dim=n_latent, output_dim=n_out, n_input_CPO=n_input_CPO, n_output_CPO=n_output_CPO, parameters=flow_parameters)

# prior_param = tf.zeros((batch_size_tf, 2*n_latent), tf.float32)
# prior_dist = distributions.DiagonalGaussianDistribution(params=prior_param)

# lay_1 = tf.layers.dense(inputs = x_input, units = 200, use_bias = True, activation = tf.nn.relu) 
# lay_2 = lay_1+tf.layers.dense(inputs = lay_1, units = 200, use_bias = True, activation = tf.nn.relu) 
# lay_3 = lay_2+tf.layers.dense(inputs = lay_2, units = 200, use_bias = True, activation = tf.nn.relu) 
# lay_4 = lay_3+tf.layers.dense(inputs = lay_3, units = 200, use_bias = True, activation = tf.nn.relu) 
# # lay_5 = tf.layers.dense(inputs = lay_4, units = 200, use_bias = True, activation = tf.nn.relu) 
# # lay_6 = tf.layers.dense(inputs = lay_5, units = 200, use_bias = True, activation = tf.nn.relu) 
# z_x = tf.layers.dense(inputs = lay_4, units = n_latent, use_bias = True, activation = None) 
# log_pdf_z_x = prior_dist.log_pdf(z_x)
# x_rec, log_pdf_x_rec = riemannian_flow.transform(z_x, log_pdf_z_x)

# # margin = 0.05
# # rec_cost = 100*tf.reduce_mean(tf.reduce_sum(tf.nn.relu((x_rec-x_input)**2-margin**2), axis=1))
# rec_cost = 100*tf.reduce_mean(tf.reduce_sum((x_rec-x_input)**2, axis=1))


# optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.9, epsilon=1e-08)
# # optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.99, epsilon=1e-08)# good
# # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08)
# # optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.5, beta2=0.9, epsilon=1e-08)
# cost_step = optimizer.minimize(rec_cost)
# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)

# print('Start Timer: ')
# start = time.time();

# rec_data_manifolds = None
# rec_grid_manifolds = None
# for epoch in range(1, n_epochs+1):
#     perm_indeces = np.random.permutation(np.arange(n_training_samples))     
#     training_data_manifold_scrambled = data_manifold[:n_training_samples,:][perm_indeces,:]
#     for i in range(math.ceil(training_data_manifold_scrambled.shape[0]/float(train_batch_size))):
#         curr_batch_np = training_data_manifold_scrambled[i*train_batch_size:min((i+1)*train_batch_size, training_data_manifold_scrambled.shape[0]), :]
#         fd = {x_input: curr_batch_np,}
#         _, rec_cost_np, z_x_np, log_pdf_z_x_np, x_rec_np, log_pdf_x_rec_np = sess.run([cost_step, rec_cost, z_x, log_pdf_z_x, x_rec, log_pdf_x_rec], feed_dict=fd)
    
#     if epoch % vis_epoch_rate == 0: 
#         print('Eval and Visualize: Epoch, Time: {:d} {:.3f}'.format(epoch, time.time()-start))
#         total_rec_cost_np = 0
#         for i in range(math.ceil(data_manifold.shape[0]/float(batch_size))):
#             curr_batch_np = data_manifold[i*batch_size:min((i+1)*batch_size, data_manifold.shape[0]), :]
#             fd = {x_input: curr_batch_np,}
#             rec_cost_np, z_x_np, log_pdf_z_x_np, x_rec_np, log_pdf_x_rec_np = sess.run([rec_cost, z_x, log_pdf_z_x, x_rec, log_pdf_x_rec], feed_dict=fd)
#             rec_data_manifold[i*batch_size:min((i+1)*batch_size, data_manifold.shape[0]), :] = x_rec_np
#             total_rec_cost_np += rec_cost_np

#         for i in range(math.ceil(grid_manifold.shape[0]/float(batch_size))):
#             curr_batch_np = grid_manifold[i*batch_size:min((i+1)*batch_size, grid_manifold.shape[0]), :]
#             fd = {x_input: curr_batch_np,}
#             rec_cost_np, z_x_np, log_pdf_z_x_np, x_rec_np, log_pdf_x_rec_np = sess.run([rec_cost, z_x, log_pdf_z_x, x_rec, log_pdf_x_rec], feed_dict=fd)
#             rec_grid_manifold[i*batch_size:min((i+1)*batch_size, grid_manifold.shape[0]), :] = x_rec_np

#         print(total_rec_cost_np/data_manifold.shape[0])
#         if rec_data_manifolds is None: rec_data_manifolds = rec_data_manifold[np.newaxis, ...]
#         else: rec_data_manifolds = np.concatenate([rec_data_manifolds, rec_data_manifold[np.newaxis, ...]], axis=0)

#         if rec_grid_manifolds is None: rec_grid_manifolds = rec_grid_manifold[np.newaxis, ...]
#         else: rec_grid_manifolds = np.concatenate([rec_grid_manifolds, rec_grid_manifold[np.newaxis, ...]], axis=0)

# end = time.time()
# print('Overall Time: {:.3f}\n'.format((end - start)))

# if not os.path.exists(exp_dir): os.makedirs(exp_dir)
# np.save(exp_dir+'data_manifold.npy', data_manifold, allow_pickle=True, fix_imports=True)
# np.save(exp_dir+'grid_manifold.npy', grid_manifold, allow_pickle=True, fix_imports=True)
# np.save(exp_dir+'rec_data_manifolds.npy', rec_data_manifolds, allow_pickle=True, fix_imports=True)
# np.save(exp_dir+'rec_grid_manifolds.npy', rec_grid_manifolds, allow_pickle=True, fix_imports=True)


























