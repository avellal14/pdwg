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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler

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

def obj_fun(X):
    func_scale_1 = 0.8
    func_scale_2 = 0.2
    vals = np.zeros((X.shape[0],))
    for i in range(X.shape[0]):
        vals[i] = func_scale_1*(func_scale_2*X[i,0]**2-0.5*np.cos(2*np.pi*X[i,0])+X[i,1]**2-0.5*np.cos(2*np.pi*X[i,1]))
    return vals

use_gpu = False 
if platform.dist()[0] == 'Ubuntu': 
    print('On Collab!!!!!')
    use_gpu = True

exp_dir = str(Path.home())+'/ExperimentalResults/RNF_EXP/'

n_samples = 10000
range_1_min = -1
range_1_max = 1

n_epochs = 25
vis_epoch_rate = 1

train_batch_size = 10
n_latent = 2
n_out = 3
n_input_CPO, n_output_CPO = 30, 30

data_manifold_xy = np.random.uniform(0, 1, (n_samples, 2))*(range_1_max-range_1_min)+range_1_min
# data_manifold_xy = np.random.randn(n_samples, 2)*0.4
data_manifold_z = obj_fun(data_manifold_xy)[:, np.newaxis]
data_manifold = np.concatenate([data_manifold_xy, data_manifold_z], axis=1)
rec_data_manifold = np.zeros(data_manifold.shape)

################################# TF training ##################################################################

if use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("os.environ['CUDA_VISIBLE_DEVICES'], ", os.environ['CUDA_VISIBLE_DEVICES'])

x_input = tf.placeholder(tf.float32, [None, n_out])
batch_size_tf = tf.shape(x_input)[0]

n_parameter = transforms.RiemannianFlow.required_num_parameters(n_latent, n_out, n_input_CPO, n_output_CPO)
flow_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
riemannian_flow = transforms.RiemannianFlow(input_dim=n_latent, output_dim=n_out, n_input_CPO=n_input_CPO, n_output_CPO=n_output_CPO, parameters=flow_parameters)

prior_param = tf.zeros((batch_size_tf, 2*n_latent), tf.float32)
prior_dist = distributions.DiagonalGaussianDistribution(params=prior_param)

lay_1 = tf.layers.dense(inputs = x_input, units = 100, use_bias = True, activation = tf.nn.relu) 
lay_2 = tf.layers.dense(inputs = lay_1, units = 100, use_bias = True, activation = tf.nn.relu) 
lay_3 = tf.layers.dense(inputs = lay_2, units = 100, use_bias = True, activation = tf.nn.relu) 
lay_4 = tf.layers.dense(inputs = lay_3, units = 100, use_bias = True, activation = tf.nn.relu) 
z_x = tf.layers.dense(inputs = lay_4, units = n_latent, use_bias = True, activation = None) 
log_pdf_z_x = prior_dist.log_pdf(z_x)
x_rec, log_pdf_x_rec = riemannian_flow.transform(z_x, log_pdf_z_x)

rec_cost = tf.reduce_mean(tf.reduce_sum((x_rec-x_input)**2, axis=1))
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.99, epsilon=1e-08)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.1, beta2=0.1, epsilon=1e-08)
cost_step = optimizer.minimize(rec_cost)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()  
sess.run(init)

print('Start Timer: ')
start = time.time();

rec_data_manifolds = None
for epoch in range(1, n_epochs+1):
    batch_size = train_batch_size
    perm_indeces = np.random.permutation(np.arange(data_manifold.shape[0])) 
    data_manifold_scrambled = data_manifold[perm_indeces,:]
    for i in range(math.ceil(data_manifold_scrambled.shape[0]/float(batch_size))):
        curr_batch_np = data_manifold_scrambled[i*batch_size:min((i+1)*batch_size, data_manifold_scrambled.shape[0]), :]
        fd = {x_input: curr_batch_np,}
        _, rec_cost_np, z_x_np, log_pdf_z_x_np, x_rec_np, log_pdf_x_rec_np = sess.run([cost_step, rec_cost, z_x, log_pdf_z_x, x_rec, log_pdf_x_rec], feed_dict=fd)
    
    if epoch % vis_epoch_rate == 0: 
        print('Eval and Visualize: Epoch, Time: {:d} {:.3f}'.format(epoch, time.time()-start))
        total_rec_cost_np = 0
        batch_size = 500
        for i in range(math.ceil(data_manifold.shape[0]/float(batch_size))):
            curr_batch_np = data_manifold[i*batch_size:min((i+1)*batch_size, data_manifold.shape[0]), :]
            fd = {x_input: curr_batch_np,}
            rec_cost_np, z_x_np, log_pdf_z_x_np, x_rec_np, log_pdf_x_rec_np = sess.run([rec_cost, z_x, log_pdf_z_x, x_rec, log_pdf_x_rec], feed_dict=fd)
            rec_data_manifold[i*batch_size:min((i+1)*batch_size, data_manifold.shape[0]), :] = x_rec_np
            total_rec_cost_np += rec_cost_np
        print(total_rec_cost_np/data_manifold.shape[0])
        if rec_data_manifolds is None: rec_data_manifolds = rec_data_manifold[np.newaxis, ...]
        else: rec_data_manifolds = np.concatenate([rec_data_manifolds, rec_data_manifold[np.newaxis, ...]], axis=0)

end = time.time()
print('Overall Time: {:.3f}\n'.format((end - start)))

if not os.path.exists(exp_dir): os.makedirs(exp_dir)
np.save(exp_dir+'data_manifold.npy', data_manifold, allow_pickle=True, fix_imports=True)
np.save(exp_dir+'rec_data_manifolds.npy', rec_data_manifolds, allow_pickle=True, fix_imports=True)


















        # print(i, i*batch_size, min((i+1)*batch_size, data_manifold_scrambled.shape[0]-i*batch_size), curr_batch_np.shape[0])

    # z0_np, log_pdf_z0_np, z_np, log_pdf_z_np = sess.run([z0, log_pdf_z0, z, log_pdf_z], feed_dict={z0_batch: z0_batch_np, log_pdf_z0_batch: log_pdf_z0_batch_np})


# z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
# log_pdf_z0 = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)
