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

# dim = 100
# v = np.random.randn(dim,1)
# v_norm = np.sqrt(np.sum(v**2))
# v_dir = v/v_norm
# householder = np.eye(dim)-2*np.dot(v_dir, v_dir.T)
# np.diag(householder)

# J1 = special_ortho_group.rvs(3)
# J3 = special_ortho_group.rvs(5)
# J2 = special_ortho_group.rvs(5)[:,:3]
# JA = np.dot(J3, np.dot(J2, J1))
# print(np.dot(JA.T,JA))


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

# def obj_fun(X):
#     func_scale_1 = 0.8
#     func_scale_2 = 0.2
#     vals = np.zeros((X.shape[0],))
#     for i in range(X.shape[0]):
#         vals[i] = func_scale_1*(func_scale_2*X[i,0]**2-0.5*np.cos(2*np.pi*X[i,0])+X[i,1]**2-0.5*np.cos(2*np.pi*X[i,1]))
#     return vals

def obj_fun(X):
    func_scale_1 = 0.8
    func_scale_2 = 0.2
    vals = np.zeros((X.shape[0],))
    for i in range(X.shape[0]):
        vals[i] = func_scale_1*(func_scale_2*(X[i,0]**2+X[i,1]**2)-0.5*(np.cos(2*np.pi*X[i,0])+np.cos(2*np.pi*X[i,1])))
    return vals

mix_1 = 0.5
l1 = multivariate_normal(mean=[0,0], cov=[[2,0],[0,2]])
mix_2_1 = 0.25
mix_2_2 = 0.25
mix_2_3 = 0.25
mix_2_4 = 0.25
l2_1 = multivariate_normal(mean=[0.5,0.5],  cov=[[0.5,0],[0,0.5]])
l2_2 = multivariate_normal(mean=[0.5,-0.5], cov=[[0.5,0],[0,0.5]])
l2_3 = multivariate_normal(mean=[-0.5,0.5], cov=[[0.5,0],[0,0.5]])
l2_4 = multivariate_normal(mean=[-0.5,-0.5], cov=[[0.5,0],[0,0.5]])

def density_fun(X):
    density = mix_1*l1.pdf(X)+(1-mix_1)*(mix_2_1*l2_1.pdf(X)+mix_2_2*l2_2.pdf(X)+mix_2_3*l2_3.pdf(X)+mix_2_4*l2_4.pdf(X))
    return density[:, np.newaxis]

use_gpu = False 
if platform.dist()[0] == 'Ubuntu': 
    print('On Collab!!!!!')
    use_gpu = True

exp_dir = str(Path.home())+'/ExperimentalResults/RNF_EXP/'
if not os.path.exists(exp_dir): os.makedirs(exp_dir)

range_1_min = -1
range_1_max = 1

resolution = 200
n_samples = 20000
n_training_samples = 20000

n_epochs = 50
vis_epoch_rate = 1

batch_size = 500
train_batch_size = 50
n_latent = 2
n_out = 3
n_input_CPO, n_output_CPO = 15, 15

data_manifold_xy = np.random.uniform(0, 1, (n_samples, 2))*(range_1_max-range_1_min)+range_1_min
data_manifold_z = obj_fun(data_manifold_xy)[:, np.newaxis]
data_manifold = np.concatenate([data_manifold_xy, data_manifold_z], axis=1)

densities = density_fun(data_manifold_xy)
pdb.set_trace()


full_grid_xy_samples, full_grid_xy, full_x0_range, full_x1_range = get_full_grid_samples(resolution=resolution, range_min=range_1_min, range_max=range_1_max)
full_grid_z_samples = obj_fun(full_grid_xy_samples)[:, np.newaxis]
grid_manifold = np.concatenate([full_grid_xy_samples, full_grid_z_samples], axis=1)

rec_data_manifold = np.zeros(data_manifold.shape)
rec_grid_manifold = np.zeros(grid_manifold.shape)

################################# TF training ##################################################################

if use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("os.environ['CUDA_VISIBLE_DEVICES'], ", os.environ['CUDA_VISIBLE_DEVICES'])

x_input = tf.placeholder(tf.float32, [None, n_out])
batch_size_tf = tf.shape(x_input)[0]

n_RF_parameter = transforms.RiemannianFlow.required_num_parameters(n_latent, n_out, n_input_CPO, n_output_CPO)
n_HF_parameter = transforms.HouseholdRotationFlow.required_num_parameters(n_out)
riemannian_flow_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_RF_parameter, use_bias = False, activation = None)
householder_flow_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_HF_parameter, use_bias = False, activation = None)
riemannian_flow = transforms.RiemannianFlow(input_dim=n_latent, output_dim=n_out, n_input_CPO=n_input_CPO, n_output_CPO=n_output_CPO, parameters=riemannian_flow_parameters)
householder_flow = transforms.HouseholdRotationFlow(input_dim=n_out, parameters=householder_flow_parameters)
serial_flow = transforms.SerialFlow([riemannian_flow, householder_flow])

prior_param = tf.zeros((batch_size_tf, 2*n_latent), tf.float32)
prior_dist = distributions.DiagonalGaussianDistribution(params=prior_param)

lay_1 = tf.layers.dense(inputs = x_input, units = 200, use_bias = True, activation = tf.nn.relu) 
lay_2 = lay_1+tf.layers.dense(inputs = lay_1, units = 200, use_bias = True, activation = tf.nn.relu) 
lay_3 = lay_2+tf.layers.dense(inputs = lay_2, units = 200, use_bias = True, activation = tf.nn.relu) 
lay_4 = lay_3+tf.layers.dense(inputs = lay_3, units = 200, use_bias = True, activation = tf.nn.relu) 
z_x = tf.layers.dense(inputs = lay_4, units = n_latent, use_bias = True, activation = None) 
log_pdf_z_x = prior_dist.log_pdf(z_x)
x_rec, log_pdf_x_rec = serial_flow.transform(z_x, log_pdf_z_x)
rec_cost = 100*tf.reduce_mean(tf.reduce_sum((x_rec-x_input)**2, axis=1))
# rec_cost = tf.reduce_mean(tf.reduce_sum((x_rec-x_input)**2, axis=1))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.1, beta2=0.1, epsilon=1e-08)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9, momentum=0., epsilon=1e-10) # good with overall rotation added
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.99, beta2=0.999, epsilon=1e-08) # good with overall rotation added
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.5, beta2=0.9, epsilon=1e-08) # good with overall rotation added
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.9, epsilon=1e-08) # good without overall rotation added
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.99, epsilon=1e-08) # good without overall rotation added
cost_step = optimizer.minimize(rec_cost)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()  
sess.run(init)

print('Start Timer: ')
start = time.time();

rec_data_manifolds = None
rec_grid_manifolds = None
for epoch in range(1, n_epochs+1):
    perm_indeces = np.random.permutation(np.arange(n_training_samples))     
    training_data_manifold_scrambled = data_manifold[:n_training_samples,:][perm_indeces,:]
    print(epoch, math.ceil(training_data_manifold_scrambled.shape[0]/float(train_batch_size)))
    for i in range(math.ceil(training_data_manifold_scrambled.shape[0]/float(train_batch_size))):
        curr_batch_np = training_data_manifold_scrambled[i*train_batch_size:min((i+1)*train_batch_size, training_data_manifold_scrambled.shape[0]), :]
        fd = {x_input: curr_batch_np,}
        _, rec_cost_np, z_x_np, log_pdf_z_x_np, x_rec_np, log_pdf_x_rec_np = sess.run([cost_step, rec_cost, z_x, log_pdf_z_x, x_rec, log_pdf_x_rec], feed_dict=fd)
    
    if epoch % vis_epoch_rate == 0: 
        print('Eval and Visualize: Epoch, Time: {:d} {:.3f}'.format(epoch, time.time()-start))
        total_rec_cost_np = 0
        for i in range(math.ceil(data_manifold.shape[0]/float(batch_size))):
            curr_batch_np = data_manifold[i*batch_size:min((i+1)*batch_size, data_manifold.shape[0]), :]
            fd = {x_input: curr_batch_np,}
            rec_cost_np, z_x_np, log_pdf_z_x_np, x_rec_np, log_pdf_x_rec_np = sess.run([rec_cost, z_x, log_pdf_z_x, x_rec, log_pdf_x_rec], feed_dict=fd)
            rec_data_manifold[i*batch_size:min((i+1)*batch_size, data_manifold.shape[0]), :] = x_rec_np
            total_rec_cost_np += rec_cost_np

        for i in range(math.ceil(grid_manifold.shape[0]/float(batch_size))):
            curr_batch_np = grid_manifold[i*batch_size:min((i+1)*batch_size, grid_manifold.shape[0]), :]
            fd = {x_input: curr_batch_np,}
            rec_cost_np, z_x_np, log_pdf_z_x_np, x_rec_np, log_pdf_x_rec_np = sess.run([rec_cost, z_x, log_pdf_z_x, x_rec, log_pdf_x_rec], feed_dict=fd)
            rec_grid_manifold[i*batch_size:min((i+1)*batch_size, grid_manifold.shape[0]), :] = x_rec_np

        print(total_rec_cost_np/data_manifold.shape[0])
        if rec_data_manifolds is None: rec_data_manifolds = rec_data_manifold[np.newaxis, ...]
        else: rec_data_manifolds = np.concatenate([rec_data_manifolds, rec_data_manifold[np.newaxis, ...]], axis=0)

        if rec_grid_manifolds is None: rec_grid_manifolds = rec_grid_manifold[np.newaxis, ...]
        else: rec_grid_manifolds = np.concatenate([rec_grid_manifolds, rec_grid_manifold[np.newaxis, ...]], axis=0)

end = time.time()
print('Overall Time: {:.3f}\n'.format((end - start)))

np.save(exp_dir+'data_manifold.npy', data_manifold, allow_pickle=True, fix_imports=True)
np.save(exp_dir+'grid_manifold.npy', grid_manifold, allow_pickle=True, fix_imports=True)
np.save(exp_dir+'rec_data_manifolds.npy', rec_data_manifolds, allow_pickle=True, fix_imports=True)
np.save(exp_dir+'rec_grid_manifolds.npy', rec_grid_manifolds, allow_pickle=True, fix_imports=True)































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


























