"""Random variable transformation classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import helper 
import pdb 
import numpy as np
import math 
from random import shuffle 
from pathlib import Path
import os

batch_size = 1000
input_dim = 1
latent_dim = 2

iter_tf = tf.placeholder(tf.float32)
lambda_z = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [None, input_dim])
epsilon = tf.placeholder(tf.float32, [None, latent_dim])
z_prior = tf.placeholder(tf.float32, [None, latent_dim])

tile_rate = 784
x_input = tf.tile(x, [1, tile_rate])

# layer_1 = tf.layers.dense(inputs = tf.concat([x_input, epsilon], axis=1), units = 1000, use_bias = True, activation = tf.nn.relu)
layer_1 = tf.layers.dense(inputs = tf.concat([x_input, epsilon], axis=1), units = 500, use_bias = True, activation = tf.nn.relu)
layer_2 = tf.layers.dense(inputs = layer_1, units = 500, use_bias = True, activation = tf.nn.relu)
# layer_3 = tf.layers.dense(inputs = layer_2, units = 500, use_bias = True, activation = tf.nn.relu)
# layer_4 = tf.layers.dense(inputs = layer_3, units = 500, use_bias = True, activation = tf.nn.relu)
z = tf.layers.dense(inputs = layer_2, units = latent_dim, use_bias = True, activation = None)

# layer_1_rec = tf.layers.dense(inputs = z, units = 1000, use_bias = True, activation = tf.nn.relu)
layer_1_rec = tf.layers.dense(inputs = z, units = 500, use_bias = True, activation = tf.nn.relu)
layer_2_rec = tf.layers.dense(inputs = layer_1_rec, units = 500, use_bias = True, activation = tf.nn.relu)
# layer_3_rec = tf.layers.dense(inputs = layer_2_rec, units = 500, use_bias = True, activation = tf.nn.relu)
# layer_4_rec = tf.layers.dense(inputs = layer_3_rec, units = 500, use_bias = True, activation = tf.nn.relu)
x_rec = tf.layers.dense(inputs = layer_2_rec, units = input_dim*tile_rate, use_bias = True, activation = None)

rec_cost = tf.reduce_mean(helper.safe_tf_sqrt(tf.reduce_sum((x_rec-x_input)**2, axis=1)))
MMD = helper.compute_MMD(z, z_prior, positive_only=True)

start, timescale = 200, 1500
# start, timescale = 0, 1
lambda_z_comp = helper.hardstep((iter_tf-float(start))/float(timescale))
cost = MMD+lambda_z_comp*rec_cost

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08)
cost_step = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()  
sess.run(init)

if not os.path.exists(str(Path.home())+'/ExperimentalResults/MMD_test/'): os.makedirs(str(Path.home())+'/ExperimentalResults/MMD_test/')
file_path = str(Path.home())+'/ExperimentalResults/MMD_test/results.txt'
with open(file_path, "w") as text_file:
    text_file.write('')

for i in range(0, 5000):
    fd = {iter_tf: i, lambda_z: 1, x: np.random.randn(batch_size, input_dim), epsilon: np.random.randn(batch_size, latent_dim), z_prior: np.random.randn(batch_size, latent_dim)}
    _, cost_np, MMD_np, rec_cost_np, lambda_z_comp_np, z_np, z_prior_np = sess.run([cost_step, cost, MMD, rec_cost, lambda_z_comp, z, z_prior], feed_dict = fd)
    if i % 100 == 0:
        print("Iteration: "+str(i) + " Cost: "+str(cost_np) + " MMD_np: "+str(MMD_np) + " lambda_z_comp_np: "+str(lambda_z_comp_np) + " rec_cost_np: "+str(rec_cost_np))
        with open(file_path, "a") as text_file:
            text_file.write(str(i) + ', '+ str(cost_np) + ', ' +str(MMD_np) + ', ' + str(rec_cost_np) + ', ' + str(lambda_z_comp_np)+'\n')

    if i % 500 == 0:
        fd = {iter_tf: i, lambda_z: 1, x: np.random.randn(10000, input_dim), epsilon: np.random.randn(10000, latent_dim), z_prior: np.random.randn(10000, latent_dim)}
        z_np, z_prior_np = sess.run([z, z_prior], feed_dict = fd)
        helper.dataset_plotter([z_np,], colors=['r',], point_thickness = 4, save_dir = str(Path.home())+'/ExperimentalResults/MMD_test/MMD_posterior/', postfix = '_MMD_posterior_'+str(i)+'_e', postfix2 = '_MMD_posterior'+'_m')
        helper.dataset_plotter([z_prior_np,], colors=['g',], point_thickness = 4, save_dir = str(Path.home())+'/ExperimentalResults/MMD_test/MMD_prior/', postfix = '_MMD_prior_'+str(i)+'_e', postfix2 = '_MMD_prior'+'_m')
        helper.dataset_plotter([z_np, z_prior_np], point_thickness = 4, save_dir = str(Path.home())+'/ExperimentalResults/MMD_test/MMD_prior_posterior/', postfix = '_MMD_prior_posterior_'+str(i)+'_e', postfix2 = '_MMD_prior_posterior'+'_m')
        os.system("python ploter_mmd.py")





