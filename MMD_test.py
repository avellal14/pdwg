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

batch_size = 1000
input_dim = 1
latent_dim = 2

iter_tf = tf.placeholder(tf.float32)
lambda_z = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [None, input_dim])
epsilon = tf.placeholder(tf.float32, [None, latent_dim])
z_prior = tf.placeholder(tf.float32, [None, latent_dim])

# x = tf.random_normal((batch_size, input_dim), 0, 1, dtype=tf.float32)
# epsilon = tf.random_normal((batch_size, latent_dim), 0, 1, dtype=tf.float32)
# z_prior = tf.random_normal((batch_size, latent_dim), 0, 1, dtype=tf.float32)

layer_1 = tf.layers.dense(inputs = tf.concat([x, epsilon], axis=1), units = 500, use_bias = True, activation = tf.nn.relu)
layer_2 = tf.layers.dense(inputs = layer_1, units = 500, use_bias = True, activation = tf.nn.relu)
layer_3 = tf.layers.dense(inputs = layer_2, units = 500, use_bias = True, activation = tf.nn.relu)
layer_4 = tf.layers.dense(inputs = layer_3, units = 500, use_bias = True, activation = tf.nn.relu)
z = tf.layers.dense(inputs = layer_4, units = latent_dim, use_bias = True, activation = None)

layer_1_rec = tf.layers.dense(inputs = z, units = 500, use_bias = True, activation = tf.nn.relu)
layer_2_rec = tf.layers.dense(inputs = layer_1_rec, units = 500, use_bias = True, activation = tf.nn.relu)
layer_3_rec = tf.layers.dense(inputs = layer_2_rec, units = 500, use_bias = True, activation = tf.nn.relu)
layer_4_rec = tf.layers.dense(inputs = layer_3_rec, units = 500, use_bias = True, activation = tf.nn.relu)
x_rec = tf.layers.dense(inputs = layer_4_rec, units = input_dim, use_bias = True, activation = None)

rec_cost = tf.reduce_mean(tf.reduce_sum((x_rec-x)**2, axis=1))
MMD = helper.compute_MMD(z, z_prior, positive_only=True)

start, timescale = 10000, 3
cost = MMD+helper.hardstep((iter_tf-float(start))/float(timescale))*rec_cost

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08)
cost_step = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()  
sess.run(init)

for i in range(0, 100000):
    fd = {iter_tf: i, lambda_z: 1, x: np.random.randn(batch_size, input_dim), epsilon: np.random.randn(batch_size, latent_dim), z_prior: np.random.randn(batch_size, latent_dim)}
    _, cost_np, z_np, z_prior_np = sess.run([cost_step, cost, z, z_prior], feed_dict = fd)
    if i % 20 == 0:
        print("Iteration: "+str(i) + " Cost: "+str(cost_np))

    if i % 100 == 0:
        fd = {iter_tf: i, lambda_z: 1, x: np.random.randn(10000, input_dim), epsilon: np.random.randn(10000, latent_dim), z_prior: np.random.randn(10000, latent_dim)}
        z_np, z_prior_np = sess.run([z, z_prior], feed_dict = fd)
        helper.dataset_plotter([z_np,], colors=['r',], point_thickness = 4, save_dir = '~/MMD_posterior/', postfix = '_MMD_posterior_'+str(i)+'_e', postfix2 = '_MMD_posterior'+'_m')
        helper.dataset_plotter([z_prior_np,], colors=['g',], point_thickness = 4, save_dir = '~/MMD_prior/', postfix = '_MMD_prior_'+str(i)+'_e', postfix2 = '_MMD_prior'+'_m')
        helper.dataset_plotter([z_np, z_prior_np], point_thickness = 4, save_dir = '~/MMD_prior_posterior/', postfix = '_MMD_prior_posterior_'+str(i)+'_e', postfix2 = '_MMD_prior_posterior'+'_m')

pdb.set_trace()
