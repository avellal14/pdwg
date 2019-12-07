# BASED ON https://github.com/igul222/improved_wgan_training
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np
import helper 
import distributions
import transforms
import tensorflow as tf
import copy

import models.PDWGANCannon.tflib as lib
import models.PDWGANCannon.tflib.ops.linear
import models.PDWGANCannon.tflib.ops.conv2d
import models.PDWGANCannon.tflib.ops.batchnorm
import models.PDWGANCannon.tflib.ops.deconv2d
# import models.PDWGANCannon.tflib.save_images
# import models.PDWGANCannon.tflib.cifar10
# import models.PDWGANCannon.tflib.inception_score
import models.PDWGANCannon.tflib.plot
	
class PriorMapGaussian():
	def __init__(self, config, name = '/PriorMapGaussian'):
		self.name = name
		self.config = config
		self.constructed = False
 
	def forward(self, x, name = ''):
		with tf.variable_scope("PriorMapGaussian", reuse=self.constructed):
			input_flat = x[0]			
			mu_pre_sig = tf.zeros(shape=(tf.shape(input_flat)[0], 2*self.config['n_latent']))
			self.constructed = True
			return mu_pre_sig

class Encoder():
	def __init__(self, config, name = '/Encoder'):
		self.name = name
		self.config = config
		self.activation_function = self.config['enc_activation_function']
		self.normalization_mode = self.config['enc_normalization_mode']
		self.constructed = False

	def forward(self, x, noise=None, name = ''):
		with tf.variable_scope("Encoder", reuse=self.constructed):
			if len(self.config['data_properties']['flat']) > 0:
				n_output_size = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['flat']])				
				x_batched_inp_flat = tf.reshape(x['flat'], [-1,  *x['flat'].get_shape().as_list()[2:]])
				
				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = self.activation_function)
				lay2_flat = helper.FCResnetLayer(lay1_flat, units = self.config['n_flat'], activation = self.activation_function)
				latent_flat = tf.layers.dense(inputs = lay2_flat, units = self.config['n_latent'], activation = None)
				z_flat = tf.reshape(latent_flat, [-1, x['flat'].get_shape().as_list()[1], self.config['n_latent']])
				z = z_flat

			if len(self.config['data_properties']['image']) > 0:								
				image_shape = (self.config['data_properties']['image'][0]['size'][-3:-1])
				n_image_size = np.prod(image_shape)
				n_output_channels = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['image']])
				image_input = tf.reshape(x['image'], [-1,  *x['image'].get_shape().as_list()[2:]])
				
				reduce_units = self.config['n_filter']
				# # # 28x28xn_channels
				if image_shape == (28, 28): ## works with 512
					lay1_image = tf.layers.conv2d(inputs=image_input, filters=int(self.config['n_filter']/8), kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay1_image = helper.conv_layer_norm_layer(lay1_image, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						lay1_image = lib.ops.batchnorm.Batchnorm('Encoder.BN1', [0,1,2], lay1_image)
					lay1_image = self.activation_function(lay1_image)
					lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=int(self.config['n_filter']/8), kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay2_image = helper.conv_layer_norm_layer(lay2_image, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						lay2_image = lib.ops.batchnorm.Batchnorm('Encoder.BN2', [0,1,2], lay2_image)
					lay2_image = self.activation_function(lay2_image)
					lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=int(self.config['n_filter']/4), kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay3_image = helper.conv_layer_norm_layer(lay3_image, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						lay3_image = lib.ops.batchnorm.Batchnorm('Encoder.BN3', [0,1,2], lay3_image)
					lay3_image = self.activation_function(lay3_image)
					lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=int(self.config['n_filter']/4), kernel_size=[5, 5], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay4_image = helper.conv_layer_norm_layer(lay4_image, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						lay4_image = lib.ops.batchnorm.Batchnorm('Encoder.BN4', [0,1,2], lay4_image)
					lay4_image = self.activation_function(lay4_image)
					latent_image = tf.layers.conv2d(inputs=lay4_image, filters=self.config['n_filter'], kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=self.activation_function)
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])	
					
				# # # 32x32xn_channels 'n_filter': 512, 28 sec, 3.4 in 20 epochs
				if image_shape == (32, 32): ## works with 512
					lay1_image = tf.layers.conv2d(inputs=image_input, filters=int(self.config['n_filter']/4), kernel_size=[4, 4], strides=[2, 2], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay1_image = helper.conv_layer_norm_layer(lay1_image, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						lay1_image = lib.ops.batchnorm.Batchnorm('Encoder.BN1', [0,1,2], lay1_image)
					lay1_image = self.activation_function(lay1_image)
					lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=int(self.config['n_filter']/2), kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay2_image = helper.conv_layer_norm_layer(lay2_image, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						lay2_image = lib.ops.batchnorm.Batchnorm('Encoder.BN2', [0,1,2], lay2_image)
					lay2_image = self.activation_function(lay2_image)
					lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=int(self.config['n_filter']/2), kernel_size=[3, 3], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay3_image = helper.conv_layer_norm_layer(lay3_image, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						lay3_image = lib.ops.batchnorm.Batchnorm('Encoder.BN3', [0,1,2], lay3_image)
					lay3_image = self.activation_function(lay3_image)
					latent_image = tf.layers.conv2d(inputs=lay3_image, filters=self.config['n_filter'], kernel_size=[3, 3], strides=[1, 1], padding="valid", use_bias=True, activation=self.activation_function)
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])	

				# # # 64x64xn_channels
				if image_shape == (64, 64): 
					lay1_image = tf.layers.conv2d(inputs=image_input, filters=int(self.config['n_filter']/32), kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay1_image = helper.conv_layer_norm_layer(lay1_image, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						lay1_image = lib.ops.batchnorm.Batchnorm('Encoder.BN1', [0,1,2], lay1_image)
					lay1_image = self.activation_function(lay1_image)
					lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=int(self.config['n_filter']/16), kernel_size=[5, 5], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay2_image = helper.conv_layer_norm_layer(lay2_image, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						lay2_image = lib.ops.batchnorm.Batchnorm('Encoder.BN2', [0,1,2], lay2_image)
					lay2_image = self.activation_function(lay2_image)
					lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=int(self.config['n_filter']/16), kernel_size=[5, 5], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay3_image = helper.conv_layer_norm_layer(lay3_image, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						lay3_image = lib.ops.batchnorm.Batchnorm('Encoder.BN3', [0,1,2], lay3_image)
					lay3_image = self.activation_function(lay3_image)
					lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=int(self.config['n_filter']/4), kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay4_image = helper.conv_layer_norm_layer(lay4_image, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						lay4_image = lib.ops.batchnorm.Batchnorm('Encoder.BN4', [0,1,2], lay4_image)
					lay4_image = self.activation_function(lay4_image)
					latent_image = tf.layers.conv2d(inputs=lay4_image, filters=self.config['n_filter'], kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=self.activation_function)
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])	

				if self.config['encoder_mode'] == 'Deterministic':
					latent_flat = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
				if self.config['encoder_mode'] == 'Gaussian':
					latent_mu = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
					latent_pre_scale = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
					latent_flat = latent_mu+tf.nn.softplus(latent_pre_scale)*noise
				
				z_flat = tf.reshape(latent_flat, [-1, x['image'].get_shape().as_list()[1], self.config['n_latent']])

			self.constructed = True
			return z_flat

class Generator():
	def __init__(self, config, name = '/Generator'):
		self.name = name
		self.config = config
		self.activation_function = self.config['gen_activation_function']
		self.normalization_mode = self.config['gen_normalization_mode']
		self.constructed = False

	def forward(self, x, name = ''):
		with tf.variable_scope("Generator", reuse=self.constructed):
			out_dict = {'flat': None, 'image': None}
			if len(self.config['data_properties']['flat']) > 0:
				n_output_size = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['flat']])
				x_batched_inp_flat = tf.reshape(x, [-1,  x.get_shape().as_list()[-1]])
				
				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = self.activation_function)
				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_flat'], activation = self.activation_function)
				lay3_flat = tf.layers.dense(inputs = lay2_flat, units = self.config['n_flat'], activation = self.activation_function)
				flat_param = tf.layers.dense(inputs = lay3_flat, units = n_output_size, activation = None)
				out_dict['flat'] = tf.reshape(flat_param, [-1, x.get_shape().as_list()[1], n_output_size])

			if len(self.config['data_properties']['image']) > 0:

				image_shape = (self.config['data_properties']['image'][0]['size'][-3:-1])
				n_image_size = np.prod(image_shape)
				n_output_channels = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['image']])
				x_batched_inp_flat = tf.reshape(x, [-1,  x.get_shape().as_list()[-1]])

				if image_shape == (28, 28):
					layer_1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = 8*8*self.config['n_filter'], activation = self.activation_function, use_bias = True)
					layer_1 = tf.reshape(layer_1_flat, [-1, 8, 8, self.config['n_filter']])
					layer_2 = tf.layers.conv2d_transpose(inputs=layer_1, filters=int(self.config['n_filter']/4), kernel_size=[5, 5], strides=[2,2], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						layer_2 = helper.conv_layer_norm_layer(layer_2, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						layer_2 = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,1,2], layer_2)
					layer_2 = self.activation_function(layer_2)
					layer_3 = tf.layers.conv2d_transpose(inputs=layer_2, filters=int(self.config['n_filter']/4), kernel_size=[5, 5], strides=[1,1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						layer_3 = helper.conv_layer_norm_layer(layer_3, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						layer_3 = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,1,2], layer_3)
					layer_3 = self.activation_function(layer_3)
					layer_4 = tf.layers.conv2d_transpose(inputs=layer_3, filters=int(self.config['n_filter']/8), kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						layer_4 = helper.conv_layer_norm_layer(layer_4, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						layer_4 = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0,1,2], layer_4)
					layer_4 = self.activation_function(layer_4)
					layer_5 = tf.layers.conv2d_transpose(inputs=layer_4, filters=int(self.config['n_filter']/8), kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						layer_5 = helper.conv_layer_norm_layer(layer_5, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						layer_5 = lib.ops.batchnorm.Batchnorm('Generator.BN5', [0,1,2], layer_5)
					layer_5 = self.activation_function(layer_5)
					output = tf.layers.conv2d_transpose(inputs=layer_5, filters=n_output_channels, kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=None)
					image_param = helper.tf_center_crop_image(output, resize_ratios=[28,28])

				# # # 32x32xn_channels 'n_filter': 512, 28 sec, 3.4 in 20 epochs
				if image_shape == (32, 32): ## works with 128 ? 
					layer_1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = 8*8*self.config['n_filter'], activation = self.activation_function, use_bias = True)
					layer_1 = tf.reshape(layer_1_flat, [-1, 8, 8, self.config['n_filter']])
					layer_2 = tf.layers.conv2d_transpose(inputs=layer_1, filters=int(self.config['n_filter']/2), kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						layer_2 = helper.conv_layer_norm_layer(layer_2, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						layer_2 = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,1,2], layer_2)
					layer_2 = self.activation_function(layer_2)
					layer_3 = tf.layers.conv2d_transpose(inputs=layer_2, filters=int(self.config['n_filter']/4), kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						layer_3 = helper.conv_layer_norm_layer(layer_3, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						layer_3 = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,1,2], layer_3)
					layer_3 = self.activation_function(layer_3)
					layer_4 = tf.layers.conv2d_transpose(inputs=layer_3, filters=int(self.config['n_filter']/4), kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						layer_4 = helper.conv_layer_norm_layer(layer_4, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						layer_4 = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0,1,2], layer_4)
					layer_4 = self.activation_function(layer_4)
					output = tf.layers.conv2d_transpose(inputs=layer_4, filters=n_output_channels, kernel_size=[5, 5], strides=[2,2], padding="valid", use_bias=True, activation=None)
					image_param = helper.tf_center_crop_image(output, resize_ratios=[32,32])

				# # # 64x64xn_channels
				if image_shape == (64, 64):
					layer_1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = 11*11*self.config['n_filter'], activation = self.activation_function, use_bias = True)
					layer_1 = tf.reshape(layer_1_flat, [-1, 11, 11, self.config['n_filter']])
					layer_2 = tf.layers.conv2d_transpose(inputs=layer_1, filters=int(self.config['n_filter']/4), kernel_size=[5, 5], strides=[2,2], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						layer_2 = helper.conv_layer_norm_layer(layer_2, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						layer_2 = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,1,2], layer_2)
					layer_2 = self.activation_function(layer_2)
					layer_3 = tf.layers.conv2d_transpose(inputs=layer_2, filters=int(self.config['n_filter']/16), kernel_size=[5, 5], strides=[2,2], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						layer_3 = helper.conv_layer_norm_layer(layer_3, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						layer_3 = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,1,2], layer_3)
					layer_3 = self.activation_function(layer_3)
					layer_4 = tf.layers.conv2d_transpose(inputs=layer_3, filters=int(self.config['n_filter']/16), kernel_size=[5, 5], strides=[1,1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						layer_4 = helper.conv_layer_norm_layer(layer_4, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						layer_4 = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,1,2], layer_4)
					layer_4 = self.activation_function(layer_4)
					layer_5 = tf.layers.conv2d_transpose(inputs=layer_4, filters=int(self.config['n_filter']/32), kernel_size=[5, 5], strides=[1,1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						layer_5 = helper.conv_layer_norm_layer(layer_5, channel_index=3)
					elif self.normalization_mode == 'Batch Norm': 
						layer_5 = lib.ops.batchnorm.Batchnorm('Generator.BN5', [0,1,2], layer_5)
					layer_5 = self.activation_function(layer_5)
					output = tf.layers.conv2d_transpose(inputs=layer_5, filters=n_output_channels, kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=None)
					image_param = output
				

				out_dict['image'] = tf.reshape(image_param, [-1, x.get_shape().as_list()[1], *image_shape, n_output_channels])

			self.constructed = True
			return out_dict




