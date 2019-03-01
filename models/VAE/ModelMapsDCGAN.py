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

activation_function = tf.nn.relu
# activation_function = tf.nn.elu
# activation_function = tf.nn.softplus
# activation_function = tf.nn.tanh
# activation_function = helper.selu
# activation_function = helper.lrelu

class PriorMap():
	def __init__(self, config, name = '/PriorMap'):
		self.name = name
		self.config = config
		self.constructed = False
 
	def forward(self, x, name = ''):
		with tf.variable_scope("PriorMap", reuse=self.constructed):
			input_flat = x[0]
			mu_log_sig = tf.zeros(shape=(tf.shape(input_flat)[0], 2*self.config['n_latent']))
			# range_uniform = tf.concat([(-1)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent'])), tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent']))], axis=1)
			self.constructed = True
			return mu_log_sig
			# return range_uniform

class EpsilonMap():
	def __init__(self, config, name = '/EpsilonMap'):
		self.name = name
		self.config = config
		self.constructed = False
 
	def forward(self, x, name = ''):
		with tf.variable_scope("EpsilonMap", reuse=self.constructed):
			input_flat = x[0]
			mu_log_sig = tf.zeros(shape=(tf.shape(input_flat)[0], 2*self.config['n_latent']))
			# range_uniform = tf.concat([(-1)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent'])), tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent']))], axis=1)
			self.constructed = True
			return mu_log_sig
			# return range_uniform

class Encoder():
	def __init__(self, config, name = '/Encoder'):
		self.name = name
		self.config = config
		self.activation_function = activation_function
		self.constructed = False

	def forward(self, x, epsilon_sample=None, name = ''):
		with tf.variable_scope("Encoder", reuse=self.constructed):
			if len(self.config['data_properties']['flat']) > 0:
				n_output_size = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['flat']])				
				x_batched_inp_flat = tf.reshape(x['flat'], [-1,  *x['flat'].get_shape().as_list()[2:]])
				
				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_encoder'], activation = activation_function)
				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_encoder'], activation = activation_function)
				latent_flat = tf.layers.dense(inputs = lay2_flat, units = self.config['n_latent'], activation = None)
				z_flat = tf.reshape(latent_flat, [-1, x['flat'].get_shape().as_list()[1], self.config['n_latent']])
				z = z_flat

			if len(self.config['data_properties']['image']) > 0:								
				image_shape = (self.config['data_properties']['image'][0]['size'][-3:-1])
				n_image_size = np.prod(image_shape)
				n_output_channels = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['image']])
				x_batched_inp_image = tf.reshape(x['image'], [-1,  *x['image'].get_shape().as_list()[2:]])
				x_batched_inp_flat = tf.reshape(x['image'], [-1,  np.prod(x['image'].get_shape().as_list()[2:])])

				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_decoder'], activation = activation_function)
				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_decoder'], activation = activation_function)
				lay3_flat = tf.layers.dense(inputs = lay2_flat, units = self.config['n_decoder'], activation = activation_function)
				lay4_flat = tf.layers.dense(inputs = lay3_flat, units = self.config['n_decoder'], activation = activation_function)
				latent_image_flat = tf.layers.dense(inputs = lay4_flat, units = self.config['n_decoder'], activation = activation_function)



				# # # 28x28xn_channels
				# if image_shape == (28, 28):
				# 	n_filters = 16
				# 	lay1_image = tf.layers.conv2d(inputs=x_batched_inp_image, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
				# 	lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
				# 	lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=2*n_filters, kernel_size=[4, 4], strides=[1, 1], padding="valid", activation=activation_function)
				# 	latent_image = tf.layers.conv2d(inputs=lay3_image, filters=3*n_filters, kernel_size=[3, 3], strides=[1, 1], padding="valid", activation=activation_function)
				# 	latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])

				# # 64x64xn_channels
				# if image_shape == (64, 64):
				# 	n_filters = 16
				# 	lay1_image = tf.layers.conv2d(inputs=x_batched_inp_image, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
				# 	lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
				# 	lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=3*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
				# 	lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=4*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
				# 	latent_image = tf.layers.conv2d(inputs=lay4_image, filters=4*n_filters, kernel_size=[3, 3], strides=[1, 1], padding="valid", activation=activation_function)
				# 	latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])

				# # # 128x128xn_channels
				# if image_shape == (128, 128):
				# 	n_filters = 12
				# 	lay1_image = tf.layers.conv2d(inputs=x_batched_inp_image, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
				# 	lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
				# 	lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=3*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
				# 	lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=4*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
				# 	lay5_image = tf.layers.conv2d(inputs=lay4_image, filters=5*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
				# 	latent_image = tf.layers.conv2d(inputs=lay5_image, filters=6*n_filters, kernel_size=[3, 3], strides=[1, 1], padding="valid", activation=activation_function)
				# 	latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])

				if epsilon_sample is not None:
					lay1_flat = tf.layers.dense(inputs = tf.concat([latent_image_flat, epsilon_sample],axis=-1), units = self.config['n_encoder'], activation = activation_function)
				else:
					lay1_flat = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_encoder'], activation = activation_function)
				# lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_encoder'], activation = activation_function)
				latent_param_flat = tf.layers.dense(inputs = lay1_flat, units = 2*self.config['n_latent'], activation = None)
				z_param_flat = tf.reshape(latent_param_flat, [-1, x['image'].get_shape().as_list()[1], 2*self.config['n_latent']])

			self.constructed = True
			return z_param_flat

class EncodingPlan():
	def __init__(self, config, name = '/EncodingPlan'):
		self.name = name
		self.config = config
		self.activation_function = activation_function
		self.constructed = False

	def forward(self, x, epsilon_sample=None, name = ''):
		with tf.variable_scope("EncodingPlan", reuse=self.constructed):
			if len(self.config['data_properties']['flat']) > 0:
				n_output_size = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['flat']])				
				x_batched_inp_flat = tf.reshape(x['flat'], [-1,  *x['flat'].get_shape().as_list()[2:]])
				
				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_encoder'], activation = activation_function)
				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_encoder'], activation = activation_function)
				latent_flat = tf.layers.dense(inputs = lay2_flat, units = self.config['n_latent'], activation = None)
				z_flat = tf.reshape(latent_flat, [-1, x['flat'].get_shape().as_list()[1], self.config['n_latent']])
				z = z_flat

			if len(self.config['data_properties']['image']) > 0:								
				image_shape = (self.config['data_properties']['image'][0]['size'][-3:-1])
				n_image_size = np.prod(image_shape)
				n_output_channels = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['image']])
				x_batched_inp_image = tf.reshape(x['image'], [-1,  *x['image'].get_shape().as_list()[2:]])

				# # 28x28xn_channels
				if image_shape == (28, 28):
					n_filters = 16
					lay1_image = tf.layers.conv2d(inputs=x_batched_inp_image, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
					lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
					lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=2*n_filters, kernel_size=[4, 4], strides=[1, 1], padding="valid", activation=activation_function)
					latent_image = tf.layers.conv2d(inputs=lay3_image, filters=3*n_filters, kernel_size=[3, 3], strides=[1, 1], padding="valid", activation=activation_function)
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])

				# 64x64xn_channels
				if image_shape == (64, 64):
					n_filters = 16
					lay1_image = tf.layers.conv2d(inputs=x_batched_inp_image, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
					lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
					lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=3*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
					lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=4*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
					latent_image = tf.layers.conv2d(inputs=lay4_image, filters=4*n_filters, kernel_size=[3, 3], strides=[1, 1], padding="valid", activation=activation_function)
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])

				# # 128x128xn_channels
				if image_shape == (128, 128):
					n_filters = 12
					lay1_image = tf.layers.conv2d(inputs=x_batched_inp_image, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
					lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
					lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=3*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=activation_function)
					lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=4*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
					lay5_image = tf.layers.conv2d(inputs=lay4_image, filters=5*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
					latent_image = tf.layers.conv2d(inputs=lay5_image, filters=6*n_filters, kernel_size=[3, 3], strides=[1, 1], padding="valid", activation=activation_function)
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])

				if epsilon_sample is not None:
					lay1_flat = tf.layers.dense(inputs = tf.concat([latent_image_flat, epsilon_sample],axis=-1), units = self.config['n_encoder'], activation = activation_function)
				else:
					lay1_flat = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_encoder'], activation = activation_function)
				# lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_encoder'], activation = activation_function)
				latent_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_latent'], activation = None)
				z_flat = tf.reshape(latent_flat, [-1, x['image'].get_shape().as_list()[1], self.config['n_latent']])
				z = z_flat

			self.constructed = True
			return z

class Decoder():
	def __init__(self, config, name = '/Decoder'):
		self.name = name
		self.config = config
		self.activation_function = activation_function
		self.constructed = False

	def forward(self, x, name = ''):
		with tf.variable_scope("Decoder", reuse=self.constructed):
			out_dict = {'flat': None, 'image': None}
			if len(self.config['data_properties']['flat']) > 0:
				n_output_size = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['flat']])
				x_batched_inp_flat = tf.reshape(x, [-1,  x.get_shape().as_list()[-1]])
				
				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_decoder'], activation = activation_function)
				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_decoder'], activation = activation_function)
				lay3_flat = tf.layers.dense(inputs = lay2_flat, units = self.config['n_decoder'], activation = activation_function)
				lay4_flat = tf.layers.dense(inputs = lay3_flat, units = self.config['n_decoder'], activation = activation_function)
				lay5_flat = tf.layers.dense(inputs = lay4_flat, units = self.config['n_decoder'], activation = activation_function)
				flat_param = tf.layers.dense(inputs = lay5_flat, units = n_output_size, activation = None)
				out_dict['flat'] = tf.reshape(flat_param, [-1, x.get_shape().as_list()[1], n_output_size])

			if len(self.config['data_properties']['image']) > 0:
				image_shape = (self.config['data_properties']['image'][0]['size'][-3:-1])
				n_image_size = np.prod(image_shape)
				n_output_channels = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['image']])
				x_batched_inp_flat = tf.reshape(x, [-1,  x.get_shape().as_list()[-1]])

				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_decoder'], activation = activation_function)
				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_decoder'], activation = activation_function)
				lay4_flat = tf.layers.dense(inputs = lay2_flat, units = self.config['n_decoder'], activation = activation_function)
				image_param_flat = tf.layers.dense(inputs = lay4_flat, units = n_output_channels*n_image_size, activation = None)
				# image_param_flat_mean = tf.layers.dense(inputs = lay4_flat, units = int(n_output_channels*n_image_size/2), activation = None)
				# image_param_flat_log_std = tf.layers.dense(inputs = lay4_flat, units = int(n_output_channels*n_image_size/2), activation = None)
				# image_param_flat = tf.concat([image_param_flat_mean, image_param_flat_log_std], axis=-1)
				out_dict['image'] = tf.reshape(image_param_flat, [-1, x.get_shape().as_list()[1], *image_shape, n_output_channels])

				# # # 28x28xn_channels
				# if image_shape == (28, 28):
				# 	b_batch_norm = False
				# 	n_filters = 64

				# 	lay1_image = tf.layers.dense(inputs = x_batched_inp_flat, units = 8*n_filters*4*4, activation = None)
				# 	lay2_image = tf.reshape(lay1_image, [-1, 4, 4, 8*n_filters])
				# 	if b_batch_norm: lay3_image = tf.nn.relu(helper.batch_norm()(lay2_image))
				# 	else: lay3_image = tf.nn.relu(lay2_image) #h0
					
				# 	lay4_image = tf.layers.conv2d_transpose(inputs=lay3_image, filters=4*n_filters, kernel_size=[5, 5], strides=[1, 1], activation=None)
				# 	if b_batch_norm: lay5_image = tf.nn.relu(helper.batch_norm()(lay4_image))
				# 	else: lay5_image = tf.nn.relu(lay4_image) #h1

				# 	lay6_image = tf.layers.conv2d_transpose(inputs=lay5_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], activation=None)
				# 	lay7_image = helper.tf_center_crop_image(lay6_image, resize_ratios=[16,16])
				# 	if b_batch_norm: lay8_image = tf.nn.relu(helper.batch_norm()(lay7_image))
				# 	else: lay8_image = tf.nn.relu(lay7_image) #h2

				# 	lay9_image = tf.layers.conv2d_transpose(inputs=lay8_image, filters=n_output_channels, kernel_size=[5, 5], strides=[2, 2], activation=None)
				# 	lay10_image = helper.tf_center_crop_image(lay9_image, resize_ratios=[28,28])
				# 	image_param = lay10_image

				# # 64x64xn_channels
				# if image_shape == (64, 64):
				# 	b_batch_norm = False
				# 	n_filters = 64
				# 	lay1_image = tf.layers.dense(inputs = x_batched_inp_flat, units = 8*n_filters*4*4, activation = None)

				# 	lay2_image = tf.reshape(lay1_image, [-1, 4, 4, 8*n_filters])
				# 	if b_batch_norm: lay3_image = tf.nn.relu(helper.batch_norm()(lay2_image))
				# 	else: lay3_image = tf.nn.relu(lay2_image) #h0

				# 	lay4_image = tf.layers.conv2d_transpose(inputs=lay3_image, filters=4*n_filters, kernel_size=[5, 5], strides=[1, 1], activation=None)
				# 	if b_batch_norm: lay5_image = tf.nn.relu(helper.batch_norm()(lay4_image))
				# 	else: lay5_image = tf.nn.relu(lay4_image) #h1

				# 	lay6_image = tf.layers.conv2d_transpose(inputs=lay5_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], activation=None)
				# 	lay7_image = helper.tf_center_crop_image(lay6_image, resize_ratios=[16,16])
				# 	if b_batch_norm: lay8_image = tf.nn.relu(helper.batch_norm()(lay7_image))
				# 	else: lay8_image = tf.nn.relu(lay7_image) #h2

				# 	lay9_image = tf.layers.conv2d_transpose(inputs=lay8_image, filters=1*n_filters, kernel_size=[5, 5], strides=[2, 2], activation=None)
				# 	lay10_image = helper.tf_center_crop_image(lay9_image, resize_ratios=[32,32])
				# 	if b_batch_norm: lay11_image = tf.nn.relu(helper.batch_norm()(lay10_image))
				# 	else: lay11_image = tf.nn.relu(lay10_image) #h3
					
				# 	lay12_image = tf.layers.conv2d_transpose(inputs=lay11_image, filters=n_output_channels, kernel_size=[5, 5], strides=[2, 2], activation=None)
				# 	lay13_image = helper.tf_center_crop_image(lay12_image, resize_ratios=[64,64])
				# 	image_param = lay13_image

				# # 128x128xn_channels
				# if image_shape == (128, 128):
				# 	b_batch_norm = False
				# 	n_filters = 64
				# 	lay1_image = tf.layers.dense(inputs = x_batched_inp_flat, units = 8*n_filters*4*4, activation = None)

				# 	lay2_image = tf.reshape(lay1_image, [-1, 4, 4, 8*n_filters])
				# 	if b_batch_norm: lay3_image = tf.nn.relu(helper.batch_norm()(lay2_image))
				# 	else: lay3_image = tf.nn.relu(lay2_image) #h0

				# 	lay4_image = tf.layers.conv2d_transpose(inputs=lay3_image, filters=4*n_filters, kernel_size=[5, 5], strides=[1, 1], activation=None)
				# 	if b_batch_norm: lay5_image = tf.nn.relu(helper.batch_norm()(lay4_image))
				# 	else: lay5_image = tf.nn.relu(lay4_image) #h1

				# 	lay6_image = tf.layers.conv2d_transpose(inputs=lay5_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], activation=None)
				# 	lay7_image = helper.tf_center_crop_image(lay6_image, resize_ratios=[16,16])
				# 	if b_batch_norm: lay8_image = tf.nn.relu(helper.batch_norm()(lay7_image))
				# 	else: lay8_image = tf.nn.relu(lay7_image) #h2

				# 	lay9_image = tf.layers.conv2d_transpose(inputs=lay8_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], activation=None)
				# 	lay10_image = helper.tf_center_crop_image(lay9_image, resize_ratios=[32,32])
				# 	if b_batch_norm: lay11_image = tf.nn.relu(helper.batch_norm()(lay10_image))
				# 	else: lay11_image = tf.nn.relu(lay10_image) #h3

				# 	lay12_image = tf.layers.conv2d_transpose(inputs=lay11_image, filters=1*n_filters, kernel_size=[5, 5], strides=[2, 2], activation=None)
				# 	lay13_image = helper.tf_center_crop_image(lay12_image, resize_ratios=[64,64])
				# 	if b_batch_norm: lay14_image = tf.nn.relu(helper.batch_norm()(lay13_image))
				# 	else: lay14_image = tf.nn.relu(lay13_image) #h3

				# 	lay15_image = tf.layers.conv2d_transpose(inputs=lay14_image, filters=n_output_channels, kernel_size=[5, 5], strides=[2, 2], activation=None)
				# 	lay16_image = helper.tf_center_crop_image(lay15_image, resize_ratios=[128,128])
				# 	image_param = lay16_image

				# out_dict['image'] = tf.reshape(image_param, [-1, x.get_shape().as_list()[1], *image_shape, n_output_channels])

			self.constructed = True
			return out_dict

class Discriminator():
	def __init__(self, config, name = '/Discriminator'):
		self.name = name
		self.activation_function = activation_function
		self.config = config
		self.constructed = False
	
	def forward(self, x, name = ''):
		# activation_function = tf.nn.tanh ################################################   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ################################################
		with tf.variable_scope("Discriminator", reuse=self.constructed):
			outputs = []
			if x['flat'] is not None:
				x_batched_inp_flat = tf.reshape(x['flat'], [-1,  *x['flat'].get_shape().as_list()[2:]])
				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_encoder'], activation = activation_function)
				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_encoder'], activation = activation_function)
				lay3_flat = tf.layers.dense(inputs = lay2_flat, units = self.config['n_encoder'], activation = activation_function)
				lay4_flat = tf.layers.dense(inputs = lay3_flat, units = self.config['n_encoder'], activation = activation_function)
				lay5_flat = tf.layers.dense(inputs = lay4_flat, units = self.config['n_encoder'], activation = activation_function)
				lay6_flat = tf.layers.dense(inputs = lay5_flat, units = 1, activation = None)
				outputs.append(tf.reshape(lay6_flat, [-1, 1, 1]))

			if x['image'] is not None: 
				image_shape = (self.config['data_properties']['image'][0]['size'][-3:-1])
				x_batched_inp_image = tf.reshape(x['image'], [-1,  *x['image'].get_shape().as_list()[2:]])

				# # 28x28xn_channels
				if image_shape == (28, 28):
					n_filters = 64
					b_batch_norm = False
					lay1_image = helper.lrelu(tf.layers.conv2d(inputs=x_batched_inp_image, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="same"))

					lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid")
					if b_batch_norm: lay3_image = helper.lrelu(helper.batch_norm()(lay2_image)) 
					else: lay3_image = helper.lrelu(lay2_image)

					lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=4*n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid")
					if b_batch_norm: lay5_image = helper.lrelu(helper.batch_norm()(lay4_image)) 
					else: lay5_image = helper.lrelu(lay4_image)

					lay6_image = tf.layers.conv2d(inputs=lay5_image, filters=8*n_filters, kernel_size=[3, 3], strides=[1, 1], padding="valid")
					if b_batch_norm: lay7_image = helper.lrelu(helper.batch_norm()(lay6_image)) 
					else: lay7_image = helper.lrelu(lay6_image)

					critic_image = tf.layers.conv2d(inputs=lay7_image, filters=1, kernel_size=[4, 4], strides=[1, 1], padding="valid", activation=None)
					pdb.set_trace()

				# 64x64xn_channels
				if image_shape == (64, 64):
					n_filters = 64
					b_batch_norm = False
					lay1_image = helper.lrelu(tf.layers.conv2d(inputs=x_batched_inp_image, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="same"))
					
					lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="same")
					if b_batch_norm: lay3_image = helper.lrelu(helper.batch_norm()(lay2_image)) 
					else: lay3_image = helper.lrelu(lay2_image)

					lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=4*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="same")
					if b_batch_norm: lay5_image = helper.lrelu(helper.batch_norm()(lay4_image)) 
					else: lay5_image = helper.lrelu(lay4_image)

					lay6_image = tf.layers.conv2d(inputs=lay5_image, filters=8*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="same")
					if b_batch_norm: lay7_image = helper.lrelu(helper.batch_norm()(lay6_image)) 
					else: lay7_image = helper.lrelu(lay6_image)

					critic_image = tf.layers.conv2d(inputs=lay7_image, filters=1, kernel_size=[4, 4], strides=[1, 1], padding="valid", activation=None)

				# 128x128xn_channels
				if image_shape == (128, 128):
					n_filters = 32
					b_batch_norm = False
					lay1_image = helper.lrelu(tf.layers.conv2d(inputs=x_batched_inp_image, filters=n_filters, kernel_size=[5, 5], strides=[2, 2], padding="same"))
					
					lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="same")
					if b_batch_norm: lay3_image = helper.lrelu(helper.batch_norm()(lay2_image)) 
					else: lay3_image = helper.lrelu(lay2_image)

					lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=4*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="same")
					if b_batch_norm: lay5_image = helper.lrelu(helper.batch_norm()(lay4_image)) 
					else: lay5_image = helper.lrelu(lay4_image)

					lay6_image = tf.layers.conv2d(inputs=lay5_image, filters=8*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="same")
					if b_batch_norm: lay7_image = helper.lrelu(helper.batch_norm()(lay6_image)) 
					else: lay7_image = helper.lrelu(lay6_image)

					lay8_image = tf.layers.conv2d(inputs=lay7_image, filters=16*n_filters, kernel_size=[5, 5], strides=[2, 2], padding="same")
					if b_batch_norm: lay9_image = helper.lrelu(helper.batch_norm()(lay8_image)) 
					else: lay9_image = helper.lrelu(lay8_image)

					critic_image = tf.layers.conv2d(inputs=lay8_image, filters=1, kernel_size=[4, 4], strides=[1, 1], padding="valid", activation=None)

				critic = tf.reshape(critic_image, [-1, x['image'].get_shape().as_list()[1], 1])
				outputs.append(critic)

			if len(outputs) > 1: 
				pdb.set_trace()
				merged_input = tf.concat(outputs, axis=-1)
				x_batched_inp_image = tf.reshape(x['image'], [-1,  *x['image'].get_shape().as_list()[2:]])
				input_merged = tf.reshape(merged_input, [-1, merged_input.get_shape().as_list()[-1]])
				lay1_merged = tf.layers.dense(inputs = input_merged, units = 1, activation = None)
				enc = tf.reshape(lay1_merged, [-1, x['flat'].get_shape().as_list()[1], 1])
			else: enc = outputs[0]
			self.constructed = True
			return enc








# # self.observation_map = f_o(n_decoder | n_state+n_latent+n_context). f_o(x_t | h<t, z_t, e(c_t))
# class ObservationMap():
# 	def __init__(self, config, name = '/ObservationMap'):
# 		self.name = name
# 		self.activation_function = activation_function
# 		self.config = config
# 		self.constructed = False

# 	def forward(self, x, name = ''):
# 		with tf.variable_scope("ObservationMap", reuse=self.constructed):
# 			z_new = x[0]
# 			input_flat = z_new
# 			decoder_hid = input_flat
# 			# decoder_hid = tf.layers.dense(inputs = input_flat, units = self.config['n_decoder'], activation = activation_function)
# 			self.constructed = True
# 			return decoder_hid

					# n_filters = 32
					# lay1_image = tf.layers.conv2d_transpose(inputs=x_batched_inp_image, filters=n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
					# lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=n_output_channels, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=None)
					# lay2_image = x_batched_inp_image+lay2_image
					# lay3_image = tf.layers.conv2d_transpose(inputs=lay2_image, filters=n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
					# lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=n_output_channels, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=None)
					# lay4_image = lay2_image+lay4_image
					# lay5_image = tf.layers.conv2d_transpose(inputs=lay4_image, filters=n_filters, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=activation_function)
					# y_image = tf.layers.conv2d(inputs=lay5_image, filters=n_output_channels, kernel_size=[5, 5], strides=[1, 1], padding="valid", activation=None)





				# image_sig = (1-1/tf.exp(0.1*t))+0*image_sig
				# image_sig = 1-image_neg_sig/tf.exp(0.1*t)
				# image_sig = noise

# # self.input_decoder = f_d(n_observed | n_decoder). f_d()
# class TransportPlan():
# 	def __init__(self, config, name = '/TransportPlan'):
# 		self.name = name
# 		self.config = config
# 		self.activation_function = activation_function
# 		self.constructed = False

# 	def forward(self, x, name = ''):
# 		with tf.variable_scope("TransportPlan", reuse=self.constructed):
# 			out_dict = {'flat': None, 'image': None}
# 			if len(self.config['data_properties']['flat']) > 0:
# 				n_output_size = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['flat']])				
# 				x_batched_inp_flat = tf.reshape(x['flat'], [-1,  *x['flat'].get_shape().as_list()[2:]])
				
# 				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_decoder'], activation = activation_function)
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_decoder'], activation = activation_function)
# 				flat_mu = tf.layers.dense(inputs = lay2_flat, units = n_output_size, activation = None)

# 				lay1_sig_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_decoder'], activation = activation_function)
# 				lay2_sig_flat = tf.layers.dense(inputs = lay1_sig_flat, units = self.config['n_decoder'], activation = activation_function)
# 				flat_sig = tf.layers.dense(inputs = lay2_sig_flat, units = 1, activation = tf.nn.sigmoid)

# 				flat_param = flat_sig*x_batched_inp_flat+(1-flat_sig)*(flat_mu)
# 				out_dict['flat'] = tf.reshape(flat_param, [-1, x['flat'].get_shape().as_list()[1], n_output_size])

# 			if len(self.config['data_properties']['image']) > 0:								
# 				image_shape = (self.config['data_properties']['image'][0]['size'][-3:-1])
# 				n_image_size = np.prod(image_shape)
# 				n_output_channels = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['image']])
# 				x_batched_inp_flat = tf.reshape(x['image'], [-1, np.prod(x['image'].get_shape().as_list()[2:])])
				
# 				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = 500, activation = activation_function)
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = 500, activation = activation_function)
# 				image_mu = tf.layers.dense(inputs = lay2_flat, units = n_output_channels*n_image_size, activation = tf.nn.sigmoid)

# 				lay1_sig_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = 500, activation = activation_function)
# 				lay2_sig_flat = tf.layers.dense(inputs = lay1_sig_flat, units = 500, activation = activation_function)
# 				image_sig = tf.layers.dense(inputs = lay2_sig_flat, units = 1, activation = tf.nn.sigmoid)

# 				image_param = (image_sig)*x_batched_inp_flat+(1-image_sig)*image_mu
# 				out_dict['image'] = tf.reshape(image_param, [-1, x['image'].get_shape().as_list()[1], *image_shape, n_output_channels])

# 				# image_param_flat = tf.reshape(image_param, [-1, x['image'].get_shape().as_list()[1], *image_shape[:2], n_output_channels*image_shape[-1]])
# 				# n_output_channels = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['image']])
# 				# x_batched_inp_image = tf.reshape(x['image'], [-1,  *x['image'].get_shape().as_list()[2:]])
# 				# lay1_image = tf.layers.conv2d_transpose(inputs=x_batched_inp_image, filters=64, kernel_size=[3, 3], strides=[1, 1], padding="valid", activation=activation_function)
# 				# lay2_image = tf.layers.conv2d_transpose(inputs=lay1_image, filters=64, kernel_size=[3, 3], strides=[1, 1], padding="valid", activation=activation_function)
# 				# lay3_image = tf.layers.conv2d_transpose(inputs=lay2_image, filters=n_output_channels, kernel_size=[3, 3], strides=[1, 1], padding="valid", activation=None)
# 				# image_param = helper.tf_center_crop_image(lay3_image, resize_ratios=[28,28])
# 				# image_param_image = tf.reshape(image_param, [-1, x['image'].get_shape().as_list()[1], *image_param.get_shape().as_list()[1:]])
# 				# out_dict['image'] = image_param_image#+image_param_flat				

# 			self.constructed = True
# 			return out_dict






				# image_sig = tf.layers.dense(inputs = lay2_sig_flat, units = 1, activation = tf.nn.sigmoid)*(1-2*1e-7)+1e-7
				# image_sig = tf.layers.dense(inputs = lay2_sig_flat, units = 1, activation = tf.nn.sigmoid)*(1-0.6-2*1e-7)+1e-7+0.6
				# image_sig = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = 1, use_bias = False, activation = tf.nn.sigmoid)*(1-2*1e-6)+1e-6




				# image_sig = helper.tf_print(image_sig, [tf.reduce_min(image_sig), tf.reduce_max(image_sig)])


				# image_sig = helper.tf_print(image_sig, [tf.reduce_min(image_sig), tf.reduce_max(image_sig)])


# class GeneratorDecoder():
# 	def __init__(self, config, name = '/GeneratorDecoder'):
# 		self.name = name
# 		self.activation_function = activation_function
# 		self.config = config
# 		self.constructed = False
	
# 	def forward(self, x, name = ''):
# 		with tf.variable_scope("GeneratorDecoder", reuse=self.constructed):
# 			outputs = []
# 			if x['flat'] is not None:
# 				x_batched_inp_flat = tf.reshape(x['flat'], [-1,  *x['flat'].get_shape().as_list()[2:]])
# 				# x_batched_inp_flat = x_batched_inp_flat+0.2*tf.random_normal(shape=tf.shape(x_batched_inp_flat))

# 				lay6_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_decoder'], activation = activation_function)
# 				# lay7_flat = tf.layers.dense(inputs = lay6_flat, units = self.config['n_decoder'], activation = activation_function)
# 				lay8_flat = tf.layers.dense(inputs = lay6_flat, units = self.config['n_latent'], activation = None)
# 				rec = tf.reshape(lay8_flat, [-1, 1, self.config['n_latent']])

# 			if x['image'] is not None: 
# 				x_batched_inp_flat = tf.reshape(x['image'], [-1, np.prod(x['image'].get_shape().as_list()[2:])])
# 				# x_batched_inp_flat = x_batched_inp_flat+0.1*tf.random_normal(shape=tf.shape(x_batched_inp_flat))

# 				lay5_flat = activation_function(tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_decoder'], activation = None))
# 				# lay6_flat = activation_function(tf.layers.dense(inputs = lay5_flat, units = self.config['n_decoder'], activation = None))
# 				# lay7_flat = activation_function(tf.layers.dense(inputs = lay6_flat, units = self.config['n_decoder'], activation = None))
# 				lay8_flat = tf.layers.dense(inputs = lay5_flat, units = self.config['n_latent'], activation = None)
# 				rec = tf.reshape(lay8_flat, [-1, 1, self.config['n_latent']])
			
# 			self.constructed = True
# 			return rec

# class DiscriminatorEncoder():
# 	def __init__(self, config, name = '/DiscriminatorEncoder'):
# 		self.name = name
# 		self.activation_function = activation_function
# 		self.config = config
# 		self.constructed = False
	
# 	def forward(self, x, name = ''):
# 		with tf.variable_scope("DiscriminatorEncoder", reuse=self.constructed):
# 			if x['flat'] is not None:
# 				x_batched_inp_flat = tf.reshape(x['flat'], [-1,  *x['flat'].get_shape().as_list()[2:]])
# 				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_encoder'], activation = activation_function)
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_encoder'], activation = activation_function)
# 			if x['image'] is not None: 
# 				x_batched_inp_flat = tf.reshape(x['image'], [-1, np.prod(x['image'].get_shape().as_list()[2:])])
# 				lay1_flat = tf.nn.relu(tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_encoder'], activation = None))
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_encoder'], activation = activation_function)

# 			self.constructed = True
# 			return lay2_flat

# class DiscriminatorDecoder():
# 	def __init__(self, config, name = '/DiscriminatorDecoder'):
# 		self.name = name
# 		self.activation_function = activation_function
# 		self.config = config
# 		self.constructed = False
	
# 	def forward(self, x, output_template, name = ''):
# 		with tf.variable_scope("DiscriminatorDecoder", reuse=self.constructed):
# 			out_dict = {'flat': None, 'image': None}
# 			if output_template['flat'] is not None:
# 				output_size = np.prod(output_template['flat'].get_shape().as_list()[2:])
# 				lay1_flat = tf.layers.dense(inputs = x, units = self.config['n_encoder'], activation = activation_function)
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_encoder'], activation = activation_function)
# 				lay3_flat = tf.layers.dense(inputs = lay2_flat, units = output_size, activation = None)
# 				out_dict['flat'] = tf.reshape(lay3_flat, [-1, *output_template['flat'].get_shape().as_list()[1:]])
# 			if output_template['image'] is not None: 
# 				output_size = np.prod(output_template['image'].get_shape().as_list()[2:])
# 				lay1_flat = tf.layers.dense(inputs = x, units = self.config['n_encoder'], activation = activation_function)
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_encoder'], activation = activation_function)
# 				lay3_flat = tf.layers.dense(inputs = lay2_flat, units = output_size, activation = None)
# 				out_dict['image'] = tf.reshape(lay3_flat, [-1, *output_template['image'].get_shape().as_list()[1:]])

# 			self.constructed = True
# 			return out_dict

				
				# m = (1-flat_sig)
				# flat_sig = helper.tf_print(flat_sig, [tf.reduce_min(flat_sig), tf.reduce_max(flat_sig)])
				# m = helper.tf_print(m, [tf.reduce_min(m), tf.reduce_max(m)])
				# flat_mu = helper.tf_print(flat_mu, [tf.reduce_min(flat_mu), tf.reduce_max(flat_mu)])





# # self.input_decoder = f_d(n_observed | n_decoder). f_d()
# class TransportPlan():
# 	def __init__(self, config, name = '/TransportPlan'):
# 		self.name = name
# 		self.config = config
# 		self.activation_function = activation_function
# 		self.constructed = False

# 	def forward(self, x, aux_sample=None, noise=None, t=None, name = ''):
# 		with tf.variable_scope("TransportPlan", reuse=self.constructed):
# 			out_dict = {'flat': None, 'image': None}
# 			if len(self.config['data_properties']['flat']) > 0:
# 				n_output_size = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['flat']])				
# 				x_batched_inp_flat = tf.reshape(x['flat'], [-1,  *x['flat'].get_shape().as_list()[2:]])

# 				lay1_sig_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_encoder'], activation = activation_function)
# 				lay2_sig_flat = tf.layers.dense(inputs = lay1_sig_flat, units = self.config['n_encoder'], activation = activation_function)
# 				gating = tf.layers.dense(inputs = lay2_sig_flat, units = 1, activation = tf.nn.sigmoid)
		
# 				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_encoder'], activation = activation_function)
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_encoder'], activation = activation_function)
# 				y_flat = tf.layers.dense(inputs = lay2_flat, units = n_output_size, activation = None)

# 				# flat_param = gating*x_batched_inp_flat+(1-gating)*(y_flat)
# 				flat_param = y_flat
# 				out_dict['flat'] = tf.reshape(flat_param, [-1, x['flat'].get_shape().as_list()[1], n_output_size])

# 			if len(self.config['data_properties']['image']) > 0:								
# 				image_shape = (self.config['data_properties']['image'][0]['size'][-3:-1])
# 				n_image_size = np.prod(image_shape)
# 				n_output_channels = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['image']])
# 				x_batched_inp_flat = tf.reshape(x['image'], [-1, np.prod(x['image'].get_shape().as_list()[2:])])

# 				strength = 15/255.
# 				n_steps = 3
# 				input_tt = x_batched_inp_flat
# 				for i in range(n_steps):
# 					lay1_flat = tf.layers.dense(inputs = input_tt, units = 100, activation = activation_function)
# 					y_addition = tf.layers.dense(inputs = lay1_flat, units = n_output_channels*n_image_size, activation = tf.nn.sigmoid)
# 					input_tt = input_tt+(strength/n_steps)*y_addition
				
# 				image_param = input_tt
# 				gating = noise

# 				out_dict['image'] = tf.reshape(image_param, [-1, x['image'].get_shape().as_list()[1], *image_shape, n_output_channels])

# 			self.constructed = True
# 			return out_dict, gating


