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

import models.WGANGPCannon.tflib as lib
import models.WGANGPCannon.tflib.ops.linear
import models.WGANGPCannon.tflib.ops.conv2d
import models.WGANGPCannon.tflib.ops.batchnorm
import models.WGANGPCannon.tflib.ops.deconv2d
import models.WGANGPCannon.tflib.save_images
import models.WGANGPCannon.tflib.cifar10
import models.WGANGPCannon.tflib.inception_score
import models.WGANGPCannon.tflib.plot

# def Encoder(inputs, n_samples):
#     n_latents = LATENT_SIZE
#     noise = tf.random_normal([n_samples, n_latents])

#     output = tf.reshape(inputs, [-1, 3, 32, 32])

#     output = lib.ops.conv2d.Conv2D('Encoder.1', 3, DIM, 5, output, stride=2)
#     output = LeakyReLU(output)

#     output = lib.ops.conv2d.Conv2D('Encoder.2', DIM, 2*DIM, 5, output, stride=2)
#     output = LeakyReLU(output)

#     output = lib.ops.conv2d.Conv2D('Encoder.3', 2*DIM, 4*DIM, 5, output, stride=2)
#     output = LeakyReLU(output)

#     output = tf.reshape(output, [-1, 4*4*4*DIM])
#     output = tf.concat([output, noise], axis=1)

#     output = lib.ops.linear.Linear('Encoder.4', 4*4*4*DIM+n_latents, 400, output)
#     output = LeakyReLU(output)

#     output = lib.ops.linear.Linear('Encoder.Output', 400, n_latents, output)

#     return output


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

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

class Diverger():
	def __init__(self, config, name = '/Diverger'):
		self.name = name
		self.config = config
		self.activation_function = self.config['div_activation_function']
		self.normalization_mode = self.config['div_normalization_mode']
		self.constructed = False

	def forward(self, z, name = ''):
		with tf.variable_scope("Diverger", reuse=self.constructed):
			z_batched_inp_flat = tf.reshape(z, [-1,  *z.get_shape().as_list()[2:]])
			lay1_flat = tf.layers.dense(inputs = z_batched_inp_flat, units = 4*self.config['n_latent'], activation = self.activation_function, use_bias = True)
			lay2_flat = tf.layers.dense(inputs = lay1_flat, units = 4*self.config['n_latent'], activation = self.activation_function, use_bias = True)
			lay3_flat = tf.layers.dense(inputs = lay2_flat, units = 4*self.config['n_latent'], activation = self.activation_function, use_bias = True)
			critic_flat = tf.layers.dense(inputs = lay3_flat, units = 1, activation = tf.nn.sigmoid)

			self.constructed = True
			return tf.reshape(critic_flat, [-1, 1, 1])

class Encoder():
	def __init__(self, config, n_output = None, name = '/Encoder'):
		self.name = name
		self.config = config
		self.activation_function = self.config['enc_activation_function']
		self.normalization_mode = self.config['enc_normalization_mode']
		if n_output is None: self.n_output = self.config['n_latent']
		else: self.n_output = n_output
		self.constructed = False

	def forward(self, x, noise=None, name = ''):
		with tf.variable_scope("Encoder", reuse=self.constructed):
			if len(self.config['data_properties']['flat']) > 0:
				n_output_size = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['flat']])				
				x_batched_inp_flat = tf.reshape(x['flat'], [-1,  *x['flat'].get_shape().as_list()[2:]])
				
				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = self.activation_function)
				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_flat'], activation = self.activation_function)
				latent_flat = tf.layers.dense(inputs = lay2_flat, units = self.n_output, activation = None)
				z_flat = tf.reshape(latent_flat, [-1, x['flat'].get_shape().as_list()[1], self.n_output])
				z = z_flat

			if len(self.config['data_properties']['image']) > 0:								
				image_shape = (self.config['data_properties']['image'][0]['size'][-3:-1])
				n_image_size = np.prod(image_shape)
				n_output_channels = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['image']])
				x_batched_inp_image = tf.reshape(x['image'], [-1,  *x['image'].get_shape().as_list()[2:]])
				
				if self.config['encoder_mode'] == 'Deterministic' or self.config['encoder_mode'] == 'Gaussian' or self.config['encoder_mode'] == 'UnivApproxNoSpatial':
					image_input = x_batched_inp_image
				if self.config['encoder_mode'] == 'UnivApprox' or self.config['encoder_mode'] == 'UnivApproxSine':
					noise_spatial = tf.tile(noise[:, np.newaxis, np.newaxis, :], [1, *x_batched_inp_image.get_shape().as_list()[1:3], 1])
					x_and_noise_image = tf.concat([x_batched_inp_image, noise_spatial], axis=-1)
					image_input = x_and_noise_image
				
				# # 28x28xn_channels
				if image_shape == (28, 28):

					lay1_image = tf.layers.conv2d(inputs=image_input, filters=self.config['n_filter'], kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay2_image = self.activation_function(helper.conv_layer_norm_layer(lay1_image))
					elif self.normalization_mode == 'Batch Norm': 
						lay2_image = self.activation_function(helper.batch_norm()(lay1_image))
					else: lay2_image = self.activation_function(lay1_image)

					lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=1*self.config['n_filter'], kernel_size=[5, 5], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay4_image = self.activation_function(helper.conv_layer_norm_layer(lay3_image))
					elif self.normalization_mode == 'Batch Norm': 
						lay4_image = self.activation_function(helper.batch_norm()(lay3_image))
					else: lay4_image = self.activation_function(lay3_image)

					lay5_image = tf.layers.conv2d(inputs=lay4_image, filters=2*self.config['n_filter'], kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay6_image = self.activation_function(helper.conv_layer_norm_layer(lay5_image))
					elif self.normalization_mode == 'Batch Norm': 
						lay6_image = self.activation_function(helper.batch_norm()(lay5_image))
					else: lay6_image = self.activation_function(lay5_image)

					latent_image = tf.layers.conv2d(inputs=lay6_image, filters=2*self.config['n_filter'], kernel_size=[3, 3], strides=[1, 1], padding="valid", use_bias=True, activation=self.activation_function)
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])

				# # 32x32xn_channels
				if image_shape == (32, 32):
					
					lay1_image = tf.layers.conv2d(inputs=image_input, filters=self.config['n_filter'], kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay2_image = self.activation_function(helper.conv_layer_norm_layer(lay1_image))
					elif self.normalization_mode == 'Batch Norm': 
						lay2_image = self.activation_function(helper.batch_norm()(lay1_image))
					else: lay2_image = self.activation_function(lay1_image)

					lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=1*self.config['n_filter'], kernel_size=[5, 5], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay4_image = self.activation_function(helper.conv_layer_norm_layer(lay3_image))
					elif self.normalization_mode == 'Batch Norm': 
						lay4_image = self.activation_function(helper.batch_norm()(lay3_image))
					else: lay4_image = self.activation_function(lay3_image)
					
					lay5_image = tf.layers.conv2d(inputs=lay4_image, filters=2*self.config['n_filter'], kernel_size=[5, 5], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay6_image = self.activation_function(helper.conv_layer_norm_layer(lay5_image))
					elif self.normalization_mode == 'Batch Norm': 
						lay6_image = self.activation_function(helper.batch_norm()(lay5_image))
					else: lay6_image = self.activation_function(lay5_image)

					latent_image = tf.layers.conv2d(inputs=lay6_image, filters=2*self.config['n_filter'], kernel_size=[3, 3], strides=[1, 1], padding="valid", use_bias=True, activation=self.activation_function)
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])
					
				# 64x64xn_channels
				if image_shape == (64, 64):

					lay1_image = tf.layers.conv2d(inputs=image_input, filters=self.config['n_filter'], kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay2_image = self.activation_function(helper.conv_layer_norm_layer(lay1_image))
					elif self.normalization_mode == 'Batch Norm': 
						lay2_image = self.activation_function(helper.batch_norm()(lay1_image))
					else: lay2_image = self.activation_function(lay1_image)

					lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=self.config['n_filter'], kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay4_image = self.activation_function(helper.conv_layer_norm_layer(lay3_image))
					elif self.normalization_mode == 'Batch Norm': 
						lay4_image = self.activation_function(helper.batch_norm()(lay3_image))
					else: lay4_image = self.activation_function(lay3_image)

					lay5_image = tf.layers.conv2d(inputs=lay4_image, filters=2*self.config['n_filter'], kernel_size=[5, 5], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay6_image = self.activation_function(helper.conv_layer_norm_layer(lay5_image))
					elif self.normalization_mode == 'Batch Norm': 
						lay6_image = self.activation_function(helper.batch_norm()(lay5_image))
					else: lay6_image = self.activation_function(lay5_image)

					lay7_image = tf.layers.conv2d(inputs=lay6_image, filters=3*self.config['n_filter'], kernel_size=[5, 5], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					if self.normalization_mode == 'Layer Norm': 
						lay8_image = self.activation_function(helper.conv_layer_norm_layer(lay7_image))
					elif self.normalization_mode == 'Batch Norm': 
						lay8_image = self.activation_function(helper.batch_norm()(lay7_image))
					else: lay8_image = self.activation_function(lay7_image)
					
					latent_image = tf.layers.conv2d(inputs=lay8_image, filters=3*self.config['n_filter'], kernel_size=[3, 3], strides=[1, 1], padding="valid", use_bias=True, activation=self.activation_function)
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])

				if self.config['encoder_mode'] == 'Deterministic':
					lay1_flat = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_flat'], use_bias = True, activation = self.activation_function)
					latent_flat = tf.layers.dense(inputs = lay1_flat, units = self.n_output, use_bias = True, activation = None)
				if self.config['encoder_mode'] == 'Gaussian':
					lay1_flat = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_flat'], use_bias = True, activation = self.activation_function)
					latent_mu = tf.layers.dense(inputs = lay1_flat, units = self.n_output, use_bias = True, activation = None)
					latent_log_sig = tf.layers.dense(inputs = lay1_flat, units = self.n_output, use_bias = True, activation = None)
					latent_flat = latent_mu+tf.nn.softplus(latent_log_sig)*noise
				if self.config['encoder_mode'] == 'UnivApprox' or self.config['encoder_mode'] == 'UnivApproxNoSpatial':
					lay1_concat = tf.layers.dense(inputs = tf.concat([latent_image_flat, noise],axis=-1), units = self.config['n_flat'], use_bias = True, activation = self.activation_function)
					latent_flat = tf.layers.dense(inputs = lay1_concat, units = self.n_output, use_bias = True, activation = None)
				if self.config['encoder_mode'] == 'UnivApproxSine':
					lay1_concat = tf.layers.dense(inputs = tf.concat([latent_image_flat, noise],axis=-1), units = self.config['n_flat'], use_bias = True, activation = self.activation_function)
					latent_correction = tf.layers.dense(inputs = lay1_concat, units = self.n_output, use_bias = True, activation = None)
					latent_output = tf.layers.dense(inputs = lay1_concat, units = self.n_output, use_bias = True, activation = None)
					latent_flat = latent_output+tf.sin(self.config['enc_sine_freq']*noise)-latent_correction

				z_flat = tf.reshape(latent_flat, [-1, x['image'].get_shape().as_list()[1], self.n_output])
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

				noise = x_batched_inp_flat
				n_latents = self.config['n_latent']
				DIM = self.config['n_filter']

				output = lib.ops.linear.Linear('Generator.Input', n_latents, 4*4*4*DIM, noise)
				output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
				output = tf.nn.relu(output)
				output = tf.reshape(output, [-1, 4*DIM, 4, 4])

				output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
				output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
				output = tf.nn.relu(output)

				output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
				output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
				output = tf.nn.relu(output)

				output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)
				output = tf.nn.sigmoid(output)
				image_param = tf.transpose(output, perm=[0,2,3,1])
				
				out_dict['image'] = tf.reshape(image_param, [-1, x.get_shape().as_list()[1], *image_shape, n_output_channels])

			self.constructed = True
			return out_dict

class Critic():
	def __init__(self, config, name = '/Critic'):
		self.name = name
		self.config = config
		self.activation_function = self.config['cri_activation_function']
		self.normalization_mode = self.config['cri_normalization_mode']
		self.constructed = False

	def forward(self, x, name = ''):
		with tf.variable_scope("Critic", reuse=self.constructed):
			outputs = []
			if x['flat'] is not None:
				x_batched_inp_flat = tf.reshape(x['flat'], [-1,  *x['flat'].get_shape().as_list()[2:]])
				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = self.activation_function)
				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_flat'], activation = self.activation_function)
				lay3_flat = tf.layers.dense(inputs = lay2_flat, units = self.config['n_flat'], activation = self.activation_function)
				lay4_flat = tf.layers.dense(inputs = lay3_flat, units = 1, activation = None)
				outputs.append(tf.reshape(lay4_flat, [-1, 1, 1]))

			if x['image'] is not None: 
				image_shape = (self.config['data_properties']['image'][0]['size'][-3:-1])
				x_batched_inp_image = tf.reshape(x['image'], [-1,  *x['image'].get_shape().as_list()[2:]])

				MODE = 'wgan-gp'
				DIM = self.config['n_filter']
				output = tf.transpose(x_batched_inp_image, perm=[0,3,1,2]) #tf.reshape(inputs, [-1, 3, 32, 32])

				output = lib.ops.conv2d.Conv2D('Critic.1', 3, DIM, 5, output, stride=2)
				output = LeakyReLU(output)

				output = lib.ops.conv2d.Conv2D('Critic.2', DIM, 2*DIM, 5, output, stride=2)
				if MODE != 'wgan-gp':
					pdb.set_trace()
					output = lib.ops.batchnorm.Batchnorm('Critic.BN2', [0,2,3], output)
				output = LeakyReLU(output)

				output = lib.ops.conv2d.Conv2D('Critic.3', 2*DIM, 4*DIM, 5, output, stride=2)
				if MODE != 'wgan-gp':
					pdb.set_trace()
					output = lib.ops.batchnorm.Batchnorm('Critic.BN3', [0,2,3], output)
				output = LeakyReLU(output)

				output = tf.reshape(output, [-1, 4*4*4*DIM])
				output = lib.ops.linear.Linear('Critic.Output', 4*4*4*DIM, 1, output)

				critic_image = output[:, np.newaxis, np.newaxis, :]

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




























				# if noise is not None:
				# 	lay1_noise = helper.lrelu(tf.layers.dense(inputs = noise, units = self.config['n_flat'], use_bias = False, activation = None))
				# 	lay2_noise = (tf.layers.dense(inputs = lay1_noise, units = self.config['n_latent'], use_bias = False, activation = None))
				# 	# lay3_noise = helper.lrelu(tf.layers.dense(inputs = lay1_noise, units = self.config['n_latent'], use_bias = False, activation = None))

				# 	lay1_data = helper.lrelu(tf.layers.dense(inputs = tf.concat([latent_image_flat, noise],axis=-1), units = self.config['n_flat'], use_bias = True, activation = None))
				# 	lay2_data = (tf.layers.dense(inputs = lay1_data, units = self.config['n_latent'], use_bias = True, activation = None))
				# 	# lay3_data = helper.lrelu(tf.layers.dense(inputs = lay1_data, units = self.config['n_latent'], use_bias = True, activation = None))

				# 	# lay1_concat = helper.lrelu(tf.layers.dense(inputs = tf.concat([latent_image_flat, noise],axis=-1), units = self.config['n_flat'], use_bias = True, activation = None))
				# 	# lay2_concat = helper.lrelu(tf.layers.dense(inputs = lay1_concat, units = self.config['n_flat'], use_bias = True, activation = None))
				# 	# lay3_concat = (tf.layers.dense(inputs = lay2_concat, units = self.config['n_latent'], use_bias = True, activation = None))

				# 	# lay1_data_mu = helper.lrelu(tf.layers.dense(inputs = latent_image_flat, units = self.config['n_flat'], use_bias = True, activation = None))
				# 	# lay1_data_logsig = helper.lrelu(tf.layers.dense(inputs = latent_image_flat, units = self.config['n_flat'], use_bias = True, activation = None))
				# 	latent_flat = lay2_noise*lay2_data
				# 	# latent_flat = lay3_noise*lay3_data
				# 	# latent_flat = lay3_concat*noise
				# 	# latent_flat = lay3_concat+noise



# # self.observation_map = f_o(n_flat | n_state+n_latent+n_context). f_o(x_t | h<t, z_t, e(c_t))
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
# 			# decoder_hid = tf.layers.dense(inputs = input_flat, units = self.config['n_flat'], activation = activation_function)
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

# # self.input_decoder = f_d(n_observed | n_flat). f_d()
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
				
# 				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = activation_function)
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_flat'], activation = activation_function)
# 				flat_mu = tf.layers.dense(inputs = lay2_flat, units = n_output_size, activation = None)

# 				lay1_sig_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = activation_function)
# 				lay2_sig_flat = tf.layers.dense(inputs = lay1_sig_flat, units = self.config['n_flat'], activation = activation_function)
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

# 				lay6_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = activation_function)
# 				# lay7_flat = tf.layers.dense(inputs = lay6_flat, units = self.config['n_flat'], activation = activation_function)
# 				lay8_flat = tf.layers.dense(inputs = lay6_flat, units = self.config['n_latent'], activation = None)
# 				rec = tf.reshape(lay8_flat, [-1, 1, self.config['n_latent']])

# 			if x['image'] is not None: 
# 				x_batched_inp_flat = tf.reshape(x['image'], [-1, np.prod(x['image'].get_shape().as_list()[2:])])
# 				# x_batched_inp_flat = x_batched_inp_flat+0.1*tf.random_normal(shape=tf.shape(x_batched_inp_flat))

# 				lay5_flat = activation_function(tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = None))
# 				# lay6_flat = activation_function(tf.layers.dense(inputs = lay5_flat, units = self.config['n_flat'], activation = None))
# 				# lay7_flat = activation_function(tf.layers.dense(inputs = lay6_flat, units = self.config['n_flat'], activation = None))
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
# 				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = activation_function)
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_flat'], activation = activation_function)
# 			if x['image'] is not None: 
# 				x_batched_inp_flat = tf.reshape(x['image'], [-1, np.prod(x['image'].get_shape().as_list()[2:])])
# 				lay1_flat = tf.nn.relu(tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = None))
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_flat'], activation = activation_function)

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
# 				lay1_flat = tf.layers.dense(inputs = x, units = self.config['n_flat'], activation = activation_function)
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_flat'], activation = activation_function)
# 				lay3_flat = tf.layers.dense(inputs = lay2_flat, units = output_size, activation = None)
# 				out_dict['flat'] = tf.reshape(lay3_flat, [-1, *output_template['flat'].get_shape().as_list()[1:]])
# 			if output_template['image'] is not None: 
# 				output_size = np.prod(output_template['image'].get_shape().as_list()[2:])
# 				lay1_flat = tf.layers.dense(inputs = x, units = self.config['n_flat'], activation = activation_function)
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_flat'], activation = activation_function)
# 				lay3_flat = tf.layers.dense(inputs = lay2_flat, units = output_size, activation = None)
# 				out_dict['image'] = tf.reshape(lay3_flat, [-1, *output_template['image'].get_shape().as_list()[1:]])

# 			self.constructed = True
# 			return out_dict

				
				# m = (1-flat_sig)
				# flat_sig = helper.tf_print(flat_sig, [tf.reduce_min(flat_sig), tf.reduce_max(flat_sig)])
				# m = helper.tf_print(m, [tf.reduce_min(m), tf.reduce_max(m)])
				# flat_mu = helper.tf_print(flat_mu, [tf.reduce_min(flat_mu), tf.reduce_max(flat_mu)])





# # self.input_decoder = f_d(n_observed | n_flat). f_d()
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

# 				lay1_sig_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = activation_function)
# 				lay2_sig_flat = tf.layers.dense(inputs = lay1_sig_flat, units = self.config['n_flat'], activation = activation_function)
# 				gating = tf.layers.dense(inputs = lay2_sig_flat, units = 1, activation = tf.nn.sigmoid)
		
# 				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = activation_function)
# 				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_flat'], activation = activation_function)
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


