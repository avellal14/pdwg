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
	
class PriorMapUniform():
	def __init__(self, config, name = '/PriorMapUniform'):
		self.name = name
		self.config = config
		self.constructed = False
 
	def forward(self, x, name = ''):
		with tf.variable_scope("PriorMapUniform", reuse=self.constructed):
			input_flat = x[0]
			range_uniform = tf.concat([(-1)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent'])), (1)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent']))], axis=1)
			self.constructed = True
			return range_uniform

class PriorMapGaussian():
	def __init__(self, config, name = '/PriorMapGaussian'):
		self.name = name
		self.config = config
		self.constructed = False
 
	def forward(self, x, name = ''):
		with tf.variable_scope("PriorMapGaussian", reuse=self.constructed):
			input_flat = x[0]
			mu_log_sig = tf.zeros(shape=(tf.shape(input_flat)[0], 2*self.config['n_latent']))
			self.constructed = True
			return mu_log_sig

class AmbientMap():
	def __init__(self, config, name = '/AmbientMap'):
		self.name = name
		self.config = config
		self.constructed = False
 
	def forward(self, x, name = ''):
		with tf.variable_scope("AmbientMap", reuse=self.constructed):
			input_flat = x[0]
			range_uniform = tf.concat([(0.05)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent'])), (1-0.05)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent']))], axis=1)
			self.constructed = True
			return range_uniform

class PriorMapBeta():
	def __init__(self, config, name = '/PriorMapBeta'):
		self.name = name
		self.config = config
		self.constructed = False
 
	def forward(self, x, name = ''):
		with tf.variable_scope("PriorMapBeta", reuse=self.constructed):
			input_flat = x[0]
			# range_uniform = tf.concat([(10)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent'])), (3)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent']))], axis=1)
			range_uniform = tf.concat([(40)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent'])), (12)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent']))], axis=1)
			self.constructed = True
			return range_uniform

class PriorMapBetaInverted():
	def __init__(self, config, name = '/PriorMapBetaInverted'):
		self.name = name
		self.config = config
		self.constructed = False
 
	def forward(self, x, name = ''):
		with tf.variable_scope("PriorMapBetaInverted", reuse=self.constructed):
			input_flat = x[0]
			# range_uniform = tf.concat([(3)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent'])), (10)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent']))], axis=1)
			range_uniform = tf.concat([(12)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent'])), (40)*tf.ones(shape=(tf.shape(input_flat)[0], self.config['n_latent']))], axis=1)
			self.constructed = True
			return range_uniform

class PriorMapBernoulli():
	def __init__(self, config, name = '/PriorMapBernoulli'):
		self.name = name
		self.config = config
		self.constructed = False
 
	def forward(self, x, name = ''):
		with tf.variable_scope("PriorMapBernoulli", reuse=self.constructed):
			input_flat = x[0]
			pre_mu = tf.zeros(shape=(tf.shape(input_flat)[0], self.config['n_latent']))
			self.constructed = True
			return pre_mu

class FlowMap():
	def __init__(self, config, name = '/FlowMap'):
		self.name = name
		self.config = config
		self.constructed = False

	def forward(self, batch, name = ''):
		with tf.variable_scope("FlowMap", reuse=self.constructed):
			parameters_list = []
			# flow_to_use = transforms.PiecewisePlanarScalingFlow
			# flow_to_use = transforms.RealNVPFlow
			flow_to_use = transforms.NonLinearIARFlow
			parameters_list.append(10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(self.config['n_latent']), use_bias = False, activation = None))
			parameters_list.append(10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(self.config['n_latent']), use_bias = False, activation = None))
			parameters_list.append(10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(self.config['n_latent']), use_bias = False, activation = None))

			n_output = np.prod(batch['observed']['properties']['image'][0]['size'][2:])
			parameters_list.append(1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = transforms.RiemannianFlow.required_num_parameters(self.config['n_latent'], n_output, n_input_NOM=self.config['rnf_prop']['n_input_NOM'], n_output_NOM=self.config['rnf_prop']['n_output_NOM']), use_bias = False, activation = None))
			parameters_list.append(1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = transforms.CompoundRotationFlow.required_num_parameters(n_output), use_bias = False, activation = None))

			self.constructed = True
			return parameters_list

class WolfMap():
	def __init__(self, config, name = '/WolfMap'):
		self.name = name
		self.config = config
		self.constructed = False

	def forward(self, batch, name = ''):
		with tf.variable_scope("WolfMap", reuse=self.constructed):
			parameters_list = []
			# flow_to_use = transforms.PiecewisePlanarScalingFlow
			# flow_to_use = transforms.RealNVPFlow
			flow_to_use = transforms.NonLinearIARFlow
			parameters_list.append(10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(self.config['n_latent']), use_bias = False, activation = None))
			parameters_list.append(10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(self.config['n_latent']), use_bias = False, activation = None))
			parameters_list.append(10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(self.config['n_latent']), use_bias = False, activation = None))
			
			self.constructed = True
			return parameters_list

class FlowMap2():
	def __init__(self, config, name = '/FlowMap'):
		self.name = name
		self.config = config
		self.constructed = False

	def forward(self, name = ''):
		with tf.variable_scope("FlowMap", reuse=self.constructed):
			parameters_list = []
			flow_to_use = transforms.NonLinearIARFlow
			parameters_list.append(1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(2*self.config['n_latent']), use_bias = False, activation = None))
			parameters_list.append(1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(2*self.config['n_latent']), use_bias = False, activation = None))
			parameters_list.append(1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(2*self.config['n_latent']), use_bias = False, activation = None))
			parameters_list.append(1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(2*self.config['n_latent']), use_bias = False, activation = None))
			parameters_list.append(1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(2*self.config['n_latent']), use_bias = False, activation = None))
			parameters_list.append(1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(2*self.config['n_latent']), use_bias = False, activation = None))
			parameters_list.append(1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = flow_to_use.required_num_parameters(2*self.config['n_latent']), use_bias = False, activation = None))

			self.constructed = True
			return parameters_list

class InfoMapGaussian():
	def __init__(self, config, name = '/InfoMap'):
		self.name = name
		self.config = config
		self.activation_function = self.config['div_activation_function']
		self.normalization_mode = self.config['div_normalization_mode']
		self.constructed = False

	def forward(self, z, name = ''):
		with tf.variable_scope("InfoMap", reuse=self.constructed):
			z_batched_inp_flat = tf.reshape(z, [-1,  *z.get_shape().as_list()[1:]])
			lay1_flat = tf.layers.dense(inputs = z_batched_inp_flat, units = 2*self.config['n_latent'], activation = self.activation_function, use_bias = True)
			# lay2_flat = tf.layers.dense(inputs = lay1_flat, units = 2*self.config['n_latent'], activation = self.activation_function, use_bias = True)
			# lay3_flat = tf.layers.dense(inputs = lay2_flat, units = 2*self.config['n_latent'], activation = self.activation_function, use_bias = True)
			lay2_flat = helper.FCResnetLayer(lay1_flat, units = 2*self.config['n_latent'], activation = self.activation_function)
			lay3_flat = helper.FCResnetLayer(lay2_flat, units = 2*self.config['n_latent'], activation = self.activation_function)
			
			mu = tf.layers.dense(inputs = lay3_flat, units = self.config['n_latent'], activation = None)
			if self.config['infomax_mode'] == 'GaussianFixedForAll': log_sig = tf.zeros(shape=tf.shape(mu))
			elif self.config['infomax_mode'] == 'GaussianSingleLearnable': log_sig = tf.tile(tf.layers.dense(inputs = tf.ones(shape=(tf.shape(z)[0], 1)), units = 1, use_bias = False, activation = None), [1, self.config['n_latent']])
			elif self.config['infomax_mode'] == 'GaussianLearnablePerDim': log_sig = tf.layers.dense(inputs = tf.ones(shape=(tf.shape(z)[0], 1)), units = self.config['n_latent'], use_bias = False, activation = None)
			elif self.config['infomax_mode'] == 'GaussianFull': log_sig = tf.layers.dense(inputs = lay3_flat, units = self.config['n_latent'], activation = None) 
			# elif self.config['infomax_mode'] == 'GaussianFullBounded': log_sig = helper.upper_bounded_nonlinearity(tf.layers.dense(inputs = lay3_flat, units = self.config['n_latent'], activation = None), max_value=0.5) 
			# elif self.config['infomax_mode'] == 'GaussianFullBounded': log_sig = helper.lower_bounded_nonlinearity(helper.upper_bounded_nonlinearity(tf.layers.dense(inputs = lay3_flat, units = self.config['n_latent'], activation = None), max_value=1), min_value=-1) 
			elif self.config['infomax_mode'] == 'GaussianFullBounded': log_sig = tf.nn.tanh(tf.layers.dense(inputs = lay3_flat, units = self.config['n_latent'], activation = None)) 

			mu_log_sig = tf.concat([mu, log_sig],axis=1)			
			self.constructed = True
			return mu_log_sig

class InfoMapBernoulli():
	def __init__(self, config, name = '/InfoMap'):
		self.name = name
		self.config = config
		self.activation_function = self.config['div_activation_function']
		self.normalization_mode = self.config['div_normalization_mode']
		self.constructed = False

	def forward(self, z, name = ''):
		with tf.variable_scope("InfoMap", reuse=self.constructed):
			z_batched_inp_flat = tf.reshape(z, [-1,  *z.get_shape().as_list()[1:]])
			lay1_flat = tf.layers.dense(inputs = z_batched_inp_flat, units = 2*self.config['n_latent'], activation = self.activation_function, use_bias = True)
			# lay2_flat = tf.layers.dense(inputs = lay1_flat, units = 2*self.config['n_latent'], activation = self.activation_function, use_bias = True)
			# lay3_flat = tf.layers.dense(inputs = lay2_flat, units = 2*self.config['n_latent'], activation = self.activation_function, use_bias = True)
			lay2_flat = helper.FCResnetLayer(lay1_flat, units = 2*self.config['n_latent'], activation = self.activation_function)
			lay3_flat = helper.FCResnetLayer(lay2_flat, units = 2*self.config['n_latent'], activation = self.activation_function)
			
			pre_mu = tf.layers.dense(inputs = lay3_flat, units = self.config['n_latent'], activation = None)
			self.constructed = True
			return pre_mu

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
			lay1_flat = tf.layers.dense(inputs = z_batched_inp_flat, units = 4*self.config['n_latent'], activation = None, use_bias = True)
			# lay1_flat = helper.dense_layer_norm_layer(lay1_flat)
			lay1_flat = self.activation_function(lay1_flat)

			lay2_flat = tf.layers.dense(inputs = lay1_flat, units = 4*self.config['n_latent'], activation = None, use_bias = True)
			# lay2_flat = helper.dense_layer_norm_layer(lay2_flat)
			lay2_flat = self.activation_function(lay2_flat)

			# lay2_flat = helper.FCResnetLayer(lay1_flat, units = 4*self.config['n_latent'], activation = self.activation_function)
			div_flat = tf.layers.dense(inputs = lay2_flat, units = 1, activation = None)
			self.constructed = True
			return tf.reshape(div_flat, [-1, 1, 1])

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
				x_batched_inp_image = tf.reshape(x['image'], [-1,  *x['image'].get_shape().as_list()[2:]])
				
				if self.config['encoder_mode'] == 'Deterministic' or self.config['encoder_mode'] == 'Gaussian' or self.config['encoder_mode'] == 'GaussianLeastVariance' or 'UnivApproxNoSpatial' in self.config['encoder_mode']:
					image_input = x_batched_inp_image
				if self.config['encoder_mode'] == 'UnivApprox' or self.config['encoder_mode'] == 'UnivApproxSine':
					noise_spatial = tf.tile(noise[:, np.newaxis, np.newaxis, :], [1, *x_batched_inp_image.get_shape().as_list()[1:3], 1])
					x_and_noise_image = tf.concat([x_batched_inp_image, noise_spatial], axis=-1)
					image_input = x_and_noise_image
				
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
					
				# # # # 32x32xn_channels
				# if image_shape == (32, 32): ## works with 512
				# 	lay1_image = tf.layers.conv2d(inputs=image_input, filters=int(self.config['n_filter']/8), kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=None)
				# 	if self.normalization_mode == 'Layer Norm': 
				# 		lay1_image = helper.conv_layer_norm_layer(lay1_image, channel_index=3)
				# 	elif self.normalization_mode == 'Batch Norm': 
				# 		lay1_image = lib.ops.batchnorm.Batchnorm('Encoder.BN1', [0,1,2], lay1_image)
				# 	lay1_image = self.activation_function(lay1_image)
				# 	lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=int(self.config['n_filter']/8), kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=None)
				# 	if self.normalization_mode == 'Layer Norm': 
				# 		lay2_image = helper.conv_layer_norm_layer(lay2_image, channel_index=3)
				# 	elif self.normalization_mode == 'Batch Norm': 
				# 		lay2_image = lib.ops.batchnorm.Batchnorm('Encoder.BN2', [0,1,2], lay2_image)
				# 	lay2_image = self.activation_function(lay2_image)
				# 	lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=int(self.config['n_filter']/4), kernel_size=[4, 4], strides=[1, 1], padding="valid", use_bias=True, activation=None)
				# 	if self.normalization_mode == 'Layer Norm': 
				# 		lay3_image = helper.conv_layer_norm_layer(lay3_image, channel_index=3)
				# 	elif self.normalization_mode == 'Batch Norm': 
				# 		lay3_image = lib.ops.batchnorm.Batchnorm('Encoder.BN3', [0,1,2], lay3_image)
				# 	lay3_image = self.activation_function(lay3_image)
				# 	lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=int(self.config['n_filter']/4), kernel_size=[5, 5], strides=[1, 1], padding="valid", use_bias=True, activation=None)
				# 	if self.normalization_mode == 'Layer Norm': 
				# 		lay4_image = helper.conv_layer_norm_layer(lay4_image, channel_index=3)
				# 	elif self.normalization_mode == 'Batch Norm': 
				# 		lay4_image = lib.ops.batchnorm.Batchnorm('Encoder.BN4', [0,1,2], lay4_image)
				# 	lay4_image = self.activation_function(lay4_image)
				# 	latent_image = tf.layers.conv2d(inputs=lay4_image, filters=self.config['n_filter'], kernel_size=[5, 5], strides=[2, 2], padding="valid", use_bias=True, activation=self.activation_function)
				# 	latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])	

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
					# lay1_flat = tf.layers.dense(inputs = latent_image_flat, units = 2*self.config['n_latent'], use_bias = True, activation = self.activation_function)
					latent_flat_det = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
					latent_flat = latent_flat_det
				if self.config['encoder_mode'] == 'Gaussian':
					# lay1_flat = tf.layers.dense(inputs = latent_image_flat, units = 2*self.config['n_latent'], use_bias = True, activation = self.activation_function)
					latent_flat_det = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
					latent_pre_scale = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
					latent_flat = latent_flat_det+tf.nn.softplus(latent_pre_scale)*noise
				if self.config['encoder_mode'] == 'GaussianLeastVariance':
					# lay1_flat = tf.layers.dense(inputs = latent_image_flat, units = 2*self.config['n_latent'], use_bias = True, activation = self.activation_function)
					latent_flat_det = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
					latent_pre_scale = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
					latent_variance = tf.nn.softplus(latent_pre_scale)+0.1
					latent_flat = latent_flat_det+latent_variance*noise
				if self.config['encoder_mode'] == 'UnivApprox' or 'UnivApproxNoSpatial' in self.config['encoder_mode']:
					# worked for MNIST
					if self.config['encoder_mode'] == 'UnivApproxNoSpatial_dense_comb':
						latent_flat_det = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
						lay1_concat = tf.layers.dense(inputs = tf.concat([latent_flat_det, noise],axis=-1), units = 2*self.config['n_latent'], use_bias = True, activation = self.activation_function)
						# lay2_concat = tf.layers.dense(inputs = lay1_concat, units = 2*self.config['n_latent'], use_bias = True, activation = self.activation_function)
						# lay3_concat = tf.layers.dense(inputs = lay2_concat, units = 2*self.config['n_latent'], use_bias = True, activation = self.activation_function)
						latent_flat_stoch = tf.layers.dense(inputs = lay1_concat, units = self.config['n_latent'], use_bias = False, activation = None)
						latent_flat = latent_flat_stoch

					elif self.config['encoder_mode'] == 'UnivApproxNoSpatial_dense_additive':
						latent_flat_det = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
						lay1_concat = tf.layers.dense(inputs = tf.concat([latent_flat_det, noise],axis=-1), units = 2*self.config['n_latent'], use_bias = True, activation = self.activation_function)
						lay2_concat = tf.layers.dense(inputs = lay1_concat, units = 2*self.config['n_latent'], use_bias = True, activation = self.activation_function)
						lay3_concat = tf.layers.dense(inputs = lay2_concat, units = 2*self.config['n_latent'], use_bias = True, activation = self.activation_function)
						latent_flat_stoch = tf.layers.dense(inputs = lay3_concat, units = self.config['n_latent'], use_bias = False, activation = None)
						latent_flat = latent_flat_det+latent_flat_stoch

					elif self.config['encoder_mode'] == 'UnivApproxNoSpatial_resnet_comb':
						latent_flat_det = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
						lay1_concat = helper.FCResnetLayer_v2(tf.concat([latent_flat_det, noise],axis=-1), units = 2*self.config['n_latent'], reduce_activation = self.activation_function)
						lay2_concat = helper.FCResnetLayer_v2(lay1_concat, units = 2*self.config['n_latent'], reduce_activation = self.activation_function)
						lay3_concat = helper.FCResnetLayer_v2(lay2_concat, units = 2*self.config['n_latent'], reduce_activation = self.activation_function)
						latent_flat_stoch = helper.FCResnetLayer_v2(lay3_concat, units = self.config['n_latent'], reduce_activation = self.activation_function)
						latent_flat = latent_flat_stoch

					# worked for CIFAR
					elif self.config['encoder_mode'] == 'UnivApproxNoSpatial_resnet_additive':
						latent_flat_det = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_latent'], use_bias = True, activation = None)
						lay1_concat = helper.FCResnetLayer_v2(tf.concat([latent_flat_det, noise],axis=-1), units = 2*self.config['n_latent'], reduce_activation = self.activation_function)
						lay2_concat = helper.FCResnetLayer_v2(lay1_concat, units = 2*self.config['n_latent'], reduce_activation = self.activation_function)
						lay3_concat = helper.FCResnetLayer_v2(lay2_concat, units = 2*self.config['n_latent'], reduce_activation = self.activation_function)
						latent_flat_stoch = helper.FCResnetLayer_v2(lay3_concat, units = self.config['n_latent'], reduce_activation = self.activation_function)
						latent_flat = latent_flat_det+latent_flat_stoch

					else: pdb.set_trace()
					
				if self.config['encoder_mode'] == 'UnivApproxSine':
					lay1_reduced = tf.layers.dense(inputs = latent_image_flat, units = 2*self.config['n_latent'], use_bias = True, activation = self.activation_function)
					latent_flat_det = tf.layers.dense(inputs = lay1_reduced, units = self.config['n_latent'], use_bias = True, activation = None)
					lay1_concat = tf.layers.dense(inputs = tf.concat([latent_flat_det, noise],axis=-1), units = 2*self.config['n_latent'], use_bias = True, activation = self.activation_function)
					latent_correction = tf.layers.dense(inputs = lay1_concat, units = self.config['n_latent'], use_bias = True, activation = None)
					latent_output = tf.layers.dense(inputs = lay1_concat, units = self.config['n_latent'], use_bias = True, activation = None)
					latent_flat = latent_output+tf.sin(self.config['enc_sine_freq']*noise)-latent_correction

				z_flat = tf.reshape(latent_flat, [-1, x['image'].get_shape().as_list()[1], self.config['n_latent']])
				z_flat_det = tf.reshape(latent_flat_det, [-1, x['image'].get_shape().as_list()[1], self.config['n_latent']])
				
				# noise_reconst_cost = None
				# if self.config['encoder_mode'] == 'UnivApprox' or 'UnivApproxNoSpatial' in self.config['encoder_mode']:
				# 	rec_layer_1 = tf.layers.dense(inputs = z_flat, units = 2*self.config['n_latent'], activation = self.activation_function, use_bias=True)
				# 	rec_layer_2 = helper.FCResnetLayer(rec_layer_1, units = 2*self.config['n_latent'], activation = self.activation_function)
				# 	noise_reconst = tf.layers.dense(inputs = rec_layer_2, units = noise.get_shape().as_list()[-1], activation = None, use_bias=True)
				# 	noise_diff = noise-noise_reconst
				# 	noise_reconst_cost = tf.reduce_mean(tf.reduce_sum(noise_diff**2, axis = [-1,]))
			
			self.constructed = True
			return z_flat, z_flat_det

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
					output = tf.layers.conv2d_transpose(inputs=layer_5, filters=n_output_channels, kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=tf.nn.sigmoid)
					image_param = helper.tf_center_crop_image(output, resize_ratios=[28,28])

				# # # # 32x32xn_channels
				# if image_shape == (32, 32): ## works with 128 ? 
				# 	layer_1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = 8*8*self.config['n_filter'], activation = self.activation_function, use_bias = True)
				# 	layer_1 = tf.reshape(layer_1_flat, [-1, 8, 8, self.config['n_filter']])
				# 	layer_2 = tf.layers.conv2d_transpose(inputs=layer_1, filters=int(self.config['n_filter']/4), kernel_size=[5, 5], strides=[2,2], padding="valid", use_bias=True, activation=None)
				# 	if self.normalization_mode == 'Layer Norm': 
				# 		layer_2 = helper.conv_layer_norm_layer(layer_2, channel_index=3)
				# 	elif self.normalization_mode == 'Batch Norm': 
				# 		layer_2 = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,1,2], layer_2)
				# 	layer_2 = self.activation_function(layer_2)
				# 	layer_3 = tf.layers.conv2d_transpose(inputs=layer_2, filters=int(self.config['n_filter']/4), kernel_size=[5, 5], strides=[1,1], padding="valid", use_bias=True, activation=None)
				# 	if self.normalization_mode == 'Layer Norm': 
				# 		layer_3 = helper.conv_layer_norm_layer(layer_3, channel_index=3)
				# 	elif self.normalization_mode == 'Batch Norm': 
				# 		layer_3 = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,1,2], layer_3)
				# 	layer_3 = self.activation_function(layer_3)
				# 	layer_4 = tf.layers.conv2d_transpose(inputs=layer_3, filters=int(self.config['n_filter']/8), kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=None)
				# 	if self.normalization_mode == 'Layer Norm': 
				# 		layer_4 = helper.conv_layer_norm_layer(layer_4, channel_index=3)
				# 	elif self.normalization_mode == 'Batch Norm': 
				# 		layer_4 = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0,1,2], layer_4)
				# 	layer_4 = self.activation_function(layer_4)
				# 	layer_5 = tf.layers.conv2d_transpose(inputs=layer_4, filters=int(self.config['n_filter']/8), kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=None)
				# 	if self.normalization_mode == 'Layer Norm': 
				# 		layer_5 = helper.conv_layer_norm_layer(layer_5, channel_index=3)
				# 	elif self.normalization_mode == 'Batch Norm': 
				# 		layer_5 = lib.ops.batchnorm.Batchnorm('Generator.BN5', [0,1,2], layer_5)
				# 	layer_5 = self.activation_function(layer_5)
				# 	output = tf.layers.conv2d_transpose(inputs=layer_5, filters=n_output_channels, kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=tf.nn.sigmoid)
				# 	image_param = output

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
					output = tf.layers.conv2d_transpose(inputs=layer_4, filters=n_output_channels, kernel_size=[5, 5], strides=[2,2], padding="valid", use_bias=True, activation=tf.nn.sigmoid)
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
					output = tf.layers.conv2d_transpose(inputs=layer_5, filters=n_output_channels, kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=tf.nn.sigmoid)
					image_param = output

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
				x_batched_inp_image_flattened = tf.reshape(x['image'], [-1, np.prod(x['image'].get_shape().as_list()[2:])])

				# # 28x28xn_channels
				if image_shape == (28, 28):
					output = tf.transpose(x_batched_inp_image, perm=[0,3,1,2]) #tf.reshape(inputs, [-1, 3, 32, 32])
					output = lib.ops.conv2d.Conv2D('Critic.1', 3, self.config['n_filter'], 5, output, stride=2)
					output = LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.2', self.config['n_filter'], 2*self.config['n_filter'], 5, output, stride=2)
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm': 
						output = lib.ops.batchnorm.Batchnorm('Critic.BN2', [0,2,3], output)
					output = LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.3', 2*self.config['n_filter'], 4*self.config['n_filter'], 5, output, stride=2)					
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm':
						output = lib.ops.batchnorm.Batchnorm('Critic.BN3', [0,2,3], output) 
					output = LeakyReLU(output)

					output = tf.reshape(output, [-1, 4*4*4*self.config['n_filter']])
					output = lib.ops.linear.Linear('Critic.Output', 4*4*4*self.config['n_filter'], 1, output)
					critic_image = output[:, np.newaxis, np.newaxis, :]

				# # 32x32xn_channels
				if image_shape == (32, 32):

					output = tf.transpose(x_batched_inp_image, perm=[0,3,1,2]) #tf.reshape(inputs, [-1, 3, 32, 32])
					output = lib.ops.conv2d.Conv2D('Critic.1', 3, self.config['n_filter'], 5, output, stride=2)
					output = LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.2', self.config['n_filter'], 2*self.config['n_filter'], 5, output, stride=2)
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm': 
						output = lib.ops.batchnorm.Batchnorm('Critic.BN2', [0,2,3], output)
					output = LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.3', 2*self.config['n_filter'], 4*self.config['n_filter'], 5, output, stride=2)					
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm':
						output = lib.ops.batchnorm.Batchnorm('Critic.BN3', [0,2,3], output) 
					output = LeakyReLU(output)

					output = tf.reshape(output, [-1, 4*4*4*self.config['n_filter']])
					output = lib.ops.linear.Linear('Critic.Output', 4*4*4*self.config['n_filter'], 1, output)
					critic_image = output[:, np.newaxis, np.newaxis, :]
					
				# 64x64xn_channels
				if image_shape == (64, 64):
					output = tf.transpose(x_batched_inp_image, perm=[0,3,1,2]) #tf.reshape(inputs, [-1, 3, 32, 32])
					output = lib.ops.conv2d.Conv2D('Critic.1', 3, self.config['n_filter'], 5, output, stride=2)
					output = LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.2', self.config['n_filter'], 2*self.config['n_filter'], 5, output, stride=2)
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm': 
						output = lib.ops.batchnorm.Batchnorm('Critic.BN2', [0,2,3], output)
					output = LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.3', 2*self.config['n_filter'], 4*self.config['n_filter'], 5, output, stride=2)					
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm':
						output = lib.ops.batchnorm.Batchnorm('Critic.BN3', [0,2,3], output) 
					output = LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.4', 4*self.config['n_filter'], 4*self.config['n_filter'], 5, output, stride=2)					
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm':
						output = lib.ops.batchnorm.Batchnorm('Critic.BN4', [0,2,3], output) 
					output = LeakyReLU(output)

					output = tf.reshape(output, [-1, 4*4*4*self.config['n_filter']])
					output = lib.ops.linear.Linear('Critic.Output', 4*4*4*self.config['n_filter'], 1, output)
					critic_image = output[:, np.newaxis, np.newaxis, :]
				
				lay1_flat = tf.layers.dense(inputs = x_batched_inp_image_flattened, units = 1000, activation = tf.nn.leaky_relu)
				lay2_flat = tf.layers.dense(inputs = lay1_flat, units = 1000, activation = None)
				lay3_flat = tf.reduce_mean(lay2_flat**2, axis=[1,], keep_dims=True)

				critic_image_updated = critic_image + lay3_flat[:, np.newaxis, np.newaxis, :]
				# critic_image_updated = lay4_flat[:, np.newaxis, np.newaxis, :]
				print('!!!!!!!!!!!!!!!!!!!!!        I UPDATED THE MODEL MAPS WITH FLAT           !!!!!!!!!!!!!!!!!!!!!')
				critic = tf.reshape(critic_image_updated, [-1, x['image'].get_shape().as_list()[1], 1])
				
				# critic = tf.reshape(critic_image, [-1, x['image'].get_shape().as_list()[1], 1])
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






