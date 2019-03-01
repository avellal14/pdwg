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

class FlowMap():
	def __init__(self, config, name = '/FlowMap'):
		self.name = name
		self.config = config
		self.constructed = False

	def forward(self, name = ''):
		with tf.variable_scope("FlowMap", reuse=self.constructed):
			n_parameter = transforms.HouseholdRotationFlow.required_num_parameters(512)
			parameters = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)

			self.constructed = True
			return parameters


class PriorTransformMap():
    def __init__(self, config, name = '/PriorTransformMap'):
        self.name = name
        self.config = config
        self.constructed = False
        self.activation_function = self.config['gen_activation_function']
 
    def forward(self, z, name = ''):
        with tf.variable_scope("PriorTransformMap", reuse=self.constructed):
            z_batched_inp_flat = tf.reshape(z, [-1,  *z.get_shape().as_list()[2:]])

            lay1_flat = tf.layers.dense(inputs = z_batched_inp_flat, units = 4*self.config['n_latent'], activation = self.activation_function, use_bias = True)
            lay2_flat = helper.FCResnetLayer(lay1_flat, units = 4*self.config['n_latent'], activation = self.activation_function)
            lay3_flat = helper.FCResnetLayer(lay2_flat, units = 4*self.config['n_latent'], activation = self.activation_function)
            transformed_z = tf.layers.dense(inputs = lay3_flat, units = self.config['n_latent'], activation = None)
            transformed_z_flat = tf.reshape(transformed_z, [-1, 1, *z.get_shape().as_list()[2:]])
            
            self.constructed = True
            return transformed_z_flat


class PriorExpandMap():
	def __init__(self, config, name = '/PriorExpandMap'):
		self.name = name
		self.config = config
		self.constructed = False
		self.activation_function = self.config['gen_activation_function']

	def forward(self, z, name = ''):
		with tf.variable_scope("PriorExpandMap", reuse=self.constructed):
			z_batched_inp_flat = tf.reshape(z, [-1,  *z.get_shape().as_list()[2:]])
			input_dim = z_batched_inp_flat.get_shape().as_list()[-1]

			layer_1 = tf.layers.dense(inputs = z_batched_inp_flat, units = self.config['n_flat'], activation = self.activation_function, use_bias=True)
			layer_2 = helper.FCResnetLayer(layer_1, units = self.config['n_latent'], activation = self.activation_function)
			layer_3 = helper.FCResnetLayer(layer_2, units = self.config['n_latent'], activation = self.activation_function)
			out = z_batched_inp_flat+tf.layers.dense(inputs = layer_3, units = input_dim, activation = None, use_bias=True)

			self.constructed = True
			return out

# class PriorExpandMap():
# 	def __init__(self, config, name = '/PriorExpandMap'):
# 		self.name = name
# 		self.config = config
# 		self.constructed = False
# 		self.activation_function = self.config['gen_activation_function']
# 		self.n_steps = 5
# 		self.begin_hidden_size = 10
# 		self.input_dim = None

# 	def forward(self, z, name = ''):
# 		with tf.variable_scope("PriorExpandMap", reuse=self.constructed):
# 			z_batched_inp_flat = tf.reshape(z, [-1,  *z.get_shape().as_list()[2:]])
# 			if self.input_dim is None: 
# 				self.input_dim = z_batched_inp_flat.get_shape().as_list()[-1]

# 			end_hidden_size = min(self.config['n_flat'], self.input_dim/2)
# 			hidden_step_increase = (end_hidden_size-self.begin_hidden_size)/(self.n_steps-1)
# 			hidden_sizes = np.round(np.arange(self.begin_hidden_size, end_hidden_size+hidden_step_increase, hidden_step_increase))
# 			print('PriorExpandMap input size: ', self.input_dim)
# 			print('PriorExpandMap hidden sizes: ', hidden_sizes)

# 			all_out = []
# 			for i in range(self.n_steps):
# 				curr_dim = int(hidden_sizes[i])
# 				curr_input = z_batched_inp_flat[:, :curr_dim]
# 				dummy_out = tf.tile(curr_input, [1, int(np.ceil(self.input_dim/curr_dim))])[:, :self.input_dim]
# 				curr_hidden_1 = tf.layers.dense(inputs = curr_input, units = self.config['n_flat'], activation = self.activation_function, use_bias=True)
# 				curr_hidden_2 = helper.FCResnetLayer(curr_hidden_1, units = self.config['n_latent'], activation = self.activation_function)
# 				curr_hidden_3 = helper.FCResnetLayer(curr_hidden_2, units = self.config['n_latent'], activation = self.activation_function)

# 				curr_out = dummy_out + tf.layers.dense(inputs = curr_hidden_3, units = self.input_dim, activation = None, use_bias=True)
# 				all_out.append(curr_out)

# 			all_out_concat = tf.concat([e[:,np.newaxis,:] for e in all_out], axis=1)
# 			self.constructed = True
# 			return all_out, all_out_concat
				
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
			lay2_flat = helper.FCResnetLayer(lay1_flat, units = 4*self.config['n_latent'], activation = self.activation_function)
			lay3_flat = helper.FCResnetLayer(lay2_flat, units = 4*self.config['n_latent'], activation = self.activation_function)
			critic_flat = tf.layers.dense(inputs = lay3_flat, units = 1, activation = None)

			self.constructed = True
			return tf.reshape(critic_flat, [-1, 1, 1])

class Separator():
	def __init__(self, config, n_output = None, name = '/Separator'):
		self.name = name
		self.config = config
		self.activation_function = self.config['enc_activation_function']
		self.normalization_mode = self.config['enc_normalization_mode']
		if n_output is None: self.n_output = self.config['n_latent']
		else: self.n_output = n_output
		self.constructed = False

	def forward(self, z_plus, name = ''):
		with tf.variable_scope("Separator", reuse=self.constructed):

			epsilon_layer_1 = tf.layers.dense(inputs = z_plus, units = self.config['n_flat'], activation = self.activation_function, use_bias=True)
			epsilon_layer_2 = helper.FCResnetLayer(epsilon_layer_1, units = self.config['n_flat'], activation = self.activation_function)
			epsilon_layer_3 = helper.FCResnetLayer(epsilon_layer_2, units = self.config['n_flat'], activation = self.activation_function)
			epsilon_layer_4 = helper.FCResnetLayer(epsilon_layer_3, units = self.config['n_flat'], activation = self.activation_function)
			epsilon_reconst = tf.layers.dense(inputs = epsilon_layer_4, units = z_plus.get_shape().as_list()[-1], activation = None, use_bias=True)

			# x_transformed_layer_1 = tf.layers.dense(inputs = z_plus, units = self.config['n_flat'], activation = self.activation_function, use_bias=True)
			x_transformed_layer_1 = tf.layers.dense(inputs = tf.concat([z_plus, epsilon_reconst], axis=1), units = self.config['n_flat'], activation = self.activation_function, use_bias=True)
			x_transformed_layer_2 = helper.FCResnetLayer(x_transformed_layer_1, units = self.config['n_flat'], activation = self.activation_function)
			x_transformed_layer_3 = helper.FCResnetLayer(x_transformed_layer_2, units = self.config['n_flat'], activation = self.activation_function)
			x_transformed_layer_4 = helper.FCResnetLayer(x_transformed_layer_3, units = self.config['n_flat'], activation = self.activation_function)
			x_transformed_reconst = tf.layers.dense(inputs = x_transformed_layer_4, units = z_plus.get_shape().as_list()[-1], activation = None, use_bias=True)

			self.constructed = True
			return x_transformed_reconst, epsilon_reconst

class Decomposer():
	def __init__(self, config, n_output = None, name = '/Decomposer'):
		self.name = name
		self.config = config
		self.activation_function = self.config['enc_activation_function']
		self.normalization_mode = self.config['enc_normalization_mode']
		if n_output is None: self.n_output = self.config['n_latent']
		else: self.n_output = n_output
		self.constructed = False
		self.n_steps = 1

	def forward(self, x, noise=None, name = ''):
		with tf.variable_scope("Decomposer", reuse=self.constructed):
			x_batched_inp_flat = tf.reshape(x, [-1,  np.prod(x.get_shape().as_list()[2:])])
			begin_hidden_size = 5
			# end_hidden_size = min(self.config['n_flat'], x_batched_inp_flat.get_shape().as_list()[-1]/2)
			# hidden_step_increase = (end_hidden_size-begin_hidden_size)/(self.n_steps-1)
			# hidden_sizes = np.round(np.arange(begin_hidden_size, end_hidden_size+hidden_step_increase, hidden_step_increase))
			# hidden_sizes = hidden_sizes*0+int(x_batched_inp_flat.get_shape().as_list()[-1]/self.n_steps)
			# hidden_sizes = hidden_sizes*0+self.config['n_flat']
			# hidden_sizes = [self.config['n_flat'],]
			hidden_sizes = [128,]
			print('Decomposer input size: ', x_batched_inp_flat.get_shape().as_list()[-1])
			print('Decomposer hidden sizes: ', hidden_sizes)
			
			all_inputs = []
			all_hidden = []
			all_reconst = []
			curr_input = x_batched_inp_flat
			for i in range(len(hidden_sizes)):
				all_inputs.append(curr_input)
				curr_hidden_size = int(hidden_sizes[i])
				curr_hidden = tf.layers.dense(inputs = curr_input, units = curr_hidden_size, activation = self.activation_function, use_bias=True)
				curr_reconst = tf.layers.dense(inputs = curr_hidden, units = x_batched_inp_flat.get_shape().as_list()[-1], activation = None, use_bias=True)
				curr_input = curr_input-curr_reconst
				all_hidden.append(curr_hidden)
				all_reconst.append(curr_reconst)
			residual = curr_input
			decomposition = [*all_reconst, residual]
			deterioration = [*all_inputs, residual]
			
			decomposition_concat = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in decomposition], axis=1)
			deterioration_concat = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in deterioration], axis=1)
			decomposition_diff = deterioration_concat[:, :-1, :]-decomposition_concat[:, :-1, :]
			reconstruction_costs = tf.reduce_sum(decomposition_diff**2, axis = [-1,])
			
			if len(x.get_shape()) == 3:
				decomposition_out = decomposition_concat
				deterioration_out = deterioration_concat
				decomposition_concat_out = tf.reshape(decomposition_out, [-1, 1, np.prod(decomposition_out.get_shape().as_list()[1:])])
				deterioration_concat_out = tf.reshape(deterioration_out, [-1, 1, np.prod(deterioration_out.get_shape().as_list()[1:])])
			elif len(x.get_shape()) == 5:						
				decomposition_out = tf.reshape(decomposition_concat, [-1, decomposition_concat.get_shape().as_list()[1], *x.get_shape().as_list()[2:]])
				deterioration_out = tf.reshape(deterioration_concat, [-1, deterioration_concat.get_shape().as_list()[1], *x.get_shape().as_list()[2:]])
				decomposition_concat_out = tf.reshape(tf.transpose(decomposition_out, [0,2,3,4,1]), \
					[-1, *decomposition_out.get_shape().as_list()[2:4], decomposition_out.get_shape().as_list()[1]*decomposition_out.get_shape().as_list()[-1]])[:, np.newaxis, :,:,:]        
				deterioration_concat_out = tf.reshape(tf.transpose(deterioration_out, [0,2,3,4,1]), \
					[-1, *deterioration_out.get_shape().as_list()[2:4], deterioration_out.get_shape().as_list()[1]*deterioration_out.get_shape().as_list()[-1]])[:, np.newaxis, :,:,:]        
			else: pdb.set_trace()

			self.constructed = True
			return decomposition_out, deterioration_out, decomposition_concat_out, deterioration_concat_out, all_hidden, reconstruction_costs

# # timescale_beta, starttime_beta = 15, 3
# # beta = helper.hardstep((self.epoch-float(starttime_beta))/float(timescale_beta))
# beta = 1
# self.decomposition_out_posterior, self.deterioration_out_posterior, self.decomposition_concat_out_posterior, self.deterioration_concat_out_posterior, self.all_hidden_posterior, self.decomposer_cost_posterior = self.Decomposer.forward(self.posterior_latent_code[:,np.newaxis,:], beta)
# self.decomposition_out_prior, self.deterioration_out_prior, self.decomposition_concat_out_prior, self.deterioration_concat_out_prior, self.all_hidden_prior, _ = self.Decomposer.forward(self.prior_dist.sample()[:,np.newaxis,:])
# class Decomposer():
# 	def __init__(self, config, n_output = None, name = '/Decomposer'):
# 		self.name = name
# 		self.config = config
# 		self.activation_function = self.config['enc_activation_function']
# 		self.normalization_mode = self.config['enc_normalization_mode']
# 		if n_output is None: self.n_output = self.config['n_latent']
# 		else: self.n_output = n_output
# 		self.constructed = False
# 		self.n_steps = 1

# 	def forward(self, x, beta=1, noise=None, name = ''):
# 		with tf.variable_scope("Decomposer", reuse=self.constructed):
# 			x_batched_inp_flat = tf.reshape(x, [-1,  np.prod(x.get_shape().as_list()[2:])])

# 			all_inputs = []
# 			all_hidden = []
# 			all_reconst = []
# 			curr_input = x_batched_inp_flat
# 			all_inputs.append(curr_input)
# 			input_size = x_batched_inp_flat.get_shape().as_list()[-1]
# 			latent_size = input_size
# 			# latent_size = 64

# 			enc_hidden_1 = tf.layers.dense(inputs = curr_input, units = self.config['n_flat'], activation = self.activation_function, use_bias=True)
# 			enc_hidden = enc_hidden_1+tf.layers.dense(inputs = enc_hidden_1, units = self.config['n_flat'], activation = self.activation_function, use_bias=True)
# 			enc_mu = tf.layers.dense(inputs = enc_hidden, units = latent_size, activation = None, use_bias=True)
# 			enc_log_sig = tf.layers.dense(inputs = enc_hidden, units = latent_size, activation = None, use_bias=True)
# 			prior_dist = distributions.DiagonalGaussianDistribution(params=tf.zeros([tf.shape(enc_mu)[0], 2*latent_size]))
# 			posterior_dist = distributions.DiagonalGaussianDistribution(params=tf.concat([enc_mu, enc_log_sig], axis=1))
# 			latent_sample = posterior_dist.sample()

# 			neg_ent_posterior = posterior_dist.log_pdf(latent_sample)
# 			neg_cross_ent_posterior = prior_dist.log_pdf(latent_sample)

# 			empirical_KL_per_sample = neg_ent_posterior-neg_cross_ent_posterior
# 			KL_per_sample = distributions.KLDivDiagGaussianVsDiagGaussian().forward(posterior_dist, prior_dist)

# 			dec_hidden_1 = tf.layers.dense(inputs = latent_sample, units = self.config['n_flat'], activation = self.activation_function, use_bias=True)
# 			dec_hidden = dec_hidden_1+tf.layers.dense(inputs = dec_hidden_1, units = self.config['n_flat'], activation = self.activation_function, use_bias=True)
# 			dec_mu = tf.layers.dense(inputs = dec_hidden, units = input_size, activation = None, use_bias=True)
# 			dec_log_sig = tf.layers.dense(inputs = dec_hidden, units = input_size, activation = None, use_bias=True)
# 			out_dist = distributions.DiagonalGaussianDistribution(params=tf.concat([dec_mu, dec_log_sig], axis=1))
# 			log_densities = out_dist.log_pdf(curr_input)
			
# 			cost = -tf.reduce_mean(log_densities)+beta*tf.reduce_mean(KL_per_sample)
# 			residual = (curr_input-dec_mu) #/(tf.exp(dec_log_sig)+1e-7)
# 			all_hidden.append(latent_sample)
# 			all_reconst.append(dec_mu)
# 			decomposition = [*all_reconst, residual]
# 			deterioration = [*all_inputs, residual]
			
# 			decomposition_concat = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in decomposition], axis=1)
# 			deterioration_concat = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in deterioration], axis=1)
# 			reconstruction_cost = cost
			
# 			if len(x.get_shape()) == 3:
# 				decomposition_out = decomposition_concat
# 				deterioration_out = deterioration_concat
# 				decomposition_concat_out = tf.reshape(decomposition_out, [-1, 1, np.prod(decomposition_out.get_shape().as_list()[1:])])
# 				deterioration_concat_out = tf.reshape(deterioration_out, [-1, 1, np.prod(deterioration_out.get_shape().as_list()[1:])])
# 			elif len(x.get_shape()) == 5:						
# 				decomposition_out = tf.reshape(decomposition_concat, [-1, decomposition_concat.get_shape().as_list()[1], *x.get_shape().as_list()[2:]])
# 				deterioration_out = tf.reshape(deterioration_concat, [-1, deterioration_concat.get_shape().as_list()[1], *x.get_shape().as_list()[2:]])
# 				decomposition_concat_out = tf.reshape(tf.transpose(decomposition_out, [0,2,3,4,1]), \
# 					[-1, *decomposition_out.get_shape().as_list()[2:4], decomposition_out.get_shape().as_list()[1]*decomposition_out.get_shape().as_list()[-1]])[:, np.newaxis, :,:,:]        
# 				deterioration_concat_out = tf.reshape(tf.transpose(deterioration_out, [0,2,3,4,1]), \
# 					[-1, *deterioration_out.get_shape().as_list()[2:4], deterioration_out.get_shape().as_list()[1]*deterioration_out.get_shape().as_list()[-1]])[:, np.newaxis, :,:,:]        
# 			else: pdb.set_trace()

# 			self.constructed = True
# 			return decomposition_out, deterioration_out, decomposition_concat_out, deterioration_concat_out, all_hidden, reconstruction_cost


# dropout_rate = 0.5
# dropout_noise = tf.stop_gradient(tf.cast(tf.random_uniform(shape=(tf.shape(self.posterior_latent_code)[0], self.config['n_flat'])) < (1-dropout_rate), tf.float32))
# _, _, _, _, _, self.decomposer_cost_posterior = self.Decomposer.forward(self.posterior_latent_code[:,np.newaxis,:], noise=dropout_noise)
# not_dropout_noise = tf.ones(shape=(tf.shape(self.posterior_latent_code)[0], self.config['n_flat']))
# # not_dropout_noise = tf.stop_gradient(tf.cast(tf.random_uniform(shape=(tf.shape(self.posterior_latent_code)[0], self.config['n_flat'])) < (1-dropout_rate), tf.float32))
# self.decomposition_out_posterior, self.deterioration_out_posterior, self.decomposition_concat_out_posterior, self.deterioration_concat_out_posterior, self.all_hidden_posterior, _ = self.Decomposer.forward(self.posterior_latent_code[:,np.newaxis,:], noise=not_dropout_noise)
# self.decomposition_out_prior,     self.deterioration_out_prior,     self.decomposition_concat_out_prior,     self.deterioration_concat_out_prior,     self.all_hidden_prior,     _ = self.Decomposer.forward(self.prior_dist.sample()[:,np.newaxis,:], noise=not_dropout_noise)

# class Decomposer():
# 	def __init__(self, config, n_output = None, name = '/Decomposer'):
# 		self.name = name
# 		self.config = config
# 		self.activation_function = self.config['enc_activation_function']
# 		self.normalization_mode = self.config['enc_normalization_mode']
# 		if n_output is None: self.n_output = self.config['n_latent']
# 		else: self.n_output = n_output
# 		self.constructed = False
# 		self.n_steps = 1

# 	def forward(self, x, noise=None, name = ''):
# 		with tf.variable_scope("Decomposer", reuse=self.constructed):
# 			x_batched_inp_flat = tf.reshape(x, [-1,  np.prod(x.get_shape().as_list()[2:])])

# 			all_inputs = []
# 			all_hidden = []
# 			all_reconst = []
# 			curr_input = x_batched_inp_flat
# 			all_inputs.append(curr_input)
# 			curr_hidden = tf.layers.dense(inputs = curr_input, units = self.config['n_flat'], activation = self.activation_function, use_bias=True)
# 			curr_reconst = tf.layers.dense(inputs = curr_hidden*noise, units = x_batched_inp_flat.get_shape().as_list()[-1], activation = None, use_bias=True)
			
# 			curr_input = curr_input-curr_reconst
# 			all_hidden.append(curr_hidden)
# 			all_reconst.append(curr_reconst)
# 			residual = curr_input
# 			decomposition = [*all_reconst, residual]
# 			deterioration = [*all_inputs, residual]
			
# 			decomposition_concat = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in decomposition], axis=1)
# 			deterioration_concat = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in deterioration], axis=1)
# 			decomposition_diff = deterioration_concat[:, :-1, :]-decomposition_concat[:, :-1, :]
# 			reconstruction_cost = tf.reduce_mean(tf.reduce_sum(decomposition_diff**2, axis = [-1,]))
			
# 			if len(x.get_shape()) == 3:
# 				decomposition_out = decomposition_concat
# 				deterioration_out = deterioration_concat
# 				decomposition_concat_out = tf.reshape(decomposition_out, [-1, 1, np.prod(decomposition_out.get_shape().as_list()[1:])])
# 				deterioration_concat_out = tf.reshape(deterioration_out, [-1, 1, np.prod(deterioration_out.get_shape().as_list()[1:])])
# 			elif len(x.get_shape()) == 5:						
# 				decomposition_out = tf.reshape(decomposition_concat, [-1, decomposition_concat.get_shape().as_list()[1], *x.get_shape().as_list()[2:]])
# 				deterioration_out = tf.reshape(deterioration_concat, [-1, deterioration_concat.get_shape().as_list()[1], *x.get_shape().as_list()[2:]])
# 				decomposition_concat_out = tf.reshape(tf.transpose(decomposition_out, [0,2,3,4,1]), \
# 					[-1, *decomposition_out.get_shape().as_list()[2:4], decomposition_out.get_shape().as_list()[1]*decomposition_out.get_shape().as_list()[-1]])[:, np.newaxis, :,:,:]        
# 				deterioration_concat_out = tf.reshape(tf.transpose(deterioration_out, [0,2,3,4,1]), \
# 					[-1, *deterioration_out.get_shape().as_list()[2:4], deterioration_out.get_shape().as_list()[1]*deterioration_out.get_shape().as_list()[-1]])[:, np.newaxis, :,:,:]        
# 			else: pdb.set_trace()

# 			self.constructed = True
# 			return decomposition_out, deterioration_out, decomposition_concat_out, deterioration_concat_out, all_hidden, reconstruction_cost

# class Decomposer():
# 	def __init__(self, config, n_output = None, name = '/Decomposer'):
# 		self.name = name
# 		self.config = config
# 		self.activation_function = self.config['enc_activation_function']
# 		self.normalization_mode = self.config['enc_normalization_mode']
# 		if n_output is None: self.n_output = self.config['n_latent']
# 		else: self.n_output = n_output
# 		self.constructed = False
# 		self.n_steps = 5

# 	def forward(self, x, noise=None, name = ''):
# 		with tf.variable_scope("Decomposer", reuse=self.constructed):
# 			x_batched_inp_flat = tf.reshape(x, [-1,  np.prod(x.get_shape().as_list()[2:])])
# 			begin_hidden_size = 5
# 			end_hidden_size = min(self.config['n_flat'], x_batched_inp_flat.get_shape().as_list()[-1]/2)
# 			hidden_step_increase = (end_hidden_size-begin_hidden_size)/(self.n_steps-1)
# 			hidden_sizes = np.round(np.arange(begin_hidden_size, end_hidden_size+hidden_step_increase, hidden_step_increase))
# 			# hidden_sizes = hidden_sizes*0+int(x_batched_inp_flat.get_shape().as_list()[-1]/self.n_steps)
# 			# hidden_sizes = hidden_sizes*0+self.config['n_flat']
# 			print('Decomposer input size: ', x_batched_inp_flat.get_shape().as_list()[-1])
# 			print('Decomposer hidden sizes: ', hidden_sizes)
			
# 			all_inputs = []
# 			all_hidden = []
# 			all_reconst = []
# 			curr_input = x_batched_inp_flat
# 			for i in range(self.n_steps):
# 				all_inputs.append(curr_input)
# 				curr_hidden_size = int(hidden_sizes[i])
# 				curr_hidden_1 = tf.layers.dense(inputs = curr_input, units = self.config['n_flat'], activation = self.activation_function, use_bias=True)
# 				curr_hidden = tf.layers.dense(inputs = curr_hidden_1, units = curr_hidden_size, activation = self.activation_function, use_bias=True)
# 				curr_reconst = tf.layers.dense(inputs = curr_hidden, units = x_batched_inp_flat.get_shape().as_list()[-1], activation = None, use_bias=True)
# 				curr_input = curr_input-curr_reconst
# 				all_hidden.append(curr_hidden)
# 				all_reconst.append(curr_reconst)
# 			residual = curr_input
# 			decomposition = [*all_reconst, residual]
# 			deterioration = [*all_inputs, residual]
			
# 			decomposition_concat = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in decomposition], axis=1)
# 			deterioration_concat = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in deterioration], axis=1)
# 			decomposition_diff = deterioration_concat[:, :-1, :]-decomposition_concat[:, :-1, :]
# 			reconstruction_cost = tf.reduce_mean(tf.reduce_sum(decomposition_diff**2, axis = [-1,]))
			
# 			if len(x.get_shape()) == 3:
# 				decomposition_out = decomposition_concat
# 				deterioration_out = deterioration_concat
# 				decomposition_concat_out = tf.reshape(decomposition_out, [-1, 1, np.prod(decomposition_out.get_shape().as_list()[1:])])
# 				deterioration_concat_out = tf.reshape(deterioration_out, [-1, 1, np.prod(deterioration_out.get_shape().as_list()[1:])])
# 			elif len(x.get_shape()) == 5:						
# 				decomposition_out = tf.reshape(decomposition_concat, [-1, decomposition_concat.get_shape().as_list()[1], *x.get_shape().as_list()[2:]])
# 				deterioration_out = tf.reshape(deterioration_concat, [-1, deterioration_concat.get_shape().as_list()[1], *x.get_shape().as_list()[2:]])
# 				decomposition_concat_out = tf.reshape(tf.transpose(decomposition_out, [0,2,3,4,1]), \
# 					[-1, *decomposition_out.get_shape().as_list()[2:4], decomposition_out.get_shape().as_list()[1]*decomposition_out.get_shape().as_list()[-1]])[:, np.newaxis, :,:,:]        
# 				deterioration_concat_out = tf.reshape(tf.transpose(deterioration_out, [0,2,3,4,1]), \
# 					[-1, *deterioration_out.get_shape().as_list()[2:4], deterioration_out.get_shape().as_list()[1]*deterioration_out.get_shape().as_list()[-1]])[:, np.newaxis, :,:,:]        
# 			else: pdb.set_trace()

# 			self.constructed = True
# 			return decomposition_out, deterioration_out, decomposition_concat_out, deterioration_concat_out, all_hidden, reconstruction_cost


class Encoder():
	def __init__(self, config, name = '/Encoder'):
		self.name = name
		self.config = config
		self.activation_function = self.config['enc_activation_function']
		self.normalization_mode = self.config['enc_normalization_mode']
		self.constructed = False

	def forward(self, f, name = ''):
		with tf.variable_scope("Encoder", reuse=self.constructed):
			f_flat = tf.reshape(f, [-1, f.get_shape().as_list()[-1]])
			lay1_flat = helper.FCResnetLayer_v3(f_flat, units = 5*self.config['n_latent'], output_unit_rate=3, activation=self.activation_function, output_activation=self.activation_function)
			lay2_flat = helper.FCResnetLayer_v3(lay1_flat, units = 5*self.config['n_latent'], output_unit_rate=1, activation=self.activation_function, output_activation=self.activation_function)
			lay3_flat = helper.FCResnetLayer_v3(lay2_flat, units = 5*self.config['n_latent'], output_unit_rate=1, activation=self.activation_function, output_activation=self.activation_function)
			z_flat = tf.layers.dense(inputs = lay3_flat, units = self.config['n_latent'], activation = None, use_bias = True)

			self.constructed = True
			return tf.reshape(z_flat, [-1, 1, self.config['n_latent']])

class Generator():
	def __init__(self, config, name = '/Generator'):
		self.name = name
		self.config = config
		self.activation_function = self.config['gen_activation_function']
		self.normalization_mode = self.config['gen_normalization_mode']
		self.constructed = False

	def forward(self, z, name = ''):
		with tf.variable_scope("Generator", reuse=self.constructed):
			z_flat = tf.reshape(z, [-1, z.get_shape().as_list()[-1]])
			lay1_flat = helper.FCResnetLayer_v3(z_flat, units = 5*self.config['n_latent'], output_unit_rate=10, activation=self.activation_function, output_activation=self.activation_function)
			lay2_flat = helper.FCResnetLayer_v3(lay1_flat, units = 5*self.config['n_latent'], output_unit_rate=1, activation=self.activation_function, output_activation=self.activation_function)
			lay3_flat = helper.FCResnetLayer_v3(lay2_flat, units = 5*self.config['n_latent'], output_unit_rate=1, activation=self.activation_function, output_activation=self.activation_function)
			f_flat = tf.layers.dense(inputs = lay3_flat, units = 512, activation = tf.nn.sigmoid, use_bias = True)

			self.constructed = True
			return tf.reshape(f_flat, [-1, 1, 512])

class PreEnc():
	def __init__(self, config, n_output = None, name = '/PreEnc'):
		self.name = name
		self.config = config
		self.activation_function = self.config['enc_activation_function']
		self.normalization_mode = self.config['enc_normalization_mode']
		if n_output is None: self.n_output = 512
		else: self.n_output = n_output
		self.constructed = False

	def forward(self, x, noise=None, name = ''):
		with tf.variable_scope("PreEnc", reuse=self.constructed):
			if len(self.config['data_properties']['flat']) > 0:				
				n_output_size = helper.list_sum([distributions.DistributionsAKA[e['dist']].num_params(e['size'][-1]) for e in self.config['data_properties']['flat']])				
				x_batched_inp_flat = tf.reshape(x['flat'], [-1,  *x['flat'].get_shape().as_list()[2:]])
				
				lay1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = self.config['n_flat'], activation = self.activation_function)
				lay2_flat = helper.FCResnetLayer(lay1_flat, units = self.config['n_flat'], activation = self.activation_function)
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
					# noise_spatial = tf.tile(0.1*noise[:, np.newaxis, np.newaxis, :], [1, *x_batched_inp_image.get_shape().as_list()[1:3], 1])
					noise_spatial = tf.tile(noise[:, np.newaxis, np.newaxis, :], [1, *x_batched_inp_image.get_shape().as_list()[1:3], 1])
					x_and_noise_image = tf.concat([x_batched_inp_image, noise_spatial], axis=-1)
					image_input = x_and_noise_image
				
				reduce_units = self.config['n_filter']
				# # 28x28xn_channels
				if image_shape == (28, 28):					
					lay1_image = helper.ConvResnetLayer_v2(image_input, units = 1*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Downsample')
					lay2_image = helper.ConvResnetLayer_v2(lay1_image, units = 2*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Downsample')
					lay3_image = helper.ConvResnetLayer_v2(lay2_image, units = 4*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Downsample')
					latent_image = helper.ConvResnetLayer_v2(lay3_image, units = 8*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Downsample')
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])

				# # # 32x32xn_channels
				if image_shape == (32, 32): ## works with 512
					lay1_image = tf.layers.conv2d(inputs=image_input, filters=self.config['n_filter'], kernel_size=[4, 4], strides=[2, 2], padding="valid", use_bias=True, activation=None)
					lay1_image = helper.conv_layer_norm_layer(lay1_image, channel_index=3)
					lay1_image = self.activation_function(lay1_image)
					lay2_image = tf.layers.conv2d(inputs=lay1_image, filters=2*self.config['n_filter'], kernel_size=[3, 3], strides=[2, 2], padding="valid", use_bias=True, activation=None)
					lay2_image = helper.conv_layer_norm_layer(lay2_image, channel_index=3)
					lay2_image = self.activation_function(lay2_image)
					lay3_image = tf.layers.conv2d(inputs=lay2_image, filters=2*self.config['n_filter'], kernel_size=[3, 3], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					lay3_image = helper.conv_layer_norm_layer(lay3_image, channel_index=3)
					lay3_image = self.activation_function(lay3_image)
					lay4_image = tf.layers.conv2d(inputs=lay3_image, filters=4*self.config['n_filter'], kernel_size=[3, 3], strides=[1, 1], padding="valid", use_bias=True, activation=None)
					lay4_image = helper.conv_layer_norm_layer(lay4_image, channel_index=3)
					lay4_image = self.activation_function(lay4_image)
					latent_image = tf.layers.conv2d(inputs=lay3_image, filters=4*self.config['n_filter'], kernel_size=[3, 3], strides=[1, 1], padding="valid", use_bias=True, activation=self.activation_function)
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])	

				# 64x64xn_channels
				if image_shape == (64, 64):
					lay1_image = helper.ConvResnetLayer_v2(image_input, units = 1*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Downsample')
					lay2_image = helper.ConvResnetLayer_v2(lay1_image, units = 1*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Downsample')
					lay3_image = helper.ConvResnetLayer_v2(lay2_image, units = 2*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Downsample')
					lay4_image = helper.ConvResnetLayer_v2(lay3_image, units = 4*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Downsample')
					latent_image = helper.ConvResnetLayer_v2(lay3_image, units = 8*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Downsample')
					latent_image_flat = tf.reshape(latent_image, [-1, np.prod(latent_image.get_shape().as_list()[1:])])					

				if self.config['encoder_mode'] == 'Deterministic':
					lay1_flat = tf.layers.dense(inputs = latent_image_flat, units = 2*self.n_output, use_bias = True, activation = self.activation_function)
					latent_flat = tf.layers.dense(inputs = lay1_flat, units = self.n_output, use_bias = True, activation = tf.nn.sigmoid)
					# latent_flat_norm = helper.safe_tf_sqrt(tf.reduce_sum(latent_flat**2, axis=1, keep_dims=True))					
					# latent_flat = latent_flat/(latent_flat_norm+1e-7)
				# if self.config['encoder_mode'] == 'Gaussian':
				# 	lay1_flat = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_flat'], use_bias = True, activation = self.activation_function)
				# 	latent_mu = tf.layers.dense(inputs = lay1_flat, units = self.n_output, use_bias = True, activation = None)
				# 	latent_log_sig = tf.layers.dense(inputs = lay1_flat, units = self.n_output, use_bias = True, activation = None)
				# 	latent_flat = latent_mu+tf.nn.softplus(latent_log_sig)*noise
				# if self.config['encoder_mode'] == 'UnivApprox' or self.config['encoder_mode'] == 'UnivApproxNoSpatial':
				# 	lay1_reduced = tf.layers.dense(inputs = latent_image_flat, units = self.config['n_flat'], use_bias = True, activation = self.activation_function)
				# 	x_transformed = tf.layers.dense(inputs = lay1_reduced, units = self.n_output, use_bias = True, activation = None)
				# 	lay1_concat = tf.layers.dense(inputs = tf.concat([latent_image_flat, x_transformed, noise],axis=-1), units = self.config['n_flat'], use_bias = True, activation = self.activation_function)
				# 	lay2_concat = helper.FCResnetLayer(lay1_concat, units = self.config['n_flat'], activation = self.activation_function)
				# 	lay3_concat = helper.FCResnetLayer(lay2_concat, units = self.config['n_flat'], activation = self.activation_function)
				# 	latent_flat_add = tf.layers.dense(inputs = lay3_concat, units = self.n_output, use_bias = False, activation = None)
				# 	latent_flat = x_transformed+latent_flat_add
				# if self.config['encoder_mode'] == 'UnivApproxSine':
				# 	lay1_concat = tf.layers.dense(inputs = tf.concat([latent_image_flat, noise],axis=-1), units = self.config['n_flat'], use_bias = True, activation = self.activation_function)
				# 	latent_correction = tf.layers.dense(inputs = lay1_concat, units = self.n_output, use_bias = True, activation = None)
				# 	latent_output = tf.layers.dense(inputs = lay1_concat, units = self.n_output, use_bias = True, activation = None)
				# 	latent_flat = latent_output+tf.sin(self.config['enc_sine_freq']*noise)-latent_correction

				z_flat = tf.reshape(latent_flat, [-1, x['image'].get_shape().as_list()[1], self.n_output])
			self.constructed = True
			
			return z_flat

class PostGen():
	def __init__(self, config, name = '/PostGen'):
		self.name = name
		self.config = config
		self.activation_function = self.config['gen_activation_function']
		self.normalization_mode = self.config['gen_normalization_mode']
		self.constructed = False

	def forward(self, x, name = ''):
		with tf.variable_scope("PostGen", reuse=self.constructed):
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

				reduce_units = self.config['n_filter']
				# # 28x28xn_channels
				if image_shape == (28, 28):
					layer_1_flat = helper.FCResnetLayer_v2(x_batched_inp_flat, units=2*2*8*self.config['n_filter'], reduce_units=256, activation=self.activation_function, normalization_mode=self.normalization_mode)
					layer_1 = tf.reshape(layer_1_flat, [-1, 2, 2, 8*self.config['n_filter']])
					layer_2 = helper.ConvResnetLayer_v2(layer_1, units = 4*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Upsample')
					layer_3 = helper.ConvResnetLayer_v2(layer_2, units = 2*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Upsample')
					layer_4 = helper.ConvResnetLayer_v2(layer_3, units = 1*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Upsample')
					output = helper.ConvResnetLayer_v2(layer_4, units = n_output_channels, reduce_units=reduce_units, activation=tf.nn.sigmoid, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Upsample')
					image_param = output[:, 2:-2, 2:-2, :]

				# # # 32x32xn_channels
				if image_shape == (32, 32): ## works with 128 ? 
					layer_1_flat = tf.layers.dense(inputs = x_batched_inp_flat, units = 8*8*self.config['n_filter'], activation = self.activation_function, use_bias = True)
					layer_1 = tf.reshape(layer_1_flat, [-1, 8, 8, self.config['n_filter']])
					layer_2 = tf.layers.conv2d_transpose(inputs=layer_1, filters=self.config['n_filter'], kernel_size=[3, 3], strides=[1,1], padding="valid", use_bias=True, activation=None)
					layer_2 = helper.conv_layer_norm_layer(layer_2, channel_index=3)
					layer_2 = self.activation_function(layer_2)
					layer_3 = tf.layers.conv2d_transpose(inputs=layer_2, filters=self.config['n_filter'], kernel_size=[3, 3], strides=[1,1], padding="valid", use_bias=True, activation=None)
					layer_3 = helper.conv_layer_norm_layer(layer_3, channel_index=3)
					layer_3 = self.activation_function(layer_3)
					layer_4 = tf.layers.conv2d_transpose(inputs=layer_3, filters=self.config['n_filter'], kernel_size=[4, 4], strides=[1,1], padding="valid", use_bias=True, activation=None)
					layer_4 = helper.conv_layer_norm_layer(layer_4, channel_index=3)
					layer_4 = self.activation_function(layer_4)
					output = tf.layers.conv2d_transpose(inputs=layer_4, filters=n_output_channels, kernel_size=[4, 4], strides=[2,2], padding="valid", use_bias=True, activation=tf.nn.sigmoid)
					image_param = output

				# 64x64xn_channels
				if image_shape == (64, 64):
					layer_1_flat = helper.FCResnetLayer_v2(x_batched_inp_flat, units=2*2*8*self.config['n_filter'], reduce_units=256, activation=self.activation_function, normalization_mode=self.normalization_mode)
					layer_1 = tf.reshape(layer_1_flat, [-1, 2, 2, 8*self.config['n_filter']])
					layer_2 = helper.ConvResnetLayer_v2(layer_1, units = 4*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Upsample')
					layer_3 = helper.ConvResnetLayer_v2(layer_2, units = 2*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Upsample')
					layer_4 = helper.ConvResnetLayer_v2(layer_3, units = 1*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Upsample')
					layer_5 = helper.ConvResnetLayer_v2(layer_4, units = 1*self.config['n_filter'], reduce_units=reduce_units, activation=self.activation_function, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Upsample')
					output = helper.ConvResnetLayer_v2(layer_5, units = n_output_channels, reduce_units=reduce_units, activation=tf.nn.sigmoid, reduce_activation=self.activation_function, normalization_mode=self.normalization_mode, image_modify='Upsample')
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
					output = helper.LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.2', self.config['n_filter'], 2*self.config['n_filter'], 5, output, stride=2)
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm': 
						output = lib.ops.batchnorm.Batchnorm('Critic.BN2', [0,2,3], output)
					output = helper.LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.3', 2*self.config['n_filter'], 4*self.config['n_filter'], 5, output, stride=2)					
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm':
						output = lib.ops.batchnorm.Batchnorm('Critic.BN3', [0,2,3], output) 
					output = helper.LeakyReLU(output)

					output = tf.reshape(output, [-1, 4*4*4*self.config['n_filter']])
					output = lib.ops.linear.Linear('Critic.Output', 4*4*4*self.config['n_filter'], 1, output)
					critic_image = output[:, np.newaxis, np.newaxis, :]

				# # 32x32xn_channels
				if image_shape == (32, 32):

					output = tf.transpose(x_batched_inp_image, perm=[0,3,1,2]) #tf.reshape(inputs, [-1, 3, 32, 32])
					output = lib.ops.conv2d.Conv2D('Critic.1', 3, self.config['n_filter'], 5, output, stride=2)
					output = helper.LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.2', self.config['n_filter'], 2*self.config['n_filter'], 5, output, stride=2)
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm': 
						output = lib.ops.batchnorm.Batchnorm('Critic.BN2', [0,2,3], output)
					output = helper.LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.3', 2*self.config['n_filter'], 4*self.config['n_filter'], 5, output, stride=2)					
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm':
						output = lib.ops.batchnorm.Batchnorm('Critic.BN3', [0,2,3], output) 
					output = helper.LeakyReLU(output)

					output = tf.reshape(output, [-1, 4*4*4*self.config['n_filter']])
					output = lib.ops.linear.Linear('Critic.Output', 4*4*4*self.config['n_filter'], 1, output)
					critic_image = output[:, np.newaxis, np.newaxis, :]
					
				# 64x64xn_channels
				if image_shape == (64, 64):
					output = tf.transpose(x_batched_inp_image, perm=[0,3,1,2]) #tf.reshape(inputs, [-1, 3, 32, 32])
					output = lib.ops.conv2d.Conv2D('Critic.1', 3, self.config['n_filter'], 5, output, stride=2)
					output = helper.LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.2', self.config['n_filter'], 2*self.config['n_filter'], 5, output, stride=2)
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm': 
						output = lib.ops.batchnorm.Batchnorm('Critic.BN2', [0,2,3], output)
					output = helper.LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.3', 2*self.config['n_filter'], 4*self.config['n_filter'], 5, output, stride=2)					
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm':
						output = lib.ops.batchnorm.Batchnorm('Critic.BN3', [0,2,3], output) 
					output = helper.LeakyReLU(output)

					output = lib.ops.conv2d.Conv2D('Critic.4', 4*self.config['n_filter'], 4*self.config['n_filter'], 5, output, stride=2)					
					if self.normalization_mode == 'Layer Norm':
						output = helper.conv_layer_norm_layer(output, channel_index=1)
					elif self.normalization_mode == 'Batch Norm':
						output = lib.ops.batchnorm.Batchnorm('Critic.BN4', [0,2,3], output) 
					output = helper.LeakyReLU(output)

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




























				# if noise is not None:
				# 	lay1_noise = helper.LeakyReLU(tf.layers.dense(inputs = noise, units = self.config['n_flat'], use_bias = False, activation = None))
				# 	lay2_noise = (tf.layers.dense(inputs = lay1_noise, units = self.config['n_latent'], use_bias = False, activation = None))
				# 	# lay3_noise = helper.LeakyReLU(tf.layers.dense(inputs = lay1_noise, units = self.config['n_latent'], use_bias = False, activation = None))

				# 	lay1_data = helper.LeakyReLU(tf.layers.dense(inputs = tf.concat([latent_image_flat, noise],axis=-1), units = self.config['n_flat'], use_bias = True, activation = None))
				# 	lay2_data = (tf.layers.dense(inputs = lay1_data, units = self.config['n_latent'], use_bias = True, activation = None))
				# 	# lay3_data = helper.LeakyReLU(tf.layers.dense(inputs = lay1_data, units = self.config['n_latent'], use_bias = True, activation = None))

				# 	# lay1_concat = helper.LeakyReLU(tf.layers.dense(inputs = tf.concat([latent_image_flat, noise],axis=-1), units = self.config['n_flat'], use_bias = True, activation = None))
				# 	# lay2_concat = helper.LeakyReLU(tf.layers.dense(inputs = lay1_concat, units = self.config['n_flat'], use_bias = True, activation = None))
				# 	# lay3_concat = (tf.layers.dense(inputs = lay2_concat, units = self.config['n_latent'], use_bias = True, activation = None))

				# 	# lay1_data_mu = helper.LeakyReLU(tf.layers.dense(inputs = latent_image_flat, units = self.config['n_flat'], use_bias = True, activation = None))
				# 	# lay1_data_logsig = helper.LeakyReLU(tf.layers.dense(inputs = latent_image_flat, units = self.config['n_flat'], use_bias = True, activation = None))
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


