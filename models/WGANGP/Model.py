from __future__ import print_function

import os
import re
import pdb
import time
import math
import numpy as np
import argparse
import copy

import models.WGANGP.ModelMapsDCGAN3 as ModelMaps

import distributions 
import helper
import tensorflow as tf

class Model():
    def __init__(self, global_args, name = 'Model'):
        self.name = name
        self.bModules = False 
        self.fixedContext = False
        self.config = global_args
        self.temperature_object = helper.TemperatureObjectTF(self.config['initial_temp'], self.config['max_step_temp'])

    def generate_modules(self, batch):
        self.Discriminator = ModelMaps.Discriminator({**self.config, 'data_properties': batch['observed']['properties']})
        self.DivergenceLatent = ModelMaps.DivergenceLatent({**self.config, 'data_properties': batch['observed']['properties']})
        self.Decoder = ModelMaps.Decoder({**self.config, 'data_properties': batch['observed']['properties']})        
        self.Encoder = ModelMaps.Encoder({**self.config, 'data_properties': batch['observed']['properties']})        
        self.EncoderSampler = ModelMaps.EncoderSampler({**self.config, 'data_properties': batch['observed']['properties']})        
        self.PriorMap = ModelMaps.PriorMap(self.config)
        self.EpsilonMap = ModelMaps.EpsilonMap(self.config)
        self.bModules = True

    def euclidean_distance_squared(self, x, y, axis=[-1], keep_dims=True):
        return tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims)

    def metric_distance_sq(self, x, y):
        try: metric_distance = self.euclidean_distance_squared(x['image'], y['image'], axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis]
        except: metric_distance = self.euclidean_distance_squared(x['flat'], y['flat'], axis=[-1], keep_dims=True)
        return metric_distance

    def rbf_kernel(self, z_1, z_2=None, sigma_z_sq=1):
        if z_2 is None: 
            squared_dists_all = (z_1[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2
            squared_dists = helper.tf_remove_diagonal(squared_dists_all, first_dim_size=tf.shape(z_1)[0], second_dim_size=tf.shape(z_1)[0])
        else:
            squared_dists = (z_2[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2

        z_diff_norm_sq = tf.reduce_sum(squared_dists, axis=[-1], keep_dims=True)
        n_dim = z_1.get_shape().as_list()[-1]
        sigma_k_sq = 2*n_dim*sigma_z_sq
        return tf.exp(-z_diff_norm_sq/sigma_k_sq)

    def inv_multiquadratics_kernel(self, z_1, z_2=None, sigma_z_sq=1):
        if z_2 is None: 
            squared_dists_all = (z_1[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2
            squared_dists = helper.tf_remove_diagonal(squared_dists_all, first_dim_size=tf.shape(z_1)[0], second_dim_size=tf.shape(z_1)[0])
        else:
            squared_dists = (z_2[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2

        z_diff_norm_sq = tf.reduce_sum(squared_dists, axis=[-1], keep_dims=True)
        n_dim = z_1.get_shape().as_list()[-1]
        C = 2*n_dim*sigma_z_sq
        return C/(C+z_diff_norm_sq)

    def rational_quadratic_kernel(self, z_1, z_2=None, alpha=1):
        if z_2 is None: 
            squared_dists_all = (z_1[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2
            squared_dists = helper.tf_remove_diagonal(squared_dists_all, first_dim_size=tf.shape(z_1)[0], second_dim_size=tf.shape(z_1)[0])
        else:
            squared_dists = (z_2[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2

        z_diff_norm_sq = tf.reduce_sum(squared_dists, axis=[-1], keep_dims=True)
        C = 1+z_diff_norm_sq/(2*alpha)
        if alpha == 1: return C
        else: return tf.pow(C, -alpha)

    def circular_shift(self, z):
        return tf.concat([z[1:,:], z[0, np.newaxis,:]], axis=0)        

    def interpolate_latent_codes(self, z, size=1, number_of_steps=10):
        z_a = z[:size,:] 
        z_b = z[size:2*size,:] 
        t = (tf.range(number_of_steps, dtype=np.float32)/float(number_of_steps-1))[np.newaxis, :, np.newaxis]
        z_interp = t*z_a[:, np.newaxis, :]+(1-t)*z_b[:, np.newaxis, :]
        return z_interp

    def generative_model(self, batch, additional_inputs_tf):
        self.gen_epoch = additional_inputs_tf[0]
        self.gen_b_identity = additional_inputs_tf[1]

        if len(batch['observed']['properties']['flat'])>0:
            for e in batch['observed']['properties']['flat']: e['dist']='dirac'
        else:
            for e in batch['observed']['properties']['image']: e['dist']='dirac'

        self.gen_input_sample = batch['observed']['data']
        self.gen_input_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.gen_input_sample)

        try: self.n_time = batch['observed']['properties']['flat'][0]['size'][1]
        except: self.n_time = batch['observed']['properties']['image'][0]['size'][1]
        try: self.gen_batch_size_tf = tf.shape(self.input_sample['flat'])[0]
        except: self.gen_batch_size_tf = tf.shape(self.input_sample['image'])[0]
        
        self.gen_prior_param = self.PriorMap.forward((tf.zeros(shape=(self.gen_batch_size_tf, 1)),))
        self.gen_prior_dist = distributions.DiagonalGaussianDistribution(params = self.gen_prior_param)
        self.gen_prior_latent_code = self.gen_prior_dist.sample()
        self.gen_prior_latent_code_expanded = tf.reshape(self.gen_prior_latent_code, [-1, 1, *self.gen_prior_latent_code.get_shape().as_list()[1:]])
        self.gen_neg_ent_prior = self.prior_dist.log_pdf(self.gen_prior_latent_code)
        self.gen_mean_neg_ent_prior = tf.reduce_mean(self.gen_neg_ent_prior)

        self.gen_obs_sample_param = self.Decoder.forward(self.gen_prior_latent_code_expanded)
        self.gen_obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.gen_obs_sample_param)
        self.gen_obs_sample = self.gen_obs_sample_dist.sample(b_mode=True)

    def inference(self, batch, additional_inputs_tf):
        self.epoch = additional_inputs_tf[0]
        self.b_identity = additional_inputs_tf[1]
        
        if len(batch['observed']['properties']['flat'])>0:
            for e in batch['observed']['properties']['flat']: e['dist']='dirac'
        else:
            for e in batch['observed']['properties']['image']: e['dist']='dirac'

        self.input_sample = batch['observed']['data']
        self.input_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.input_sample)

        if not self.bModules: self.generate_modules(batch)
        try: self.n_time = batch['observed']['properties']['flat'][0]['size'][1]
        except: self.n_time = batch['observed']['properties']['image'][0]['size'][1]
        try: self.batch_size_tf = tf.shape(self.input_sample['flat'])[0]
        except: self.batch_size_tf = tf.shape(self.input_sample['image'])[0]

        #############################################################################
        # GENERATOR 

        self.prior_param = self.PriorMap.forward((tf.zeros(shape=(self.batch_size_tf, 1)),))
        self.prior_dist = distributions.DiagonalGaussianDistribution(params = self.prior_param)
        self.prior_latent_code = self.prior_dist.sample()
        self.prior_latent_code_expanded = tf.reshape(self.prior_latent_code, [-1, 1, *self.prior_latent_code.get_shape().as_list()[1:]])
        self.neg_ent_prior = self.prior_dist.log_pdf(self.prior_latent_code)
        self.mean_neg_ent_prior = tf.reduce_mean(self.neg_ent_prior)

        self.obs_sample_param = self.Decoder.forward(self.prior_latent_code_expanded)
        self.obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.obs_sample_param)
        self.obs_sample = self.obs_sample_dist.sample(b_mode=True)

        if os.path.exists('./np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz'): 
            np_constant_prior_sample = np.load('./np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz')
        else:
            np_constant_prior_sample = np.random.normal(loc=0., scale=1., size=[400, self.prior_latent_code.get_shape().as_list()[-1]])
            np.save('./np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz', np_constant_prior_sample)    
        
        self.constant_prior_latent_code = tf.constant(np.asarray(np_constant_prior_sample), dtype=np.float32)
        self.constant_prior_latent_code_expanded = tf.reshape(self.constant_prior_latent_code, [-1, 1, *self.constant_prior_latent_code.get_shape().as_list()[1:]])

        self.constant_obs_sample_param = self.Decoder.forward(self.constant_prior_latent_code_expanded)
        self.constant_obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.constant_obs_sample_param)
        self.constant_obs_sample = self.constant_obs_sample_dist.sample(b_mode=True)
        
        #############################################################################
        # ENCODER 

        b_deterministic = False
        if b_deterministic: 
            self.epsilon = None
        else:
            self.epsilon_param = self.PriorMap.forward((tf.zeros(shape=(self.batch_size_tf, 1)),))
            self.epsilon_dist = distributions.DiagonalGaussianDistribution(params = self.prior_param)        
            self.epsilon = self.epsilon_dist.sample()

        self.posterior_latent_code_expanded = self.EncoderSampler.forward(self.input_sample, noise=self.epsilon) 
        self.posterior_latent_code = self.posterior_latent_code_expanded[:,0,:]
        self.interpolated_posterior_latent_code = self.interpolate_latent_codes(self.posterior_latent_code, size=self.batch_size_tf//2)
        self.interpolated_obs = self.Decoder.forward(self.interpolated_posterior_latent_code) 

        self.reconst_param = self.Decoder.forward(self.posterior_latent_code_expanded) 
        self.reconst_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.reconst_param)
        self.reconst_sample = self.reconst_dist.sample(b_mode=True)

        self.kernel_function = self.inv_multiquadratics_kernel
        self.k_post_prior = tf.reduce_mean(self.kernel_function(self.posterior_latent_code, self.prior_dist.sample()))
        self.k_post_post = tf.reduce_mean(self.kernel_function(self.posterior_latent_code))
        self.k_prior_prior = tf.reduce_mean(self.kernel_function(self.prior_dist.sample()))
        self.MMD = self.k_prior_prior+self.k_post_post-2*self.k_post_prior

        self.disc_posterior = self.DivergenceLatent.forward(self.posterior_latent_code_expanded)        
        self.disc_prior = self.DivergenceLatent.forward(self.prior_latent_code_expanded)   
        self.mean_z_divergence = tf.reduce_mean(tf.log(10e-7+self.disc_prior))+tf.reduce_mean(tf.log(10e-7+1-self.disc_posterior))
        self.mean_z_divergence_minimizer = tf.reduce_mean(tf.log(10e-7+1-self.disc_posterior))

        #############################################################################
        # REGULARIZER 

        self.uniform_dist = distributions.UniformDistribution(params = tf.concat([tf.zeros(shape=(self.batch_size_tf, 1)), tf.ones(shape=(self.batch_size_tf, 1))], axis=1))
        self.uniform_sample = self.uniform_dist.sample()
        
        self.reg_trivial_sample_param = {'image': None, 'flat': None}
        try: 
            self.reg_trivial_sample_param['image'] = self.uniform_sample[:, np.newaxis, :, np.newaxis, np.newaxis]*self.obs_sample['image']+\
                                                    (1-self.uniform_sample[:, np.newaxis, :, np.newaxis, np.newaxis])*self.input_sample['image']
        except: 
            self.reg_trivial_sample_param['flat'] = self.uniform_sample[:, np.newaxis, :]*self.obs_sample['flat']+\
                                                   (1-self.uniform_sample[:, np.newaxis, :])*self.input_sample['flat']
        self.reg_trivial_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.reg_trivial_sample_param)
        self.reg_trivial_sample = self.reg_trivial_dist.sample(b_mode=True)

        self.reg_target_dist = self.obs_sample_dist
        self.reg_target_sample = self.obs_sample
        self.reg_dist = self.reg_trivial_dist
        self.reg_sample = self.reg_trivial_sample 
        
        #############################################################################
        # CRITIC 

        self.critic_real = self.Discriminator.forward(self.input_sample)
        self.critic_gen = self.Discriminator.forward(self.obs_sample)
        self.critic_reg_trivial = self.Discriminator.forward(self.reg_trivial_sample)
        self.critic_reg = self.critic_reg_trivial

        try:
            self.trivial_grad = tf.gradients(self.critic_reg_trivial, [self.reg_trivial_sample['image']])[0]
            self.trivial_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.trivial_grad**2, axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis])
        except: 
            self.trivial_grad = tf.gradients(self.critic_reg_trivial, [self.reg_trivial_sample['flat']])[0]
            self.trivial_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.trivial_grad**2, axis=[-1,], keep_dims=True))      

        self.trivial_grad_norm_1_penalties = ((self.trivial_grad_norm-1)**2)

        self.mean_critic_real = tf.reduce_mean(self.critic_real)
        self.mean_critic_gen = tf.reduce_mean(self.critic_gen)
        self.mean_critic_reg = tf.reduce_mean(self.critic_reg)
        self.mean_OT_dual = self.mean_critic_real-self.mean_critic_gen

        #############################################################################
        # OBJECTIVES
        self.enc_reg_strength, self.disc_reg_strength = 100, 10

        self.real_reconst_distances_sq = self.metric_distance_sq(self.input_sample, self.reconst_sample)
        self.OT_primal = tf.reduce_mean(helper.safe_tf_sqrt(self.real_reconst_distances_sq))
        self.mean_OT_primal = tf.reduce_mean(self.OT_primal)

        self.enc_reg_cost = self.MMD # self.mean_z_divergence_minimizer
        self.mean_POT_primal = self.mean_OT_primal+self.enc_reg_strength*self.enc_reg_cost
        
        self.disc_reg_cost = tf.reduce_mean(self.trivial_grad_norm_1_penalties)

        # WGAN-GP
        # self.div_cost = -self.mean_z_divergence
        self.enc_cost = self.mean_POT_primal
        self.disc_cost = -self.mean_OT_dual+self.disc_reg_strength*self.disc_reg_cost
        self.gen_cost = -self.mean_critic_gen



















































# from __future__ import print_function

# import os
# import re
# import pdb
# import time
# import math
# import numpy as np
# import argparse
# import copy

# import models.WGANGP.ModelMapsDCGAN as ModelMaps

# import distributions 
# import helper
# import tensorflow as tf

# class Model():
#     def __init__(self, global_args, name = 'Model'):
#         self.name = name
#         self.bModules = False 
#         self.fixedContext = False
#         self.config = global_args
#         self.temperature_object = helper.TemperatureObjectTF(self.config['initial_temp'], self.config['max_step_temp'])

#     def generate_modules(self, batch):
#         self.Discriminator = ModelMaps.Discriminator({**self.config, 'data_properties': batch['observed']['properties']})
#         self.Decoder = ModelMaps.Decoder({**self.config, 'data_properties': batch['observed']['properties']})        
#         self.EncodingPlan = ModelMaps.EncodingPlan({**self.config, 'data_properties': batch['observed']['properties']})        
#         self.PriorMap = ModelMaps.PriorMap(self.config)
#         self.EpsilonMap = ModelMaps.EpsilonMap(self.config)
#         self.bModules = True

#     def euclidean_distance_squared(self, x, y, axis=[-1], keep_dims=True):
#         return tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims)

#     def metric_distance_sq(self, x, y):
#         try: metric_distance = self.euclidean_distance_squared(x['image'], y['image'], axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis]
#         except: metric_distance = self.euclidean_distance_squared(x['flat'], y['flat'], axis=[-1], keep_dims=True)
#         return metric_distance

#     def generative_model(self, batch, additional_inputs_tf):
#         self.gen_epoch = additional_inputs_tf[0]
#         self.gen_b_identity = additional_inputs_tf[1]

#         if len(batch['observed']['properties']['flat'])>0:
#             for e in batch['observed']['properties']['flat']: e['dist']='dirac'
#         else:
#             for e in batch['observed']['properties']['image']: e['dist']='dirac'

#         self.gen_input_sample = batch['observed']['data']
#         self.gen_input_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.gen_input_sample)

#         try: self.n_time = batch['observed']['properties']['flat'][0]['size'][1]
#         except: self.n_time = batch['observed']['properties']['image'][0]['size'][1]
#         try: batch_size_tf = tf.shape(self.input_sample['flat'])[0]
#         except: batch_size_tf = tf.shape(self.input_sample['image'])[0]
        
#         self.gen_prior_param = self.PriorMap.forward((tf.zeros(shape=(batch_size_tf, 1)),))
#         self.gen_prior_dist = distributions.DiagonalGaussianDistribution(params = self.gen_prior_param)
#         self.gen_prior_latent_code = self.gen_prior_dist.sample()
#         self.gen_neg_ent_prior = self.prior_dist.log_pdf(self.gen_prior_latent_code)
#         self.gen_mean_neg_ent_prior = tf.reduce_mean(self.gen_neg_ent_prior)

#         self.gen_prior_latent_code_expanded = tf.reshape(self.gen_prior_latent_code, [-1, 1, *self.gen_prior_latent_code.get_shape().as_list()[1:]])
#         self.gen_obs_sample_param = self.Decoder.forward(self.gen_prior_latent_code_expanded)
#         self.gen_obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.gen_obs_sample_param)
#         self.gen_obs_sample = self.gen_obs_sample_dist.sample(b_mode=True)

#     def inference(self, batch, additional_inputs_tf):
#         self.epoch = additional_inputs_tf[0]
#         self.b_identity = additional_inputs_tf[1]
        
#         if len(batch['observed']['properties']['flat'])>0:
#             for e in batch['observed']['properties']['flat']: e['dist']='dirac'
#         else:
#             for e in batch['observed']['properties']['image']: e['dist']='dirac'

#         self.input_sample = batch['observed']['data']
#         self.input_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.input_sample)

#         if not self.bModules: self.generate_modules(batch)
#         try: self.n_time = batch['observed']['properties']['flat'][0]['size'][1]
#         except: self.n_time = batch['observed']['properties']['image'][0]['size'][1]
#         try: batch_size_tf = tf.shape(self.input_sample['flat'])[0]
#         except: batch_size_tf = tf.shape(self.input_sample['image'])[0]
        
#         self.prior_param = self.PriorMap.forward((tf.zeros(shape=(batch_size_tf, 1)),))
#         self.prior_dist = distributions.DiagonalGaussianDistribution(params = self.prior_param)
#         self.prior_latent_code = self.prior_dist.sample()
#         self.neg_ent_prior = self.prior_dist.log_pdf(self.prior_latent_code)
#         self.mean_neg_ent_prior = tf.reduce_mean(self.neg_ent_prior)

#         self.uniform_dist = distributions.UniformDistribution(params = tf.concat([tf.zeros(shape=(batch_size_tf, 1)), tf.ones(shape=(batch_size_tf, 1))], axis=1))
#         self.convex_mix = self.uniform_dist.sample()

#         self.prior_latent_code_expanded = tf.reshape(self.prior_latent_code, [-1, 1, *self.prior_latent_code.get_shape().as_list()[1:]])
#         self.obs_sample_param = self.Decoder.forward(self.prior_latent_code_expanded)
#         self.obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.obs_sample_param)
#         self.obs_sample = self.obs_sample_dist.sample(b_mode=True)

#         #############################################################################
#         # WGAN-GP
#         self.posterior_latent_code = self.prior_latent_code
#         self.posterior_latent_code_expanded = tf.reshape(self.posterior_latent_code, [-1, 1, *self.posterior_latent_code.get_shape().as_list()[1:]])
        
#         self.reconst_dist = self.obs_sample_dist
#         self.reconst_sample = self.obs_sample
#         self.reg_target_dist = self.obs_sample_dist
#         self.reg_target_sample = self.obs_sample

#         self.neg_cross_ent_posterior = self.prior_dist.log_pdf(self.posterior_latent_code)
#         self.mean_neg_cross_ent_posterior = tf.reduce_mean(self.neg_cross_ent_posterior)
#         # #################################################################################

#         self.reg_sample_param = {'image': None, 'flat': None}
#         try: 
#             self.reg_sample_param['image'] = self.convex_mix[:, np.newaxis, :, np.newaxis, np.newaxis]*self.reg_target_sample['image']+\
#                                             (1-self.convex_mix[:, np.newaxis, :, np.newaxis, np.newaxis])*self.input_sample['image']
#         except: 
#             self.reg_sample_param['flat'] = self.convex_mix[:, np.newaxis, :]*self.reg_target_sample['flat']+\
#                                             (1-self.convex_mix[:, np.newaxis, :])*self.input_sample['flat']

#         self.reg_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.reg_sample_param)
#         self.reg_sample = self.reg_dist.sample(b_mode=True)

#         # self.disc_input, self.disc_input_axis_sizes = helper.tf_batchify_across_dim([self.input_sample, self.obs_sample, self.reg_sample], axis=0)
#         # self.disc_out = self.Discriminator.forward(self.disc_input) 
#         # self.critic_real, self.critic_gen, self.critic_reg = helper.tf_debatchify_across_dim(self.disc_out, self.disc_input_axis_sizes)

#         self.critic_real = self.Discriminator.forward(self.input_sample)
#         self.critic_gen = self.Discriminator.forward(self.obs_sample)
#         self.critic_reg = self.Discriminator.forward(self.reg_sample)

#         try:
#             self.convex_grad = tf.gradients(self.critic_reg, [self.reg_sample['image']])[0]
#             self.convex_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.convex_grad**2, axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis])
#         except: 
#             self.convex_grad = tf.gradients(self.critic_reg, [self.reg_sample['flat']])[0]
#             self.convex_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.convex_grad**2, axis=[-1], keep_dims=True))      
#         self.gradient_penalties = ((self.convex_grad_norm-1)**2)

#         self.mean_critic_real = tf.reduce_mean(self.critic_real)
#         self.mean_critic_gen = tf.reduce_mean(self.critic_gen)
#         self.mean_critic_reg = tf.reduce_mean(self.critic_reg)
#         self.mean_gradient_penalty = tf.reduce_mean(self.gradient_penalties)
        
#         self.regularizer_cost = 10*self.mean_gradient_penalty
#         self.discriminator_cost = -self.mean_critic_real+self.mean_critic_gen+self.regularizer_cost
#         self.generator_cost = -self.mean_critic_gen
#         self.transporter_cost = None
