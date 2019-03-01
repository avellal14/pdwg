from __future__ import print_function

import os
import re
import pdb
import time
import math
import numpy as np
import argparse
import copy

import models.WGANGPCannon.ModelMapsDCGAN3 as ModelMaps

import distributions 
import helper
import tensorflow as tf

class Model():
    def __init__(self, global_args, name = 'Model'):
        self.name = name
        self.bModules = False 
        self.fixedContext = False
        self.config = global_args
        self.compute_all_regularizers = False

        # self.config['sample_distance_mode'] # Euclidean, Quadratic  
        # self.config['kernel_mode'] # InvMultiquadratics, RadialBasisFunction, RationalQuadratic  
        # self.config['encoder_mode'] # Deterministic, Gaussian, UnivApprox, UnivApproxNoSpatial, UnivApproxSine
        # self.config['divergence_mode'] # GAN, NS-GAN, MMD     
        # self.config['dual_dist_mode'] # Coupling, Prior, CouplingAndPrior
        # self.config['critic_reg_mode'] # ['Coupling Gradient Vector', 'Coupling Gradient Norm', 'Trivial Gradient Norm', 'Uniform Gradient Norm', 'Coupling Lipschitz', 'Trivial Lipschitz', 'Uniform Lipschitz']

        self.sample_distance_function = self.euclidean_distance
        if self.config['kernel_mode'] == 'InvMultiquadratics': self.kernel_function = self.inv_multiquadratics_kernel
        if self.config['kernel_mode'] == 'RadialBasisFunction': self.kernel_function = self.rbf_kernel
        if self.config['kernel_mode'] == 'RationalQuadratic': self.kernel_function = self.rational_quadratic_kernel

    def generate_modules(self, batch):
        self.PriorMap = ModelMaps.PriorMap(self.config)
        self.EpsilonMap = ModelMaps.EpsilonMap(self.config)
        self.Diverger = ModelMaps.Diverger({**self.config, 'data_properties': batch['observed']['properties']})
        self.Encoder = ModelMaps.Encoder({**self.config, 'data_properties': batch['observed']['properties']})        
        self.Generator = ModelMaps.Generator({**self.config, 'data_properties': batch['observed']['properties']})        
        self.Critic = ModelMaps.Critic({**self.config, 'data_properties': batch['observed']['properties']})
        self.bModules = True

    def euclidean_distance_squared(self, x, y, axis=[-1], keep_dims=True):
        return tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims)

    def metric_distance_sq(self, x, y):
        try: metric_distance = self.euclidean_distance_squared(x['image'], y['image'], axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis]
        except: metric_distance = self.euclidean_distance_squared(x['flat'], y['flat'], axis=[-1], keep_dims=True)
        return metric_distance

    def euclidean_distance(self, a, b):
        return helper.safe_tf_sqrt(self.metric_distance_sq(a, b))

    def quadratic_distance(self, a, b):
        return (self.metric_distance_sq(a, b))

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

    def compute_MMD(self, sample_batch_1, sample_batch_2, mode='His'):
        if mode == 'Mine':
            k_sample_1_2 = tf.reduce_mean(self.kernel_function(sample_batch_1, sample_batch_2))
            k_sample_1_1 = tf.reduce_mean(self.kernel_function(sample_batch_1))
            k_sample_2_2 = tf.reduce_mean(self.kernel_function(sample_batch_2))
            MMD = k_sample_2_2+k_sample_1_1-2*k_sample_1_2
        else:
            sample_qz, sample_pz = sample_batch_1, sample_batch_2
            sigma2_p = 1 ** 2
            n = tf.cast(tf.shape(sample_qz)[0], tf.int32)
            nf = tf.cast(tf.shape(sample_qz)[0], tf.float32)

            norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
            dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
            distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

            norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
            dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
            distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

            dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
            distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

            Cbase = 2.
            stat = 0.
            scales = [.1, .2, .5, 1., 2., 5., 10.]
            for scale in scales:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
            MMD = stat/len(scales)
        return MMD

    def stable_div(self, div_func, batch_input, batch_rand_dirs):
        n_transforms = batch_rand_dirs.get_shape().as_list()[0]
        transformed_batch_input = self.apply_householder_reflections(batch_input, batch_rand_dirs)
        list_transformed_batch_input = tf.split(transformed_batch_input, n_transforms, axis=1)
        integral = 0
        for e in list_transformed_batch_input: 
            t_i_batch_input_inverted = tf.reverse(e[:,0,:], [0])
            integral += div_func(t_i_batch_input_inverted, batch_input)
        integral /= n_transforms
        return integral

    def apply_householder_reflections(self, batch_input, batch_rand_dirs):
        v_householder = batch_rand_dirs[np.newaxis, :,:]
        batch_input_expand = batch_input[:,np.newaxis, :]
        householder_inner_expand = tf.reduce_sum(v_householder*batch_input_expand, axis=2, keep_dims=True)  
        return batch_input_expand-2*householder_inner_expand*v_householder

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
        # self.gen_neg_ent_prior = self.prior_dist.log_pdf(self.gen_prior_latent_code)
        # self.gen_mean_neg_ent_prior = tf.reduce_mean(self.gen_neg_ent_prior)

        self.gen_obs_sample_param = self.Generator.forward(self.gen_prior_latent_code_expanded)
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
        # self.neg_ent_prior = self.prior_dist.log_pdf(self.prior_latent_code)
        # self.mean_neg_ent_prior = tf.reduce_mean(self.neg_ent_prior)

        self.obs_sample_param = self.Generator.forward(self.prior_latent_code_expanded)
        self.obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.obs_sample_param)
        self.obs_sample = self.obs_sample_dist.sample(b_mode=True)

        if not os.path.exists('./fixed_samples/'): os.makedirs('./fixed_samples/')
        if os.path.exists('./fixed_samples/np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz'): 
            np_constant_prior_sample = np.load('./np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz')
        else:
            np_constant_prior_sample = np.random.normal(loc=0., scale=1., size=[400, self.prior_latent_code.get_shape().as_list()[-1]])
            np.save('./fixed_samples/np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz', np_constant_prior_sample)    
        
        self.constant_prior_latent_code = tf.constant(np.asarray(np_constant_prior_sample), dtype=np.float32)
        self.constant_prior_latent_code_expanded = tf.reshape(self.constant_prior_latent_code, [-1, 1, *self.constant_prior_latent_code.get_shape().as_list()[1:]])

        self.constant_obs_sample_param = self.Generator.forward(self.constant_prior_latent_code_expanded)
        self.constant_obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.constant_obs_sample_param)
        self.constant_obs_sample = self.constant_obs_sample_dist.sample(b_mode=True)
        
        #############################################################################
        # ENCODER 
        

        if self.config['encoder_mode'] == 'Deterministic': 
            self.epsilon = None
        if self.config['encoder_mode'] == 'Gaussian' or self.config['encoder_mode'] == 'UnivApprox' or self.config['encoder_mode'] == 'UnivApproxNoSpatial' or self.config['encoder_mode'] == 'UnivApproxSine': 
            self.epsilon_param = self.PriorMap.forward((tf.zeros(shape=(self.batch_size_tf, 1)),))
            self.epsilon_dist = distributions.DiagonalGaussianDistribution(params = self.epsilon_param)        
            self.epsilon = self.epsilon_dist.sample()

        self.posterior_latent_code_expanded = self.Encoder.forward(self.input_sample, noise=self.epsilon) 
        self.posterior_latent_code = self.posterior_latent_code_expanded[:,0,:]
        self.interpolated_posterior_latent_code = self.interpolate_latent_codes(self.posterior_latent_code, size=self.batch_size_tf//2)
        self.interpolated_obs = self.Generator.forward(self.interpolated_posterior_latent_code) 

        self.reconst_param = self.Generator.forward(self.posterior_latent_code_expanded) 
        self.reconst_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.reconst_param)
        self.reconst_sample = self.reconst_dist.sample(b_mode=True)

        ### Primal Penalty
        if self.config['divergence_mode'] == 'MMD': 
            self.MMD = self.compute_MMD(self.posterior_latent_code, self.prior_dist.sample())
            self.enc_reg_cost = self.MMD
        if self.config['divergence_mode'] == 'INV-MMD': 
            batch_rand_vectors = tf.random_normal(shape=[self.config['enc_inv_MMD_n_trans'], self.config['n_latent']])
            batch_rand_dirs = batch_rand_vectors/helper.safe_tf_sqrt(tf.reduce_sum((batch_rand_vectors**2), axis=1, keep_dims=True))            
            self.Inv_MMD = self.stable_div(self.compute_MMD, self.posterior_latent_code, batch_rand_dirs)
            self.MMD = self.compute_MMD(self.posterior_latent_code, self.prior_dist.sample())
            self.enc_reg_cost = self.MMD + self.config['enc_inv_MMD_strength']*self.Inv_MMD
        elif self.config['divergence_mode'] == 'GAN' or self.config['divergence_mode'] == 'NS-GAN':
            self.div_posterior = self.Diverger.forward(self.posterior_latent_code_expanded)        
            self.div_prior = self.Diverger.forward(self.prior_latent_code_expanded)   
            self.mean_z_divergence = tf.reduce_mean(tf.log(10e-7+self.div_prior))+tf.reduce_mean(tf.log(10e-7+1-self.div_posterior))
            if self.config['divergence_mode'] == 'NS-GAN': 
                self.enc_reg_cost = -tf.reduce_mean(tf.log(10e-7+self.div_posterior))
            elif self.config['divergence_mode'] == 'GAN': 
                self.enc_reg_cost = tf.reduce_mean(tf.log(10e-7+1-self.div_posterior))

        
        #############################################################################
        # REGULARIZER 
        self.uniform_dist = distributions.UniformDistribution(params = tf.concat([tf.zeros(shape=(self.batch_size_tf, 1)), tf.ones(shape=(self.batch_size_tf, 1))], axis=1))

        ### Visualized regularization sample
        self.reg_target_dist = self.reconst_dist
        self.reg_dist = None
                
        ### Uniform Sample From Geodesic of Trivial Coupling Pairs
        if self.compute_all_regularizers or \
           'Trivial Gradient Norm' in self.config['critic_reg_mode'] or \
           'Trivial Lipschitz' in self.config['critic_reg_mode']:
            self.uniform_sample_b = self.uniform_dist.sample()
            self.trivial_line_sample_param = {'image': None, 'flat': None}
            try: 
                self.uniform_sample_b_expanded = self.uniform_sample_b[:, np.newaxis, :, np.newaxis, np.newaxis]
                self.trivial_line_sample_param['image'] = self.uniform_sample_b_expanded*self.obs_sample['image']+\
                                                          (1-self.uniform_sample_b_expanded)*self.input_sample['image']
            except: 
                self.uniform_sample_b_expanded = self.uniform_sample_b[:, np.newaxis, :]
                self.trivial_line_sample_param['flat'] = self.uniform_sample_b_expanded*self.obs_sample['flat']+\
                                                         (1-self.uniform_sample_b_expanded)*self.input_sample['flat']
            self.trivial_line_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.trivial_line_sample_param)
            self.trivial_line_sample = self.trivial_line_dist.sample(b_mode=True)
            if self.reg_dist is None: self.reg_dist = self.trivial_line_dist

        #############################################################################
        # CRITIC 

        self.critic_real = self.Critic.forward(self.input_sample)
        self.critic_gen = self.Critic.forward(self.obs_sample)
        self.critic_rec = self.Critic.forward(self.reconst_sample)
        
        self.mean_critic_real = tf.reduce_mean(self.critic_real)
        self.mean_critic_gen = tf.reduce_mean(self.critic_gen)
        self.mean_critic_rec = tf.reduce_mean(self.critic_rec)

        self.mean_critic_reg = None

        if self.compute_all_regularizers or \
           'Trivial Gradient Norm' in self.config['critic_reg_mode'] or \
           'Trivial Lipschitz' in self.config['critic_reg_mode']:
            self.critic_trivial_line = self.Critic.forward(self.trivial_line_sample)
            if self.mean_critic_reg is None: self.mean_critic_reg = tf.reduce_mean(self.critic_trivial_line)
            try:
                self.trivial_line_grad = tf.gradients(self.critic_trivial_line, [self.trivial_line_sample['image']])[0]
                self.trivial_line_grad_norm = tf.sqrt(tf.reduce_sum(tf.square(self.trivial_line_grad), axis=[-1,-2,-3], keep_dims=False))[:,np.newaxis,:]
            except: 
                self.trivial_line_grad = tf.gradients(self.critic_trivial_line, [self.trivial_line_sample['flat']])[0]
                self.trivial_line_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.trivial_line_grad**2, axis=[-1,], keep_dims=True))      

        self.cri_reg_cost = 0

        if self.compute_all_regularizers or 'Trivial Gradient Norm' in self.config['critic_reg_mode']:
            self.trivial_line_grad_norm_1_penalties = ((self.trivial_line_grad_norm-1.)**2)
            self.mean_trivial_line_grad_norm_1_penalties = tf.reduce_mean(self.trivial_line_grad_norm_1_penalties)
            if 'Trivial Gradient Norm' in self.config['critic_reg_mode']: 
                print('Adding Trivial Gradient Norm Penalty')
                self.cri_reg_cost += self.mean_trivial_line_grad_norm_1_penalties

        if len(self.config['critic_reg_mode'])>0: self.cri_reg_cost /= len(self.config['critic_reg_mode'])

        #############################################################################

        # OBJECTIVES
        ### Divergence
        if self.config['divergence_mode'] == 'GAN' or self.config['divergence_mode'] == 'NS-GAN':
            self.div_cost = -self.config['enc_reg_strength']*self.mean_z_divergence

        ### Encoder
        self.OT_primal = self.sample_distance_function(self.input_sample, self.reconst_sample)
        self.mean_OT_primal = tf.reduce_mean(self.OT_primal)
        

        # self.mean_OT_primal = helper.tf_print(self.mean_OT_primal, [self.mean_OT_primal])
        

        self.mean_POT_primal = self.mean_OT_primal #+self.config['enc_reg_strength']*self.enc_reg_cost
        self.enc_cost = self.mean_POT_primal

        ### Generator
        if self.config['dual_dist_mode'] == 'Coupling':
            self.gen_cost = -self.mean_critic_rec
        elif self.config['dual_dist_mode'] == 'Prior':
            self.gen_cost = -self.mean_critic_gen
        elif self.config['dual_dist_mode'] == 'CouplingAndPrior':
            self.gen_cost = -0.5*(self.mean_critic_rec+self.mean_critic_gen)

        ### Critic
        if self.config['dual_dist_mode'] == 'Coupling':
            self.mean_OT_dual = self.mean_critic_real-self.mean_critic_rec
        elif self.config['dual_dist_mode'] == 'Prior':
            self.mean_OT_dual = self.mean_critic_real-self.mean_critic_gen
        elif self.config['dual_dist_mode'] == 'CouplingAndPrior':
            self.mean_OT_dual = self.mean_critic_real-0.5*(self.mean_critic_rec+self.mean_critic_gen)

        self.cri_cost = -self.mean_OT_dual+self.config['cri_reg_strength']*self.cri_reg_cost



