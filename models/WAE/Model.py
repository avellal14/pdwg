from __future__ import print_function

import os
import re
import pdb
import time
import math
import numpy as np
import argparse
import copy
from pathlib import Path

# import models.WGANGP.ModelMapsDCGAN3 as ModelMaps
# import models.WGANGP.ModelMapsDCGAN3INTENSITY as ModelMaps
# import models.PDWGANCannon.ModelMapsDCGAN3 as ModelMaps

import models.PDWGANCannon.ModelMapsDCGAN3 as ModelMaps
# import models.WGANGP.ModelMapsDCGAN3 as ModelMaps
# import models.WAECompactFlow.ModelMaps as ModelMaps

import distributions 
import helper
import tensorflow as tf

class Model():
    def __init__(self, global_args, name = 'Model'):
        self.name = name
        self.bModules = False 
        self.fixedContext = False
        self.config = global_args

        # self.config['sample_distance_mode'] # Euclidean, Quadratic  
        # self.config['kernel_mode'] # InvMultiquadratics, RadialBasisFunction, RationalQuadratic  
        # self.config['encoder_mode'] # Deterministic, Gaussian, UnivApprox, UnivApproxNoSpatial, UnivApproxSine
        # self.config['divergence_mode'] # GAN, NS-GAN, MMD, INV-MMD, SUM-MMD   
        # self.config['dual_dist_mode'] # Coupling, Prior
        # self.config['critic_reg_mode'] # ['Coupling Gradient Vector', 'Coupling Gradient Norm', 'Trivial Gradient Norm', 'Uniform Gradient Norm', 'Coupling Lipschitz', 'Trivial Lipschitz', 'Uniform Lipschitz']

        if self.config['sample_distance_mode'] == 'Euclidean': self.sample_distance_function = self.euclidean_distance
        if self.config['sample_distance_mode'] == 'Quadratic': self.sample_distance_function = self.quadratic_distance
        if self.config['kernel_mode'] == 'InvMultiquadratics': self.kernel_function = self.inv_multiquadratics_kernel
        if self.config['kernel_mode'] == 'RadialBasisFunction': self.kernel_function = self.rbf_kernel
        if self.config['kernel_mode'] == 'RationalQuadratic': self.kernel_function = self.rational_quadratic_kernel

    def generate_modules(self, batch):
        self.PriorMap = ModelMaps.PriorMap(self.config)
        self.EpsilonMap = ModelMaps.EpsilonMap(self.config)
        self.Diverger = ModelMaps.Diverger({**self.config, 'data_properties': batch['observed']['properties']})
        self.Decomposer = ModelMaps.Decomposer({**self.config, 'data_properties': batch['observed']['properties']})        
        self.Separator = ModelMaps.Separator({**self.config, 'data_properties': batch['observed']['properties']})        
        self.Encoder = ModelMaps.Encoder({**self.config, 'data_properties': batch['observed']['properties']})        
        self.Generator = ModelMaps.Generator({**self.config, 'data_properties': batch['observed']['properties']})        
        self.Critic = ModelMaps.Critic({**self.config, 'data_properties': batch['observed']['properties']})
        self.bModules = True


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

        # self.gen_prior_param_low = -3*tf.ones(shape=(self.batch_size_tf, self.config['n_latent']))
        # self.gen_prior_param_high = 3*tf.ones(shape=(self.batch_size_tf, self.config['n_latent']))
        # self.gen_prior_param= tf.concat([self.gen_prior_param_low, self.gen_prior_param_high], axis = 1)
        # self.gen_prior_dist = distributions.UniformDistribution(params = self.gen_prior_param)

        self.gen_prior_latent_code = self.gen_prior_dist.sample()
        self.gen_prior_latent_code_expanded = self.gen_prior_latent_code[:,np.newaxis,:]
        self.gen_neg_ent_prior = self.prior_dist.log_pdf(self.gen_prior_latent_code)
        self.gen_mean_neg_ent_prior = tf.reduce_mean(self.gen_neg_ent_prior)

        # self.gen_prior_input_transformed_reconst, _ = self.Separator.forward(self.gen_prior_latent_code)

        self.gen_obs_sample_param = self.Generator.forward(self.gen_prior_latent_code_expanded)
        # self.gen_obs_sample_param = self.Generator.forward(self.gen_prior_input_transformed_reconst[:,np.newaxis,:])
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

        # self.prior_param_low = -3*tf.ones(shape=(self.batch_size_tf, self.config['n_latent']))
        # self.prior_param_high = 3*tf.ones(shape=(self.batch_size_tf, self.config['n_latent']))
        # self.prior_param = tf.concat([self.prior_param_low, self.prior_param_high], axis = 1)
        # self.prior_dist = distributions.UniformDistribution(params = self.prior_param)

        self.prior_latent_code = self.prior_dist.sample()
        self.prior_latent_code_expanded = self.prior_latent_code[:, np.newaxis, :]
        self.neg_ent_prior = self.prior_dist.log_pdf(self.prior_latent_code)
        self.mean_neg_ent_prior = tf.reduce_mean(self.neg_ent_prior)
        
        # self.prior_input_transformed_reconst, _ = self.Separator.forward(self.prior_latent_code)
        
        self.obs_sample_param = self.Generator.forward(self.prior_latent_code_expanded)
        # self.obs_sample_param = self.Generator.forward(self.prior_input_transformed_reconst[:,np.newaxis,:])
        self.obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.obs_sample_param)
        self.obs_sample = self.obs_sample_dist.sample(b_mode=True)
        
        if not os.path.exists(str(Path.home())+'/ExperimentalResults/FixedSamples/'): os.makedirs(str(Path.home())+'/ExperimentalResults/FixedSamples/')
        if os.path.exists(str(Path.home())+'/ExperimentalResults/FixedSamples/np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz'): 
            np_constant_prior_sample = np.load(str(Path.home())+'/ExperimentalResults/FixedSamples/np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz')
        else:
            np_constant_prior_sample = np.random.normal(loc=0., scale=1., size=[400, self.prior_latent_code.get_shape().as_list()[-1]])
            np.save(str(Path.home())+'/ExperimentalResults/FixedSamples/np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz', np_constant_prior_sample)    
        
        self.constant_prior_latent_code = tf.constant(np.asarray(np_constant_prior_sample), dtype=np.float32)
        self.constant_prior_latent_code_expanded = self.constant_prior_latent_code[:, np.newaxis, :]

        # self.constant_prior_input_transformed_reconst, _ = self.Separator.forward(self.constant_prior_latent_code)

        self.constant_obs_sample_param = self.Generator.forward(self.constant_prior_latent_code_expanded)
        # self.constant_obs_sample_param = self.Generator.forward(self.constant_prior_input_transformed_reconst[:, np.newaxis, :])
        self.constant_obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.constant_obs_sample_param)
        self.constant_obs_sample = self.constant_obs_sample_dist.sample(b_mode=True)
                
        if self.config['n_latent'] == 2: 
            grid_scale = 3
            x = np.linspace(-grid_scale, grid_scale, 20)
            y = np.linspace(grid_scale, -grid_scale, 20)
            xv, yv = np.meshgrid(x, y)
            np_constant_prior_grid_sample = np.concatenate((xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis][:]), axis=1)
        
            self.constant_prior_grid_latent_code = tf.constant(np.asarray(np_constant_prior_grid_sample), dtype=np.float32)
            self.constant_prior_grid_latent_code_expanded = tf.reshape(self.constant_prior_grid_latent_code, [-1, 1, *self.constant_prior_grid_latent_code.get_shape().as_list()[1:]])

            self.constant_obs_grid_sample_param = self.Generator.forward(self.constant_prior_grid_latent_code_expanded)
            self.constant_obs_grid_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.constant_obs_grid_sample_param)
            self.constant_obs_grid_sample = self.constant_obs_grid_sample_dist.sample(b_mode=True)
                
        #############################################################################
        # ENCODER 
        self.uniform_dist = distributions.UniformDistribution(params = tf.concat([tf.zeros(shape=(self.batch_size_tf, 1)), tf.ones(shape=(self.batch_size_tf, 1))], axis=1))
        self.uniform_sample = self.uniform_dist.sample()

        if self.config['encoder_mode'] == 'Deterministic': 
            self.epsilon = None
            self.epsilon_contract = None
            self.epsilon_2 = None
        if self.config['encoder_mode'] == 'Gaussian' or self.config['encoder_mode'] == 'UnivApprox' or self.config['encoder_mode'] == 'UnivApproxNoSpatial' or self.config['encoder_mode'] == 'UnivApproxSine': 
            self.epsilon_param = self.PriorMap.forward((tf.zeros(shape=(self.batch_size_tf, 1)),))
            self.epsilon_dist = distributions.DiagonalGaussianDistribution(params = self.epsilon_param)        
            self.epsilon = self.epsilon_dist.sample()
            self.epsilon_contract = self.epsilon*self.uniform_sample
            self.epsilon_2 = self.epsilon_dist.sample()

        self.posterior_latent_code_expanded, self.input_transformed = self.Encoder.forward(self.input_sample, noise=self.epsilon)
        self.posterior_latent_code = self.posterior_latent_code_expanded[:,0,:]
        # self.posterior_latent_code_contract_expanded, _ = self.Encoder.forward(self.input_sample, noise=self.epsilon_contract)
        # self.posterior_latent_code_contract = self.posterior_latent_code_contract_expanded[:,0,:]
        # self.posterior_latent_code_expanded_2, _ = self.Encoder.forward(self.input_sample, noise=self.epsilon_2)
        # self.posterior_latent_code_2 = self.posterior_latent_code_expanded_2[:,0,:]
        # self.posterior_latent_code_diff = self.posterior_latent_code_2-self.posterior_latent_code
        
        # self.contraction_distance_sq = tf.reduce_sum((self.posterior_latent_code*self.uniform_sample-self.posterior_latent_code_contract)**2, axis=-1, keep_dims=True)
        # self.mean_contraction_distance_sq = tf.reduce_mean(self.contraction_distance_sq)

        # self.posterior_latent_code_diff_norm_sq = tf.reduce_mean((self.posterior_latent_code_diff**2), axis=1, keep_dims=True)
        # self.mean_posterior_latent_code_diff_norm_sq = tf.reduce_mean(self.posterior_latent_code_diff_norm_sq)
        # self.prior_latent_code_diff_norm_sq = tf.reduce_mean(((self.prior_dist.sample()-self.prior_dist.sample())**2), axis=1, keep_dims=True)
        # self.mean_prior_latent_code_diff_norm_sq = tf.reduce_mean(self.prior_latent_code_diff_norm_sq)

        self.interpolated_posterior_latent_code = self.interpolate_latent_codes(self.posterior_latent_code, size=self.batch_size_tf//2)
        self.interpolated_obs = self.Generator.forward(self.interpolated_posterior_latent_code) 

        self.input_transformed_reconst, self.epsilon_reconst = self.Separator.forward(self.posterior_latent_code)
        self.input_transformed_reconst_cost = tf.reduce_mean(tf.reduce_sum((self.input_transformed-self.input_transformed_reconst)**2, axis = [-1,])) 
        self.epsilon_reconst_cost = tf.reduce_mean(tf.reduce_sum((self.epsilon-self.epsilon_reconst)**2, axis = [-1,])) 


        self.reconst_param = self.Generator.forward(self.posterior_latent_code_expanded) 
        # self.reconst_param = self.Generator.forward(self.input_transformed_reconst[:, np.newaxis, :]) 
        # self.reconst_param = self.Generator.forward(self.input_transformed[:, np.newaxis, :]) 
        # self.tiny_perturb = 0.1*self.epsilon_dist.sample()
        # self.reconst_param = self.Generator.forward((self.input_transformed+self.tiny_perturb)[:, np.newaxis, :]) 
        self.reconst_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.reconst_param)
        self.reconst_sample = self.reconst_dist.sample(b_mode=True)

        ### Primal Penalty

        # if self.config['divergence_mode'] == 'MUTUALINFOMMD': 
        #     # n_dim_MMD = 20
        #     # shuffled_posterior_latent_code = helper.tf_differentiable_random_shuffle_with_axis(self.posterior_latent_code, axis=1)
        #     # shuffled_epsilon = helper.tf_differentiable_random_shuffle_with_axis(self.epsilon, axis=1)
        #     # rand_subset_shuffled_posterior_latent_code_1 = shuffled_epsilon[:self.batch_size_tf//2, :][:, :n_dim_MMD]
        #     # rand_subset_shuffled_posterior_latent_code_2 = shuffled_epsilon[self.batch_size_tf//2:, :][:, :n_dim_MMD]
        #     # rand_subset_shuffled_epsilon = shuffled_epsilon[:self.batch_size_tf//2, :][:, :n_dim_MMD]
        #     # rand_subset_shuffled_not_epsilon = self.epsilon_dist.sample()[self.batch_size_tf//2:, :][:, :n_dim_MMD]
        #     # self.MMD = - helper.compute_MMD(tf.concat([rand_subset_shuffled_posterior_latent_code_1, rand_subset_shuffled_epsilon], axis=1), 
        #     #                               tf.concat([rand_subset_shuffled_posterior_latent_code_2, rand_subset_shuffled_not_epsilon], axis=1))
            
        #     self.Inv_MMD = helper.compute_MMD(tf.concat([self.posterior_latent_code, self.epsilon], axis=1), 
        #                                     tf.concat([self.posterior_latent_code, self.epsilon_dist.sample()], axis=1))
        #     self.MMD = helper.compute_MMD(self.posterior_latent_code, self.prior_dist.sample()) 
        #     self.enc_reg_cost = self.MMD - self.config['enc_inv_MMD_strength']*tf.nn.sigmoid(10*self.Inv_MMD)
        # if self.config['divergence_mode'] == 'MMD': 
        #     self.MMD = helper.compute_MMD(self.posterior_latent_code, self.prior_dist.sample())
        #     self.enc_reg_cost = self.MMD
        if self.config['divergence_mode'] == 'MMD-DENOISE':
            self.MMD = helper.compute_MMD(self.posterior_latent_code, self.prior_dist.sample())
            # self.MMD = helper.compute_MMD(self.input_transformed, self.prior_dist.sample())
            # self.enc_reg_cost = self.MMD+self.config['enc_inv_MMD_strength']*self.noise_reconst_cost
            # self.enc_reg_cost = tf.nn.relu(self.MMD) + self.config['enc_inv_MMD_strength']*self.epsilon_reconst_cost
            self.enc_reg_cost = self.MMD + self.config['enc_inv_MMD_strength']*self.epsilon_reconst_cost

        # if self.config['divergence_mode'] == 'MMD-DECOMPOSED': 
        #     self.decomposition_out_posterior, self.deterioration_out_posterior, self.decomposition_concat_out_posterior, self.deterioration_concat_out_posterior, self.all_hidden_posterior, self.decomposer_costs_posterior = self.Decomposer.forward(self.posterior_latent_code[:,np.newaxis,:])
        #     self.decomposition_out_prior, self.deterioration_out_prior, self.decomposition_concat_out_prior, self.deterioration_concat_out_prior, self.all_hidden_prior, self.decomposer_costs_prior = self.Decomposer.forward(self.prior_dist.sample()[:,np.newaxis,:])
 
        #     self.MMD_reconst_list = []
        #     self.MMD_residual_list = []
        #     self.MMD_hidden_list = []
        #     for i in range(len(self.all_hidden_posterior)):
        #         curr_reconst_MMD = helper.compute_MMD(self.decomposition_out_posterior[:, i, :], self.decomposition_out_prior[:, i, :])
        #         curr_residual_MMD = helper.compute_MMD(self.deterioration_out_posterior[:, i, :], self.deterioration_out_prior[:, i, :])
        #         curr_hidden_MMD = helper.compute_MMD(self.all_hidden_posterior[i], self.all_hidden_prior[i])
        #         self.MMD_reconst_list.append(curr_reconst_MMD[np.newaxis])
        #         self.MMD_residual_list.append(curr_residual_MMD[np.newaxis])
        #         self.MMD_hidden_list.append(curr_hidden_MMD[np.newaxis])
            
        #     self.MMD_last_residual = helper.compute_MMD(self.decomposition_out_posterior[:, -1, :], self.decomposition_out_prior[:, -1, :])
        #     self.MMD_hidden_concat = tf.concat([*self.MMD_hidden_list, self.MMD_last_residual[np.newaxis]], axis=0)
        #     self.MMD_residual_concat = tf.concat([*self.MMD_residual_list, self.MMD_last_residual[np.newaxis]], axis=0)
        #     self.MMD_reconst_concat = tf.concat([*self.MMD_reconst_list, self.MMD_last_residual[np.newaxis]], axis=0)
            
        #     # self.MMD = tf.reduce_sum(self.MMD_hidden_concat)
        #     # self.MMD = tf.reduce_sum(self.MMD_residual_concat)
        #     # self.MMD = tf.reduce_sum(self.MMD_reconst_concat)
        #     # self.MMD = self.MMD_residual_concat[0]+self.MMD_hidden_concat[0] 

        #     self.MMD = tf.nn.relu(100-tf.reduce_mean(self.decomposer_costs_posterior))  #helper.compute_MMD(self.posterior_latent_code, self.prior_dist.sample()) 
        #     # self.enc_reg_cost = self.MMD #+ self.config['enc_inv_MMD_strength']*(-tf.reduce_mean(self.decomposer_costs_posterior))
        #     self.enc_reg_cost = self.MMD
        # if self.config['divergence_mode'] == 'MMD-CONTRACT': 
        #     self.MMD = helper.compute_MMD(self.posterior_latent_code, self.prior_dist.sample())
        #     self.enc_reg_cost = self.MMD+0.1*self.mean_contraction_distance_sq
        # if self.config['divergence_mode'] == 'MMD-MEANDIFF': 
        #     self.MMD = helper.compute_MMD(self.posterior_latent_code, self.prior_dist.sample())
        #     self.enc_reg_cost = self.MMD + 0.1*(self.mean_prior_latent_code_diff_norm_sq-self.mean_posterior_latent_code_diff_norm_sq)**2
        # if self.config['divergence_mode'] == 'NEW-MMD': 
        #     batch_rand_vectors = tf.random_normal(shape=[self.config['enc_inv_MMD_n_trans']*self.config['enc_inv_MMD_n_reflect'], self.config['n_latent']])
        #     batch_rand_dirs = batch_rand_vectors/helper.safe_tf_sqrt(tf.reduce_sum((batch_rand_vectors**2), axis=1, keep_dims=True)) 
        #     batch_rand_dirs_expanded = tf.reshape(batch_rand_dirs, [self.config['enc_inv_MMD_n_trans'], self.config['enc_inv_MMD_n_reflect'], self.config['n_latent']])           
        #     batch_rand_dirs_expanded = tf.stop_gradient(batch_rand_dirs_expanded)          
        #     transformed_batch_input = self.posterior_latent_code[np.newaxis, :, :]
        #     for i in range(self.config['enc_inv_MMD_n_reflect']):
        #         transformed_batch_input = helper.apply_householder_reflections2(transformed_batch_input, batch_rand_dirs_expanded[:, i, :])
        #     transformed_batch_input_inverse = tf.reverse(transformed_batch_input, [1])
            
        #     self.MMD = helper.compute_MMD(tf.concat([self.posterior_latent_code, transformed_batch_input_inverse[0,:,:]], axis=1), 
        #                                 tf.concat([self.prior_dist.sample(), tf.reverse(self.posterior_latent_code, [0])], axis=1), mode='His')
        #     self.enc_reg_cost = self.MMD
        # if self.config['divergence_mode'] == 'DIFF-MMD2':
        #     posterior_latent_diffs = helper.circular_shift(self.posterior_latent_code)-helper.circular_shift(self.circular_shift(self.posterior_latent_code)) 
        #     prior_latent_diffs = self.prior_dist.sample()-self.prior_dist.sample()
        #     self.MMD = helper.compute_MMD(tf.concat([self.posterior_latent_code, posterior_latent_diffs], axis=1), tf.concat([self.prior_dist.sample(), prior_latent_diffs], axis=1))
        #     self.enc_reg_cost = self.MMD
        # if self.config['divergence_mode'] == 'DIFF-MMD':
        #     prior_latent_code_1 = self.prior_dist.sample()
        #     prior_latent_code_2 = self.prior_dist.sample()
        #     prior_latent_code_diff = prior_latent_code_2-prior_latent_code_1
        #     self.MMD = helper.compute_MMD(tf.concat([self.posterior_latent_code, self.posterior_latent_code_diff], axis=1), tf.concat([prior_latent_code_1, prior_latent_code_diff], axis=1))
        #     self.enc_reg_cost = self.MMD
        # if self.config['divergence_mode'] == 'SUBSET-MMD': # assumes normal distributed prior
        #     self.MMD = helper.compute_MMD(self.posterior_latent_code, self.prior_dist.sample())
        #     subset_sizes = 2**np.arange(1, int(np.log2(self.posterior_latent_code.get_shape().as_list()[1]))+1)
        #     for subset_size in subset_sizes:
        #         batch_input_shuffled = helper.tf_differentiable_random_shuffle_with_axis(self.posterior_latent_code, axis=1)
        #         self.MMD += helper.compute_MMD(batch_input_shuffled[:, :subset_size], self.posterior_latent_code[:, :subset_size])
        #     self.enc_reg_cost = self.MMD     
        # if self.config['divergence_mode'] == 'SUM-MMD':
        #     self.Inv_MMD = helper.sum_div(helper.compute_MMD, self.posterior_latent_code)
        #     self.MMD = helper.compute_MMD(self.posterior_latent_code, self.prior_dist.sample())
        #     self.enc_reg_cost = self.MMD + self.config['enc_inv_MMD_strength']*self.Inv_MMD
        # if self.config['divergence_mode'] == 'INV-MMD': 
        #     batch_rand_vectors = tf.random_normal(shape=[self.config['enc_inv_MMD_n_trans'], self.config['n_latent']])
        #     batch_rand_dirs = batch_rand_vectors/helper.safe_tf_sqrt(tf.reduce_sum((batch_rand_vectors**2), axis=1, keep_dims=True))            
        #     self.Inv_MMD = helper.stable_div(helper.compute_MMD, self.posterior_latent_code, batch_rand_dirs)
        #     self.MMD = helper.compute_MMD(self.posterior_latent_code, self.prior_dist.sample())
        #     self.enc_reg_cost = self.MMD + self.config['enc_inv_MMD_strength']*self.Inv_MMD
        # if self.config['divergence_mode'] == 'INV-MMD2': 
        #     batch_rand_vectors = tf.random_normal(shape=[self.config['enc_inv_MMD_n_trans']*self.config['enc_inv_MMD_n_reflect'], self.config['n_latent']])
        #     batch_rand_dirs = batch_rand_vectors/helper.safe_tf_sqrt(tf.reduce_sum((batch_rand_vectors**2), axis=1, keep_dims=True)) 
        #     batch_rand_dirs_expanded = tf.reshape(batch_rand_dirs, [self.config['enc_inv_MMD_n_trans'], self.config['enc_inv_MMD_n_reflect'], self.config['n_latent']])           
        #     batch_rand_dirs_expanded = tf.stop_gradient(batch_rand_dirs_expanded)          
        #     self.Inv_MMD = helper.stable_div_expanded(helper.compute_MMD, self.posterior_latent_code, batch_rand_dirs_expanded)
        #     self.MMD = helper.compute_MMD(self.posterior_latent_code, self.prior_dist.sample())
        #     self.enc_reg_cost = self.MMD + self.config['enc_inv_MMD_strength']*self.Inv_MMD
        # if self.config['divergence_mode'] == 'GAN' or self.config['divergence_mode'] == 'NS-GAN':
        #     self.div_posterior = self.Diverger.forward(self.posterior_latent_code_expanded)        
        #     self.div_prior = self.Diverger.forward(self.prior_latent_code_expanded)   
        #     self.mean_z_divergence = tf.reduce_mean(tf.log(10e-7+self.div_prior))+tf.reduce_mean(tf.log(10e-7+1-self.div_posterior))
        #     if self.config['divergence_mode'] == 'NS-GAN': 
        #         self.enc_reg_cost = -tf.reduce_mean(tf.log(10e-7+self.div_posterior))
        #     elif self.config['divergence_mode'] == 'GAN': 
        #         self.enc_reg_cost = tf.reduce_mean(tf.log(10e-7+1-self.div_posterior))

        #############################################################################
        # REGULARIZER

        self.reg_target_dist = self.reconst_dist
        self.reg_target_sample = self.reconst_sample
        self.reg_dist = self.reconst_dist
        self.reg_sample = self.reconst_sample
        
        #############################################################################
        # OBJECTIVES
        ### Divergence
        # if self.config['divergence_mode'] == 'GAN' or self.config['divergence_mode'] == 'NS-GAN':
        #     self.div_cost = self.config['enc_reg_strength']*(-self.mean_z_divergence)
        #     # self.div_cost = (-self.mean_z_divergence)
        # elif self.config['divergence_mode'] == 'MMD-DENOISE' or self.config['divergence_mode'] == 'MMD-DECOMPOSED':

        # if self.config['divergence_mode'] == 'MMD-DECOMPOSED':
        #     self.div_cost = tf.reduce_mean(self.decomposer_costs_posterior) + tf.nn.relu(100-tf.reduce_mean(self.decomposer_costs_prior))
            # self.div_cost = tf.reduce_mean(self.decomposer_costs_posterior) -tf.reduce_mean(self.decomposer_costs_prior)

        ### Encoder
        b_use_timer, timescale, starttime = False, 5, 10
        self.OT_primal = helper.sample_distance_function(self.input_sample, self.reconst_sample)
        self.mean_OT_primal = tf.reduce_mean(self.OT_primal)
        if b_use_timer:
            self.mean_POT_primal = self.mean_OT_primal+helper.hardstep((self.epoch-float(starttime))/float(timescale)+0.0001)*self.config['enc_reg_strength']*self.enc_reg_cost
        else:
            self.mean_POT_primal = self.mean_OT_primal+self.config['enc_reg_strength']*self.enc_reg_cost
        self.enc_cost = self.mean_POT_primal

        ### Critic
        # self.cri_reg_cost = self.input_transformed_reconst_cost
        self.cri_cost = self.epsilon_reconst_cost

        ### Generator
        self.gen_cost = self.mean_OT_primal








            # alpha_MMD = 0.5
            # n_transformations = 10
            # v_householder_pre = tf.stop_gradient(self.prior_dist.sample())[:n_transformations, :]
            # v_householder = v_householder_pre/helper.safe_tf_sqrt(tf.reduce_sum((v_householder_pre**2), axis=1, keep_dims=True))
            # v_householder = v_householder[np.newaxis, :,:]
            # batched_posterior_latent_code = self.posterior_latent_code[:,np.newaxis, :]
            # householder_inner = tf.reduce_sum(v_householder*batched_posterior_latent_code, axis=2, keep_dims=True)  
            # transformed_posterior_latent_code = batched_posterior_latent_code-2*householder_inner*v_householder
            # list_transformed_posterior = tf.split(transformed_posterior_latent_code, n_transformations, axis=1)
            # self.Inv_MMD = 0
            # for e in list_transformed_posterior: 
            #     self.Inv_MMD += helper.compute_MMD(e[:,0,:], self.posterior_latent_code)
            # self.Inv_MMD /= n_transformations

            # self.MMD = helper.compute_MMD(self.posterior_latent_code, self.prior_dist.sample())
            # self.enc_reg_cost = (1-alpha_MMD)*self.MMD + alpha_MMD*100*self.Inv_MMD





            # self.decomposition_out_posterior_scaled = self.decomposition_out_posterior
            # self.decomposition_out_prior_scaled = self.decomposition_out_prior
            # self.decomposition_out_posterior_scaled = self.decomposition_out_posterior*(tf.range(self.decomposition_out_posterior.get_shape().as_list()[1],dtype=tf.float32)[np.newaxis, :, np.newaxis]+1)
            # self.decomposition_out_prior_scaled = self.decomposition_out_prior*(tf.range(self.decomposition_out_prior.get_shape().as_list()[1],dtype=tf.float32)[np.newaxis, :, np.newaxis]+1)
            











        # # #################################################################################
        # self.reg_sample_param = {'image': None, 'flat': None}
        # try: 
        #     self.reg_sample_param['image'] = self.uniform_sample_1[:, np.newaxis, :, np.newaxis, np.newaxis]*self.reg_target_sample['image']+\
        #                                     (1-self.uniform_sample_1[:, np.newaxis, :, np.newaxis, np.newaxis])*self.input_sample['image']
        # except: 
        #     self.reg_sample_param['flat'] = self.uniform_sample_1[:, np.newaxis, :]*self.reg_target_sample['flat']+\
        #                                     (1-self.uniform_sample_1[:, np.newaxis, :])*self.input_sample['flat']
        # self.reg_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.reg_sample_param)
        # self.reg_sample = self.reg_dist.sample(b_mode=True)

        # self.critic_real = self.Discriminator.forward(self.input_sample)
        # self.critic_gen = self.Discriminator.forward(self.obs_sample)
        # self.critic_reg = self.Discriminator.forward(self.reg_sample)

        # self.real_reg_distances_sq = self.metric_distance_sq(self.input_sample, self.reg_sample)
        # self.real_reg_slopes_sq = ((self.critic_real-self.critic_reg)**2)/(self.real_reg_distances_sq+1e-7)

        # try:
        #     self.convex_grad = tf.gradients(self.critic_reg, [self.reg_sample['image']])[0]
        #     self.convex_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.convex_grad**2, axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis])
        # except: 
        #     self.convex_grad = tf.gradients(self.critic_reg, [self.reg_sample['flat']])[0]
        #     self.convex_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.convex_grad**2, axis=[-1], keep_dims=True))      
        # self.gradient_penalties = ((self.convex_grad_norm-1)**2)

        # self.mean_critic_real = tf.reduce_mean(self.critic_real)
        # self.mean_critic_gen = tf.reduce_mean(self.critic_gen)
        # self.mean_critic_reg = tf.reduce_mean(self.critic_reg)

        # self.mean_real_reg_slopes_sq = tf.reduce_mean(self.real_reg_slopes_sq)
        # self.mean_gradient_penalty = tf.reduce_mean(self.gradient_penalties)

        # self.regularizer_cost = 10*self.mean_gradient_penalty
        # self.discriminator_cost = -self.mean_critic_real+self.mean_critic_gen+self.regularizer_cost
        # self.generator_cost = -self.mean_critic_gen
        # self.transporter_cost = -self.mean_real_reg_slopes_sq

# #################################################################################
        # self.reg_sample_param = {'image': None, 'flat': None}
        # try: 
        #     self.reg_sample_param['image'] = self.uniform_sample_1[:, np.newaxis, :, np.newaxis, np.newaxis]*self.reg_target_sample['image']+\
        #                                     (1-self.uniform_sample_1[:, np.newaxis, :, np.newaxis, np.newaxis])*self.input_sample['image']
        # except: 
        #     self.reg_sample_param['flat'] = self.uniform_sample_1[:, np.newaxis, :]*self.reg_target_sample['flat']+\
        #                                     (1-self.uniform_sample_1[:, np.newaxis, :])*self.input_sample['flat']
        # self.reg_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.reg_sample_param)
        # self.reg_sample = self.reg_dist.sample(b_mode=True)

        # self.critic_real = self.Discriminator.forward(self.input_sample)
        # self.critic_gen = self.Discriminator.forward(self.obs_sample)
        # self.critic_reg = self.Discriminator.forward(self.reg_sample)

        # self.real_reg_distances_sq = self.metric_distance_sq(self.input_sample, self.reg_sample)
        # self.real_reg_slopes_sq = ((self.critic_real-self.critic_reg))/(helper.safe_tf_sqrt(self.real_reg_distances_sq)+1e-7)

        # try:
        #     self.convex_grad = tf.gradients(self.critic_reg, [self.reg_sample['image']])[0]
        #     self.convex_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.convex_grad**2, axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis])
        # except: 
        #     self.convex_grad = tf.gradients(self.critic_reg, [self.reg_sample['flat']])[0]
        #     self.convex_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.convex_grad**2, axis=[-1], keep_dims=True))      
        # self.gradient_penalties = ((self.convex_grad_norm-1)**2)

        # self.mean_critic_real = tf.reduce_mean(self.critic_real)
        # self.mean_critic_gen = tf.reduce_mean(self.critic_gen)
        # self.mean_critic_reg = tf.reduce_mean(self.critic_reg)

        # self.mean_real_reg_slopes_sq = tf.reduce_mean(self.real_reg_slopes_sq)
        # self.mean_gradient_penalty = tf.reduce_mean(self.gradient_penalties)

        # self.regularizer_cost = 10*self.mean_gradient_penalty
        # self.discriminator_cost = -self.mean_critic_real+self.mean_critic_gen+self.regularizer_cost
        # self.generator_cost = -self.mean_critic_gen
        # self.transporter_cost = -self.mean_real_reg_slopes_sq


# #################################################################################





        # self.posterior_latent_code = self.prior_dist.sample()
        # self.posterior_latent_code_expanded = tf.reshape(self.posterior_latent_code, [-1, 1, *self.posterior_latent_code.get_shape().as_list()[1:]])
        # self.posterior_latent_code_logpdf = self.prior_dist.log_pdf(self.posterior_latent_code)
        # self.transport_param = self.TransportPlan.forward(self.input_sample)
        # self.transport_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.transport_param)
        # transport_sample = self.transport_dist.sample(b_mode=True)
        # self.obs_reconst_dist = self.transport_dist

        # latent_sample_out = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in latent_list], axis=1)
        # dec_input = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in dec_input_list], axis=1)
        # obs_param = self.Generator.forward(dec_input)  

        # obs_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = obs_param) 
        # obs_sample_out = obs_dist.sample(b_mode=True) 



# def normalized_euclidean_distance_squared(self, x, y, axis=[-1], keep_dims=True):
#         # return tf.sqrt(tf.reduce_sum(tf.abs(x-y), axis=axis, keep_dims=keep_dims))
#         # return tf.reduce_sum(tf.abs(x-y), axis=axis, keep_dims=keep_dims)
#         # return tf.sqrt(tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims))
        
#         # return 10*tf.reduce_sum(tf.clip_by_value(tf.abs(x-y), 1e-10, np.inf), axis=axis, keep_dims=keep_dims)

#         # return tf.clip_by_value(tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims), 1e-10, np.inf)**(1/2)
#         # return tf.sqrt(tf.clip_by_value(tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims), 1e-7, np.inf))
        
#         # scale = 1/np.sqrt(np.prod(x.get_shape().as_list()[2:]))
#         # return scale*tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims)

#         return tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims)
#         # c_xx = tf.reduce_sum((x*x), axis=axis, keep_dims=keep_dims)
#         # c_yy = tf.reduce_sum((y*y), axis=axis, keep_dims=keep_dims)
#         # c_xy = tf.reduce_sum((x*y), axis=axis, keep_dims=keep_dims)
#         # return c_xx+c_yy-2*c_xy
        
#         # return 100*tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims)
#         # return tf.reduce_sum(tf.clip_by_value(tf.abs(x-y), 1e-2, np.inf), axis=axis, keep_dims=keep_dims)
#         # return tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims)#+tf.reduce_sum(tf.clip_by_value(tf.abs(x-y), 1e-2, np.inf), axis=axis, keep_dims=keep_dims)

#         # return 2-2*tf.exp(-tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims)/5)



        # train_outs_dict = {'generator_cost': train_generator_cost, 
        #                    'discriminator_cost': train_discriminator_cost,
        #                    'transporter_cost': train_transporter_cost,
        #                    # 'penalty_term': tf.reduce_mean(critic_real)-tf.reduce_mean(critic_gen),
        #                    'penalty_term': penalty_term,
        #                    'critic_real': critic_real,
        #                    'critic_gen': critic_gen,
        #                    'mean_transport_cost': mean_transport_cost,
        #                    'convex_mask': convex_mask,
        #                    'input_sample': input_sample,
        #                    'transport_sample': transport_sample,
        #                    'expected_log_pdf_prior': expected_log_pdf_prior,
        #                    'expected_log_pdf_agg_post': expected_log_pdf_agg_post}

        # test_outs_dict = {'generator_cost': test_generator_cost, 
        #                   'discriminator_cost': test_discriminator_cost,
        #                   'transporter_cost': test_transporter_cost,
        #                   # 'penalty_term': tf.reduce_mean(critic_real)-tf.reduce_mean(critic_gen),
        #                   'penalty_term': penalty_term,
        #                   'critic_real': critic_real,
        #                   'critic_gen': critic_gen,
        #                   'mean_transport_cost': mean_transport_cost,
        #                   'convex_mask': convex_mask,
        #                   'input_sample': input_sample,
        #                   'transport_sample': transport_sample,
        #                   'expected_log_pdf_prior': expected_log_pdf_prior,
        #                   'expected_log_pdf_agg_post': expected_log_pdf_agg_post}

        
        # train_transporter_cost = -tf.reduce_mean((critic_real-critic_transported)/(transport_cost+1e-7))

        # try: transport_cost = self.normalized_euclidean_distance_squared(transport_sample['flat'], batch['observed']['data']['flat'], axis=[-1])
        # except: transport_cost = self.normalized_euclidean_distance_squared(transport_sample['image'], batch['observed']['data']['image'], axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis]
        # mean_transport_cost = tf.reduce_mean(transport_cost)

        # try: transport_cost_random = self.normalized_euclidean_distance_squared(mixture_sample['flat'], batch['observed']['data']['flat'], axis=[-1])
        # except: transport_cost_random = self.normalized_euclidean_distance_squared(mixture_sample['image'], batch['observed']['data']['image'], axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis]
        # mean_transport_cost_random = tf.reduce_mean(transport_cost_random)
        

        # penalty_term = 100*tf.reduce_mean((((critic_real-critic_transported)/(transport_cost+1e-7))-1)**2) # works somewhat
        # penalty_term = 10*tf.reduce_mean((((critic_real-critic_transported)/(transport_cost+1e-7))-1)**2) # works better
        # penalty_term = 100*tf.reduce_mean(((critic_real-critic_transported)-transport_cost))**2)

        # penalty_term = 10*tf.reduce_mean((((critic_real-critic_transported_random)/(transport_cost_random+1e-7))-1)**2) # works better

       

        # [self.prior_dist_list, self.posterior_dist_list, prior_param_list, posterior_param_list, latent_list, kl_sample_list, \
        #  kl_analytic_list, state_list, obs_encoded_list, context_encoded_list, dec_input_list] = helper.generate_empty_lists(11, self.n_time)


        # [self.prior_dist_list, self.posterior_dist_list, self.posterior_base_dist_list, prior_param_list, posterior_param_list,
        #  posterior_transform_list, latent_list, latent_list_aux, kl_sample_list, kl_analytic_list, state_list, obs_encoded_list, context_encoded_list, 
        #  dec_input_list, dec_input_list_aux] = helper.generate_empty_lists(15, self.n_time)


    # def inference(self, batch, additional_inputs_tf):
    #     epoch = additional_inputs_tf[0]
    #     b_identity = additional_inputs_tf[1]
    #     meanp = additional_inputs_tf[2]
    #     p_real = additional_inputs_tf[3]

    #     if len(batch['observed']['properties']['flat'])>0:
    #         for e in batch['observed']['properties']['flat']: e['dist']='dirac'
    #     else:
    #         for e in batch['observed']['properties']['image']: e['dist']='dirac'

    #     if not self.bModules: self.generate_modules(batch)
    #     try: self.n_time = batch['observed']['properties']['flat'][0]['size'][1]
    #     except: self.n_time = batch['observed']['properties']['image'][0]['size'][1]
    #     try: batch_size_tf = tf.shape(batch['observed']['data']['flat'])[0]
    #     except: batch_size_tf = tf.shape(batch['observed']['data']['image'])[0]
        
    #     [self.prior_dist_list, self.posterior_dist_list, self.posterior_base_dist_list, prior_param_list, posterior_param_list,
    #      posterior_transform_list, latent_list, latent_list_aux, kl_sample_list, kl_analytic_list, state_list, obs_encoded_list, context_encoded_list, 
    #      dec_input_list, dec_input_list_aux] = helper.generate_empty_lists(15, self.n_time)

    #     uniform_dist = distributions.UniformDistribution(params = tf.concat([tf.zeros(shape=(batch_size_tf, 1)), tf.ones(shape=(batch_size_tf, 1))], axis=1))
    #     uniform_w = uniform_dist.sample()

    #     prior_param_list[0] = self.PriorMap.forward((tf.zeros(shape=(batch_size_tf, 1)),))
    #     self.prior_dist_list[0] = distributions.DiagonalGaussianDistribution(params = prior_param_list[0])
    #     latent_list[0] = self.prior_dist_list[0].sample()
    #     dec_input_list[0] = self.ObservationMap.forward((latent_list[0],))
    #     # latent_list_aux[0] = self.prior_dist_list[0].sample()
    #     # dec_input_list_aux[0] = self.ObservationMap.forward((latent_list_aux[0],))

    #     self.prior_dist_list[0] = distributions.DiagonalGaussianDistribution(params = prior_param_list[0])
        
    #     real_transport_dist_mean = tf.ones(shape=(batch_size_tf, 1))*p_real
    #     real_transport_dist = distributions.BernoulliDistribution(params = real_transport_dist_mean, b_sigmoid=False)
    #     real_transport_sample = real_transport_dist.sample()        
    #     self.AAA = real_transport_sample

    #     self.latent_sample_out = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in latent_list], axis=1)
    #     dec_input = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in dec_input_list], axis=1)
    #     # dec_input_aux = tf.concat([tf.reshape(e, [-1, 1, *e.get_shape().as_list()[1:]]) for e in dec_input_list_aux], axis=1)

    #     obs_param = self.Generator.forward(dec_input)
    #     # obs_param_aux = self.Generator.forward(dec_input_aux) 

    #     self.obs_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = obs_param)
    #     obs_sample = self.obs_dist.sample(b_mode=True)
    #     self.obs_sample = obs_sample

    #     # obs_dist_aux = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = obs_param)
    #     # obs_sample_aux = obs_dist_aux.sample(b_mode=True)

    #     # ###############################################################################
    #     # dec_rec_input = self.EncodingPlan.forward(batch['observed']['data'])
    #     # rec_param = self.Generator.forward(dec_rec_input) 
    #     # self.rec_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = rec_param)
    #     # rec_sample = self.rec_dist.sample(b_mode=True)

    #     # transport_param, convex_mask = rec_param, uniform_w
    #     # self.transport_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = transport_param)
    #     # transport_sample = self.transport_dist.sample(b_mode=True)
    #     # ##############################################################################
    #     # bb_var_inference_noise = self.prior_dist_list[0].sample()[:, :50]
    #     # # bb_var_inference_noise = None
    #     # dec_rec_input = self.EncodingPlan.forward(batch['observed']['data'], noise=bb_var_inference_noise)
    #     # rec_param = self.Generator.forward(dec_rec_input)
    #     # self.rec_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = rec_param)
    #     # rec_sample = self.rec_dist.sample(b_mode=True)

    #     # transport_param, convex_mask = self.MixingPlan.forward(batch['observed']['data'], rec_sample, noise=uniform_w, t=epoch)
    #     # self.transport_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = transport_param)
    #     # transport_sample = self.transport_dist.sample(b_mode=True)
    #     # ################################################################################


    #     transport_param, convex_mask = self.TransportPlan.forward(batch['observed']['data'], rec_sample=None, aux_sample=None, noise=uniform_w, b_identity=b_identity, real_transport_sample=real_transport_sample)
    #     # transport_param, convex_mask = self.TransportPlan.forward(batch['observed']['data'], rec_sample=rec_sample, aux_sample=None, noise=uniform_w, b_identity=b_identity)
    #     self.transport_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = transport_param)
    #     transport_sample = self.transport_dist.sample(b_mode=True)
    #     self.rec_dist = self.transport_dist
    #     #################################################################################

    #     real_features = self.TransportCostFeatureMap.forward(batch['observed']['data'])
    #     transport_features = self.TransportCostFeatureMap.forward(transport_sample)

    #     # try: transport_cost = self.normalized_euclidean_distance_squared(transport_sample['flat'], batch['observed']['data']['flat'], axis=[-1])
    #     # except: transport_cost = self.normalized_euclidean_distance_squared(transport_sample['image'], batch['observed']['data']['image'], axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis]

    #     try: transport_cost = self.normalized_euclidean_distance_squared(transport_features['flat'], real_features['flat'], axis=[-1])
    #     except: transport_cost = self.normalized_euclidean_distance_squared(transport_features['image'], real_features['image'], axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis]
        

    #     mean_transport_cost = tf.reduce_mean(transport_cost)

    #     critic_real = self.Discriminator.forward(transport_sample)
    #     critic_gen = self.Discriminator.forward(obs_sample)
        
    #     # critic_real = (self.Discriminator.forward(transport_sample)-meanp)/stdp
    #     # critic_gen = (self.Discriminator.forward(obs_sample)-meanp)/stdp


    #     # train_discriminator_cost = -tf.reduce_mean(critic_real)+tf.reduce_mean(critic_gen)
    #     # # train_generator_cost = -tf.reduce_mean(critic_gen)
    #     # train_generator_cost = tf.reduce_mean(transport_cost)+tf.reduce_mean(critic_real)-tf.reduce_mean(critic_gen)
    #     # train_transporter_cost = tf.reduce_mean(transport_cost)+tf.reduce_mean(critic_real)

    #     # train_discriminator_cost = (-tf.reduce_mean(critic_real)+tf.reduce_mean(critic_gen))
    #     # train_generator_cost = (-tf.reduce_mean(critic_gen))
    #     # train_transporter_cost = tf.reduce_mean(transport_cost)+tf.reduce_mean(critic_real)

    #     # scale_factor = (epoch**2)
    #     # train_discriminator_cost = (-tf.reduce_mean(critic_real)+tf.reduce_mean(critic_gen))/scale_factor
    #     # train_generator_cost = (-tf.reduce_mean(critic_gen))/scale_factor
    #     # train_transporter_cost = tf.reduce_mean(transport_cost)+tf.reduce_mean(critic_real)/scale_factor

    #     # scale_factor = (1)
    #     # m = scale_factor/(scale_factor+1)
    #     # train_discriminator_cost = (-tf.reduce_mean(critic_real)+tf.reduce_mean(critic_gen))
    #     # train_generator_cost = (-tf.reduce_mean(critic_gen))
    #     # train_transporter_cost = 2*(m*tf.reduce_mean(transport_cost)+(1-m)*tf.reduce_mean(critic_real))

    #     scale_factor = epoch
    #     train_discriminator_cost = (-tf.reduce_mean(critic_real)+tf.reduce_mean(critic_gen))/scale_factor
    #     train_generator_cost = tf.reduce_mean(transport_cost)+(tf.reduce_mean(critic_real)-tf.reduce_mean(critic_gen))/scale_factor
    #     train_transporter_cost = tf.reduce_mean(transport_cost)+tf.reduce_mean(critic_real)/scale_factor

    #     expected_log_pdf_prior = tf.reduce_mean(self.prior_dist_list[0].log_pdf(self.prior_dist_list[0].sample()))
    #     # expected_log_pdf_agg_post = tf.reduce_mean(self.prior_dist_list[0].log_pdf(dec_rec_input[:,0,:]))
    #     expected_log_pdf_agg_post = expected_log_pdf_prior

    #     test_discriminator_cost = train_discriminator_cost
    #     test_generator_cost = train_generator_cost           
    #     test_transporter_cost = train_transporter_cost           

    #     input_sample = batch['observed']['data']
    #     train_outs_dict = {'generator_cost': train_generator_cost, 
    #                        'discriminator_cost': train_discriminator_cost,
    #                        'transporter_cost': train_transporter_cost,
    #                        'critic_real': critic_real,
    #                        'critic_gen': critic_gen,
    #                        'mean_transport_cost': mean_transport_cost,
    #                        'convex_mask': convex_mask,
    #                        'input_sample': input_sample,
    #                        'transport_sample': transport_sample,
    #                        'expected_log_pdf_prior': expected_log_pdf_prior,
    #                        'expected_log_pdf_agg_post': expected_log_pdf_agg_post}

    #     test_outs_dict = {'generator_cost': test_generator_cost, 
    #                       'discriminator_cost': test_discriminator_cost,
    #                       'transporter_cost': test_transporter_cost,
    #                       'critic_real': critic_real,
    #                       'critic_gen': critic_gen,
    #                       'mean_transport_cost': mean_transport_cost,
    #                       'convex_mask': convex_mask,
    #                       'input_sample': input_sample,
    #                       'transport_sample': transport_sample,
    #                       'expected_log_pdf_prior': expected_log_pdf_prior,
    #                       'expected_log_pdf_agg_post': expected_log_pdf_agg_post}


    #     return train_outs_dict, test_outs_dict

