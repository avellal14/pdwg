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

import models.WAECompactFlow.ModelMaps as ModelMaps

import distributions 
import transforms
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
        # self.config['divergence_mode'] # GAN, NS-GAN, MMD, INV-MMD   
        # self.config['dual_dist_mode'] # Coupling, Prior
        # self.config['critic_reg_mode'] # ['Coupling Gradient Vector', 'Coupling Gradient Norm', 'Trivial Gradient Norm', 'Uniform Gradient Norm', 'Coupling Lipschitz', 'Trivial Lipschitz', 'Uniform Lipschitz']

        if self.config['sample_distance_mode'] == 'Euclidean': self.sample_distance_function = helper.euclidean_distance
        if self.config['sample_distance_mode'] == 'Quadratic': self.sample_distance_function = helper.quadratic_distance
        if self.config['kernel_mode'] == 'InvMultiquadratics': self.kernel_function = helper.inv_multiquadratics_kernel
        if self.config['kernel_mode'] == 'RadialBasisFunction': self.kernel_function = helper.rbf_kernel
        if self.config['kernel_mode'] == 'RationalQuadratic': self.kernel_function = helper.rational_quadratic_kernel

    def generate_modules(self, batch):
        self.PriorMap = ModelMaps.PriorMapGaussian(self.config)
        self.FlowMap = ModelMaps.FlowMap(self.config)
        self.Diverger = ModelMaps.Diverger({**self.config, 'data_properties': batch['observed']['properties']})
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
        
        self.gen_pre_prior_param = self.PriorMap.forward((tf.zeros(shape=(self.gen_batch_size_tf, 1)),))
        self.gen_pre_prior_dist = distributions.DiagonalGaussianDistribution(params = self.gen_pre_prior_param)
        self.gen_pre_prior_latent_code = self.gen_pre_prior_dist.sample()
        self.gen_pre_prior_latent_code_log_pdf = self.gen_pre_prior_dist.log_pdf(self.gen_pre_prior_latent_code)

        self.gen_flow_param_list = self.FlowMap.forward()
        self.gen_flow_object = transforms.SerialFlow([\
                                                      transforms.NonLinearIARFlow(self.gen_flow_param_list[0], self.config['n_latent']), 
                                                      transforms.NonLinearIARFlow(self.gen_flow_param_list[1], self.config['n_latent']),
                                                      transforms.NonLinearIARFlow(self.gen_flow_param_list[2], self.config['n_latent']),
                                                      transforms.NonLinearIARFlow(self.gen_flow_param_list[3], self.config['n_latent']),
                                                      transforms.NonLinearIARFlow(self.gen_flow_param_list[4], self.config['n_latent']),
                                                      ])
        self.gen_prior_latent_code, self.gen_prior_latent_code_log_pdf = self.gen_flow_object.inverse_transform(self.gen_pre_prior_latent_code, self.gen_pre_prior_latent_code_log_pdf)

        self.gen_obs_sample_param = self.Generator.forward(self.gen_prior_latent_code[:, np.newaxis, :])
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

        self.pre_prior_param = self.PriorMap.forward((tf.zeros(shape=(self.batch_size_tf, 1)),))
        self.pre_prior_dist = distributions.DiagonalGaussianDistribution(params = self.pre_prior_param)        
        self.pre_prior_latent_code = self.pre_prior_dist.sample()
        self.pre_prior_latent_code_log_pdf = self.pre_prior_dist.log_pdf(self.pre_prior_latent_code)
        
        self.flow_param_list = self.FlowMap.forward()
        self.flow_object = transforms.SerialFlow([ \
                                                  transforms.NonLinearIARFlow(self.flow_param_list[0], self.config['n_latent']), 
                                                  transforms.NonLinearIARFlow(self.flow_param_list[1], self.config['n_latent']),
                                                  transforms.NonLinearIARFlow(self.flow_param_list[2], self.config['n_latent']),
                                                  transforms.NonLinearIARFlow(self.flow_param_list[3], self.config['n_latent']),
                                                  transforms.NonLinearIARFlow(self.flow_param_list[4], self.config['n_latent']),
                                                  ])
        self.prior_latent_code, self.prior_latent_code_log_pdf = self.flow_object.inverse_transform(self.pre_prior_latent_code, self.pre_prior_latent_code_log_pdf)

        self.obs_sample_param = self.Generator.forward(self.prior_latent_code[:, np.newaxis, :])
        self.obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.obs_sample_param)
        self.obs_sample = self.obs_sample_dist.sample(b_mode=True)        

        if not os.path.exists(str(Path.home())+'/ExperimentalResults/FixedSamples/'): os.makedirs(str(Path.home())+'/ExperimentalResults/FixedSamples/')
        if os.path.exists(str(Path.home())+'/ExperimentalResults/FixedSamples/np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz'): 
            np_constant_prior_sample = np.load(str(Path.home())+'/ExperimentalResults/FixedSamples/np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz')
        else:
            np_constant_prior_sample = np.random.normal(loc=0., scale=1., size=[400, self.prior_latent_code.get_shape().as_list()[-1]])
            np.save(str(Path.home())+'/ExperimentalResults/FixedSamples/np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz', np_constant_prior_sample)    
        
        self.constant_pre_prior_latent_code = tf.constant(np.asarray(np_constant_prior_sample), dtype=np.float32)
        self.constant_prior_latent_code, _ = self.flow_object.inverse_transform(self.constant_pre_prior_latent_code, tf.zeros(shape=(self.batch_size_tf, 1)))

        self.constant_obs_sample_param = self.Generator.forward(self.constant_prior_latent_code[:, np.newaxis, :])
        self.constant_obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.constant_obs_sample_param)
        self.constant_obs_sample = self.constant_obs_sample_dist.sample(b_mode=True)
    
        if self.config['n_latent'] == 2: 
            grid_scale = 3
            x = np.linspace(-grid_scale, grid_scale, 20)
            y = np.linspace(grid_scale, -grid_scale, 20)
            xv, yv = np.meshgrid(x, y)
            np_constant_prior_grid_sample = np.concatenate((xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis][:]), axis=1)
            self.constant_prior_grid_latent_code = tf.constant(np.asarray(np_constant_prior_grid_sample), dtype=np.float32)

            self.constant_obs_grid_sample_param = self.Generator.forward(self.constant_prior_grid_latent_code[:, np.newaxis, :])
            self.constant_obs_grid_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.constant_obs_grid_sample_param)
            self.constant_obs_grid_sample = self.constant_obs_grid_sample_dist.sample(b_mode=True)
                
        #############################################################################
        # ENCODER 
        self.epsilon_param = self.PriorMap.forward((tf.zeros(shape=(self.batch_size_tf, 1)),))
        self.epsilon_dist = distributions.DiagonalGaussianDistribution(params=self.epsilon_param) 
        
        if self.config['encoder_mode'] == 'Deterministic': 
            self.epsilon = None
        if self.config['encoder_mode'] == 'Gaussian' or self.config['encoder_mode'] == 'UnivApprox' or self.config['encoder_mode'] == 'UnivApproxNoSpatial' or self.config['encoder_mode'] == 'UnivApproxSine':         
            self.epsilon = self.epsilon_dist.sample()

        # self.pre_posterior_latent_code = 0.9*tf.nn.sigmoid(self.Encoder.forward(self.input_sample, noise=self.epsilon))[:,0,:]
        self.pre_posterior_latent_code = self.Encoder.forward(self.input_sample, noise=self.epsilon)[:,0,:]
        self.nball_param = tf.concat([self.pre_posterior_latent_code, 0.05*tf.ones(shape=(self.batch_size_tf, 1))], axis=1)
        self.nball_dist = distributions.UniformBallDistribution(params=self.nball_param) 
        self.posterior_latent_code = self.nball_dist.sample()
        self.posterior_latent_code_log_pdf = -np.log(50000)+self.nball_dist.log_pdf(self.posterior_latent_code)

        self.transformed_posterior_latent_code, self.transformed_posterior_latent_code_log_pdf = self.flow_object.transform(self.posterior_latent_code, self.posterior_latent_code_log_pdf)
        self.KL_transformed_prior_per = self.transformed_posterior_latent_code_log_pdf-self.pre_prior_dist.log_pdf(self.transformed_posterior_latent_code)
        self.KL_transformed_prior = tf.reduce_mean(self.KL_transformed_prior_per)
        
        # self.FF_1, self.FF_2 = transforms.InverseOpenIntervalDimensionFlow(self.config['n_latent']).transform(self.posterior_latent_code, self.posterior_latent_code_log_pdf)

        self.interpolated_posterior_latent_code = helper.interpolate_latent_codes(self.posterior_latent_code, size=self.batch_size_tf//2)
        self.interpolated_obs = self.Generator.forward(self.interpolated_posterior_latent_code) 

        self.reconst_param = self.Generator.forward(self.posterior_latent_code[:, np.newaxis, :]) 
        self.reconst_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.reconst_param)
        self.reconst_sample = self.reconst_dist.sample(b_mode=True)

        ### Primal Penalty
        
        #############################################################################
        # REGULARIZER

        self.reg_target_dist = self.reconst_dist
        self.reg_target_sample = self.reconst_sample
        self.reg_dist = self.reconst_dist
        self.reg_sample = self.reconst_sample
        
        #############################################################################

        # OBJECTIVES
        # Divergence
        # if self.config['divergence_mode'] == 'GAN' or self.config['divergence_mode'] == 'NS-GAN':
        #     self.div_cost = -(tf.reduce_mean(tf.log(tf.nn.sigmoid(self.div_posterior)+10e-7))+tf.reduce_mean(tf.log(1-tf.nn.sigmoid(self.div_prior)+10e-7)))
        # if self.config['divergence_mode'] == 'WGAN-GP':
        #     uniform_dist = distributions.UniformDistribution(params = tf.concat([tf.zeros(shape=(self.batch_size_tf, 1)), tf.ones(shape=(self.batch_size_tf, 1))], axis=1))
        #     uniform_w = uniform_dist.sample()
        #     self.trivial_line = uniform_w[:,np.newaxis,:]*self.pre_posterior_latent_code_expanded+(1-uniform_w[:,np.newaxis,:])*self.prior_latent_code_expanded
        #     self.div_trivial_line = self.Diverger.forward(self.trivial_line)
        #     self.trivial_line_grad = tf.gradients(tf.reduce_sum(self.div_trivial_line), [self.trivial_line])[0]
        #     self.trivial_line_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.trivial_line_grad**2, axis=-1, keep_dims=False)[:,:,np.newaxis])
        #     self.trivial_line_grad_norm_1_penalties = ((self.trivial_line_grad_norm-1)**2)
        #     self.div_reg_cost = tf.reduce_mean(self.trivial_line_grad_norm_1_penalties)
        #     # self.div_cost = -(tf.reduce_mean(self.div_posterior)-tf.reduce_mean(self.div_prior))+10*self.div_reg_cost
        self.div_cost = self.KL_transformed_prior

        # ### Encoder
        # b_use_timer, timescale, starttime = False, 10, 5
        self.OT_primal = self.sample_distance_function(self.input_sample, self.reconst_sample)
        self.mean_OT_primal = tf.reduce_mean(self.OT_primal)
        # if b_use_timer:
        #     self.mean_POT_primal = self.mean_OT_primal+helper.hardstep((self.epoch-float(starttime))/float(timescale))*self.config['enc_reg_strength']*self.enc_reg_cost
        # else:
        #     self.mean_POT_primal = self.mean_OT_primal+self.config['enc_reg_strength']*self.enc_reg_cost
        # self.enc_cost = self.mean_POT_primal
        self.enc_cost = self.mean_OT_primal

        # ### Critic
        # # self.cri_cost = helper.compute_MMD(self.pre_prior_latent_code, self.prior_latent_code)
        # if self.config['divergence_mode'] == 'NS-GAN': 
        #     self.cri_cost = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.div_prior)+10e-7))+self.config['enc_reg_strength']*helper.compute_MMD(self.pre_prior_latent_code, self.prior_latent_code)
        # elif self.config['divergence_mode'] == 'GAN': 
        #     self.cri_cost = tf.reduce_mean(tf.log(1-tf.nn.sigmoid(self.div_prior)+10e-7))+self.config['enc_reg_strength']*helper.compute_MMD(self.pre_prior_latent_code, self.prior_latent_code)
        # elif self.config['divergence_mode'] == 'WGAN-GP': 
        #     self.cri_cost = -tf.reduce_mean(self.div_prior)+self.config['enc_reg_strength']*helper.compute_MMD(self.pre_prior_latent_code, self.prior_latent_code)

        ### Generator
        self.gen_cost = self.mean_OT_primal




