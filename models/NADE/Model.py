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

import models.NADE.ModelMaps as ModelMaps

import helper
import distributions 
import transforms
import tensorflow as tf


class Model():
    def __init__(self, global_args, name = 'Model'):
        self.name = name
        self.bModules = False 
        self.fixedContext = False
        self.config = global_args

        if self.config['sample_distance_mode'] == 'Euclidean': self.sample_distance_function = helper.euclidean_distance
        if self.config['sample_distance_mode'] == 'Quadratic': self.sample_distance_function = helper.quadratic_distance
        
    def generate_modules(self, batch):
        self.PriorMap = ModelMaps.PriorMapGaussian(self.config)
        self.EpsilonMap = ModelMaps.PriorMapGaussian(self.config)    
        self.Encoder = ModelMaps.Encoder({**self.config, 'data_properties': batch['observed']['properties']})        
        self.Generator = ModelMaps.Generator({**self.config, 'data_properties': batch['observed']['properties']})        
        self.bModules = True

    def generative_model(self, batch, additional_inputs_tf):
        self.gen_epoch = additional_inputs_tf[0]
        self.gen_b_identity = additional_inputs_tf[1]
        
        empirical_observed_properties = copy.deepcopy(batch['observed']['properties'])
        for e in empirical_observed_properties['flat']: e['dist']='dirac'
        for e in empirical_observed_properties['image']: e['dist']='dirac'

        self.gen_input_sample = batch['observed']['data']
        self.gen_input_dist = distributions.ProductDistribution(sample_properties = empirical_observed_properties, params = self.gen_input_sample)

        try: self.n_time = batch['observed']['properties']['flat'][0]['size'][1]
        except: self.n_time = batch['observed']['properties']['image'][0]['size'][1]
        try: self.gen_batch_size_tf = tf.shape(self.input_sample['flat'])[0]
        except: self.gen_batch_size_tf = tf.shape(self.input_sample['image'])[0]
        
        self.gen_prior_param = self.PriorMap.forward((tf.zeros(shape=(self.gen_batch_size_tf, 1)),))
        self.gen_prior_dist = distributions.DiagonalGaussianDistribution(params = self.gen_prior_param)
        self.gen_prior_latent_code = self.gen_prior_dist.sample()

        self.gen_obs_sample_param = self.Generator.forward(self.gen_prior_latent_code[:,np.newaxis,:])
        self.gen_obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.gen_obs_sample_param)
        self.gen_obs_sample = self.gen_obs_sample_dist.sample()

    def inference(self, batch, additional_inputs_tf):
        self.epoch = additional_inputs_tf[0]
        self.b_identity = additional_inputs_tf[1]
        
        empirical_observed_properties = copy.deepcopy(batch['observed']['properties'])
        for e in empirical_observed_properties['flat']: e['dist']='dirac'
        for e in empirical_observed_properties['image']: e['dist']='dirac'

        self.input_sample = batch['observed']['data']        
        self.input_dist = distributions.ProductDistribution(sample_properties = empirical_observed_properties, params = self.input_sample)

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

        self.obs_sample_param = self.Generator.forward(self.prior_latent_code[:, np.newaxis, :])
        self.obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.obs_sample_param)
        self.obs_sample = self.obs_sample_dist.sample()

        self.obs_log_pdf = self.obs_sample_dist.log_pdf(self.input_sample)

        if not os.path.exists(str(Path.home())+'/ExperimentalResults/FixedSamples/'): os.makedirs(str(Path.home())+'/ExperimentalResults/FixedSamples/')
        if os.path.exists(str(Path.home())+'/ExperimentalResults/FixedSamples/np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz'): 
            np_constant_prior_sample = np.load(str(Path.home())+'/ExperimentalResults/FixedSamples/np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz')
        else:
            np_constant_prior_sample = np.random.normal(loc=0., scale=1., size=[400, self.prior_latent_code.get_shape().as_list()[-1]])
            np.save(str(Path.home())+'/ExperimentalResults/FixedSamples/np_constant_prior_sample_'+str(self.prior_latent_code.get_shape().as_list()[-1])+'.npz', np_constant_prior_sample)    
        
        self.constant_prior_latent_code = tf.constant(np.asarray(np_constant_prior_sample), dtype=np.float32)
        self.constant_obs_sample_param = self.Generator.forward(self.constant_prior_latent_code[:, np.newaxis, :])
        self.constant_obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.constant_obs_sample_param)
        self.constant_obs_sample = self.constant_obs_sample_dist.sample()
                
        if self.config['n_latent'] == 2: 
            grid_scale = 3
            x = np.linspace(-grid_scale, grid_scale, 20)
            y = np.linspace(grid_scale, -grid_scale, 20)
            xv, yv = np.meshgrid(x, y)
            np_constant_prior_grid_sample = np.concatenate((xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis][:]), axis=1)
        
            self.constant_prior_grid_latent_code = tf.constant(np.asarray(np_constant_prior_grid_sample), dtype=np.float32)
            self.constant_obs_grid_sample_param = self.Generator.forward(self.constant_prior_grid_latent_code[:, np.newaxis, :])
            self.constant_obs_grid_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.constant_obs_grid_sample_param)
            self.constant_obs_grid_sample = self.constant_obs_grid_sample_dist.sample()
                
        #############################################################################
        # ENCODER 

        if self.config['encoder_mode'] == 'Gaussian': 
            self.epsilon_param = self.EpsilonMap.forward((tf.zeros(shape=(self.batch_size_tf, 1)),))
            self.epsilon_dist = distributions.DiagonalGaussianDistribution(params = self.epsilon_param)        
            self.epsilon = self.epsilon_dist.sample()
        else:
            self.epsilon = None

        self.posterior_latent_code_expanded = self.Encoder.forward(self.input_sample, noise=self.epsilon)
        self.posterior_latent_code = self.posterior_latent_code_expanded[:,0,:]

        self.reconst_param = self.Generator.forward(self.posterior_latent_code_expanded) 
        self.reconst_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.reconst_param)
        self.reconst_sample = self.reconst_dist.sample(b_mode=True)
        self.reconst_log_pdf = self.reconst_dist.log_pdf(self.input_sample)

        self.interpolated_posterior_latent_code = helper.interpolate_latent_codes(self.posterior_latent_code, size=self.batch_size_tf//2)
        self.interpolated_obs = self.Generator.forward(self.interpolated_posterior_latent_code) 

        #############################################################################
        # REGULARIZER

        self.reg_target_dist = self.reconst_dist
        self.reg_target_sample = self.reconst_sample
        self.reg_dist = self.reconst_dist
        self.reg_sample = self.reconst_sample
        
        #############################################################################
        # OBJECTIVES

        ## Encoder
        self.mean_neg_log_pdf = -tf.reduce_mean(self.reconst_log_pdf)

        self.OT_primal = self.sample_distance_function(self.input_sample, self.reconst_sample)
        self.mean_OT_primal = tf.reduce_mean(self.OT_primal)

        # overall_cost = self.mean_neg_log_pdf
        timescale, start_time, min_tradeoff = 5, 10, 0.000001
        tradeoff = (1-2*min_tradeoff)*helper.hardstep((self.epoch-float(start_time))/float(timescale))+min_tradeoff
        overall_cost = tradeoff*self.mean_neg_log_pdf+(1-tradeoff)*self.mean_OT_primal

        self.enc_cost = overall_cost

        ### Generator
        self.gen_cost = overall_cost
































        # timescale, start_time = 5, 10
        # update_pre_std = helper.hardstep((self.epoch-float(start_time))/float(timescale))
        # temp_mean = self.reconst_param['image'][..., :int(self.reconst_param['image'].get_shape().as_list()[-1]/2.)]
        # temp_pre_std = self.reconst_param['image'][..., int(self.reconst_param['image'].get_shape().as_list()[-1]/2.):]
        # temp_pre_std = (1-update_pre_std)*(4)*tf.ones(tf.shape(temp_pre_std), tf.float32)+update_pre_std*temp_pre_std
        # self.reconst_param['image'] = tf.concat([temp_mean, temp_pre_std], axis=-1)












