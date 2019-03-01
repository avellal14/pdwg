from __future__ import print_function

import os
import re
import pdb
import time
import math
import numpy as np
import argparse
import copy

import models.WGANGP.ModelMapsDCGAN as ModelMaps

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
        self.Decoder = ModelMaps.Decoder({**self.config, 'data_properties': batch['observed']['properties']})        
        self.Encoder = ModelMaps.Encoder({**self.config, 'data_properties': batch['observed']['properties']})        
        self.EncodingPlan = ModelMaps.EncodingPlan({**self.config, 'data_properties': batch['observed']['properties']})        
        self.PriorMap = ModelMaps.PriorMap(self.config)
        self.EpsilonMap = ModelMaps.EpsilonMap(self.config)
        self.bModules = True

    def euclidean_distance_squared(self, x, y, axis=[-1], keep_dims=True):
        return tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims)

    def metric_distance_sq(self, x, y):
        try: metric_distance = self.euclidean_distance_squared(x['image'], y['image'], axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis]
        except: metric_distance = self.euclidean_distance_squared(x['flat'], y['flat'], axis=[-1], keep_dims=True)
        return metric_distance

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
        try: batch_size_tf = tf.shape(self.input_sample['flat'])[0]
        except: batch_size_tf = tf.shape(self.input_sample['image'])[0]
        
        self.gen_prior_param = self.PriorMap.forward((tf.zeros(shape=(batch_size_tf, 1)),))
        self.gen_prior_dist = distributions.DiagonalGaussianDistribution(params = self.gen_prior_param)
        self.gen_prior_latent_code = self.gen_prior_dist.sample()
        self.gen_neg_ent_prior = self.prior_dist.log_pdf(self.gen_prior_latent_code)
        self.gen_mean_neg_ent_prior = tf.reduce_mean(self.gen_neg_ent_prior)

        self.gen_prior_latent_code_expanded = tf.reshape(self.gen_prior_latent_code, [-1, 1, *self.gen_prior_latent_code.get_shape().as_list()[1:]])
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
        try: batch_size_tf = tf.shape(self.input_sample['flat'])[0]
        except: batch_size_tf = tf.shape(self.input_sample['image'])[0]
        
        self.prior_param = self.PriorMap.forward((tf.zeros(shape=(batch_size_tf, 1)),))
        self.prior_dist = distributions.DiagonalGaussianDistribution(params = self.prior_param)
        self.prior_latent_code = self.prior_dist.sample()
        self.neg_ent_prior = self.prior_dist.log_pdf(self.prior_latent_code)
        self.mean_neg_ent_prior = tf.reduce_mean(self.neg_ent_prior)

        self.uniform_dist = distributions.UniformDistribution(params = tf.concat([tf.zeros(shape=(batch_size_tf, 1)), tf.ones(shape=(batch_size_tf, 1))], axis=1))
        self.convex_mix = self.uniform_dist.sample()

        self.prior_latent_code_expanded = tf.reshape(self.prior_latent_code, [-1, 1, *self.prior_latent_code.get_shape().as_list()[1:]])
        self.obs_sample_param = self.Decoder.forward(self.prior_latent_code_expanded)
        self.obs_sample_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.obs_sample_param)
        self.obs_sample = self.obs_sample_dist.sample(b_mode=True)

        #############################################################################
        b_use_reconst_as_reg = True
        self.posterior_param_expanded = self.Encoder.forward(self.input_sample)
        self.posterior_param = self.posterior_param_expanded[:,0,:]
        self.posterior_dist = distributions.DiagonalGaussianDistribution(params = self.posterior_param)
        self.posterior_latent_code = self.posterior_dist.sample()
        self.posterior_latent_code_expanded = self.posterior_latent_code[:,np.newaxis,:]

        self.reconst_param = self.Decoder.forward(self.posterior_latent_code_expanded) 
        self.reconst_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.reconst_param)
        self.reconst_sample = self.reconst_dist.sample(b_mode=True)

        if b_use_reconst_as_reg:
            self.reg_target_dist = self.reconst_dist
            self.reg_target_sample = self.reconst_sample
        else:
            self.reg_target_dist = self.obs_sample_dist
            self.reg_target_sample = self.obs_sample

        self.neg_cross_ent_posterior = self.prior_dist.log_pdf(self.posterior_latent_code)
        self.mean_neg_cross_ent_posterior = tf.reduce_mean(self.neg_cross_ent_posterior)
        self.neg_ent_posterior = self.posterior_dist.log_pdf(self.posterior_latent_code)
        self.mean_neg_ent_posterior = tf.reduce_mean(self.neg_ent_posterior)

        # self.kl_posterior_prior = distributions.KLDivDiagGaussianVsDiagGaussian().forward(self.posterior_dist, self.prior_dist)
        # self.mean_kl_posterior_prior = tf.reduce_mean(self.kl_posterior_prior) 

        self.mean_kl_posterior_prior = -self.mean_neg_cross_ent_posterior+self.mean_neg_ent_posterior
        #############################################################################

        self.reg_sample_param = {'image': None, 'flat': None}
        try: 
            self.reg_sample_param['image'] = self.convex_mix[:, np.newaxis, :, np.newaxis, np.newaxis]*self.reg_target_sample['image']+\
                                            (1-self.convex_mix[:, np.newaxis, :, np.newaxis, np.newaxis])*self.input_sample['image']
        except: 
            self.reg_sample_param['flat'] = self.convex_mix[:, np.newaxis, :]*self.reg_target_sample['flat']+\
                                            (1-self.convex_mix[:, np.newaxis, :])*self.input_sample['flat']
        self.reg_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = self.reg_sample_param)
        self.reg_sample = self.reg_dist.sample(b_mode=True)

        self.critic_real = self.Discriminator.forward(self.input_sample)
        self.critic_gen = self.Discriminator.forward(self.obs_sample)
        self.critic_reg = self.Discriminator.forward(self.reg_sample)
        self.critic_reconst = self.Discriminator.forward(self.reconst_sample)

        self.critic_slopes = -self.critic_reconst

        lambda_t = 1
        self.real_reconst_distances_sq = self.metric_distance_sq(self.input_sample, self.reconst_sample)
        self.autoencode_costs = self.real_reconst_distances_sq/2+lambda_t*self.mean_kl_posterior_prior

        try:
            self.convex_grad = tf.gradients(self.critic_reg, [self.reg_sample['image']])[0]
            self.convex_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.convex_grad**2, axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis])
        except: 
            self.convex_grad = tf.gradients(self.critic_reg, [self.reg_sample['flat']])[0]
            self.convex_grad_norm = helper.safe_tf_sqrt(tf.reduce_sum(self.convex_grad**2, axis=[-1], keep_dims=True))      
        self.gradient_penalties = ((self.convex_grad_norm-1)**2)

        self.mean_critic_real = tf.reduce_mean(self.critic_real)
        self.mean_critic_gen = tf.reduce_mean(self.critic_gen)
        self.mean_critic_reg = tf.reduce_mean(self.critic_reg)
        self.mean_autoencode_cost = tf.reduce_mean(self.autoencode_costs)
        self.mean_gradient_penalty = tf.reduce_mean(self.gradient_penalties)

        self.regularizer_cost = 10*self.mean_gradient_penalty
        self.discriminator_cost = -self.mean_critic_real+self.mean_critic_gen+self.regularizer_cost
        self.generator_cost = -self.mean_critic_gen
        self.transporter_cost = self.mean_autoencode_cost


    












        # # #################################################################################
        # self.reg_sample_param = {'image': None, 'flat': None}
        # try: 
        #     self.reg_sample_param['image'] = self.convex_mix[:, np.newaxis, :, np.newaxis, np.newaxis]*self.reg_target_sample['image']+\
        #                                     (1-self.convex_mix[:, np.newaxis, :, np.newaxis, np.newaxis])*self.input_sample['image']
        # except: 
        #     self.reg_sample_param['flat'] = self.convex_mix[:, np.newaxis, :]*self.reg_target_sample['flat']+\
        #                                     (1-self.convex_mix[:, np.newaxis, :])*self.input_sample['flat']
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
        #     self.reg_sample_param['image'] = self.convex_mix[:, np.newaxis, :, np.newaxis, np.newaxis]*self.reg_target_sample['image']+\
        #                                     (1-self.convex_mix[:, np.newaxis, :, np.newaxis, np.newaxis])*self.input_sample['image']
        # except: 
        #     self.reg_sample_param['flat'] = self.convex_mix[:, np.newaxis, :]*self.reg_target_sample['flat']+\
        #                                     (1-self.convex_mix[:, np.newaxis, :])*self.input_sample['flat']
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
        # obs_param = self.Decoder.forward(dec_input)  

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

    #     obs_param = self.Decoder.forward(dec_input)
    #     # obs_param_aux = self.Decoder.forward(dec_input_aux) 

    #     self.obs_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = obs_param)
    #     obs_sample = self.obs_dist.sample(b_mode=True)
    #     self.obs_sample = obs_sample

    #     # obs_dist_aux = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = obs_param)
    #     # obs_sample_aux = obs_dist_aux.sample(b_mode=True)

    #     # ###############################################################################
    #     # dec_rec_input = self.EncodingPlan.forward(batch['observed']['data'])
    #     # rec_param = self.Decoder.forward(dec_rec_input) 
    #     # self.rec_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = rec_param)
    #     # rec_sample = self.rec_dist.sample(b_mode=True)

    #     # transport_param, convex_mask = rec_param, uniform_w
    #     # self.transport_dist = distributions.ProductDistribution(sample_properties = batch['observed']['properties'], params = transport_param)
    #     # transport_sample = self.transport_dist.sample(b_mode=True)
    #     # ##############################################################################
    #     # bb_var_inference_noise = self.prior_dist_list[0].sample()[:, :50]
    #     # # bb_var_inference_noise = None
    #     # dec_rec_input = self.EncodingPlan.forward(batch['observed']['data'], noise=bb_var_inference_noise)
    #     # rec_param = self.Decoder.forward(dec_rec_input)
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

