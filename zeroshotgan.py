import sys


import sys

import sys
import inspect
import traceback

import os
import pdb
import time
import shutil
import argparse

# # # MNIST
# parser = argparse.ArgumentParser(description='Tensorflow Gan Models')
# parser.add_argument('--global_exp_dir', type=str, default='./experimentsDAEGAN40', help='Directory to put the experiments.')
# parser.add_argument('--exp_dir_postfix', type=str, default='', help='Directory to put the experiment postfix.')
# parser.add_argument('--dataset_dir', type=str, default='../dataset/dataset_both2/scripted/*.npz', help='Directory of data.')
# parser.add_argument('--restore_dir', type=str, default='/7cb1611a0a584269a019dc8342aee213/checkpoint/', help='Directory of restore experiment.')
# parser.add_argument('--restore', type=bool, default=False, help='Restore model.')

# parser.add_argument('--analyticKL', type=bool, default=True, help='Type of KL divergence to use.')
# parser.add_argument('--transformedQ', type=bool, default=False, help='Use posterior Transform.')
# parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train.')
# parser.add_argument('--batch_size', type=int, default=50, help='Input batch size for training.')
# parser.add_argument('--time_steps', type=int, default=1, help='Number of timesteps')
# parser.add_argument('--hierarchy_rate', type=int, default=1, help='Number of timesteps')

# parser.add_argument('--optimizer_class', type=str, default='Adam', help='Optimizer type.')
# parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial momentum.')
# parser.add_argument('--weight_decay', type=float, default=0, help='Initial weight decay.')
# parser.add_argument('--initial_temp', type=float, default=1, help='Initial temperature for KL divergence.')
# parser.add_argument('--max_step_temp', type=float, default=15000, help='Starting step for temp=1.')

# parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, help='random seed')
# parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')

# parser.add_argument('--n_encoder', type=int, default=400, help='n_encoder.')
# parser.add_argument('--n_decoder', type=int, default=400, help='n_decoder.')
# parser.add_argument('--n_context', type=int, default=1, help='n_context.')
# parser.add_argument('--n_state', type=int, default=1, help='n_state.')
# parser.add_argument('--n_latent', type=int, default=100, help='n_latent.')
# global_args = parser.parse_args()
# global_args.curr_epoch = 1

# from datasetLoaders.MnistLoader import DataLoader
# data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)

# # # # #############################################################################################################################

# # # # TOY
# parser = argparse.ArgumentParser(description='Tensorflow Gan Models')
# parser.add_argument('--global_exp_dir', type=str, default='./experimentsGanTests25', help='Directory to put the experiments.')
# parser.add_argument('--exp_dir_postfix', type=str, default='', help='Directory to put the experiment postfix.')
# parser.add_argument('--dataset_dir', type=str, default='../dataset/dataset_both2/scripted/*.npz', help='Directory of data.')
# parser.add_argument('--restore_dir', type=str, default='/7cb1611a0a584269a019dc8342aee213/checkpoint/', help='Directory of restore experiment.')
# parser.add_argument('--restore', type=bool, default=False, help='Restore model.')

# parser.add_argument('--analyticKL', type=bool, default=True, help='Type of KL divergence to use.')
# parser.add_argument('--transformedQ', type=bool, default=True, help='Use posterior Transform.')
# parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train.')
# parser.add_argument('--batch_size', type=int, default=50, help='Input batch size for training.')
# parser.add_argument('--time_steps', type=int, default=1, help='Number of timesteps')
# parser.add_argument('--hierarchy_rate', type=int, default=1, help='Number of timesteps')

# parser.add_argument('--optimizer_class', type=str, default='Adam', help='Optimizer type.')
# parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial momentum.')
# parser.add_argument('--weight_decay', type=float, default=0, help='Initial weight decay.')
# parser.add_argument('--initial_temp', type=float, default=0.1, help='Initial temperature for KL divergence.')
# parser.add_argument('--max_step_temp', type=float, default=15000, help='Starting step for temp=1.')

# parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, help='random seed')
# parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')

# parser.add_argument('--n_encoder', type=int, default=100, help='n_encoder.')
# parser.add_argument('--n_decoder', type=int, default=100, help='n_decoder.')
# parser.add_argument('--n_context', type=int, default=1, help='n_context.')
# parser.add_argument('--n_state', type=int, default=1, help='n_state.')
# parser.add_argument('--n_latent', type=int, default=2, help='n_latent.')
# global_args = parser.parse_args()
# global_args.curr_epoch = 1

# # from datasetLoaders.ToyDataLoader import DataLoader
# # data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)
# from datasetLoaders.RandomManifoldDataLoader import DataLoader
# data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)

##############################################################################################################################

# # CNN FEATURES
parser = argparse.ArgumentParser(description='Tensorflow Gan Models')
parser.add_argument('--global_exp_dir', type=str, default='./experimentsZERO', help='Directory to put the experiments.')
parser.add_argument('--exp_dir_postfix', type=str, default='', help='Directory to put the experiment postfix.')
parser.add_argument('--dataset_dir', type=str, default='../dataset/dataset_both2/scripted/*.npz', help='Directory of data.')
parser.add_argument('--restore_dir', type=str, default='/7cb1611a0a584269a019dc8342aee213/checkpoint/', help='Directory of restore experiment.')
parser.add_argument('--restore', type=bool, default=False, help='Restore model.')

parser.add_argument('--analyticKL', type=bool, default=True, help='Type of KL divergence to use.')
parser.add_argument('--transformedQ', type=bool, default=False, help='Use posterior Transform.')
parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=30, help='Input batch size for training.')
parser.add_argument('--time_steps', type=int, default=1, help='Number of timesteps')
parser.add_argument('--hierarchy_rate', type=int, default=1, help='Number of timesteps')

parser.add_argument('--optimizer_class', type=str, default='Adam', help='Optimizer type.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial momentum.')
parser.add_argument('--weight_decay', type=float, default=0, help='Initial weight decay.')
parser.add_argument('--initial_temp', type=float, default=1, help='Initial temperature for KL divergence.')
parser.add_argument('--max_step_temp', type=float, default=15000, help='Starting step for temp=1.')

parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')

parser.add_argument('--n_encoder', type=int, default=400, help='n_encoder.')
parser.add_argument('--n_decoder', type=int, default=400, help='n_decoder.')
parser.add_argument('--n_context', type=int, default=1, help='n_context.')
parser.add_argument('--n_state', type=int, default=1, help='n_state.')
parser.add_argument('--n_latent', type=int, default=200, help='n_latent.')
global_args = parser.parse_args()
global_args.curr_epoch = 1

from datasetLoaders.FeatureAttributeLoader import DataLoader
data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)

from models.ZEROSHOTGAN.Model import Model
import distributions 
import helper

import random
import numpy as np
import tensorflow as tf
random.seed(global_args.seed)
np.random.seed(global_args.seed)
tf.set_random_seed(global_args.seed)

global_args.exp_dir = helper.get_exp_dir(global_args)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("os.environ['CUDA_VISIBLE_DEVICES'], ", os.environ['CUDA_VISIBLE_DEVICES'])
print("TENSORBOARD: Linux:\npython -m tensorflow.tensorboard --logdir=model1:"+\
    os.path.realpath(global_args.exp_dir)+" --port="+str(20000+int(global_args.exp_dir[-4:-1], 16))+" &")
print("TENSORBOARD: Mac:\nhttp://0.0.0.0:"+str(20000+int(global_args.exp_dir[-4:-1], 16)))
print("\n\n\n")

# shutil.copyfile('./models/SLVM.py', global_args.exp_dir+'SLVM.py')
# shutil.copyfile('./models/ModelGTM.py', global_args.exp_dir+'ModelGTM.py')

_, _, batch = next(data_loader)
with tf.Graph().as_default():
    tf.set_random_seed(global_args.seed)
    model = Model(vars(global_args))

    global_step = tf.Variable(0.0, name='global_step', trainable=False)
    with tf.variable_scope("training"):
        tf.set_random_seed(global_args.seed)
        
        additional_inputs_tf = tf.placeholder(tf.float32, [1])
        batch_tf, input_dict_func = helper.tf_batch_and_input_dict(batch, additional_inputs_tf)
        train_outs_dict, test_outs_dict = model.inference(batch_tf, additional_inputs_tf)
        generative_dict = model.generative_model(batch_tf)
        inference_obs_dist = model.obs_dist

        discriminator_vars = [v for v in tf.trainable_variables() if 'Discriminator' in v.name]
        generator_vars = [v for v in tf.trainable_variables() if 'Discriminator' not in v.name] 

        # Weight clipping
        discriminator_vars_flat_concat = tf.concat([tf.reshape(e, [-1]) for e in discriminator_vars], axis=0)
        max_abs_discriminator_vars = tf.reduce_max(tf.abs(discriminator_vars_flat_concat))
        clip_op_list = []
        for e in discriminator_vars:
            clip_op_list.append(tf.assign(e, tf.clip_by_value(e, -0.01, 0.01)))

    if global_args.optimizer_class == 'RmsProp':
        train_generator_step_tf = tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, 
            momentum=0.9).minimize(train_outs_dict['generator_cost'], var_list=generator_vars, global_step=global_step)
        train_discriminator_step_tf = tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, 
            momentum=0.9).minimize(train_outs_dict['discriminator_cost'], var_list=discriminator_vars, global_step=global_step)
    elif global_args.optimizer_class == 'Adam':
        train_generator_step_tf = tf.train.AdamOptimizer(learning_rate=0.0001, 
            beta1=0.5, beta2=0.999, epsilon=1e-08).minimize(train_outs_dict['generator_cost'], var_list=generator_vars, global_step=global_step)
        train_discriminator_step_tf = tf.train.AdamOptimizer(learning_rate=0.0001, 
            beta1=0.5, beta2=0.999, epsilon=1e-08).minimize(train_outs_dict['discriminator_cost'], var_list=discriminator_vars, global_step=global_step)

    helper.variable_summaries(train_outs_dict['generator_cost'], '/generator_cost')
    helper.variable_summaries(train_outs_dict['discriminator_cost'], '/discriminator_cost')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    merged_summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(global_args.exp_dir+'/summaries', sess.graph)
    sess.run(init)

    if global_args.restore:
        print("=> Loading checkpoint: '{}'".format(global_args.global_exp_dir+global_args.restore_dir))
        try: 
            helper.load_checkpoint(saver, sess, global_args.global_exp_dir+global_args.restore_dir)  
            print("=> Loaded checkpoint: '{}'".format(global_args.global_exp_dir+global_args.restore_dir))
        except: print("=> FAILED to load checkpoint: '{}'".format(global_args.global_exp_dir+global_args.restore_dir))

    def train(epoch):
        data_loader.train()
        train_gen_loss_accum, train_dis_loss_accum, train_likelihood_accum, train_kl_accum, batch_size_accum = 0, 0, 0, 0, 0
        start = time.time();
        for batch_idx, curr_batch_size, batch in data_loader: 

            disc_train_step_np = sess.run([train_discriminator_step_tf], feed_dict = input_dict_func(batch, np.asarray([0,])))
            if batch_idx % 5 !=0: continue
            gen_train_step_np, generator_cost_np, discriminator_cost_np = \
                sess.run([train_generator_step_tf, train_outs_dict['generator_cost'], train_outs_dict['discriminator_cost']],
                          feed_dict = input_dict_func(batch, np.asarray([0,])))

            max_discriminator_weight = sess.run(max_abs_discriminator_vars)
            train_gen_loss_accum += curr_batch_size*generator_cost_np
            train_dis_loss_accum += curr_batch_size*discriminator_cost_np
            batch_size_accum += curr_batch_size

            if batch_idx % global_args.log_interval == 0:
                end = time.time();
                print('Train: Epoch {} [{:7d} ()]\tGenerator Cost: {:.6f}\tDiscriminator Cost: {:.6f}\tTime: {:.3f}, Max disc weight {:.6f}'.format(
                      epoch, batch_idx * curr_batch_size, generator_cost_np, discriminator_cost_np, (end - start), max_discriminator_weight))

                with open(global_args.exp_dir+"training_traces.txt", "a") as text_file:
                    text_file.write(str(generator_cost_np) + ', ' + str(discriminator_cost_np) + '\n')
                start = time.time()
    
        summary_str = sess.run(merged_summaries, feed_dict = input_dict_func(batch, np.asarray([0,])))
        summary_writer.add_summary(summary_str, (tf.train.global_step(sess, global_step)))
        
        checkpoint_time = 1
        if data_loader.__module__ == 'datasetLoaders.RandomManifoldDataLoader' or data_loader.__module__ == 'datasetLoaders.ToyDataLoader':
            checkpoint_time = 20

        if epoch % checkpoint_time == 0:
            print('====> Average Train: Epoch: {}\tGenerator Cost: {:.6f}\tDiscriminator Cost: {:.6f}'.format(
                  epoch, train_gen_loss_accum/batch_size_accum, train_dis_loss_accum/batch_size_accum))

            # helper.draw_bar_plot(rate_similarity_gen_np[:,0,0], y_min_max = [0,1], save_dir=global_args.exp_dir+'Visualization/inversion_weight/', postfix='inversion_weight'+str(epoch))
            # helper.draw_bar_plot(effective_z_cost_np[:,0,0], thres = [np.mean(effective_z_cost_np), np.max(effective_z_cost_np)], save_dir=global_args.exp_dir+'Visualization/inversion_cost/', postfix='inversion_cost'+str(epoch))
            # helper.draw_bar_plot(disc_cost_gen_np[:,0,0], thres = [0, 0], save_dir=global_args.exp_dir+'Visualization/disc_cost/', postfix='disc_cost'+str(epoch))
            
            # if data_loader.__module__ == 'datasetLoaders.RandomManifoldDataLoader' or data_loader.__module__ == 'datasetLoaders.ToyDataLoader':
            #     helper.visualize_datasets(sess, input_dict_func(batch), data_loader.dataset, generative_dict['obs_sample_out'], generative_dict['latent_sample_out'],
            #         save_dir=global_args.exp_dir+'Visualization/', postfix=str(epoch)) 
                
            #     xmin, xmax, ymin, ymax, X_dense, Y_dense = -3.5, 3.5, -3.5, 3.5, 250, 250
            #     xlist = np.linspace(xmin, xmax, X_dense)
            #     ylist = np.linspace(ymin, ymax, Y_dense)
            #     X, Y = np.meshgrid(xlist, ylist)
            #     XY = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)

            #     batch['observed']['data']['flat'] = XY[:, np.newaxis, :]
            #     disc_cost_real_np = sess.run(train_outs_dict['critic_real'], feed_dict = input_dict_func(batch, np.asarray([0,])))

            #     f = np.reshape(disc_cost_real_np[:,0,0], [Y_dense, X_dense])
            #     helper.plot_ffs(X, Y, f, save_dir=global_args.exp_dir+'Visualization/discriminator_function/', postfix='discriminator_function'+str(epoch))
            # else:
            #     distributions.visualizeProductDistribution(sess, input_dict_func(batch), batch, inference_obs_dist, generative_dict['obs_dist'], 
            #     save_dir=global_args.exp_dir+'Visualization/Train/', postfix='train_'+str(epoch))

            checkpoint_path1 = global_args.exp_dir+'checkpoint/'
            checkpoint_path2 = global_args.exp_dir+'checkpoint2/'
            print('====> Saving checkpoint. Epoch: ', epoch); start_tmp = time.time()
            helper.save_checkpoint(saver, sess, global_step, checkpoint_path1) 
            end_tmp = time.time(); print('Checkpoint path: '+checkpoint_path1+'   ====> It took: ', end_tmp - start_tmp)
            if epoch % 60 == 0: 
                print('====> Saving checkpoint backup. Epoch: ', epoch); start_tmp = time.time()
                helper.save_checkpoint(saver, sess, global_step, checkpoint_path2) 
                end_tmp = time.time(); print('Checkpoint path: '+checkpoint_path2+'   ====> It took: ', end_tmp - start_tmp)

    def test(epoch):
        data_loader.eval()
        test_gen_loss_accum, test_dis_loss_accum, test_likelihood_accum, test_kl_accum, batch_size_accum = 0, 0, 0, 0, 0
        start = time.time()

        for batch_idx, curr_batch_size, batch in data_loader: 
            test_generator_cost_np, test_discriminator_cost_np = sess.run([test_outs_dict['generator_cost'], test_outs_dict['discriminator_cost']], 
                feed_dict = input_dict_func(batch, np.asarray([0,])))            

            test_gen_loss_accum += curr_batch_size*test_generator_cost_np
            test_dis_loss_accum += curr_batch_size*test_discriminator_cost_np
            batch_size_accum += curr_batch_size

        end = time.time();
        print('====> Average Test: Epoch {}\tGenerator Cost: {:.6f}\tDiscriminator Cost: {:.6f}\tTime: {:.3f}'.format(
              epoch, test_gen_loss_accum/batch_size_accum, test_dis_loss_accum/batch_size_accum, (end - start)))

        with open(global_args.exp_dir+"test_traces.txt", "a") as text_file:
            text_file.write(str(test_gen_loss_accum/batch_size_accum) + ', ' + str(test_dis_loss_accum/batch_size_accum) + '\n')

        # distributions.visualizeProductDistribution(sess, input_dict_func(batch), batch, inference_obs_dist, generative_dict['obs_dist'], 
        #     save_dir = global_args.exp_dir+'Visualization/Test/', postfix='test_'+str(epoch))


    print('Starting training.')
    while global_args.curr_epoch < global_args.epochs + 1:
        train(global_args.curr_epoch)
        if global_args.curr_epoch % 20 == 0: 
            test(global_args.curr_epoch)
        global_args.curr_epoch += 1
       








            






            