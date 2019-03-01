import sys

import inspect
import traceback

import os
import pdb
import time
import shutil
import argparse
from sklearn.manifold import TSNE

# # # # FLOWERS
# parser = argparse.ArgumentParser(description='Tensorflow Gan Models')
# parser.add_argument('--global_exp_dir', type=str, default='./experimentsFLOWERSVAE', help='Directory to put the experiments.')
# parser.add_argument('--exp_dir_postfix', type=str, default='', help='Directory to put the experiment postfix.')
# parser.add_argument('--dataset_dir', type=str, default='../dataset/dataset_both2/scripted/*.npz', help='Directory of data.')
# parser.add_argument('--restore_dir', type=str, default='/7cb1611a0a584269a019dc8342aee213/checkpoint/', help='Directory of restore experiment.')
# parser.add_argument('--restore', type=bool, default=False, help='Restore model.')

# parser.add_argument('--analyticKL', type=bool, default=True, help='Type of KL divergence to use.')
# parser.add_argument('--transformedQ', type=bool, default=False, help='Use posterior Transform.')
# parser.add_argument('--epochs', type=int, default=1000000000, help='Number of epochs to train.')
# parser.add_argument('--batch_size', type=int, default=50, help='Input batch size for training.')
# parser.add_argument('--time_steps', type=int, default=1, help='Number of timesteps')
# parser.add_argument('--hierarchy_rate', type=int, default=1, help='Number of timesteps')

# parser.add_argument('--optimizer_class', type=str, default='Adam', help='Optimizer type.')
# parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial momentum.')
# parser.add_argument('--weight_decay', type=float, default=0, help='Initial weight decay.')
# parser.add_argument('--gradient_clipping', type=float, default=1, help='Initial weight decay.')
# parser.add_argument('--initial_temp', type=float, default=1, help='Initial temperature for KL divergence.')
# parser.add_argument('--max_step_temp', type=float, default=15000, help='Starting step for temp=1.')

# parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, help='random seed')
# parser.add_argument('--log_interval', type=int, default=40, help='how many batches to wait before logging training status')
# parser.add_argument('--vis_interval', type=int, default=20, help='how many batches to wait before visualizing training status')
# parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
# parser.add_argument('--save_checkpoints', type=bool, default=False, help='store the checkpoints?')
# parser.add_argument('--test_epoch', type=int, default=60, help='test epoch repeat')
# parser.add_argument('--latent_vis_epoch_rate', type=list, default=[150,1], help='latent epoch repeat')
# parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[20,1], help='reconst epoch repeat')
# parser.add_argument('--inception_score_epoch_rate', type=list, default=[50,1], help='compute inception score')
# parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[50,1], help='compute pigeonhole score')

# parser.add_argument('--n_encoder', type=int, default=400, help='n_encoder.')
# parser.add_argument('--n_decoder', type=int, default=400, help='n_decoder.')
# parser.add_argument('--n_context', type=int, default=1, help='n_context.')
# parser.add_argument('--n_state', type=int, default=1, help='n_state.')
# parser.add_argument('--n_latent', type=int, default=50, help='n_latent.')
# global_args = parser.parse_args()
# global_args.curr_epoch = 1

# from datasetLoaders.FlowersQueueLoader import DataLoader
# data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)

# # # # #############################################################################################################################

# # # CUB
# parser = argparse.ArgumentParser(description='Tensorflow Gan Models')
# parser.add_argument('--global_exp_dir', type=str, default='./experimentsCUBVAE', help='Directory to put the experiments.')
# parser.add_argument('--exp_dir_postfix', type=str, default='', help='Directory to put the experiment postfix.')
# parser.add_argument('--dataset_dir', type=str, default='../dataset/dataset_both2/scripted/*.npz', help='Directory of data.')
# parser.add_argument('--restore_dir', type=str, default='/7cb1611a0a584269a019dc8342aee213/checkpoint/', help='Directory of restore experiment.')
# parser.add_argument('--restore', type=bool, default=False, help='Restore model.')

# parser.add_argument('--analyticKL', type=bool, default=True, help='Type of KL divergence to use.')
# parser.add_argument('--transformedQ', type=bool, default=False, help='Use posterior Transform.')
# parser.add_argument('--epochs', type=int, default=1000000000, help='Number of epochs to train.')
# parser.add_argument('--batch_size', type=int, default=50, help='Input batch size for training.')
# parser.add_argument('--time_steps', type=int, default=1, help='Number of timesteps')
# parser.add_argument('--hierarchy_rate', type=int, default=1, help='Number of timesteps')

# parser.add_argument('--optimizer_class', type=str, default='Adam', help='Optimizer type.')
# parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial momentum.')
# parser.add_argument('--weight_decay', type=float, default=0, help='Initial weight decay.')
# parser.add_argument('--gradient_clipping', type=float, default=1, help='Initial weight decay.')
# parser.add_argument('--initial_temp', type=float, default=1, help='Initial temperature for KL divergence.')
# parser.add_argument('--max_step_temp', type=float, default=15000, help='Starting step for temp=1.')

# parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, help='random seed')
# parser.add_argument('--log_interval', type=int, default=40, help='how many batches to wait before logging training status')
# parser.add_argument('--vis_interval', type=int, default=20, help='how many batches to wait before visualizing training status')
# parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
# parser.add_argument('--save_checkpoints', type=bool, default=False, help='store the checkpoints?')
# parser.add_argument('--test_epoch', type=int, default=60, help='test epoch repeat')
# parser.add_argument('--latent_vis_epoch_rate', type=list, default=[150,1], help='latent epoch repeat')
# parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[20,1], help='reconst epoch repeat')
# parser.add_argument('--inception_score_epoch_rate', type=list, default=[50,1], help='compute inception score')
# parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[50,1], help='compute pigeonhole score')

# parser.add_argument('--n_encoder', type=int, default=400, help='n_encoder.')
# parser.add_argument('--n_decoder', type=int, default=400, help='n_decoder.')
# parser.add_argument('--n_context', type=int, default=1, help='n_context.')
# parser.add_argument('--n_state', type=int, default=1, help='n_state.')
# parser.add_argument('--n_latent', type=int, default=50, help='n_latent.')
# global_args = parser.parse_args()
# global_args.curr_epoch = 1

# from datasetLoaders.CubQueueLoader import DataLoader
# data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)

# # # # #############################################################################################################################

# # # CELEB-A
# parser = argparse.ArgumentParser(description='Tensorflow Gan Models')
# parser.add_argument('--global_exp_dir', type=str, default='./experimentsCELEBVAE', help='Directory to put the experiments.')
# parser.add_argument('--exp_dir_postfix', type=str, default='', help='Directory to put the experiment postfix.')
# parser.add_argument('--dataset_dir', type=str, default='../dataset/dataset_both2/scripted/*.npz', help='Directory of data.')
# parser.add_argument('--restore_dir', type=str, default='/7cb1611a0a584269a019dc8342aee213/checkpoint/', help='Directory of restore experiment.')
# parser.add_argument('--restore', type=bool, default=False, help='Restore model.')

# parser.add_argument('--analyticKL', type=bool, default=True, help='Type of KL divergence to use.')
# parser.add_argument('--transformedQ', type=bool, default=False, help='Use posterior Transform.')
# parser.add_argument('--epochs', type=int, default=1000000000, help='Number of epochs to train.')
# parser.add_argument('--batch_size', type=int, default=50, help='Input batch size for training.')
# parser.add_argument('--time_steps', type=int, default=1, help='Number of timesteps')
# parser.add_argument('--hierarchy_rate', type=int, default=1, help='Number of timesteps')

# parser.add_argument('--optimizer_class', type=str, default='Adam', help='Optimizer type.')
# parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial momentum.')
# parser.add_argument('--weight_decay', type=float, default=0, help='Initial weight decay.')
# parser.add_argument('--gradient_clipping', type=float, default=5, help='Initial weight decay.')
# parser.add_argument('--initial_temp', type=float, default=1, help='Initial temperature for KL divergence.')
# parser.add_argument('--max_step_temp', type=float, default=15000, help='Starting step for temp=1.')

# parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, help='random seed')
# parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
# parser.add_argument('--vis_interval', type=int, default=1, help='how many batches to wait before visualizing training status')
# parser.add_argument('--in_between_vis', type=int, default=1, help='how many reports to wait before visualizing training status')
# parser.add_argument('--save_checkpoints', type=bool, default=False, help='store the checkpoints?')
# parser.add_argument('--test_epoch', type=int, default=40, help='test epoch repeat')
# parser.add_argument('--latent_vis_epoch_rate', type=list, default=[50,1], help='latent epoch repeat')
# parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[1,0], help='reconst epoch repeat')
# parser.add_argument('--inception_score_epoch_rate', type=list, default=[0,1], help='compute inception score')
# parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[0,1], help='compute pigeonhole score')

# parser.add_argument('--n_encoder', type=int, default=400, help='n_encoder.')
# parser.add_argument('--n_decoder', type=int, default=400, help='n_decoder.')
# parser.add_argument('--n_context', type=int, default=1, help='n_context.')
# parser.add_argument('--n_state', type=int, default=1, help='n_state.')
# parser.add_argument('--n_latent', type=int, default=50, help='n_latent.')
# global_args = parser.parse_args()
# global_args.curr_epoch = 1

# from datasetLoaders.CelebA1QueueLoader2 import DataLoader
# data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)

# # # # # # # #############################################################################################################################

# # MNIST
parser = argparse.ArgumentParser(description='Tensorflow Gan Models')
parser.add_argument('--global_exp_dir', type=str, default='./experimentsMNISTVAE', help='Directory to put the experiments.')
parser.add_argument('--exp_dir_postfix', type=str, default='', help='Directory to put the experiment postfix.')
parser.add_argument('--dataset_dir', type=str, default='../dataset/dataset_both2/scripted/*.npz', help='Directory of data.')
parser.add_argument('--restore_dir', type=str, default='/7cb1611a0a584269a019dc8342aee213/checkpoint/', help='Directory of restore experiment.')
parser.add_argument('--restore', type=bool, default=False, help='Restore model.')

parser.add_argument('--analyticKL', type=bool, default=True, help='Type of KL divergence to use.')
parser.add_argument('--transformedQ', type=bool, default=False, help='Use posterior Transform.')
parser.add_argument('--epochs', type=int, default=1000000000, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=50, help='Input batch size for training.')
parser.add_argument('--time_steps', type=int, default=1, help='Number of timesteps')
parser.add_argument('--hierarchy_rate', type=int, default=1, help='Number of timesteps')

parser.add_argument('--optimizer_class', type=str, default='Adam', help='Optimizer type.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial momentum.')
parser.add_argument('--weight_decay', type=float, default=0, help='Initial weight decay.')
parser.add_argument('--gradient_clipping', type=float, default=100, help='Initial weight decay.')
parser.add_argument('--initial_temp', type=float, default=1, help='Initial temperature for KL divergence.')
parser.add_argument('--max_step_temp', type=float, default=15000, help='Starting step for temp=1.')

parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--vis_interval', type=int, default=1, help='how many batches to wait before visualizing training status')
parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
parser.add_argument('--save_checkpoints', type=bool, default=False, help='store the checkpoints?')
parser.add_argument('--test_epoch', type=int, default=20, help='test epoch repeat')
parser.add_argument('--latent_vis_epoch_rate', type=list, default=[0,10], help='latent epoch repeat')
parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[1,0], help='reconst epoch repeat')
parser.add_argument('--inception_score_epoch_rate', type=list, default=[0,1], help='compute inception score')
parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[0,1], help='compute pigeonhole score')

parser.add_argument('--n_encoder', type=int, default=400, help='n_encoder.')
parser.add_argument('--n_decoder', type=int, default=400, help='n_decoder.')
parser.add_argument('--n_context', type=int, default=1, help='n_context.')
parser.add_argument('--n_state', type=int, default=1, help='n_state.')
parser.add_argument('--n_latent', type=int, default=400, help='n_latent.')
global_args = parser.parse_args()
global_args.curr_epoch = 1

# from datasetLoaders.MnistLoader import DataLoader
from datasetLoaders.ColorMnistBackLoader import DataLoader
data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)

# #############################################################################################################################


# # # # TOY
# parser = argparse.ArgumentParser(description='Tensorflow Gan Models')
# parser.add_argument('--global_exp_dir', type=str, default='./experimentsTOYVAE', help='Directory to put the experiments.')
# parser.add_argument('--exp_dir_postfix', type=str, default='', help='Directory to put the experiment postfix.')
# parser.add_argument('--dataset_dir', type=str, default='../dataset/dataset_both2/scripted/*.npz', help='Directory of data.')
# parser.add_argument('--restore_dir', type=str, default='/7cb1611a0a584269a019dc8342aee213/checkpoint/', help='Directory of restore experiment.')
# parser.add_argument('--restore', type=bool, default=False, help='Restore model.')

# parser.add_argument('--analyticKL', type=bool, default=True, help='Type of KL divergence to use.')
# parser.add_argument('--transformedQ', type=bool, default=True, help='Use posterior Transform.')
# parser.add_argument('--epochs', type=int, default=1000000000, help='Number of epochs to train.')
# parser.add_argument('--batch_size', type=int, default=100, help='Input batch size for training.')
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
# parser.add_argument('--vis_interval', type=int, default=20, help='how many batches to wait before visualizing training status')
# parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
# parser.add_argument('--save_checkpoints', type=bool, default=False, help='store the checkpoints?')
# parser.add_argument('--test_epoch', type=int, default=20, help='test epoch')
# parser.add_argument('--latent_vis_epoch_rate', type=list, default=[1,-1], help='latent epoch repeat')
# parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[1,-1], help='reconst epoch repeat')
# parser.add_argument('--inception_score_epoch_rate', type=list, default=[50,1], help='compute inception score')
# parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[50,1], help='compute pigeonhole score')

# parser.add_argument('--n_encoder', type=int, default=400, help='n_encoder.')
# parser.add_argument('--n_decoder', type=int, default=400, help='n_decoder.')
# parser.add_argument('--n_context', type=int, default=1, help='n_context.')
# parser.add_argument('--n_state', type=int, default=1, help='n_state.')
# parser.add_argument('--n_latent', type=int, default=50, help='n_latent.')
# global_args = parser.parse_args()
# global_args.curr_epoch = 1

# from datasetLoaders.ToyDataLoader import DataLoader
# data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)
# # from datasetLoaders.RandomManifoldDataLoader import DataLoader
# # data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)

# ##############################################################################################################################

from models.VAE.Model import Model

from models.EVAL.InceptionScore import InceptionScore
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
# if os.path.exists('/var/scratch/mcgemici/'): 
#     temp_dir = '/var/scratch/mcgemici/experiments/'+global_args.exp_dir
#     shutil.move(global_args.exp_dir, temp_dir) 
#     os.symlink(os.path.abspath(temp_dir), os.path.abspath(global_args.exp_dir))
#     global_args.exp_dir = temp_dir

_, _, batch = next(data_loader)
try: 
    fixed_batch_data = batch['observed']['data']['image'].copy()
    random_batch_data = batch['observed']['data']['image'].copy()
except: pass
with tf.Graph().as_default():
    tf.set_random_seed(global_args.seed)
    model = Model(vars(global_args))

    global_step = tf.Variable(0.0, name='global_step', trainable=False)
    with tf.variable_scope("training"):
        tf.set_random_seed(global_args.seed)
        
        additional_inputs_tf = tf.placeholder(tf.float32, [2])
        batch_tf, input_dict_func = helper.tf_batch_and_input_dict(batch, additional_inputs_tf)
        model.inference(batch_tf, additional_inputs_tf)
        model.generative_model(batch_tf, additional_inputs_tf)

        generator_vars = [v for v in tf.trainable_variables() if 'Decoder' in v.name or 'Encoder' in v.name] 


    # if global_args.optimizer_class == 'Adam':
    #     if data_loader.__module__ == 'datasetLoaders.RandomManifoldDataLoader' or data_loader.__module__ == 'datasetLoaders.ToyDataLoader':
    #         generator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08), 
    #                                   loss=model.generator_cost, var_list=generator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
    #     else:
    #         generator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08), 
    #                                   loss=model.generator_cost, var_list=generator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)


    if global_args.optimizer_class == 'Adam':
        if data_loader.__module__ == 'datasetLoaders.RandomManifoldDataLoader' or data_loader.__module__ == 'datasetLoaders.ToyDataLoader':
            generator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0001, momentum=0.95), 
                                      loss=model.generator_cost, var_list=generator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
        else:
            generator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0001, momentum=0.95), 
                                      loss=model.generator_cost, var_list=generator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
            
            
    helper.variable_summaries(model.generator_cost, '/generator_cost')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    merged_summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(global_args.exp_dir+'/summaries', sess.graph)
    sess.run(init)

    try:
        real_data_large_batch = None
        while real_data_large_batch is None or real_data_large_batch.shape[0]<300:
            _, _, batch = next(data_loader)
            if real_data_large_batch is None: real_data_large_batch = batch['observed']['data']['image'].copy()
            else: real_data_large_batch = np.concatenate([real_data_large_batch, batch['observed']['data']['image']], axis=0)
        helper.visualize_images2(real_data_large_batch[:int(np.sqrt(real_data_large_batch.shape[0]))**2, ...],
        block_size=[int(np.sqrt(real_data_large_batch.shape[0])), int(np.sqrt(real_data_large_batch.shape[0]))],
        save_dir=global_args.exp_dir+'Visualization/Train/', postfix='_real_sample_only')
    except: pass

    if global_args.restore:
        print("=> Loading checkpoint: '{}'".format(global_args.global_exp_dir+global_args.restore_dir))
        try: 
            helper.load_checkpoint(saver, sess, global_args.global_exp_dir+global_args.restore_dir)  
            print("=> Loaded checkpoint: '{}'".format(global_args.global_exp_dir+global_args.restore_dir))
        except: print("=> FAILED to load checkpoint: '{}'".format(global_args.global_exp_dir+global_args.restore_dir))
    
    print('Loading inception model.')
    InceptionScoreModel = InceptionScore(sess)
    print('Successfully loaded inception model.')

    def train(epoch):
        global random_batch_data, fixed_batch_data
        trans_steps, disc_steps, gen_steps = 0, 0, 0
        report_count = 0
        data_loader.train()
        train_gen_loss_accum, train_dis_loss_accum, train_likelihood_accum, train_kl_accum, batch_size_accum = 0, 0, 0, 0, 0
        start = time.time();

        hyperparam_dict = {'b_identity': 0.}
        helper.update_dict_from_file(hyperparam_dict, './hyperparam_file.py')
        
        random_idx = np.random.randint(0, high=data_loader.curr_max_iter-1)
        for batch_idx, curr_batch_size, batch in data_loader: 
            if random_idx == batch_idx: 
                try:
                    random_batch_data = batch['observed']['data']['image'].copy()
                except: pass

            hyper_param = np.asarray([epoch, hyperparam_dict['b_identity']])
            curr_feed_dict = input_dict_func(batch, hyper_param)

            np_gen_step, np_generator_cost = \
            sess.run([generator_step_tf, model.generator_cost], feed_dict = curr_feed_dict)
            gen_steps = gen_steps+1

            train_gen_loss_accum += curr_batch_size*np_generator_cost
            batch_size_accum += curr_batch_size

            if batch_idx % global_args.log_interval == 0:
                report_count = report_count+1

                np_generator_cost = sess.run(model.generator_cost, feed_dict = curr_feed_dict)
                np_mean_neg_ent_posterior, np_mean_neg_cross_ent_posterior, np_mean_kl_posterior_prior = \
                sess.run([model.mean_neg_ent_posterior, model.mean_neg_cross_ent_posterior, model.mean_kl_posterior_prior], feed_dict = curr_feed_dict)
                
                end = time.time();
                report_format = 'Train: Epoch {} [{:7d}] Time {:.2f} Generator Cost {:.2f} ' +\
                                't {:2d} d {:2d} g {:2d} ne_post {:.1f} nce_post_prior {:.1f}  kl {:.2f}'

                print(report_format.format(epoch, batch_idx * curr_batch_size, (end - start), \
                      np_generator_cost, trans_steps, disc_steps, gen_steps, np_mean_neg_ent_posterior, 
                      np_mean_neg_cross_ent_posterior, np_mean_kl_posterior_prior))

                with open(global_args.exp_dir+"training_traces.txt", "a") as text_file:
                    text_file.write(str(np_generator_cost) + '\n')
                start = time.time()

                if global_args.in_between_vis>0 and report_count % global_args.in_between_vis == 0: 
                    batch['observed']['data']['image'] = random_batch_data
                    curr_feed_dict = input_dict_func(batch, hyper_param)
                    distributions.visualizeProductDistribution4(sess, curr_feed_dict, batch, model.input_dist, model.reg_dist, model.reg_target_dist, model.reconst_dist, model.obs_sample_dist, model.gen_obs_sample_dist, 
                    save_dir=global_args.exp_dir+'Visualization/Train/Random/', postfix='train_'+str(epoch))
                    
                    batch['observed']['data']['image'] = fixed_batch_data
                    curr_feed_dict = input_dict_func(batch, hyper_param)
                    distributions.visualizeProductDistribution4(sess, curr_feed_dict, batch, model.input_dist, model.reg_dist, model.reg_target_dist, model.reconst_dist, model.obs_sample_dist, model.gen_obs_sample_dist, 
                    save_dir=global_args.exp_dir+'Visualization/Train/Fixed/', postfix='train_fixed_'+str(epoch))

        train_gen_loss_accum /= batch_size_accum

        summary_str = sess.run(merged_summaries, feed_dict = curr_feed_dict)
        summary_writer.add_summary(summary_str, (tf.train.global_step(sess, global_step)))
                   
        if epoch % global_args.vis_interval == 0:
            print('====> Average Train: Epoch: {}\tGenerator Cost: {:.3f}'.format(
                  epoch, train_gen_loss_accum))

            if data_loader.__module__ == 'datasetLoaders.RandomManifoldDataLoader' or data_loader.__module__ == 'datasetLoaders.ToyDataLoader':
                helper.visualize_datasets(sess, input_dict_func(batch), data_loader.dataset, generative_dict['obs_sample_out'],
                                          generative_dict['latent_sample_out'], train_outs_dict['transport_sample'], train_outs_dict['input_sample'],
                                          save_dir=global_args.exp_dir+'Visualization/', postfix=str(epoch)) 

                xmin, xmax, ymin, ymax, X_dense, Y_dense = -2.5, 2.5, -2.5, 2.5, 250, 250
                xlist = np.linspace(xmin, xmax, X_dense)
                ylist = np.linspace(ymin, ymax, Y_dense)
                X, Y = np.meshgrid(xlist, ylist)
                XY = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)

                batch['observed']['data']['flat'] = XY[:, np.newaxis, :]
                disc_cost_real_np = sess.run(train_outs_dict['critic_real'], feed_dict = input_dict_func(batch, hyper_param))

                batch['observed']['data']['flat'] = data_loader.dataset[:, np.newaxis, :]
                disc_cost_real_real_np = sess.run(train_outs_dict['critic_real'], feed_dict = input_dict_func(batch, hyper_param))

                disc_mean = disc_cost_real_real_np.mean()
                disc_std = disc_cost_real_real_np.std()
                disc_max = disc_mean+2*disc_std
                disc_min = disc_mean-2*disc_std

                np.clip(disc_cost_real_np, disc_min, disc_max, out=disc_cost_real_np)
                f = np.reshape(disc_cost_real_np[:,0,0], [Y_dense, X_dense])
                helper.plot_ffs(X, Y, f, save_dir=global_args.exp_dir+'Visualization/discriminator_function/', postfix='discriminator_function'+str(epoch))
                
            else:
                batch['observed']['data']['image'] = random_batch_data
                curr_feed_dict = input_dict_func(batch, hyper_param)
                distributions.visualizeProductDistribution4(sess, curr_feed_dict, batch, model.input_dist, model.reg_dist, model.reg_target_dist, model.reconst_dist, model.obs_sample_dist, model.gen_obs_sample_dist, 
                save_dir=global_args.exp_dir+'Visualization/Train/Random/', postfix='train_'+str(epoch))
                
                batch['observed']['data']['image'] = fixed_batch_data
                curr_feed_dict = input_dict_func(batch, hyper_param)

                distributions.visualizeProductDistribution4(sess, curr_feed_dict, batch, model.input_dist, model.reg_dist, model.reg_target_dist, model.reconst_dist, model.obs_sample_dist, model.gen_obs_sample_dist, 
                save_dir=global_args.exp_dir+'Visualization/Train/Fixed/', postfix='train_fixed_'+str(epoch))

            checkpoint_path1 = global_args.exp_dir+'checkpoint/'
            checkpoint_path2 = global_args.exp_dir+'checkpoint2/'
            print('====> Saving checkpoint. Epoch: ', epoch); start_tmp = time.time()
            if global_args.save_checkpoints: helper.save_checkpoint(saver, sess, global_step, checkpoint_path1) 
            end_tmp = time.time(); print('Checkpoint path: '+checkpoint_path1+'   ====> It took: ', end_tmp - start_tmp)
            if epoch % 60 == 0: 
                print('====> Saving checkpoint backup. Epoch: ', epoch); start_tmp = time.time()
                if global_args.save_checkpoints: helper.save_checkpoint(saver, sess, global_step, checkpoint_path2) 
                end_tmp = time.time(); print('Checkpoint path: '+checkpoint_path2+'   ====> It took: ', end_tmp - start_tmp)

    def test(epoch):
        global random_batch_data, fixed_batch_data
        data_loader.eval()
        test_gen_loss_accum, test_dis_loss_accum, test_batch_size_accum = 0, 0, 0
        start = time.time()

        hyperparam_dict = {'b_identity': 0.}
        helper.update_dict_from_file(hyperparam_dict, './hyperparam_file.py')

        random_idx = np.random.randint(0, high=data_loader.curr_max_iter-1)
        for batch_idx, curr_batch_size, batch in data_loader: 
            if random_idx == batch_idx: 
                try:
                    random_batch_data = batch['observed']['data']['image'].copy()
                except: pass

            hyper_param = np.asarray([epoch, hyperparam_dict['b_identity']])
            curr_feed_dict = input_dict_func(batch, hyper_param)       
            
            test_np_generator_cost = sess.run(model.generator_cost, feed_dict = curr_feed_dict)

            test_gen_loss_accum += curr_batch_size*test_np_generator_cost
            test_batch_size_accum += curr_batch_size

        test_gen_loss_accum /= test_batch_size_accum

        end = time.time();
        print('====> Average Test: Epoch {} Time {:.2f} Generator Cost {:.2f}'.format(
              epoch, (end - start), test_gen_loss_accum))

        with open(global_args.exp_dir+"test_traces.txt", "a") as text_file:
            text_file.write(str(test_gen_loss_accum) + '\n')

        if not (data_loader.__module__ == 'datasetLoaders.RandomManifoldDataLoader' or data_loader.__module__ == 'datasetLoaders.ToyDataLoader'):
            batch['observed']['data']['image'] = random_batch_data
            curr_feed_dict = input_dict_func(batch, hyper_param)
            distributions.visualizeProductDistribution4(sess, curr_feed_dict, batch, model.input_dist, model.reg_dist, model.reg_target_dist, model.reconst_dist, model.obs_sample_dist, model.gen_obs_sample_dist, 
            save_dir = global_args.exp_dir+'Visualization/Test/Random/', postfix='test_'+str(epoch))
            
            batch['observed']['data']['image'] = fixed_batch_data
            curr_feed_dict = input_dict_func(batch, hyper_param)
            distributions.visualizeProductDistribution4(sess, curr_feed_dict, batch, model.input_dist, model.reg_dist, model.reg_target_dist, model.reconst_dist, model.obs_sample_dist, model.gen_obs_sample_dist, 
            save_dir = global_args.exp_dir+'Visualization/Test/Fixed/', postfix='test_fixed_'+str(epoch))

    def visualize(epoch):
        data_loader.eval()

        hyperparam_dict = {'b_identity': 0.}
        helper.update_dict_from_file(hyperparam_dict, './hyperparam_file.py')
        all_np_posterior_latent_code= None
        all_np_prior_latent_code = None
        all_np_input_sample = None
        all_np_reconst_sample = None
        all_labels_np = None

        print('\n*************************************   VISUALIZATION STAGE   *************************************\n')
        print('Obtaining visualization data.')
        start = time.time();
        for batch_idx, curr_batch_size, batch in data_loader: 
            hyper_param = np.asarray([epoch, hyperparam_dict['b_identity']])
            curr_feed_dict = input_dict_func(batch, hyper_param)    

            if (global_args.latent_vis_epoch_rate[0]>0 and global_args.curr_epoch % global_args.latent_vis_epoch_rate[0] == global_args.latent_vis_epoch_rate[1]): 
                np_posterior_latent_code, np_prior_latent_code = sess.run([model.posterior_latent_code, model.prior_latent_code], feed_dict = curr_feed_dict)
                if all_np_posterior_latent_code is None: all_np_posterior_latent_code = np_posterior_latent_code
                else: all_np_posterior_latent_code = np.concatenate([all_np_posterior_latent_code, np_posterior_latent_code], axis=0)
                if all_np_prior_latent_code is None: all_np_prior_latent_code = np_prior_latent_code
                else: all_np_prior_latent_code = np.concatenate([all_np_prior_latent_code, np_prior_latent_code], axis=0)

            if (global_args.reconst_vis_epoch_rate[0]>0 and global_args.curr_epoch % global_args.reconst_vis_epoch_rate[0] == global_args.reconst_vis_epoch_rate[1]): 
                np_input_sample, np_reconst_sample = sess.run([model.input_sample['image'], model.reconst_sample['image']], feed_dict = curr_feed_dict)
                if all_np_input_sample is None or all_np_input_sample.shape[0]<300:
                    if all_np_input_sample is None: all_np_input_sample = np_input_sample
                    else: all_np_input_sample = np.concatenate([all_np_input_sample, np_input_sample], axis=0)
                if all_np_reconst_sample is None or all_np_reconst_sample.shape[0]<300:
                    if all_np_reconst_sample is None: all_np_reconst_sample = np_reconst_sample
                    else: all_np_reconst_sample = np.concatenate([all_np_reconst_sample, np_reconst_sample], axis=0)

            if batch['context']['data']['flat'] is not None:
              if all_labels_np is None: all_labels_np = batch['context']['data']['flat'][:,0,:]
              else: all_labels_np = np.concatenate([all_labels_np, batch['context']['data']['flat'][:,0,:]], axis=0)
        end = time.time();
        print('Obtained visualization data: Time: {:.3f}'.format((end - start)))

        if (global_args.pigeonhole_score_epoch_rate[0]>0 and global_args.curr_epoch % global_args.pigeonhole_score_epoch_rate[0] == global_args.pigeonhole_score_epoch_rate[1]) or \
           (global_args.inception_score_epoch_rate[0]>0 and global_args.curr_epoch % global_args.inception_score_epoch_rate[0] == global_args.inception_score_epoch_rate[1]):
            n_random_samples = 50000
            print('Obtaining {:d} random samples.'.format(n_random_samples))
            start = time.time();
            random_samples_from_model = np.zeros((n_random_samples, 1, data_loader.image_size, data_loader.image_size, 3))
            curr_index = 0 
            while curr_index<n_random_samples:
                curr_samples = sess.run(model.gen_obs_sample['image'], feed_dict = curr_feed_dict)
                end_index = min(curr_index+curr_samples.shape[0], n_random_samples)
                random_samples_from_model[curr_index:end_index, ...] = curr_samples[:end_index-curr_index,...]
                curr_index = end_index
            end = time.time()
            print('Obtained random samples: Time: {:.3f}'.format((end - start)))
            
            if global_args.pigeonhole_score_epoch_rate[0]>0 and global_args.curr_epoch % global_args.pigeonhole_score_epoch_rate[0] == global_args.pigeonhole_score_epoch_rate[1]: 
                print('Computing pidgeon-hole score.')
                start = time.time();
                pigeonhole_mean, pigeonhole_std = helper.pigeonhole_score(random_samples_from_model, subset=500, neigh=0.05)
                end = time.time()
                print('Pidgeon-hole Score -- Time: {:.3f} Mean: {:.3f} Std: {:.3f}'.format((end - start), pigeonhole_mean, pigeonhole_std))
            if global_args.inception_score_epoch_rate[0]>0 and global_args.curr_epoch % global_args.inception_score_epoch_rate[0] == global_args.inception_score_epoch_rate[1]:
                print('Computing inception score.')
                start = time.time();
                inception_mean, inception_std = InceptionScoreModel.inception_score(random_samples_from_model)
                end = time.time()
                print('Inception Score -- Time: {:.3f} Mean: {:.3f} Std: {:.3f}'.format((end - start), inception_mean, inception_std))

        if global_args.reconst_vis_epoch_rate[0]>0 and global_args.curr_epoch % global_args.reconst_vis_epoch_rate[0] == global_args.reconst_vis_epoch_rate[1]: 
            b_zero_one_range = False
            print('Visualizing reconstructions')
            start = time.time();
            if b_zero_one_range: np.clip(all_np_reconst_sample, 0, 1, out=all_np_reconst_sample) 
            all_sample_reconst = helper.interleave_data([all_np_input_sample, all_np_reconst_sample])
            helper.visualize_images2(all_sample_reconst[:2*int(np.sqrt(all_np_input_sample.shape[0]))**2, ...], 
            block_size=[int(np.sqrt(all_np_input_sample.shape[0])), 2*int(np.sqrt(all_np_input_sample.shape[0]))], 
            save_dir=global_args.exp_dir+'Visualization/test_real_sample_reconst/', postfix = '_test_real_sample_reconst'+str(epoch))
            
            end = time.time()
            print('Visualized reconstructions: Time: {:.3f}'.format((end - start)))

        if global_args.latent_vis_epoch_rate[0]>0 and global_args.curr_epoch % global_args.latent_vis_epoch_rate[0] == global_args.latent_vis_epoch_rate[1]: 
            print('Visualizing latents.')
            start = time.time();
            # all_np_prior_latent_code = all_np_prior_latent_code[:int(all_np_prior_latent_code.shape[0]*0.4), :]
            # all_np_prior_latent_code = all_np_prior_latent_code[:5000, :]
            n_vis_samples = 5000
            all_np_prior_latent_code = all_np_prior_latent_code[:n_vis_samples, :]

            chosen_indeces = np.random.permutation(np.arange(all_np_posterior_latent_code.shape[0]))
            chosen_indeces = chosen_indeces[:n_vis_samples]
            all_np_posterior_latent_code = all_np_posterior_latent_code[chosen_indeces, :]
            if all_labels_np is not None: all_labels_np = all_labels_np[chosen_indeces, :]

            all_tsne_input = np.concatenate([all_np_posterior_latent_code, all_np_prior_latent_code], axis=0)
            chosen_indeces2 = np.random.permutation(np.arange(all_tsne_input.shape[0]))
            all_tsne = TSNE().fit_transform(all_tsne_input[chosen_indeces2, :])

            all_tsne_centered = all_tsne-np.mean(all_tsne, axis=0)[np.newaxis, :]
            all_tsne_normalized = all_tsne_centered/(np.std(all_tsne_centered, axis=0)[np.newaxis, :]+1e-7)
            all_tsne_normalized_posterior = all_tsne_normalized[:all_np_posterior_latent_code.shape[0],:]
            all_tsne_normalized_prior = all_tsne_normalized[all_np_posterior_latent_code.shape[0]:,:]

            class_wise = []
            class_wise_sizes = []
            if all_labels_np is None:
              class_wise.append(all_tsne_normalized_posterior)
            else:
              for c in range(all_labels_np.shape[1]):
                class_wise_z = all_tsne_normalized_posterior[all_labels_np[:,c].astype(bool),:]
                class_wise_sizes.append(class_wise_z.shape[0])
                class_wise.append(class_wise_z)
              for i in range(len(class_wise)):
                class_wise[i] = class_wise[i][:min(class_wise_sizes),:]

            helper.dataset_plotter(class_wise, save_dir = global_args.exp_dir+'Visualization/z_projection_posterior/', postfix = '_z_projection_posterior'+str(epoch), postfix2 = 'z_projection_posterior')
            helper.dataset_plotter([all_tsne_normalized_prior,], save_dir = global_args.exp_dir+'Visualization/z_projection_prior/', postfix = '_z_projection_prior'+str(epoch), postfix2 = 'z_projection_prior')
            helper.dataset_plotter([all_tsne_normalized_prior, all_tsne_normalized_posterior], save_dir = global_args.exp_dir+'Visualization/z_projection_prior_posterior/', postfix = '_z_projection_prior_posterior'+str(epoch), postfix2 = 'z_projection_prior_posterior')
            
            end = time.time()
            print('Visualized latents: Time: {:.3f}\n'.format((end - start)))

    print('Starting training.')
    while global_args.curr_epoch < global_args.epochs + 1:
        train(global_args.curr_epoch)
        if global_args.curr_epoch % global_args.test_epoch == 0: 
            test(global_args.curr_epoch)
        visualize(global_args.curr_epoch)
        print('Experiment Directory: ', global_args.exp_dir)

        global_args.curr_epoch += 1            







































# helper.draw_bar_plot(convex_mask_np, y_min_max = [0,1], save_dir=global_args.exp_dir+'Visualization/convex_mask/', postfix='convex_mask'+str(epoch))

    # if global_args.optimizer_class == 'RmsProp':
    #     if data_loader.__module__ == 'datasetLoaders.RandomManifoldDataLoader' or data_loader.__module__ == 'datasetLoaders.ToyDataLoader':
    #         generator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.5), 
    #                                   loss=model.generator_cost, var_list=generator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
    #         discriminator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.5), 
    #                                       loss=model.discriminator_cost, var_list=discriminator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
    #         if model.transporter_cost is not None:
    #             transport_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.5), 
    #                                       loss=model.transporter_cost, var_list=transport_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
    #     else:
    #         generator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.9), 
    #                                   loss=model.generator_cost, var_list=generator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
    #         discriminator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.5), 
    #                                       loss=model.discriminator_cost, var_list=discriminator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
    #         if model.transporter_cost is not None:
    #             transport_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.5), 
    #                                       loss=model.transporter_cost, var_list=transport_vars, global_step=global_step, clip_param=global_args.gradient_clipping)

        
        # helper.visualize_images2(all_np_reconst_sample[:int(np.sqrt(all_np_reconst_sample.shape[0]))**2, ...],
        # block_size=[int(np.sqrt(all_np_reconst_sample.shape[0])), int(np.sqrt(all_np_reconst_sample.shape[0]))],
        # save_dir=global_args.exp_dir+'Visualization/test_reconstruction_only/', postfix='_test_reconstruction_only')

        # helper.visualize_images2(all_np_input_sample[:int(np.sqrt(all_np_input_sample.shape[0]))**2, ...],
        # block_size=[int(np.sqrt(all_np_input_sample.shape[0])), int(np.sqrt(all_np_input_sample.shape[0]))],
        # save_dir=global_args.exp_dir+'Visualization/test_real_sample_only/', postfix='_test_real_sample_only')

