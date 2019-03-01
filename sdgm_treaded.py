import sys

import sys
import inspect
import traceback

import os
import pdb
import time
import shutil
import argparse

# # MNIST
# parser = argparse.ArgumentParser(description='Tensorflow Sequential Latent Variable Models')
# parser.add_argument('--global_exp_dir', type=str, default='./experimentsMnist', help='Directory to put the experiments.')
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
# parser.add_argument('--log_interval', type=int, default=500, help='how many batches to wait before logging training status')

# parser.add_argument('--n_encoder', type=int, default=400, help='n_encoder.')
# parser.add_argument('--n_decoder', type=int, default=400, help='n_decoder.')
# parser.add_argument('--n_context', type=int, default=1, help='n_context.')
# parser.add_argument('--n_state', type=int, default=1, help='n_state.')
# parser.add_argument('--n_latent', type=int, default=50, help='n_latent.')
# global_args = parser.parse_args()
# global_args.curr_epoch = 1

# from datasetLoaders.MnistLoader import DataLoader
# data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)

# ##############################################################################################################################


# # TOY
parser = argparse.ArgumentParser(description='Tensorflow Sequential Latent Variable Models')
parser.add_argument('--global_exp_dir', type=str, default='./experimentsL2', help='Directory to put the experiments.')
parser.add_argument('--exp_dir_postfix', type=str, default='', help='Directory to put the experiment postfix.')
parser.add_argument('--dataset_dir', type=str, default='../dataset/dataset_both2/scripted/*.npz', help='Directory of data.')
parser.add_argument('--restore_dir', type=str, default='/7cb1611a0a584269a019dc8342aee213/checkpoint/', help='Directory of restore experiment.')
parser.add_argument('--restore', type=bool, default=False, help='Restore model.')

parser.add_argument('--analyticKL', type=bool, default=True, help='Type of KL divergence to use.')
parser.add_argument('--transformedQ', type=bool, default=True, help='Use posterior Transform.')
parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=50, help='Input batch size for training.')
parser.add_argument('--time_steps', type=int, default=1, help='Number of timesteps')
parser.add_argument('--hierarchy_rate', type=int, default=1, help='Number of timesteps')

parser.add_argument('--optimizer_class', type=str, default='Adam', help='Optimizer type.')
parser.add_argument('--learning_rate', type=float, default=0.000001, help='Initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial momentum.')
parser.add_argument('--weight_decay', type=float, default=0, help='Initial weight decay.')
parser.add_argument('--initial_temp', type=float, default=0.1, help='Initial temperature for KL divergence.')
parser.add_argument('--max_step_temp', type=float, default=15000, help='Starting step for temp=1.')

parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging training status')

parser.add_argument('--n_encoder', type=int, default=100, help='n_encoder.')
parser.add_argument('--n_decoder', type=int, default=100, help='n_decoder.')
parser.add_argument('--n_context', type=int, default=1, help='n_context.')
parser.add_argument('--n_state', type=int, default=1, help='n_state.')
parser.add_argument('--n_latent', type=int, default=20, help='n_latent.')
global_args = parser.parse_args()
global_args.curr_epoch = 1

# from datasetLoaders.ToyDataLoader import DataLoader
# data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)
from datasetLoaders.RandomManifoldDataLoader import DataLoader
data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)

##############################################################################################################################

from models.SLVM.SLVM import SLVM
from models.SLVML2.SLVML2 import SLVM

import distributions 
import helper

import random
import numpy as np
random.seed(global_args.seed)
np.random.seed(global_args.seed)
import tensorflow as tf
tf.set_random_seed(global_args.seed)

global_args.exp_dir = helper.get_exp_dir(global_args)

os.environ['CUDA_VISIBLE_DEVICES'] = ''
print("os.environ['CUDA_VISIBLE_DEVICES'], ", os.environ['CUDA_VISIBLE_DEVICES'])
print("TENSORBOARD: Linux:\npython -m tensorflow.tensorboard --logdir=model1:"+\
    os.path.realpath(global_args.exp_dir)+" --port="+str(20000+int(global_args.exp_dir[-4:-1], 16))+" &")
print("TENSORBOARD: Mac:\nhttp://0.0.0.0:"+str(20000+int(global_args.exp_dir[-4:-1], 16)))
print("\n\n\n")

shutil.copyfile('./models/SLVM.py', global_args.exp_dir+'SLVM.py')
shutil.copyfile('./models/ModelGTM.py', global_args.exp_dir+'ModelGTM.py')

_, _, batch = next(data_loader)
with tf.Graph().as_default():
    tf.set_random_seed(global_args.seed)
    model = SLVM(vars(global_args))

    global_step = tf.Variable(0.0, name='global_step', trainable=False)
    with tf.variable_scope("training"):
        tf.set_random_seed(global_args.seed)
        batch_tf, input_dict_func = helper.tf_batch_and_input_dict(batch)

        train_out_list, test_out_list = model.inference(batch_tf, global_step)
        batch_loss_tf = train_out_list[0]
        obs_dist = model.obs_dist
        sample_obs_dist, obs_sample_out_tf, latent_sample_out_tf = model.generative_model(batch_tf)

    if global_args.optimizer_class == 'RmsProp':
        train_step_tf = tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, 
            momentum=0.9).minimize(batch_loss_tf, global_step=global_step)
    elif global_args.optimizer_class == 'Adam':
        train_step_tf = tf.train.AdamOptimizer(learning_rate=global_args.learning_rate, 
            beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(batch_loss_tf, global_step=global_step)

    helper.variable_summaries(batch_loss_tf, '/batch_loss_tf')
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
        train_loss, curr_batch_size_accum = 0, 0 
        start = time.time();
        for batch_idx, curr_batch_size, batch in data_loader: 
            train_step, batch_loss, train_elbo_per_sample, train_likelihood, train_kl, curr_temp =\
                sess.run([train_step_tf, *train_out_list], feed_dict = input_dict_func(batch))
            train_loss += curr_batch_size*batch_loss
            curr_batch_size_accum += curr_batch_size

            if batch_idx % global_args.log_interval == 0:
                end = time.time();
                print('Train Epoch: {} [{:7d} ()]\tLoss: {:.6f}\tLikelihood: {:.6f}\tKL: {:.6f}\tTime: {:.3f}, Temperature: {:.3f}'.format(
                      epoch, batch_idx * curr_batch_size, batch_loss, train_likelihood, train_kl, (end - start), curr_temp))
                with open(global_args.exp_dir+"training_traces.txt", "a") as text_file:
                    trace_string = str(batch_loss) + ', ' + str(train_likelihood)+ ', ' + str(train_kl)+ ', ' + str(curr_temp) + '\n' 
                    text_file.write(trace_string)
                start = time.time()
        
        summary_str = sess.run(merged_summaries, feed_dict=input_dict_func(batch))
        summary_writer.add_summary(summary_str, (tf.train.global_step(sess, global_step)))
        
        if epoch % 10 == 0:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss/curr_batch_size_accum))
            
            distributions.visualizeProductDistribution(sess, input_dict_func(batch), batch, obs_dist, sample_obs_dist, 
                save_dir=global_args.exp_dir+'Visualization/', postfix='train')
            if data_loader.__module__ == 'datasetLoaders.RandomManifoldDataLoader' or data_loader.__module__ == 'datasetLoaders.ToyDataLoader':
                helper.visualize_datasets(sess, input_dict_func(batch), data_loader.dataset, obs_sample_out_tf, latent_sample_out_tf,
                    save_dir=global_args.exp_dir+'Visualization/', postfix=str(epoch)) 

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
        test_loss_accum, test_likelihood_accum, test_kl_accum, batch_size_accum = 0, 0, 0, 0
        start = time.time()

        for batch_idx, curr_batch_size, batch in data_loader: 
            test_batch_loss, test_elbo_per_sample, test_likelihood, test_kl, curr_temp =\
                sess.run(test_out_list, feed_dict = input_dict_func(batch))
            test_loss_accum += curr_batch_size*test_batch_loss
            test_likelihood_accum += curr_batch_size*test_likelihood 
            test_kl_accum += curr_batch_size*test_kl 
            batch_size_accum += curr_batch_size

        test_loss_accum /= batch_size_accum
        test_likelihood_accum /= batch_size_accum
        test_kl_accum /= batch_size_accum

        end = time.time();
        print('\n\n====> Test Epoch: {} [{:7d} ()]\tLoss: {:.6f}\tLikelihood: {:.6f}\tKL: {:.6f}\tTime: {:.3f}, Temperature: {:.3f}\n\n'.format(
            epoch, batch_size_accum, test_loss_accum, test_likelihood_accum, test_kl_accum, (end - start), curr_temp))
        with open(global_args.exp_dir+"test_traces.txt", "a") as text_file:
            trace_string = str(test_loss_accum)+ ', ' + str(test_likelihood_accum)+ ', '+\
                           str(test_kl_accum)+ ', ' + str(curr_temp) + '\n' 
            text_file.write(trace_string)
        distributions.visualizeProductDistribution(sess, input_dict_func(batch), batch, obs_dist, sample_obs_dist, 
            save_dir = global_args.exp_dir+'Visualization/', postfix = 'test')

    print('Starting training.')
    while global_args.curr_epoch < global_args.epochs + 1:
        train(global_args.curr_epoch)
        if global_args.curr_epoch % 20 == 0: 
            test(global_args.curr_epoch)
        global_args.curr_epoch += 1
       








# from datasetLoaders.ActionSeqLoader import DataLoader
# data_loader = DataLoader(batch_size = global_args.batch_size, cuda = global_args.cuda, 
#     time_steps = global_args.time_steps, data_path_pattern = global_args.dataset_dir)
