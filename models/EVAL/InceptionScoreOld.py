# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import gc

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy
import scipy.misc
import math
import sys
import pdb
import time

class InceptionScore():
    def __init__(self, session, name = 'InceptionScore'):
        self.name = name
        self.model_dir = './models/EVAL/imagenet_model'
        self.data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        self.session = session
        self.output, self.pool3 = self._init_inception()

        # This function is called automatically.
    def _init_inception(self):
      if not os.path.exists(self.model_dir):
        os.makedirs(self.model_dir)
      filename = self.data_url.split('/')[-1]
      filepath = os.path.join(self.model_dir, filename)
      if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (
              filename, float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(self.data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
      tarfile.open(filepath, 'r:gz').extractall(self.model_dir)
      with tf.gfile.FastGFile(os.path.join(
          self.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
      # Works with an arbitrary minibatch size.
      pool3 = self.session.graph.get_tensor_by_name('pool_3:0')
      ops = pool3.graph.get_operations()
      for op_idx, op in enumerate(ops):
          for o in op.outputs:
              try:
                  shape_1 = o.get_shape() 
                  shape = [s.value for s in shape_1]
                  new_shape = []
                  for j, s in enumerate(shape):
                      if s == 1 and j == 0:
                          new_shape.append(None)
                      else:
                          new_shape.append(s)
                  o._shape = tf.TensorShape(new_shape)
                  # print(shape_1, o.get_shape())
              except: 
                pass
                # pdb.set_trace()
      w_matrix = self.session.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
      # w_matrix = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]

      logits = tf.matmul(pool3[:,0,0,:], w_matrix)
      softmax = tf.nn.softmax(logits)
      return softmax[:,:1000], pool3


    # def inception_score(self, images_tensor):
    #   split_images = np.split((images_tensor[:,0,:,:,:]*255), images_tensor.shape[0], axis=0) 
    #   split_images = [split_image[0] for split_image in split_images]
    #   for split_image in split_images:
    #     np.clip(split_image, 0, 255, out=split_image)
    #   # split_images = [scipy.misc.imresize(split_image, (255, 255), interp='bilinear') for split_image in split_images]

    #   assert(type(split_images) == list)
    #   assert(type(split_images[0]) == np.ndarray)
    #   assert(len(split_images[0].shape) == 3)
    #   assert(np.max(split_images[0]) > 10)
    #   assert(np.min(split_images[0]) >= 0.0)
    #   mean, std = self.get_inception_score(split_images, splits=10)
    #   return mean, std
    
    # # Call this function with list of images. Each of elements should be a 
    # # numpy array with values ranging from 0 to 255.
    # def get_inception_score(self, images, splits=10):
    #   assert(type(images) == list)
    #   assert(type(images[0]) == np.ndarray)
    #   assert(len(images[0].shape) == 3)
    #   assert(np.max(images[0]) > 10)
    #   assert(np.min(images[0]) >= 0.0)
    #   inps = []
    #   for img in images:
    #     img = img.astype(np.float32)
    #     inps.append(np.expand_dims(img, 0))
    #   bs = 100
    #   preds = []
    #   n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    #   for i in range(n_batches):
    #       inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
    #       inp = np.concatenate(inp, 0)
    #       pred = self.session.run(self.output, {'ExpandDims:0': inp})
    #       preds.append(pred)
    #   preds = np.concatenate(preds, 0)
    #   scores = []
    #   for i in range(splits):
    #     part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
    #     kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    #     kl = np.mean(np.sum(kl, 1))
    #     scores.append(np.exp(kl))
    #   return np.mean(scores), np.std(scores)


    # def get_activations(self, images, batch_size=50, verbose=True):
    #   """Calculates the activations of the pool_3 layer for all images.

    #   Params:
    #   -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
    #                    must lie between 0 and 256.
    #   -- sess        : current session
    #   -- batch_size  : the images numpy array is split into batches with batch size
    #                    batch_size. A reasonable batch size depends on the disposable hardware.
    #   -- verbose    : If set to True and parameter out_step is given, the number of calculated
    #                    batches is reported.
    #   Returns:
    #   -- A numpy array of dimension (num images, 2048) that contains the
    #      activations of the given tensor when feeding inception with the query tensor.
    #   """
    #   # inception_layer = _get_inception_layer(sess)
    #   # images = images[:,0,:,:,:]
    #   d0 = images.shape[0]
    #   if batch_size > d0:
    #       print("warning: batch size is bigger than the data size. setting batch size to data size")
    #       batch_size = d0
    #   n_batches = d0//batch_size
    #   n_used_imgs = n_batches*batch_size
    #   pred_arr = np.empty((n_used_imgs,2048))
    #   for i in range(n_batches):
    #       if verbose:
    #           print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
    #       start = i*batch_size
    #       end = start + batch_size
    #       batch = images[start:end]
    #       # pred = self.session.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
    #       pred = self.session.run(self.pool3, {'ExpandDims:0': batch})
    #       pred_arr[start:end] = pred.reshape(batch_size,-1)
    #   if verbose:
    #       print(" done")
    #   return pred_arr

    # def calculate_activation_statistics(self, images, batch_size=50, verbose=True):
    #   """Calculation of the statistics used by the FID.
    #   Params:
    #   -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
    #                    must lie between 0 and 255.
    #   -- sess        : current session
    #   -- batch_size  : the images numpy array is split into batches with batch size
    #                    batch_size. A reasonable batch size depends on the available hardware.
    #   -- verbose     : If set to True and parameter out_step is given, the number of calculated
    #                    batches is reported.
    #   Returns:
    #   -- mu    : The mean over samples of the activations of the pool_3 layer of
    #              the incption model.
    #   -- sigma : The covariance matrix of the activations of the pool_3 layer of
    #              the incption model.
    #   """
    #   act = self.get_activations(images, batch_size, verbose)
    #   mu = np.mean(act, axis=0)
    #   sigma = np.cov(act, rowvar=False)
    #   return mu, sigma


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
      """Numpy implementation of the Frechet Distance.
      The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
      and X_2 ~ N(mu_2, C_2) is
              d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

      Params:
      -- mu1 : Numpy array containing the activations of the pool_3 layer of the
               inception net ( like returned by the function 'get_predictions')
      -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                 on an representive data set.
      -- sigma2: The covariance matrix over activations of the pool_3 layer,
                 precalcualted on an representive data set.

      Returns:
      -- dist  : The Frechet Distance.

      Raises:
      -- InvalidFIDException if nan occures.
      """
      m = np.square(mu1 - mu2).sum()
      s = scipy.linalg.sqrtm(np.dot(sigma1, sigma2))
      dist = m + np.trace(sigma1+sigma2 - 2*s)
      if np.isnan(dist):
          print("nan occured in distance calculation.")
          dist = -1
      return dist
    
    # def frechet_distance(self, random_samples_from_model, dataset_to_use, data_loader):
    #   assert(dataset_to_use in ['IMAGENET', 'BEDROOM', 'CELEB', 'CAT', 'FLOWERS', 'CUB', 'CIFAR10'])
    #   preprocessed_path = self.model_dir+'/'+dataset_to_use+'_activation_mean_sig.npz'
    #   try:
    #     print('Trying to load from preprocessed FID mu and sigma file: ', preprocessed_path)
    #     f = np.load(preprocessed_path)
    #     real_mu, real_sigma = f['mu'][:], f['sigma'][:]
    #     f.close()
    #   except: 
    #     print('Failed. Creating processed file for FID mu and sigma.')
    #     start = time.time()
    #     real_mu, real_sigma = self.calculate_activation_statistics(data_loader.train_data, batch_size=100)
    #     np.savez_compressed(preprocessed_path, mu=real_mu, sigma=real_sigma)
    #     end = time.time()
    #     print('Success creating preprocessed FID mu and sigma file. Time: ', (end-start))

    #   print('Computing mu and sigma for generated samples.')
    #   model_mu, model_sigma = self.calculate_activation_statistics((random_samples_from_model[:,0,:,:,:]*255).astype(np.uint8), batch_size=100)
    #   fid_value = self.calculate_frechet_distance(model_mu, model_sigma, real_mu, real_sigma)
    #   return fid_value

    def batch_covariance(self, input_mat, mu=None, batch_size=500):
      if mu is None: mu = np.mean(input_mat, axis=0)
      input_mat_centered = input_mat-mu[np.newaxis,:]
      input_size = input_mat_centered.shape[0]
      n_batches = int(np.ceil(float(input_size)/float(batch_size)))
      acc = None 
      for i in range(n_batches):
        curr_batch = input_mat_centered[i*batch_size:(i+1)*batch_size, :]
        curr_cov = np.dot(curr_batch.T, curr_batch) 
        if acc is None: acc = curr_cov
        else: acc = acc + curr_cov
      acc = acc / (input_size-1)
      return acc

    def compute_fid_and_inception_score(self, network_input, batch_size=100, verbose=True):
      network_input_min, network_input_max = np.min(network_input), np.max(network_input)
      assert(type(network_input) == np.ndarray)
      assert(network_input.dtype == np.uint8)
      assert(len(network_input.shape) == 4)
      assert(network_input_min >= 0)
      assert(network_input_max > 1)
      assert(network_input_max <= 255)

      # np.clip(network_input, 0, 255, out=network_input)
      if batch_size > network_input.shape[0]:
          print("warning: batch size is bigger than the data size. setting batch size to data size")
          batch_size = network_input.shape[0]

      preds = np.empty((network_input.shape[0], 1000))
      activations = np.empty((network_input.shape[0], 2048))
      n_batches = int(math.ceil(float(network_input.shape[0]) / float(batch_size)))
      for i in range(n_batches):
        if verbose: print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start, end = i*batch_size, min((i + 1) * batch_size, network_input.shape[0])
        input_to = network_input[start:end, ...]
        pdbself.session.graph.get_tensor_by_name('ExpandDims:0')
        print(self.session.graph.get_tensor_by_name('ExpandDims:0').shape, input_to.shape)
        pool3, pred = self.session.run([self.pool3, self.output], {'ExpandDims:0': input_to})
        activations[start:end, :] = pool3[:,0,0,:]
        preds[start:end, :] = pred
      
      print("\nComputing activation statistics.")
      gc.collect()
      activation_mu = np.mean(activations, axis=0)
      activation_sigma = self.batch_covariance(activations, activation_mu, batch_size=1000)
      # activation_sigma = np.cov(activations, rowvar=False)
      gc.collect()

      print("Computing inception score.")
      scores = []
      splits=10
      batch_size_split = int(math.ceil(float(network_input.shape[0]) / float(splits)))
      for i in range(splits):
        start, end = i*batch_size_split, min((i + 1) * batch_size_split, network_input.shape[0])
        part = preds[start:end, :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl_mean = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl_mean))
      inception_mean, inception_std = np.mean(scores), np.std(scores)
      
      return activation_mu, activation_sigma, inception_mean, inception_std

    def fid_and_inception_score(self, random_samples_from_model, dataset_to_use, data_loader):      
      assert(dataset_to_use in ['IMAGENET', 'BEDROOM', 'CELEB', 'CAT', 'FLOWERS', 'CUB', 'CIFAR10'])
      preprocessed_path = self.model_dir+'/'+dataset_to_use+'_activation_mean_sig.npz'
      reset = False
      try:
        assert(not reset)
        print('Trying to load from preprocessed Inception statistics file: ', preprocessed_path)
        f = np.load(preprocessed_path)
        if len(f['real_inception_std'].shape) == 0:
          real_activation_mu, real_activation_sigma, real_inception_mean, real_inception_std = f['real_activation_mu'][:], f['real_activation_sigma'][:], f['real_inception_mean']+0, f['real_inception_std']+0
        else:
          real_activation_mu, real_activation_sigma, real_inception_mean, real_inception_std = f['real_activation_mu'][:], f['real_activation_sigma'][:], f['real_inception_mean'][:], f['real_inception_std'][:]
        f.close()
      except: 
        print('Failed. Creating file for Inception statistics.')
        start = time.time()
        if dataset_to_use == 'CELEB': real_input = data_loader.train_data
        if dataset_to_use in ['CIFAR10', 'CAT', 'FLOWERS', 'CUB']: real_input = (data_loader.train_data*255).astype(np.uint8)
        if dataset_to_use in ['IMAGENET', 'BEDROOM']: pdb.set_trace()
        
        real_activation_mu, real_activation_sigma, real_inception_mean, real_inception_std = self.compute_fid_and_inception_score(real_input)
        np.savez_compressed(preprocessed_path, real_activation_mu = real_activation_mu, real_activation_sigma = real_activation_sigma, \
                            real_inception_mean=real_inception_mean, real_inception_std=real_inception_std)
        end = time.time()
        print('Success creating file for Inception statistics. Time: ', (end-start))

      print('Computing Inception statistics for generated samples.')
      model_input = (random_samples_from_model[:,0,:,:,:]*255).astype(np.uint8)
      model_activation_mu, model_activation_sigma, model_inception_mean, model_inception_std = self.compute_fid_and_inception_score(model_input)
      model_fid_value = self.calculate_frechet_distance(model_activation_mu, model_activation_sigma, real_activation_mu, real_activation_sigma)
      return model_fid_value, model_inception_mean, model_inception_std, real_inception_mean, real_inception_std

      # split_images = np.split((images_tensor[:,0,:,:,:]*255), images_tensor.shape[0], axis=0) 
      # split_images = [split_image[0] for split_image in split_images]
      # for split_image in split_images:
      #   np.clip(split_image, 0, 255, out=split_image)
      # # split_images = [scipy.misc.imresize(split_image, (255, 255), interp='bilinear') for split_image in split_images]

      # assert(type(split_images) == list)
      # assert(type(split_images[0]) == np.ndarray)
      # assert(len(split_images[0].shape) == 3)
      # assert(np.max(split_images[0]) > 10)
      # assert(np.min(split_images[0]) >= 0.0)
      # mean, std = self.get_inception_score(split_images, splits=10)
      # return mean, std

# The Inception Score calculation has 3 mistakes.

# It uses an outdated Inception network that in fact outputs a 1008-vector of classes (see the following GitHub issue):
# It turns out that the 1008 size softmax output is an artifact of dimension back-compatibility with a older, Google-internal system. Newer versions of the inception model have 1001 output classes, where one is an "other" class used in training. You shouldn't need to pay any attention to the extra 8 outputs.

# Fix: See link for the new inception Model.

# It calculates the kl-divergence directly using logs, which leads to numerical instabilities (can output nan instead of inf). Instead, scipy.stats.entropy should be used.
# kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
# kl = np.mean(np.sum(kl, 1))
# Fix: Replace the above with something along the lines of the following:

# py = np.mean(part, axis=0)
# l = np.mean([entropy(part[i, :], py) for i in range(part.shape[0])])
# It calculates the mean of the exponential of the split rather than the exponential of the mean:
# Here is the code in inception_score.py which does this:

#     scores.append(np.exp(kl))
# return np.mean(scores), np.std(scores)
# This is clearly problematic, as can easily be seen in a very simple case with a x~Bernoulli(0.5) random variable that E[e^x] = .5(e^(0) + e^(1)) != e^(.5(0)+.5(1)) = e^[E[x]]. This can further be seen with an example w/ a uniform random variable, where the split-mean over-estimates the exponential.

# import numpy as np
# data = np.random.uniform(low=0., high=15., size=1000)
# split_data = np.split(data, 10)
# np.mean([np.exp(np.mean(x)) for x in split_data]) # 1608.25
# np.exp(np.mean(data)) # 1477.25
# Fix: Do not calculate the mean of the exponential of the split, and instead calculate the exponential of the mean of the KL-divergence over all 50,000 inputs.
