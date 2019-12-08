import pdb
import numpy as np
import math
import scipy.misc
import pickle
import zlib
import os
import time
from pathlib import Path
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib

class DataLoader:
	def __init__(self, batch_size, time_steps, data_type='regular', b_fixed_amplitute=True, b_context=True, cuda=False):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.mode = 'Train'
		self.cuda = cuda
		self.b_context = b_context
		self.iter = 0
		self.image_size = 28
		self.b_fixed_amplitute = b_fixed_amplitute
		self.min_interval_val = 0.0001

		if data_type == 'regular':
			self.dataset_path = str(Path.home())+'/datasets/mnist_data/dataset/'
		elif data_type == 'binary':
			self.dataset_path = str(Path.home())+'/datasets/mnist_data/dataset_binary/'
		elif data_type == 'rotation':
			self.dataset_path = str(Path.home())+'/datasets/mnist_data/dataset_rotation/'

		print('Loading color mnist data')
		start = time.time()
		reset_data = False
		try: 
			assert(not reset_data)
			print('Trying to load from processed train file.')
			self.train_data = np.load(self.dataset_path+'color_'+data_type+'_fixed_amp_'+str(self.b_fixed_amplitute)+'_train_data.npy')
			self.train_label = np.load(self.dataset_path+'color_'+data_type+'_fixed_amp_'+str(self.b_fixed_amplitute)+'_train_label.npy')
			self.test_data = np.load(self.dataset_path+'color_'+data_type+'_fixed_amp_'+str(self.b_fixed_amplitute)+'_test_data.npy')
			self.test_label = np.load(self.dataset_path+'color_'+data_type+'_fixed_amp_'+str(self.b_fixed_amplitute)+'_test_label.npy')
			print('Success loading train-test data from processed test file.')
		except:
			print('Failed. Creating processed files.')
			start_indiv = time.time()
			self.dataset = input_data.read_data_sets(self.dataset_path, one_hot=True)

			self.n_train_examples = self.dataset.train.num_examples
			self.n_test_examples = self.dataset.test.num_examples

			self.raw_train_data, self.train_label = self.dataset.train.next_batch(self.dataset.train.num_examples, shuffle=False)
			self.raw_test_data, self.test_label = self.dataset.test.next_batch(self.dataset.test.num_examples, shuffle=False)
			
			self.raw_train_data = self.raw_train_data.astype(np.float32)
			self.raw_test_data = self.raw_test_data.astype(np.float32)
			self.train_label = self.train_label.astype(np.float32)
			self.test_label = self.test_label.astype(np.float32)

			self.train_data = np.zeros((self.n_train_examples, self.image_size, self.image_size, 3), dtype=np.float32)
			self.test_data = np.zeros((self.n_test_examples, self.image_size, self.image_size, 3), dtype=np.float32)

			self.train_data[:,:,:,0][:] = self.raw_train_data.reshape(-1, self.image_size, self.image_size)
			self.train_data[:,:,:,1][:] = self.raw_train_data.reshape(-1, self.image_size, self.image_size)
			self.train_data[:,:,:,2][:] = self.raw_train_data.reshape(-1, self.image_size, self.image_size)
			
			self.test_data[:,:,:,0][:] = self.raw_test_data.reshape(-1, self.image_size, self.image_size)
			self.test_data[:,:,:,1][:] = self.raw_test_data.reshape(-1, self.image_size, self.image_size)
			self.test_data[:,:,:,2][:] = self.raw_test_data.reshape(-1, self.image_size, self.image_size)
			
			self.train_data = self.colorify(self.train_data, b_fixed_amplitute=self.b_fixed_amplitute)
			self.train_data = self.noisify(self.train_data)

			self.test_data = self.colorify(self.test_data, b_fixed_amplitute=self.b_fixed_amplitute)
			self.test_data = self.noisify(self.test_data)

			np.save(self.dataset_path+'color_'+data_type+'_fixed_amp_'+str(self.b_fixed_amplitute)+'_train_data.npy', self.train_data)
			np.save(self.dataset_path+'color_'+data_type+'_fixed_amp_'+str(self.b_fixed_amplitute)+'_train_label.npy', self.train_label)
			np.save(self.dataset_path+'color_'+data_type+'_fixed_amp_'+str(self.b_fixed_amplitute)+'_test_data.npy', self.test_data)
			np.save(self.dataset_path+'color_'+data_type+'_fixed_amp_'+str(self.b_fixed_amplitute)+'_test_label.npy', self.test_label)

		self.train_data = (1-2*self.min_interval_val)*self.train_data+self.min_interval_val
		self.test_data = (1-2*self.min_interval_val)*self.test_data+self.min_interval_val

		end = time.time()
		print('Loaded color mnist data. Time: ', (end-start))
		
		self.n_train_examples = self.train_data.shape[0]
		self.n_test_examples = self.test_data.shape[0]

		self.train_max_iter = np.floor(self.n_train_examples/(self.batch_size*self.time_steps))
		self.test_max_iter = np.floor(self.n_test_examples/(self.batch_size*self.time_steps))

		self.train()
		self.reset()

		self.batch = {}
		if self.b_context:
			self.batch['context'] = {'properties': {'flat': [{'dist': 'cat', 'name': 'Digit Class', 'size': tuple([self.batch_size, self.time_steps, 10])}], 
													'image': []},
									 'data':       {'flat': None,
									 		  	    'image': None}}
		else:
			self.batch['context'] = {'properties': {'flat': [], 'image': []},
									 'data':       {'flat': None, 'image': None}}

		self.batch['observed'] = {'properties': {'flat': [], 
												 'image': [{'dist': 'intr', 'name': 'Digit Image', 'size': tuple([self.batch_size, self.time_steps, self.image_size, self.image_size, 3])}]},
								  'data':       {'flat': None,
								 		  	     'image': None}}
	
	def colorify(self, input_mat, b_fixed_amplitute=False):
		if not b_fixed_amplitute:
			return input_mat*(0.3+0.7*np.random.uniform(size=(input_mat.shape[0], 1, 1, 3)))
		else:
			# hue = np.tile(2*np.pi*np.random.uniform(size=(input_mat.shape[0], 1, 1, 1)), (1,input_mat.shape[1],input_mat.shape[2],1))
			# saturation = np.tile(100*np.random.uniform(size=(input_mat.shape[0], 1, 1, 1)), (1,input_mat.shape[1],input_mat.shape[2],1))
			hue = np.tile(np.random.uniform(size=(input_mat.shape[0], 1, 1, 1)), (1,input_mat.shape[1],input_mat.shape[2],1)).astype(np.float32)
			saturation = 0.25+0.75*np.tile(np.random.uniform(size=(input_mat.shape[0], 1, 1, 1)), (1,input_mat.shape[1],input_mat.shape[2],1)).astype(np.float32)
			intensity = input_mat[:, :, :, 0, np.newaxis]	
			hsv = np.concatenate([hue, saturation, intensity], axis=3).astype(np.float32)
			return matplotlib.colors.hsv_to_rgb(hsv).astype(np.float32)

	def noisify(self, input_mat, noise_scale=0.1, pixel_rate=0.1):
		noise = np.random.binomial(1, pixel_rate, size=(input_mat.shape[0], self.image_size, self.image_size, 3))*\
				(2*noise_scale*np.random.uniform(size=(input_mat.shape[0], self.image_size, self.image_size, 3))-noise_scale)
		noisified = input_mat+noise
		return np.clip(noisified, 0, 1)

	def train(self, randomize=True):
		self.mode = 'Train'
		self.curr_data_order = np.arange(self.train_data.shape[0])
		if randomize: self.curr_data_order=np.random.permutation(self.curr_data_order)
		self.curr_data = self.train_data[self.curr_data_order, ...]
		self.curr_label = self.train_label[self.curr_data_order, ...]
		self.curr_max_iter = self.train_max_iter
		self.reset()

	def eval(self, randomize=False):
		self.mode = 'Test'
		self.curr_data_order = np.arange(self.test_data.shape[0])
		if randomize: self.curr_data_order=np.random.permutation(self.curr_data_order)
		self.curr_data = self.test_data[self.curr_data_order, ...]
		self.curr_label = self.test_label[self.curr_data_order, ...]
		self.curr_max_iter = self.test_max_iter
		self.reset()

	def reset(self):
		self.iter = 0
	
	def next_batch(self):
		self._batch_observed_data_image = \
			self.curr_data[self.iter*self.batch_size*self.time_steps: (self.iter+1)*self.batch_size*self.time_steps,:,:,:].reshape(\
			self.batch_size, self.time_steps, self.image_size, self.image_size, 3)
		
		self._batch_observed_data_label = \
			self.curr_label[self.iter*self.batch_size*self.time_steps: (self.iter+1)*self.batch_size*self.time_steps,:].reshape(\
			self.batch_size, self.time_steps, 10)

	def __iter__(self):
		return self
	
	def __next__(self):
		if self.iter == int(self.curr_max_iter): 
			raise StopIteration		

		self.next_batch()

		if self.b_context: self.batch['context']['data']['flat'] = self._batch_observed_data_label
		self.batch['observed']['data']['image'] = self._batch_observed_data_image

		self.iter += 1
		return self.iter, self.batch_size, self.batch 






	# def colorify(self):
	# 	self._batch_obs = self._batch_obs*(0.3+0.7*np.random.uniform(size=(self.batch_size*self.time_steps, 1, 3)))

	# # def adjust_range(self):
	# 	# self._batch_obs = self._batch_obs*2-1

	# def noisify(self, noise_scale=0.1, pixel_rate=0.1):
	# 	# noise = np.random.normal(scale=scale, size=(self.batch_size*self.time_steps, self.image_size*self.image_size, 3))
	# 	noise = np.random.binomial(1, pixel_rate, size=(self.batch_size*self.time_steps, self.image_size*self.image_size, 3))*(2*noise_scale*np.random.uniform(size=(self.batch_size*self.time_steps, self.image_size*self.image_size, 3))-noise_scale)
	# 	self._batch_obs = self._batch_obs+noise
	# 	self._batch_obs = np.clip(self._batch_obs, 0, 1)
	
	# def reset(self):
	# 	self._batch_obs = np.zeros((self.batch_size*self.time_steps, self.image_size*self.image_size, 3))
	# 	self.iter = 0
	
	# def __next__(self):
	# 	if self.iter == int(self.curr_max_iter): 
	# 		raise StopIteration		
		
	# 	self.colorify()
	# 	# self.adjust_range()
	# 	self.noisify()
	# 	# self._batch_obs = np.clip(self._batch_obs, -1, 1)
	# 	if self.b_context: self.batch['context']['data']['flat'] = context.reshape(self.batch_size, self.time_steps, -1)
	# 	self.batch['observed']['data']['image'] = self._batch_obs.reshape(self.batch_size, self.time_steps, self.image_size, self.image_size, 3)

	# 	self.iter += 1
	# 	return self.iter, self.batch_size, self.batch 





# loader = DataLoader(batch_size = 15, time_steps = 10)
# context_fc, context_conv, obs = next(loader)
# pdb.set_trace()

# [('observed', 'flat', 'cat', 'action_code', (20, 1, 4)),
#  ('observed', 'flat', 'cat', 'interaction_with_hero', (20, 1, 11)),
#  ('observed', 'flat', 'cat', 'interaction_with_ability', (20, 1, 31)),
#  ('observed', 'flat', 'cat', 'interaction_with_building', (20, 1, 37)),
#  ('observed', 'flat', 'cat', 'interaction_with_runes', (20, 1, 7)),
#  ('observed', 'flat', 'cat', 'interaction_with_misc', (20, 1, 4)),
#  ('observed', 'flat', 'cat', 'action_minimap', (20, 1, 101)),
#  ('observed', 'flat', 'cont', 'target_location', (20, 1, 4)),
#  ('observed', 'flat', 'bern', 'target_location_valid', (20, 1, 2))]






