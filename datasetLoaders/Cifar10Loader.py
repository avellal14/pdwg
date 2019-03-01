import pdb
import numpy as np
import math
import scipy.misc
import pickle
import zlib
import os
import time
from six.moves import cPickle
from pathlib import Path

class DataLoader:
	def __init__(self, batch_size, time_steps, data_type='regular', b_context=True, cuda=False):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.mode = 'Train'
		self.cuda = cuda
		self.b_context = b_context
		self.iter = 0
		self.image_size = 32
		self.train_subsample_size = None

		if data_type == 'regular':
			# self.dataset_path = '/home/mcgemici/datasets/cifar10_data/cifar-10-batches-py/'
			# self.dataset_path = '../datasets/cifar10_data/cifar-10-batches-py/'
			self.dataset_path = str(Path.home())+'/datasets/cifar10_data/cifar-10-batches-py/'

		print('Loading cifar10 data')
		start = time.time()
		reset_data = False
		try: 
			assert(not reset_data)
			print('Trying to load from processed train file.')
			self.train_data = np.load(self.dataset_path+data_type+'train_data.npy')
			self.train_label = np.load(self.dataset_path+data_type+'train_label.npy')
			self.test_data = np.load(self.dataset_path+data_type+'test_data.npy')
			self.test_label = np.load(self.dataset_path+data_type+'test_label.npy')
			print('Success loading train-test data from processed test file.')
		except:
			print('Failed. Creating processed files.')
			start_indiv = time.time()
			
			train_filename_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
			test_filename_list = ['test_batch',]

			self.raw_train_data = None
			self.raw_train_label = None
			for e in train_filename_list:
				f = open(self.dataset_path+e, 'rb')
				datadict = cPickle.load(f, encoding='latin1')
				f.close()
				if self.raw_train_data is None: self.raw_train_data = np.asarray(datadict["data"], dtype=np.uint8)
				else: self.raw_train_data = np.concatenate((self.raw_train_data, np.asarray(datadict["data"], dtype=np.uint8)), axis=0) 
				if self.raw_train_label is None: self.raw_train_label = np.asarray(datadict["labels"], dtype=np.uint8)
				else: self.raw_train_label = np.concatenate((self.raw_train_label, np.asarray(datadict["labels"], dtype=np.uint8)), axis=0) 

			self.raw_test_data = None
			self.raw_test_label = None
			for e in test_filename_list:
				f = open(self.dataset_path+e, 'rb')
				datadict = cPickle.load(f, encoding='latin1')
				f.close()
				if self.raw_test_data is None: self.raw_test_data = np.asarray(datadict["data"], dtype=np.uint8)
				else: self.raw_test_data = np.concatenate((self.raw_test_data, np.asarray(datadict["data"], dtype=np.uint8)), axis=0) 
				if self.raw_test_label is None: self.raw_test_label = np.asarray(datadict["labels"], dtype=np.uint8)
				else: self.raw_test_label = np.concatenate((self.raw_test_label, np.asarray(datadict["labels"], dtype=np.uint8)), axis=0) 

			self.train_data = self.raw_train_data.reshape(-1, 3, self.image_size, self.image_size).astype(np.float32)
			self.test_data = self.raw_test_data.reshape(-1, 3, self.image_size, self.image_size).astype(np.float32)
			self.train_data = np.transpose(self.train_data, (0, 2, 3, 1)) 
			self.test_data = np.transpose(self.test_data, (0, 2, 3, 1)) 
			self.train_data = self.train_data/255.
			self.test_data = self.test_data/255.

			self.train_label = np.zeros((self.raw_train_label.shape[0], 10), dtype=np.float32)
			self.test_label = np.zeros((self.raw_test_label.shape[0], 10), dtype=np.float32)
			self.train_label[np.arange(self.train_label.shape[0]), self.raw_train_label] = 1
			self.test_label[np.arange(self.test_label.shape[0]), self.raw_test_label] = 1

			np.save(self.dataset_path+data_type+'train_data.npy', self.train_data)
			np.save(self.dataset_path+data_type+'train_label.npy', self.train_label)
			np.save(self.dataset_path+data_type+'test_data.npy', self.test_data)
			np.save(self.dataset_path+data_type+'test_label.npy', self.test_label)

		end = time.time()
		print('Loaded cifar10 data. Time: ', (end-start))
		
		if self.train_subsample_size is None:
			self.n_train_examples = self.train_data.shape[0]
		else:
			self.n_train_examples = min(self.train_data.shape[0], self.train_subsample_size)
			self.train_data = self.train_data[:self.n_train_examples, ...]
			self.train_label = self.train_label[:self.n_train_examples, ...]
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
												 # 'image': [{'dist': 'cont', 'name': 'Digit Image', 'size': tuple([self.batch_size, self.time_steps, self.image_size, self.image_size, 3])}]},
												 'image': [{'dist': 'bern', 'name': 'Digit Image', 'size': tuple([self.batch_size, self.time_steps, self.image_size, self.image_size, 3])}]},
								  'data':       {'flat': None,
								 		  	     'image': None}}

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






