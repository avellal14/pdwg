import pdb
import numpy as np
import math
import scipy.misc
import pickle
import zlib
import os
import time
from tensorflow.examples.tutorials.mnist import input_data


class DataLoader:
	def __init__(self, batch_size, time_steps, data_type='regular', b_context=True, cuda=False):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.mode = 'Train'
		self.cuda = cuda
		self.b_context = b_context
		self.iter = 0
		self.image_size = 28

		# if data_type == 'regular':
		# 	self.dataset_path = './datasetLoaders/mnist/dataset/'
		# elif data_type == 'binary':
		# 	self.dataset_path = './datasetLoaders/mnist/dataset_binary/'
		# elif data_type == 'rotation':
		# 	self.dataset_path = './datasetLoaders/mnist/dataset_rotation/'

		if data_type == 'regular':
			self.dataset_path = '~/datasets/mnist/dataset/'
		elif data_type == 'binary':
			self.dataset_path = '~/datasets/mnist/dataset_binary/'
		elif data_type == 'rotation':
			self.dataset_path = '~/datasets/mnist/dataset_rotation/'

		self.dataset = input_data.read_data_sets(self.dataset_path, one_hot=True)
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
												 'image': [{'dist': 'bern', 'name': 'Digit Image', 'size': tuple([self.batch_size, self.time_steps, self.image_size, self.image_size, 1])}]},
												 # 'image': [{'dist': 'cont', 'name': 'Digit Image', 'size': tuple([self.batch_size, self.time_steps, self.image_size, self.image_size, 1])}]},
								  'data':       {'flat': None,
								 		  	     'image': None}}

	def train(self):
		self.mode = 'Train'
		self.curr_loader = self.dataset.train
		self.curr_max_iter = self.dataset.train.num_examples/(self.batch_size*self.time_steps)
		self.reset()

	def eval(self):
		self.mode = 'Test'
		self.curr_loader = self.dataset.test
		self.curr_max_iter = self.dataset.test.num_examples/(self.batch_size*self.time_steps)
		self.reset()

	def reset(self):
		self._batch_obs = np.zeros((self.batch_size*self.time_steps, self.image_size*self.image_size, 3))
		self.iter = 0

	def __iter__(self):
		return self
	
	def __next__(self):
		if self.iter == int(self.curr_max_iter): 
			raise StopIteration		
		
		obs, context = self.curr_loader.next_batch(self.batch_size*self.time_steps, shuffle=True)
		if self.b_context: self.batch['context']['data']['flat'] = context.reshape(self.batch_size, self.time_steps, -1)
		self.batch['observed']['data']['image'] = obs.reshape(self.batch_size, self.time_steps, self.image_size, self.image_size, 1)

		self.iter += 1
		return self.iter, self.batch_size, self.batch 











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






