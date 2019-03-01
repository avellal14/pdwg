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
	def __init__(self, batch_size, time_steps, data_type='regular', b_context=False, cuda=False):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.mode = 'Train'
		self.cuda = cuda
		self.b_context = b_context
		self.iter = 0

		# if data_type == 'regular':
		# 	self.dataset_path = './datasetLoaders/fashionMnist/'
		
		if data_type == 'regular':
			self.dataset_path = '~/datasets/fashionMnist/'


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
												 'image': [{'dist': 'bern', 'name': 'Digit Image', 'size': tuple([self.batch_size, self.time_steps, 28, 28, 1])}]},
								  'data':       {'flat': None,
								 		  	     'image': None}}
	
	def reset(self):
		if self.mode == 'Train':
			self.curr_loader = self.dataset.train
			self.max_iter = self.dataset.train.num_examples/(self.batch_size*self.time_steps)
		elif self.mode == 'Test':
			self.curr_loader = self.dataset.test
			self.max_iter = self.dataset.test.num_examples/(self.batch_size*self.time_steps)

	def train(self):
		self.mode = 'Train'
		self.reset()

	def eval(self):
		self.mode = 'Test'
		self.reset()

	def __iter__(self):
		return self
	
	def __next__(self):
		if self.iter == int(self.max_iter): 
			self.iter = 0
			self.reset()
			raise StopIteration		
		
		obs, context = self.curr_loader.next_batch(self.batch_size*self.time_steps)
		if self.b_context: self.batch['context']['data']['flat'] = context.reshape(self.batch_size, self.time_steps, -1)
		self.batch['observed']['data']['image'] = obs.reshape(self.batch_size, self.time_steps, 28, 28, 1)

		self.iter += 1
		return self.iter, self.batch_size, self.batch 



