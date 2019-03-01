import sys

import pdb
import numpy as np
import math
import scipy.misc
import pickle
import zlib
import os
import uuid
import time
from datasetLoaders.data_tread import returnQueue

def gather_data(start_inds, episode_length, data, set_value = None):
	start_inds_np = np.asarray(start_inds, dtype=int)
	all_inds_np = np.tile(start_inds_np[:,np.newaxis], [1, episode_length])+np.arange(episode_length)
	# if set_value is not None: 
	# 	if (all_inds_np[:,-1]>=data.shape[0]).sum()>0: pdb.set_trace()
	# 	data[all_inds_np, ...] = set_value
	return data[all_inds_np, ...]

def sample_wo_rep2(available_mask, sample_size, time_steps):
	num_items = available_mask.shape[0]
	available_mask[:time_steps] = False
	# available_mask[-time_steps:] = False
	arange  = np.arange(num_items)
	try: end_indeces = np.random.choice(arange[available_mask], size=sample_size, replace=False, p=None)
	except: pdb.set_trace()

	# gather_data(end_indeces-time_steps+1, 2*time_steps, available_mask, set_value = False)
	return available_mask, end_indeces

def extract_from_data_dict(data_dict, start_indeces, time_steps = 1):
	keys = np.asarray([e.split('.') for e in list(data_dict.keys())])
	context_image = []
	context_flat = []
	obs_flat = []
	obs_types = []

	for k in range(keys.shape[0]):

		gathered = gather_data(start_indeces, time_steps, curr_data)
		if keys[k, 0] == 'context' and keys[k, 1] == 'image':
			# context_image.append(np.transpose(gathered, axes=[0,1,4,2,3]))
			context_image.append(gathered)
		if keys[k, 0] == 'context' and keys[k, 1] == 'flat': 
			context_flat.append(gathered.reshape(gathered.shape[0], gathered.shape[1], -1))
		if keys[k, 0] == 'observed' and keys[k, 1] == 'flat': 
			obs_flat.append(gathered.reshape(gathered.shape[0], gathered.shape[1], -1))
			obs_types.append((*keys[k, :], gathered.shape))
	
	# preprocess(context_image, context_flat, obs_flat, keys)
	context_image_concat, context_flat_concat, obs_flat_concat = None, None, None
	if len(context_image) > 0:
		context_image_concat = np.concatenate(context_image, axis = -1)
	if len(context_flat) > 0:
		context_flat_concat = np.concatenate(context_flat, axis = -1)
	if len(obs_flat) > 0:
		obs_flat_concat = np.concatenate(obs_flat, axis = -1)
	return context_image_concat, context_flat_concat, obs_flat_concat, obs_types

def get_prioritized_samples(arr, priority, batch_size, time_steps, sampled):
	effective_batchs = [0]*(max(priority)+1)
	end_indeces = [[]]*(max(priority)+1)
	for i, dim in enumerate(priority):
		filtered = (arr[..., dim]==True)

		if i == len(priority)-1: effective_batchs[dim] = batch_size - np.sum(effective_batchs)
		else: effective_batchs[dim] = min(math.floor(batch_size/2.), filtered.sum()) 
		
		if effective_batchs[dim] > 0: 
			updated_available, end_indeces[dim] = sample_wo_rep2(filtered, effective_batchs[dim], time_steps)
			np.copyto(sampled, (sampled+(updated_available==False))) 
		else: end_indeces[dim] = []

	return np.concatenate(end_indeces, axis=0)

def sample_from_single_file(data_dict, batch_size, time_steps, sampled, update_mode=False, mode = 'Train'):
	end_indeces = get_prioritized_samples(arr = data_dict[''], priority = [2, 0],\
										  batch_size = batch_size, time_steps = time_steps, sampled = sampled)

	if end_indeces.shape[0] == 0: return None, None, None, None
	start_indeces = end_indeces-time_steps+1
	np.random.shuffle(start_indeces)
	context_image, context_flat, obs, obs_types = extract_from_data_dict(data_dict, start_indeces, time_steps = time_steps)
	return context_image, context_flat, obs, obs_types

class DataLoader:
	def __init__(self, batch_size, time_steps, cuda = False, data_path_pattern = './*.npz', file_batch_size = 5, file_refresh_rate = 100, train_max_iter = 1000):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.mode = 'Train'
		self.cuda = cuda
		self.iter = 0
		self.queue = returnQueue(data_path_pattern)
		self.file_blacklist = []

		self.file_batch_size = file_batch_size
		self.file_refresh_rate = file_refresh_rate

		self.train_max_iter = train_max_iter
		self.max_iter = self.train_max_iter
		self.iter = 0
		
		self.batch = {}
		self.batch['context'] = {'properties': {'flat': [], 'image': []}, 'data': {'flat': None, 'image': None}}
		self.batch['observed'] = {'properties': {'flat': [], 'image': []}, 'data': {'flat': None, 'image': None}}

		self.training_sampled = {}
		self.curr_file_batch = [None]*self.file_batch_size
		self.get_new_files()
		self.next_batch()
	def get_new_files(self):
		for i in range(self.file_batch_size):
			self.curr_file_batch[i] = self.queue.get()
			while self.curr_file_batch[i][1] in self.file_blacklist: 
				self.curr_file_batch[i] = self.queue.get()

	def next_batch(self):		
		for i in range(self.file_batch_size):
			file_data_dict, filename = self.curr_file_batch[i]
			file_num_time_steps = file_data_dict[list(file_data_dict.keys())[0]].shape[0]
			if filename not in self.training_sampled: self.training_sampled[filename] = np.zeros((file_num_time_steps), dtype=bool)
			file_minibatch_size = math.ceil(self.batch_size/self.file_batch_size)
			context_image, context_flat, obs, obs_types = sample_from_single_file(file_data_dict, file_minibatch_size, self.time_steps, self.training_sampled[filename], True, self.mode)

			if self.batch['context']['data']['flat'] is None and context_flat is not None: 
				self._batch_context_data_flat = np.zeros((self.batch_size, self.time_steps, *context_flat.shape[2:]), dtype=np.float32)
				if self.cuda: self._batch_context_data_flat_tensor = self._batch_context_data_flat
				else: self._batch_context_data_flat_tensor = self._batch_context_data_flat
				self.batch['context']['data']['flat'] = (self._batch_context_data_flat_tensor)
				self.batch['context']['properties']['flat'] = [{'dist': 'cont', 'name': 'Ocean State', 'size': (self.batch_size, self.time_steps, *context_flat.shape[2:])}]

			if self.batch['context']['data']['image'] is None and context_image is not None: 
				self._batch_context_data_image = np.zeros((self.batch_size, self.time_steps, *context_image.shape[2:]), dtype=np.float32)
				if self.cuda: self._batch_context_data_image_tensor = self._batch_context_data_image
				else: self._batch_context_data_image_tensor = self._batch_context_data_image
				self.batch['context']['data']['image'] = (self._batch_context_data_image_tensor)
				self.batch['context']['properties']['image'] = [{'dist': 'cont', 'name': 'Ocean State Minimap', 'size': (self.batch_size, self.time_steps, *context_image.shape[2:])}]

			if self.batch['observed']['data']['flat'] is None and obs is not None: 
				self._batch_observed_data_flat = np.zeros((self.batch_size, self.time_steps, *obs.shape[2:]), dtype=np.float32)
				if self.cuda: self._batch_observed_data_flat_tensor = self._batch_observed_data_flat
				else: self._batch_observed_data_flat_tensor = self._batch_observed_data_flat
				self.batch['observed']['data']['flat'] = (self._batch_observed_data_flat_tensor)
				self.batch['observed']['properties']['flat'] = [{'dist': e[2], 'name': 'Ocean Action/'+e[3], 'size': (self.batch_size, self.time_steps, *e[4][2:])} for e in obs_types]				

			if context_flat is not None: self._batch_context_data_flat[i*file_minibatch_size:(i+1)*file_minibatch_size, ...] = context_flat			
			if context_image is not None: self._batch_context_data_image[i*file_minibatch_size:(i+1)*file_minibatch_size, ...] = context_image
			if obs is not None: self._batch_observed_data_flat[i*file_minibatch_size:(i+1)*file_minibatch_size, ...] = obs

			if self.cuda: 
				self._batch_context_data_flat_tensor = self._batch_context_data_flat
				self._batch_context_data_image_tensor = self._batch_context_data_image
				self._batch_observed_data_flat_tensor = self._batch_observed_data_flat
			
			self.batch['context']['data']['flat'] = self._batch_context_data_flat_tensor
			self.batch['context']['data']['image'] = self._batch_context_data_image_tensor
			self.batch['observed']['data']['flat']= self._batch_observed_data_flat_tensor


	def train(self):
		self.mode = 'Train'
		self.max_iter = self.train_max_iter

	def eval(self):
		self.mode = 'Test'
		self.max_iter = int(self.train_max_iter/10.0)

	def reset(self):
		self.iter = 0

	def __iter__(self):
		return self
	
	def __next__(self):
		if self.iter == self.max_iter: 
			self.iter = 0
			raise StopIteration
		
		if self.iter % self.file_refresh_rate == 0: self.get_new_files()
		self.next_batch()

		self.iter += 1
		return self.iter-1, self.batch_size, self.batch 
	

