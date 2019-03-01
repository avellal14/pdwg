import pdb
import numpy as np
import math
import scipy.misc
import pickle
import zlib
import os
import time
import scipy.io as sio
# from tensorflow.examples.tutorials.mnist import input_data


class DataLoader:
	def __init__(self, batch_size, time_steps, data_type='regular', b_context=False, cuda=False):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.mode = 'Train'
		self.cuda = cuda
		self.b_context = b_context
		self.iter = 0
		self.num_labels = 200

		self.attribute_path = './datasetLoaders/FeatureAttribute/CUB/att_splits.mat'
		self.feature_path = './datasetLoaders/FeatureAttribute/CUB/res101.mat'

		att_mat_contents = sio.loadmat(self.attribute_path)
		feat_mat_contents = sio.loadmat(self.feature_path)

		label_att = att_mat_contents['att'].T
		features = feat_mat_contents['features'].T
		labels = feat_mat_contents['labels']
		attributes = label_att[labels-1,:][:,0,:]
		
		num_examples = int((features.shape[0]-features.shape[0]%50))
		features = features[:num_examples, ...]
		labels = labels[:num_examples, ...]
		labels_one_hot = np.zeros((labels.shape[0], self.num_labels), np.float32) 
		labels_one_hot[np.arange(labels.shape[0]), labels[:,0]-1] = 1
		attributes = attributes[:num_examples, ...]

		self.dataset = {'train': {'features': features[int(features.shape[0]/5):,...], 'labels': labels[int(labels.shape[0]/5):,...], 
								  'labels_one_hot': labels_one_hot[int(labels.shape[0]/5):,...], 'attributes': attributes[int(attributes.shape[0]/5):,...]},
						'test': {'features': features[:int(features.shape[0]/5),...], 'labels': labels[:int(labels.shape[0]/5),...], 
								 'labels_one_hot': labels_one_hot[:int(labels.shape[0]/5),...], 'attributes': attributes[:int(attributes.shape[0]/5),...]}}
		self.train()

		self.batch = {}
		self.batch['observed'] = {'properties': {'flat': [{'dist': 'cont', 'name': 'Resnet Features', 'size': tuple([self.batch_size, self.time_steps, self.dataset['train']['features'].shape[1]])}], 
												'image': {}},
								 'data':       {'flat': None,
											    'image': None}}
		
		self.batch['context'] = {'properties': {'image': {}, 
												 'flat': {'attributes': {'dist': 'cont', 'name': 'Attributes', 'size': tuple([self.batch_size, self.time_steps, self.dataset['train']['attributes'].shape[1]]) }, 
												 		  'labels': {'dist': 'cat', 'name': 'Labels', 'size': tuple([self.batch_size, self.time_steps, self.dataset['train']['labels_one_hot'].shape[1]])} }},
								  'data':       {'flat': None,
												 'image': None}}

	def reset(self):
		self._batch_obs = np.zeros((self.batch_size*self.time_steps, self.dataset['train']['features'].shape[1]))
		self._batch_attributes = np.zeros((self.batch_size*self.time_steps, self.dataset['train']['attributes'].shape[1]))
		self._batch_labels = np.zeros((self.batch_size*self.time_steps, self.dataset['train']['labels_one_hot'].shape[1]))

		if self.mode == 'Train':
			self.curr_loader = self.dataset['train']
			self.max_iter = self.dataset['train']['features'].shape[0]/(self.batch_size*self.time_steps)
		elif self.mode == 'Test':
			self.curr_loader = self.dataset['test']
			self.max_iter = self.dataset['test']['features'].shape[0]/(self.batch_size*self.time_steps)

	def train(self):
		self.mode = 'Train'
		self.reset()

	def eval(self):
		self.mode = 'Test'
		self.reset()

	def get_batch(self, iteration, curr_loader, batch_size):
		self._batch_obs[:] = curr_loader['features'][iteration*batch_size:(iteration+1)*batch_size, ...]
		self._batch_attributes[:] = curr_loader['attributes'][iteration*batch_size:(iteration+1)*batch_size, ...]
		self._batch_labels[:] = curr_loader['labels_one_hot'][iteration*batch_size:(iteration+1)*batch_size, ...]
		return self._batch_obs, self._batch_attributes, self._batch_labels

	def __iter__(self):
		return self
	
	def __next__(self):
		if self.iter == int(self.max_iter): 
			self.iter = 0
			self.reset()
			raise StopIteration		
		
		obs, attributes, labels = self.get_batch(self.iter, self.curr_loader, self.batch_size*self.time_steps)
		self.batch['context']['data']['flat'] = {'attributes': attributes.reshape(self.batch_size, self.time_steps, -1), 'labels': labels.reshape(self.batch_size, self.time_steps, -1)}
		self.batch['observed']['data']['flat'] = obs.reshape(self.batch_size, self.time_steps, -1)

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






