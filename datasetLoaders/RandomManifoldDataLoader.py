import pdb
import numpy as np
import math
import scipy.misc
import pickle
import zlib
import os
import time
import helper
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets.samples_generator import make_swiss_roll

class DataLoader:
	def __init__(self, batch_size, time_steps, data_type='manifold', n_dims = 2,  cuda=False):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.data_type = data_type 
		self.n_dims = n_dims
		self.cuda = cuda
		self.mode = 'Train'
		self.iter = 0

		if self.data_type == 'manifold':
			self.dataset = None
			manifolds = [(30, 0.1, [0, 0, 0]), (-45, 0.1, [1, 2, 0]), (0, 0.1, [0, 0, 0])]
			if 15000 % len(manifolds) != 0: pdb.set_trace()
			for it, (degree, sigma, mu) in enumerate(manifolds):
				degreeRad = float(degree)*np.pi/180.

				mean = np.asarray(mu).astype(np.float32)[np.newaxis, :]
				centeredG = np.random.randn(int((15000+15000)/len(manifolds)), 3).astype(np.float32)
				centeredG[:, 0] = centeredG[:, 0]*2
				centeredG[:, 1] = np.tanh(centeredG[:, 0])+centeredG[:, 1]*sigma
				centeredG[:, 2] = centeredG[:, 2]*sigma

				rotation_mat = np.asarray([np.cos(degreeRad), np.sin(degreeRad), -np.sin(degreeRad), np.cos(degreeRad)]).reshape(2, 2)
				rotated_centered = centeredG
				centeredG[:,:2] = np.matmul(centeredG[:,:2], rotation_mat) 

				curr_samples = mean+rotated_centered
				if self.dataset is None: self.dataset = curr_samples
				else: self.dataset = np.concatenate([self.dataset, curr_samples], axis=0)

			if self.n_dims == 2:
				self.dataset = self.dataset[:, :2]						
			rand_index = np.arange(self.dataset.shape[0])
			rand_index = np.random.permutation(rand_index)
			self.dataset = self.dataset[rand_index, ...]
			self.dataset = self.dataset-1
			self.dataset = self.dataset*1/3.
			if self.dataset.shape[0] != 30000: pdb.set_trace()
		
		elif self.data_type == 'linear_degenerate_manifold' or self.data_type == 'nonlinear_degenerate_manifold':

			self.dataset = None
			manifolds = [(30, 0, [0, 0, 0]), (-45, 0, [1, 2, 0]), (0, 0, [0, 0, 0])]
			if 15000 % len(manifolds) != 0: pdb.set_trace()
			for it, (degree, sigma, mu) in enumerate(manifolds):
				degreeRad = float(degree)*np.pi/180.

				mean = np.asarray(mu).astype(np.float32)[np.newaxis, :]
				centeredG = np.random.randn(int((15000+15000)/len(manifolds)), 3).astype(np.float32)
				centeredG[:, 0] = centeredG[:, 0]*2
				centeredG[:, 1] = np.tanh(centeredG[:, 0])+centeredG[:, 1]*sigma
				centeredG[:, 2] = centeredG[:, 2]*sigma

				rotation_mat = np.asarray([np.cos(degreeRad), np.sin(degreeRad), -np.sin(degreeRad), np.cos(degreeRad)]).reshape(2, 2)
				rotated_centered = centeredG
				centeredG[:,:2] = np.matmul(centeredG[:,:2], rotation_mat) 

				curr_samples = mean+rotated_centered
				if self.dataset is None: self.dataset = curr_samples
				else: self.dataset = np.concatenate([self.dataset, curr_samples], axis=0)

			if self.n_dims == 2:
				self.dataset = self.dataset[:, :2]						
			rand_index = np.arange(self.dataset.shape[0])
			rand_index = np.random.permutation(rand_index)
			self.dataset = self.dataset[rand_index, ...]
			self.dataset = self.dataset-1
			self.dataset = self.dataset*1/3.
			if self.dataset.shape[0] != 30000: pdb.set_trace()

			# helper.dataset_plotter([self.dataset,], show_also=True)

		# elif self.data_type == 'linear_degenerate_manifold' or self.data_type == 'nonlinear_degenerate_manifold':

		# 	self.dataset = None
		# 	manifolds = [(30, 0.3, [0, 0, 0]), (-45, 0.3, [1, 2, 0]), (0, 0.3, [0, 0, 0])]
		# 	if 15000 % len(manifolds) != 0: pdb.set_trace()
		# 	for it, (degree, sigma, mu) in enumerate(manifolds):
		# 		degreeRad = float(degree)*np.pi/180.

		# 		mean = np.asarray(mu).astype(np.float32)[np.newaxis, :]
		# 		centeredG = np.random.randn(int((15000+15000)/len(manifolds)), 3).astype(np.float32)
		# 		centeredG[:, 0] = centeredG[:, 0]*2
		# 		centeredG[:, 1] = np.tanh(centeredG[:, 0])+centeredG[:, 1]*sigma
		# 		centeredG[:, 2] = centeredG[:, 2]*sigma

		# 		rotation_mat = np.asarray([np.cos(degreeRad), np.sin(degreeRad), -np.sin(degreeRad), np.cos(degreeRad)]).reshape(2, 2)
		# 		rotated_centered = centeredG
		# 		centeredG[:,:2] = np.matmul(centeredG[:,:2], rotation_mat) 

		# 		curr_samples = mean+rotated_centered
		# 		if self.dataset is None: self.dataset = curr_samples
		# 		else: self.dataset = np.concatenate([self.dataset, curr_samples], axis=0)

		# 	if self.n_dims == 2:
		# 		self.dataset = self.dataset[:, :2]						
		# 	rand_index = np.arange(self.dataset.shape[0])
		# 	rand_index = np.random.permutation(rand_index)
		# 	self.dataset = self.dataset[rand_index, ...]
		# 	self.dataset = self.dataset-1
		# 	self.dataset = self.dataset*1/3.
		# 	if self.dataset.shape[0] != 30000: pdb.set_trace()
			
		# 	if self.data_type == 'linear_degenerate_manifold':
		# 		self.dataset = np.concatenate([self.dataset, np.zeros((self.dataset.shape[0],1))], axis=-1)
		# 	elif self.data_type == 'nonlinear_degenerate_manifold':
		# 		self.dataset = np.concatenate([self.dataset, -np.tanh(self.dataset[:,0,np.newaxis])], axis=-1)

		# 	helper.dataset_plotter([self.dataset,], show_also=True)

		# 	pdb.set_trace()

		# helper.dataset_plotter([self.dataset,])
		# pdb.set_trace()

		self.train()
		self.reset()
		self.batch = {}
		self.batch['context'] = {'properties': {'flat': [], 'image': []},
									 'data':       {'flat': None, 'image': None}}
		self.batch['observed'] = {'properties': {'flat': [{'dist': 'cont', 'name': 'Toy Data', 'size': tuple([self.batch_size, self.time_steps, self.n_dims])}], 
												 'image': []},
								  'data':       {'flat': None, 'image': None}}
	
	def reset(self):
		if self.mode == 'Train':
			self.curr_loader = self.dataset[:15000, ...]
			rand_index = np.arange(self.curr_loader.shape[0])
			rand_index = np.random.permutation(rand_index)
			self.curr_loader = self.curr_loader[rand_index, ...]
			self.max_iter = 15000/(self.batch_size*self.time_steps)
		elif self.mode == 'Test':
			self.curr_loader = self.dataset[15000:, ...]
			rand_index = np.arange(self.curr_loader.shape[0])
			rand_index = np.random.permutation(rand_index)
			self.curr_loader = self.curr_loader[rand_index, ...]
			self.max_iter = 15000/(self.batch_size*self.time_steps)

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
		
		obs = self.curr_loader[self.iter*(self.batch_size*self.time_steps):(self.iter+1)*(self.batch_size*self.time_steps), ...]
		self.batch['observed']['data']['flat'] = obs.reshape(self.batch_size, self.time_steps, self.n_dims)

		self.iter += 1
		return self.iter, self.batch_size, self.batch 






