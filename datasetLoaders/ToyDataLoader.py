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
	def __init__(self, batch_size, time_steps, data_type='circular_8_gaussians', n_dims = 2,  cuda=False):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.data_type = data_type 
		self.n_dims = n_dims
		self.cuda = cuda
		self.mode = 'Train'
		self.iter = 0

		if self.data_type == 'swiss_roll':
			self.dataset, _ = make_swiss_roll(15000+15000, 0.2)
			if self.n_dims == 3:
				self.dataset = self.dataset.astype(np.float32)/7.
			if self.n_dims == 2:
				self.dataset = self.dataset[:, [0, 2]].astype(np.float32)/7.
		elif self.data_type == 'grid_25_gaussians':
			sigma = 0.05
			self.dataset = None
			for i in range(-2, 3):
				for j in range(-2, 3):
					for k in range(-2, 3):
						mean = np.asarray([i, j, k]).astype(np.float32)[np.newaxis, :]
						curr_samples = mean+np.random.randn(120+120, 3).astype(np.float32)*sigma
						if self.dataset is None: self.dataset = curr_samples
						else: self.dataset = np.concatenate([self.dataset, curr_samples], axis=0)
			if self.n_dims == 2:
				self.dataset = self.dataset[:, :2]						
			rand_index = np.arange(self.dataset.shape[0])
			rand_index = np.random.permutation(rand_index)
			self.dataset = self.dataset[rand_index, ...]
		elif self.data_type == 'grid_9_gaussians':
			sigma = 0.05
			self.dataset = None
			for i in [-1, 0, 1]:
				for j in [-1, 0, 1]:
					for k in [-1, 0, 1]:
						mean = np.asarray([i, j, k]).astype(np.float32)[np.newaxis, :]
						curr_samples = mean+np.random.randn(550+550, 3).astype(np.float32)*sigma
						if self.dataset is None: self.dataset = curr_samples
						else: self.dataset = np.concatenate([self.dataset, curr_samples], axis=0)
			curr_samples = np.random.randn(150+150, 3).astype(np.float32)
			self.dataset = np.concatenate([self.dataset, curr_samples], axis=0) 			
			if self.n_dims == 2:
				self.dataset = self.dataset[:, :2]						
			rand_index = np.arange(self.dataset.shape[0])
			rand_index = np.random.permutation(rand_index)
			self.dataset = self.dataset[rand_index, ...]
		elif self.data_type == 'circular_8_gaussians':
			sigma = 0.05
			self.dataset = None
			for (i, j) in [[-1.6, 0], [-1, 1], [-1, -1], [0, 1.4], [0, -1.4], [1, 1], [1, -1], [1.6, 0]]:
				for k in [-1, 1]:
					mean = np.asarray([i, j, k]).astype(np.float32)[np.newaxis, :]
					curr_samples = mean+np.random.randn(1875, 3).astype(np.float32)*sigma
					if self.dataset is None: self.dataset = curr_samples
					else: self.dataset = np.concatenate([self.dataset, curr_samples], axis=0)
			if self.n_dims == 2:
				self.dataset = self.dataset[:, :2]						
			rand_index = np.arange(self.dataset.shape[0])
			rand_index = np.random.permutation(rand_index)
			self.dataset = self.dataset[rand_index, ...]
		elif self.data_type == 'star_8_gaussians':
			sigma = 0.25
			self.dataset = None
			for it, (i, j) in enumerate([[1.6, 0], [1, 1], [0, 1.4], [-1, 1], [-1.6, 0], [-1, -1], [0, -1.4], [1, -1]]):
				for k in [-1, 1]:
					mean = np.asarray([i, j, k]).astype(np.float32)[np.newaxis, :]
					rand_part = np.random.randn(1875, 3).astype(np.float32)
					rand_part[:, 1] = rand_part[:, 1]*sigma
					rand_part[:, 0] = rand_part[:, 0]*0.5
					rotation_mat = np.asarray([np.cos(it*np.pi/4), np.sin(it*np.pi/4), -np.sin(it*np.pi/4), np.cos(it*np.pi/4)]).reshape(2, 2)
					rand_part[:,:2] = np.matmul(rand_part[:,:2], rotation_mat)
					curr_samples = mean+rand_part
					if self.dataset is None: self.dataset = curr_samples
					else: self.dataset = np.concatenate([self.dataset, curr_samples], axis=0)
			if self.n_dims == 2:
				self.dataset = self.dataset[:, :2]
			rand_index = np.arange(self.dataset.shape[0])
			rand_index = np.random.permutation(rand_index)
			self.dataset = self.dataset[rand_index, ...]
			# self.dataset = self.dataset*2
		elif self.data_type == 'circular_3_gaussians':
			sigma = 0.3
			self.dataset = None
			# for (i, j) in [[-1.6, 0], [-1, 1], [-1, -1], [0, 1.4], [0, -1.4], [1, 1], [1, -1], [1.6, 0]]:
			for (i, j) in [[-1.6, 0], [1, 1], [-1, -1]]:
				for k in [-1, 1]:
					mean = np.asarray([i, j, k]).astype(np.float32)[np.newaxis, :]
					curr_samples = mean+np.random.randn(5000, 3).astype(np.float32)*sigma
					if self.dataset is None: self.dataset = curr_samples
					else: self.dataset = np.concatenate([self.dataset, curr_samples], axis=0)
			if self.n_dims == 2:
				self.dataset = self.dataset[:, :2]						
			rand_index = np.arange(self.dataset.shape[0])
			rand_index = np.random.permutation(rand_index)
			self.dataset = self.dataset[rand_index, ...]

		elif self.data_type == 'linear_degenerate_circular_3_gaussians' or self.data_type == 'nonlinear_degenerate_circular_3_gaussians':
			sigma = 1
			degree1 = 30
			degree2 = 45
			self.dataset = None
			# for (i, j) in [[-1.6, 0], [-1, 1], [-1, -1], [0, 1.4], [0, -1.4], [1, 1], [1, -1], [1.6, 0]]:
			for (i, j) in [[-1.6, 0], [2, 2], [-2, -2]]:
				for k in [-1, 1]:
					mean = np.asarray([i, j, k]).astype(np.float32)[np.newaxis, :]
					curr_samples = mean+np.random.randn(5000, 3).astype(np.float32)*sigma
					if self.dataset is None: self.dataset = curr_samples
					else: self.dataset = np.concatenate([self.dataset, curr_samples], axis=0)
			if self.n_dims == 2:
				self.dataset = self.dataset[:, :2]						

			if self.data_type == 'linear_degenerate_circular_3_gaussians':
				self.dataset = np.concatenate([self.dataset, np.zeros((self.dataset.shape[0],1))], axis=-1)
			elif self.data_type == 'nonlinear_degenerate_circular_3_gaussians':
				self.dataset = np.concatenate([self.dataset, 8*np.tanh(self.dataset[:,0,np.newaxis])], axis=-1)

			# -60, 30 to -13 18  45 12
			degree1 = 15
			degree2 = 0		
			degree3 = -30			
			degreeRad1 = float(degree1)*np.pi/180.
			degreeRad2 = float(degree2)*np.pi/180.
			degreeRad3 = float(degree3)*np.pi/180.
			rotation_mat1 = np.asarray([np.cos(degreeRad1), np.sin(degreeRad1), 0, -np.sin(degreeRad1), np.cos(degreeRad1), 0, 0, 0, 1]).reshape(3, 3)
			rotation_mat2 = np.asarray([1, 0, 0, 0, np.cos(degreeRad2), np.sin(degreeRad2), 0, -np.sin(degreeRad2), np.cos(degreeRad2)]).reshape(3, 3)
			rotation_mat3 = np.asarray([np.cos(degreeRad3), 0, np.sin(degreeRad3), 0, 1, 0, -np.sin(degreeRad3), 0, np.cos(degreeRad3)]).reshape(3, 3)
			
			self.dataset = np.matmul(self.dataset, rotation_mat1) 
			self.dataset = np.matmul(self.dataset, rotation_mat2) 
			self.dataset = np.matmul(self.dataset, rotation_mat3) 

			rand_index = np.arange(self.dataset.shape[0])
			rand_index = np.random.permutation(rand_index)
			self.dataset = self.dataset[rand_index, ...]

			self.n_dims = 3
		else:
			pdb.set_trace()
		self.dataset = self.dataset*0.5
		
		# helper.dataset_plotter([self.dataset,], show_also=True)
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






