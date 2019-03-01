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
import glob

class DataLoader:
	def __init__(self, batch_size, time_steps, cuda = False, image_mode='cubic', image_size = 64):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.mode = 'Train'
		self.cuda = cuda
		self.iter = 0
		self.image_size = image_size
		self.all_to_test_ratio = 5

		self.dataset_path = '/home/mcgemici/datasets/cat_data/cats_'+str(image_size)+'_'+image_mode+'/'
		self._batch_observed_data_image = np.zeros((self.batch_size, self.time_steps, self.image_size, self.image_size, 3), np.float32)
		
		self.all_files = glob.glob(self.dataset_path+'*.jpg')
		self.n_examples = min(len(self.all_files), 250000)
		self.all_files = self.all_files[:self.n_examples]
		
		self.all_data_order = np.arange(self.all_data.shape[0])
		self.all_data_order=np.random.permutation(self.all_data_order)
		pdb.set_trace()
		self.all_files = self.all_files[self.all_data_order.aslist()]


		self.n_test_examples = int(float(self.n_examples)/self.all_to_test_ratio)
		self.n_train_examples = self.n_examples-self.n_test_examples

		self.train_files = self.all_files[:self.n_train_examples]
		self.test_files = self.all_files[self.n_train_examples:]

		self.train_max_iter = np.floor(self.n_train_examples/(self.batch_size*self.time_steps))
		self.test_max_iter = np.floor(self.n_test_examples/(self.batch_size*self.time_steps))

		self.train_data = np.zeros((self.n_train_examples, self.image_size, self.image_size, 3), np.uint8)
		self.test_data = np.zeros((self.n_test_examples, self.image_size, self.image_size, 3), np.uint8)

		print('Loading cat data')
		start = time.time()
		reset_data = True
		try: 
			assert(not reset_data)
			print('Trying to load from processed train file.')
			self.train_data = np.load(self.dataset_path+'train.npy')
			print('Success loading train data from processed test file.')
		except:
			print('Failed. Creating processed file for train.')
			start_indiv = time.time()
			for i, filename in enumerate(self.train_files): 
				if i % 10000 == 0: 
					end_indiv = time.time()
					print(i, len(self.train_files), end_indiv-start_indiv)
					start_indiv = time.time()
				self.train_data[i,:,:,:] = scipy.misc.imread(filename)
			np.save(self.dataset_path+'train.npy', self.train_data)
		
		try: 
			assert(not reset_data)
			print('Trying to load from processed test file.')
			self.test_data = np.load(self.dataset_path+'test.npy')
			print('Success loading test data from processed test file.')
		except:
			print('Failed. Creating processed file for test.')
			start_indiv = time.time()
			for i, filename in enumerate(self.test_files): 
				if i % 10000 == 0: 
					end_indiv = time.time()
					print(i, len(self.test_files), end_indiv-start_indiv)
					start_indiv = time.time()
				self.test_data[i,:,:,:] = scipy.misc.imread(filename)
			np.save(self.dataset_path+'test.npy', self.test_data)

		self.train_data = self.train_data.astype(np.float32)
		self.test_data = self.test_data.astype(np.float32)
		self.train_data = self.normalize(self.train_data) 
		self.test_data = self.normalize(self.test_data) 

		end = time.time()
		print('Loaded cat data. Time: ', (end-start))

		self.batch = {}
		self.batch['context'] = {'properties': {'flat': [], 'image': []},
							     'data':       {'flat': None, 'image': None}}

		self.batch['observed'] = {'properties': {'flat': [], 
												 'image': [{'dist': 'cont', 'name': 'Face Image', 'size': tuple([self.batch_size, self.time_steps, self.image_size, self.image_size, 3])}]},
								  'data':       {'flat': None,
												 'image': None}}
		self.train()
	
	def normalize(self, x, noise_ify=False):
		if noise_ify :
			return (x+np.random.uniform(low=-1.0, high=1.0, size=x.shape))/(255.)
		else:
			return (x)/(255.)

	def denormalize(self, x):
		return x*(255.)

	def train(self, randomize=True):
		self.mode = 'Train'
		self.curr_data_order = np.arange(self.train_data.shape[0])
		if randomize: self.curr_data_order=np.random.permutation(self.curr_data_order)
		self.curr_data = self.train_data[self.curr_data_order, ...]
		self.curr_max_iter = self.train_max_iter
		self.reset()

	def eval(self, randomize=False):
		self.mode = 'Test'
		self.curr_data_order = np.arange(self.test_data.shape[0])
		if randomize: self.curr_data_order=np.random.permutation(self.curr_data_order)
		self.curr_data = self.test_data[self.curr_data_order, ...]
		self.curr_max_iter = self.test_max_iter
		self.reset()

	def reset(self):
		self.iter = 0
	
	def next_batch(self):
		self._batch_observed_data_image = \
			self.curr_data[self.iter*self.batch_size*self.time_steps: (self.iter+1)*self.batch_size*self.time_steps,:,:,:].reshape(\
			self.batch_size, self.time_steps, self.image_size, self.image_size, 3)

	# def next_batch(self):
	# 	self._batch_observed_data_image = \
	# 		self.curr_data[self.iter*self.batch_size*self.time_steps: (self.iter+1)*self.batch_size*self.time_steps,:,:,:].reshape(\
	# 		self.batch_size, self.time_steps, self.image_size, self.image_size, 3).astype(np.float32)
	# 	self._batch_observed_data_image = self.normalize(self._batch_observed_data_image)

	def __iter__(self):
		return self
	
	def __next__(self):
		if self.iter == self.curr_max_iter: 
			raise StopIteration
		
		self.next_batch()
		self.batch['observed']['data']['image'] = self._batch_observed_data_image

		self.iter += 1
		return self.iter-1, self.batch_size, self.batch 
	


		# import time
		# allfiles = glob.glob(self.dataset_path+'splits/train/*.jpg')
		# alltimes = 0
		# for i, f in enumerate(allfiles):
		# 	start = time.time()
		# 	scipy.misc.imread(f)
		# 	end = time.time()
		# 	print(i,len(allfiles), (end-start))
		# 	alltimes = alltimes+(end-start)


		# # allfiles = glob.glob('/home/mcgemici/lsun_celebA_data/celebA_64_bilinear/splits/train/*.jpg')
		# # alltimes2 = 0
		# # for i, f in enumerate(allfiles):
		# # 	start = time.time()
		# # 	scipy.misc.imread(f)
		# # 	end = time.time()
		# # 	print(i,len(allfiles), (end-start))
		# # 	alltimes2 = alltimes2+(end-start)
		# # # print('\n\n', alltimes)
		# # # print('\n\n')

		# # print('\n\n', alltimes2)
		# # print('\n\n')
		# pdb.set_trace()
		

		
		# import time
		# allfiles = glob.glob('/home/mcgemici/lsun_celebA_data/celebA_32_mid/splits/train/*.jpg')
		# alltimes = 0
		# for i, f in enumerate(allfiles):
		# 	start = time.time()
		# 	scipy.misc.imread(f)
		# 	end = time.time()
		# 	print(i,len(allfiles), (end-start))
		# 	alltimes = alltimes+(end-start)


		# allfiles = glob.glob('/home/mcgemici/lsun_celebA_data/celebA_64_bilinear/splits/train/*.jpg')
		# alltimes2 = 0
		# for i, f in enumerate(allfiles):
		# 	start = time.time()
		# 	scipy.misc.imread(f)
		# 	end = time.time()
		# 	print(i,len(allfiles), (end-start))
		# 	alltimes2 = alltimes2+(end-start)
		# # print('\n\n', alltimes)
		# # print('\n\n')

		# print('\n\n', alltimes2)
		# print('\n\n')
		# pdb.set_trace()

