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
from datasetLoaders.data_tread import returnQueue

class DataLoader:
	def __init__(self, batch_size, time_steps, cuda = False, data_path_pattern = './*.npz', image_size = 64, file_batch_size = 5000, file_refresh_rate = 100):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.mode = 'Train'
		self.cuda = cuda
		self.iter = 0
		self.curr_file_index = 0
		self.image_size = image_size
		# self.dataset_path = '/home/mcgemici/lsun_celebA_data/celebA_'+str(self.image_size)+'_mid/'
		self.dataset_path = '/home/mcgemici/lsun_celebA_data/celebA_'+str(self.image_size)+'_bilinear/'
		# self.dataset_path = '/home/mcgemici/lsun_celebA_data/celebA_'+str(self.image_size)+'_closeup/'
		self._batch_observed_data_image = np.zeros((self.batch_size, self.time_steps, self.image_size, self.image_size, 3), np.float32)

		self.file_batch_size = file_batch_size
		self.file_refresh_rate = file_refresh_rate


		self.train_path = self.dataset_path+'splits/train/*.jpg'
		self.test_path = self.dataset_path+'splits/test/*.jpg'
		self.valid_path = self.dataset_path+'splits/valid/*.jpg'
		
		self.train_files = glob.glob(self.train_path)
		self.test_files = glob.glob(self.test_path)
		self.valid_files = glob.glob(self.valid_path)



		self.n_train_examples = len(self.train_files)
		self.n_test_examples = len(self.test_files)
		self.n_valid_examples = len(self.valid_files)

		self.train_queue = returnQueue(self.train_path)
		self.test_queue = returnQueue(self.test_path)
		self.valid_queue = returnQueue(self.valid_path)
		self.curr_queue = self.train_queue

		self.train_max_iter = np.floor(self.n_train_examples/(self.batch_size*self.time_steps))
		self.test_max_iter = np.floor(self.n_test_examples/(self.batch_size*self.time_steps))
		self.valid_max_iter = np.floor(self.n_valid_examples/(self.batch_size*self.time_steps))
		self.curr_max_iter = self.train_max_iter

		self.batch = {}
		self.batch['context'] = {'properties': {'flat': [], 'image': []},
							     'data':       {'flat': None, 'image': None}}

		self.batch['observed'] = {'properties': {'flat': [], 
												 'image': [{'dist': 'bern', 'name': 'Face Image', 'size': tuple([self.batch_size, self.time_steps, self.image_size, self.image_size, 3])}]},
								  'data':       {'flat': None,
												 'image': None}}
		self.training_sampled = {}
		self.curr_file_batch = [None]*self.file_batch_size
		self.get_new_files()
		self.next_batch()

	def get_new_files(self):
		for i in range(self.file_batch_size): self.curr_file_batch[i] = self.curr_queue.get()

	def normalize(self, x, noise_ify=False):
		if noise_ify :
			return (x+np.random.uniform(low=-1.0, high=1.0, size=x.shape))/(255.)
		else:
			return (x)/(255.)

	def denormalize(self, x):
		return x*(255.)

	def next_batch(self):		
		for i in range(self.batch_size):
			for j in range(self.time_steps):
				rgb_int, filename = self.curr_file_batch[self.curr_file_index]
				self.curr_file_index += 1
				self._batch_observed_data_image[i,j,:,:,:] = rgb_int
				self._batch_observed_data_image[i,j,:,:,:] = self.normalize(self._batch_observed_data_image[i,j,:,:,:])
				if self.curr_file_index == self.file_batch_size: 
					self.curr_file_index = 0
					self.get_new_files()
					# print("\n\n\n\n\n getting new files  \n\n\n\n")

	def train(self):
		self.mode = 'Train'
		self.curr_queue = self.train_queue
		self.curr_max_iter = self.train_max_iter
		self.reset()

	def eval(self):
		self.mode = 'Test'
		self.curr_queue = self.test_queue
		self.curr_max_iter = self.test_max_iter
		self.reset()

	def reset(self):
		self.iter = 0
		# print("\n\n\n\n\n resetting  \n\n\n\n")
		self.train_queue = returnQueue(self.train_path)
		self.test_queue = returnQueue(self.test_path)
		self.valid_queue = returnQueue(self.valid_path)		

	def __iter__(self):
		return self
	
	def __next__(self):
		if self.iter == self.curr_max_iter: 
			# print("\n\n\n\n\n finished epoch  \n\n\n\n")
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

