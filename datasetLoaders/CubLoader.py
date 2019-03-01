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
	def __init__(self, batch_size, time_steps, cuda = False, image_mode='aspect', image_size = 64):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.image_mode = image_mode
		self.mode = 'Train'
		self.cuda = cuda
		self.iter = 0
		self.all_to_test_ratio = 10
		self.image_size = image_size
		self.dataset_path = '/home/mcgemici/datasets/cub_data/CUB_200_2011/images_processed_'+self.image_mode+'_'+str(self.image_size)+'/'
		self._batch_observed_data_image = np.zeros((self.batch_size, self.time_steps, self.image_size, self.image_size, 3), np.float32)

		self.filename_array = []
		self.filename_dict = {}
		self.raw_file_names = glob.glob(self.dataset_path+'*/*.jpg')
		for filename in self.raw_file_names: self.filename_array.append(filename.split('/')[-2:])
		for a,b in self.filename_array:
			if a.split('.')[0] not in self.filename_dict: self.filename_dict[a.split('.')[0]] = {'name': a.split('.')[1]}
			if 'files' not in self.filename_dict[a.split('.')[0]]: self.filename_dict[a.split('.')[0]]['files']=[]
			self.filename_dict[a.split('.')[0]]['files'].append(b)
		self.n_classes = len(self.filename_dict)
		self.class_file_counts =[len(self.filename_dict[c]['files']) for c in self.filename_dict]

		self.n_examples = len(self.filename_array)
		self.n_test_examples = int(float(self.n_examples)/self.all_to_test_ratio)
		self.n_train_examples = self.n_examples-self.n_test_examples

		self.train_max_iter = np.floor(self.n_train_examples/(self.batch_size*self.time_steps))
		self.test_max_iter = np.floor(self.n_test_examples/(self.batch_size*self.time_steps))

		self.all_data = np.zeros((self.n_examples, self.image_size, self.image_size, 3), np.float32)
		self.all_labels = np.zeros((self.n_examples, 1), np.float32)
		
		print('Loading cub data')
		start = time.time()		
		for i, filename in enumerate(self.raw_file_names): 
			self.all_data[i,:,:,:] = scipy.misc.imread(filename)
			self.all_labels[i,:] = float(filename.split('/')[-2].split('.')[0])
		self.all_data = self.normalize(self.all_data) 
		end = time.time()
		print('Loaded cub data. Time: ', (end-start))
		
		self.all_data_order = np.arange(self.all_data.shape[0])
		self.all_data_order=np.random.permutation(self.all_data_order)
		self.all_data = self.all_data[self.all_data_order, ...]

		self.train_data = self.all_data[:self.n_train_examples,:,:,:]
		self.train_labels = self.all_labels[:self.n_train_examples,:]
		self.test_data = self.all_data[self.n_train_examples:,:,:,:]
		self.test_labels = self.all_data[self.n_train_examples:,:]

		self.batch = {}
		self.batch['context'] = {'properties': {'flat': [], 'image': []},
							     'data':       {'flat': None, 'image': None}}

		self.batch['observed'] = {'properties': {'flat': [], 
												 'image': [{'dist': 'bern', 'name': 'Face Image', 'size': tuple([self.batch_size, self.time_steps, self.image_size, self.image_size, 3])}]},
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

