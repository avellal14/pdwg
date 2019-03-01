
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import pdb
import scipy 
from os import listdir
import os
import os.path
import glob
from os.path import isfile, join
import subprocess
import shutil

print('\n\n\n')
list_of_exp_folders = ['./experimentsCELEB/',]
experiments_to_preserve = ['d52ac346e9eb4330a650218c6d0cf9d1', 
						   '8069d73192fd44dd868d84c4c2c1aac6', 
						   '46b9de3f958d4cc0b212c560567f39b4', 
						   'e4a1317f77f5420389a66d5eba21649e', 
						   '1b764d2089504c959d3a5a98ee05760b',
						   '00e2ec7f317548c6addb65e9e05c5393', 
						   '49b7121cd48c45e0884b1778ca840d96',
						   '85cf79b6ff964883b030bfb4050d9d43',
						   'a7a7503456fb41c28397ef2e973760a1',
						   'e40c27d4ae0d4bf3aa3a849f457820e8',
						  ]
all_experiments = []
for e in list_of_exp_folders:
	all_experiments = all_experiments+glob.glob(e+'*')

experiment_indeces_to_preserve = []
experiment_indeces_to_remove = []
for ind in range(len(all_experiments)):
	in_preserve = False
	for preserve in experiments_to_preserve:
		if preserve in all_experiments[ind]:
			experiment_indeces_to_preserve.append(ind)
			in_preserve = True
	if not in_preserve: experiment_indeces_to_remove.append(ind)
for ind in experiment_indeces_to_preserve: 
	print('Preserve :', all_experiments[ind])

print('\n\n\n')
print('*************************************************')
print('*************************************************')
print('*************************************************\n\n\n')
print('\n\n\n')

for ind in experiment_indeces_to_remove: 
	print('Removing :', all_experiments[ind])
	try: shutil.rmtree(all_experiments[ind]) 
	except: print('Failed :', all_experiments[ind])

