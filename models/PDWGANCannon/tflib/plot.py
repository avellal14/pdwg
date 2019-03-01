import numpy as np
import pdb
import platform

import matplotlib
if platform.dist()[0] == 'Ubuntu': 
	print('On collab!')
else: 
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
# import cPickle as pickle
import _pickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
	_iter[0] += 1

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush():
	prints = []

	for name, vals in _since_last_flush.items():

		prints.append("{}\t{}".format(name, np.mean( list(vals.values()) ) ))
		_since_beginning[name].update(vals)

		x_vals = np.sort( list(_since_beginning[name].keys()) )
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name)
		plt.savefig(name.replace(' ', '_')+'_m.png')

	print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
	_since_last_flush.clear()

	with open('log.pkl', 'wb') as f:
		# pdb.set_trace()
		# pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)		
		pickle.dump(dict(_since_beginning), f, protocol=2)