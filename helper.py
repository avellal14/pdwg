
import pdb
import numpy as np
import math
import scipy
import scipy.misc
import matplotlib
from scipy.misc import imsave

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

matplotlib.use('TkAgg')

# import platform
# if platform.dist()[0] == 'centos':
# 	matplotlib.use('Agg')
# elif platform.dist()[0] == 'debian': 
# 	matplotlib.use('Agg')
# elif platform.dist()[0] == 'Ubuntu': 
# 	print('On Collab')
# # else: 
	# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import scipy.stats as st
import seaborn as sns
import pickle
import zlib
import os
import uuid
import glob
import time
import shutil, errno
import string 
import pdb
import copy
import skimage
from scipy import signal
from skimage import color

plt.rcParams['axes.linewidth'] = 2

# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# print_tensors_in_checkpoint_file(file_name=global_args.global_exp_dir+global_args.restore_dir+'/checkpoints/checkpoint', tensor_name='',all_tensors='')

class MyAxes3D(axes3d.Axes3D):

    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # set visibility of some features False 
        self.set_some_features_visibility(False)
        # draw the axes
        super(MyAxes3D, self).draw(renderer)
        # set visibility of some features True. 
        # This could be adapted to set your features to desired visibility, 
        # e.g. storing the previous values and restoring the values
        self.set_some_features_visibility(True)

        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        # disable draw grid
        zaxis.axes._draw_grid = False

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw :
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw :
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2], 
                             tmp_planes[1], tmp_planes[0], 
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes

        # disable draw grid
        zaxis.axes._draw_grid = draw_grid_old

def tf_print(input_tensor, list_of_print_tensors):
	input_tensor = tf.Print(input_tensor, list_of_print_tensors, message="Log values:")
	return input_tensor

def get_report_formatted(report, sess, curr_feed_dict):
	compute_list = []
	for e in report: 
		if e[2] is None: compute_list.append(e[1])
	computed_list = sess.run(compute_list, feed_dict = curr_feed_dict)

	report_value_list = []
	report_format = ''    
	curr_ind = 0
	for e in report: 

		if e[2] is None: 
			e[2] = computed_list[curr_ind]
			curr_ind += 1
		report_format = report_format + ' ' + e[0]
		report_value_list.append(e[2])
	return report_format[1:], report_value_list

def current_platform():
	from sys import platform
	return platform

def save_specs_file(exp_dir, global_args):
	filestring = 'Directory: '+exp_dir+ '\n\n Specs: \n\n'+ str(global_args)+'\n\n'
	with open(exp_dir+"Specs.txt", "w") as text_file:
	    text_file.write(filestring)

def get_exp_dir(global_args):
	# if global_args.restore:
	if False:
		exp_dir = global_args.global_exp_dir + '/' + global_args.restore_dir
	else:
		random_name = str(uuid.uuid4().hex)
		exp_dir = global_args.global_exp_dir + '/' + random_name
		if len(global_args.exp_dir_postfix)>0: exp_dir = exp_dir + '_' + global_args.exp_dir_postfix 
	exp_dir = exp_dir+ '/'
	print('\n\nEXPERIMENT RESULT DIRECTORY: '+ exp_dir + '\n\n')

	if not os.path.exists(exp_dir): os.makedirs(exp_dir)
	save_specs_file(exp_dir, global_args)
	return exp_dir

def list_hyperparameters(exp_folder):
    spec_file_path = exp_folder+'Specs.txt'
    target_file_path = exp_folder+'Listed_Specs.txt'
    with open(spec_file_path, "r") as text_file: data_lines = text_file.readlines()
    all_data_str = ''.join(data_lines)
    all_data_str = all_data_str.split('Namespace', 1)[-1]
    all_data_str = all_data_str.rstrip("\n")
    all_data_str = all_data_str[1:-1]
    
    split_list = all_data_str.split(',')
    full_list = []
    curr = []
    for e in split_list:
        if '=' in e:
            full_list.append(''.join(curr))
            curr = []
        curr.append(e)
    full_list = full_list[1:]
    pro_full_list = []
    for e in full_list: 
        pro_full_list.append(e.strip().replace("'", '"')) 
    pro_full_list.sort()
    with open(target_file_path, "w") as text_file: text_file.write('\n'.join(pro_full_list))

def debugger():
	import sys, ipdb, traceback
	def info(type, value, tb):
	   traceback.print_exception(type, value, tb)
	   print
	   ipdb.pm()
	sys.excepthook = info

def load_data_compressed(path):
    with open(path, 'rb') as f:
        return pickle.loads(zlib.decompress(f.read()))

def read_cropped_celebA(filename, size=64):
	rgb = scipy.misc.imread(filename)
	if size==64:
		crop_size = 108
		x_start = (rgb.shape[0]-crop_size)/2
		y_start = (rgb.shape[1]-crop_size)/2
		rgb_cropped = rgb[x_start:x_start+crop_size,y_start:y_start+crop_size,:]
		rgb_scaled = scipy.imresize(rgb_cropped, (size, size), interp='bilinear')
	return rgb_scaled

def split_tensor_np(tensor, dimension, list_of_sizes):
	split_list = []
	curr_ind = 0
	for s in list_of_sizes:		
		split_list.append(np.take(tensor, range(curr_ind, curr_ind+s), axis=dimension))
		curr_ind += s
	return split_list

def add_object_to_collection(x, name):
    if type(x) is list:
        for i in range(len(x)):
            tf.add_to_collection('list/'+name+'_'+str(i), x[i])
    else:
        tf.add_to_collection(name, x)

def slice_parameters(parameters, start, size):
    sliced_param = tf.slice(parameters, [0, start], [-1, size])
    new_start = start+size
    return sliced_param, new_start

def hardstep(x):
	return (1-2*tf.nn.relu(0.5*(1-x))+tf.nn.relu(-x))

def tf_suppress_max_non_max(x):
	x_max = tf.reduce_max(x, axis=1, keep_dims=True)
	non_max_indeces = tf.sign(x_max-x)
	max_indeces = 1-non_max
	non_max_suppressed = max_indeces*x_max
	max_suppressed = x-non_max_suppressed
	return max_suppressed, non_max_suppressed

def parametric_relu(x, positive, negative):
         f1 = 0.5 * (positive + negative)
         f2 = 0.5 * (positive - negative)
         return f1 * x + f2 * tf.abs(x)

def polinomial_nonlin(x, coefficients):
	y_k = 0
	for order in range(coefficients.get_shape().as_list()[-1]):
		coefficient_batch_vector = coefficients[:,order][:, np.newaxis]
		pdb.set_trace()
		y_k = y_k+coefficient_batch_vector*x**order
	return y_k

def sigmoid_J(x):
	return (1-tf.sigmoid(x))*tf.sigmoid(x)

def upper_bounded_nonlinearity(x, max_value=1):
	return max_value-tf.nn.softplus(max_value-x)

def lower_bounded_nonlinearity(x, min_value=-1):
	return tf.nn.softplus(x-min_value)+min_value

def tanh_J(x):
	return 1-tf.nn.tanh(x)**2

def relu_J(x):
	return (tf.abs(x)+x)/(2*x)

def relu_abs(x):
	return tf.nn.relu(x)+tf.nn.relu(-x)

def parametric_relu_J(x, positive, negative):
	b_positive_x = (tf.abs(x)+x)/(2*x)
	return b_positive_x*positive+(1-b_positive_x)*negative

def polinomial_nonlin_J(x, coefficients):
	new_coefficients = coefficients[:, 1:]
	new_coefficients_dim = new_coefficients.get_shape().as_list()[-1]
	pdb.set_trace()
	new_coefficients = new_coefficients*tf.linspace(1.0, new_coefficients_dim, new_coefficients_dim)[np.newaxis, :]
	return polinomial_nonlin(x, coefficients)

class batch_norm(object):
	def __init__(self, epsilon=1e-5, decay = 0.9, name="batch_norm"):
		self.epsilon  = epsilon
		self.decay = decay
		self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
											data_format='NHWC', 
											decay=self.decay, 
											updates_collections=None,
											epsilon=self.epsilon,
											scale=True,
											is_training=train)

def conv_layer_norm_layer(input_layer, channel_index=3):
    input_layer_offset = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = input_layer.get_shape().as_list()[channel_index], use_bias = False, activation = None)[0]
    input_layer_scale = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = input_layer.get_shape().as_list()[channel_index], use_bias = False, activation = None)[0]
    return layer_norm(input_layer, [-1,-2,-3], channel_index, input_layer_offset, input_layer_scale)

def dense_layer_norm_layer(input_layer): # NOT SURE ABOUT THE CORRECTNESS OF THIS
    conv_input_layer = input_layer[:, np.newaxis, np.newaxis, :]
    channel_index = 3
    input_layer_offset = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = conv_input_layer.get_shape().as_list()[channel_index], use_bias = False, activation = None)[0]
    input_layer_scale = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = conv_input_layer.get_shape().as_list()[channel_index], use_bias = False, activation = None)[0]
    res = layer_norm(conv_input_layer, [-1,-2,-3], channel_index, input_layer_offset, input_layer_scale)
    return res[:, 0, 0, :]

def layer_norm(x, norm_axes, channel_index, channel_offset, channel_scale):
    # norm_axes = [-1,-2,-3]
    # norm_axes = [-1]
    mean, var = tf.nn.moments(x, norm_axes, keep_dims=True)
    frame = [1, 1, 1]
    frame[channel_index-1] = -1
    offset = tf.reshape(channel_offset, frame)
    scale = tf.reshape(channel_scale, frame)
    return tf.nn.batch_normalization(x, mean, var, offset, scale+1, 1e-5)

def wasserstein_metric_1D_gaussian_columns(x_samples, y_samples, mode='Metric Squared'):
    assert (len(x_samples.get_shape().as_list()) == 2)
    assert (len(y_samples.get_shape().as_list()) == 2)

    x_mean = tf.reduce_mean(x_samples, axis=0)[np.newaxis,:]
    y_mean = tf.reduce_mean(y_samples, axis=0)[np.newaxis,:]
    x_var = tf.reduce_mean((x_samples-x_mean)**2, axis=0)[np.newaxis,:]
    x_std = safe_tf_sqrt(x_var)
    y_var = tf.reduce_mean((y_samples-y_mean)**2, axis=0)[np.newaxis,:]
    y_std = safe_tf_sqrt(y_var)

    # THEY ARE THE SAME Fretchet and 2-Wasserstein 
    # d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)). FID score paper
    # d^2 = ||mu_1 - mu_2||^2 + ||C_1^(1/2) - C_2^(1/2)||^2_Frobenius. http://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/
    # if mode == 'Metric Squared':
    #     return (x_mean-y_mean)**2+(x_var+y_var-2*safe_tf_sqrt(x_var*y_var))
    # elif mode == 'Metric':
    #     return safe_tf_sqrt((x_mean-y_mean)**2+(x_var+y_var-2*safe_tf_sqrt(x_var*y_var)))
    if mode == 'Metric Squared':
        return (x_mean-y_mean)**2+(x_std-y_std)**2        
    elif mode == 'Metric':
        return safe_tf_sqrt((x_mean-y_mean)**2+(x_std-y_std)**2)

def wasserstein_metric_nD_gaussian_columns(x_samples, y_samples, mode='Metric Squared'):
    assert (len(x_samples.get_shape().as_list()) == 2)
    assert (len(y_samples.get_shape().as_list()) == 2)

    x_mean = tf.reduce_mean(x_samples, axis=0)[np.newaxis,:]
    y_mean = tf.reduce_mean(y_samples, axis=0)[np.newaxis,:]
    x_var = tf.reduce_mean((x_samples-x_mean)**2, axis=0)[np.newaxis,:]
    x_std = safe_tf_sqrt(x_var)
    y_var = tf.reduce_mean((y_samples-y_mean)**2, axis=0)[np.newaxis,:]
    y_std = safe_tf_sqrt(y_var)

    pdb.set_trace()
    # THEY ARE THE SAME Fretchet and 2-Wasserstein 
    # d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)). FID score paper
    # d^2 = ||mu_1 - mu_2||^2 + ||C_1^(1/2) - C_2^(1/2)||^2_Frobenius. http://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/
    # if mode == 'Metric Squared':
    #     return (x_mean-y_mean)**2+(x_var+y_var-2*safe_tf_sqrt(x_var*y_var))
    # elif mode == 'Metric':
    #     return safe_tf_sqrt((x_mean-y_mean)**2+(x_var+y_var-2*safe_tf_sqrt(x_var*y_var)))
    if mode == 'Metric Squared':
        return (x_mean-y_mean)**2+(x_std-y_std)**2        
    elif mode == 'Metric':
        return safe_tf_sqrt((x_mean-y_mean)**2+(x_std-y_std)**2)


def get_object_from_collection(object_type, name):
    if object_type is list:
        out = []
        while True:
            try: out.append(tf.get_collection('list/'+name+'_'+str(len(out)))[0])
            except: break;
        return out
    else:
        return tf.get_collection(name)[0]

def split_tensor(tensor, dimension, list_of_sizes):
	split_list = []
	curr_ind = 0
	for s in list_of_sizes:
		split_list.append(tensor.narrow(dimension, curr_ind, s).contiguous())
		curr_ind += s
	return split_list

def split_tensor_tf(tensor, dimension, list_of_sizes):
	if tensor is None: return []
	if dimension == -1: dimension = len(tensor.get_shape().as_list())-1
	return tf.split(tensor, list_of_sizes, axis=dimension)

def list_sum(list_input):
	if len(list_input) == 0: return 0
	summed = None
	for e in list_input:
		if summed is None: summed = e
		else: summed = summed+e
	return summed

def pigeonhole_score(random_samples_from_model, subset=500, neigh=0.1):
	n_dim = np.prod(random_samples_from_model.shape[2:])
	max_euclidean_distance = np.sqrt(n_dim)
	random_samples_from_model_flat = random_samples_from_model.reshape(random_samples_from_model.shape[0]*random_samples_from_model.shape[1],-1)
	rates_of_matches = []
	for i in range(int(random_samples_from_model_flat.shape[0]/subset)):
		condensed_valid_distances = scipy.spatial.distance.pdist(random_samples_from_model_flat[i*subset:(i+1)*subset,:], metric='euclidean')
		rates_of_matches.append(np.mean((condensed_valid_distances<neigh*max_euclidean_distance)))
	return np.mean(rates_of_matches), np.std(rates_of_matches)

def list_product(list_input):
	if len(list_input) == 0: return 0
	else: return int(np.asarray(list_input).prod())

def list_merge(list_of_lists_input):
	out = []
	for e in list_of_lists_input:  out = out+e
	return out

def list_remove_none(list_input):
	return [x for x in list_input if x is not None]

def generate_empty_lists(num_lists, list_length):
	return [[None]*list_length for i in range(num_lists)]

def interleave_data(list_of_data):
	data_size = list_of_data[0].shape
	alldata = np.zeros((data_size[0]*len(list_of_data), *data_size[1:]))
	for i in range(len(list_of_data)):
		alldata[i::len(list_of_data)] = list_of_data[i]
	return  alldata

def extract_vis_data(reshape_func, obs, obs_param_out, batch_size):
    batch_reshaped = reshape_func(obs.view(-1, *obs.size()[2:]))
    batch_reshaped = [e.contiguous().view(obs.size(0), obs.size(1), *e.size()[1:]) for e in batch_reshaped]
    vis_data = [(batch_reshaped[i][:batch_size, ...].data.numpy(), 
                 obs_param_out[i][:batch_size, ...].data.numpy()) for i in range(len(obs_param_out))]
    return vis_data

def visualize_images(visualized_list, batch_size = 20, time_size = 30, save_dir = './', postfix = ''):
	visualized_list = visualized_list[..., :3]
	if not os.path.exists(save_dir): os.makedirs(save_dir)	
	batch_size = min(visualized_list.shape[0], batch_size)
	time_size = min(visualized_list.shape[1], time_size)
	block_size = [batch_size, time_size]
	padding = [5, 5]
	image_size = visualized_list.shape[-3:]
	canvas = np.ones([image_size[0]*block_size[0]+ padding[0]*(block_size[0]+1), 
					  image_size[1]*block_size[1]+ padding[1]*(block_size[1]+1), image_size[2]])

	for i in range(block_size[0]):
		start_coor = padding[0] + i*(image_size[0]+padding[0])
		for t in range(block_size[1]):
			y_start = (t+1)*padding[1]+t*image_size[1]
			canvas[start_coor:start_coor+image_size[0], y_start:y_start+image_size[1], :] =  visualized_list[i][t]
	if canvas.shape[2] == 1: canvas = np.repeat(canvas, 3, axis=2)
	scipy.misc.toimage(canvas).save(save_dir+'imageMatrix_'+postfix+'.png')

def visualize_images2(visualized_list, block_size, max_examples=20, save_dir = './', postfix = '', postfix2 = None):
	visualized_list = visualized_list[..., :3]
	assert(visualized_list.shape[0] == np.prod(block_size))
	padding = [1, 1]
	image_size = visualized_list.shape[-3:]
	canvas = np.ones([image_size[0]*min(block_size[0], max_examples)+ padding[0]*(min(block_size[0], max_examples)+1), 
					  image_size[1]*block_size[1]+ padding[1]*(block_size[1]+1), image_size[2]])

	visualized_list = visualized_list.reshape(*block_size, *visualized_list.shape[-3:])
	for i in range(min(block_size[0], max_examples)):
		start_coor = padding[0] + i*(image_size[0]+padding[0])
		for t in range(block_size[1]):
			y_start = (t+1)*padding[1]+t*image_size[1]
			canvas[start_coor:start_coor+image_size[0], y_start:y_start+image_size[1], :] =  visualized_list[i][t]
	if canvas.shape[2] == 1: canvas = np.repeat(canvas, 3, axis=2)

	if not os.path.exists(save_dir): os.makedirs(save_dir)
	scipy.misc.toimage(canvas).save(save_dir+'imageMatrix_'+postfix+'.png')
	# if postfix2 is None: scipy.misc.toimage(canvas).save(save_dir+'/../imageMatrix.png')
	# else: scipy.misc.toimage(canvas).save(save_dir+'/../imageMatrix_'+postfix2+'.png')
	if postfix2 is not None: scipy.misc.toimage(canvas).save(save_dir+'/../imageMatrix_'+postfix2+'.png')

	# canvas_int = (canvas*255).astype('uint8')
	# imsave(save_dir+'imageMatrix_'+postfix+'_2.png', canvas_int)
	# from PIL import Image
	# result = Image.fromarray(canvas_int)
	# result.save(save_dir+'imageMatrix_'+postfix+'_3.png', format='JPEG', subsampling=0, quality=100)

def draw_quivers(data, path):
	max_value =  np.abs(data).max()
	# max_value =  0.5
	soa = np.concatenate((np.zeros(data.T.shape), data.T),axis = 1)
	X, Y, U, V = zip(*soa)
	plt.figure()
	ax = plt.gca()
	colors = np.asarray([[1, 0, 0, 1], [0, 1, 0., 1], [0, 0, 1., 1]])
	ax.quiver(X, Y, U, V, color = colors, angles='xy', scale_units='xy', scale=1)
	ax.set_xlim([-max_value, max_value])
	ax.set_ylim([-max_value, max_value])
	plt.savefig(path)
	plt.close('all')

def draw_bar_plot(vector, y_min_max = None, thres = None, save_dir = './', postfix = ''):
	fig = plt.figure()
	plt.cla()
	plt.bar(np.arange(vector.shape[0]), vector, align='center')

	if thres is not None:
		plt.plot([0., vector.shape[0]], [thres[0], thres[0]], "r--")
		plt.plot([0., vector.shape[0]], [thres[1], thres[1]], "g--")
	plt.xlim(0, vector.shape[0])
	if y_min_max is not None:
		plt.ylim(*y_min_max)

	if not os.path.exists(save_dir): os.makedirs(save_dir)
	plt.savefig(save_dir+'barplot_'+postfix+'.png')
	plt.close('all')

def plot_ffs(xx, yy, f, save_dir = './', postfix = ''):
	# import matplotlib.mlab as mlab
	# delta = 0.025
	# x = np.arange(-3.0, 3.0, delta)
	# y = np.arange(-2.0, 2.0, delta)
	# X, Y = np.meshgrid(x, y)
	# Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
	# Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
	# # difference of Gaussians
	# Z = 10.0 * (Z2 - Z1)


	# # Create a simple contour plot with labels using default colors.  The
	# # inline argument to clabel will control whether the labels are draw
	# # over the line segments of the contour, removing the lines beneath
	# # the label
	# plt.figure()
	# CS = plt.contour(X.flatten(), Y.flatten(), Z.flatten())
	# plt.clabel(CS, inline=1, fontsize=10)
	# plt.title('Simplest default with labels')
	# plt.savefig('barplot_'+postfix+'_2.png')
	# pdb.set_trace()

	fig = plt.figure(figsize=(15, 15), dpi=300)
	plt.cla()
	plt.imshow(np.flipud(f), cmap='rainbow', interpolation='none')
	# # plt.xlim(0, vector.shape[0])
	# # if y_min_max is not None:
	# # 	plt.ylim(*y_min_max)
	if not os.path.exists(save_dir+'barplot/'): os.makedirs(save_dir+'barplot/')
	plt.savefig(save_dir+'barplot/barplot_'+postfix+'.png')
	plt.close('all')

	fig = plt.figure(figsize=(15, 15), dpi=300)
	plt.cla()
	heatmap = plt.pcolor(xx, yy, f, cmap='RdBu', vmin=np.min(f), vmax=np.max(f))
	plt.colorbar(heatmap)
	if not os.path.exists(save_dir+'barplot_3/'): os.makedirs(save_dir+'barplot_3/')
	plt.savefig(save_dir+'barplot_3/barplot_3_'+postfix+'.png')
	plt.close('all')


	fig = plt.figure(figsize=(15, 15), dpi=300)
	plt.cla()
	# plt.imshow(f, cmap='rainbow', interpolation='none')
	ax = fig.gca()
	cfset = ax.contourf(xx, yy, f, cmap='Blues')
	# # plt.xlim(0, vector.shape[0])
	# # if y_min_max is not None:
	# # 	plt.ylim(*y_min_max)
	if not os.path.exists(save_dir+'barplot_2/'): os.makedirs(save_dir+'barplot_2/')
	plt.savefig(save_dir+'barplot_2/barplot_2_'+postfix+'.png')
	plt.close('all')

def visualize_flat(visualized_list, batch_size = 10, save_dir = './', postfix = ''):
	canvas = -1*np.ones((int(len(visualized_list)/3)*4, max([e.shape[2] for e in visualized_list[:int(len(visualized_list)/3)]])))
	num_items = int(len(visualized_list)/3)
	cont_item = num_items-2
	time_step = -1
	for b in range(min(batch_size, visualized_list[0].shape[0])):
		canvas.fill(-1)
		for i in range(num_items):
			mat = np.concatenate([visualized_list[i][..., np.newaxis], 
								  visualized_list[i+int(len(visualized_list)/3)][..., np.newaxis],
								  visualized_list[i+2*int(len(visualized_list)/3)][..., np.newaxis]], axis = -1)
			data = mat[b][time_step]
			canvas[i*4:i*4+3, 0:data.shape[0]] = data.T
		plt.matshow(canvas)
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		plt.savefig(save_dir+'flatMatrix_'+str(b)+'_'+str(time_step)+'_'+postfix+'.png')
		plt.close('all')

def sharpness_eval_np(input_images): ## B x W x H x 3
	filter_mat = np.array([[0, 1, 0], [1, -4 ,1], [0, 1, 0]])[np.newaxis, :, :, np.newaxis]	
	if input_images.shape[-1] == 3: 
        grey_input_images = color.rgb2grey(input_images[:,0,:,:,:])[:,:,:,np.newaxis]
    elif input_images.shape[-1] == 1:
        grey_input_images = input_images[:,0,:,:,:]
    else:
        pdb.set_trace()
	all_vars = np.zeros((grey_input_images.shape[0]))
	for i in range(grey_input_images.shape[0]):
		curr_edges = signal.convolve2d(grey_input_images[i,:,:,0], filter_mat[0,:,:,0], mode='valid')
		all_vars[i] = np.var(curr_edges.flatten())
	return all_vars #np.mean(all_vars) # scalar

def visualize_flat2(visualized_list, batch_size = 10, save_dir = './', postfix = ''):
	time_step = -1
	width = 0.35
	num_dimensions_to_visualize = min(50, visualized_list[1][0][time_step].shape[0])
	ind = np.arange(num_dimensions_to_visualize)  # the x locations for the groups
	for i in range(batch_size):
		rects1 = plt.bar(ind, visualized_list[0][i][time_step][:num_dimensions_to_visualize], width, color='b')
		rects2 = plt.bar(ind + width, visualized_list[1][i][time_step][:num_dimensions_to_visualize], width, color='g')
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		plt.savefig(save_dir+'flatMatrix_'+str(i)+'_'+str(time_step)+'_'+postfix+'.png')
		plt.close('all')

def dataset_plotter_old(data_list, save_dir = './', postfix = '', postfix2 = None, show_also = False):
	colors = ['r', 'g', 'b', 'k']
	alphas = [1, 0.3, 0.3, 1.]
	if data_list[0].shape[1]==3:
		fig = plt.figure()
		plt.cla()
		ax = p3.Axes3D(fig)
		ax.view_init(7, -80)
		for i, data in enumerate(data_list):
			ax.plot3D(data[:, 0], data[:, 1], data[:, 2], 'o', color = colors[i])

	if data_list[0].shape[1]==2:
		fig = plt.figure(figsize=(8, 8), dpi=150)
		plt.cla()
		ax = fig.gca()
		# xmin, xmax = -3.5, 3.5
		# ymin, ymax = -3.5, 3.5

		xmin, xmax = -1.5, 1.5
		ymin, ymax = -1.5, 1.5

		# if len(data_list)==3: pdb.set_trace()
		for i, data in enumerate(data_list):			
			plt.scatter(data[:, 0], data[:, 1], color = colors[i], s=0.5, alpha=alphas[i]) #, edgecolors='none'
			# plt.scatter(data_list[i][:, 0], data_list[i][:, 1], color = colors[i], s=0.5, alpha=alphas[i]) #, edgecolors='none'
		ax.set_xlim([xmin, xmax])
		ax.set_ylim([ymin, ymax])
		plt.axes().set_aspect('equal')

	if not os.path.exists(save_dir): os.makedirs(save_dir)
	plt.savefig(save_dir+'datasets_'+postfix+'.png')
	if postfix2 is None: plt.savefig(save_dir+'/../datasets.png')
	else: plt.savefig(save_dir+'/../datasets_'+postfix2+'.png')
	# if postfix2 == 'data_only_3': pdb.set_trace()

	if show_also: plt.show()
	else: plt.close('all')

def dataset_plotter(data_list, ranges=None, tie=False, point_thickness=0.5, colors=None, save_dir = './', postfix = '', postfix2 = None, show_also = False):
	if colors is None: colors = ['r', 'g', 'b', 'k', 'c', 'm', 'gold', 'teal', 'springgreen', 'lightcoral', 'darkgray']
	# alphas = [1, 0.3, 0.3, 1.]
	# alphas = [1, 0.6, 0.4, 1.]
	alphas = [0.3, 1, 1, 1.]

	# if data_list[0].shape[1]==3:
	# 	fig = plt.figure()
	# 	plt.cla()
	# 	ax = p3.Axes3D(fig)
	# 	ax.view_init(11, 0)
	# 	for i, data in enumerate(data_list):
	# 		ax.plot3D(data[:, 0], data[:, 1], data[:, 2], 'o', color = colors[i])
	n_lines = 50

	if data_list[0].shape[1]==3:
		fig = plt.figure(figsize=(8, 8))
		plt.cla()
		ax = fig.add_subplot(111, projection='3d')
		# ax.set_title('z-axis left side')
		ax = fig.add_axes(MyAxes3D(ax, 'l'))
		# example_surface(ax) # draw an example surface
		# cm = plt.get_cmap("RdYlGn")
		
		# fig = plt.figure()
		# plt.cla()
		# ax = p3.Axes3D(fig)
		# ax.view_init(11, 0)
		for i, data in enumerate(data_list):
			X, Y, Z = data[:, 0], data[:, 1], data[:, 2]
			# ax.scatter(X,Y,Z, c=cm.coolwarm(Z), linewidth=0)
			ax.scatter(X,Y,Z, color = colors[i], linewidth=0)
			# ax.plot3D(X,Y,Z, 'o', facecolors=cm.jet(Z)) #color = colors[i])


			# squareshape = [int(np.sqrt(X.shape[0])), int(np.sqrt(X.shape[0]))]
			# X = X.reshape(squareshape)
			# Y = Y.reshape(squareshape)
			# Z = Z.reshape(squareshape)
			# surf = ax.plot_surface(data[:, 0], data[:, 1], data[:, 2], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			# surf = ax.plot_surface(data[:, 0], data[:, 1], data[:, 2], 'o', color = colors[i], antialiased=False)#, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			# surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm, linewidth=1, antialiased=False)#,color = cm.coolwarm)

			# ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.3)
			# cset = ax.contour(X, Y, Z, zdir='z', offset=-15, cmap=cm.coolwarm)
			# cset = ax.contour(X, Y, Z, zdir='x', offset=-15, cmap=cm.coolwarm)
			# cset = ax.contour(X, Y, Z, zdir='y', offset=15, cmap=cm.coolwarm)


		# 	surf = ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
		# fig.colorbar(surf, shrink=0.5, aspect=5)
			
		# ax.set_zlim(-1.01, 1.01)
		# ax.zaxis.set_major_locator(LinearLocator(10))
		# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

		
		# Add a color bar which maps values to colors.
		# fig.colorbar(surf, shrink=0.5, aspect=5)

		if ranges is None: ranges = (-5, 5)
		ax.set_xlim(*ranges)
		ax.set_ylim(*ranges)
		ax.set_zlim(*ranges)

	if data_list[0].shape[1]==2:
		fig = plt.figure(figsize=(8, 8), dpi=150)
		plt.cla()
		ax = fig.gca()
		xmin, xmax = -3., 3.
		ymin, ymax = -3., 3.
		# if len(data_list)==3: pdb.set_trace()
		for i, data in enumerate(data_list):
			if len(data_list)<=4: 			
				plt.scatter(data[:, 0], data[:, 1], color = colors[i], s=point_thickness, alpha=alphas[i]) #, edgecolors='none'
			else:
				plt.scatter(data[:, 0], data[:, 1], color = colors[i], s=3) #, edgecolors='none'
			# plt.scatter(data_list[i][:, 0], data_list[i][:, 1], color = colors[i], s=0.5, alpha=alphas[i]) #, edgecolors='none'
		if tie and len(data_list)==3:
			norms = np.sum((data_list[1]-data_list[2])**2, axis=1)
			norm_order = np.argsort(norms)[::-1]
			# norm_order = np.argsort(norms)

			x_all, y_all = [], []
			for j in range(min(data_list[1].shape[0], n_lines)):
				# print(j, norm_order[j])
				x1, y1 = [data_list[1][norm_order[j]][0], data_list[2][norm_order[j]][0]], [data_list[1][norm_order[j]][1], data_list[2][norm_order[j]][1]]
				# np.concatenate([data_list[0][norm_order,0,np.newaxis], data_list[1][norm_order,0,np.newaxis]],axis=1)[:2,:]
				# np.concatenate([data_list[0][:,0,np.newaxis], data_list[1][:,0,np.newaxis]],axis=1)
				# pdb.set_trace()
				plt.plot(x1, y1, 'k')

		ax.set_xlim([xmin, xmax])
		ax.set_ylim([ymin, ymax])
		plt.axes().set_aspect('equal')

	ax = plt.gca()
	try:
		ax.set_axis_bgcolor((1., 1., 1.))
	except:
		ax.set_facecolor((1., 1., 1.))

	ax.spines['bottom'].set_color('black')
	ax.spines['top'].set_color('black')
	ax.spines['right'].set_color('black')
	ax.spines['left'].set_color('black')

	if not os.path.exists(save_dir): os.makedirs(save_dir)
	plt.savefig(save_dir+'datasets_'+postfix+'.png', bbox_inches='tight')
	if postfix2 is None: plt.savefig(save_dir+'/../datasets.png', bbox_inches='tight')
	else: plt.savefig(save_dir+'/../datasets_'+postfix2+'.png', bbox_inches='tight')

	if show_also: plt.show()
	else: plt.close('all')


def plot2D_dist(data, save_dir = './', postfix = ''):
	x = data[:, 0]
	y = data[:, 1]
	# xmin, xmax = x.min()-0.1, x.max()+0.1
	# ymin, ymax = y.min()-0.1, y.max()+0.1
	
	# xmin, xmax = -3.5, 3.5
	# ymin, ymax = -3.5, 3.5

	xmin, xmax = -1.5, 1.5
	ymin, ymax = -1.5, 1.5

	# Peform the kernel density estimate
	xx, yy = np.mgrid[xmin:xmax:125j, ymin:ymax:125j]
	positions = np.vstack([xx.ravel(), yy.ravel()])
	values = np.vstack([x, y])
	kernel = st.gaussian_kde(values)
	f = np.reshape(kernel(positions).T, xx.shape)

	fig = plt.figure(figsize=(8, 8), dpi=150)
	ax = fig.gca()
	# Contourf plot

	cfset = ax.contourf(xx, yy, f, cmap='Blues')
	## Or kernel density estimate plot instead of the contourf plot
	#ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
	# Contour plot
	print(len(xx))
	print(xx.shape)
	print(len(yy))
	print(yy.shape)
	print(len(f))
	print(f.shape)
	cset = ax.contour(xx, yy, f, colors='k')
	# Label plot
	ax.set_xlabel('Y1')
	ax.set_ylabel('Y0')
	ax.set_xlim([xmin, xmax])
	ax.set_ylim([ymin, ymax])

	plt.axes().set_aspect('equal')
	if not os.path.exists(save_dir+'/plot2D_dist/'): os.makedirs(save_dir+'/plot2D_dist/')
	plt.savefig(save_dir+'plot2D_dist/distribution_'+postfix+'.png')
	plt.savefig(save_dir+'distribution.png')
	ax.clabel(cset, inline=1, fontsize=10)

	if not os.path.exists(save_dir+'/plot2D_dist_b_labeled/'): os.makedirs(save_dir+'/plot2D_dist_b_labeled/')
	plt.savefig(save_dir+'plot2D_dist_b_labeled/distribution_b_labeled_'+postfix+'.png')
	plt.savefig(save_dir+'distribution_b_labeled.png')
	plt.close('all')


def plot2D_dist2(Y):
	Y = np.random.multivariate_normal((0, 0), [[0.8, 0.05], [0.05, 0.7]], 100)
	ax = sns.kdeplot(Y, shade = True, cmap = "PuBu")
	ax.patch.set_facecolor('white')
	ax.collections[0].set_alpha(0)
	ax.set_xlabel('$Y_1$', fontsize = 15)
	ax.set_ylabel('$Y_0$', fontsize = 15)
	plt.xlim(-3, 3)
	plt.ylim(-3, 3)
	plt.plot([-3, 3], [-3, 3], color = "black", linewidth = 1)
	plt.savefig('./ff_dist2.png')
	# plt.show()
	pdb.set_trace()

def visualize_datasets(sess, feed_dict, dataset, obs_sample_out_tf, latent_sample_out_tf, transported_sample_out_tf=None, input_sample_out_tf=None, save_dir = './', postfix = ''):
	n_sampled = 0
	all_obs_sample_out = None
	all_latent_sample_out = None
	all_transp_sample_out = None
	all_input_sample_out = None

	while n_sampled < dataset.shape[0]:
		try: input_sample_out, transported_sample_out, obs_sample_out, latent_sample_out = sess.run([input_sample_out_tf['flat'], transported_sample_out_tf['flat'], obs_sample_out_tf['flat'], latent_sample_out_tf], feed_dict = feed_dict)
		except: obs_sample_out, latent_sample_out = sess.run([obs_sample_out_tf['flat'], latent_sample_out_tf], feed_dict = feed_dict)

		if all_obs_sample_out is None: all_obs_sample_out = obs_sample_out.reshape(-1, obs_sample_out.shape[-1])
		else: all_obs_sample_out = np.concatenate([all_obs_sample_out, obs_sample_out.reshape(-1, obs_sample_out.shape[-1])], axis=0)
		if all_latent_sample_out is None: all_latent_sample_out = latent_sample_out.reshape(-1, latent_sample_out.shape[-1])
		else: all_latent_sample_out = np.concatenate([all_latent_sample_out, latent_sample_out.reshape(-1, latent_sample_out.shape[-1])], axis=0)
		try: 
			if all_transp_sample_out is None: all_transp_sample_out = transported_sample_out.reshape(-1, transported_sample_out.shape[-1])
			# else: all_transp_sample_out = np.concatenate([all_transp_sample_out, transported_sample_out.reshape(-1, transported_sample_out.shape[-1])], axis=0)
			if all_input_sample_out is None: all_input_sample_out = input_sample_out.reshape(-1, input_sample_out.shape[-1])
			# else: all_input_sample_out = np.concatenate([all_input_sample_out, input_sample_out.reshape(-1, input_sample_out.shape[-1])], axis=0)

		except: pass

		n_sampled += obs_sample_out.reshape(-1, obs_sample_out.shape[-1]).shape[0]

	all_obs_sample_out = all_obs_sample_out[:dataset.shape[0], ...]
	all_latent_sample_out = all_latent_sample_out[:dataset.shape[0], ...]

	# try: 
	# 	all_transp_sample_out = all_transp_sample_out[:dataset.shape[0], ...]
	# 	all_input_sample_out = all_input_sample_out[:dataset.shape[0], ...]
	# except: pass

	print('Mean: ', all_obs_sample_out[:,0].mean(), all_obs_sample_out[:,1].mean())
	print('Variance: ', all_obs_sample_out[:,0].std(), all_obs_sample_out[:,1].std())

	dataset_plotter([all_obs_sample_out], save_dir = save_dir+'/dataset_plotter_data_only/', postfix = postfix+'_data_only', postfix2 = 'data_only')
	dataset_plotter([dataset, all_obs_sample_out], save_dir = save_dir+'/dataset_plotter_data_real/', postfix = postfix+'_data_real', postfix2 = 'data_real')
	try: 
		dataset_plotter([dataset, all_input_sample_out, all_transp_sample_out], save_dir = save_dir+'/dataset_plotter_data_transport_lined/', tie=True, postfix = postfix+'_data_transport_lined', postfix2 = 'data_transport_lined')
		dataset_plotter([dataset, all_transp_sample_out], save_dir = save_dir+'/dataset_plotter_data_transport/', postfix = postfix+'_data_transport', postfix2 = 'data_transport')
	except: pass

	# plot2D_dist(all_obs_sample_out, save_dir = save_dir, postfix = postfix)

def visualizeTransitions(sess, input_dict, generative_dict, save_dir = '.', postfix = ''):
	interpolated_sample_np, interpolated_further_sample_np, interpolated_sample_begin_np, interpolated_sample_end_np = \
		sess.run([generative_dict['interpolated_sample']['image'], generative_dict['interpolated_further_sample']['image'], 
				  generative_dict['interpolated_sample_begin']['image'], generative_dict['interpolated_sample_end']['image']], feed_dict = input_dict)
	
	interpolated_sample_linear_np = sess.run(generative_dict['interpolated_sample_linear']['image'], feed_dict = input_dict)
	# samples_params_np = np.array([interpolated_sample_begin_np, interpolated_sample_np, interpolated_further_sample_np, interpolated_sample_end_np])
	samples_params_np = np.array([interpolated_sample_begin_np, *interpolated_sample_linear_np, interpolated_sample_end_np])
	vis_data = np.concatenate(samples_params_np, axis=1)
	np.clip(vis_data, 0, 1, out=vis_data)
	visualize_images(np.concatenate(samples_params_np, axis=1), save_dir = save_dir, postfix = postfix+'_'+'image')
	visualize_images(vis_data, save_dir = save_dir, postfix = postfix+'_'+'image_2')

def visualize_datasets2(sess, feed_dict_func, data_loader, dataset, obs_sample_out_tf, save_dir = './', postfix = ''):
	data_loader.train()
	all_obs_sample_out, batch_all = None, None
	n_sampled, num_batches, i = 0, 10, 0
	for batch_idx, curr_batch_size, batch in data_loader:
		if i == num_batches: break
		if batch_all is None: batch_all = copy.deepcopy(batch) 
		else:
			try: batch_all['observed']['data']['flat'] = np.concatenate([batch_all['observed']['data']['flat'], batch['observed']['data']['flat']], axis=0)
			except: batch_all['observed']['data']['image'] = np.concatenate([batch_all['observed']['data']['image'], batch['observed']['data']['image']], axis=0)
		i += 1
	feed_dict = feed_dict_func(batch_all)

	while n_sampled < dataset.shape[0]:
		obs_sample_out = sess.run(obs_sample_out_tf['flat'], feed_dict = feed_dict)
		if all_obs_sample_out is None: all_obs_sample_out = obs_sample_out.reshape(-1, obs_sample_out.shape[-1])
		else: all_obs_sample_out = np.concatenate([all_obs_sample_out, obs_sample_out.reshape(-1, obs_sample_out.shape[-1])], axis=0)
		n_sampled += obs_sample_out.reshape(-1, obs_sample_out.shape[-1]).shape[0]
	all_obs_sample_out = all_obs_sample_out[:dataset.shape[0], ...]

	dataset_plotter([all_obs_sample_out], save_dir = save_dir+'/dataset_plotter_data_only_2/', postfix = postfix+'_data_only_2', postfix2 = 'data_only_2')
	dataset_plotter([all_obs_sample_out, dataset], save_dir = save_dir+'/dataset_plotter_data_real_2/', postfix = postfix+'_data_real_2', postfix2 = 'data_real_2')
	# plot2D_dist(all_obs_sample_out, save_dir = save_dir, postfix = postfix)

def visualize_datasets3(sess, feed_dict_func, data_loader, dataset, obs_sample_out_tf, save_dir = './', postfix = ''):
	n_sampled = 0
	all_obs_sample_out = None
	data_loader.train()
	for batch_idx, curr_batch_size, batch in data_loader:
		obs_sample_out = sess.run(obs_sample_out_tf['flat'], feed_dict = feed_dict_func(batch))
		if all_obs_sample_out is None: all_obs_sample_out = obs_sample_out.reshape(-1, obs_sample_out.shape[-1])
		else: all_obs_sample_out = np.concatenate([all_obs_sample_out, obs_sample_out.reshape(-1, obs_sample_out.shape[-1])], axis=0)
		n_sampled += obs_sample_out.reshape(-1, obs_sample_out.shape[-1]).shape[0]
	all_obs_sample_out = all_obs_sample_out[:n_sampled, ...]

	dataset_plotter([all_obs_sample_out], save_dir = save_dir+'/dataset_plotter_data_only_3/', postfix = postfix+'_data_only_3', postfix2 = 'data_only_3')
	dataset_plotter([all_obs_sample_out, dataset], save_dir = save_dir+'/dataset_plotter_data_real_3/', postfix = postfix+'_data_real_3', postfix2 = 'data_real_3')
	# plot2D_dist(all_obs_sample_out, save_dir = save_dir, postfix = postfix)

def visualize_datasets4(sess, feed_dict_func, data_loader, obs_sample_begin_tf, obs_sample_end_tf, obs_sample_out_tf, save_dir = './', postfix = ''):
	data_loader.train()
	all_obs_sample_begin, all_obs_sample_end, all_obs_sample_out, batch_all = None, None, None, None
	n_sampled, num_batches, i = 0, 1, 0
	for batch_idx, curr_batch_size, batch in data_loader:
		if i == num_batches: break
		if batch_all is None: batch_all = copy.deepcopy(batch) 
		else:
			try: batch_all['observed']['data']['flat'] = np.concatenate([batch_all['observed']['data']['flat'], batch['observed']['data']['flat']], axis=0)
			except: batch_all['observed']['data']['image'] = np.concatenate([batch_all['observed']['data']['image'], batch['observed']['data']['image']], axis=0)
		i += 1
	feed_dict = feed_dict_func(batch_all)

	while n_sampled < 20000:
		obs_sample_begin, obs_sample_end, obs_sample_out = sess.run([obs_sample_begin_tf['flat'], obs_sample_end_tf['flat'], obs_sample_out_tf['flat']], feed_dict = feed_dict)
		if all_obs_sample_begin is None: all_obs_sample_begin = obs_sample_begin.reshape(-1, obs_sample_begin.shape[-1])
		else: all_obs_sample_begin = np.concatenate([all_obs_sample_begin, obs_sample_begin.reshape(-1, obs_sample_begin.shape[-1])], axis=0)
		if all_obs_sample_end is None: all_obs_sample_end = obs_sample_end.reshape(-1, obs_sample_end.shape[-1])
		else: all_obs_sample_end = np.concatenate([all_obs_sample_end, obs_sample_end.reshape(-1, obs_sample_end.shape[-1])], axis=0)
		if all_obs_sample_out is None: all_obs_sample_out = obs_sample_out.reshape(-1, obs_sample_out.shape[-1])
		else: all_obs_sample_out = np.concatenate([all_obs_sample_out, obs_sample_out.reshape(-1, obs_sample_out.shape[-1])], axis=0)
		n_sampled += obs_sample_out.reshape(-1, obs_sample_out.shape[-1]).shape[0]

	dataset_plotter([all_obs_sample_out], save_dir = save_dir+'/dataset_plotter_data_only_4/', postfix = postfix+'_data_only_4', postfix2 = 'data_only_4')
	dataset_plotter([all_obs_sample_out, all_obs_sample_begin, all_obs_sample_end], save_dir = save_dir+'/dataset_plotter_data_real_4/', postfix = postfix+'_data_real_4', postfix2 = 'data_real_4')
	# plot2D_dist(all_obs_sample_out, save_dir = save_dir, postfix = postfix)

def tf_differentiable_specific_shuffle_with_axis(batch_input, specific_order, axis=0):
	list_of_sizes = [1,]*batch_input.get_shape().as_list()[axis]
	split_tensors = tf.split(batch_input, list_of_sizes, axis=axis)
	rearranged_split_tensors = [split_tensors[i] for i in specific_order]
	return tf.concat(rearranged_split_tensors, axis)

def tf_differentiable_random_shuffle_with_axis(batch_input, axis=0):
	permute_list = None
	if axis == 0: 
		pdb.set_trace()
		input_batch_input = batch_input
	else: 
		permute_list = [*range(len(batch_input.shape))]
		permute_list[0] = axis
		permute_list[axis] = 0
		input_batch_input = tf.transpose(batch_input, perm=permute_list)

	eye_matrix = tf.eye(input_batch_input.get_shape().as_list()[0])
	permute_matrix = tf.stop_gradient(tf.random_shuffle(eye_matrix))

	transposed_shape = input_batch_input.get_shape().as_list()
	transposed_shape_none_replaced = [-1 if a==None else a for a in transposed_shape]
	input_batch_input_collapsed = tf.reshape(input_batch_input, [input_batch_input.get_shape().as_list()[0], -1])

	result_collapsed = tf.matmul(permute_matrix, input_batch_input_collapsed)
	result =  tf.reshape(result_collapsed, transposed_shape_none_replaced)

	if axis == 0: output_result = result
	else: output_result = tf.transpose(result, perm=permute_list)

	return output_result

def tf_nondifferentiable_random_shuffle_with_axis(batch_input, axis=0):
	permute_list = None
	if axis == 0: input_batch_input = batch_input
	else: 
		permute_list = [*range(len(batch_input.shape))]
		permute_list[0] = axis
		permute_list[axis] = 0
		input_batch_input = tf.transpose(batch_input, perm=permute_list)

	result = tf.random_shuffle(input_batch_input)

	if axis == 0: output_result = result
	else: output_result = tf.transpose(result, perm=permute_list)

	return output_result

def safe_tf_sqrt(x, clip_value=1e-5):
	return tf.sqrt(tf.clip_by_value(x, clip_value, np.inf))

# def lrelu(x, leak=0.2):
# 	f1 = 0.5 * (1 + leak)
# 	f2 = 0.5 * (1 - leak)
# 	return f1 * x + f2 * tf.abs(x)

def SeLu(x, lambda_var=1.0507009873554804934193349852946, alpha_var=1.6732632423543772848170429916717):
	positive_x = 0.5*(x+tf.abs(x))
	negative_x = 0.5*(x-tf.abs(x))
	negative = alpha_var*(tf.exp(negative_x)-1)
	return lambda_var*(positive_x+negative)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def FCResnetLayer(x, units, activation=LeakyReLU, output_activation=None):
	x_size = x.get_shape().as_list()[-1]
	pre_layer = tf.layers.dense(inputs = x, units = units, activation = activation, use_bias=True)
	out = x+tf.layers.dense(inputs = pre_layer, units = x_size, activation = None, use_bias=True)
	if output_activation is not None: out = output_activation(out)
	return out

def FCResnetLayer_v2(x, units, reduce_units=None, activation=None, reduce_activation=LeakyReLU, normalization_mode='None'):	
	shortcut = x 
	if x.get_shape().as_list()[-1] != units:
		shortcut = tf.layers.dense(inputs = x, units = units, activation = None, use_bias=True)
		if normalization_mode == 'Layer Norm': 
			shortcut = conv_layer_norm_layer(shortcut[:,np.newaxis,np.newaxis,:], channel_index=3)[:,0,0,:]
		elif normalization_mode == 'Batch Norm': 
			shortcut = batch_norm()(shortcut)

	if reduce_units is None: reduce_units = units
	reduce_layer = tf.layers.dense(inputs = x, units = reduce_units, activation = None, use_bias=True)
	if normalization_mode == 'Layer Norm': 
		reduce_layer = conv_layer_norm_layer(reduce_layer[:,np.newaxis,np.newaxis,:], channel_index=3)[:,0,0,:]
	elif normalization_mode == 'Batch Norm': 
		reduce_layer = batch_norm()(reduce_layer)
	reduce_layer = reduce_activation(reduce_layer)

	add_layer = tf.layers.dense(inputs = reduce_layer, units = units, activation = None, use_bias=True)
	if normalization_mode == 'Layer Norm': 
		add_layer = conv_layer_norm_layer(add_layer[:,np.newaxis,np.newaxis,:], channel_index=3)[:,0,0,:]
	elif normalization_mode == 'Batch Norm': 
		add_layer = batch_norm()(add_layer)

	out = shortcut+add_layer
	if activation is not None: out = activation(out)
	return out

def FCResnetLayer_v3(x, units, output_unit_rate=1, activation=LeakyReLU, output_activation=LeakyReLU):
	x_size = x.get_shape().as_list()[-1]
	pre_layer = tf.layers.dense(inputs = x, units = units, activation = activation, use_bias=True)
	if output_unit_rate == 1: shortcut = x
	else: shortcut = tf.tile(x, [1, output_unit_rate])

	out = shortcut+tf.layers.dense(inputs = pre_layer, units = output_unit_rate*x_size, activation = None, use_bias=True)
	if output_activation is not None: out = output_activation(out)
	return out

def ConvResnetLayer_v1(x, units, reduce_units=None, activation=None, output_activation=None):
	if reduce_units is None: reduce_units = units
	reduce_layer = tf.layers.conv2d(inputs=x, filters=reduce_units, kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=True, activation=activation)
	added_layer = reduce_layer+tf.layers.conv2d(inputs=reduce_layer, filters=reduce_units, kernel_size=[1, 1], strides=[1, 1], padding="valid", use_bias=True, activation=activation)
	output = tf.layers.conv2d(inputs=added_layer, filters=units, kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=True, activation=None)
	return output

def ConvResnetLayer_v2(x, units, reduce_units=None, activation=None, reduce_activation=LeakyReLU, normalization_mode='None', image_modify='None'):
	if reduce_units is None: reduce_units = units
	
	shortcut = x 
	if image_modify != 'None' or x.get_shape().as_list()[-1] != units:
		if image_modify == 'Downsample': 
			shortcut = tf.layers.conv2d(inputs=x, filters=units, kernel_size=[3, 3], strides=[2,2], padding="same", use_bias=True, activation=None)
		elif image_modify == 'Upsample':
			shortcut = tf.layers.conv2d_transpose(inputs=x, filters=units, kernel_size=[3, 3], strides=[2,2], padding="same", use_bias=True, activation=None)			 
		elif image_modify == 'None':
			shortcut = tf.layers.conv2d(inputs=x, filters=units, kernel_size=[1, 1], strides=[1,1], padding="same", use_bias=True, activation=None)
		else: pdb.set_trace()
		if normalization_mode == 'Layer Norm': 
			shortcut = conv_layer_norm_layer(shortcut, channel_index=3)
		elif normalization_mode == 'Batch Norm': 
			shortcut = batch_norm()(shortcut)

	if image_modify == 'Downsample': 
		reduce_layer = tf.layers.conv2d(inputs=x, filters=reduce_units, kernel_size=[3, 3], strides=[2,2], padding="same", use_bias=True, activation=None)	
	elif image_modify == 'Upsample':
		reduce_layer = tf.layers.conv2d_transpose(inputs=x, filters=reduce_units, kernel_size=[3, 3], strides=[2,2], padding="same", use_bias=True, activation=None)			
	elif image_modify == 'None':
		reduce_layer = tf.layers.conv2d(inputs=x, filters=reduce_units, kernel_size=[3, 3], strides=[1,1], padding="same", use_bias=True, activation=None)
	else: pdb.set_trace()
	if normalization_mode == 'Layer Norm': 
		reduce_layer = conv_layer_norm_layer(reduce_layer, channel_index=3)
	elif normalization_mode == 'Batch Norm': 
		reduce_layer = batch_norm()(reduce_layer)
	reduce_layer = reduce_activation(reduce_layer)

	add_layer = tf.layers.conv2d(inputs=reduce_layer, filters=units, kernel_size=[3, 3], strides=[1,1], padding="same", use_bias=True, activation=None)
	if normalization_mode == 'Layer Norm': 
		add_layer = conv_layer_norm_layer(add_layer, channel_index=3)
	elif normalization_mode == 'Batch Norm': 
		add_layer = batch_norm()(add_layer)
	
	out = shortcut + add_layer
	if activation is not None: out = activation(out)
	return out

def random_rot_mat(dim, mode='SO(n)'):
	# https://statweb.stanford.edu/~cgates/PERSI/papers/subgroup-rand-var.pdf
	print('Creating uniformly random rotation matrix in ' + mode + ' with n=' + str(dim) + '.')
	assert (mode == 'SO(n)' or mode == 'O(n)')
	assert (dim > 0)

	intermediate_rotation = np.zeros((dim, dim))
	curr_rot = None
	start = time.time()
	for curr_dim in range(1, dim+1):
		if curr_dim == 1:
			if mode == 'SO(n)':
				if dim % 2 == 0: curr_rot = -1*np.ones((1,1))
				else: curr_rot = 1*np.ones((1,1))
			elif mode == 'O(n)':
				curr_rot = (2*np.random.randint(2)-1)*np.ones((1,1))
		else:
			intermediate_rotation[-curr_dim+1:,-curr_dim+1:] = curr_rot
			intermediate_rotation[-curr_dim, -curr_dim] = 1.

			e1 = np.zeros(curr_dim)[:, np.newaxis]
			e1[0, 0] = 1
			v_vec = np.random.randn(curr_dim)[:, np.newaxis]
			v_norm = np.sqrt(np.sum(v_vec**2))
			v = v_vec/v_norm

			householder_vec = (e1-v)
			# householder_vec = np.random.randn(curr_dim)[:, np.newaxis] # leads to non-uniform dist
			householder_norm = np.sqrt(np.sum(householder_vec**2))
			householder_dir = householder_vec/householder_norm

			substract = 2*np.dot(householder_dir, np.dot(householder_dir.T, intermediate_rotation[-curr_dim:, -curr_dim:]))
			# substract = 2*np.dot(np.dot(householder_dir, householder_dir.T), intermediate_rotation[-curr_dim:, -curr_dim:]) # much slower
			curr_rot = intermediate_rotation[-curr_dim:, -curr_dim:]-substract

	end = time.time()
	print('It took (sec): ', (end - start))
	return curr_rot

def euclidean_distance_squared(x, y, axis=[-1], keep_dims=True):
    return tf.reduce_sum((x-y)**2, axis=axis, keep_dims=keep_dims)

def metric_distance_sq(x, y):
    try: metric_distance = euclidean_distance_squared(x['image'], y['image'], axis=[-1,-2,-3], keep_dims=False)[:,:,np.newaxis]
    except: metric_distance = euclidean_distance_squared(x['flat'], y['flat'], axis=[-1], keep_dims=True)
    return metric_distance

def euclidean_distance(a, b):
    return safe_tf_sqrt(metric_distance_sq(a, b))

def quadratic_distance(a, b):
    return metric_distance_sq(a, b)

def rbf_kernel(z_1, z_2=None, sigma_z_sq=1):
    if z_2 is None: 
        squared_dists_all = (z_1[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2
        squared_dists = tf_remove_diagonal(squared_dists_all, first_dim_size=tf.shape(z_1)[0], second_dim_size=tf.shape(z_1)[0])
    else:
        squared_dists = (z_2[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2

    z_diff_norm_sq = tf.reduce_sum(squared_dists, axis=[-1], keep_dims=True)
    n_dim = z_1.get_shape().as_list()[-1]
    sigma_k_sq = 2*n_dim*sigma_z_sq
    return tf.exp(-z_diff_norm_sq/sigma_k_sq)

def inv_multiquadratics_kernel(z_1, z_2=None, sigma_z_sq=1):
    if z_2 is None: 
        squared_dists_all = (z_1[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2
        squared_dists = tf_remove_diagonal(squared_dists_all, first_dim_size=tf.shape(z_1)[0], second_dim_size=tf.shape(z_1)[0])
    else:
        squared_dists = (z_2[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2

    z_diff_norm_sq = tf.reduce_sum(squared_dists, axis=[-1], keep_dims=True)
    n_dim = z_1.get_shape().as_list()[-1]
    C = 2*n_dim*sigma_z_sq
    return C/(C+z_diff_norm_sq)

def rational_quadratic_kernel(z_1, z_2=None, alpha=1):
    if z_2 is None: 
        squared_dists_all = (z_1[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2
        squared_dists = tf_remove_diagonal(squared_dists_all, first_dim_size=tf.shape(z_1)[0], second_dim_size=tf.shape(z_1)[0])
    else:
        squared_dists = (z_2[:,np.newaxis,:]-z_1[np.newaxis,:,:])**2

    z_diff_norm_sq = tf.reduce_sum(squared_dists, axis=[-1], keep_dims=True)
    C = 1+z_diff_norm_sq/(2*alpha)
    if alpha == 1: return C
    else: return tf.pow(C, -alpha)

def circular_shift(z):
    return tf.concat([z[1:,:], z[0, np.newaxis,:]], axis=0)        

def interpolate_latent_codes(z, size=1, number_of_steps=10):
    z_a = z[:size,:] 
    z_b = z[size:2*size,:] 
    t = (tf.range(number_of_steps, dtype=np.float32)/float(number_of_steps-1))[np.newaxis, :, np.newaxis]
    z_interp = t*z_a[:, np.newaxis, :]+(1-t)*z_b[:, np.newaxis, :]
    return z_interp

def squeeze_realnpv_depthspace(image, squeeze_order=[1,2,3,4], verbose=True):
    assert (len(squeeze_order) == 4 and 1 in squeeze_order and 2 in squeeze_order and 3 in squeeze_order and 4 in squeeze_order)
    if verbose:
        order_str_list = [' top left ', ' top right ', ' bottom left ', ' bottom right ']
        print('Squeeze order: ', order_str_list[squeeze_order[0]-1], order_str_list[squeeze_order[1]-1], 
                                 order_str_list[squeeze_order[2]-1], order_str_list[squeeze_order[3]-1])
    # image is  batch x height x width x channel
    assert (image.get_shape()[1].value % 2 == 0)
    assert (image.get_shape()[2].value % 2 == 0)

    squeezed_regular = tf.nn.space_to_depth(image, 2)
    regular_order = tf.split(tf.reshape(squeezed_regular, [-1, squeezed_regular.get_shape()[1].value, squeezed_regular.get_shape()[2].value, 4, image.get_shape()[3]]), 4, axis=3)
    desired_order = [regular_order[squeeze_order[i]-1][:,:,:,0,:] for i in range(len(squeeze_order))]
    squeezed = tf.concat(desired_order, axis=-1)
    # squeezed is batch x height/2 x width/2 x channel*4 where original channels in input image are next to each other
    return squeezed

def squeeze_realnpv(image, squeeze_order=[1,2,3,4], verbose=True):
    assert (len(squeeze_order) == 4 and 1 in squeeze_order and 2 in squeeze_order and 3 in squeeze_order and 4 in squeeze_order)
    if verbose:
        order_str_list = [' top left ', ' top right ', ' bottom left ', ' bottom right ']
        print('Squeeze order: ', order_str_list[squeeze_order[0]-1], order_str_list[squeeze_order[1]-1], 
                                 order_str_list[squeeze_order[2]-1], order_str_list[squeeze_order[3]-1])
    # image is  batch x height x width x channel
    assert (image.get_shape()[1].value % 2 == 0)
    assert (image.get_shape()[2].value % 2 == 0)

    top_rows = image[:, 0::2, :, :]
    bottom_rows = image[:, 1::2, :, :]
    top_rows_left_cols = top_rows[:, :, 0::2, :]
    top_rows_right_cols = top_rows[:, :, 1::2, :]
    bottom_rows_left_cols = bottom_rows[:, :, 0::2, :]
    bottom_rows_right_cols = bottom_rows[:, :, 1::2, :]

    regular_order = [top_rows_left_cols, top_rows_right_cols, bottom_rows_left_cols, bottom_rows_right_cols]
    desired_order = [regular_order[squeeze_order[i]-1] for i in range(len(squeeze_order))]
    squeezed = tf.concat(desired_order, axis=-1)
    # squeezed is batch x height/2 x width/2 x channel*4 where original channels in input image are next to each other
    return squeezed

def unsqueeze_realnpv_depthspace(squeezed, squeeze_order=[1,2,3,4], verbose=True):
    assert (len(squeeze_order) == 4 and 1 in squeeze_order and 2 in squeeze_order and 3 in squeeze_order and 4 in squeeze_order)
    if verbose:
        order_str_list = [' top left ', ' top right ', ' bottom left ', ' bottom right ']
        print('Unsqueezing squeeze order: ', order_str_list[squeeze_order[0]-1], order_str_list[squeeze_order[1]-1], 
                                             order_str_list[squeeze_order[2]-1], order_str_list[squeeze_order[3]-1])
    # squeezed is batch x height/2 x width/2 x channel*4
    assert (squeezed.get_shape()[3].value % 4 == 0)
    out_n_channels = int(squeezed.get_shape()[3].value/4.)
    desired_order = tf.split(tf.reshape(squeezed, [-1, squeezed.get_shape()[1].value, squeezed.get_shape()[2].value, 4, out_n_channels]), 4, axis=3)
    regular_order = [None]*4
    for i in range(len(desired_order)): regular_order[squeeze_order[i]-1] = desired_order[i][:,:,:,0,:]
    image = tf.nn.depth_to_space(tf.concat(regular_order, axis=-1), 2)
    return image

def unsqueeze_realnpv(squeezed, squeeze_order=[1,2,3,4], verbose=True):
    assert (len(squeeze_order) == 4 and 1 in squeeze_order and 2 in squeeze_order and 3 in squeeze_order and 4 in squeeze_order)
    if verbose:
        order_str_list = [' top left ', ' top right ', ' bottom left ', ' bottom right ']
        print('Unsqueezing squeeze order: ', order_str_list[squeeze_order[0]-1], order_str_list[squeeze_order[1]-1], 
                                             order_str_list[squeeze_order[2]-1], order_str_list[squeeze_order[3]-1])
    # squeezed is batch x height/2 x width/2 x channel*4
    assert (squeezed.get_shape()[3].value % 4 == 0)
    out_n_channels = int(squeezed.get_shape()[3].value/4.)

    desired_order = [squeezed[:, :, :, 0*out_n_channels:1*out_n_channels], squeezed[:, :, :, 1*out_n_channels:2*out_n_channels],
                     squeezed[:, :, :, 2*out_n_channels:3*out_n_channels], squeezed[:, :, :, 3*out_n_channels:4*out_n_channels]]

    regular_order = [None]*4
    for i in range(len(desired_order)): regular_order[squeeze_order[i]-1] = desired_order[i]
    top_rows_left_cols, top_rows_right_cols, bottom_rows_left_cols, bottom_rows_right_cols = regular_order

    top_rows_left_cols_split = tf.split(top_rows_left_cols, top_rows_left_cols.get_shape()[2].value, axis=2)
    top_rows_right_cols_split = tf.split(top_rows_right_cols, top_rows_right_cols.get_shape()[2].value, axis=2)
    bottom_rows_left_cols_split = tf.split(bottom_rows_left_cols, bottom_rows_left_cols.get_shape()[2].value, axis=2)
    bottom_rows_right_cols_split = tf.split(bottom_rows_right_cols, bottom_rows_right_cols.get_shape()[2].value, axis=2)
    
    assert (len(top_rows_left_cols_split) == len(top_rows_right_cols_split))
    assert (len(bottom_rows_left_cols_split) == len(bottom_rows_right_cols_split))
    assert (len(top_rows_left_cols_split) == len(bottom_rows_left_cols_split))

    top_rows_split = []
    bottom_rows_split = []
    for i in range(len(top_rows_left_cols_split)):
        top_rows_split.append(top_rows_left_cols_split[i])
        top_rows_split.append(top_rows_right_cols_split[i])
        bottom_rows_split.append(bottom_rows_left_cols_split[i])
        bottom_rows_split.append(bottom_rows_right_cols_split[i])

    top_rows = tf.concat(top_rows_split, axis=2)
    bottom_rows = tf.concat(bottom_rows_split, axis=2)
    top_rows_rowsplit = tf.split(top_rows, top_rows.get_shape()[1].value, axis=1)
    bottom_rows_rowsplit = tf.split(bottom_rows, bottom_rows.get_shape()[1].value, axis=1)

    assert (len(top_rows_rowsplit) == len(bottom_rows_rowsplit))

    all_split = []
    for i in range(len(top_rows_rowsplit)):
        all_split.append(top_rows_rowsplit[i])
        all_split.append(bottom_rows_rowsplit[i])

    # image is  batch x height x width x channel
    image = tf.concat(all_split, axis=1)
    return image

def build_checkerboard_np(w, h, one_left_upper=True):
    if one_left_upper:
        re = np.r_[w*[1,0]] # even-numbered rows
        ro = np.r_[w*[0,1]]  # odd-numbered rows
        return np.row_stack(h*(re, ro))
    else:
        re = np.r_[w*[0,1]] # even-numbered rows
        ro = np.r_[w*[1,0]]  # odd-numbered rows
        return np.row_stack(h*(re, ro))

def tf_build_checker_board_for_images(image, one_left_upper=True):
    assert (image.get_shape()[1].value % 2 == 0)
    assert (image.get_shape()[2].value % 2 == 0)
    mask_np = build_checkerboard_np(int(image.get_shape()[1].value/2), int(image.get_shape()[2].value/2), one_left_upper=one_left_upper)[np.newaxis, :, :, np.newaxis]
    # print(mask_np[0,:,:,0])
    mask = tf.constant(mask_np, tf.float32)
    return mask

def compute_MMD_OLD(sample_batch_1, sample_batch_2, mode='His', positive_only=False):
    if mode == 'Mine':
        k_sample_1_2 = tf.reduce_mean(self.kernel_function(sample_batch_1, sample_batch_2))
        k_sample_1_1 = tf.reduce_mean(self.kernel_function(sample_batch_1))
        k_sample_2_2 = tf.reduce_mean(self.kernel_function(sample_batch_2))
        MMD = k_sample_2_2+k_sample_1_1-2*k_sample_1_2
    else:
        sample_qz, sample_pz = sample_batch_1, sample_batch_2
        sigma2_p = 1 ** 2
        n = tf.cast(tf.shape(sample_qz)[0], tf.int32) # batch size int
        nf = tf.cast(tf.shape(sample_qz)[0], tf.float32) # batch size float
        n_dim = sample_qz.get_shape().as_list()[-1]

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True) # norm sq of pz
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True) 
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        Cbase = 2.*n_dim*sigma2_p
        stat = 0.
        # for scale in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
        # for scale in [0.01, 0.1, 0.5, 1, 2, 10, 100]:
        for scale in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
        MMD = stat
    if positive_only:
        MMD = tf.nn.relu(MMD)+1e-7
    return MMD

def compute_MMD(sample_1, sample_2, weight_1=None, weight_2=None, kernel_type='InvMultiquadratics', kernel_mode='Sum', rectify=False, scales = [0.01, 0.1, 1, 10, 100]):
    assert (kernel_mode == 'Mid' or kernel_mode == 'Sum' or kernel_mode == 'Mean' or kernel_mode == 'Min' or kernel_mode == 'Max')
    assert (len(sample_1.get_shape().as_list()) == 2 and len(sample_2.get_shape().as_list()) == 2)
    assert (sample_1.get_shape()[1].value == sample_2.get_shape()[1].value)
    if weight_1 is not None: 
        assert (len(weight_1.get_shape().as_list()) == 2) 
        assert (weight_1.get_shape()[1].value == 1)
    if weight_2 is not None: 
        assert (len(weight_2.get_shape().as_list()) == 2)
        assert (weight_2.get_shape()[1].value == 1)

    batch_size_tf = tf.shape(sample_1)[0]
    batch_size_tf_float = tf.cast(batch_size_tf, tf.float32) # batch size float
    n_dim_tf = tf.shape(sample_1)[1]
    n_dim_tf_float = tf.cast(n_dim_tf, tf.float32) # batch size float

    sigma_sq_p = 1.0**2
    c_base = 2*n_dim_tf_float*sigma_sq_p
    
    off_diag_ones = 1.0-tf.eye(batch_size_tf)

    norm_sq_1 = tf.reduce_sum(tf.square(sample_1), axis=1, keepdims=True)
    dotprods_mat_1 = tf.matmul(sample_1, sample_1, transpose_b=True)
    distance_sq_mat_1 = norm_sq_1+tf.transpose(norm_sq_1)-2*dotprods_mat_1

    norm_sq_2 = tf.reduce_sum(tf.square(sample_2), axis=1, keepdims=True)
    dotprods_mat_2 = tf.matmul(sample_2, sample_2, transpose_b=True) 
    distance_sq_mat_2 = norm_sq_2+tf.transpose(norm_sq_2)-2*dotprods_mat_2

    dotprods_mat_1row_2col = tf.matmul(sample_1, sample_2, transpose_b=True)
    distance_sq_mat_1row_2col = norm_sq_1+tf.transpose(norm_sq_2)-2*dotprods_mat_1row_2col

    all_mmds = []
    for scale in scales:
        c_smooth = scale*c_base 

        if kernel_type == 'InvMultiquadratics':
            ker_mat_1 = inv_multiquadratics_from_l2_dist_sq(c_smooth, distance_sq_mat_1)
            ker_mat_2 = inv_multiquadratics_from_l2_dist_sq(c_smooth, distance_sq_mat_2)
            ker_mat_1row_2col = inv_multiquadratics_from_l2_dist_sq(c_smooth, distance_sq_mat_1row_2col)
        else:
            pdb.set_trace()

        if weight_1 is not None:
            ker_mat_1 = ker_mat_1*weight_1*tf.transpose(weight_1)
            ker_mat_1row_2col = ker_mat_1row_2col*weight_1
        if weight_2 is not None:
            ker_mat_2 = ker_mat_2*weight_2*tf.transpose(weight_2)
            ker_mat_1row_2col = ker_mat_1row_2col*tf.transpose(weight_2)

        ker_mat_1_plus_2_diag_suppressed =  off_diag_ones*(ker_mat_1+ker_mat_2)

        mmd_pos = tf.reduce_sum(ker_mat_1_plus_2_diag_suppressed)/(batch_size_tf_float*batch_size_tf_float-batch_size_tf_float)
        mmd_neg = 2*tf.reduce_sum(ker_mat_1row_2col)/(batch_size_tf_float*batch_size_tf_float)

        mmd = mmd_pos-mmd_neg
        all_mmds.append(mmd)

    if kernel_mode == 'Mid':
        overall_mmd = all_mmds[int(np.ceil(float(len(all_mmds))/2.0))]
    elif kernel_mode == 'Sum':
        overall_mmd = tf.add_n(all_mmds)
    elif kernel_mode == 'Mean':
        overall_mmd = tf.add_n(all_mmds)/float(len(all_mmds))
    elif kernel_mode == 'Min':
        overall_mmd = tf.reduce_min(tf.concat([e[np.newaxis] for e in all_mmds], axis=0))
    elif kernel_mode == 'Max':
        overall_mmd = tf.reduce_max(tf.concat([e[np.newaxis] for e in all_mmds], axis=0))

    if rectify: overall_mmd = tf.nn.relu(overall_mmd)+1e-7
    return overall_mmd

def sum_div(div_func, batch_input):
    # batch_input_transformed = (self.circular_shift(batch_input)+batch_input)/np.sqrt(2)
    batch_input_transformed = (self.circular_shift(batch_input)+batch_input)/2

    # integral = div_func(tf.reverse(batch_input, axis=[0,]) , batch_input_transformed)
    integral = div_func(batch_input, batch_input_transformed)
    return integral

def stable_div(div_func, batch_input, batch_rand_dirs):
    n_transforms = batch_rand_dirs.get_shape().as_list()[0]
    transformed_batch_input = self.apply_householder_reflections(batch_input, batch_rand_dirs)
    transformed_batch_input_inverse = tf.reverse(transformed_batch_input, [0])
    list_transformed_batch_input = tf.split(transformed_batch_input_inverse, n_transforms, axis=1)
    integral = 0
    for e in list_transformed_batch_input: 
        integral += div_func(e[:,0,:], batch_input)
    integral /= n_transforms
    return integral

def stable_div_expanded(div_func, batch_input, batch_rand_dirs_expanded, mode='max', projection_dim=10):        
    n_transforms = batch_rand_dirs_expanded.get_shape().as_list()[0]
    n_reflections = batch_rand_dirs_expanded.get_shape().as_list()[1]
    
    # batch_input_to_transform = batch_input[:self.batch_size_tf//2, :]
    # batch_input_to_compare = batch_input[self.batch_size_tf//2:, :]
    batch_input_to_transform = batch_input
    batch_input_to_compare = batch_input

    transformed_batch_input = batch_input_to_transform[np.newaxis, :, :]
    for i in range(n_reflections):
        transformed_batch_input = self.apply_householder_reflections2(transformed_batch_input, batch_rand_dirs_expanded[:, i, :])
    transformed_batch_input_inverse = tf.reverse(transformed_batch_input, [1])
    
    if mode == 'max':
        list_of_divergences = []
        for j in range(n_transforms):
            list_of_divergences.append(div_func(transformed_batch_input_inverse[j,:,:][:,:projection_dim], batch_input_to_compare[:,:projection_dim])[np.newaxis])
        list_of_divergences_tf = tf.concat(list_of_divergences, axis=0)
        max_stable_div = tf.reduce_max(list_of_divergences_tf)
        stable_div = max_stable_div
    elif mode == 'mean':
        integral = 0
        for j in range(n_transforms):
            integral += div_func(transformed_batch_input_inverse[j,:,:][:,:projection_dim], batch_input_to_compare[:,:projection_dim])
        ave_stable_div = integral/n_transforms
        stable_div = ave_stable_div
    else: pdb.set_trace()

    return stable_div

# def stable_div_expanded(div_func, batch_input, batch_rand_dirs_expanded):        
#     n_transforms = batch_rand_dirs_expanded.get_shape().as_list()[0]
#     n_reflections = batch_rand_dirs_expanded.get_shape().as_list()[1]
    
#     transformed_batch_input = batch_input[np.newaxis, :, :]
#     for i in range(n_reflections):
#         transformed_batch_input = self.apply_householder_reflections2(transformed_batch_input, batch_rand_dirs_expanded[:, i, :])

#     pdb.set_trace()
#     batch_input_to_transform = batch_input[self.batch_size_tf//2:, :]
#     batch_input_to_compare = batch_input[:self.batch_size_tf//2, :]

#     integral1 = 0
#     for j in range(n_transforms):
#         integral1 += div_func(transformed_batch_input_inverse[j,:,:], batch_input_to_compare)
#     integral1 /= n_transforms
    
#     batch_input_to_transform = batch_input[self.batch_size_tf//2:, :]
#     batch_input_to_compare = batch_input[:self.batch_size_tf//2, :]
#     transformed_batch_input = batch_input_to_transform[np.newaxis, :, :]
#     for i in range(n_reflections):
#         transformed_batch_input = self.apply_householder_reflections2(transformed_batch_input, batch_rand_dirs_expanded[:, i, :])
#     transformed_batch_input_inverse = transformed_batch_input
#     integral2 = 0
#     for j in range(n_transforms):
#         integral2 += div_func(transformed_batch_input_inverse[j,:,:], batch_input_to_compare)
#     integral2 /= n_transforms
    
#     integral = (integral1+integral2)/2.
#     return integral

def apply_single_householder_reflection(batch_input, v_householder):
	v_householder_expanded = v_householder[np.newaxis, :]
	householder_inner = tf.reduce_sum(v_householder_expanded*batch_input, axis=1, keep_dims=True)  
	batch_out = batch_input-2*householder_inner*v_householder_expanded
	return batch_out

def apply_rotations_reflections(batch_input, batch_rand_dirs_expanded):
	v_householder = batch_rand_dirs[np.newaxis, :,:]
	batch_input_expand = batch_input[:,np.newaxis, :]
	householder_inner_expand = tf.reduce_sum(v_householder*batch_input_expand, axis=2, keep_dims=True)  
	return batch_input_expand-2*householder_inner_expand*v_householder

def apply_householder_reflections(batch_input, batch_rand_dirs):
	v_householder = batch_rand_dirs[np.newaxis, :,:]
	batch_input_expand = batch_input[:,np.newaxis, :]
	householder_inner_expand = tf.reduce_sum(v_householder*batch_input_expand, axis=2, keep_dims=True)  
	return batch_input_expand-2*householder_inner_expand*v_householder

def apply_householder_reflections2(batch_input, batch_rand_dirs):
	v_householder = batch_rand_dirs[:, np.newaxis,:]
	householder_inner_expand = tf.reduce_sum(v_householder*batch_input, axis=2, keep_dims=True)  
	return batch_input-2*householder_inner_expand*v_householder

def variable_summaries(var, name):
	mean = tf.reduce_mean(var)
	tf.summary.scalar('mean/' + name, mean)
	stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	tf.summary.scalar('stddev/' + name, stddev)
	tf.summary.scalar('max/' + name, tf.reduce_max(var))
	tf.summary.scalar('min/' + name, tf.reduce_min(var))
	tf.summary.histogram(name, var)

def visualize_vectors(visualized_list, batch_size = 10, save_dir = './', postfix = ''):
	time_step = -1
	for i in range(int(len(visualized_list)/3.)):
		mat = np.concatenate([visualized_list[i][..., np.newaxis], 
				  visualized_list[i+int(len(visualized_list)/3)][..., np.newaxis],
				  visualized_list[i+2*int(len(visualized_list)/3)][..., np.newaxis]], axis = -1)
		for b in range(min(batch_size, visualized_list[0].shape[0])):
			data = mat[b][time_step]
			if not os.path.exists(save_dir): os.makedirs(save_dir)
			draw_quivers(data, save_dir+'flatRelPos_obs_'+str(i)+'_example_'+str(b)+'_t_'+str(time_step)+'_'+postfix+'.png')

def copy_dir2(dir1, dir2):	
	if os.path.exists(dir1) and not os.path.exists(dir2): os.makedirs(dir2)
	file_list = glob.glob(dir1+'/*')
	for e in file_list:
		pdb.set_trace()

		shutil.copyfile(e, dir2+e[len(dir1):])

def copy_dir(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def save_checkpoint(saver, sess, global_step, exp_dir):
	if not os.path.exists(exp_dir): os.makedirs(exp_dir)
	# checkpoint_file = os.path.join(exp_dir , 'model.ckpt')
	# saver.save(sess, exp_dir+'/model' global_step=global_step)
	# saver.export_meta_graph(os.path.join(exp_dir , 'model.meta'))
	saver.save(sess, exp_dir+'/model')
	saver.export_meta_graph(exp_dir+'/model.meta')

def read_specs_file(path):
	epoch_info = -1
	with open(path, "r") as text_file:
		all_string = text_file.read()
		all_string_clean = all_string[all_string.find('Namespace(')+10: all_string.find(')\n\n')]
		all_string_split = all_string_clean.split(',')
		for i in range(len(all_string_split)):
			if 'curr_epoch' in (all_string_split[i]):
				epoch_info = int(all_string_split[i].split('=')[-1])	
	assert (epoch_info>0)
	return epoch_info

def load_checkpoint(saver, sess, checkpoint_exp_dir):	
	# saver = tf.train.import_meta_graph(os.path.join(checkpoint_exp_dir , 'model.meta'))
	checkpoint_epoch = read_specs_file(checkpoint_exp_dir+'Specs.txt')
	saver.restore(sess, checkpoint_exp_dir+'/checkpoint/model')
	return checkpoint_epoch 

	# ckpt = tf.train.get_checkpoint_state(exp_dir+'/checkpoints')
	# if ckpt and ckpt.model_checkpoint_path:
	# 	saver.restore(sess, ckpt.model_checkpoint_path)
	# 	print('Loaded checkpoint from file: ' + ckpt.model_checkpoint_path)
	# else: print('No Checkpoint..')


def visualize_categorical_data(vis_data, save_dir):
	for i in range(len(vis_data)):
		if i+1 == len(vis_data):
			real_action_map = vis_data[i][0][:, :-1].reshape(-1, 1, 10, 10)
			prob_action_map = vis_data[i][1][:, :-1].reshape(-1, 1, 10, 10)
			visualize_images([real_action_map, prob_action_map], save_dir+'/vis_cat_2/', [1, 10, 10])
		else:
			plt.cla()
			f, axes = plt.subplots(vis_data[i][0].shape[0], sharex=True, sharey=True, figsize=(8, 15))
			for j, axis in enumerate(axes):
				sample = vis_data[i][0][j]
				probs = vis_data[i][1][j]	
				markers,stems,base = axis.stem(sample, 'r', markerfmt='ro', bottom=0)
				for stem in stems: stem.set_linewidth(6)
				markers,stems,base = axis.stem(probs, 'g', markerfmt='go', bottom=0)
				for stem in stems: stem.set_linewidth(6)
				axis.axhline(0, color='blue', lw=2)

			f.subplots_adjust(hspace=0.3)
			plt.savefig(save_dir+'/vis_cat_2/VAEVis_'+str(i)+'.png')
	plt.close('all')

def visualize_categorical_data_series(vis_data, save_dir, time_index = 0, postfix = ''):
	if not os.path.exists(save_dir+'vis/'): os.makedirs(save_dir+'vis/')
	
	for i in range(len(vis_data)):
		category = vis_data[i]
		if i+1 == len(vis_data):
			real_action_map = category[0][:, time_index, :-1].reshape(-1, 1, 10, 10)
			prob_action_map = category[1][:, time_index, :-1].reshape(-1, 1, 10, 10)
			visualize_images([real_action_map, prob_action_map], save_dir+'vis/', [1, 10, 10], postfix = postfix)
		else:
			plt.cla()
			f, axes = plt.subplots(category[0].shape[0], sharex=True, sharey=True, figsize=(8, 15))
			for j, axis in enumerate(axes):
				sample = category[0][j]
				probs = category[1][j]	
				markers,stems,base = axis.stem(sample[time_index], 'r', markerfmt='ro', bottom=0)
				for stem in stems: stem.set_linewidth(6)
				markers,stems,base = axis.stem(probs[time_index], 'g', markerfmt='go', bottom=0)
				for stem in stems: stem.set_linewidth(6)
				axis.axhline(0, color='blue', lw=2)

			f.subplots_adjust(hspace=0.3)
			plt.savefig(save_dir+'vis/'+'VAEVis_'+str(i)+'_'+postfix+'.png')
	plt.close('all')

class TemperatureObjectTF:
	def __init__(self, start_temp, max_steps):
		self.start_temp = start_temp
		self.max_steps = max_steps
		self.train()

	def train(self):
		self.mode = 'Train'

	def eval(self):
		self.mode = 'Test'
	
	def temp_step(self, t):
		if self.mode == 'Test': return 1.0
		return tf.minimum(1.0, self.start_temp+t/float(self.max_steps))

class TemperatureObject:
	def __init__(self, start_temp, max_steps):
		self.start_temp = start_temp
		self.max_steps = max_steps
		self.t = 0
		self.train()

	def train(self):
		self.mode = 'Train'

	def eval(self):
		self.mode = 'Test'
	
	def temp(self):
		if self.mode == 'Test': return 1.0
		return min(1.0, self.start_temp+float(self.t)/float(self.max_steps))

	def temp_step(self):
		curr_temp = self.temp()
		if self.mode == 'Train': self.t += 1
		return curr_temp

def serialize_model(model, path=None):
	pass

def deserialize_model(path, model=None):
	pass


def update_dict_from_file(dict_to_update, filename):
	try: 
		with open(filename) as f:
			content = f.readlines()
		for line in content:
			if line[0] != '#':  
				line_list = line.split(':')
				if len(line_list)==2 and line_list[0] in dict_to_update:
					try: dict_to_update[line_list[0]] = float(line_list[1])
					except: pass
	except: pass
	return dict_to_update

class PrintSnooper:
    def __init__(self, stdout):
        self.stdout = stdout
    def write(self, s):
        self.stdout.write('====print====\n')
        traceback.print_stack()
        self.stdout.write(s)
        self.stdout.write("\n")
    def flush(self):
        self.stdout.flush()

def tf_batch_and_input_dict(batch_template, additional_inputs_template=None):
    batch_tf = {}
    for d in ['context', 'observed']:
        if d not in batch_tf: batch_tf[d] = {}
        for a in ['properties', 'data']:
            if a not in batch_tf[d]: batch_tf[d][a] = {}
            for t in ['flat', 'image']:
                if a == 'properties': batch_tf[d][a][t] = batch_template[d][a][t]
                elif a == 'data': 
                    if batch_template[d][a][t] is None: batch_tf[d][a][t] = None
                    else: 
                    	batch_tf[d][a][t] = tf.placeholder(tf.float32, [None, *batch_template[d][a][t].shape[1:]]) 
                    	# batch_tf[d][a][t] = tf.placeholder(tf.float32, [*batch_template[d][a][t].shape]) 

    def input_dict_func(batch, additional_inputs=None):
        input_dict = {}
        for d in ['context', 'observed']:
            for t in batch_tf[d]['data']:
                if batch_tf[d]['data'][t] is not None:
                    input_dict[batch_tf[d]['data'][t]] = batch[d]['data'][t]
        if additional_inputs is not None: input_dict = {additional_inputs_template: additional_inputs, **input_dict}
        return input_dict
    return batch_tf, input_dict_func

def _log_gamma(x):
	return scipy.special.gammaln(x).astype(np.float32)

def log_gamma(x):
	log_gamma = tf.py_func(_log_gamma, [x,], [tf.float32], name="log_gamma", stateful=False)[0]
	log_gamma.set_shape(x.get_shape().as_list())
	return log_gamma

def _beta_samples(alpha=0.5, beta=0.5):
	assert (alpha.shape == beta.shape)
	return np.random.beta(alpha, beta).astype(np.float32)

def beta_samples(alpha, beta):
	beta_samples = tf.py_func(_beta_samples, [alpha, beta], [tf.float32], name="beta_samples", stateful=False)[0]
	beta_samples.set_shape(alpha.get_shape().as_list())
	return beta_samples

def _triangular_ones(full_size, trilmode = 0):
  return np.tril(np.ones(full_size), trilmode).astype(np.float32)

def _block_triangular_ones(full_size, block_size, trilmode = 0):
  O = np.ones(block_size) 
  Z = np.zeros(block_size) 
  string_matrix = np.empty(full_size, dtype="<U3")
  string_matrix[:] = 'O'
  string_matrix = np.tril(string_matrix, trilmode)
  string_matrix[string_matrix == ''] = 'Z'
  stringForBlocks = ''
  for i in range(string_matrix.shape[0]):
    for j in range(string_matrix.shape[1]):
      stringForBlocks = stringForBlocks + string_matrix[i,j]
      if j!=string_matrix.shape[1]-1: stringForBlocks = stringForBlocks + ','
    if i!=string_matrix.shape[0]-1: stringForBlocks = stringForBlocks + ';'
  return np.bmat(stringForBlocks).astype(np.float32)

def _block_diagonal_ones(full_size, block_size):
  O = np.ones(block_size) 
  Z = np.zeros(block_size) 
  string_matrix = np.empty(full_size, dtype="<U3")
  string_matrix[:] = 'Z'
  np.fill_diagonal(string_matrix, 'O')
  stringForBlocks = ''
  for i in range(string_matrix.shape[0]):
    for j in range(string_matrix.shape[1]):
      stringForBlocks = stringForBlocks + str(string_matrix[i,j])
      if j!=string_matrix.shape[1]-1: stringForBlocks = stringForBlocks + ','
    if i!=string_matrix.shape[0]-1: stringForBlocks = stringForBlocks + ';'
  return np.bmat(stringForBlocks).astype(np.float32)

def triangular_ones(full_size, trilmode = 0):
    return tf.py_func(_triangular_ones, [full_size, trilmode], [tf.float32], name="triangular_ones", stateful=False)[0]

def block_triangular_ones(full_size, block_size, trilmode = 0):
    return tf.py_func(_block_triangular_ones, [full_size, block_size, trilmode], 
                      [tf.float32], name="block_triangular_ones", stateful=False)[0]

def block_diagonal_ones(full_size, block_size):
    return tf.py_func(_block_diagonal_ones, [full_size, block_size],
                      [tf.float32], name="block_diagonal_ones", stateful=False)[0]

def tf_remove_diagonal(tensor, first_dim_size=5, second_dim_size=5):
	lower_triangular_mask = triangular_ones([first_dim_size, second_dim_size], trilmode = -1)
	for i in range(len(tensor.get_shape())-2): lower_triangular_mask = lower_triangular_mask[..., np.newaxis]
	upper_triangular_mask = 1-triangular_ones([first_dim_size, second_dim_size], trilmode = 0)
	for i in range(len(tensor.get_shape())-2): upper_triangular_mask = upper_triangular_mask[..., np.newaxis]

	tensor_lt = (tensor*lower_triangular_mask)[1:,...]
	tensor_ut = (tensor*upper_triangular_mask)[:-1,...]
	tensor_without_diag = tensor_lt+tensor_ut
	return tensor_without_diag

# random_data = tf.random_normal((2, 2, 3, 3), 0, 1, dtype=tf.float32)
# total = tf_remove_diagonal(random_data, first_dim_size=random_data.get_shape().as_list()[0], second_dim_size=random_data.get_shape().as_list()[1])

# # init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# # sess.run(init)
# # out1, out2, out3, out4 = sess.run([random_data_lt, random_data_ut, total, random_data])
# out1, out2 = sess.run([total, random_data])

# print(out1.shape)
# print(out1)

# print(out2.shape)
# print(out2)

# # print(out3.shape)
# # print(out3)

# # print(out4.shape)
# # print(out4)
# pdb.set_trace()

def tf_single_zero():
	return tf.ones(shape=(1))[0]*0

def _batch_matmul(a, b):
    if a.get_shape()[0].value == 1:
      return tf.matmul(b, a[0, :, :], transpose_a=False, transpose_b=True)
    else:
      # return tf.batch_matmul(a, tf.expand_dims(b, 2))[:, :, 0]
      return tf.matmul(a, tf.expand_dims(b, 2))[:, :, 0]

def householder_matrix(n, k):
	H_mat = np.eye(n)
	u_var = np.random.randn(k)
	if k == 1:
		H_ref = 1
	else:
		H_ref = np.eye(k) - (2/np.dot(u_var, u_var))*np.outer(u_var, u_var)
	H_mat[n-k:, n-k:] = H_ref
	return H_mat

def householder_rotations(n, k_start=1):
	M_k_start_list = []
	W = None 
	for k in range(k_start, n+1):
		H_k = householder_matrix(n, k)
		M_k_start_list.append(H_k)
		if W is None: W = H_k
		else: W = np.matmul(H_k, W)
	return W

def householder_matrix_tf(batch, n, k, init_reflection=1, u_var=None):
	if k == 1:
		H_ref = np.sign(init_reflection)*tf.tile(tf.eye(k)[np.newaxis, :, :], [batch, 1, 1])
	else:
		if u_var is None: pdb.set_trace() # u_var = tf.random_normal((batch, k, 1), 0, 1, dtype=tf.float32)
		else: u_var = u_var[:, :, np.newaxis]
		H_ref = tf.eye(k)[np.newaxis, :, :] - (2./tf.matmul(u_var, u_var, transpose_a=True))*tf.matmul(u_var, u_var, transpose_b=True)
	
	H_mat_id = tf.tile(tf.concat([tf.eye(n-k)[np.newaxis, :, :], tf.zeros(shape=(k, n-k))[np.newaxis, :, :]], axis=1), [batch, 1, 1])		
	H_ref_column = tf.concat([tf.tile(tf.zeros(shape=(n-k, k))[np.newaxis, :, :], [batch, 1, 1]), H_ref], axis=1)
	H_mat = tf.concat([H_mat_id, H_ref_column], axis=2)
	return H_mat

def householder_rotations_tf(n, k_start=1, init_reflection=1, params=None, mode='uniform'):
	assert (mode == 'uniform' or mode == 'simple')

	M_k_start_list = []
	W = None 
	batch = 1
	if params.get_shape().as_list()[1] == 0: params = None
	if params is not None: 
		batch = tf.shape(params)[0]
		params_split = tf.split(params, list(range(max(2, k_start), n+1)), axis=1)
	
	for k in range(k_start, n+1):
		if k % 20 == 0: print('Householder rotation progress: '+ str(k) + '/' + str(n))
		if params is None or k == 1: u_var = None
		else: u_var = params_split[k-max(2, k_start)]

		if u_var is None or mode == 'simple':
			H_k = householder_matrix_tf(batch, n, k, float(init_reflection), u_var)
		elif mode == 'uniform':
			e1_tf = tf_standard_basis_vector(u_var.get_shape().as_list()[1], 0)[np.newaxis, :]
			e1_minus_v = e1_tf-(u_var/safe_tf_sqrt(tf.reduce_sum(u_var**2, axis=1, keep_dims=True)))
			H_k = householder_matrix_tf(batch, n, k, float(init_reflection), e1_minus_v)
		
		M_k_start_list.append(H_k)
		if W is None: W = H_k
		else: W = tf.matmul(H_k, W)
	return W

def householder_rotation_vectors_tf(n, k_start=1, init_reflection=1, params=None, mode='uniform'):
	assert (mode == 'uniform' or mode == 'simple')

	list_householder_dir_vecs = []
	batch = 1
	if params.get_shape().as_list()[1] == 0: params = None
	if params is not None: 
		batch = tf.shape(params)[0]
		params_split = tf.split(params, list(range(max(2, k_start), n+1)), axis=1)

	for k in range(k_start, n+1):
		if k % 20 == 0: print('Householder rotation progress: '+ str(k) + '/' + str(n))
		if params is None or k == 1: u_var = None
		else: u_var = params_split[k-max(2, k_start)]

		if k == 1:
			list_householder_dir_vecs.append(float(init_reflection))
		else:
			if u_var is None: 
				pdb.set_trace() # u_var = tf.random_normal((batch, k), 0, 1, dtype=tf.float32)
			
			if mode == 'uniform':
				e1_tf = tf_standard_basis_vector(u_var.get_shape().as_list()[1], 0)[np.newaxis, :]
				e1_minus_v = e1_tf-(u_var/safe_tf_sqrt(tf.reduce_sum(u_var**2, axis=1, keep_dims=True)))
				u_dir = e1_minus_v/safe_tf_sqrt(tf.reduce_sum(e1_minus_v**2, axis=1, keep_dims=True))
			elif mode == 'simple':
				u_dir = u_var/safe_tf_sqrt(tf.reduce_sum(u_var**2, axis=1, keep_dims=True))

			list_householder_dir_vecs.append(u_dir)
	return list_householder_dir_vecs

def tf_standard_basis_vector(dim, index):
	vec = np.zeros(dim)
	vec[index] = 1
	return tf.constant(vec, tf.float32)

def tf_resize_image(x, resize_ratios=[2,2], b_unknown_shape=False, mode='nearest'):
    assert (mode == 'nearest' or mode == 'bicubic')
    if b_unknown_shape:
        new_shape_1 = tf.cast(resize_ratios[0]*tf.shape(x)[1], tf.int32)
        new_shape_2 = tf.cast(resize_ratios[1]*tf.shape(x)[2], tf.int32)
    else:
        new_shape_1 = int(resize_ratios[0]*x.get_shape().as_list()[1])
        new_shape_2 = int(resize_ratios[1]*x.get_shape().as_list()[2])
    if mode == 'bicubic': method = tf.image.ResizeMethod.AREA
    elif mode == 'nearest': method = tf.image.ResizeMethod.NEAREST_NEIGHBOR 
    return tf.image.resize_images(x, [new_shape_1, new_shape_2], method=method)

def tf_center_crop_image(x, resize_ratios=[28,28]):
	shape_0 = x.get_shape().as_list()[1]
	shape_1 = x.get_shape().as_list()[2]
	start_0 = int((shape_0-resize_ratios[0])/2)
	start_1 = int((shape_1-resize_ratios[1])/2)
	return x[:, start_0:start_0+resize_ratios[0], start_1:start_1+resize_ratios[1], :]

def normalized_bell_np(x):
	return 2./(1+np.exp(-8*x-4))+2./(1+np.exp(8*x-4))-3

def tf_jacobian_1(y, x):
	y_flat = tf.reshape(y, (-1,))
	jacobian_flat = tf.stack([tf.gradients(y_i, x)[0] for y_i in tf.unstack(y_flat)])
	jacobian = tf.reshape(jacobian_flat, y.shape.concatenate(x.shape))
	return tf.reshape(jacobian, [*y.get_shape().as_list(), -1, *x.get_shape().as_list()[1:]])

def tf_jacobian(y, x): # the fast one
	y_flat = tf.reshape(y, [-1])
	n = y_flat.shape[0]
	loop_vars = [tf.constant(0, tf.int32), tf.TensorArray(tf.float32, size=n)]
	aa = lambda j, _: j < n
	bb = lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[j], x)))
	_, jacobian = tf.while_loop(aa, bb, loop_vars)
	jacobian = jacobian.stack()
	return tf.reshape(jacobian, [*y.get_shape().as_list(), -1, *x.get_shape().as_list()[1:]])

def tf_batchify_across_dim(list_of_data, axis=0):
	list_of_tensors_flat = [e['flat'] for e in list_of_data] 
	list_of_tensors_image = [e['image'] for e in list_of_data] 
	output = {'flat': None, 'image': None}
	if None not in list_of_tensors_flat: 
		list_of_sizes = [e.get_shape().as_list()[axis] for e in list_of_tensors_flat] 
		if None in list_of_sizes:
			list_of_sizes = [tf.shape(e)[axis] for e in list_of_tensors_flat] 
		output['flat'] = tf.concat(list_of_tensors_flat, axis=axis)
	else:
		list_of_sizes = [e.get_shape().as_list()[axis] for e in list_of_tensors_image] 
		if None in list_of_sizes:
			list_of_sizes = [tf.shape(e)[axis] for e in list_of_tensors_image] 
		output['image'] = tf.concat(list_of_tensors_image, axis=axis)

	return output, {'axis': axis, 'list_of_sizes': list_of_sizes}

def tf_debatchify_across_dim(concatenated, axis_sizes):
	if type(concatenated) is not dict:
		list_of_data = split_tensor_tf(concatenated, axis_sizes['axis'], axis_sizes['list_of_sizes'])		
	else:
		if concatenated['flat'] is not None:
			list_of_tensors_flat = split_tensor_tf(concatenated['flat'], axis_sizes['axis'], axis_sizes['list_of_sizes']) 
			list_of_data = [{'image': None, 'flat': e} for e in list_of_tensors_flat]
		else:
			list_of_tensors_image = split_tensor_tf(concatenated['image'], axis_sizes['axis'], axis_sizes['list_of_sizes']) 
			list_of_data = [{'image': e, 'flat': None} for e in list_of_tensors_image]
	return list_of_data

def tf_jacobian_3(y, x):
	y_list = tf.unstack(tf.reshape(y, [-1]))
	jacobian_list = [tf.gradients(y_, x)[0] for y_ in y_list]  # list [grad(y0, x), grad(y1, x), ...]
	jacobian = tf.stack(jacobian_list)
	return tf.reshape(jacobian, [*y.get_shape().as_list(), -1, *x.get_shape().as_list()[1:]])

def tf_batch_reduced_jacobian(y, x):
	batch_jacobian = tf_jacobian(tf.reduce_sum(y, axis=[0], keep_dims=False), x)
	return tf.transpose(batch_jacobian, perm=[1, 0, 2])

def triangular_matrix_mask(output_struct, input_struct):
	mask = np.greater_equal(output_struct[:, np.newaxis]-input_struct[np.newaxis, :], 0).astype(np.float32)
	return mask

def get_mask_list_for_MADE(input_dim, layer_expansions, add_mu_log_sigma_layer=False, b_normalization=True):
	layer_expansions = [1, *layer_expansions]
	layer_structures = []
	for e in layer_expansions:
		# np.random.shuffle(np.repeat(np.arange(1, input_dim+1), e))
		layer_structures.append(np.repeat(np.arange(1, input_dim+1), e))
	if add_mu_log_sigma_layer:
		layer_structures.append(np.repeat(np.arange(1, input_dim+1)[:,np.newaxis], 2, axis=1).T.flatten())
	masks = []
	for l in range(len(layer_structures)-1):
		masks.append(triangular_matrix_mask(layer_structures[l+1], layer_structures[l])[np.newaxis, :, :])
	if b_normalization:
		for i in range(len(masks)):
			# normalize rows
			# masks[i] = (masks[i].shape[2])*masks[i]/masks[i].sum(2)[:,:,np.newaxis] 
			# masks[i] = masks[i]/np.sqrt(masks[i].sum(2))[:,:,np.newaxis] 
			masks[i] = np.sqrt(masks[i].shape[2])*masks[i]/np.sqrt(masks[i].sum(2))[:,:,np.newaxis] 
			# masks[i] = masks[i]/np.sqrt(masks[i].sum(2))[:,:,np.newaxis] 
	return masks

def tf_get_mask_list_for_MADE(input_dim, layer_expansions, add_mu_log_sigma_layer=False, b_normalization=True):
	masks_np = get_mask_list_for_MADE(input_dim, layer_expansions, add_mu_log_sigma_layer=add_mu_log_sigma_layer, b_normalization=b_normalization)
	masks_tf = []
	for mask_np in masks_np:
		masks_tf.append(tf.py_func(lambda x: x, [mask_np.astype(np.float32),], [tf.float32], name="masks_list", stateful=False)[0])
	return masks_tf

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "{} {}".format(s, size_name[i])

def visualize_images_from_dists(sess, input_dict, batch, inference_sample, generative_sample, save_dir = '.', postfix = ''):
	sample = batch['observed']['data']
	sample_properties = batch['observed']['properties']
	for obs_type in ['flat', 'image']:
		if obs_type=='image' and sample[obs_type] is not None:
			real_sample_np = sample[obs_type]
			inference_sample_np = sess.run(inference_sample[obs_type], feed_dict = input_dict)
			generative_sample_np = sess.run(generative_sample[obs_type], feed_dict = input_dict)
			
			samples_params_np = np.array([np.array([]), real_sample_np, inference_sample_np, generative_sample_np])[1:]
			samples_params_np_interleaved = interleave_data(samples_params_np)
			visualize_images(samples_params_np_interleaved, save_dir = save_dir, postfix = postfix+'_'+obs_type)

			# MNIST
			image_min = 0
			image_max = 1

			inference_sample_np_clipped = np.clip(inference_sample_np, image_min, image_max)
			generative_sample_np_clipped = np.clip(generative_sample_np, image_min, image_max)
			samples_params_np = np.array([np.array([]), real_sample_np, inference_sample_np_clipped, generative_sample_np_clipped])[1:]
			samples_params_np_interleaved = interleave_data(samples_params_np)
			visualize_images(samples_params_np_interleaved, save_dir = save_dir, postfix = postfix+'_'+obs_type+'_2')

			# REAL IMAGES
			image_min = -1
			image_max = 1

			inference_sample_np_clipped = np.clip(inference_sample_np, image_min, image_max)
			generative_sample_np_clipped = np.clip(generative_sample_np, image_min, image_max)
			samples_params_np = np.array([np.array([]), real_sample_np, inference_sample_np_clipped, generative_sample_np_clipped])[1:]
			samples_params_np_interleaved = interleave_data(samples_params_np)
			visualize_images(samples_params_np_interleaved, save_dir = save_dir, postfix = postfix+'_'+obs_type+'_3')

# Values for gate_gradients.
GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2
def clipped_optimizer_minimize(optimizer, loss, global_step=None, var_list=None,
							   gate_gradients=GATE_OP, aggregation_method=None,
							   colocate_gradients_with_ops=False, name=None,
							   grad_loss=None, clip_param=None):

	grads_and_vars = optimizer.compute_gradients(
		loss, var_list=var_list, gate_gradients=gate_gradients,
		aggregation_method=aggregation_method,
		colocate_gradients_with_ops=colocate_gradients_with_ops,
		grad_loss=grad_loss)
	
	if clip_param is not None and clip_param>0:
		clipped_grads_and_vars = [(tf.clip_by_norm(grad, clip_param), var) if grad is not None else (grad, var) for grad, var in grads_and_vars]
	elif clip_param<0: pdb.set_trace()
	else: clipped_grads_and_vars = grads_and_vars
	
	vars_with_grad = [v for g, v in clipped_grads_and_vars if g is not None]

	if not vars_with_grad:
		raise ValueError(
			"No gradients provided for any variable, check your graph for ops"
			" that do not support gradients, between variables %s and loss %s." %
			([str(v) for _, v in clipped_grads_and_vars], loss))

	return optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step, name=name)



# Binary stochastic neuron with straight through estimator
# https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
def binaryRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name) 

        # For Tensorflow v0.11 and below use:
        #with g.gradient_override_map({"Floor": "Identity"}):
        #    return tf.round(x, name=name)

def bernoulliSample(x):
    """
    Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
    using the straight through estimator for the gradient.

    E.g.,:
    if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
    and the gradient will be pass-through (identity).
    """
    g = tf.get_default_graph()

    with ops.name_scope("BernoulliSample") as name:
        with g.gradient_override_map({"Ceil": "Identity","Sub": "BernoulliSample_ST"}):
            return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)

@ops.RegisterGradient("BernoulliSample_ST")
def bernoulliSample_ST(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]

def passThroughSigmoid(x, slope=1):
    """Sigmoid that uses identity function as its gradient"""
    g = tf.get_default_graph()
    with ops.name_scope("PassThroughSigmoid") as name:
        with g.gradient_override_map({"Sigmoid": "Identity"}):
            return tf.sigmoid(x, name=name)

def binaryStochastic_ST(x, slope_tensor=None, pass_through=False, stochastic=False):
    """
    Sigmoid followed by either a random sample from a bernoulli distribution according
    to the result (binary stochastic neuron) (default), or a sigmoid followed by a binary
    step function (if stochastic == False). Uses the straight through estimator.
    See https://arxiv.org/abs/1308.3432.

    Arguments:
    * x: the pre-activation / logit tensor
    * slope_tensor: if passThrough==False, slope adjusts the slope of the sigmoid function
        for purposes of the Slope Annealing Trick (see http://arxiv.org/abs/1609.01704)
    * pass_through: if True (default), gradient of the entire function is 1 or 0;
        if False, gradient of 1 is scaled by the gradient of the sigmoid (required if
        Slope Annealing Trick is used)
    * stochastic: binary stochastic neuron if True (default), or step function if False
    """
    if slope_tensor is None:
        slope_tensor = tf.constant(1.0)

    if pass_through:
        p = passThroughSigmoid(x)
    else:
        p = tf.sigmoid(slope_tensor*x)

    if stochastic:
        return bernoulliSample(p)
    else:
        return binaryRound(p) # if x>=9e-8 then return 1 else 0

def depthwise_conv2d_transpose(value, filter, output_shape, strides, padding='SAME', name=None):
    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    value = ops.convert_to_tensor(value, name="value")
    filter = ops.convert_to_tensor(filter, name="filter")
    return gen_nn_ops.depthwise_conv2d_native_backprop_input(
        input_sizes=output_shape_,
        filter=filter,
        out_backprop=value,
        strides=strides,
        padding=padding,
        name=name)

def tf_manual_conv2d_n_parameters(filter_height, filter_width, input_channels, output_channels):
    return output_channels + filter_height*filter_width*input_channels*output_channels

def upsample_bilinear_2x(input):
    output_shape = input.get_shape().as_list()
    output_shape[1] = output_shape[1]*2
    output_shape[2] = output_shape[2]*2
    output_shape[3] = output_shape[3]*7

    f = [[0.25, 0.5, 0.25],
         [0.5, 1, 0.5],
         [0.25, 0.5, 0.25]]
    f = np.array(f)
    f = np.expand_dims(f, 2)
    f = np.expand_dims(f, 3)
    f = np.tile(f, (1, 1, input.get_shape().as_list()[3], 1))
    f = f.astype(np.float32)
    # f = np.tile(f, [1, 1, 1, 7])

    print(input, f.shape, output_shape, [1,2,2,1])
    return depthwise_conv2d_transpose(input, f, output_shape, [1,2,2,1])


def tf_manual_batched_conv2d(input_ims, filters, strides = [1, 1], padding='VALID', transpose=False):
    # input_ims shape: batch_size x height x width x input_channels
    # filters shape: batch_size x filter_height x filter_width x input_channels x output_channels
    height = input_ims.get_shape()[1].value
    width = input_ims.get_shape()[2].value
    input_channels = input_ims.get_shape()[3].value
    filter_height = filters.get_shape()[1].value
    filter_width = filters.get_shape()[2].value
    output_channels = filters.get_shape()[4].value
    if transpose:
        if padding == 'VALID': 
            projected_out_height_low, projected_out_height_high = (height-1)*strides[0]+filter_height-1+1, height*strides[0]+filter_height-1
            projected_out_width_low, projected_out_width_high = (width-1)*strides[1]+filter_width-1+1,  width*strides[1]+filter_width-1
            projected_out_height, projected_out_width = projected_out_height_low, projected_out_width_low, 
        elif padding == 'SAME': 
            projected_out_height_low, projected_out_height_high = (height-1)*strides[0]+1, height*strides[0]
            projected_out_width_low, projected_out_width_high =  (width-1)*strides[1]+1, width*strides[1]
            projected_out_height, projected_out_width = projected_out_height_high, projected_out_width_high
    else:
        if padding == 'VALID':
            projected_out_height = int(np.ceil((float(height-filter_height+1)/float(strides[0]))))
            projected_out_width = int(np.ceil((float(width-filter_width+1)/float(strides[1]))))
        elif padding == 'SAME':
            projected_out_height = int(np.ceil(float(height)/float(strides[0])))
            projected_out_width = int(np.ceil(float(width)/float(strides[1])))
    assert (input_channels == filters.get_shape()[3].value)
    assert (padding == 'VALID' or padding == 'SAME')
    assert (len(strides) == 2 and strides[0] > 0 and strides[1] > 0)

    input_ims_r = tf.reshape(tf.transpose(input_ims, [1, 2, 0, 3]), [1, height, width, -1])
    filters_r = tf.reshape(tf.transpose(filters, [1, 2, 0, 3, 4]), [filter_height, filter_width, -1, output_channels])
    filters_r_2 = tf.reshape(tf.transpose(filters, [1, 2, 0, 4, 3]), [filter_height, filter_width, -1, output_channels])

    if transpose: 
        # output_shape = [tf.shape(input_ims)[0], projected_out_height, projected_out_width, output_channels] 
        # out_conv_1 = tf.nn.conv2d_transpose(input_ims, tf.transpose(filters[0,:,:,:,:], [0,1,3,2]), strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape) 
        # out_conv = depthwise_conv2d_transpose(input_ims, tf.transpose(filters, [0,1,2,4,3]), strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape) 
        # out_conv_2 = depthwise_conv2d_transpose(input_ims, filters, strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape) 

        # output_shape_1 = [1, projected_out_height, projected_out_width, input_ims.get_shape()[0].value*input_channels*output_channels] 
        # output_shape_2 = [1, projected_out_height, projected_out_width, tf.shape(input_ims)[0]*input_channels*output_channels] 
        # output_shape_3 = [1, projected_out_height, projected_out_width, -1] 

        # out_conv = tf.nn.depthwise_conv2d(input_ims_r, filter=filters_r, strides=[1, strides[0], strides[1], 1], padding=padding)
        # orig = gen_nn_ops.depthwise_conv2d_native_backprop_input(input_sizes=[1,16,15,30], filter=filters_r, out_backprop=out_conv, strides=[1, strides[0], strides[1], 1], padding=padding)
        # orig2 = gen_nn_ops.depthwise_conv2d_native_backprop_input(input_sizes=[1,16,15,11], filter=tf.transpose(filters_r, [0, 1, 3, 2]), out_backprop=out_conv, strides=[1, strides[0], strides[1], 1], padding=padding)

        # filter_k = tf.reshape(filters_r, [filters_r.get_shape()[0].value, filters_r.get_shape()[1].value, -1, 1])

        # (1, 16, 15, 30) --> (1, 8, 8, 330) using (5, 4, 30, 11) |||||||||||  (1, 8, 8, 30) --> (1, 16, 15, 330) using (5, 4, 30, 11)

        # out_conv_2 = depthwise_conv2d_transpose(input_ims_r, filter=tf.transpose(filters_r, [0,1,3,2]), strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape_2) 
        # out_conv_3 = depthwise_conv2d_transpose(input_ims_r, filter=filters_r, strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape_2) 
        # out_conv_4 = depthwise_conv2d_transpose(input_ims_r, filter=filters_r_2, strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape_2) 

        # out_conv_5 = depthwise_conv2d_transpose(input_ims_r, filter=tf.transpose(filters_r, [0,1,3,2]), strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape_3) 
        # out_conv_6 = depthwise_conv2d_transpose(input_ims_r, filter=filters_r, strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape_3) 
        # out_conv_7 = depthwise_conv2d_transpose(input_ims_r, filter=filters_r_2, strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape_3) 

        # out_conv_7 = depthwise_conv2d_transpose(input_ims_r, filter=filters_r, strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape_2) 
        # out_conv_7 = depthwise_conv2d_transpose(input_ims_r, filter=tf.transpose(filters_r, [0, 1, 3, 2]), strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape_2) 
        
        # sosf = upsample_bilinear_2x(tf.tile(input_ims_r, [5,1,1,1])[:,:,:,:28])
        # out_conv_7 = depthwise_conv2d_transpose(input_ims_r, filter=filters_r, strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape_1) 
        # out_conv_8 = depthwise_conv2d_transpose(input_ims_r, filter=tf.transpose(filters_r, [0, 1, 3, 2]), strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape_1) 
        # out_conv_1 = depthwise_conv2d_transpose(input_ims_r, filter=filters_r, strides=[1, strides[0], strides[1], 1], output_shape=output_shape_1) 
        # out_conv_2 = depthwise_conv2d_transpose(input_ims_r, filter=tf.transpose(filters_r, [0, 1, 3, 2]), strides=[1, strides[0], strides[1], 1], output_shape=output_shape_1) 

        # init = tf.initialize_all_variables()
        # sess = tf.InteractiveSession()  
        # sess.run(init)
        # # sess.run(input_ims_r).shape
        # # sess.run(filters_r).shape
        # pdb.set_trace()
        # sess.run(out_conv).shape
        # sess.run(orig).shape
        # sess.run(orig2).shape

        input_ims_outted = tf.layers.conv2d(inputs=input_ims, filters=output_channels, kernel_size=[1, 1], strides=[1, 1], padding="VALID", use_bias=True, activation=None)
        input_ims_outted_r = tf.reshape(tf.transpose(input_ims_outted, [1, 2, 0, 3]), [1, input_ims_outted.get_shape()[1].value, input_ims_outted.get_shape()[2].value, -1])
        new_filter = filters[:,:,:,0,:,np.newaxis]
        new_filter_r = tf.reshape(tf.transpose(new_filter, [1, 2, 0, 3, 4]), [new_filter.get_shape()[1].value, new_filter.get_shape()[2].value, -1, new_filter.get_shape()[4].value])
        out_conv = gen_nn_ops.depthwise_conv2d_native_backprop_input(input_sizes=[1, projected_out_height, projected_out_width, tf.shape(input_ims)[0]*output_channels], filter=new_filter_r, out_backprop=input_ims_outted_r, strides=[1, strides[0], strides[1], 1], padding=padding)
        out_reshaped = tf.reshape(out_conv, [out_conv.get_shape()[1].value, out_conv.get_shape()[2].value, -1, output_channels, 1])
        out = tf.transpose(out_reshaped, [2, 0, 1, 3, 4])[:,:,:,:,0]
        
        init = tf.initialize_all_variables()
        sess = tf.InteractiveSession()  
        sess.run(init)
        # sess.run(out).shape
        pdb.set_trace()
    else: 
        out_conv = tf.nn.depthwise_conv2d(input_ims_r, filter=filters_r, strides=[1, strides[0], strides[1], 1], padding=padding)
        out_reshaped = tf.reshape(out_conv, [out_conv.get_shape()[1].value, out_conv.get_shape()[2].value, -1, input_channels, output_channels])
        out = tf.reduce_sum(tf.transpose(out_reshaped, [2, 0, 1, 3, 4]), axis=3) # sum across input_channels
        pdb.set_trace()
    
    
    out_height = out.get_shape()[1].value 
    out_width = out.get_shape()[2].value
    # out_conv shape: 1 x out_height x out_width x batch_size*input_channels*output_channels
    assert(out_height == projected_out_height)
    assert(out_width == projected_out_width)
    # out shape is: batch_size x out_height x out_width x out_channels 
    return out

def tf_manual_batched_conv2d_layer(input_ims, filters, biases, strides = [1, 1], padding='VALID', use_bias = True, nonlinearity=None, transpose=False, verbose=False):
    # input_ims shape: batch_size x height x width x input_channels
    # filters shape: batch_size x filter_height x filter_width x input_channels x output_channels
    # biases shape: batch_size x output_channels
    conv_ims = tf_manual_batched_conv2d(input_ims, filters, strides=strides, padding=padding, transpose=transpose)
    if verbose: print('\nConvolution from, '+ str(input_ims.get_shape().as_list()) + ' to, ' + str(conv_ims.get_shape().as_list())+'\n')
    if use_bias: conv_ims_biased = conv_ims + biases[:, np.newaxis, np.newaxis, :]
    else: conv_ims_biased = conv_ims
    if nonlinearity is not None: return nonlinearity(conv_ims_biased)
    else: return conv_ims_biased

def tf_manual_conv2d(input_ims, filters, strides = [1, 1], padding='VALID', transpose=False):
    # input_ims shape: batch_size x height x width x input_channels
    # filters shape: filter_height x filter_width x input_channels x output_channels
    height = input_ims.get_shape()[1].value
    width = input_ims.get_shape()[2].value
    input_channels = input_ims.get_shape()[3].value
    filter_height = filters.get_shape()[0].value
    filter_width = filters.get_shape()[1].value
    output_channels = filters.get_shape()[3].value
    if transpose:
        if padding == 'VALID': 
            projected_out_height_low, projected_out_height_high = (height-1)*strides[0]+filter_height-1+1, height*strides[0]+filter_height-1
            projected_out_width_low, projected_out_width_high = (width-1)*strides[1]+filter_width-1+1,  width*strides[1]+filter_width-1
            projected_out_height, projected_out_width = projected_out_height_low, projected_out_width_low, 
        elif padding == 'SAME': 
            projected_out_height_low, projected_out_height_high = (height-1)*strides[0]+1, height*strides[0]
            projected_out_width_low, projected_out_width_high =  (width-1)*strides[1]+1, width*strides[1]
            projected_out_height, projected_out_width = projected_out_height_high, projected_out_width_high
    else:
        if padding == 'VALID':
            projected_out_height = int(np.ceil((float(height-filter_height+1)/float(strides[0]))))
            projected_out_width = int(np.ceil((float(width-filter_width+1)/float(strides[1]))))
        elif padding == 'SAME':
            projected_out_height = int(np.ceil(float(height)/float(strides[0])))
            projected_out_width = int(np.ceil(float(width)/float(strides[1])))
    assert (input_channels == filters.get_shape()[2].value)
    assert (padding == 'VALID' or padding == 'SAME')
    assert (len(strides) == 2 and strides[0] > 0 and strides[1] > 0)

    if transpose: 
        output_shape = [tf.shape(input_ims)[0], projected_out_height, projected_out_width, output_channels] 
        out_conv = tf.nn.conv2d_transpose(input_ims, tf.transpose(filters, [0,1,3,2]), strides=[1, strides[0], strides[1], 1], padding=padding, output_shape=output_shape) 
    else: 
        out_conv = tf.nn.conv2d(input_ims, filters, strides=[1, strides[0], strides[1], 1], padding=padding)

    out_height = out_conv.get_shape()[1].value 
    out_width = out_conv.get_shape()[2].value
    assert(out_height == projected_out_height)
    assert(out_width == projected_out_width)
    # out_conv shape is: batch_size x out_height x out_width x out_channels 
    return out_conv

def tf_manual_conv2d_layer(input_ims, filters, biases, strides = [1, 1], padding='VALID', use_bias = True, nonlinearity=None, transpose=False, verbose=False):
    # input_ims shape: batch_size x height x width x input_channels
    # filters shape: filter_height x filter_width x input_channels x output_channels
    # biases shape: output_channels
    conv_ims = tf_manual_conv2d(input_ims, filters, strides=strides, padding=padding, transpose=transpose)
    if verbose: print('\nConvolution from, '+ str(input_ims.get_shape().as_list()) + ' to, ' + str(conv_ims.get_shape().as_list())+'\n')
    if use_bias: conv_ims_biased = conv_ims + biases[np.newaxis, np.newaxis, np.newaxis, :]
    else: conv_ims_biased = conv_ims
    if nonlinearity is not None: return nonlinearity(conv_ims_biased)
    else: return conv_ims_biased

# batch_size = 10
# im_height = 16
# im_width = 15
# n_in_channels = 3

# filter_height = 5
# filter_width = 4
# n_out_channels = 11

# strides = [2, 2]
# padding='SAME'
# nonlinearity = tf.nn.relu
# use_bias = True
# transpose = True
# verbose = True
# random_param = False

# input_ims =  tf.random_uniform(shape=(batch_size, im_height, im_width, n_in_channels), dtype=tf.float32) 

# if random_param:
#     input_filters_fixed =  tf.random_uniform(shape=(filter_height, filter_width, n_in_channels, n_out_channels), dtype=tf.float32)
#     input_filters_fixed_batched = tf.tile(input_filters_fixed[np.newaxis, :, :, : ,:], [batch_size, 1, 1, 1, 1])
#     input_filters_batched = tf.random_uniform(shape=(batch_size, filter_height, filter_width, n_in_channels, n_out_channels), dtype=tf.float32)
#     input_biases_fixed = tf.random_uniform(shape=(n_out_channels,), dtype=tf.float32)
#     input_biases_fixed_batched = tf.tile(input_biases_fixed[np.newaxis,:], [batch_size, 1])
#     input_biases_batched = tf.random_uniform(shape=(batch_size, n_out_channels), dtype=tf.float32)
# else:
#     n_parameter = tf_manual_conv2d_n_parameters(filter_height, filter_width, n_in_channels, n_out_channels)
#     fixed_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
#     batched_parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(batch_size, 1)), units = n_parameter, use_bias = False, activation = None)

#     param_index = 0
#     input_filters_fixed_vec, param_index = slice_parameters(fixed_parameters, param_index, filter_height*filter_width*n_in_channels*n_out_channels) 
#     input_biases_fixed_vec, param_index = slice_parameters(fixed_parameters, param_index, n_out_channels) 

#     param_index = 0
#     input_filters_batched_vec, param_index = slice_parameters(batched_parameters, param_index, filter_height*filter_width*n_in_channels*n_out_channels) 
#     input_biases_batched_vec, param_index = slice_parameters(batched_parameters, param_index, n_out_channels) 

#     input_filters_fixed = tf.reshape(input_filters_fixed_vec[0, ...], [filter_height, filter_width, n_in_channels, n_out_channels])
#     input_filters_fixed_batched = tf.tile(input_filters_fixed[np.newaxis, :, :, : ,:], [batch_size, 1, 1, 1, 1])
#     input_filters_batched = tf.reshape(input_filters_batched_vec, [-1, filter_height, filter_width, n_in_channels, n_out_channels])

#     input_biases_fixed = input_biases_fixed_vec[0, :]
#     input_biases_fixed_batched = tf.tile(input_biases_fixed[np.newaxis,:], [batch_size, 1])
#     input_biases_batched = input_biases_batched_vec

# conv_ims_fixed = tf_manual_conv2d_layer(input_ims, input_filters_fixed, input_biases_fixed, strides=strides, padding=padding, use_bias=use_bias, nonlinearity=nonlinearity, transpose=transpose, verbose=verbose)
# conv_ims_fixed_batched = tf_manual_batched_conv2d_layer(input_ims, input_filters_fixed_batched, input_biases_fixed_batched, strides=strides, padding=padding, use_bias=use_bias, nonlinearity=nonlinearity, transpose=transpose, verbose=verbose)
# conv_ims_batched = tf_manual_batched_conv2d_layer(input_ims, input_filters_batched, input_biases_batched, strides=strides, padding=padding, use_bias=use_bias, nonlinearity=nonlinearity, transpose=transpose, verbose=verbose)

# pdb.set_trace()

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# input_ims_np, conv_ims_fixed_np, conv_ims_fixed_batched_np, conv_ims_batched_np = sess.run([input_ims, conv_ims_fixed, conv_ims_fixed_batched, conv_ims_batched])
# print(np.abs(conv_ims_batched_np-conv_ims_fixed_np).max())
# print(np.abs(conv_ims_fixed_batched_np-conv_ims_fixed_np).max())

# import time
# print('Start Timer: ')
# start = time.time();
# for i in range(10000):
#     conv_ims_fixed_np = sess.run(conv_ims_fixed)
# end = time.time()
# print('Time: {:.3f}\n'.format((end - start)))

# import time
# print('Start Timer: ')
# start = time.time();
# for i in range(10000):
#     conv_ims_fixed_np = sess.run(conv_ims_fixed_batched)
# end = time.time()
# print('Time: {:.3f}\n'.format((end - start)))
# pdb.set_trace()


# print((projected_out_height_low, projected_out_height_high), (projected_out_width_low, projected_out_width_high))
# print(projected_out_height, projected_out_width) 
# print('\n\n\n')
# # # first three are the same, last two must be different in height and width
# print(tf.nn.conv2d(tf.random_uniform(shape=(1, projected_out_height, projected_out_width, 3), dtype=tf.float32), filters, strides=[1, strides[0], strides[1], 1], padding=padding))
# print(tf.nn.conv2d(tf.random_uniform(shape=(1, projected_out_height_low, projected_out_width_low, 3), dtype=tf.float32), filters, strides=[1, strides[0], strides[1], 1], padding=padding))
# print(tf.nn.conv2d(tf.random_uniform(shape=(1, projected_out_height_high, projected_out_width_high, 3), dtype=tf.float32), filters, strides=[1, strides[0], strides[1], 1], padding=padding))
# print(tf.nn.conv2d(tf.random_uniform(shape=(1, projected_out_height_low-1, projected_out_width_low-1, 3), dtype=tf.float32), filters, strides=[1, strides[0], strides[1], 1], padding=padding))
# print(tf.nn.conv2d(tf.random_uniform(shape=(1, projected_out_height_high+1, projected_out_width_high+1, 3), dtype=tf.float32), filters, strides=[1, strides[0], strides[1], 1], padding=padding))
# print('\n\n\n')



# batch_size = 10
# im_size = 12
# n_channels = 3

# squeeze_func = squeeze_realnpv_depthspace
# # squeeze_func = squeeze_realnpv
# # unsqueeze_func = unsqueeze_realnpv_depthspace
# unsqueeze_func = unsqueeze_realnpv

# input_ims =  tf.random_uniform(shape=(batch_size, im_size, im_size, n_channels), dtype=tf.float32) # required for some transforms#
# squeezed_ims = squeeze_func(input_ims, squeeze_order=[1,4,2,3], verbose=True)
# unsqueezed_ims = unsqueeze_func(squeezed_ims, squeeze_order=[1,4,2,3], verbose=True)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)

# import time
# print('Start Timer: ')
# start = time.time();
# for i in range(50000):
#     input_ims_np, squeezed_ims_np, unsqueezed_ims_np = sess.run([input_ims, squeezed_ims, unsqueezed_ims])
# end = time.time()
# print('Time: {:.3f}\n'.format((end - start)))
# # np.abs(input_ims_np-unsqueezed_ims_np).max()
# pdb.set_trace()

















# input_tf = tf.placeholder(tf.float32, [5])
# z = binaryStochastic_ST(input_tf)
# f_d = {input_tf: np.asarray([-10, -1e-8, 0, 9e-8, 10])}

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# z_np = sess.run(z, feed_dict = f_d)
# print(z_np)
# pdb.set_trace()



















# print(get_mask_list_for_MADE(3, [2,4], add_mu_sigma_layer=True))


# n = 4
# print(normalized_bell_np(np.arange(-1,1+1/n,1/n)))

# dim = 5
# batch = 4 
# k_start = 5
# params = tf.random_normal((batch, sum(list(range(max(2, k_start), dim+1)))), 0, 1, dtype=tf.float32)
# # params = tf.ones((batch, sum(list(range(max(2, k_start), dim+1)))))
# # params = None

# rot_matrix_tf = householder_rotations_tf(n=dim, k_start=k_start, init_reflection=-1, params=params)
# rot_matrix_tf2 = householder_rotations_tf(n=dim, k_start=k_start, init_reflection=1, params=params)
# rot_matrix_tf_a = tf.matmul(rot_matrix_tf, rot_matrix_tf, transpose_a=True)
# rot_matrix_tf_b = tf.matmul(rot_matrix_tf, rot_matrix_tf, transpose_a=True)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# out1, out2, out3, out4 = sess.run([rot_matrix_tf, rot_matrix_tf2, rot_matrix_tf_a, rot_matrix_tf_b])

# print(out1.shape)
# print(out1)
# print(out2.shape)
# print(out2)
# print(out3.shape)
# print(out3)
# print(out4.shape)
# print(out4)
# pdb.set_trace()


# end_vectors = []
# end_vectors2= []
# start_vector = np.asarray([[1., 0., 0.]]).T
# for i in range(out1.shape[0]):
# 	rot_matrix = out1[i]
# 	rot_matrix2 = out2[i]
# 	end_vectors.append(np.matmul(rot_matrix, start_vector))
# 	end_vectors2.append(np.matmul(rot_matrix2, start_vector))
# dataset = np.concatenate(end_vectors, axis=1).T
# dataset2 = np.concatenate(end_vectors2, axis=1).T

# dataset_plotter([dataset,], show_also=True)
# pdb.set_trace()
# dataset_plotter([dataset2,], show_also=True)
# pdb.set_trace()
# dataset_plotter([dataset, dataset2], show_also=True)
# pdb.set_trace()


# block_size = [1, 1]
# full_size = [3, 3]
# block_triangular = block_triangular_ones(full_size, block_size, trilmode=-1)
# block_diagonal = block_diagonal_ones(full_size, block_size)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# out1, out2 = sess.run([block_triangular, block_diagonal])
# print(out1)
# print(out2)
# pdb.set_trace()



# sys.stdout = PrintSnooper(sys.stdout) # put it to the beginning of script

# print('postload')
# for var in tf.trainable_variables():
#     print('var', var.name, var.get_shape(), sess.run(tf.reduce_sum(var ** 2)))




# def save_checkpoint(saver, sess, global_step, exp_dir):
# 	if not os.path.exists(exp_dir): os.makedirs(exp_dir)
# 	# checkpoint_file = os.path.join(exp_dir , 'model.ckpt')
# 	# saver.save(sess, exp_dir+'/model' global_step=global_step)
# 	# saver.export_meta_graph(os.path.join(exp_dir , 'model.meta'))
# 	saver.save(sess, exp_dir+'/model', global_step=global_step)
# 	saver.export_meta_graph(exp_dir+'/model.meta')


# def load_checkpoint(saver, sess, exp_dir):	
# 	# saver = tf.train.import_meta_graph(os.path.join(exp_dir , 'model.meta'))
# 	saver.restore(sess, exp_dir+'model')

# 	# ckpt = tf.train.get_checkpoint_state(exp_dir+'/checkpoints')
# 	# if ckpt and ckpt.model_checkpoint_path:
# 	# 	saver.restore(sess, ckpt.model_checkpoint_path)
# 	# 	print('Loaded checkpoint from file: ' + ckpt.model_checkpoint_path)
# 	# else: print('No Checkpoint..')



# def network(x, y, alpha):
# 	image_shape = (28, 28, 1)
# 	x_flat = tf.reshape(x, [-1, 1, 784]) 
# 	y_flat = tf.reshape(y, [-1, 1, 784]) 
# 	concat_input = tf.concat([x_flat, y_flat, alpha], axis=-1)
# 	n_output_size = 784
# 	concat_input_flat = tf.reshape(concat_input, [-1,  concat_input.get_shape().as_list()[-1]])
# 	pdb.set_trace()

# 	lay1_flat = tf.layers.dense(inputs = concat_input_flat, units = self.config['n_decoder'], activation = activation_function)
# 	# lay1_flat = tf.concat([lay1_flat, tf.reshape(alpha, [-1,  alpha.get_shape().as_list()[-1]])], axis=-1)
# 	lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_decoder'], activation = activation_function)
# 	# lay2_flat = tf.concat([lay2_flat, tf.reshape(alpha, [-1,  alpha.get_shape().as_list()[-1]])], axis=-1)
# 	lay3_flat = tf.layers.dense(inputs = lay2_flat, units = self.config['n_decoder'], activation = activation_function)
# 	# lay3_flat = tf.concat([lay3_flat, tf.reshape(alpha, [-1,  alpha.get_shape().as_list()[-1]])], axis=-1)
# 	lay5_flat = tf.layers.dense(inputs = lay1_flat, units = n_output_size, activation = None)
# 	out = tf.reshape(lay5_flat, [-1, *x.get_shape().as_list()[1:]])
	


# N = 1
# batch_size = 30
# with tf.Graph().as_default():
# 	x = tf.placeholder(tf.float32, shape=[None, N])
# 	spur_1 = tf.placeholder(tf.float32, shape=[None, 784])
# 	spur_2 = tf.placeholder(tf.float32, shape=[None, 784])
# 	input_x = tf.concat([x, spur], axis=-1)
# 	h = tf.layers.dense(inputs = input_x, units = 784, activation = tf.nn.sigmoid)
# 	y = tf.layers.dense(inputs = h, units = 784, activation = tf.nn.sigmoid)
# 	vv = tf.gradients(y, x)
# 	jacobian = tf_batch_reduced_jacobian(y, x)

# 	init = tf.global_variables_initializer()
# 	saver = tf.train.Saver()
# 	sess = tf.InteractiveSession()
# 	sess.run(init)

# x_val = np.random.randn(batch_size, N)
# spur1_val = np.random.randn(batch_size, 784)
# spur2_val = np.random.randn(batch_size, 784)
# y_val, vv_val = sess.run([y, vv], feed_dict={x:x_val, spur:spur_val})
# start = time.time(); jacobian_val = sess.run(jacobian, feed_dict={x:x_val, spur_1:spur1_val, spur_1:spur1_val}); end = time.time(); timing= end-start
# print(jacobian_val)
# print('Timing (s),', timing)
# pdb.set_trace()








# N = 1
# batch_size = 30
# with tf.Graph().as_default():
# 	x = tf.placeholder(tf.float32, shape=[None, N])
# 	spur = tf.placeholder(tf.float32, shape=[None, 400])
# 	input_x = tf.concat([x, spur], axis=-1)
# 	h = tf.layers.dense(inputs = input_x, units = 784, activation = tf.nn.sigmoid)
# 	y = tf.layers.dense(inputs = h, units = 784, activation = tf.nn.sigmoid)
# 	vv = tf.gradients(y, x)
# 	jacobian = tf_batch_reduced_jacobian(y, x)

# 	init = tf.global_variables_initializer()
# 	saver = tf.train.Saver()
# 	sess = tf.InteractiveSession()
# 	sess.run(init)

# x_val = np.random.randn(batch_size, N)
# spur_val = np.random.randn(batch_size, 400)
# y_val, vv_val = sess.run([y, vv], feed_dict={x:x_val, spur:spur_val})
# start = time.time(); 
# for i in range(100):
# 	jacobian_val = sess.run(jacobian, feed_dict={x:x_val, spur:spur_val}); 
# end = time.time(); timing= end-start
# print(jacobian_val)
# print('Timing (s),', timing)
# pdb.set_trace()




# jacobian1 = tf_jacobian(tf.reduce_sum(y, axis=[0], keep_dims=True), x)
# jacobian2 = tf_jacobian(tf.reduce_sum(y, axis=[0], keep_dims=False), x)
# jacobian3 = tf_jacobian(y, x)

# start = time.time(); jacobian_val1 = sess.run(jacobian1, feed_dict={x:x_val}); end = time.time(); timing1= end-start
# start = time.time(); jacobian_val2 = sess.run(jacobian2, feed_dict={x:x_val}); end = time.time(); timing2= end-start
# start = time.time(); jacobian_val3 = sess.run(jacobian3, feed_dict={x:x_val}); end = time.time(); timing3= end-start
# print('Timing (s),', timing1)
# print('Timing (s),', timing2)
# print('Timing (s),', timing3)
# print(jacobian_val1)
# print(jacobian_val2)
# print(jacobian_val3)
# nonzero_jacobian = np.concatenate([jacobian_val3[0,:,0,:][np.newaxis,:,:], jacobian_val3[1,:,1,:][np.newaxis,:,:], jacobian_val3[2,:,2,:][np.newaxis,:,:]], axis=0)
# print('Different?', np.any(np.abs(jacobian_val2.transpose([1,0,2])-nonzero_jacobian)>0))
# print('Different?', np.any(np.abs(jacobian_val1-jacobian_val3)>0))
# print('Different?', np.any(np.abs(jacobian_val2-jacobian_val3)>0))

