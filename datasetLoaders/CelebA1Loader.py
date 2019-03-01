import pdb
import numpy as np
import math
import scipy.misc
import pickle
import zlib
import os
import time
import scipy.io as sio
import helper
from PIL import Image
from glob import glob
import tensorflow as tf

# from tensorflow.examples.tutorials.mnist import input_data



class DataLoader:
	def __init__(self, batch_size, time_steps, image_size=32, data_type='regular', b_context=False, cuda=True):
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.image_size = image_size
		self.mode = 'Train'
		self.cuda = cuda
		self.b_context = b_context
		self.iter = 0

		self.dataset_path = '/home/mcgemici/lsun_celebA_data/celebA2/'
		if self.cuda: self.data_format = 'NCHW'
		else: self.data_format = 'NHWC'

		self.train_loader = get_loader(self.dataset_path, self.batch_size, self.image_size, self.data_format, split='train', is_grayscale=False)
		self.test_loader = get_loader(self.dataset_path, self.batch_size, self.image_size, self.data_format, split='test', is_grayscale=False)
		self.valid_loader = get_loader(self.dataset_path, self.batch_size, self.image_size, self.data_format, split='valid', is_grayscale=False)
		pdb.set_trace()

		x = norm_img(self.train_loader)

		self.train_path = '/home/mcgemici/lsun_celebA_data/celebA2/splits/train/'
		self.test_path = '/home/mcgemici/lsun_celebA_data/celebA2/splits/test/'
		self.valid_path = '/home/mcgemici/lsun_celebA_data/celebA2/splits/valid/'
		
		data_loader = get_loader(data_path, config.batch_size, config.input_scale_size,
            config.data_format, config.split)

		pdb.set_trace()

		self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
		imreadImg = imread(self.data[0]);
		if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
			self.c_dim = imread(self.data[0]).shape[-1]
		else: self.c_dim = 1


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


def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)


def get_loader(root, batch_size, scale_size, data_format, split='train', is_grayscale=False, seed=123):
    dataset_name = os.path.basename(root)
    root = os.path.join(root, 'splits', split)

    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(root, ext))

        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png
        
        if len(paths) != 0:
            break

        # pdb.set_trace()
    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    if dataset_name in ['CelebA']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    else:
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    return tf.to_float(queue)


