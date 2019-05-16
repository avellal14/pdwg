"""Random variable transformation classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import helper 
import pdb 
import numpy as np
import math 
from random import shuffle 
import time
import scipy
from scipy import special

class PlanarFlow():
    """
    Planar Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, input_dim, parameters, name='planar_transform'):   
        self._parameter_scale = 1
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim
        
        assert (self._input_dim > 1)

        self._parameters.get_shape().assert_is_compatible_with([None, PlanarFlow.required_num_parameters(self._input_dim)])

        self._w = tf.slice(self._parameters, [0, 0], [-1, self._input_dim])
        self._u = tf.slice(self._parameters, [0, self._input_dim], [-1, self._input_dim])
        self._b = tf.slice(self._parameters, [0, 2*self._input_dim], [-1, 1])
        self._w_t_u = tf.reduce_sum(self._w*self._u, axis=[1], keep_dims=True)
        self._w_t_w = tf.reduce_sum(self._w*self._w, axis=[1], keep_dims=True)
        self._u_tilde = (self._u+(((tf.log(1e-7+1+tf.exp(self._w_t_u))-1)+self._w_t_u)/self._w_t_w)*self._w)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):
        return input_dim+input_dim+1

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        affine = tf.reduce_sum(z0*self._w, axis=[1], keep_dims=True) + self._b
        h = tf.tanh(affine)
        z = z0+self._u_tilde*h

        h_prime_w = (1-tf.pow(h, 2))*self._w
        log_abs_det_jacobian = tf.log(1e-7+tf.abs(1+tf.reduce_sum(h_prime_w*self._u_tilde, axis=[1], keep_dims=True)))
        log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        return z, log_pdf_z

class RadialFlow():
    """
    Radial Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, input_dim, parameters, name='radial_transform'):   
        self._parameter_scale = 1
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim
        
        assert (self._input_dim > 1)

        self._parameters.get_shape().assert_is_compatible_with([None, RadialFlow.required_num_parameters(self._input_dim)])

        self._alpha = tf.slice(self._parameters, [0, 0], [-1, 1])
        self._beta = tf.slice(self._parameters, [0, 1], [-1, 1])
        self._z_ref = tf.slice(self._parameters, [0, 2], [-1, self._input_dim])
        self._alpha_tilde = tf.log(1e-7+1+tf.exp(self._alpha))
        self._beta_tilde = tf.log(1e-7+1+tf.exp(self._beta)) - self._alpha_tilde

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):
        return 1+1+input_dim

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        z_diff = z0 - self._z_ref
        r = tf.sqrt(tf.reduce_sum(tf.square(z_diff), axis=[1], keep_dims=True))
        h = 1/(self._alpha_tilde + r)
        scale = self._beta_tilde * h
        z = z0 + scale * z_diff

        log_abs_det_jacobian = tf.log(1e-7+tf.abs(tf.pow(1 + scale, self._input_dim - 1) * (1 + scale * (1 - h * r))))
        log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        return z, log_pdf_z

class SpecificOrderDimensionFlow():
    """
    Specific Order Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, input_dim, order=None, parameters=None, name='specific_order_dimension_transform'):   
        print('Unusable in a safe fashion: This creates as random ordering which changes from random seed to random seed '+ 
              'and is not saved in any checkpoint, since the order is not stored in a tf.Variable. Therefore, if the seed '+
              'does not match or if there is a random call order change, then the loaded model from the checkpoint will be wrong.')
        quit()
        self._input_dim = input_dim 
        if order is None: # a specific but random order
            # self._order = [*range(self._input_dim)]
            # shuffle(self._order) # works for subset of O(n) but not SO(n) 
            print('SpecificOrderDimensionFlow, creating random order: ')
            start = time.time()
            n_swaps = 10*self._input_dim
            self._order = [*range(self._input_dim)]     
            for t in range(n_swaps):
                index_1 = np.random.randint(len(self._order))
                index_2 = index_1
                while index_2 == index_1: index_2 = np.random.randint(len(self._order))
                temp = self._order[index_1]
                self._order[index_1] = self._order[index_2]
                self._order[index_2] = temp
            assert (n_swaps % 2 == 0) # SO(n)
            print('Time: {:.3f}\n'.format((time.time() - start)))
        else: self._order = order
        self._inverse_order = [-1]*self._input_dim
        for i in range(self._input_dim): self._inverse_order[self._order[i]] = i
        
        assert (len(self._order) == self._input_dim)
        assert (len(self._inverse_order) == self._input_dim)
        assert (parameters is None)
        assert (self._input_dim > 1)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):  
        return 0

    def get_batched_rot_matrix(self):
        batched_rot_matrix_np = np.zeros((1, self._input_dim, self._input_dim))
        for i in range(len(self._order)): batched_rot_matrix_np[0, i, self._order[i]] = 1
        return tf.constant(batched_rot_matrix_np, tf.float32)

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        z = helper.tf_differentiable_specific_shuffle_with_axis(z0, self._order, axis=1)
        log_pdf_z = log_pdf_z0
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)

        z0 = helper.tf_differentiable_specific_shuffle_with_axis(z, self._inverse_order, axis=1)
        log_pdf_z0 = log_pdf_z
        return z0, log_pdf_z0

class CustomSpecificOrderDimensionFlow():
    """
    Custom Specific Order Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, input_dim, order=None, parameters=None, name='custom_specific_order_dimension_transform'):   
        print('Unusable in a safe fashion: This creates as random ordering which changes from random seed to random seed '+ 
              'and is not saved in any checkpoint, since the order is not stored in a tf.Variable. Therefore, if the seed '+
              'does not match or if there is a random call order change, then the loaded model from the checkpoint will be wrong.')
        quit()
        self._input_dim = input_dim 
        self._sodf_1 = SpecificOrderDimensionFlow(input_dim=int(self._input_dim/2.))
        self._sodf_2 = SpecificOrderDimensionFlow(input_dim=int(self._input_dim/2.))
        
        assert (self._input_dim % 2 == 0)
        assert (parameters is None)
        assert (self._input_dim > 1)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):  
        return 0

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        z0_1 = z0[:, :int(self._input_dim/2.)] 
        z0_2 = z0[:, int(self._input_dim/2.):]
        # z_1, _ = self._sodf_1.transform(z0_1, tf.zeros(shape=[tf.shape(z0_1)[0], 1]))
        z_1 = z0_1
        z_2, _ = self._sodf_2.transform(z0_2, tf.zeros(shape=[tf.shape(z0_2)[0], 1]))
        z = tf.concat([z_1, z_2], axis=1)
        log_pdf_z = log_pdf_z0
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)

        z_1 = z[:, :int(self._input_dim/2.)] 
        z_2 = z[:, int(self._input_dim/2.):]
        # z0_1, _ = self._sodf_1.inverse_transform(z_1, tf.zeros(shape=[tf.shape(z_1)[0], 1]))
        z0_1 = z_1
        z0_2, _ = self._sodf_2.inverse_transform(z_2, tf.zeros(shape=[tf.shape(z_2)[0], 1]))
        z0 = tf.concat([z0_1, z0_2], axis=1)
        log_pdf_z0 = log_pdf_z
        return z0, log_pdf_z0

class InverseOrderDimensionFlow():
    """
    Inverse Order Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, input_dim, parameters=None, name='inverse_order_dimension_transform'):   
        self._input_dim = input_dim
        assert (parameters is None)
        assert (self._input_dim > 1)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):  
        return 0

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        z = tf.reverse(z0, axis=[-1,])
        log_pdf_z = log_pdf_z0
        return z, log_pdf_z

class PermuteDimensionFlow():
    """
    Permute Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, input_dim, parameters=None, slide_to_higher=True, name='permute_dimension_transform'):   
        self._input_dim = input_dim
        self._slide_to_higher = slide_to_higher
        self._n_slide_dims = int(float(self._input_dim)/3.)+1
        assert (self._n_slide_dims != 0)
        assert (parameters is None)
        assert (self._input_dim > 1)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):  
        return 0

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        if self._slide_to_higher:
            z = tf.concat([tf.slice(z0, [0, self._n_slide_dims], [-1, self._input_dim-self._n_slide_dims]), tf.slice(z0, [0, 0], [-1, self._n_slide_dims])], axis=1)
        else:
            z = tf.concat([tf.slice(z0, [0, self._input_dim-self._n_slide_dims], [-1, self._n_slide_dims]), tf.slice(z0, [0, 0], [-1, self._input_dim-self._n_slide_dims])], axis=1)
        log_pdf_z = log_pdf_z0
        return z, log_pdf_z

class ScaleDimensionFlow():
    """
    Scale Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, input_dim, parameters=None, scale=1/5., name='scale_dimension_transform'):   
        self._input_dim = input_dim
        self._scale = scale
        assert (parameters is None)
        assert (self._input_dim > 1)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):  
        return 0

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        z = z0*self._scale
        log_abs_det_jacobian = self._input_dim*tf.log(1e-7+self._scale)
        log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        return z, log_pdf_z

class OpenIntervalDimensionFlow():
    """
    Open Interval Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, input_dim, parameters=None, zero_one=True, name='open_interval_dimension_transform'):  #real
        self._input_dim = input_dim
        self._zero_one = zero_one
        assert (parameters is None)
        assert (self._input_dim > 1)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):  
        return 0

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        if self._zero_one:
            # z = tf.nn.sigmoid(z0)
            z = tf.nn.sigmoid(5.*z0)
            # log_abs_det_jacobian = tf.reduce_sum(z0-2*tf.log(1e-7+tf.exp(z0)+1), axis=[1], keep_dims=True)
            # log_abs_det_jacobian = tf.reduce_sum(tf.log(1e-7+z)+tf.log(1e-7+(1-z)), axis=[1], keep_dims=True)
            log_abs_det_jacobian = tf.reduce_sum(np.log(5.)+(tf.log(1e-7+z)+tf.log(1e-7+(1-z))), axis=[1], keep_dims=True)
        else:
            # z = tf.nn.tanh(z0)
            z = tf.nn.tanh(2.5*z0)
            # log_abs_det_jacobian = tf.reduce_sum(tf.log(1e-7+(1-z))+tf.log(1e-7+(1+z)), axis=[1], keep_dims=True)
            log_abs_det_jacobian = tf.reduce_sum(np.log(2.5)+tf.log(1e-7+(1-z))+tf.log(1e-7+(1+z)), axis=[1], keep_dims=True)
        
        log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)

        if self._zero_one:
            # z0 = (tf.log(z)-tf.log(1-z))
            z0 = (tf.log(z)-tf.log(1-z))/5.
            # log_abs_det_jacobian = -tf.reduce_sum(z0-2*tf.log(1e-7+tf.exp(z0)+1), axis=[1], keep_dims=True)
            log_abs_det_jacobian = -tf.reduce_sum(np.log(5.)+tf.log(1e-7+z)+tf.log(1e-7+(1-z)), axis=[1], keep_dims=True)
        else:
            # z0 = (0.5*(tf.log(1+z)-tf.log(1-z)))
            z0 = (0.5*(tf.log(1+z)-tf.log(1-z)))/2.5
            # log_abs_det_jacobian = -tf.reduce_sum(tf.log(1e-7+1-z)+tf.log(1e-7+1+z), axis=[1], keep_dims=True)
            log_abs_det_jacobian = -tf.reduce_sum(np.log(2.5)+tf.log(1e-7+1-z)+tf.log(1e-7+1+z), axis=[1], keep_dims=True)
        
        log_pdf_z0 = log_pdf_z - log_abs_det_jacobian
        return z0, log_pdf_z0

class InverseOpenIntervalDimensionFlow():
    """
    Inverse Open Interval Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, input_dim, parameters=None, name='InverseOpenIntervalDimensionFlow'):  #real
        self._input_dim = input_dim
        self._inverse_flow_object = OpenIntervalDimensionFlow(self._input_dim)
        assert (parameters is None)
        assert (self._input_dim > 1)
    
    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):  
        return 0

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        z, log_pdf_z = self._inverse_flow_object.inverse_transform(z0, log_pdf_z0)
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)

        z0, log_pdf_z0 = self._inverse_flow_object.transform(z, log_pdf_z)
        return z0, log_pdf_z0

class SpecificRotationFlow():
    """
    Specific Rotation Flow class. SO(n) fixed random rotation
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """

    def __init__(self, input_dim, parameters=None, name='specific_rotation_transform'):   
        self._parameters = parameters
        self._input_dim = input_dim
        assert (self._parameters is None)
        self._batched_rot_matrix = self.get_batched_rot_matrix() 

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim): 
        return 0

    def get_batched_rot_matrix(self):
        return tf.constant(helper.random_rot_mat(self._input_dim, mode='SO(n)'), dtype=tf.float32)[np.newaxis, :, :]
        # return tf.Variable(tf.constant(helper.random_rot_mat(self._input_dim, mode='SO(n)'), dtype=tf.float32), trainable=False)[np.newaxis, :, :]

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        z = tf.matmul(z0, self._batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=True)
        log_pdf_z = log_pdf_z0 
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)
        
        z0 = tf.matmul(z, self._batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=False)
        log_pdf_z0 = log_pdf_z 
        return z0, log_pdf_z0

class ManyReflectionRotationFlow():
    """
    Specific Rotation Flow class. SO(n) 
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    n_steps = 100

    def __init__(self, input_dim, parameters, name='many_reflection_rotation_transform'):   
        self._parameter_scale = 1.
        self._parameters = parameters
        if self._parameters is not None: self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim

        assert (ManyReflectionRotationFlow.n_steps % 2 == 0) # Required for SO(n)

        self._parameters.get_shape().assert_is_compatible_with([None, ManyReflectionRotationFlow.required_num_parameters(self._input_dim)])

        param_index = 0
        self._householder_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim*ManyReflectionRotationFlow.n_steps) 
        self._householder_vec = tf.reshape(self._householder_vec, [-1, ManyReflectionRotationFlow.n_steps, self._input_dim])
        self._householder_vec_dir = self._householder_vec/helper.safe_tf_sqrt(tf.reduce_sum(self._householder_vec**2, axis=2, keep_dims=True))

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim): 
        return ManyReflectionRotationFlow.n_steps*input_dim

    def get_batched_rot_matrix(self):
        identity_mat = tf.constant(np.eye(self._input_dim), tf.float32)
        overall_rot_mat = None
        for i in range(ManyReflectionRotationFlow.n_steps):
            curr_dir = self._householder_vec_dir[:,i,:]
            curr_rot_mat = identity_mat-2*tf.matmul(curr_dir, curr_dir, transpose_a=True, transpose_b=False)
            if overall_rot_mat is None: overall_rot_mat = curr_rot_mat
            else: overall_rot_mat = tf.matmul(curr_rot_mat, overall_rot_mat, transpose_a=False, transpose_b=False)
        return overall_rot_mat[np.newaxis, :, :]

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        curr_z = z0
        for i in range(ManyReflectionRotationFlow.n_steps):
            curr_dir = self._householder_vec_dir[:,i,:]
            curr_dot_product = tf.reduce_sum(curr_z*curr_dir, axis=1, keep_dims=True)
            curr_z = curr_z-2*curr_dot_product*curr_dir

        z, log_pdf_z = curr_z, log_pdf_z0
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)

        curr_z = z
        for i in range(ManyReflectionRotationFlow.n_steps):
            curr_dir = self._householder_vec_dir[:,ManyReflectionRotationFlow.n_steps-i-1,:]
            curr_dot_product = tf.reduce_sum(curr_z*curr_dir, axis=1, keep_dims=True)
            curr_z = curr_z-2*curr_dot_product*curr_dir

        z0, log_pdf_z0 = curr_z, log_pdf_z
        return z0, log_pdf_z0

class HouseholdRotationFlow():
    """
    Householder Rotation Flow class. SO(n)
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    max_steps = 20

    def __init__(self, input_dim, parameters, vector_mode_rate=1, name='household_rotation_transform'):   
        self._parameter_scale = 1.
        self._parameters = parameters
        if self._parameters is not None: self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim
        self._k_start = max(self._input_dim-HouseholdRotationFlow.max_steps+1, 1)
        self._n_steps = self._input_dim-self._k_start+1
        self._init_reflection = (-1)**(self._input_dim-1)
        self._vector_mode_rate = vector_mode_rate
        if float(self._n_steps)/float(self._input_dim) <= self._vector_mode_rate: self._mode = 'vector'
        else: self._mode = 'matrix'
        print('Household Rotation Flow (# steps, input_dim, mode): ', (self._n_steps, self._input_dim, self._mode))
        
        assert (self._init_reflection == 1 or self._init_reflection == -1)
        assert (self._mode == 'matrix' or self._mode == 'vector')
        assert (HouseholdRotationFlow.max_steps % 2 == 0) # Required for SO(n)

        if self._parameters is not None:
            self._parameters.get_shape().assert_is_compatible_with([None, HouseholdRotationFlow.required_num_parameters(self._input_dim)])

        if self._mode == 'matrix':
            self._batched_rot_matrix = self.get_batched_rot_matrix() 
        elif self._mode == 'vector':
            self._list_batched_householder_dirs =  self.get_list_batched_householder_vectors() 

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim): 
        assert (HouseholdRotationFlow.max_steps % 2 == 0) # Required for SO(n)
        k_start = max(input_dim-HouseholdRotationFlow.max_steps+1, 1)
        return sum(list(range(max(2, k_start), input_dim+1)))
    
    def get_batched_rot_matrix(self):
        return helper.householder_rotations_tf(n=self.input_dim, k_start=self._k_start, init_reflection=self._init_reflection, params=self._parameters) 

    def get_list_batched_householder_vectors(self):
        return helper.householder_rotation_vectors_tf(n=self.input_dim, k_start=self._k_start, init_reflection=self._init_reflection, params=self._parameters) 
        
    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        if self._mode == 'matrix':
            if self._parameters is None or self._parameters.get_shape()[0].value == 1: #one set of parameters
                z = tf.matmul(z0, self._batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=True)
            else: # batched parameters
                z = tf.matmul(self._batched_rot_matrix, z0[:,:,np.newaxis])[:, :, 0]

        elif self._mode == 'vector':
            curr_z = z0 
            for i in range(len(self._list_batched_householder_dirs)):
                curr_batched_householder_dir = self._list_batched_householder_dirs[i]
                start_ind = None
                if isinstance(curr_batched_householder_dir, float): 
                    start_ind = self._input_dim-1
                    reflected = curr_z[:, start_ind:]*curr_batched_householder_dir
                else: 
                    start_ind = self._input_dim-curr_batched_householder_dir.get_shape().as_list()[1]
                    reflected = curr_z[:, start_ind:]-2*curr_batched_householder_dir*tf.reduce_sum(curr_z[:, start_ind:]*curr_batched_householder_dir, axis=1, keep_dims=True)
                curr_z = tf.concat([curr_z[:, :start_ind], reflected], axis=1)
            z = curr_z

        log_pdf_z = log_pdf_z0 
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)
        
        if self._parameters is None or self._parameters.get_shape()[0].value == 1: #one set of parameters
            if self._mode == 'matrix':
                z0 = tf.matmul(z, self._batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=False)
            elif self._mode == 'vector':
                curr_z = z
                for i in range(len(self._list_batched_householder_dirs)):
                    curr_batched_householder_dir = self._list_batched_householder_dirs[len(self._list_batched_householder_dirs)-i-1]
                    start_ind = None
                    if isinstance(curr_batched_householder_dir, float): 
                        start_ind = self._input_dim-1
                        reflected = curr_z[:, start_ind:]*curr_batched_householder_dir
                    else: 
                        start_ind = self._input_dim-curr_batched_householder_dir.get_shape().as_list()[1]
                        reflected = curr_z[:, start_ind:]-2*curr_batched_householder_dir*tf.reduce_sum(curr_z[:, start_ind:]*curr_batched_householder_dir, axis=1, keep_dims=True)
                    curr_z = tf.concat([curr_z[:, :start_ind], reflected], axis=1)
                z0 = curr_z

        else: # batched parameters
            print('Parameters depend on unknown z0. Therefore, there is no analytical inverse.')
            quit()

        log_pdf_z0 = log_pdf_z 
        return z0, log_pdf_z0

CompoundHouseholdRotationFlow = ManyReflectionRotationFlow
# class CompoundHouseholdRotationFlow():
    # """
    # Compound Householder Rotation Flow class. SO(n)
    # Args:
    #   parameters: parameters of transformation all appended.
    #   input_dim : input dimensionality of the transformation. 
    # Raises:
    #   ValueError: 
    # """
    # max_steps = 100

    # def __init__(self, input_dim, parameters, vector_mode_rate=1, name='compound_household_rotation_transform'):   
    #     self._parameter_scale = 1.
    #     self._parameters = parameters
    #     if self._parameters is not None: self._parameters = self._parameter_scale*self._parameters
    #     self._input_dim = input_dim
    #     self._k_start = max(self._input_dim-CompoundHouseholdRotationFlow.max_steps+1, 1)
    #     self._n_steps = self._input_dim-self._k_start+1
    #     self._init_reflection = (-1)**(self._input_dim-1)
    #     self._vector_mode_rate = vector_mode_rate
    #     if float(self._n_steps)/float(self._input_dim) <= self._vector_mode_rate: self._mode = 'vector'
    #     else: self._mode = 'matrix'
    #     print('Household Rotation Flow (# steps, input_dim, mode): ', (self._n_steps, self._input_dim, self._mode))
        
    #     assert (self._init_reflection == 1 or self._init_reflection == -1)
    #     assert (self._mode == 'matrix' or self._mode == 'vector')
    #     assert (CompoundHouseholdRotationFlow.max_steps % 2 == 0) # Required for SO(n)

    #     if self._parameters is not None:
    #         self._parameters.get_shape().assert_is_compatible_with([None, CompoundHouseholdRotationFlow.required_num_parameters(self._input_dim)])

    #     if self._mode == 'matrix':
    #         self._batched_rot_matrix = self.get_batched_rot_matrix() 
    #     elif self._mode == 'vector':
    #         self._list_batched_householder_dirs =  self.get_list_batched_householder_vectors() 

    # @property
    # def input_dim(self):
    #     return self._input_dim

    # @property
    # def output_dim(self):
    #     return self._input_dim

    # @staticmethod
    # def required_num_parameters(input_dim): 
    #     assert (CompoundHouseholdRotationFlow.max_steps % 2 == 0) # Required for SO(n)
    #     k_start = max(input_dim-CompoundHouseholdRotationFlow.max_steps+1, 1)
    #     return sum(list(range(max(2, k_start), input_dim+1)))
    
    # def get_batched_rot_matrix(self):
    #     return helper.householder_rotations_tf(n=self.input_dim, k_start=self._k_start, init_reflection=self._init_reflection, params=self._parameters) 

    # def get_list_batched_householder_vectors(self):
    #     return helper.householder_rotation_vectors_tf(n=self.input_dim, k_start=self._k_start, init_reflection=self._init_reflection, params=self._parameters) 
        
    # def transform(self, z0, log_pdf_z0):
    #     verify_size(z0, log_pdf_z0)

    #     if self._mode == 'matrix':
    #         if self._parameters is None or self._parameters.get_shape()[0].value == 1: #one set of parameters
    #             z = tf.matmul(z0, self._batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=True)
    #         else: # batched parameters
    #             z = tf.matmul(self._batched_rot_matrix, z0[:,:,np.newaxis])[:, :, 0]

    #     elif self._mode == 'vector':
    #         curr_z = z0 
    #         for i in range(len(self._list_batched_householder_dirs)):
    #             curr_batched_householder_dir = self._list_batched_householder_dirs[i]
    #             start_ind = None
    #             if isinstance(curr_batched_householder_dir, float): 
    #                 start_ind = self._input_dim-1
    #                 reflected = curr_z[:, start_ind:]*curr_batched_householder_dir
    #             else: 
    #                 start_ind = self._input_dim-curr_batched_householder_dir.get_shape().as_list()[1]
    #                 reflected = curr_z[:, start_ind:]-2*curr_batched_householder_dir*tf.reduce_sum(curr_z[:, start_ind:]*curr_batched_householder_dir, axis=1, keep_dims=True)
    #             curr_z = tf.concat([curr_z[:, :start_ind], reflected], axis=1)
    #         z = curr_z

    #     log_pdf_z = log_pdf_z0 
    #     return z, log_pdf_z

    # def inverse_transform(self, z, log_pdf_z):
    #     verify_size(z, log_pdf_z)
        
    #     if self._parameters is None or self._parameters.get_shape()[0].value == 1: #one set of parameters
    #         if self._mode == 'matrix':
    #             z0 = tf.matmul(z, self._batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=False)
    #         elif self._mode == 'vector':
    #             curr_z = z
    #             for i in range(len(self._list_batched_householder_dirs)):
    #                 curr_batched_householder_dir = self._list_batched_householder_dirs[len(self._list_batched_householder_dirs)-i-1]
    #                 start_ind = None
    #                 if isinstance(curr_batched_householder_dir, float): 
    #                     start_ind = self._input_dim-1
    #                     reflected = curr_z[:, start_ind:]*curr_batched_householder_dir
    #                 else: 
    #                     start_ind = self._input_dim-curr_batched_householder_dir.get_shape().as_list()[1]
    #                     reflected = curr_z[:, start_ind:]-2*curr_batched_householder_dir*tf.reduce_sum(curr_z[:, start_ind:]*curr_batched_householder_dir, axis=1, keep_dims=True)
    #                 curr_z = tf.concat([curr_z[:, :start_ind], reflected], axis=1)
    #             z0 = curr_z

    #     else: # batched parameters
    #         print('Parameters depend on unknown z0. Therefore, there is no analytical inverse.')
    #         quit()

    #     log_pdf_z0 = log_pdf_z 
    #     return z0, log_pdf_z0

class CompoundRotationFlow():
    """
    Compound Rotation Flow class. SO(n)
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    compound_structure = ['C', 'H']

    def __init__(self, input_dim, parameters, name='compound_rotation_transform'):  
        self._parameter_scale = 1.
        self._parameters = parameters
        if self._parameters is not None: self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim
        
        if self._parameters is not None:
            self._parameters.get_shape().assert_is_compatible_with([None, CompoundRotationFlow.required_num_parameters(self._input_dim)])
        
        param_index = 0
        self._constant_rot_mats_list, self._householder_flows_list, self._specific_order_dimension_flows_list = [], [], []
        for i in range(len(CompoundRotationFlow.compound_structure)):
            if CompoundRotationFlow.compound_structure[i] == 'C':
                self._constant_rot_mats_list.append(tf.Variable(tf.constant(helper.random_rot_mat(self._input_dim, mode='SO(n)'), dtype=tf.float32), trainable=False))
            elif CompoundRotationFlow.compound_structure[i] == 'H':
                curr_householder_param, param_index = helper.slice_parameters(self._parameters, param_index, CompoundHouseholdRotationFlow.required_num_parameters(self._input_dim))
                self._householder_flows_list.append(CompoundHouseholdRotationFlow(self._input_dim, curr_householder_param))
            elif CompoundRotationFlow.compound_structure[i] == 'P':
                self._specific_order_dimension_flows_list.append(SpecificOrderDimensionFlow(self._input_dim))
            else: assert (False)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim): 
        n_parameters = 0
        for i in range(len(CompoundRotationFlow.compound_structure)):
            if CompoundRotationFlow.compound_structure[i] == 'H':
                n_parameters += CompoundHouseholdRotationFlow.required_num_parameters(input_dim)
        return n_parameters 
    
    def get_batched_rot_matrix(self):
        curr_batched_rot_matrix = None

        c_index, h_index, p_index = 0, 0, 0
        for i in range(len(CompoundRotationFlow.compound_structure)):
            if CompoundRotationFlow.compound_structure[i] == 'C':
                curr_mat = self._constant_rot_mats_list[c_index][np.newaxis, :, :]
                c_index += 1
            elif CompoundRotationFlow.compound_structure[i] == 'H':
                curr_mat = self._householder_flows_list[h_index].get_batched_rot_matrix()
                h_index += 1
            elif CompoundRotationFlow.compound_structure[i] == 'P':
                curr_mat = self._specific_order_dimension_flows_list[p_index].get_batched_rot_matrix()
                p_index += 1

            if curr_batched_rot_matrix is None: curr_batched_rot_matrix = curr_mat
            else: curr_batched_rot_matrix = tf.matmul(curr_mat, curr_batched_rot_matrix, transpose_a=False, transpose_b=False)

        assert ((c_index+h_index+p_index) == len(CompoundRotationFlow.compound_structure))
        return curr_batched_rot_matrix
        
    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        c_index, h_index, p_index = 0, 0, 0
        curr_z, curr_log_pdf_z = z0, log_pdf_z0
        for i in range(len(CompoundRotationFlow.compound_structure)):
            # print('c_index, h_index, p_index ', c_index, h_index, p_index)
            if CompoundRotationFlow.compound_structure[i] == 'C':
                curr_z, curr_log_pdf_z = tf.matmul(curr_z, self._constant_rot_mats_list[c_index], transpose_a=False, transpose_b=True), curr_log_pdf_z
                c_index += 1
            elif CompoundRotationFlow.compound_structure[i] == 'H':
                curr_z, curr_log_pdf_z = self._householder_flows_list[h_index].transform(curr_z, curr_log_pdf_z)
                h_index += 1
            elif CompoundRotationFlow.compound_structure[i] == 'P':
                curr_z, curr_log_pdf_z = self._specific_order_dimension_flows_list[p_index].transform(curr_z, curr_log_pdf_z)
                p_index += 1

        assert ((c_index+h_index+p_index) == len(CompoundRotationFlow.compound_structure))

        z, log_pdf_z = curr_z, curr_log_pdf_z
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)

        if self._parameters is None or self._parameters.get_shape()[0].value == 1: #one set of parameters
            c_index = np.sum((np.asarray(CompoundRotationFlow.compound_structure) == 'C'))
            h_index = np.sum((np.asarray(CompoundRotationFlow.compound_structure) == 'H'))
            p_index = np.sum((np.asarray(CompoundRotationFlow.compound_structure) == 'P'))
            curr_z, curr_log_pdf_z = z, log_pdf_z
            for i in range(len(CompoundRotationFlow.compound_structure)-1, -1, -1):
                # print('c_index, h_index, p_index ', c_index, h_index, p_index)
                if CompoundRotationFlow.compound_structure[i] == 'C':
                    c_index -= 1
                    curr_z, curr_log_pdf_z = tf.matmul(curr_z, self._constant_rot_mats_list[c_index], transpose_a=False, transpose_b=False), curr_log_pdf_z
                elif CompoundRotationFlow.compound_structure[i] == 'H':
                    h_index -= 1
                    curr_z, curr_log_pdf_z = self._householder_flows_list[h_index].inverse_transform(curr_z, curr_log_pdf_z)
                elif CompoundRotationFlow.compound_structure[i] == 'P':
                    p_index -= 1
                    curr_z, curr_log_pdf_z = self._specific_order_dimension_flows_list[p_index].inverse_transform(curr_z, curr_log_pdf_z)
                assert (c_index >= 0 and h_index >= 0 and p_index >= 0)

            z0, log_pdf_z0 = curr_z, curr_log_pdf_z
        else: # batched parameters
            print('Parameters depend on unknown z0. Therefore, there is no analytical inverse.')
            quit()

        return z0, log_pdf_z0

class OthogonalProjectionMap():
    """
    Othogonal Projection Map class with Jacobians specified as J^TJ=I or JJ^T=I depending on which is the smaller matrix.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
      output_dim : output dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    rotation_flow_class = CompoundRotationFlow # HouseholdRotationFlow
    
    def __init__(self, input_dim, output_dim, parameters, name='orthogonal_projection_transform'):   
        self._parameter_scale = 1.
        self._parameters = parameters
        self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._mode = 'vector'

        assert (self._mode == 'matrix' or self._mode == 'vector')
        assert (self._input_dim > 0 and self._output_dim > 0)

        self._parameters.get_shape().assert_is_compatible_with([None, OthogonalProjectionMap.required_num_parameters(self._input_dim, self._output_dim)])
        
        param_index = 0
        self._rotation_param, param_index = helper.slice_parameters(self._parameters, param_index, OthogonalProjectionMap.rotation_flow_class.required_num_parameters(max(self._input_dim, self._output_dim)))
        self._input_shift_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim)
        self._output_shift_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._output_dim) 
        self._rotation_flow = OthogonalProjectionMap.rotation_flow_class(max(self._input_dim, self._output_dim), self._rotation_param) 

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @staticmethod
    def required_num_parameters(input_dim, output_dim):
        assert (input_dim > 0 and output_dim > 0)
        n_rot_param = OthogonalProjectionMap.rotation_flow_class.required_num_parameters(max(input_dim, output_dim))
        return n_rot_param+input_dim+output_dim

    def jacobian(self, z0):        
        full_batched_rot_matrix = self._rotation_flow.get_batched_rot_matrix()
        batched_rot_matrix = full_batched_rot_matrix[:, :self._output_dim, :self._input_dim]
        if self._parameters is None or self._parameters.get_shape()[0].value == 1: #one set of parameters
            jacobian = tf.tile(batched_rot_matrix, [tf.shape(z0)[0], 1, 1])
        else: # batched parameters
            jacobian = batched_rot_matrix
        return jacobian

    def transform(self, z0):
        z0_centered = z0-self._input_shift_vec

        if self._mode == 'matrix': # This is for debugging mostly, defer to rotation flow mode in general.
            full_batched_rot_matrix = self._rotation_flow.get_batched_rot_matrix()
            batched_rot_matrix = full_batched_rot_matrix[:, :self._output_dim, :self._input_dim]
            if self._parameters is None or self._parameters.get_shape()[0].value == 1: #one set of parameters
                z_proj = tf.matmul(z0_centered, batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=True)
            else: # batched parameters
                z_proj = tf.matmul(batched_rot_matrix, z0_centered[:,:,np.newaxis])[:, :, 0]

        elif self._mode == 'vector':
            if self._input_dim >= self._output_dim: 
                z_proj_full, _ = self._rotation_flow.transform(z0_centered, tf.zeros((tf.shape(z0)[0], 1), tf.float32))
                z_proj = z_proj_full[:, :self._output_dim]
            elif self._output_dim > self._input_dim: 
                z_proj, _ = self._rotation_flow.transform(tf.concat([z0_centered, tf.zeros((tf.shape(z0_centered)[0], self._output_dim-self._input_dim))], axis=1), tf.zeros((tf.shape(z0)[0], 1), tf.float32))

        z = z_proj+self._output_shift_vec
        return z


class ConnectedPiecewiseOrthogonalMap():
    """
    Connected Piecewise Orthogonal Map class with Jacobians specified as scaled multiples of orthogonal matrices.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    rotation_flow_class = HouseholdRotationFlow # CompoundRotationFlow

    def __init__(self, input_dim, parameters, margin_mode='NoGradient', scale_mode='Scale', name='connected_piecewise_orthogonal_map'):   
        self._parameter_scale = 1.
        self._parameters = parameters
        self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim
        self._margin_mode = margin_mode
        self._scale_mode = scale_mode
        self._max_bounded_scale = 5
        self._min_bounded_scale = 1/self._max_bounded_scale

        assert (self._margin_mode == 'NoGradient' or self._margin_mode == 'ST')
        assert (self._scale_mode == 'Scale' or self._scale_mode == 'BoundedScale')
        assert (self._max_bounded_scale > 1 and self._min_bounded_scale >= 0 and self._max_bounded_scale > self._min_bounded_scale)

        self._parameters.get_shape().assert_is_compatible_with([None, ConnectedPiecewiseOrthogonalMap.required_num_parameters(self._input_dim)])
        
        param_index = 0
        self._pos_rotation_param, param_index = helper.slice_parameters(self._parameters, param_index, ConnectedPiecewiseOrthogonalMap.rotation_flow_class.required_num_parameters(self._input_dim))
        self._neg_rotation_param, param_index = helper.slice_parameters(self._parameters, param_index, ConnectedPiecewiseOrthogonalMap.rotation_flow_class.required_num_parameters(self._input_dim))
        self._pos_pre_scale, param_index = helper.slice_parameters(self._parameters, param_index, 1)
        self._neg_pre_scale, param_index = helper.slice_parameters(self._parameters, param_index, 1)
        self._hyper_pre_bias, param_index = helper.slice_parameters(self._parameters, param_index, 1)
        self._hyper_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim) 
        self._output_shift_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim) 

        if self._scale_mode == 'Scale':
            self._pos_scale = tf.clip_by_value(tf.nn.softplus(self._pos_pre_scale)/np.log(1+np.exp(0)), 1e-10, np.inf)  
            self._neg_scale = tf.clip_by_value(tf.nn.softplus(self._neg_pre_scale)/np.log(1+np.exp(0)), 1e-10, np.inf)  
        elif self._scale_mode == 'BoundedScale': 
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            self._pos_scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(self._pos_pre_scale+scipy.special.logit(1/gap))*gap, 1e-10, np.inf)  
            self._neg_scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(self._neg_pre_scale+scipy.special.logit(1/gap))*gap, 1e-10, np.inf)  
        self._hyper_bias = tf.nn.softplus(self._hyper_pre_bias)/np.log(1+np.exp(0))
        self._hyper_vec_dir = self._hyper_vec/helper.safe_tf_sqrt(tf.reduce_sum(self._hyper_vec**2, axis=1, keep_dims=True))
        self._pos_rotation_flow = ConnectedPiecewiseOrthogonalMap.rotation_flow_class(self._input_dim, self._pos_rotation_param) 
        self._neg_rotation_flow = ConnectedPiecewiseOrthogonalMap.rotation_flow_class(self._input_dim, self._neg_rotation_param) 

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):
        n_rot_param = ConnectedPiecewiseOrthogonalMap.rotation_flow_class.required_num_parameters(input_dim)
        return 2*n_rot_param+1+1+1+input_dim+input_dim

    def jacobian(self, z0):
        z_centered = z0-self._hyper_vec_dir*self._hyper_bias

        margin = tf.reduce_sum(self._hyper_vec_dir*z_centered, axis=1, keep_dims=True)
        if self._margin_mode == 'NoGradient':
            pos_mask = tf.stop_gradient(tf.cast(margin>=0, tf.float32)) # margin >= 0 return 1 else 0 
        elif self._margin_mode == 'ST':
            pos_mask = helper.binaryStochastic_ST(margin) # margin >= ~9e-8 return 1 else 0
        neg_mask = 1-pos_mask
        
        pos_batched_rot_matrix = self._pos_rotation_flow.get_batched_rot_matrix()
        neg_batched_rot_matrix = self._neg_rotation_flow.get_batched_rot_matrix()
        scaled_pos_batched_rot_matrix = self._pos_scale[:, :, np.newaxis]*pos_batched_rot_matrix
        scaled_neg_batched_rot_matrix = self._neg_scale[:, :, np.newaxis]*neg_batched_rot_matrix
        jacobian = pos_mask[:, :, np.newaxis]*scaled_pos_batched_rot_matrix+neg_mask[:, :, np.newaxis]*scaled_neg_batched_rot_matrix
        return jacobian

    def transform(self, z0):
        z_centered = z0-self._hyper_vec_dir*self._hyper_bias

        margin = tf.reduce_sum(self._hyper_vec_dir*z_centered, axis=1, keep_dims=True)
        if self._margin_mode == 'NoGradient':
            pos_mask = tf.stop_gradient(tf.cast(margin>=0, tf.float32)) # margin >= 0 return 1 else 0 
        elif self._margin_mode == 'ST':
            pos_mask = helper.binaryStochastic_ST(margin) # margin >= ~9e-8 return 1 else 0
        neg_mask = 1-pos_mask
        
        z_pos_rot, _ = self._pos_rotation_flow.transform(z_centered, tf.zeros(shape=(tf.shape(z0)[0], 1), dtype=tf.float32))
        z_neg_rot, _ = self._neg_rotation_flow.transform(z_centered, tf.zeros(shape=(tf.shape(z0)[0], 1), dtype=tf.float32))
        z_pos_scale_rot = self._pos_scale*z_pos_rot
        z_neg_scale_rot = self._neg_scale*z_neg_rot

        z_scale_rot = pos_mask*z_pos_scale_rot+neg_mask*z_neg_scale_rot
        z = z_scale_rot+self._hyper_vec_dir*self._hyper_bias+self._output_shift_vec
        scales = pos_mask*self._pos_scale+neg_mask*self._neg_scale 
        log_scales = tf.log(scales)
        return z, log_scales

class PiecewisePlanarScalingMap():
    """
    Connected Piecewise Scaling Map class with Jacobians specified as scaled multiples of diagonal matrices.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    n_steps = 10

    def __init__(self, input_dim, parameters, margin_mode='NoGradient', scale_mode='Scale', name='piecewise_planar_scaling_map'):   
        self._parameter_scale = 1.
        self._parameters = parameters
        self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim
        self._margin_mode = margin_mode
        self._scale_mode = scale_mode
        self._max_bounded_scale = 5.
        self._min_bounded_scale = 0 

        assert (self._margin_mode == 'NoGradient' or self._margin_mode == 'ST')
        assert (self._scale_mode == 'Scale' or self._scale_mode == 'BoundedScale')
        assert (self._max_bounded_scale > 1 and self._min_bounded_scale >= 0 and self._max_bounded_scale > self._min_bounded_scale)

        self._parameters.get_shape().assert_is_compatible_with([None, PiecewisePlanarScalingMap.required_num_parameters(self._input_dim)])
        
        param_index = 0
        self._pos_pre_scale, param_index = helper.slice_parameters(self._parameters, param_index, PiecewisePlanarScalingMap.n_steps)
        self._neg_pre_scale, param_index = helper.slice_parameters(self._parameters, param_index, PiecewisePlanarScalingMap.n_steps)
        self._hyper_pre_bias, param_index = helper.slice_parameters(self._parameters, param_index, PiecewisePlanarScalingMap.n_steps)
        self._hyper_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim*PiecewisePlanarScalingMap.n_steps) 
        self._output_shift_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim*PiecewisePlanarScalingMap.n_steps) 
        
        self._hyper_vec = tf.reshape(self._hyper_vec, [-1, PiecewisePlanarScalingMap.n_steps, self._input_dim])
        self._output_shift_vec = tf.reshape(self._output_shift_vec, [-1, PiecewisePlanarScalingMap.n_steps, self._input_dim])

        if self._scale_mode == 'Scale':
            self._pos_scale = tf.clip_by_value(tf.nn.softplus(self._pos_pre_scale)/np.log(1+np.exp(0)), 1e-10, np.inf)  
            self._neg_scale = tf.clip_by_value(tf.nn.softplus(self._neg_pre_scale)/np.log(1+np.exp(0)), 1e-10, np.inf)  
        elif self._scale_mode == 'BoundedScale': 
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            self._pos_scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(self._pos_pre_scale+scipy.special.logit(1/gap))*gap, 1e-10, np.inf)  
            self._neg_scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(self._neg_pre_scale+scipy.special.logit(1/gap))*gap, 1e-10, np.inf)  
        self._hyper_bias = tf.nn.softplus(self._hyper_pre_bias)/np.log(1+np.exp(0))
        self._hyper_vec_dir = self._hyper_vec/helper.safe_tf_sqrt(tf.reduce_sum(self._hyper_vec**2, axis=2, keep_dims=True))

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):
        return PiecewisePlanarScalingMap.n_steps*(input_dim+input_dim+1+1+1)

    def jacobian(self, z0): 
        z, log_scales = self.transform(z0)
        scales = tf.exp(log_scales)
        return scales[:,:,np.newaxis]*tf.constant(np.eye(self._input_dim), tf.float32)[np.newaxis,:,:]

    def transform(self, z0):
        curr_z = z0
        log_scales = 0
        for i in range(PiecewisePlanarScalingMap.n_steps):
            curr_pos_scale = self._pos_scale[:,i,np.newaxis]
            curr_neg_scale = self._neg_scale[:,i,np.newaxis]
            curr_output_shift_vec = self._output_shift_vec[:,i,:]
            curr_w = self._hyper_vec_dir[:,i,:]
            curr_b = self._hyper_bias[:,i,np.newaxis]
            curr_wb = curr_w*curr_b
            curr_z_centered = curr_z-curr_wb

            curr_margin = tf.reduce_sum(curr_w*curr_z_centered, axis=1, keep_dims=True)
            if self._margin_mode == 'NoGradient':
                curr_pos_mask = tf.stop_gradient(tf.cast(curr_margin>=0, tf.float32)) # margin >= 0 return 1 else 0 
            elif self._margin_mode == 'ST':
                curr_pos_mask = helper.binaryStochastic_ST(curr_margin) # margin >= ~9e-8 return 1 else 0
            curr_neg_mask = 1-curr_pos_mask

            curr_scales = curr_pos_mask*curr_pos_scale+curr_neg_mask*curr_neg_scale
            curr_z = curr_scales*curr_z_centered+curr_wb+curr_output_shift_vec
            log_scales = log_scales+tf.log(curr_scales)
        
        z = curr_z
        return z, log_scales

    def inverse_transform(self, z):
        curr_z = z
        log_scales = 0
        for i in range(PiecewisePlanarScalingMap.n_steps):
            curr_pos_scale = self._pos_scale[:,PiecewisePlanarScalingMap.n_steps-1-i,np.newaxis]
            curr_neg_scale = self._neg_scale[:,PiecewisePlanarScalingMap.n_steps-1-i,np.newaxis]
            curr_output_shift_vec = self._output_shift_vec[:,PiecewisePlanarScalingMap.n_steps-1-i,:]
            curr_w = self._hyper_vec_dir[:,PiecewisePlanarScalingMap.n_steps-1-i,:]
            curr_b = self._hyper_bias[:,PiecewisePlanarScalingMap.n_steps-1-i,np.newaxis]
            curr_wb = curr_w*curr_b
            curr_z_centered = curr_z-curr_wb-curr_output_shift_vec

            curr_margin = tf.reduce_sum(curr_w*curr_z_centered, axis=1, keep_dims=True)
            if self._margin_mode == 'NoGradient':
                curr_pos_mask = tf.stop_gradient(tf.cast(curr_margin>=0, tf.float32)) # margin >= 0 return 1 else 0 
            elif self._margin_mode == 'ST':
                curr_pos_mask = helper.binaryStochastic_ST(curr_margin) # margin >= ~9e-8 return 1 else 0
            curr_neg_mask = 1-curr_pos_mask

            curr_scales = curr_pos_mask*(1/curr_pos_scale)+curr_neg_mask*(1/curr_neg_scale)
            curr_z = curr_scales*curr_z_centered+curr_wb
            log_scales = log_scales+tf.log(curr_scales)
        
        z0 = curr_z
        return z0, log_scales

class PiecewisePlanarScalingFlow():
    """
    Connected Piecewise Scaling Flow class with Jacobians specified as scaled multiples of diagonal matrices.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """

    def __init__(self, input_dim, parameters, name='piecewise_planar_scaling_transform'):   
        self._parameter_scale = 1.
        self._parameters = parameters
        self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim

        self._parameters.get_shape().assert_is_compatible_with([None, PiecewisePlanarScalingFlow.required_num_parameters(self._input_dim)])
        self._piecewise_planar_scaling_map = PiecewisePlanarScalingMap(self._input_dim, self._parameters)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):
        return PiecewisePlanarScalingMap.required_num_parameters(input_dim)

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        z, log_scales = self._piecewise_planar_scaling_map.transform(z0)
        log_pdf_z = log_pdf_z0-log_scales 
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)
        z0, log_scales = self._piecewise_planar_scaling_map.inverse_transform(z)
        log_pdf_z0 = log_pdf_z-log_scales 
        return z0, log_pdf_z0

class RiemannianFlow():
    """
    Projective Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    NOM_class = PiecewisePlanarScalingMap
    
    def __init__(self, input_dim, output_dim, n_input_NOM, n_output_NOM, parameters, name='riemannian_transform'):   
        self._parameter_scale = 1.
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._additional_dim = self._output_dim-self._input_dim
        self._n_input_NOM = n_input_NOM
        self._n_output_NOM = n_output_NOM

        assert (self._output_dim > self._input_dim)

        self._parameters.get_shape().assert_is_compatible_with([None, RiemannianFlow.required_num_parameters(self._input_dim, self._output_dim, self._n_input_NOM, self._n_output_NOM)])

        param_index = 0
        self._input_NOM_list = []
        for i in range(self._n_input_NOM): 
            curr_param, param_index = helper.slice_parameters(self._parameters, param_index, RiemannianFlow.NOM_class.required_num_parameters(self._input_dim))
            self._input_NOM_list.append(RiemannianFlow.NOM_class(self._input_dim, curr_param))

        proj_param, param_index = helper.slice_parameters(self._parameters, param_index, OthogonalProjectionMap.required_num_parameters(self._input_dim, self._additional_dim))
        self._proj_map = OthogonalProjectionMap(self._input_dim, self._additional_dim, proj_param)

        self._additional_NOM_list = []
        for i in range(self._n_output_NOM): 
            curr_param, param_index = helper.slice_parameters(self._parameters, param_index, RiemannianFlow.NOM_class.required_num_parameters(self._additional_dim))
            self._additional_NOM_list.append(RiemannianFlow.NOM_class(self._additional_dim, curr_param))

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def additional_dim(self):
        return self._additional_dim

    @staticmethod
    def required_num_parameters(input_dim, output_dim, n_input_NOM, n_output_NOM): 
        assert (output_dim > input_dim)
        assert (n_input_NOM >= 0 and n_output_NOM >= 0)

        additional_dim = output_dim-input_dim
        n_input_NOM_param = n_input_NOM*RiemannianFlow.NOM_class.required_num_parameters(input_dim)
        n_proj_param = OthogonalProjectionMap.required_num_parameters(input_dim, additional_dim)
        n_output_NOM_param = n_output_NOM*RiemannianFlow.NOM_class.required_num_parameters(additional_dim)
        return n_input_NOM_param+n_proj_param+n_output_NOM_param

    def jacobian(self, z0, mode='full'):
        assert (mode == 'full' or mode == 'additional')

        curr_z = z0
        input_NOM_Js = []
        for i in range(len(self._input_NOM_list)): 
            input_NOM_Js.append(self._input_NOM_list[i].jacobian(curr_z))
            curr_z, _ = self._input_NOM_list[i].transform(curr_z)

        input_NOM_z = curr_z
        proj_input_NOM_z = self._proj_map.transform(input_NOM_z)
        proj_map_J = self._proj_map.jacobian(input_NOM_z)

        curr_z = proj_input_NOM_z
        additional_NOM_Js = []
        for i in range(len(self._additional_NOM_list)): 
            additional_NOM_Js.append(self._additional_NOM_list[i].jacobian(curr_z))
            curr_z, _ = self._additional_NOM_list[i].transform(curr_z)

        overall_input_NOM_Js = None
        for i in range(len(input_NOM_Js)):
            if overall_input_NOM_Js is None: overall_input_NOM_Js = input_NOM_Js[i]
            else: overall_input_NOM_Js = tf.matmul(input_NOM_Js[i], overall_input_NOM_Js, transpose_a=False, transpose_b=False)

        overall_additional_NOM_Js = None
        for i in range(len(additional_NOM_Js)):
            if overall_additional_NOM_Js is None: overall_additional_NOM_Js = additional_NOM_Js[i]
            else: overall_additional_NOM_Js = tf.matmul(additional_NOM_Js[i], overall_additional_NOM_Js, transpose_a=False, transpose_b=False)

        if overall_input_NOM_Js is not None:
            proj_map_J_overall_input_J = tf.matmul(proj_map_J, overall_input_NOM_Js, transpose_a=False, transpose_b=False)
        else: proj_map_J_overall_input_J = proj_map_J

        if overall_additional_NOM_Js is not None:
            overall_J = tf.matmul(overall_additional_NOM_Js, proj_map_J_overall_input_J, transpose_a=False, transpose_b=False)
        else: overall_J = proj_map_J_overall_input_J

        if mode == 'full':
            overall_J = tf.concat([tf.tile(tf.eye(self._input_dim)[np.newaxis, :, :], [tf.shape(overall_J)[0],1,1]), overall_J], axis=1)
        return overall_J

    def transform(self, z0, log_pdf_z0, mode='regular'):
        assert (mode == 'regular' or mode == 'scales')
        verify_size(z0, log_pdf_z0)

        curr_z = z0
        input_NOM_log_scales = []
        for i in range(len(self._input_NOM_list)): 
            curr_z, curr_log_scales = self._input_NOM_list[i].transform(curr_z)
            input_NOM_log_scales.append(curr_log_scales)
        input_NOM_z = curr_z

        proj_input_NOM_z = self._proj_map.transform(input_NOM_z)

        curr_z = proj_input_NOM_z
        additional_NOM_log_scales = []
        for i in range(len(self._additional_NOM_list)): 
            curr_z, curr_log_scales = self._additional_NOM_list[i].transform(curr_z)
            additional_NOM_log_scales.append(curr_log_scales)
        additional_NOM_z = curr_z
        
        z = tf.concat([z0, additional_NOM_z], axis=1)
        
        if len(input_NOM_log_scales) > 0: input_NOM_log_scales_sum = tf.add_n(input_NOM_log_scales)
        else: input_NOM_log_scales_sum = tf.zeros((tf.shape(z0)[0], 1), tf.float32)
        if len(additional_NOM_log_scales) > 0: additional_NOM_log_scales_sum = tf.add_n(additional_NOM_log_scales)
        else: additional_NOM_log_scales_sum = tf.zeros((tf.shape(z0)[0], 1), tf.float32)
        overall_scales = tf.exp(input_NOM_log_scales_sum+additional_NOM_log_scales_sum)

        delta_log_pdf_z = -tf.log(1+overall_scales**2)*(min(self._additional_dim, self._input_dim)/2)
        log_pdf_z = log_pdf_z0 + delta_log_pdf_z
        if mode == 'regular': return z, log_pdf_z
        else: return z, log_pdf_z, overall_scales

class LinearIARFlow():
    """
    Linear Inverse Autoregressive Flow class. (not really, why does it not change the density? It is a subset version of it namely volume preserving skewing.)

    This is only to be used before a non-centered diagonal-covariance gaussian to archive the full
    flexibility since it does the minimal, which is volume preserving skewing (and no scaling or translation).
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, input_dim, parameters, name='linearIAR_transform'):   
        self._parameter_scale = 1.
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim
        self._mask_mat_ones = helper.triangular_ones([self._input_dim, self._input_dim], trilmode=-1)
        self._diag_mat_ones = helper.block_diagonal_ones([self._input_dim, self._input_dim], [1, 1])
        
        assert (self._input_dim > 1)
        
        self._parameters.get_shape().assert_is_compatible_with([None, LinearIARFlow.required_num_parameters(self._input_dim)])
        
        self._param_matrix = tf.reshape(self._parameters, [-1, self._input_dim, self._input_dim])
        self._mask_matrix = tf.reshape(self._mask_mat_ones, [1, self._input_dim, self._input_dim]) 
        self._diag_matrix = tf.reshape(self._diag_mat_ones, [1, self._input_dim, self._input_dim]) 
        self._cho = self._mask_matrix*self._param_matrix+self._diag_matrix

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):
        return input_dim*input_dim

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        if self._parameters.get_shape()[0].value == 1: #one set of parameters
            z = tf.matmul(z0, self._cho[0, :, :], transpose_a=False, transpose_b=True)
        else: # batched parameters
            z = tf.matmul(self._cho, z0[:,:,np.newaxis])[:, :, 0]
        log_pdf_z = log_pdf_z0 
        return z, log_pdf_z

class NonLinearIARFlow():
    """
    Non-Linear Inverse Autoregressive Flow class.

    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    layer_expansions = [10, 10]

    def __init__(self, input_dim, parameters, mode='ScaleShift', name='nonlinearIAR_transform'):   #real
        self._parameter_scale = 1
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim
        self._nonlinearity = helper.LeakyReLU # tf.nn.tanh
        self._mode = mode
        self._max_bounded_scale = 10
        self._min_bounded_scale = 1/self._max_bounded_scale

        assert (self._input_dim > 1)
        assert (self._max_bounded_scale > 1 and self._min_bounded_scale >= 0 and self._max_bounded_scale > self._min_bounded_scale)

        self._parameters.get_shape().assert_is_compatible_with([None, NonLinearIARFlow.required_num_parameters(self._input_dim)])

        self._mask_tensors = helper.tf_get_mask_list_for_MADE(self._input_dim-1, NonLinearIARFlow.layer_expansions, add_mu_log_sigma_layer=True)
        self._pre_mu_for_1 = self._parameters[:, 0, np.newaxis]
        self._pre_scale_for_1 = self._parameters[:, 1, np.newaxis]
        self._layerwise_parameters = []

        start_ind = 2
        concat_layer_expansions = [1, *NonLinearIARFlow.layer_expansions, 2]
        for l in range(len(concat_layer_expansions)-1):
            W_num_param = concat_layer_expansions[l]*(self._input_dim-1)*concat_layer_expansions[l+1]*(self._input_dim-1) # matrix
            B_num_param = concat_layer_expansions[l+1]*(self._input_dim-1) # bias
            W_l_flat = tf.slice(self._parameters, [0, start_ind], [-1, W_num_param])
            B_l_flat = tf.slice(self._parameters, [0, start_ind+W_num_param], [-1, B_num_param])
            W_l = tf.reshape(W_l_flat, [-1, concat_layer_expansions[l+1]*(self._input_dim-1), concat_layer_expansions[l]*(self._input_dim-1)])              
            B_l = tf.reshape(B_l_flat, [-1, concat_layer_expansions[l+1]*(self._input_dim-1)])
            self._layerwise_parameters.append((W_l, B_l))  
            start_ind += (W_num_param+B_num_param)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):
        n_parameters = 0
        concat_layer_expansions = [1, *NonLinearIARFlow.layer_expansions, 2]
        for l in range(len(concat_layer_expansions)-1):
            n_parameters += concat_layer_expansions[l]*(input_dim-1)*concat_layer_expansions[l+1]*(input_dim-1) # matrix
            n_parameters += concat_layer_expansions[l+1]*(input_dim-1) # bias
        n_parameters += 2
        return n_parameters
    
    def MADE(self, input_x, layerwise_parameters, masks, nonlinearity):
        input_dim = input_x.get_shape().as_list()[1]
        curr_input = input_x
        for l in range(len(layerwise_parameters)):
            W, bias = layerwise_parameters[l]
            mask = masks[l]
            W_masked = W*mask
            if W.get_shape()[0].value == 1: #one set of parameters
                affine = tf.matmul(curr_input, W_masked[0, :, :], transpose_a=False, transpose_b=True)
            else: # batched parameters
                affine = tf.matmul(W_masked, curr_input[:,:,np.newaxis])[:, :, 0]
            if l < len(layerwise_parameters)-1: curr_input = nonlinearity(affine+bias)
            else: curr_input = (affine+bias)

        pre_mu = tf.slice(curr_input, [0, 0], [-1, input_dim])
        pre_scale = tf.slice(curr_input, [0, input_dim], [-1, input_dim])
        return pre_mu, pre_scale

    def MADE_single(self, input_x, layerwise_parameters, masks, nonlinearity):
        original_input_dim = layerwise_parameters[0][0].get_shape().as_list()[-1]
        input_dim = input_x.get_shape().as_list()[1]
        curr_input = input_x

        for l in range(len(layerwise_parameters)):
            W, bias = layerwise_parameters[l]
            mask = masks[l]
            curr_input_dim = curr_input.get_shape().as_list()[1]
            if l < len(layerwise_parameters)-1: 
                W = tf.slice(W, [0, 0, 0], [-1, input_dim*NonLinearIARFlow.layer_expansions[l], curr_input_dim])
                mask = tf.slice(mask, [0, 0, 0], [-1, input_dim*NonLinearIARFlow.layer_expansions[l], curr_input_dim])
                bias = tf.slice(bias, [0, 0], [-1, input_dim*NonLinearIARFlow.layer_expansions[l]])
            else: 
                W = tf.concat([tf.slice(W, [0, input_dim-1, 0], [-1, 1, curr_input_dim]), tf.slice(W, [0, original_input_dim+input_dim-1, 0], [-1, 1, curr_input_dim])] , axis=1)
                mask = tf.concat([tf.slice(mask, [0, input_dim-1, 0], [-1, 1, curr_input_dim]), tf.slice(mask, [0, original_input_dim+input_dim-1, 0], [-1, 1, curr_input_dim])] , axis=1)
                bias = tf.concat([tf.slice(bias, [0, input_dim-1], [-1, 1]), tf.slice(bias, [0, original_input_dim+input_dim-1], [-1, 1])] , axis=1)
                                
            W_masked = W*mask
            if W.get_shape()[0].value == 1: #one set of parameters
                affine = tf.matmul(curr_input, W_masked[0, :, :], transpose_a=False, transpose_b=True)
            else: # batched parameters
                affine = tf.matmul(W_masked, curr_input[:,:,np.newaxis])[:, :, 0]
            if l < len(layerwise_parameters)-1: curr_input = nonlinearity(affine+bias)
            else: curr_input = (affine+bias)
        
        pre_mu = tf.slice(curr_input, [0, 0], [-1, 1])
        pre_scale = tf.slice(curr_input, [0, 1], [-1, 1])                
        return pre_mu, pre_scale

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)

        pre_mu_for_1 = self._pre_mu_for_1
        pre_scale_for_1 = self._pre_scale_for_1
        pre_mu_for_2toD, pre_scale_for_2toD = self.MADE(z0[:, :-1], self._layerwise_parameters, self._mask_tensors, self._nonlinearity)
        if pre_mu_for_1.get_shape().as_list()[0] == 1:            
            pre_mu = tf.concat([tf.tile(pre_mu_for_1, [tf.shape(z0)[0],1]), pre_mu_for_2toD], axis=1)
            pre_scale = tf.concat([tf.tile(pre_scale_for_1, [tf.shape(z0)[0], 1]), pre_scale_for_2toD], axis=1)
        else:    
            pre_mu = tf.concat([pre_mu_for_1, pre_mu_for_2toD], axis=1)
            pre_scale = tf.concat([pre_scale_for_1, pre_scale_for_2toD], axis=1)

        if self._mode == 'RNN':
            mu = pre_mu
            scale = tf.clip_by_value(tf.nn.sigmoid(pre_scale), 1e-10, np.inf)  
            z = scale*z0+(1-scale)*mu
            log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.nn.softplus(pre_scale)/(np.log(1+np.exp(0))), 1e-10, np.inf)  
            z = mu+scale*z0
            log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ExponentialScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.exp(pre_scale), 1e-10, np.inf)  
            z = mu+scale*z0
            log_abs_det_jacobian = tf.reduce_sum(pre_scale, axis=[1], keep_dims=True)
        elif self._mode == 'BoundedScaleShift':
            mu = pre_mu
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale+scipy.special.logit(1/gap))*gap, 1e-10, np.inf)  
            z = mu+scale*z0
            log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'VolumePreserving':
            mu = pre_mu
            scale = tf.ones(shape=tf.shape(pre_scale))
            z = mu+z0
            log_abs_det_jacobian = 0
        elif self._mode == 'BoundedVolumePreserving':
            mu = tf.nn.tanh(pre_mu)*10
            scale = tf.ones(shape=tf.shape(pre_scale))
            z = mu+z0
            log_abs_det_jacobian = 0
        else: quit()

        log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        return z, log_pdf_z
    
    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)

        pre_mu_list = []
        pre_scale_list = []
        pre_mu_for_1 = self._pre_mu_for_1
        pre_scale_for_1 = self._pre_scale_for_1
        if pre_mu_for_1.get_shape().as_list()[0] == 1:
            pre_mu_list.append(tf.tile(pre_mu_for_1, [tf.shape(z)[0], 1]))
            pre_scale_list.append(tf.tile(pre_scale_for_1, [tf.shape(z)[0], 1]))
        else:    
            pre_mu_list.append(pre_mu_for_1)
            pre_scale_list.append(pre_scale_for_1)
        
        mu_list = []
        scale_list = []
        z0_list = []
        for i in range(self._input_dim):
            if i % 20 == 0: print('Inverse transform progress: '+ str(i) + '/' + str(self._input_dim))                
            if i == 0:
                pre_mu_i, pre_scale_i = pre_mu_list[i], pre_scale_list[i]
            else:
                pre_mu_i, pre_scale_i = self.MADE_single(tf.concat(z0_list[:i], axis=1), self._layerwise_parameters, self._mask_tensors, self._nonlinearity)
                pre_mu_list.append(pre_mu_i)
                pre_scale_list.append(pre_scale_i)

            if self._mode == 'RNN':
                mu_i = pre_mu_i
                scale_i = tf.clip_by_value(tf.nn.sigmoid(pre_scale_i), 1e-10, np.inf)  
                z0_i = (z[:, i, np.newaxis]-(1-scale_i)*mu_i)/scale_i
            elif self._mode == 'ScaleShift':
                mu_i = pre_mu_i
                scale_i = tf.clip_by_value(tf.nn.softplus(pre_scale_i)/(np.log(1+np.exp(0))), 1e-10, np.inf)  
                z0_i = (z[:, i, np.newaxis]-mu_i)/scale_i
            elif self._mode == 'ExponentialScaleShift':
                mu_i = pre_mu_i
                scale_i = tf.clip_by_value(tf.exp(pre_scale_i), 1e-10, np.inf)  
                z0_i = (z[:, i, np.newaxis]-mu_i)/scale_i
            elif self._mode == 'BoundedScaleShift':
                mu_i = pre_mu_i
                gap = (self._max_bounded_scale-self._min_bounded_scale)
                scale_i = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale_i+scipy.special.logit(1/gap))*gap, 1e-10, np.inf)  
                z0_i = (z[:, i, np.newaxis]-mu_i)/scale_i
            elif self._mode == 'VolumePreserving':
                mu_i = pre_mu_i
                scale_i = tf.ones(shape=tf.shape(pre_scale_i))
                z0_i = (z[:, i, np.newaxis]-mu_i)
            elif self._mode == 'BoundedVolumePreserving':
                mu_i = tf.nn.tanh(pre_mu_i)*10
                scale_i = tf.ones(shape=tf.shape(pre_scale_i))
                z0_i = (z[:, i, np.newaxis]-mu_i)
            else: quit()
            
            mu_list.append(mu_i)
            scale_list.append(scale_i)
            z0_list.append(z0_i)  

        if self._mode == 'RNN': log_abs_det_jacobian = -tf.reduce_sum(tf.log(1e-7+tf.concat(scale_list, axis=1)), axis=[1], keep_dims=True)
        elif self._mode == 'ScaleShift': log_abs_det_jacobian = -tf.reduce_sum(tf.log(1e-7+tf.concat(scale_list, axis=1)), axis=[1], keep_dims=True)
        elif self._mode == 'ExponentialScaleShift': log_abs_det_jacobian = -tf.reduce_sum(tf.concat(pre_scale_list, axis=1), axis=[1], keep_dims=True)
        elif self._mode == 'BoundedScaleShift': log_abs_det_jacobian = -tf.reduce_sum(tf.log(1e-7+tf.concat(scale_list, axis=1)), axis=[1], keep_dims=True)
        elif self._mode == 'VolumePreserving': log_abs_det_jacobian = 0
        elif self._mode == 'BoundedVolumePreserving': log_abs_det_jacobian = 0

        z0 = tf.concat(z0_list, axis=1)
        log_pdf_z0 = log_pdf_z - log_abs_det_jacobian
        return z0, log_pdf_z0

class RealNVPFlow():
    """
    Real NVP Flow class.

    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    layer_expansions = [5, 5]
    same_dim = None

    def __init__(self, input_dim, parameters, mode='ScaleShift', name='realNVP_transform'):   #real
        self._parameter_scale = 1
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim
        self._layer_expansions = RealNVPFlow.layer_expansions
        self._nonlinearity = helper.LeakyReLU # tf.nn.tanh
        self._mode = mode
        if RealNVPFlow.same_dim is None: self._same_dim = int(float(self._input_dim)/2.)
        else: self._same_dim = RealNVPFlow.same_dim
        self._change_dim = self._input_dim - self._same_dim
        self._max_bounded_scale = 5
        self._min_bounded_scale = 1/self._max_bounded_scale

        assert (self._input_dim > 1)
        assert (self._input_dim % 2 == 0)
        assert (self._max_bounded_scale > 1 and self._min_bounded_scale >= 0 and self._max_bounded_scale > self._min_bounded_scale)

        self._parameters.get_shape().assert_is_compatible_with([None, RealNVPFlow.required_num_parameters(self._input_dim)])

        self._layerwise_parameters = []
        start_ind = 0
        concat_layer_expansions = [1, *RealNVPFlow.layer_expansions, 2]
        for l in range(len(concat_layer_expansions)-1):
            if l == 0: W_num_param = concat_layer_expansions[l]*(self._same_dim)*concat_layer_expansions[l+1]*(self._change_dim) # matrix
            else: W_num_param = concat_layer_expansions[l]*(self._change_dim)*concat_layer_expansions[l+1]*(self._change_dim) # matrix
            B_num_param = concat_layer_expansions[l+1]*(self._change_dim) # bias
            W_l_flat = tf.slice(self._parameters, [0, start_ind], [-1, W_num_param])
            B_l_flat = tf.slice(self._parameters, [0, start_ind+W_num_param], [-1, B_num_param])
            if l == 0: W_l = tf.reshape(W_l_flat, [-1, concat_layer_expansions[l+1]*(self._change_dim), concat_layer_expansions[l]*(self._same_dim)])              
            else: W_l = tf.reshape(W_l_flat, [-1, concat_layer_expansions[l+1]*(self._change_dim), concat_layer_expansions[l]*(self._change_dim)])              
            B_l = tf.reshape(B_l_flat, [-1, concat_layer_expansions[l+1]*(self._change_dim)])
            self._layerwise_parameters.append((W_l, B_l))  
            start_ind += (W_num_param+B_num_param)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):
        if RealNVPFlow.same_dim is None: same_dim = int(float(input_dim)/2.)
        else: same_dim = RealNVPFlow.same_dim
        change_dim = input_dim-same_dim

        n_parameters = 0
        concat_layer_expansions = [1, *RealNVPFlow.layer_expansions, 2]
        for l in range(len(concat_layer_expansions)-1):
            if l == 0: n_parameters += concat_layer_expansions[l]*(same_dim)*concat_layer_expansions[l+1]*(change_dim) # matrix
            else: n_parameters += concat_layer_expansions[l]*(change_dim)*concat_layer_expansions[l+1]*(change_dim) # matrix
            n_parameters += concat_layer_expansions[l+1]*(change_dim) # bias
        return n_parameters

    def transformFunction(self, input_x, layerwise_parameters, nonlinearity):
        input_dim = input_x.get_shape().as_list()[1]
        curr_input = input_x
        for l in range(len(layerwise_parameters)):
            W, bias = layerwise_parameters[l]
            if W.get_shape()[0].value == 1: #one set of parameters
                affine = tf.matmul(curr_input, W[0, :, :], transpose_a=False, transpose_b=True)
            else: # batched parameters
                affine = tf.matmul(W, curr_input[:,:,np.newaxis])[:, :, 0]
            if l < len(layerwise_parameters)-1: curr_input = nonlinearity(affine+bias)
            else: curr_input = (affine+bias)

        pre_mu = tf.slice(curr_input, [0, 0], [-1, input_dim])
        pre_scale = tf.slice(curr_input, [0, input_dim], [-1, input_dim])
        return pre_mu, pre_scale

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        
        z0_same = z0[:, :self._same_dim]
        z0_change = z0[:, self._same_dim:]
        pre_mu, pre_scale = self.transformFunction(z0_same, self._layerwise_parameters, self._nonlinearity)

        if self._mode == 'RNN':
            mu = pre_mu
            scale = tf.clip_by_value(tf.nn.sigmoid(pre_scale), 1e-10, np.inf)  
            z_change = scale*z0_change+(1-scale)*mu
            log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.nn.softplus(pre_scale)/(np.log(1+np.exp(0))), 1e-10, np.inf)
            z_change = mu+scale*z0_change
            log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ExponentialScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.exp(pre_scale), 1e-10, np.inf)
            z_change = mu+scale*z0_change
            log_abs_det_jacobian = tf.reduce_sum(pre_scale, axis=[1], keep_dims=True)
        elif self._mode == 'BoundedScaleShift':
            mu = pre_mu
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale+scipy.special.logit(1/gap))*gap, 1e-10, np.inf)
            z_change = mu+scale*z0_change
            log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'VolumePreserving':
            mu = pre_mu
            scale = tf.ones(shape=tf.shape(pre_scale))
            z_change = mu+z0_change
            log_abs_det_jacobian = 0
        elif self._mode == 'BoundedVolumePreserving':
            mu = tf.nn.tanh(pre_mu)*10
            scale = tf.ones(shape=tf.shape(pre_scale))
            z_change = mu+z0_change
            log_abs_det_jacobian = 0
        else: quit()

        z = tf.concat([z0_same, z_change], axis=1)
        log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        return z, log_pdf_z
    
    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)
        
        z_same = z[:, :self._same_dim]
        z_change = z[:, self._same_dim:]
        pre_mu, pre_scale = self.transformFunction(z_same, self._layerwise_parameters, self._nonlinearity)

        if self._mode == 'RNN':
            mu = pre_mu
            scale = tf.clip_by_value(tf.nn.sigmoid(pre_scale), 1e-10, np.inf)  
            z0_change = (z_change-(1-scale)*mu)/scale
            log_abs_det_jacobian = -tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.nn.softplus(pre_scale)/(np.log(1+np.exp(0))), 1e-10, np.inf)  
            z0_change = (z_change-mu)/scale
            log_abs_det_jacobian = -tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ExponentialScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.exp(pre_scale), 1e-10, np.inf)  
            z0_change = (z_change-mu)/scale
            log_abs_det_jacobian = -tf.reduce_sum(pre_scale, axis=[1], keep_dims=True)
        elif self._mode == 'BoundedScaleShift':
            mu = pre_mu
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale+scipy.special.logit(1/gap))*gap, 1e-10, np.inf)  
            z0_change = (z_change-mu)/scale
            log_abs_det_jacobian = -tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'VolumePreserving':
            mu = pre_mu
            scale = tf.ones(shape=tf.shape(pre_scale))
            z0_change = (z_change-mu)
            log_abs_det_jacobian = 0
        elif self._mode == 'BoundedVolumePreserving':
            mu = tf.nn.tanh(pre_mu)*10
            scale = tf.ones(shape=tf.shape(pre_scale))
            z0_change = (z_change-mu)
            log_abs_det_jacobian = 0
        else: quit()

        z0 = tf.concat([z_same, z0_change], axis=1)
        log_pdf_z0 = log_pdf_z - log_abs_det_jacobian
        return z0, log_pdf_z0

class SerialFlow():
    def __init__(self, transforms, name='serial_transform'): 
        self._transforms = transforms
    
    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        curr_z, curr_log_pdf_z = z0, log_pdf_z0
        for i in range(len(self._transforms)):
            curr_z, curr_log_pdf_z = self._transforms[i].transform(curr_z, curr_log_pdf_z)
        return curr_z, curr_log_pdf_z
    
    def inverse_transform(self, z, log_pdf_z):
        verify_size(z, log_pdf_z)
        curr_z0, curr_log_pdf_z0 = z, log_pdf_z
        for i in range(len(self._transforms)):
            curr_z0, curr_log_pdf_z0 = self._transforms[-i-1].inverse_transform(curr_z0, curr_log_pdf_z0)
        return curr_z0, curr_log_pdf_z0

def verify_size(z0, log_pdf_z0):
  z0.get_shape().assert_is_compatible_with([None, None])
  if log_pdf_z0 is not None:
    log_pdf_z0.get_shape().assert_is_compatible_with([None, 1])

def _jacobian(y, x):
    batch_size = y.get_shape()[0].value
    flat_y = tf.reshape(y, [batch_size, -1])
    num_y = flat_y.get_shape()[1].value
    one_hot = np.zeros((batch_size, num_y))
    jacobian_rows = []
    for i in range(num_y):
        one_hot.fill(0)
        one_hot[:, i] = 1
        grad_flat_y = tf.constant(one_hot, dtype=y.dtype)

        grad_x, = tf.gradients(flat_y, [x], grad_flat_y, gate_gradients=True)
        assert grad_x is not None, "Variable `y` is not computed from `x`."

        row = tf.reshape(grad_x, [batch_size, 1, -1])
        jacobian_rows.append(row)

    return tf.concat(jacobian_rows, 1)

def _log_determinant(matrices):
    _, logdet = np.linalg.slogdet(matrices)
    return logdet.reshape(len(matrices), 1)

def _check_logdet(flow, z0, log_pdf_z0, rtol=1e-5):
    z1, log_pdf_z1 = flow.transform(z0, log_pdf_z0)
    jacobian = _jacobian(z1, z0)
    
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()  
    sess.run(init)
    out_jacobian, out_log_pdf_z0, out_log_pdf_z1 = sess.run([jacobian, log_pdf_z0, log_pdf_z1])

    # The logdet will be ln|det(dz1/dz0)|.
    logdet_expected = _log_determinant(out_jacobian)
    logdet = out_log_pdf_z0 - out_log_pdf_z1

    # if np.allclose(logdet_expected, logdet, rtol=rtol):
    if np.all(np.abs(logdet_expected-logdet)<rtol):
        print('Transform update correct.')
        print('logdet_expected: ', logdet_expected)
        print('logdet: ', logdet)
    else: 
        print('Transform update incorrect!!!!!!!!!!!!!!!!')
        print('logdet_expected: ', logdet_expected)
        print('logdet: ', logdet)
        # np.abs(logdet_expected-logdet) 1e-08+rtol*np.abs(logdet)

# pdb.set_trace()

# batch_size = 12
# n_latent = 4
# n_out = 7
# n_input_NOM, n_output_NOM = 3, 4
# name = 'transform'
# transform_to_check = RiemannianFlow
# n_parameter = transform_to_check.required_num_parameters(n_latent, n_out, n_input_NOM, n_output_NOM)

# parameters = None
# if n_parameter > 0: parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)

# z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
# log_pdf_z0 = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)
# transform1 = transform_to_check(input_dim=n_latent, output_dim=n_out, n_input_NOM=n_input_NOM, n_output_NOM=n_output_NOM, parameters=parameters)
# z, log_pdf_z, all_scales = transform1.transform(z0, log_pdf_z0, mode='scales')
# additional_jacobian = transform1.jacobian(z0, mode='additional')
# full_jacobian = transform1.jacobian(z0, mode='full')

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# z0_np, log_pdf_z0_np, z_np, log_pdf_z_np, all_scales_np, jacobian_np, full_jacobian_np = sess.run([z0, log_pdf_z0, z, log_pdf_z, all_scales, additional_jacobian, full_jacobian])
# z0_np, log_pdf_z0_np, z_np, log_pdf_z_np = sess.run([z0, log_pdf_z0, z, log_pdf_z])

# print('\n\n\n')
# print('Jacobian Shape: ', jacobian_np.shape)
# print('Example Jacobian:\n', jacobian_np[0, :, :])
# if jacobian_np.shape[1] < jacobian_np.shape[2]:
#     print('Example Jacobian*Jacobian^T:\n', np.dot(jacobian_np[0, :, :], jacobian_np[0, :, :].T))
# else:
#     print('Example Jacobian^T*Jacobian:\n', np.dot(jacobian_np[0, :, :].T, jacobian_np[0, :, :]))
# print('\n\n\n')

# for i in range(jacobian_np.shape[0]):
#     if jacobian_np.shape[1] < jacobian_np.shape[2]:
#         JJT = np.dot(jacobian_np[i, :, :], jacobian_np[i, :, :].T)
#         if np.abs(np.diag(JJT)-JJT[0,0]).max() > 1e-7: print('JJT abs max diagonal offset: ', (JJT[0,0], np.abs(np.diag(JJT)-JJT[0,0]).max()))
#         if np.abs(JJT-np.diag(np.diag(JJT))).max() > 1e-7: print('JJT abs max off-diagonal: ', np.abs(JJT-np.diag(np.diag(JJT))).max())
#     else:
#         JTJ = np.dot(jacobian_np[i, :, :].T, jacobian_np[i, :, :])
#         if np.abs(np.diag(JTJ)-JTJ[0,0]).max() > 1e-7: print('JTJ abs max diagonal offset: ', (JTJ[0,0], np.abs(np.diag(JTJ)-JTJ[0,0]).max()))
#         if np.abs(JTJ-np.diag(np.diag(JTJ))).max() > 1e-7: print('JTJ abs max off-diagonal: ', np.abs(JTJ-np.diag(np.diag(JTJ))).max())

# all_scales_sq = (all_scales_np**2)[:, 0]
# all_JJT_scales = np.zeros((batch_size,))
# for i in range(jacobian_np.shape[0]):
#     if jacobian_np.shape[1] < jacobian_np.shape[2]:
#         JJT = np.dot(jacobian_np[i, :, :], jacobian_np[i, :, :].T)
#         # if JJT.shape[0]>1: print(JJT[0,0], JJT[1,1], JJT[0,0]-JJT[1,1], all_scales_sq[i], all_scales_sq[i]-JJT[0,0])
#         # else: print(JJT[0,0], all_scales_sq[i], all_scales_sq[i]-JJT[0,0])
#         all_JJT_scales[i] = JJT[0,0]
#     else:
#         JTJ = np.dot(jacobian_np[i, :, :].T, jacobian_np[i, :, :])
#         # if JTJ.shape[0]>1: print(JTJ[0,0], JTJ[1,1], JTJ[0,0]-JTJ[1,1], all_scales_sq[i], all_scales_sq[i]-JTJ[0,0])
#         # else: print(JTJ[0,0], all_scales_sq[i], all_scales_sq[i]-JTJ[0,0])
#         all_JJT_scales[i] = JTJ[0,0]
# print('scales_sq vs jacobian scales_sq max abs diff: ', np.abs(all_scales_sq-all_JJT_scales).max())

# all_vol_scales = np.zeros((batch_size,))
# all_vol_scales_2 = np.zeros((batch_size,))
# for i in range(full_jacobian_np.shape[0]):
#     curr_full_J = full_jacobian_np[i,:,:]
#     curr_JTJ = np.dot(curr_full_J.T, curr_full_J)
#     all_vol_scales[i] = np.sqrt(np.linalg.det(curr_JTJ))
#     all_vol_scales_2[i] = curr_JTJ[0,0]**(min(n_out, n_latent)/2) # works only when (n_out-n_latent)>=n_latent curr_JTJ is diagonal only then
# print('all_vol_scales vs all_vol_scales_2 max abs diff: ', np.abs(all_vol_scales-all_vol_scales_2).max())

# all_log_density_changes = -np.log(all_vol_scales)
# all_log_density_changes_computed = (log_pdf_z_np-log_pdf_z0_np)[:,0]
# print('all_log_density_changes vs all_log_density_changes_computed max abs diff: ', np.abs(all_log_density_changes-all_log_density_changes_computed).max())
# pdb.set_trace()

# batch_size = 50
# n_latent = 2
# n_out = 3
# n_input_NOM, n_output_NOM = 4, 4
# name = 'transform'
# transform_to_check = RiemannianFlow
# n_parameter = transform_to_check.required_num_parameters(n_latent, n_out, n_input_NOM, n_output_NOM)

# parameters = None
# if n_parameter > 0: parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)

# z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
# log_pdf_z0 = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)
# transform1 = transform_to_check(input_dim=n_latent, output_dim=n_out, n_input_NOM=n_input_NOM, n_output_NOM=n_output_NOM, parameters=parameters)
# z, log_pdf_z = transform1.transform(z0, log_pdf_z0)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)

# import time
# print('Start Timer: ')
# start = time.time();
# for i in range(1000):
#     z0_np, log_pdf_z0_np, z_np, log_pdf_z_np = sess.run([z0, log_pdf_z0, z, log_pdf_z])

# end = time.time()
# print('Time: {:.3f}\n'.format((end - start)))
# pdb.set_trace()

# batch_size = 20
# n_latent = 4
# name = 'transform'
# # transform_to_check = ConnectedPiecewiseOrthogonalMap
# transform_to_check = PiecewisePlanarScalingMap
# n_parameter = transform_to_check.required_num_parameters(n_latent)

# parameters = None
# if n_parameter > 0: parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)

# z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
# log_pdf_z0 = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)
# transform1 = transform_to_check(input_dim=n_latent, parameters=parameters)
# z, log_scales = transform1.transform(z0)
# jacobian = transform1.jacobian(z0)
# log_pdf_z = log_pdf_z0

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# z0_np, log_pdf_z0_np, z_np, log_scales_np, jacobian_np = sess.run([z0, log_pdf_z0, z, log_scales, jacobian])

# print('\n\n\n')
# print('Jacobian Shape: ', jacobian_np.shape)
# print('Example Jacobian:\n', jacobian_np[0, :, :])
# print('Example Jacobian*Jacobian^T:\n', np.dot(jacobian_np[0, :, :], jacobian_np[0, :, :].T))
# print('Example Jacobian^T*Jacobian:\n', np.dot(jacobian_np[0, :, :].T, jacobian_np[0, :, :]))
# print('\n\n\n')

# all_scales = ((np.exp(log_scales_np)**2))[:,0]
# for i in range(jacobian_np.shape[0]):
#     JJT = np.dot(jacobian_np[i, :, :], jacobian_np[i, :, :].T)
#     # JTJ = np.dot(jacobian_np[i, :, :].T, jacobian_np[i, :, :])
#     print(JJT[0,0], all_scales[i])
# pdb.set_trace()

# batch_size = 50
# n_latent = 1
# n_out = 3
# name = 'transform'
# transform_to_check = OthogonalProjectionMap
# n_parameter = transform_to_check.required_num_parameters(n_latent, n_out)

# parameters = None
# if n_parameter > 0: parameters = 10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)

# z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
# log_pdf_z0 = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)
# transform1 = transform_to_check(input_dim=n_latent, output_dim=n_out, parameters=parameters)
# jacobian = transform1.jacobian(z0)
# transform1._mode = 'vector'
# z_1 = transform1.transform(z0, log_pdf_z0)
# transform1._mode = 'matrix'
# z_2 = transform1.transform(z0, log_pdf_z0)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# z0_np, log_pdf_z0_np, z_1_np, z_2_np, jacobian_np = sess.run([z0, log_pdf_z0, z_1, z_2, jacobian])
# print(np.max(np.abs(z_1_np-z_2_np)))

# print('\n\n\n')
# print('Jacobian Shape: ', jacobian_np.shape)
# print('Example Jacobian:\n', jacobian_np[0, :, :])
# print('Example Jacobian*Jacobian^T:\n', np.dot(jacobian_np[0, :, :], jacobian_np[0, :, :].T))
# print('Example Jacobian^T*Jacobian:\n', np.dot(jacobian_np[0, :, :].T, jacobian_np[0, :, :]))
# print('\n\n\n')
# pdb.set_trace()

# batch_size = 4
# n_latent = 50
# name = 'transform'
# transform_to_check = ManyReflectionRotationFlow
# # transform_to_check = CompoundRotationFlow
# # transform_to_check = HouseholdRotationFlow
# n_parameter = transform_to_check.required_num_parameters(n_latent)

# parameters = None
# if n_parameter > 0: parameters = 10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)

# z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
# log_pdf_z0 = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)
# transform1 = transform_to_check(input_dim=n_latent, parameters=parameters)
# z, log_pdf_z = transform1.transform(z0, log_pdf_z0)
# z0_inv, log_pdf_z0_inv = transform1.inverse_transform(z, log_pdf_z)
# rot_mat = transform1.get_batched_rot_matrix()

# # transform1._mode = 'matrix'
# # z_2, log_pdf_z = transform1.transform(z0, log_pdf_z0)
# # z0_inv_2, log_pdf_z0_inv = transform1.inverse_transform(z_2, log_pdf_z)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# z0_np, log_pdf_z0_np, z_np, log_pdf_z_np, z0_inv_np, log_pdf_z0_inv_np, rot_mat_np = sess.run([z0, log_pdf_z0, z, log_pdf_z, z0_inv, log_pdf_z0_inv, rot_mat])
# # z0_np, log_pdf_z0_np, z_np, z_np_2, log_pdf_z_np, z0_inv_np, log_pdf_z0_inv_np = sess.run([z0, log_pdf_z0, z, z_2, log_pdf_z, z0_inv, log_pdf_z0_inv])
# # rot_mat_np = sess.run(transform1.get_batched_rot_matrix())

# # print('n_steps of reflection: ', transform1._n_steps)
# # print('initial reflection: ', transform1._init_reflection)
# print('rotation matrix diag: ')
# print(np.diag(rot_mat_np[0]))

# print('rotation determinant: ', np.linalg.det(rot_mat_np[0]))
# print(np.dot(rot_mat_np[0], rot_mat_np[0].T))
# print(np.abs(np.dot(z0_np, rot_mat_np[0].T)-z_np).max())

# print(np.max(np.abs(z0_np-z0_inv_np)))
# print(np.max(np.abs(log_pdf_z0_np-log_pdf_z0_inv_np)))

# print('Mean absolute differences:')
# print(np.mean(np.abs((z0_np-z_np)),axis=0))

# # last_vec_np = sess.run(transform1._list_batched_householder_dirs[-1])
# # print(last_vec_np)
# # pdb.set_trace()

# import matplotlib.pyplot as plt
# import numpy as np
# print(rot_mat_np[0])
# plt.imshow(rot_mat_np[0], cmap='hot', interpolation='nearest')
# plt.show()

# pdb.set_trace()






# import time
# print('Start Timer: ', transform1._mode)
# start = time.time();
# for i in range(1000):
#     z0_np, log_pdf_z0_np, z_np, log_pdf_z_np, z0_inv_np, log_pdf_z0_inv_np = sess.run([z0, log_pdf_z0, z, log_pdf_z, z0_inv, log_pdf_z0_inv])

# end = time.time()
# print('Time: {:.3f}\n'.format((end - start)))


# n_tests = 1
# batch_size = 5
# n_latent = 50
# name = 'transform'
# for transform_to_check in [\
#                            # PlanarFlow, \
#                            # RadialFlow, \
#                            # SpecificOrderDimensionFlow, \
#                            # InverseOrderDimensionFlow, \
#                            # PermuteDimensionFlow, \
#                            # ScaleDimensionFlow, \
#                            # OpenIntervalDimensionFlow, \
#                            # InverseOpenIntervalDimensionFlow, \
#                            # HouseholdRotationFlow, \
#                            # CompoundRotationFlow, \
#                            PiecewisePlanarScalingFlow,
#                            # LinearIARFlow, \
#                            # NonLinearIARFlow, \
#                            # RealNVPFlow, \
#                            ]:
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('\n\n\n')
#     print('            '+str(transform_to_check)+'               ')
#     print('\n\n\n')    
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

#     n_parameter = transform_to_check.required_num_parameters(n_latent)

#     for parameter_scale in [1, 10]:
#         parameters = None
#         if n_parameter > 0: parameters = parameter_scale*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
#         transform_object =  transform_to_check(input_dim=n_latent, parameters=parameters)

#         # z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
#         z0 = tf.random_uniform(shape=(batch_size, n_latent), dtype=tf.float32)
#         log_pdf_z0 = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)

#         for repeat in range(n_tests): _check_logdet(transform_object, z0, log_pdf_z0)

# pdb.set_trace()

















# class ConnectedPiecewiseOrthogonalMap():
#     """
#     Connected Piecewise Orthogonal Map class with Jacobians specified as scaled multiples of orthogonal matrices.
#     Args:
#       parameters: parameters of transformation all appended.
#       input_dim : input dimensionality of the transformation. 
#     Raises:
#       ValueError: 
#     """
#     rotation_flow_class = HouseholdRotationFlow # CompoundRotationFlow

#     def __init__(self, input_dim, parameters, margin_mode='NoGradient', name='connected_piecewise_orthogonal_transform'):   
#         self._parameter_scale = 1.
#         self._parameters = parameters
#         self._parameters = self._parameter_scale*self._parameters
#         self._input_dim = input_dim
#         self._margin_mode = margin_mode

#         assert (self._margin_mode == 'NoGradient' or self._margin_mode == 'ST')

#         self._parameters.get_shape().assert_is_compatible_with([None, ConnectedPiecewiseOrthogonalMap.required_num_parameters(self._input_dim)])
        
#         param_index = 0
#         self._pos_rotation_param, param_index = helper.slice_parameters(self._parameters, param_index, ConnectedPiecewiseOrthogonalMap.rotation_flow_class.required_num_parameters(self._input_dim))
#         self._neg_rotation_param, param_index = helper.slice_parameters(self._parameters, param_index, ConnectedPiecewiseOrthogonalMap.rotation_flow_class.required_num_parameters(self._input_dim))
#         self._pos_pre_scale, param_index = helper.slice_parameters(self._parameters, param_index, 1)
#         self._neg_pre_scale, param_index = helper.slice_parameters(self._parameters, param_index, 1)
#         self._hyper_pre_bias, param_index = helper.slice_parameters(self._parameters, param_index, 1)
#         self._hyper_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim) 
#         self._output_shift_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim) 

#         self._pos_scale = tf.nn.softplus(self._pos_pre_scale)/np.log(1+np.exp(0))
#         self._neg_scale = tf.nn.softplus(self._neg_pre_scale)/np.log(1+np.exp(0))
#         self._hyper_bias = tf.nn.softplus(self._hyper_pre_bias)/np.log(1+np.exp(0))
#         self._hyper_vec_dir = self._hyper_vec/helper.safe_tf_sqrt(tf.reduce_sum(self._hyper_vec**2, axis=1, keep_dims=True))
#         self._pos_rotation_flow = ConnectedPiecewiseOrthogonalMap.rotation_flow_class(self._input_dim, self._pos_rotation_param) 
#         self._neg_rotation_flow = ConnectedPiecewiseOrthogonalMap.rotation_flow_class(self._input_dim, self._neg_rotation_param) 

#     @property
#     def input_dim(self):
#         return self._input_dim

#     @property
#     def output_dim(self):
#         return self._input_dim

#     @staticmethod
#     def required_num_parameters(input_dim):
#         n_rot_param = ConnectedPiecewiseOrthogonalMap.rotation_flow_class.required_num_parameters(input_dim)
#         return 2*n_rot_param+1+1+1+input_dim+input_dim

#     def jacobian(self, z0):
#         z_centered = z0-self._hyper_vec_dir*self._hyper_bias

#         margin = tf.reduce_sum(self._hyper_vec_dir*z_centered, axis=1, keep_dims=True)
#         if self._margin_mode == 'NoGradient':
#             pos_mask = tf.stop_gradient(tf.cast(margin>=0, tf.float32)) # margin >= 0 return 1 else 0 
#         elif self._margin_mode == 'ST':
#             pos_mask = helper.binaryStochastic_ST(margin) # margin >= ~9e-8 return 1 else 0
#         neg_mask = 1-pos_mask
        
#         pos_batched_rot_matrix = self._pos_rotation_flow.get_batched_rot_matrix()
#         neg_batched_rot_matrix = self._neg_rotation_flow.get_batched_rot_matrix()
#         scaled_pos_batched_rot_matrix = self._pos_scale[:, :, np.newaxis]*pos_batched_rot_matrix
#         scaled_neg_batched_rot_matrix = self._neg_scale[:, :, np.newaxis]*neg_batched_rot_matrix
#         jacobian = pos_mask[:, :, np.newaxis]*scaled_pos_batched_rot_matrix+neg_mask[:, :, np.newaxis]*scaled_neg_batched_rot_matrix
#         return jacobian

#     def transform(self, z0):
#         z_centered = z0-self._hyper_vec_dir*self._hyper_bias

#         margin = tf.reduce_sum(self._hyper_vec_dir*z_centered, axis=1, keep_dims=True)
#         if self._margin_mode == 'NoGradient':
#             pos_mask = tf.stop_gradient(tf.cast(margin>=0, tf.float32)) # margin >= 0 return 1 else 0 
#         elif self._margin_mode == 'ST':
#             pos_mask = helper.binaryStochastic_ST(margin) # margin >= ~9e-8 return 1 else 0
#         neg_mask = 1-pos_mask
        
#         z_pos_rot, _ = self._pos_rotation_flow.transform(z_centered, tf.zeros(shape=(tf.shape(z0)[0], 1), dtype=tf.float32))
#         z_neg_rot, _ = self._neg_rotation_flow.transform(z_centered, tf.zeros(shape=(tf.shape(z0)[0], 1), dtype=tf.float32))
#         z_pos_scale_rot = self._pos_scale*z_pos_rot
#         z_neg_scale_rot = self._neg_scale*z_neg_rot

#         z_scale_rot = pos_mask*z_pos_scale_rot+neg_mask*z_neg_scale_rot
#         z = z_scale_rot+self._hyper_vec_dir*self._hyper_bias+self._output_shift_vec
#         scales = pos_mask*self._pos_scale+neg_mask*self._neg_scale 
#         return z, scales















# class RiemannianFlowOld():
#     """
#     Projective Flow class.
#     Args:
#       parameters: parameters of transformation all appended.
#       input_dim : input dimensionality of the transformation. 
#     Raises:
#       ValueError: 
#     """
#     def __init__(self, input_dim, parameters, additional_dim=3, k_start=1, init_reflection=1, manifold_nonlinearity=tf.nn.tanh, polinomial_degree=3, name='riemannian_transform'):   
#         self._parameter_scale = 1.
#         self._parameters = self._parameter_scale*parameters
#         self._input_dim = input_dim
#         self._additional_dim = additional_dim
#         self._k_start = k_start
#         self._init_reflection = init_reflection
#         self._polinomial_degree = polinomial_degree
#         self._manifold_nonlinearity = manifold_nonlinearity
#         assert (additional_dim > 0)
#         assert (self._input_dim > additional_dim)

#     @property
#     def input_dim(self):
#         return self._input_dim

#     @property
#     def output_dim(self):
#         return self._input_dim+self._additional_dim

#     def apply_manifold_nonlin(self, x_k, NonlinK_param):
#         if self._manifold_nonlinearity == tf.nn.tanh: return tf.nn.tanh(x_k), helper.tanh_J(x_k)
#         if self._manifold_nonlinearity == tf.nn.sigmoid: return tf.nn.sigmoid(x_k), helper.sigmoid_J(x_k)
#         if self._manifold_nonlinearity == tf.nn.relu: return tf.nn.relu(x_k), helper.relu_J(x_k)
#         if self._manifold_nonlinearity == helper.parametric_relu:
#             param_index = 0
#             positive, param_index = helper.slice_parameters(NonlinK_param, param_index, self._additional_dim)
#             negative, param_index = helper.slice_parameters(NonlinK_param, param_index, self._additional_dim)
#             return helper.parametric_relu(x_k, positive, negative), helper.parametric_relu_J(x_k, positive, negative)
#         if self._manifold_nonlinearity == helper.polinomial_nonlin:
#             param_index = 0
#             positive, param_index = helper.slice_parameters(NonlinK_param, param_index, self._additional_dim)
#             negative, param_index = helper.slice_parameters(NonlinK_param, param_index, self._additional_dim)            
#             pdb.set_trace()
#             return y_k, manifold_nonlinearity_J

#     def get_rotation_tensor(self, dim, rot_params):
#         rot_tensor = helper.householder_rotations_tf(n=dim, batch=tf.shape(rot_params)[0], k_start=self._k_start, init_reflection=self._init_reflection, params=rot_params)
#         return rot_tensor

#     @staticmethod
#     def required_num_parameters(input_dim, additional_dim=3, k_start=1, manifold_nonlinearity=tf.nn.tanh, polinomial_degree=3): 
#         if additional_dim == 1:
#             n_C_param = input_dim
#             n_RK_1_param = 1
#             n_RK_2_param = 1
#         else:
#             n_C_param = HouseholdRotationFlow.required_num_parameters(input_dim, k_start)
#             n_RK_1_param = HouseholdRotationFlow.required_num_parameters(additional_dim, k_start)
#             n_RK_2_param = HouseholdRotationFlow.required_num_parameters(additional_dim, k_start)
            
#         n_pre_bias_param = additional_dim
#         n_pre_scale_param = additional_dim
#         if manifold_nonlinearity == tf.nn.tanh or manifold_nonlinearity == tf.nn.sigmoid or manifold_nonlinearity == tf.nn.relu :
#             n_NonlinK_param = 0
#         elif manifold_nonlinearity == helpe.parametric_relu:
#             n_NonlinK_param = 2*additional_dim
#         elif manifold_nonlinearity == RiemannianFlow.polinomial_nonlin:
#             n_NonlinK_param = (polinomial_degree+1)*additional_dim
#         n_post_bias_param = additional_dim
#         n_post_scale_param = additional_dim

#         n_RN_param = HouseholdRotationFlow.required_num_parameters(input_dim, k_start)
#         n_RG_param = HouseholdRotationFlow.required_num_parameters(input_dim+additional_dim, k_start)
        
#         return n_C_param+n_RK_1_param+n_RK_2_param+ \
#                n_pre_bias_param+n_pre_scale_param+n_NonlinK_param+ \
#                n_post_bias_param+n_post_scale_param+ \
#                n_RN_param+n_RG_param

#     def transform(self, z0, log_pdf_z0):
#         verify_size(z0, log_pdf_z0)
#         self._parameters.get_shape().assert_is_compatible_with([None, RiemannianFlow.required_num_parameters(
#             self._input_dim, self._additional_dim, self._k_start, self._manifold_nonlinearity, self._polinomial_degree)])

#         param_index = 0
#         if self._additional_dim == 1:             
#             C_param, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim)
#             RK_1_param, param_index = helper.slice_parameters(self._parameters, param_index, 1)
#             RK_2_param, param_index = helper.slice_parameters(self._parameters, param_index, 1)
#         else: 
#             C_param, param_index = helper.slice_parameters(self._parameters, param_index, HouseholdRotationFlow.required_num_parameters(self._input_dim, self._k_start))
#             RK_1_param, param_index = helper.slice_parameters(self._parameters, param_index, HouseholdRotationFlow.required_num_parameters(self._additional_dim, self._k_start))
#             RK_2_param, param_index = helper.slice_parameters(self._parameters, param_index, HouseholdRotationFlow.required_num_parameters(self._additional_dim, self._k_start))

#         pre_bias, param_index = helper.slice_parameters(self._parameters, param_index, self._additional_dim)
#         pre_scale, param_index = helper.slice_parameters(self._parameters, param_index, self._additional_dim)
#         if self._manifold_nonlinearity == tf.nn.tanh or self._manifold_nonlinearity == tf.nn.sigmoid or self._manifold_nonlinearity == tf.nn.relu :
#             NonlinK_param = None
#         elif manifold_nonlinearity == helper.parametric_relu:
#             NonlinK_param, param_index = helper.slice_parameters(self._parameters, param_index, 2*self._additional_dim)
#         elif manifold_nonlinearity == helper.polinomial_nonlin:
#             NonlinK_param, param_index = helper.slice_parameters(self._parameters, param_index, (self._polinomial_degree+1)*self._additional_dim)   
#         post_bias, param_index = helper.slice_parameters(self._parameters, param_index, self._additional_dim)
#         post_scale, param_index = helper.slice_parameters(self._parameters, param_index, self._additional_dim)
        
#         RN_param, param_index = helper.slice_parameters(self._parameters, param_index, HouseholdRotationFlow.required_num_parameters(self._input_dim, self._k_start))
#         RG_param, param_index = helper.slice_parameters(self._parameters, param_index, HouseholdRotationFlow.required_num_parameters(self._input_dim+self._additional_dim, self._k_start))
        
#         if self._additional_dim == 1:             
#             C = C_param[:,np.newaxis,:]
#             RK_1 = RK_1_param[:,np.newaxis,:]
#             RK_2 = RK_2_param[:,np.newaxis,:]
#         else: 
#             C = self.get_rotation_tensor(self._input_dim, C_param)[:,-self._additional_dim:,:]
#             RK_1 = self.get_rotation_tensor(self._additional_dim, RK_1_param)
#             RK_2 = self.get_rotation_tensor(self._additional_dim, RK_2_param)
        
#         RN = self.get_rotation_tensor(self._input_dim, RN_param)
#         RG = self.get_rotation_tensor(self._input_dim+self._additional_dim, RG_param)

#         if self._manifold_nonlinearity == tf.nn.tanh or self._manifold_nonlinearity == tf.nn.sigmoid or self._manifold_nonlinearity == tf.nn.relu :
#             NonlinK_param = None
#         elif manifold_nonlinearity == helper.parametric_relu:
#             NonlinK_param, param_index = helper.slice_parameters(self._parameters, param_index, 2*self._additional_dim)
#         elif manifold_nonlinearity == helper.polinomial_nonlin:
#             NonlinK_param, param_index = helper.slice_parameters(self._parameters, param_index, (self._polinomial_degree+1)*self._additional_dim)
        
#         # C*z
#         if C.get_shape()[0].value == 1: #one set of parameters
#             Cz = tf.matmul(z0, C[0, :, :], transpose_a=False, transpose_b=True)
#         else: # batched parameters
#             Cz = tf.matmul(C, z0[:,:,np.newaxis])[:, :, 0]

#         # RK1*C*z
#         if RK_1.get_shape()[0].value == 1: #one set of parameters
#             RK1Cz = tf.matmul(Cz, RK_1[0, :, :], transpose_a=False, transpose_b=True)
#         else: # batched parameters
#             RK1Cz = tf.matmul(RK_1, Cz[:,:,np.newaxis])[:, :, 0]

#         pre_nonlinK = pre_bias+pre_scale*RK1Cz
#         nonlinK, nonlinK_J = self.apply_manifold_nonlin(pre_nonlinK, NonlinK_param)
#         post_nonlinK = post_bias+post_scale*nonlinK

#         # RK2*nonlin(a(C*z)+b)
#         if RK_2.get_shape()[0].value == 1: #one set of parameters
#             y_k = tf.matmul(post_nonlinK, RK_2[0, :, :], transpose_a=False, transpose_b=True)
#         else: # batched parameters
#             y_k = tf.matmul(RK_2, post_nonlinK[:,:,np.newaxis])[:, :, 0]
        
#         # RK2*nonlin(a(C*z)+b)
#         if RN.get_shape()[0].value == 1: #one set of parameters
#             y_n = tf.matmul(z0, RN[0, :, :], transpose_a=False, transpose_b=True)
#         else: # batched parameters
#             y_n = tf.matmul(RN, z0[:,:,np.newaxis])[:, :, 0]
#         y = tf.concat([y_n, y_k], axis=-1)
        
#         # full rotation
#         if RK_2.get_shape()[0].value == 1: #one set of parameters
#             z = tf.matmul(y, RG[0, :, :], transpose_a=False, transpose_b=True)
#         else: # batched parameters
#             z = tf.matmul(RG, y[:,:,np.newaxis])[:, :, 0]

#         log_volume_increase_ratio = tf.reduce_sum(0.5*tf.log(1e-7+(nonlinK_J*pre_scale*post_scale)**2+1), axis=[1], keep_dims=True)
#         log_pdf_z = log_pdf_z0 - log_volume_increase_ratio
#         return z, log_pdf_z































# batch_size = 3
# n_latent = 20
# name = 'transform'
# transform_to_check = PiecewisePlanarScalingFlow
# n_parameter = transform_to_check.required_num_parameters(n_latent)

# parameters = 10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
# z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
# log_pdf_z0 = tf.random_normal((batch_size, 1), 0, 1, dtype=tf.float32)

# transform1 = transform_to_check(n_latent, parameters)
# z, log_pdf_z = transform1.transform(z0, log_pdf_z0)
# z0_inv, log_pdf_z0_inv = transform1.inverse_transform(z, log_pdf_z)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# parameters_np, z0_np, log_pdf_z0_np, z_np, log_pdf_z_np, z0_inv_np, log_pdf_z0_inv_np = sess.run([parameters, z0, log_pdf_z0, z, log_pdf_z, z0_inv, log_pdf_z0_inv])
# np.mean(np.abs(parameters_np))   
# np.max(np.abs(z0_np-z0_inv_np)) 
# np.max(np.abs(log_pdf_z0_np-log_pdf_z0_inv_np)) 

# pdb.set_trace()


# batch_size = 100
# n_latent = 20
# name = 'transform'
# transform_to_check = NonLinearIARFlow
# n_parameter = transform_to_check.required_num_parameters(n_latent)

# with tf.Graph().as_default():
#     with tf.variable_scope("Diverger", reuse=False):
#         parameters = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
#         parameters_2 = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
#         # parameters = 10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
#         # parameters = tf.Variable(tf.random_normal([1, n_parameter], 0, 1, dtype=tf.float32))

#     z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
#     log_pdf_z0 = tf.random_normal((batch_size, 1), 0, 1, dtype=tf.float32)

#     distrib = distributions.DiagonalGaussianDistribution(params = tf.zeros(shape=(batch_size, 2*n_latent)) )
#     sample = distrib.sample()

#     nonlinearIAF_transform1 = NonLinearIARFlow(parameters, n_latent)
#     # nonlinearIAF_transform2 = NonLinearIARFlow(parameters_2, n_latent)
#     # z1, log_pdf_z1, forw_mu1, forw_scale1 = nonlinearIAF_transform1.transform(z0, log_pdf_z0)
#     z, log_pdf_z, forw_mu, forw_scale = nonlinearIAF_transform1.transform(z1, log_pdf_z1)
#     z0_inv, log_pdf_z0_inv, back_mu, back_scale = nonlinearIAF_transform1.inverse_transform(z, log_pdf_z)

#     should_be_zero = z[:, :int(n_latent/2)]
#     should_be_normal = z[:, int(n_latent/2):]
#     should_be_zero_cost = tf.reduce_mean(tf.reduce_sum((should_be_zero**2), axis=1))
#     cost = should_be_zero_cost
#     # cost = tf.reduce_mean(tf.reduce_sum((z-z0)**2, axis=1))

#     div_vars = [v for v in tf.trainable_variables() if 'Diverger' in v.name]
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08)
#     div_step = optimizer.minimize(cost, var_list=div_vars)

#     init = tf.global_variables_initializer()
#     sess = tf.InteractiveSession()
#     sess.run(init)

# for i in range(0, 100000):
#     _, parameters_np, cost_np, forw_mu_np, forw_scale_np, sample_np = sess.run([div_step, parameters, cost, forw_mu, forw_scale, sample])
#     if i % 500 == 0:

#         print("Cost: ", cost_np)
#         # print("parameters: ", parameters_np)
#         # print("forw_mu: ", forw_mu_np[0,:])
#         # print("forw_scale: ", forw_scale_np[0,:])
#         print("sample: ", sample_np[0,:])


# batch_size = 1
# n_latent = 20
# name = 'transform'
# transform_to_check = NonLinearIARFlow
# n_parameter = transform_to_check.required_num_parameters(n_latent)

# parameters = 10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
# try: transform_object =  transform_to_check(parameters, n_latent)
# except: transform_object =  transform_to_check(None, n_latent)
# z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
# log_pdf_z0 = tf.random_normal((batch_size, 1), 0, 1, dtype=tf.float32)

# nonlinearIAF_transform1 = NonLinearIARFlow(parameters, n_latent)
# z, log_pdf_z, forw_mu, forw_scale = nonlinearIAF_transform1.transform(z0, log_pdf_z0)
# z0_inv, log_pdf_z0_inv, back_mu, back_scale = nonlinearIAF_transform1.inverse_transform(z, log_pdf_z)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# parameters_np, z0_np, log_pdf_z0_np, z_np, log_pdf_z_np, z0_inv_np, log_pdf_z0_inv_np, forw_mu_np, forw_scale_np, back_mu_np, back_scale_np = sess.run([parameters, z0, log_pdf_z0, z, log_pdf_z, z0_inv, log_pdf_z0_inv, forw_mu, forw_scale, back_mu, back_scale])
# # np.mean(np.abs(parameters_np))   
# # np.min(np.abs(z0_np-z0_inv_np)) 
# # np.min(np.abs(log_pdf_z0_np-log_pdf_z0_inv_np)) 



# dim = 3
# batch = 1000 
# k_start = 1

# params = tf.random_normal((batch, sum(list(range(max(2, k_start), dim+1)))), 0, 1, dtype=tf.float32)
# # params = tf.ones((batch, sum(list(range(max(2, k_start), dim+1)))))
# # params = None

# z0 = tf.tile(np.asarray([[1., 1., 1.]]).astype(np.float32), [batch, 1])
# log_pdf_z0 = tf.random_normal((batch, 1), 0, 1, dtype=tf.float32)

# rotation_transform1 = HouseholdRotationFlow(params, dim, k_start=k_start, init_reflection=1)
# rotation_transform2 = HouseholdRotationFlow(params, dim, k_start=k_start, init_reflection=-1)
# z1, log_pdf_z1 = rotation_transform1.transform(z0, log_pdf_z0)
# z2, log_pdf_z2 = rotation_transform2.transform(z0, log_pdf_z0)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# z1_np, log_pdf_z1_np, z2_np, log_pdf_z2_np = sess.run([z1, log_pdf_z1, z2, log_pdf_z2])
# helper.dataset_plotter([z1_np, z2_np], show_also=True)
# pdb.set_trace()



# # self.prior_map = f_p(n_latent | n_state, n_context). f_p(z_t | h<t, e(c_t))
# class TransformMap():
#     def __init__(self, config, name = '/TransformMap'):
#         self.name = name
#         self.config = config
#         self.constructed = False
 
#     def forward(self, transform_class_list, name = ''):
#         with tf.variable_scope("TransformMap", reuse=self.constructed):
#             input_dim = self.config['n_latent']
#             transforms_list = []
#             for transform_to_use in transform_class_list:  
#                 n_parameter = transform_to_use.required_num_parameters(input_dim)
#                 if n_parameter>0:
#                     parameters = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
#                 else: parameters = None
#                 transforms_list.append(transform_to_use(parameters, input_dim))
#             self.constructed = True
#             return transforms_list



# class ProjectiveFlow():
#     """
#     Projective Flow class.
#     Args:
#       parameters: parameters of transformation all appended.
#       input_dim : input dimensionality of the transformation. 
#     Raises:
#       ValueError: 
#     """
#     def __init__(self, parameters, input_dim, additional_dim=3, k_start=1, init_reflection=1, name='projective_transform'):   
#         self._parameter_scale = 1.
#         self._parameters = self._parameter_scale*parameters
#         self._input_dim = input_dim
#         self._additional_dim = additional_dim
#         self._k_start = k_start
#         self._init_reflection = init_reflection
#         self._manifold_nonlinearity = tf.nn.tanh

#     @property
#     def input_dim(self):
#         return self._input_dim

#     @property
#     def output_dim(self):
#         return self._input_dim+self._additional_dim

#     @staticmethod
#     def required_num_parameters(input_dim, additional_dim=3, k_start=1):  
#         if input_dim>=additional_dim: #row independent
#             if additional_dim == 1: n_basis_param = input_dim 
#             else: n_basis_param = sum(list(range(max(2, k_start), input_dim+1)))
#             return additional_dim+n_basis_param
#         else: #additional_dim > input_dim, column independent
#             if input_dim == 1: n_basis_param = additional_dim 
#             else: n_basis_param = sum(list(range(max(2, k_start), additional_dim+1)))
#             return input_dim+n_basis_param

#     def transform(self, z0, log_pdf_z0):
#         verify_size(z0, log_pdf_z0)
#         self._parameters.get_shape().assert_is_compatible_with([None, ProjectiveFlow.required_num_parameters(self._input_dim, self._additional_dim, self._k_start)])

#         if self._input_dim>=self._additional_dim: 
#             shear_param = tf.slice(self._parameters, [0, 0], [-1, self._additional_dim])
#             basis_param = tf.slice(self._parameters, [0, self._additional_dim], [-1, -1])
#         else: 
#             shear_param = tf.slice(self._parameters, [0, 0], [-1, self._input_dim])
#             basis_param = tf.slice(self._parameters, [0, self._input_dim], [-1, -1])

#         shear_matrix = tf.nn.softplus(shear_param)/(np.log(1+np.exp(0)))
#         if self._input_dim>=self._additional_dim: 
#             if self._additional_dim == 1: basis_tensor = (basis_param/tf.sqrt(tf.reduce_sum(basis_param**2, axis=[-1], keep_dims=True)))[:,np.newaxis,:]
#             else: basis_tensor = helper.householder_rotations_tf(n=self.input_dim, batch=tf.shape(basis_param)[0], k_start=self._k_start, 
#                                                                  init_reflection=self._init_reflection, params=basis_param)[:,-self._additional_dim:,:]
#         else: 
#             if self._input_dim == 1: basis_tensor = (basis_param/tf.sqrt(tf.reduce_sum(basis_param**2, axis=[-1], keep_dims=True)))[:,:,np.newaxis]
#             else: basis_tensor = helper.householder_rotations_tf(n=self._additional_dim, batch=tf.shape(basis_param)[0], k_start=self._k_start, 
#                                                                  init_reflection=self._init_reflection, params=basis_param)[:,:,-self.input_dim:]
#         # Transformation
#         if self._input_dim>=self._additional_dim: z_project_input = z0
#         else: z_project_input = shear_matrix*z0

#         if basis_tensor.get_shape()[0].value == 1: #one set of parameters
#             z_project = tf.matmul(z_project_input, basis_tensor[0, :, :], transpose_a=False, transpose_b=True)
#         else: # batched parameters
#             z_project = tf.matmul(basis_tensor, z_project_input[:,:,np.newaxis])[:, :, 0]
        
#         if self._input_dim>=self._additional_dim: z_project_sheared = tf.concat([z0, shear_matrix*z_project], axis=1)  
#         else: z_project_sheared = tf.concat([z0, z_project], axis=1)           
#         z = self._manifold_nonlinearity(z_project_sheared)

#         # Density Update
#         if self._manifold_nonlinearity == tf.nn.tanh:
#             diagonal_nonlinearity_jacobian = 1-z**2
#         else: pdb.set_trace()

#         diagonal_nonlinearity_jacobian_1 = tf.slice(diagonal_nonlinearity_jacobian, [0, 0], [-1, self._input_dim])
#         diagonal_nonlinearity_jacobian_2 = tf.slice(diagonal_nonlinearity_jacobian, [0, self._input_dim], [-1, -1])
#         pdb.set_trace()

        
#         log_abs_det_jacobian = 0
#         log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
#         return z, log_pdf_z

