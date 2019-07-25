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

#####################################################################################
######################### Non-analytical Inverse Flows ##############################
#####################################################################################

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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        affine = tf.reduce_sum(z0*self._w, axis=[1], keep_dims=True) + self._b
        h = tf.tanh(affine)
        z = z0+self._u_tilde*h

        if log_pdf_z0 is not None: 
            h_prime_w = (1-tf.pow(h, 2))*self._w
            log_abs_det_jacobian = tf.log(1e-7+tf.abs(1+tf.reduce_sum(h_prime_w*self._u_tilde, axis=[1], keep_dims=True)))
            log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        else: log_pdf_z = None
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        print('No analytical inverse for planar flows.')
        quit()

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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        z_diff = z0 - self._z_ref
        r = tf.sqrt(tf.reduce_sum(tf.square(z_diff), axis=[1], keep_dims=True))
        h = 1/(self._alpha_tilde + r)
        scale = self._beta_tilde * h
        z = z0 + scale * z_diff

        if log_pdf_z0 is not None: 
            log_abs_det_jacobian = tf.log(1e-7+tf.abs(tf.pow(1 + scale, self._input_dim - 1) * (1 + scale * (1 - h * r))))
            log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        else: log_pdf_z = None
        return z, log_pdf_z
    
    def inverse_transform(self, z, log_pdf_z):
        print('No analytical inverse for radial flows.')
        quit()

#####################################################################################
############################## Dimension Shuffle Flows ##############################
#####################################################################################

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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        z = tf.reverse(z0, axis=[-1,])
        
        if log_pdf_z0 is not None: log_pdf_z = log_pdf_z0
        else: log_pdf_z = None
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        z0 = tf.reverse(z, axis=[-1,])

        if log_pdf_z is not None: log_pdf_z0 = log_pdf_z
        else: log_pdf_z0 = None
        return z0, log_pdf_z0

class CircularSlideDimensionFlow():
    """
    Circular Slide Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, input_dim, parameters=None, slide_left=True, name='circular_slide_dimension_transform'):   
        self._input_dim = input_dim
        self._slide_left = slide_left
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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        if self._slide_left:
            z = tf.concat([tf.slice(z0, [0, self._n_slide_dims], [-1, self._input_dim-self._n_slide_dims]), tf.slice(z0, [0, 0], [-1, self._n_slide_dims])], axis=1)
        else:
            z = tf.concat([tf.slice(z0, [0, self._input_dim-self._n_slide_dims], [-1, self._n_slide_dims]), tf.slice(z0, [0, 0], [-1, self._input_dim-self._n_slide_dims])], axis=1)
        
        if log_pdf_z0 is not None: log_pdf_z = log_pdf_z0
        else: log_pdf_z = None
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        if self._slide_left:
            z0 = tf.concat([tf.slice(z, [0, self._input_dim-self._n_slide_dims], [-1, self._n_slide_dims]), tf.slice(z, [0, 0], [-1, self._input_dim-self._n_slide_dims])], axis=1)
        else:
            z0 = tf.concat([tf.slice(z, [0, self._n_slide_dims], [-1, self._input_dim-self._n_slide_dims]), tf.slice(z, [0, 0], [-1, self._n_slide_dims])], axis=1)
        
        if log_pdf_z is not None: log_pdf_z0 = log_pdf_z
        else: log_pdf_z0 = None
        return z0, log_pdf_z0

#####################################################################################
############################## Scale and Interval Flows #############################
#####################################################################################

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
        assert (self._scale >= 1e-7)

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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        z = z0*self._scale
        
        if log_pdf_z0 is not None: 
            log_abs_det_jacobian = self._input_dim*tf.log(self._scale)
            log_pdf_z = log_pdf_z0-log_abs_det_jacobian
        else: log_pdf_z = None
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        z0 = z/self._scale
        
        if log_pdf_z is not None: 
            log_abs_det_jacobian = self._input_dim*tf.log(self._scale)
            log_pdf_z = log_pdf_z0+log_abs_det_jacobian
        else: log_pdf_z0 = None
        return z0, log_pdf_z0

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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        if self._zero_one: z = tf.nn.sigmoid(5.*z0)
        else: z = tf.nn.tanh(2.5*z0)
        
        if log_pdf_z0 is not None: 
            if self._zero_one:
                log_abs_det_jacobian = tf.reduce_sum(np.log(5.)+(tf.log(1e-7+z)+tf.log(1e-7+(1-z))), axis=[1], keep_dims=True)
            else:
                log_abs_det_jacobian = tf.reduce_sum(np.log(2.5)+tf.log(1e-7+(1-z))+tf.log(1e-7+(1+z)), axis=[1], keep_dims=True)
            log_pdf_z = log_pdf_z0-log_abs_det_jacobian
        else: log_pdf_z = None
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        if self._zero_one: z0 = (tf.log(z)-tf.log(1-z))/5.
        else: z0 = (0.5*(tf.log(1+z)-tf.log(1-z)))/2.5
        
        if log_pdf_z is not None: 
            if self._zero_one:
                log_abs_det_jacobian = tf.reduce_sum(np.log(5.)+tf.log(1e-7+z)+tf.log(1e-7+(1-z)), axis=[1], keep_dims=True)
            else:        
                log_abs_det_jacobian = tf.reduce_sum(np.log(2.5)+tf.log(1e-7+(1-z))+tf.log(1e-7+(1+z)), axis=[1], keep_dims=True)
            log_pdf_z0 = log_pdf_z+log_abs_det_jacobian
        else: log_pdf_z0 = None
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
    def __init__(self, input_dim, parameters=None, name='inverse_open_interval_dimension_transform'):  #real
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
        return self._inverse_flow_object.inverse_transform(z0, log_pdf_z0)

    def inverse_transform(self, z, log_pdf_z):
        return self._inverse_flow_object.transform(z, log_pdf_z)


#####################################################################################
##################################  Rotation Flows ##################################
#####################################################################################

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
        return tf.Variable(tf.constant(helper.random_rot_mat(self._input_dim, mode='SO(n)'), dtype=tf.float32), trainable=False)[np.newaxis, :, :]

    def transform(self, z0, log_pdf_z0):
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        z = tf.matmul(z0, self._batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=True)
        
        if log_pdf_z0 is not None: log_pdf_z = log_pdf_z0 
        else: log_pdf_z = None
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)
        
        z0 = tf.matmul(z, self._batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=False)

        if log_pdf_z is not None: log_pdf_z0 = log_pdf_z 
        else: log_pdf_z0 = None
        return z0, log_pdf_z0

class NotManyReflectionsRotationFlow():
    """
    Many Householder Reflections Rotation Flow class. SO(n) 
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    n_steps = 4

    def __init__(self, input_dim, parameters, name='not_many_reflections_rotation_transform'):   
        self._parameter_scale = 1.
        self._parameters = parameters
        if self._parameters is not None: self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim
        assert (NotManyReflectionsRotationFlow.n_steps % 2 == 0) # Required for SO(n)

        self._parameters.get_shape().assert_is_compatible_with([None, NotManyReflectionsRotationFlow.required_num_parameters(self._input_dim)])

        param_index = 0
        self._householder_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim*NotManyReflectionsRotationFlow.n_steps) 
        self._householder_vec = tf.reshape(self._householder_vec, [-1, NotManyReflectionsRotationFlow.n_steps, self._input_dim])
        self._householder_vec_dir = self._householder_vec/helper.safe_tf_sqrt(tf.reduce_sum(self._householder_vec**2, axis=2, keep_dims=True))

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim): 
        return NotManyReflectionsRotationFlow.n_steps*input_dim

    def get_batched_rot_matrix(self):
        identity_mat = tf.constant(np.eye(self._input_dim), tf.float32)[np.newaxis,:,:]
        overall_rot_mat = None
        for i in range(self._householder_vec_dir.get_shape()[1].value):
            curr_dir = self._householder_vec_dir[:,i,:]
            curr_rot_mat = identity_mat-2*tf.matmul(curr_dir[:,:,np.newaxis], curr_dir[:,np.newaxis, :], transpose_a=False, transpose_b=False)
            if overall_rot_mat is None: overall_rot_mat = curr_rot_mat
            else: overall_rot_mat = tf.matmul(curr_rot_mat, overall_rot_mat, transpose_a=False, transpose_b=False)
        return overall_rot_mat

    def transform(self, z0, log_pdf_z0):
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        curr_z = z0
        for i in range(self._householder_vec_dir.get_shape()[1].value):
            curr_dir = self._householder_vec_dir[:,i,:]
            curr_dot_product = tf.reduce_sum(curr_z*curr_dir, axis=1, keep_dims=True)
            curr_z = curr_z-2*curr_dot_product*curr_dir
        z = curr_z

        if log_pdf_z0 is not None: log_pdf_z = log_pdf_z0 
        else: log_pdf_z = None
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        curr_z = z
        for i in range(self._householder_vec_dir.get_shape()[1].value):
            curr_dir = self._householder_vec_dir[:,self._householder_vec_dir.get_shape()[1].value-i-1,:]
            curr_dot_product = tf.reduce_sum(curr_z*curr_dir, axis=1, keep_dims=True)
            curr_z = curr_z-2*curr_dot_product*curr_dir
        z0 = curr_z

        if log_pdf_z is not None: log_pdf_z0 = log_pdf_z 
        else: log_pdf_z0 = None
        return z0, log_pdf_z0

class ManyReflectionsRotationFlow():
    """
    Many Householder Reflections Rotation Flow class. SO(n) 
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    n_steps = 100

    def __init__(self, input_dim, parameters, name='many_reflections_rotation_transform'):   
        self._parameter_scale = 1.
        self._parameters = parameters
        if self._parameters is not None: self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim
        assert (ManyReflectionsRotationFlow.n_steps % 2 == 0) # Required for SO(n)

        self._parameters.get_shape().assert_is_compatible_with([None, ManyReflectionsRotationFlow.required_num_parameters(self._input_dim)])

        param_index = 0
        self._householder_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim*ManyReflectionsRotationFlow.n_steps) 
        self._householder_vec = tf.reshape(self._householder_vec, [-1, ManyReflectionsRotationFlow.n_steps, self._input_dim])
        self._householder_vec_dir = self._householder_vec/helper.safe_tf_sqrt(tf.reduce_sum(self._householder_vec**2, axis=2, keep_dims=True))

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim): 
        return ManyReflectionsRotationFlow.n_steps*input_dim

    def get_batched_rot_matrix(self):
        identity_mat = tf.constant(np.eye(self._input_dim), tf.float32)[np.newaxis,:,:]
        overall_rot_mat = None
        for i in range(self._householder_vec_dir.get_shape()[1].value):
            curr_dir = self._householder_vec_dir[:,i,:]
            curr_rot_mat = identity_mat-2*tf.matmul(curr_dir[:,:,np.newaxis], curr_dir[:,np.newaxis, :], transpose_a=False, transpose_b=False)
            if overall_rot_mat is None: overall_rot_mat = curr_rot_mat
            else: overall_rot_mat = tf.matmul(curr_rot_mat, overall_rot_mat, transpose_a=False, transpose_b=False)
        return overall_rot_mat

    def transform(self, z0, log_pdf_z0):
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        curr_z = z0
        for i in range(self._householder_vec_dir.get_shape()[1].value):
            curr_dir = self._householder_vec_dir[:,i,:]
            curr_dot_product = tf.reduce_sum(curr_z*curr_dir, axis=1, keep_dims=True)
            curr_z = curr_z-2*curr_dot_product*curr_dir
        z = curr_z

        if log_pdf_z0 is not None: log_pdf_z = log_pdf_z0 
        else: log_pdf_z = None
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        curr_z = z
        for i in range(self._householder_vec_dir.get_shape()[1].value):
            curr_dir = self._householder_vec_dir[:,self._householder_vec_dir.get_shape()[1].value-i-1,:]
            curr_dot_product = tf.reduce_sum(curr_z*curr_dir, axis=1, keep_dims=True)
            curr_z = curr_z-2*curr_dot_product*curr_dir
        z0 = curr_z

        if log_pdf_z is not None: log_pdf_z0 = log_pdf_z 
        else: log_pdf_z0 = None
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
        k_start = max(input_dim-HouseholdRotationFlow.max_steps+1, 1)
        return sum(list(range(max(2, k_start), input_dim+1)))
    
    def get_batched_rot_matrix(self):
        return helper.householder_rotations_tf(n=self.input_dim, k_start=self._k_start, init_reflection=self._init_reflection, params=self._parameters) 

    def get_list_batched_householder_vectors(self):
        return helper.householder_rotation_vectors_tf(n=self.input_dim, k_start=self._k_start, init_reflection=self._init_reflection, params=self._parameters) 
        
    def transform(self, z0, log_pdf_z0):
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        if self._mode == 'matrix':
            if self._parameters is None or self._parameters.get_shape()[0].value == 1: #one set of parameters
                z = tf.matmul(z0, self._batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=True)
            else: # batched parameters
                z = tf.matmul(self._batched_rot_matrix, z0[:,:,np.newaxis], transpose_a=False, transpose_b=False)[:, :, 0]

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

        if log_pdf_z0 is not None: log_pdf_z = log_pdf_z0 
        else: log_pdf_z = None
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)
        
        if self._mode == 'matrix':
            if self._parameters is None or self._parameters.get_shape()[0].value == 1: #one set of parameters
                z0 = tf.matmul(z, self._batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=False)
            else: # batched parameters
                z0 = tf.matmul(self._batched_rot_matrix, z[:,:,np.newaxis],  transpose_a=True, transpose_b=False)[:, :, 0]
            
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

        if log_pdf_z is not None: log_pdf_z0 = log_pdf_z 
        else: log_pdf_z0 = None
        return z0, log_pdf_z0

class CompoundRotationFlow():
    """
    Compound Rotation Flow class. SO(n)
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    H_class = ManyReflectionsRotationFlow
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
                curr_householder_param, param_index = helper.slice_parameters(self._parameters, param_index, CompoundRotationFlow.H_class.required_num_parameters(self._input_dim))
                self._householder_flows_list.append(CompoundRotationFlow.H_class(self._input_dim, curr_householder_param))
            elif CompoundRotationFlow.compound_structure[i] == 'P':
                pdb.set_trace()
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
                n_parameters += CompoundRotationFlow.H_class.required_num_parameters(input_dim)
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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        c_index, h_index, p_index = 0, 0, 0
        curr_z = z0
        for i in range(len(CompoundRotationFlow.compound_structure)):
            # print('c_index, h_index, p_index ', c_index, h_index, p_index)
            if CompoundRotationFlow.compound_structure[i] == 'C':
                curr_z, _ = tf.matmul(curr_z, self._constant_rot_mats_list[c_index], transpose_a=False, transpose_b=True), None
                c_index += 1
            elif CompoundRotationFlow.compound_structure[i] == 'H':
                curr_z, _= self._householder_flows_list[h_index].transform(curr_z, None)
                h_index += 1
            elif CompoundRotationFlow.compound_structure[i] == 'P':
                curr_z, _ = self._specific_order_dimension_flows_list[p_index].transform(curr_z, None)
                p_index += 1
        assert ((c_index+h_index+p_index) == len(CompoundRotationFlow.compound_structure))
        z = curr_z

        if log_pdf_z0 is not None: log_pdf_z = log_pdf_z0 
        else: log_pdf_z = None
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        c_index = np.sum((np.asarray(CompoundRotationFlow.compound_structure) == 'C'))
        h_index = np.sum((np.asarray(CompoundRotationFlow.compound_structure) == 'H'))
        p_index = np.sum((np.asarray(CompoundRotationFlow.compound_structure) == 'P'))
        curr_z = z
        for i in range(len(CompoundRotationFlow.compound_structure)-1, -1, -1):
            # print('c_index, h_index, p_index ', c_index, h_index, p_index)
            if CompoundRotationFlow.compound_structure[i] == 'C':
                c_index -= 1
                curr_z, _ = tf.matmul(curr_z, self._constant_rot_mats_list[c_index], transpose_a=False, transpose_b=False), None
            elif CompoundRotationFlow.compound_structure[i] == 'H':
                h_index -= 1
                curr_z, _ = self._householder_flows_list[h_index].inverse_transform(curr_z, None)
            elif CompoundRotationFlow.compound_structure[i] == 'P':
                p_index -= 1
                curr_z, _ = self._specific_order_dimension_flows_list[p_index].inverse_transform(curr_z, None)
            assert (c_index >= 0 and h_index >= 0 and p_index >= 0)
        z0 = curr_z

        if log_pdf_z is not None: log_pdf_z0 = log_pdf_z 
        else: log_pdf_z0 = None
        return z0, log_pdf_z0

#####################################################################################
################################  Orthogonal Maps ###################################
#####################################################################################

class PiecewisePlanarScalingMap():
    """
    Connected Piecewise Scaling Map class with Jacobians specified as scaled multiples of diagonal matrices.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    n_steps = 50

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
            self._pos_scale = tf.clip_by_value(tf.nn.softplus(self._pos_pre_scale)/np.log(1+np.exp(0)), 1e-7, np.inf)  
            self._neg_scale = tf.clip_by_value(tf.nn.softplus(self._neg_pre_scale)/np.log(1+np.exp(0)), 1e-7, np.inf)  
        elif self._scale_mode == 'BoundedScale': 
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            self._pos_scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(self._pos_pre_scale+scipy.special.logit(1/gap))*gap, 1e-7, np.inf)  
            self._neg_scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(self._neg_pre_scale+scipy.special.logit(1/gap))*gap, 1e-7, np.inf)  
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

class ConnectedPiecewiseOrthogonalMap():
    """
    Connected Piecewise Orthogonal Map class with Jacobians specified as scaled multiples of orthogonal matrices.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    rotation_flow_class = NotManyReflectionsRotationFlow

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
            self._pos_scale = tf.clip_by_value(tf.nn.softplus(self._pos_pre_scale)/np.log(1+np.exp(0)), 1e-7, np.inf)  
            self._neg_scale = tf.clip_by_value(tf.nn.softplus(self._neg_pre_scale)/np.log(1+np.exp(0)), 1e-7, np.inf)  
        elif self._scale_mode == 'BoundedScale': 
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            self._pos_scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(self._pos_pre_scale+scipy.special.logit(1/gap))*gap, 1e-7, np.inf)  
            self._neg_scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(self._neg_pre_scale+scipy.special.logit(1/gap))*gap, 1e-7, np.inf)  
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
        
        z_pos_rot, _ = self._pos_rotation_flow.transform(z_centered, None)
        z_neg_rot, _ = self._neg_rotation_flow.transform(z_centered, None)
        z_pos_scale_rot = self._pos_scale*z_pos_rot
        z_neg_scale_rot = self._neg_scale*z_neg_rot

        z_scale_rot = pos_mask*z_pos_scale_rot+neg_mask*z_neg_scale_rot
        z = z_scale_rot+self._hyper_vec_dir*self._hyper_bias+self._output_shift_vec
        scales = pos_mask*self._pos_scale+neg_mask*self._neg_scale 
        log_scales = tf.log(scales)
        return z, log_scales

class CompoundPiecewiseOrthogonalMap():
    """
    Compound Piecewise Orthogonal Map class with Jacobians specified as scaled multiples of orthogonal matrices.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    compound_structure = ['S', 'R']

    def __init__(self, input_dim, parameters, name='compound_piecewise_orthogonal_map'):  
        self._parameter_scale = 1.
        self._parameters = parameters
        if self._parameters is not None: self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim
        
        if self._parameters is not None:
            self._parameters.get_shape().assert_is_compatible_with([None, CompoundPiecewiseOrthogonalMap.required_num_parameters(self._input_dim)])
        
        param_index = 0
        self._planar_scale_maps_list, self._connected_piecewise_maps_list = [], []
        for i in range(len(CompoundPiecewiseOrthogonalMap.compound_structure)):
            if CompoundPiecewiseOrthogonalMap.compound_structure[i] == 'S':
                curr_householder_param, param_index = helper.slice_parameters(self._parameters, param_index, PiecewisePlanarScalingMap.required_num_parameters(self._input_dim))
                self._planar_scale_maps_list.append(PiecewisePlanarScalingMap(self._input_dim, curr_householder_param))
            elif CompoundPiecewiseOrthogonalMap.compound_structure[i] == 'R':
                curr_connected_piecewise_param, param_index = helper.slice_parameters(self._parameters, param_index, ConnectedPiecewiseOrthogonalMap.required_num_parameters(self._input_dim))
                self._connected_piecewise_maps_list.append(ConnectedPiecewiseOrthogonalMap(self._input_dim, curr_connected_piecewise_param))
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
        for i in range(len(CompoundPiecewiseOrthogonalMap.compound_structure)):
            if CompoundPiecewiseOrthogonalMap.compound_structure[i] == 'S':
                n_parameters += PiecewisePlanarScalingMap.required_num_parameters(input_dim)
            elif CompoundPiecewiseOrthogonalMap.compound_structure[i] == 'R':
                n_parameters += ConnectedPiecewiseOrthogonalMap.required_num_parameters(input_dim)
        return n_parameters 
    
    def jacobian(self, z0):
        s_index, r_index = 0, 0
        curr_z, jacobian = z0, None
        for i in range(len(CompoundPiecewiseOrthogonalMap.compound_structure)):
            if CompoundPiecewiseOrthogonalMap.compound_structure[i] == 'S':
                curr_jacobian = self._planar_scale_maps_list[s_index].jacobian(curr_z)
                if jacobian is None: jacobian = curr_jacobian
                else: jacobian = tf.matmul(curr_jacobian, jacobian, transpose_a=False, transpose_b=False)
                curr_z, _ = self._planar_scale_maps_list[s_index].transform(curr_z)
                s_index += 1
            elif CompoundPiecewiseOrthogonalMap.compound_structure[i] == 'R':
                curr_jacobian = self._connected_piecewise_maps_list[r_index].jacobian(curr_z)
                if jacobian is None: jacobian = curr_jacobian
                else: jacobian = tf.matmul(curr_jacobian, jacobian, transpose_a=False, transpose_b=False)
                curr_z, _ = self._connected_piecewise_maps_list[r_index].transform(curr_z)
                r_index += 1
        assert ((s_index+r_index) == len(CompoundPiecewiseOrthogonalMap.compound_structure))
        
        return jacobian

    def transform(self, z0):
        s_index, r_index = 0, 0
        curr_z, log_scales = z0, 0
        for i in range(len(CompoundPiecewiseOrthogonalMap.compound_structure)):
            if CompoundPiecewiseOrthogonalMap.compound_structure[i] == 'S':
                curr_z, curr_log_scales = self._planar_scale_maps_list[s_index].transform(curr_z)
                log_scales += curr_log_scales
                s_index += 1
            elif CompoundPiecewiseOrthogonalMap.compound_structure[i] == 'R':
                curr_z, curr_log_scales = self._connected_piecewise_maps_list[r_index].transform(curr_z)
                log_scales += curr_log_scales
                r_index += 1
        assert ((s_index+r_index) == len(CompoundPiecewiseOrthogonalMap.compound_structure))
        z = curr_z

        return z, log_scales

#####################################################################################
###########################  Riemannian Maps and Flows ##############################
#####################################################################################

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
    rotation_flow_class = CompoundRotationFlow
    
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

class RiemannianFlow():
    """
    Projective Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """

    NOM_class = CompoundPiecewiseOrthogonalMap
    
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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

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
        
        if log_pdf_z0 is not None: 
            if len(input_NOM_log_scales) > 0: input_NOM_log_scales_sum = tf.add_n(input_NOM_log_scales)
            else: input_NOM_log_scales_sum = tf.zeros((tf.shape(z0)[0], 1), tf.float32)
            if len(additional_NOM_log_scales) > 0: additional_NOM_log_scales_sum = tf.add_n(additional_NOM_log_scales)
            else: additional_NOM_log_scales_sum = tf.zeros((tf.shape(z0)[0], 1), tf.float32)
            overall_scales = tf.exp(input_NOM_log_scales_sum+additional_NOM_log_scales_sum)

            delta_log_pdf_z = -tf.log(1+overall_scales**2)*(min(self._additional_dim, self._input_dim)/2)
            log_pdf_z = log_pdf_z0 + delta_log_pdf_z
        else: log_pdf_z = None

        if mode == 'regular': return z, log_pdf_z
        else: return z, log_pdf_z, overall_scales

    def inverse_transform(self, z, log_pdf_z, mode='regular'):
        print('No analytical inverse for Riemannian flows.')
        quit()

#####################################################################################
#############################  Invertible Euclidean Flows ###########################
#####################################################################################

class Affine2DFlow():
    """
    A affine transformation in 2D.

    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    allowed = ['Rotation']
    # allowed = ['Rotation', 'Translation']
    # allowed = ['Rotation', 'Scale', 'Translation']
    # allowed = ['Rotation', 'Scale', 'Shear', 'Translation']

    def __init__(self, input_dim, parameters, mode='Bounded', name='affine_2D_transform'):  
        self._parameter_scale = 1.
        self._parameters = parameters
        if self._parameters is not None: self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim
        self._mode = mode
        self._shear_mode = 'first'
        assert (input_dim == 2)

        if self._parameters is not None:
            self._parameters.get_shape().assert_is_compatible_with([None, Affine2DFlow.required_num_parameters(self._input_dim)])

        param_index = 0
        self._affine_matrix = tf.eye(2)[np.newaxis, :, :]
        if 'Rotation' in Affine2DFlow.allowed:
            self._radian_angle_param, param_index = helper.slice_parameters(self._parameters, param_index, 1)
            self._sin = tf.math.sin(self._radian_angle_param)
            self._cos = tf.math.cos(self._radian_angle_param)
            self._rot_mat = tf.concat([tf.concat([self._cos[:,:,np.newaxis], -self._sin[:,:,np.newaxis]], axis=2), 
                                 tf.concat([self._sin[:,:,np.newaxis],  self._cos[:,:,np.newaxis]], axis=2)], axis=1) 
            self._affine_matrix = tf.matmul(self._affine_matrix, self._rot_mat)
        if 'Scale' in Affine2DFlow.allowed:
            self._scale_param_raw, param_index = helper.slice_parameters(self._parameters, param_index, 2)
            self._scale_param = tf.clip_by_value(tf.nn.softplus(self._scale_param_raw), 1e-7, np.inf)  
            self._scale_mat = tf.matmul(tf.eye(2)[np.newaxis, :, :], tf.tile(self._scale_param[:,np.newaxis,:], [1,2,1]))
            self._affine_matrix = tf.matmul(self._affine_matrix, self._scale_mat)
        if 'Shear' in Affine2DFlow.allowed: 
            self._shear_param, param_index = helper.slice_parameters(self._parameters, param_index, 1)
            if self._shear_mode == 'first':
                temp = tf.concat([tf.tile(tf.constant([1,], tf.float32)[np.newaxis,:], [tf.shape(self._shear_param)[0], 1]), self._shear_param], axis=1)
                self._shear_mat = tf.concat([temp[:, np.newaxis, :], tf.tile(tf.constant([0,1], tf.float32)[np.newaxis,:], [tf.shape(self._shear_param)[0], 1])[:, np.newaxis, :]], axis=1)
            else:
                temp = tf.concat([self._shear_param, tf.tile(tf.constant([1,], tf.float32)[np.newaxis,:], [tf.shape(self._shear_param)[0], 1])], axis=1)
                self._shear_mat = tf.concat([tf.tile(tf.constant([1,0], tf.float32)[np.newaxis,:], [tf.shape(self._shear_param)[0], 1])[:, np.newaxis, :], temp[:, np.newaxis, :]], axis=1)
            self._affine_matrix = tf.matmul(self._affine_matrix, self._shear_mat)
        if 'Translation' in Affine2DFlow.allowed:
            self._translation_param, param_index = helper.slice_parameters(self._parameters, param_index, 2)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim): 
        n_parameters = 0 
        if 'Rotation' in Affine2DFlow.allowed: n_parameters = n_parameters+1
        if 'Scale' in Affine2DFlow.allowed: n_parameters = n_parameters+2
        if 'Shear' in Affine2DFlow.allowed: n_parameters = n_parameters+1
        if 'Translation' in Affine2DFlow.allowed: n_parameters = n_parameters+2
        return n_parameters
    
    def transform(self, z0, log_pdf_z0):
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        z_curr = z0
        if 'Rotation' in Affine2DFlow.allowed:
            z_curr = tf.matmul(self._rot_mat, z_curr[:,:,np.newaxis], transpose_a=False, transpose_b=False)[:,:,0]
        if 'Scale' in Affine2DFlow.allowed:
            z_curr = self._scale_param*z_curr
        if 'Shear' in Affine2DFlow.allowed:
            if self._shear_mode == 'first':
                z_new_y = z_curr[:,1,np.newaxis]
                z_new_x = z_curr[:,0,np.newaxis]+self._shear_param*z_new_y
            else:
                z_new_x = z_curr[:,0,np.newaxis]
                z_new_y = z_curr[:,1,np.newaxis]+self._shear_param*z_new_x
            z_curr = tf.concat([z_new_x, z_new_y], axis=1)
        if 'Translation' in Affine2DFlow.allowed:
            z_curr = z_curr+self._translation_param
        z = z_curr 

        if 'Scale' in Affine2DFlow.allowed and log_pdf_z0 is not None: 
            log_abs_det_jacobian = tf.reduce_sum(tf.log(self._scale_param), axis=[1], keep_dims=True)

        if log_pdf_z0 is not None: log_pdf_z = log_pdf_z0-log_abs_det_jacobian
        else: log_pdf_z = log_pdf_z0
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        z_curr = z
        if 'Translation' in Affine2DFlow.allowed:
            z_curr = z_curr-self._translation_param
        if 'Shear' in Affine2DFlow.allowed:
            if self._shear_mode == 'first':
                z_new_y = z_curr[:,1,np.newaxis]
                z_new_x = z_curr[:,0,np.newaxis]-self._shear_param*z_new_y
            else:
                z_new_x = z_curr[:,0,np.newaxis]
                z_new_y = z_curr[:,1,np.newaxis]-self._shear_param*z_new_x
            z_curr = tf.concat([z_new_x, z_new_y], axis=1)
        if 'Scale' in Affine2DFlow.allowed:
            z_curr = z_curr/self._scale_param
        if 'Rotation' in Affine2DFlow.allowed:
            z_curr = tf.matmul(self._rot_mat, z_curr[:,:,np.newaxis], transpose_a=True, transpose_b=False)[:,:,0]

        z0 = z_curr

        if 'Scale' in Affine2DFlow.allowed and log_pdf_z is not None: 
            log_abs_det_jacobian = -tf.reduce_sum(tf.log(self._scale_param), axis=[1], keep_dims=True)

        if log_pdf_z is not None: log_pdf_z0 = log_pdf_z-log_abs_det_jacobian
        else: log_pdf_z0 = log_pdf_z
        return z0, log_pdf_z0

class ProperIsometricFlow():
    """
    A proper rigid transformation (also called Euclidean transformation or 
    Euclidean isometry) that is made of rotations(SO(n))+translations. 
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    rotation_class = NotManyReflectionsRotationFlow

    def __init__(self, input_dim, parameters, mode='Bounded', name='proper_isometric_transform'):  
        self._parameter_scale = 1.
        self._parameters = parameters
        if self._parameters is not None: self._parameters = self._parameter_scale*self._parameters
        self._input_dim = input_dim
        self._mode = mode
        self._max_bounded_translation = 0.02
        self._min_bounded_translation = -0.02

        if self._parameters is not None:
            self._parameters.get_shape().assert_is_compatible_with([None, ProperIsometricFlow.required_num_parameters(self._input_dim)])

        param_index = 0
        self._translation_param_raw, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim)
        self._rotation_param, param_index = helper.slice_parameters(self._parameters, param_index, ProperIsometricFlow.rotation_class.required_num_parameters(self._input_dim))
        self._rotation_flow = ProperIsometricFlow.rotation_class(input_dim=self._input_dim, parameters=self._rotation_param) 
        self._translation_param = self._translation_param_raw 
        
    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim): 
        return input_dim+ProperIsometricFlow.rotation_class.required_num_parameters(input_dim) 
    
    def transform(self, z0, log_pdf_z0):
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        z_rot, log_pdf_z = self._rotation_flow.transform(z0, log_pdf_z0)
        if self._mode == 'Regular':
            z = z_rot+self._translation_param
        elif self._mode == 'Bounded':
            gap = (self._max_bounded_translation-self._min_bounded_translation)
            translation = self._min_bounded_translation+tf.nn.sigmoid(self._translation_param)*gap
            z = z_rot+translation
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        if self._mode == 'Regular':
            z_rot = z-self._translation_param
        elif self._mode == 'Bounded':
            gap = (self._max_bounded_translation-self._min_bounded_translation)
            translation = self._min_bounded_translation+tf.nn.sigmoid(self._translation_param)*gap
            z_rot = z-translation

        z0, log_pdf_z0 = self._rotation_flow.inverse_transform(z_rot, log_pdf_z)
        return z0, log_pdf_z0

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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)
        
        z, log_scales = self._piecewise_planar_scaling_map.transform(z0)
        if log_pdf_z0 is not None: log_pdf_z = log_pdf_z0-self._input_dim*log_scales
        else: log_pdf_z = None
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        z0, log_scales = self._piecewise_planar_scaling_map.inverse_transform(z)
        if log_pdf_z is not None: log_pdf_z0 = log_pdf_z-self._input_dim*log_scales
        else: log_pdf_z0 = None
        return z0, log_pdf_z0

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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

        if self._parameters.get_shape()[0].value == 1: #one set of parameters
            z = tf.matmul(z0, self._cho[0, :, :], transpose_a=False, transpose_b=True)
        else: # batched parameters
            z = tf.matmul(self._cho, z0[:,:,np.newaxis])[:, :, 0]
        log_pdf_z = log_pdf_z0 
        return z, log_pdf_z

    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        z0 = None
        for i in range(self._input_dim):
            if i == 0:
                z0_new = z[:, np.newaxis, i]/self._cho[:, i, i, np.newaxis]
            else:
                subs_dot_prod = tf.reduce_sum(z0*self._cho[:, i, 0:i], axis=-1, keep_dims=True)
                z0_new = (z[:, np.newaxis, i]-subs_dot_prod)/self._cho[:, i, i, np.newaxis]
            if z0 is None: z0 = z0_new
            else: z0 = tf.concat([z0, z0_new], axis=-1)

        log_pdf_z0 = log_pdf_z 
        return z0, log_pdf_z0

class NonLinearIARFlow():
    """
    Non-Linear Inverse Autoregressive Flow class.

    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    # layer_expansions = [5,5,5] ## dont make it wider, make it deeper [2,2,2] instead of [5,5] for speed
    layer_expansions = [50,50,50] ## dont make it wider, make it deeper [2,2,2] instead of [5,5] for speed
    # layer_expansions = [10,10,10] ## dont make it wider, make it deeper [2,2,2] instead of [5,5] for speed

    def __init__(self, input_dim, parameters, mode='ScaleShift', name='nonlinearIAR_transform'):   #real
        self._parameter_scale = 1
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim
        self._nonlinearity = helper.LeakyReLU 
        # self._nonlinearity = tf.nn.tanh
        self._mode = mode
        self._max_bounded_scale = 1.1
        self._min_bounded_scale = 0.9
        # self._min_bounded_scale = 1/self._max_bounded_scale
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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)

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
            scale = tf.clip_by_value(tf.nn.sigmoid(pre_scale), 1e-7, np.inf)  
            z = scale*z0+(1-scale)*mu
            if log_pdf_z0 is not None: log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.nn.softplus(pre_scale)/(np.log(1+np.exp(0))), 1e-7, np.inf)  
            z = mu+scale*z0
            if log_pdf_z0 is not None: log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ExponentialScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.exp(pre_scale), 1e-7, np.inf)  
            z = mu+scale*z0
            if log_pdf_z0 is not None: log_abs_det_jacobian = tf.reduce_sum(pre_scale, axis=[1], keep_dims=True)
        elif self._mode == 'BoundedScaleShift':
            mu = pre_mu
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale)*gap, 1e-7, np.inf)  
            z = mu+scale*z0
            if log_pdf_z0 is not None: log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'BoundedScaleShiftAdjusted': ## CAUSED SOME NAN ISSUES
            mu = pre_mu
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale+scipy.special.logit(1/gap))*gap, 1e-7, np.inf)  
            z = mu+scale*z0
            if log_pdf_z0 is not None: log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'VolumePreserving':
            mu = pre_mu
            scale = tf.ones(shape=tf.shape(pre_scale))
            z = mu+z0
            if log_pdf_z0 is not None: log_abs_det_jacobian = 0
        elif self._mode == 'BoundedVolumePreserving':
            mu = tf.nn.tanh(pre_mu)*10
            scale = tf.ones(shape=tf.shape(pre_scale))
            z = mu+z0
            if log_pdf_z0 is not None: log_abs_det_jacobian = 0
        else: quit()

        if log_pdf_z0 is not None: log_pdf_z = log_pdf_z0-log_abs_det_jacobian
        else: log_pdf_z = None
        return z, log_pdf_z
    
    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

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
                scale_i = tf.clip_by_value(tf.nn.sigmoid(pre_scale_i), 1e-7, np.inf)  
                z0_i = (z[:, i, np.newaxis]-(1-scale_i)*mu_i)/scale_i
            elif self._mode == 'ScaleShift':
                mu_i = pre_mu_i
                scale_i = tf.clip_by_value(tf.nn.softplus(pre_scale_i)/(np.log(1+np.exp(0))), 1e-7, np.inf)  
                z0_i = (z[:, i, np.newaxis]-mu_i)/scale_i
            elif self._mode == 'ExponentialScaleShift':
                mu_i = pre_mu_i
                scale_i = tf.clip_by_value(tf.exp(pre_scale_i), 1e-7, np.inf)  
                z0_i = (z[:, i, np.newaxis]-mu_i)/scale_i
            elif self._mode == 'BoundedScaleShift':
                mu_i = pre_mu_i
                gap = (self._max_bounded_scale-self._min_bounded_scale)
                scale_i = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale_i)*gap, 1e-7, np.inf)  
                z0_i = (z[:, i, np.newaxis]-mu_i)/scale_i
            elif self._mode == 'BoundedScaleShiftAdjusted': ## CAUSED SOME NAN ISSUES
                mu_i = pre_mu_i
                gap = (self._max_bounded_scale-self._min_bounded_scale)
                scale_i = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale_i+scipy.special.logit(1/gap))*gap, 1e-7, np.inf)  
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
        z0 = tf.concat(z0_list, axis=1)

        if log_pdf_z is not None: 
            if self._mode == 'RNN': log_abs_det_jacobian = -tf.reduce_sum(tf.log(1e-7+tf.concat(scale_list, axis=1)), axis=[1], keep_dims=True)
            elif self._mode == 'ScaleShift': log_abs_det_jacobian = -tf.reduce_sum(tf.log(1e-7+tf.concat(scale_list, axis=1)), axis=[1], keep_dims=True)
            elif self._mode == 'ExponentialScaleShift': log_abs_det_jacobian = -tf.reduce_sum(tf.concat(pre_scale_list, axis=1), axis=[1], keep_dims=True)
            elif self._mode == 'BoundedScaleShift': log_abs_det_jacobian = -tf.reduce_sum(tf.log(1e-7+tf.concat(scale_list, axis=1)), axis=[1], keep_dims=True)
            elif self._mode == 'VolumePreserving': log_abs_det_jacobian = 0
            elif self._mode == 'BoundedVolumePreserving': log_abs_det_jacobian = 0
            log_pdf_z0 = log_pdf_z-log_abs_det_jacobian
        else: log_pdf_z0 = None
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
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)
        
        z0_same = z0[:, :self._same_dim]
        z0_change = z0[:, self._same_dim:]
        pre_mu, pre_scale = self.transformFunction(z0_same, self._layerwise_parameters, self._nonlinearity)

        if self._mode == 'RNN':
            mu = pre_mu
            scale = tf.clip_by_value(tf.nn.sigmoid(pre_scale), 1e-7, np.inf)  
            z_change = scale*z0_change+(1-scale)*mu
            if log_pdf_z0 is not None: log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.nn.softplus(pre_scale)/(np.log(1+np.exp(0))), 1e-7, np.inf)
            z_change = mu+scale*z0_change
            if log_pdf_z0 is not None: log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ExponentialScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.exp(pre_scale), 1e-7, np.inf)
            z_change = mu+scale*z0_change
            if log_pdf_z0 is not None: log_abs_det_jacobian = tf.reduce_sum(pre_scale, axis=[1], keep_dims=True)
        elif self._mode == 'BoundedScaleShift':
            mu = pre_mu
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale)*gap, 1e-7, np.inf)
            z_change = mu+scale*z0_change
            if log_pdf_z0 is not None: log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'BoundedScaleShiftAdjusted': ## CAUSED SOME NAN ISSUES
            mu = pre_mu
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale+scipy.special.logit(1/gap))*gap, 1e-7, np.inf)
            z_change = mu+scale*z0_change
            if log_pdf_z0 is not None: log_abs_det_jacobian = tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'VolumePreserving':
            mu = pre_mu
            scale = tf.ones(shape=tf.shape(pre_scale))
            z_change = mu+z0_change
            if log_pdf_z0 is not None: log_abs_det_jacobian = 0
        elif self._mode == 'BoundedVolumePreserving':
            mu = tf.nn.tanh(pre_mu)*10
            scale = tf.ones(shape=tf.shape(pre_scale))
            z_change = mu+z0_change
            if log_pdf_z0 is not None: log_abs_det_jacobian = 0
        else: quit()
        z = tf.concat([z0_same, z_change], axis=1)
        
        if log_pdf_z0 is not None: log_pdf_z = log_pdf_z0-log_abs_det_jacobian
        else: log_pdf_z = None
        return z, log_pdf_z
    
    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)

        z_same = z[:, :self._same_dim]
        z_change = z[:, self._same_dim:]
        pre_mu, pre_scale = self.transformFunction(z_same, self._layerwise_parameters, self._nonlinearity)

        if self._mode == 'RNN':
            mu = pre_mu
            scale = tf.clip_by_value(tf.nn.sigmoid(pre_scale), 1e-7, np.inf)  
            z0_change = (z_change-(1-scale)*mu)/scale
            if log_pdf_z is not None: log_abs_det_jacobian = -tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.nn.softplus(pre_scale)/(np.log(1+np.exp(0))), 1e-7, np.inf)  
            z0_change = (z_change-mu)/scale
            if log_pdf_z is not None: log_abs_det_jacobian = -tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'ExponentialScaleShift':
            mu = pre_mu
            scale = tf.clip_by_value(tf.exp(pre_scale), 1e-7, np.inf)  
            z0_change = (z_change-mu)/scale
            if log_pdf_z is not None: log_abs_det_jacobian = -tf.reduce_sum(pre_scale, axis=[1], keep_dims=True)
        elif self._mode == 'BoundedScaleShift':
            mu = pre_mu
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale)*gap, 1e-7, np.inf)  
            z0_change = (z_change-mu)/scale
            if log_pdf_z is not None: log_abs_det_jacobian = -tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'BoundedScaleShiftAdjusted': ## CAUSED SOME NAN ISSUES
            mu = pre_mu
            gap = (self._max_bounded_scale-self._min_bounded_scale)
            scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(pre_scale+scipy.special.logit(1/gap))*gap, 1e-7, np.inf)  
            z0_change = (z_change-mu)/scale
            if log_pdf_z is not None: log_abs_det_jacobian = -tf.reduce_sum(tf.log(scale), axis=[1], keep_dims=True)
        elif self._mode == 'VolumePreserving':
            mu = pre_mu
            scale = tf.ones(shape=tf.shape(pre_scale))
            z0_change = (z_change-mu)
            if log_pdf_z is not None: log_abs_det_jacobian = 0
        elif self._mode == 'BoundedVolumePreserving':
            mu = tf.nn.tanh(pre_mu)*10
            scale = tf.ones(shape=tf.shape(pre_scale))
            z0_change = (z_change-mu)
            if log_pdf_z is not None: log_abs_det_jacobian = 0
        else: quit()

        z0 = tf.concat([z_same, z0_change], axis=1)
        if log_pdf_z is not None: log_pdf_z0 = log_pdf_z-log_abs_det_jacobian
        else: log_pdf_z0 = None
        return z0, log_pdf_z0

#####################################################################################
#####################  Convolutional Invertible Euclidean Flows #####################
#####################################################################################


# class ConvNonLinearIARFlow():
#     """
#     Non-Linear Inverse Autoregressive Flow class.

#     Args:
#       parameters: parameters of transformation all appended.
#       input_dim : input dimensionality of the transformation. 
#     Raises:
#       ValueError: 
#     """

#     def __init__(self, input_dim, parameters, mode='ScaleShift', name='conv_nonlinearIAR_transform'):   #real
#         self._parameter_scale = 1
#         self._parameters = self._parameter_scale*parameters
#         self._input_dim = input_dim
#         self._nonlinearity = helper.LeakyReLU 
#         # self._nonlinearity = tf.nn.tanh
#         self._mode = mode
#         self._max_bounded_scale = 1.1
#         self._min_bounded_scale = 0.9
#         # self._min_bounded_scale = 1/self._max_bounded_scale
#         assert (self._input_dim > 1)
#         assert (self._max_bounded_scale > 1 and self._min_bounded_scale >= 0 and self._max_bounded_scale > self._min_bounded_scale)

#         self._parameters.get_shape().assert_is_compatible_with([None, NonLinearIARFlow.required_num_parameters(self._input_dim)])

#         self._mask_tensors = helper.tf_get_mask_list_for_MADE(self._input_dim-1, NonLinearIARFlow.layer_expansions, add_mu_log_sigma_layer=True)
#         self._pre_mu_for_1 = self._parameters[:, 0, np.newaxis]
#         self._pre_scale_for_1 = self._parameters[:, 1, np.newaxis]
#         self._layerwise_parameters = []

#         start_ind = 2
#         concat_layer_expansions = [1, *NonLinearIARFlow.layer_expansions, 2]
#         for l in range(len(concat_layer_expansions)-1):
#             W_num_param = concat_layer_expansions[l]*(self._input_dim-1)*concat_layer_expansions[l+1]*(self._input_dim-1) # matrix
#             B_num_param = concat_layer_expansions[l+1]*(self._input_dim-1) # bias
#             W_l_flat = tf.slice(self._parameters, [0, start_ind], [-1, W_num_param])
#             B_l_flat = tf.slice(self._parameters, [0, start_ind+W_num_param], [-1, B_num_param])
#             W_l = tf.reshape(W_l_flat, [-1, concat_layer_expansions[l+1]*(self._input_dim-1), concat_layer_expansions[l]*(self._input_dim-1)])              
#             B_l = tf.reshape(B_l_flat, [-1, concat_layer_expansions[l+1]*(self._input_dim-1)])
#             self._layerwise_parameters.append((W_l, B_l))  
#             start_ind += (W_num_param+B_num_param)

#     @property
#     def input_dim(self):
#         return self._input_dim

#     @property
#     def output_dim(self):
#         return self._input_dim









#####################################################################################
###########################  Serial Flows and Helpers ###############################
#####################################################################################

class SerialFlow():
    def __init__(self, transforms, name='serial_transform'): 
        self._transforms = transforms
    
    def transform(self, z0, log_pdf_z0):
        if log_pdf_z0 is not None: verify_size(z0, log_pdf_z0)
        curr_z, curr_log_pdf_z = z0, log_pdf_z0
        for i in range(len(self._transforms)):
            curr_z, curr_log_pdf_z = self._transforms[i].transform(curr_z, curr_log_pdf_z)
        return curr_z, curr_log_pdf_z
    
    def inverse_transform(self, z, log_pdf_z):
        if log_pdf_z is not None: verify_size(z, log_pdf_z)
        curr_z0, curr_log_pdf_z0 = z, log_pdf_z
        for i in range(len(self._transforms)):
            curr_z0, curr_log_pdf_z0 = self._transforms[-i-1].inverse_transform(curr_z0, curr_log_pdf_z0)
        return curr_z0, curr_log_pdf_z0

class GeneralInverseFlow():
    def __init__(self, transform, name='general_inverse_transform'): 
        self._transform = transform
        assert ('inverse_transform' in dir(self._transform))

    def transform(self, z0, log_pdf_z0):
        z, log_pdf_z = self._transform.inverse_transform(z0, log_pdf_z0)
        return z, log_pdf_z
    
    def inverse_transform(self, z, log_pdf_z):
        z0, log_pdf_z0 = self._transform.transform(z, log_pdf_z)
        return z0, log_pdf_z0

def verify_size(z0, log_pdf_z0):
  z0.get_shape().assert_is_compatible_with([None, None])
  if log_pdf_z0 is not None:
    log_pdf_z0.get_shape().assert_is_compatible_with([None, 1])










































