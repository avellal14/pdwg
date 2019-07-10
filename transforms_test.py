"""Random variable transformation classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import transforms 
import helper 
import pdb 
import numpy as np
import math 
from random import shuffle 
import time
import scipy
from scipy import special
import matplotlib.pyplot as plt

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
    print('\n\n\n')
    print('logdet_expected:')
    print(logdet_expected)
    print('logdet:')
    print(logdet)
    if np.all(np.abs(logdet_expected-logdet)<rtol): print('Transform update correct. Max error: ', np.abs(logdet_expected-logdet).max())
    else: print('Transform update incorrect !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Max error: ', np.abs(logdet_expected-logdet).max())
    print(np.abs(logdet_expected-logdet))

# #####################################################################################
# ######################### Euclidean Flows Jacobian Test #############################
# #####################################################################################

# n_tests = 1
# batch_size = 5
# n_input = 10
# for transform_to_check in [\
                        
#                            ######################### Non-analytical Inverse Flows ##############################
#                            # transforms.PlanarFlow, \
#                            # transforms.RadialFlow, \
                           
#                            ############################## Dimension Shuffle Flows ##############################
#                            # transforms.InverseOrderDimensionFlow, \
#                            # transforms.CircularSlideDimensionFlow, \
                           
#                            ############################## Scale and Interval Flows #############################
#                            # transforms.ScaleDimensionFlow, \
#                            # transforms.OpenIntervalDimensionFlow, \
#                            # transforms.InverseOpenIntervalDimensionFlow, \
                           
#                            ##################################  Rotation Flows ##################################
                           # transforms.SpecificRotationFlow, \
                           # transforms.NotManyReflectionsRotationFlow, \
                           # transforms.ManyReflectionsRotationFlow, \
                           # transforms.HouseholdRotationFlow, \
                           # transforms.CompoundRotationFlow, \

#                            #############################  Invertible Euclidean Flows ###########################
#                            # transforms.PiecewisePlanarScalingFlow,
#                            # transforms.LinearIARFlow, \
#                            # transforms.NonLinearIARFlow, \
#                            # transforms.RealNVPFlow, \
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

#     n_parameter = transform_to_check.required_num_parameters(n_input)

#     for parameter_scale in [1, 10]:
#         parameters = None
#         if n_parameter > 0: parameters = parameter_scale*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
#         transform_object =  transform_to_check(input_dim=n_input, parameters=parameters)

#         z0 = tf.random_uniform(shape=(batch_size, n_input), dtype=tf.float32) # required for some transforms#
#         log_pdf_z0 = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)

#         for repeat in range(n_tests): _check_logdet(transform_object, z0, log_pdf_z0)

#####################################################################################
######################### Non-analytical Inverse Flows ##############################
#####################################################################################

# No additional testing required

#####################################################################################
############################## Dimension Shuffle Flows ##############################
#####################################################################################

# batch_size = 5
# n_input = 6
# for transform_to_check in [\
#                              transforms.InverseOrderDimensionFlow, \
#                              transforms.CircularSlideDimensionFlow, \
#                           ]:
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('\n\n\n')
#     print('            '+str(transform_to_check)+'               ')
#     print('\n\n\n')    
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

#     parameters = None
#     z0 = tf.random_uniform(shape=(batch_size, n_input), dtype=tf.float32) # required for some transforms#
#     transform = transform_to_check(input_dim=n_input, parameters=parameters)
#     z, _ = transform.transform(z0, None)
#     z0_inv, _ = transform.inverse_transform(z, None)

#     init = tf.initialize_all_variables()
#     sess = tf.InteractiveSession()  
#     sess.run(init)
#     z0_np, z_np, z0_inv_np = sess.run([z0, z, z0_inv])
    
#     print('\n\n')
#     print('Max absolute difference between z0 and z0_inv: ', np.abs((z0_np-z0_inv_np)).max())
#     print(np.max(np.abs((z0_np-z0_inv_np)), axis=0))
    
#     print('\n\n')
#     print('Examples:')
#     print('\n\n')
#     print('z0:')
#     print(z0_np[0,:])
#     print('z:')
#     print(z_np[0,:])
#     print('z0_inv:')
#     print(z0_inv_np[0,:])

#####################################################################################
############################## Scale and Interval Flows #############################
#####################################################################################

# batch_size = 5
# n_input = 6
# for transform_to_check in [\
#                              transforms.ScaleDimensionFlow, \
#                              transforms.OpenIntervalDimensionFlow, \
#                              transforms.InverseOpenIntervalDimensionFlow, \
#                           ]:
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('\n\n\n')
#     print('            '+str(transform_to_check)+'               ')
#     print('\n\n\n')    
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

#     parameters = None
#     z0 = tf.random_uniform(shape=(batch_size, n_input), dtype=tf.float32) # required for some transforms#
#     transform = transform_to_check(input_dim=n_input, parameters=parameters)
#     z, _ = transform.transform(z0, None)
#     z0_inv, _ = transform.inverse_transform(z, None)

#     init = tf.initialize_all_variables()
#     sess = tf.InteractiveSession()  
#     sess.run(init)
#     z0_np, z_np, z0_inv_np = sess.run([z0, z, z0_inv])
    
#     print('\n\n')
#     print('Max absolute difference between z0 and z0_inv: ', np.abs((z0_np-z0_inv_np)).max())
#     print(np.max(np.abs((z0_np-z0_inv_np)), axis=0))

#     print('\n\n')
#     print('Examples:')
#     print('\n\n')
#     print('z0:')
#     print(z0_np[0,:])
#     print('z:')
#     print(z_np[0,:])
#     print('z0_inv:')
#     print(z0_inv_np[0,:])

# ####################################################################################
# #################################  Rotation Flows ##################################
# ####################################################################################

# batch_size = 5
# n_input = 6

# for transform_to_check in [\
#                            transforms.SpecificRotationFlow, \
#                            transforms.NotManyReflectionsRotationFlow, \
#                            transforms.ManyReflectionsRotationFlow, \
#                            transforms.HouseholdRotationFlow, \
#                            transforms.CompoundRotationFlow, \
#                           ]:
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('\n\n\n')
#     print('            '+str(transform_to_check)+'               ')
#     print('\n\n\n')    
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

#     n_parameter = transform_to_check.required_num_parameters(n_input)
#     parameters, parameters_batch = None, None
#     if n_parameter > 0: parameters = 10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
#     if n_parameter > 0: parameters_batch = 10*tf.layers.dense(inputs = tf.ones(shape=(batch_size, 1)), units = n_parameter, use_bias = False, activation = None)

#     z0 = tf.random_uniform(shape=(batch_size, n_input), dtype=tf.float32) # required for some transforms#
#     transform = transform_to_check(input_dim=n_input, parameters=parameters)
#     transform_batch = transform_to_check(input_dim=n_input, parameters=parameters_batch)
#     z, _ = transform.transform(z0, None)
#     z0_inv, _ = transform.inverse_transform(z, None)
#     rot_mat = transform.get_batched_rot_matrix()

#     z_batch, _ = transform_batch.transform(z0, None)
#     z0_batch_inv, _ = transform_batch.inverse_transform(z_batch, None)
#     rot_batch_mat = transform_batch.get_batched_rot_matrix()

#     if hasattr(transform, '_mode'):
#         transform._mode = 'matrix'
#         transform._batched_rot_matrix = transform.get_batched_rot_matrix() 
#         transform_batch._mode = 'matrix'
#         transform_batch._batched_rot_matrix = transform_batch.get_batched_rot_matrix() 

#         z_mat_mode, _  = transform.transform(z0, None)
#         z0_inv_mat_mode, _ = transform.inverse_transform(z_mat_mode, None)
#         z_batch_mat_mode, _ = transform_batch.transform(z0, None)
#         z0_batch_inv_mat_mode, _ = transform_batch.inverse_transform(z_batch_mat_mode, None)

#         transform._mode = 'vector'
#         transform_batch._mode = 'vector'
#     else:
#         z_mat_mode = z
#         z0_inv_mat_mode = z0_inv
#         z_batch_mat_mode = z_batch
#         z0_batch_inv_mat_mode = z0_batch_inv

#     init = tf.initialize_all_variables()
#     sess = tf.InteractiveSession()  
#     sess.run(init)
#     rot_mat_np, rot_batch_mat_np, z0_np, z_np, z0_inv_np, z_batch_np, z0_batch_inv_np, z_mat_mode_np, z0_inv_mat_mode_np, z_batch_mat_mode_np, z0_batch_inv_mat_mode_np = \
#         sess.run([rot_mat, rot_batch_mat, z0, z, z0_inv, z_batch, z0_batch_inv, z_mat_mode, z0_inv_mat_mode, z_batch_mat_mode, z0_batch_inv_mat_mode])
    
#     print('\n\n')
#     print('Max absolute difference between z0 and z0_inv: ', np.abs((z0_np-z0_inv_np)).max())
#     print(np.max(np.abs((z0_np-z0_inv_np)), axis=0))
#     print('\n\n')
#     print('Max absolute difference between z0 and z0_batch_inv: ', np.abs((z0_np-z0_batch_inv_np)).max())
#     print(np.max(np.abs((z0_np-z0_batch_inv_np)), axis=0))
#     print('\n\n')
#     print('Max absolute difference between z0 and z0_batch_inv_mat_mode: ', np.abs((z0_np-z0_batch_inv_mat_mode_np)).max())
#     print(np.max(np.abs((z0_np-z0_batch_inv_mat_mode_np)), axis=0))
        
#     print('\n\n')
#     print('Examples:')
#     print('\n\n')
#     print('z0:')
#     print(z0_np[0,:])
#     print('z:')
#     print(z_np[0,:])
#     print('z0_inv:')
#     print(z0_inv_np[0,:])

#     print('\n\n')
#     print('Examples batched:')
#     print('\n\n')
#     print('z0:')
#     print(z0_np[0,:])
#     print('z_batch:')
#     print(z_batch_np[0,:])
#     print('z0_batch_inv:')
#     print(z0_batch_inv_np[0,:])

#     print('\n\n')
#     print('Max absolute difference between z and z_mat_mode: ', np.abs((z_np-z_mat_mode_np)).max())
#     print(np.max(np.abs((z_np-z_mat_mode_np)), axis=0))
#     print('\n\n')
#     print('Max absolute difference between z_batch and z_batch_mat_mode: ', np.abs((z_batch_np-z_batch_mat_mode_np)).max())
#     print(np.max(np.abs((z_batch_np-z_batch_mat_mode_np)), axis=0))

#     print('\n\n')
#     print('Rotation matrix shape: ', rot_mat_np.shape)
#     assert(rot_mat_np.shape[0] == 1 and rot_mat_np.shape[1] == n_input and rot_mat_np.shape[2] == n_input)
#     print('Rotation matrix diagonal: ')
#     print(np.diag(rot_mat_np[0]))    

#     print('\n\n')
#     print('Rotation matrix batched shape: ', rot_batch_mat_np.shape)
#     if n_parameter > 0:
#         assert(rot_batch_mat_np.shape[0] == batch_size and rot_batch_mat_np.shape[1] == n_input and rot_batch_mat_np.shape[2] == n_input)
#     else:
#         assert(rot_batch_mat_np.shape[0] == 1 and rot_batch_mat_np.shape[1] == n_input and rot_batch_mat_np.shape[2] == n_input)
#     print('Rotation matrix batched diagonal: ')
#     print(np.diag(rot_batch_mat_np[0]))    

#     print('Rotation matrix determinant: ', np.linalg.det(rot_mat_np[0]))
#     print('Rotation matrix eye-R*R^T == 0, Error:', np.abs(np.eye(rot_mat_np[0].shape[0])-np.dot(rot_mat_np[0], rot_mat_np[0].T)).max())

#     print('Rotation matrix batched determinant: ', np.linalg.det(rot_batch_mat_np[0]))
#     print('Rotation matrix batched eye-R*R^T == 0, Error:', np.abs(np.eye(rot_batch_mat_np[0].shape[0])-np.dot(rot_batch_mat_np[0], rot_batch_mat_np[0].T)).max())

#     # plt.imshow(rot_mat_np[0], cmap='hot', interpolation='nearest')
#     # plt.show()

#####################################################################################
#################################  Orthogonal Maps ##################################
#####################################################################################

# batch_size = 5
# n_input = 6
# for transform_to_check in [\
#                            transforms.PiecewisePlanarScalingMap,
#                            transforms.ConnectedPiecewiseOrthogonalMap, \
#                            transforms.CompoundPiecewiseOrthogonalMap, \
#                           ]:
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('\n\n\n')
#     print('            '+str(transform_to_check)+'               ')
#     print('\n\n\n')    
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

#     n_parameter = transform_to_check.required_num_parameters(n_input)
#     parameters = None
#     if n_parameter > 0: parameters = 10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)

#     z0 = tf.random_uniform(shape=(batch_size, n_input), dtype=tf.float32) # required for some transforms#
#     transform = transform_to_check(input_dim=n_input, parameters=parameters)
#     z, log_scales = transform.transform(z0)
#     jacobian = transform.jacobian(z0)

#     init = tf.initialize_all_variables()
#     sess = tf.InteractiveSession()  
#     sess.run(init)
#     z0_np, z_np, log_scales_np, jacobian_np = sess.run([z0, z, log_scales, jacobian])
    
#     print('\n\n')
#     print('Examples:')
#     print('\n\n')
#     print('z0:')
#     print(z0_np[0,:])
#     print('z:')
#     print(z_np[0,:])
#     print('log_scales:')
#     print(log_scales_np[0,:])

#     print('\n\n\n')
#     print('Example Jacobian Shape: ', jacobian_np.shape)
#     print('Example Jacobian: \n', jacobian_np[0, :, :])
#     print('Example Jacobian*Jacobian^T:\n', np.dot(jacobian_np[0, :, :], jacobian_np[0, :, :].T))
#     print('Example Jacobian^T*Jacobian:\n', np.dot(jacobian_np[0, :, :].T, jacobian_np[0, :, :]))
#     print('\n\n\n')

#     scales_sq = ((np.exp(log_scales_np)**2))[:,0]
#     scales_sq_jacobian = np.zeros(scales_sq.shape)
#     for i in range(jacobian_np.shape[0]):
#         JJT = np.dot(jacobian_np[i, :, :], jacobian_np[i, :, :].T)
#         scales_sq_jacobian[i] = JJT[0,0]
    
#     print('scales^2 from log_scales vs scales^2 from jacobian, Error: ', np.abs(scales_sq_jacobian-scales_sq).max())
#     print(np.abs(scales_sq_jacobian-scales_sq))

#####################################################################################
###########################  Riemannian Maps and Flows ##############################
#####################################################################################

# batch_size = 5
# n_input = 4
# transform_to_check = transforms.OthogonalProjectionMap

# for n_out in [2, 4, 6]:
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('\n\n\n')
#     print('            '+str(transform_to_check)+'               ')
#     print('       (n_input, n_out): '+str((n_input, n_out))+'    ')
#     print('\n\n\n')    
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

#     n_parameter = transform_to_check.required_num_parameters(n_input, n_out)
#     parameters = None
#     if n_parameter > 0: parameters = 10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)

#     z0 = tf.random_normal((batch_size, n_input), 0, 1, dtype=tf.float32)
#     transform = transform_to_check(input_dim=n_input, output_dim=n_out, parameters=parameters)

#     z = transform.transform(z0)
#     jacobian = transform.jacobian(z0)
#     transform._mode = 'matrix'
#     z_mat_mode  = transform.transform(z0)
#     transform._mode = 'vector'

#     init = tf.initialize_all_variables()
#     sess = tf.InteractiveSession()  
#     sess.run(init)
#     z0_np, z_np, z_mat_mode_np, jacobian_np = sess.run([z0, z, z_mat_mode, jacobian])

#     print('\n\n\n')
#     print('Examples:')
#     print('\n\n')
#     print('z0:')
#     print(z0_np[0,:])
#     print('z:')
#     print(z_np[0,:])

#     print('\n\n\n')
#     print('Example Jacobian Shape: ', jacobian_np.shape)
#     print('Example Jacobian: \n', jacobian_np[0, :, :])
#     print('Example Jacobian*Jacobian^T:\n', np.dot(jacobian_np[0, :, :], jacobian_np[0, :, :].T))
#     print('Example Jacobian^T*Jacobian:\n', np.dot(jacobian_np[0, :, :].T, jacobian_np[0, :, :]))
    
#     JJ_smaller = np.zeros((jacobian_np.shape[0], min(n_input, n_out), min(n_input, n_out)))
#     if n_input > n_out: 
#         for i in range(jacobian_np.shape[0]): 
#             JJ_smaller[i, :, :] = np.dot(jacobian_np[i, :, :], jacobian_np[i, :, :].T)
#     else:
#         for i in range(jacobian_np.shape[0]): 
#             JJ_smaller[i, :, :] = np.dot(jacobian_np[i, :, :].T, jacobian_np[i, :, :])
    
#     print('\n\n\n')
#     print('JJ^T or J^T*J (whichever is smaller matrix) eye-JJ_smaller == 0, Error:', np.abs(JJ_smaller-np.eye(min(n_input, n_out))[np.newaxis,:,:]).max())
#     print(np.abs(JJ_smaller-np.eye(min(n_input, n_out))[np.newaxis,:,:]).max(2).max(1))

#     print('\n\n\n')
#     print('Max absolute difference between z and z_mat_mode: ', np.abs((z_np-z_mat_mode_np)).max())
#     print(np.max(np.abs((z_np-z_mat_mode_np)), axis=0))


batch_size = 10
n_input = 3
n_input_NOM, n_output_NOM = 4, 3
transform_to_check = transforms.RiemannianFlow

# for n_out in [5, 6, 7]:
for n_out in [7,]:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('\n\n\n')
    print('            '+str(transform_to_check)+'               ')
    print('       (n_input, n_out): '+str((n_input, n_out))+'    ')
    print('\n\n\n')    
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    n_parameter = transform_to_check.required_num_parameters(n_input, n_out, n_input_NOM, n_output_NOM)
    parameters = None
    if n_parameter > 0: parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)

    z0 = tf.random_normal((batch_size, n_input), 0, 1, dtype=tf.float32)
    log_pdf_z0 = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)
    transform = transform_to_check(input_dim=n_input, output_dim=n_out, n_input_NOM=n_input_NOM, n_output_NOM=n_output_NOM, parameters=parameters)

    z, log_pdf_z, all_scales = transform.transform(z0, log_pdf_z0, mode='scales')
    additional_jacobian = transform.jacobian(z0, mode='additional')
    full_jacobian = transform.jacobian(z0, mode='full')

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()  
    sess.run(init)
    z0_np, log_pdf_z0_np, z_np, log_pdf_z_np, all_scales_np, jacobian_np, full_jacobian_np = sess.run([z0, log_pdf_z0, z, log_pdf_z, all_scales, additional_jacobian, full_jacobian])

    print('\n\n')
    print('Examples:')
    print('\n\n')
    print('z0:')
    print(z0_np[0,:])
    print('z:')
    print(z_np[0,:])

    print('\n\n\n')
    print('Example Additional Jacobian Shape: ', jacobian_np.shape)
    print('Example Additional Jacobian: \n', jacobian_np[0, :, :])
    print('Example Additional Jacobian*Jacobian^T:\n', np.dot(jacobian_np[0, :, :], jacobian_np[0, :, :].T))
    print('Example Additional Jacobian^T*Jacobian:\n', np.dot(jacobian_np[0, :, :].T, jacobian_np[0, :, :]))
    
    n_additional = n_out-n_input
    JJ_smaller = np.zeros((jacobian_np.shape[0], min(n_input, n_additional), min(n_input, n_additional)))
    if n_input > n_additional: 
        for i in range(jacobian_np.shape[0]): 
            JJ_smaller[i, :, :] = np.dot(jacobian_np[i, :, :], jacobian_np[i, :, :].T)
    else:
        for i in range(jacobian_np.shape[0]): 
            JJ_smaller[i, :, :] = np.dot(jacobian_np[i, :, :].T, jacobian_np[i, :, :])
    
    JJ_smaller_normalized = JJ_smaller/JJ_smaller[:,0,0][:,np.newaxis,np.newaxis]

    print('\n\n\n')
    print('Additional Jacobian*Jacobian^T or Jacobian^T*Jacobian is a scaled identity matrix:')
    print('Additional JJ^T or J^T*J (whichever is smaller matrix) eye-JJ_smaller/JJ_smaller[0,0] == 0, Error:', np.abs(JJ_smaller_normalized-np.eye(min(n_input, n_additional))[np.newaxis,:,:]).max())
    print(np.abs(JJ_smaller_normalized-np.eye(min(n_input, n_additional))[np.newaxis,:,:]).max(2).max(1))

    scales_sq = ((all_scales_np**2))[:,0]
    scales_sq_jacobian = JJ_smaller[:,0,0]
    
    print('\n\n\n')
    print('scales^2 from all_scales vs scales^2 from additional jacobian, Error: ', np.abs(scales_sq_jacobian-scales_sq).max())
    print(np.abs(scales_sq_jacobian-scales_sq))

    print('\n\n\n')
    print('Example Full Jacobian Shape: ', full_jacobian_np.shape)
    print('Example Full Jacobian: \n', full_jacobian_np[0, :, :])
    print('Example Full Jacobian*Jacobian^T:\n', np.dot(full_jacobian_np[0, :, :], full_jacobian_np[0, :, :].T))
    print('Example Full Jacobian^T*Jacobian:\n', np.dot(full_jacobian_np[0, :, :].T, full_jacobian_np[0, :, :]))
    
    full_JTJ = np.zeros((full_jacobian_np.shape[0], n_input, n_input))
    for i in range(full_jacobian_np.shape[0]): 
        full_JTJ[i, :, :] = np.dot(full_jacobian_np[i, :, :].T, full_jacobian_np[i, :, :])
    full_JTJ_normalized = full_JTJ/full_JTJ[:,0,0][:,np.newaxis,np.newaxis]

    scale_change = 1/np.exp(log_pdf_z_np-log_pdf_z0_np)
    scale_change_full_jacobian = np.zeros(scale_change.shape)

    for i in range(full_jacobian_np.shape[0]): 
        scale_change_full_jacobian[i] = np.sqrt(np.linalg.det(full_JTJ[i, :, :]))
    
    print('\n\n\n')
    print('scale_change from log_pdfs vs scale_change from full jacobian, Error: ', np.abs(scale_change-scale_change_full_jacobian).max())
    print(np.abs(scale_change-scale_change_full_jacobian))

    print('\n\n\n')
    print('Check that Full Jacobian J^T*J is always scaled multiple of identity matrix when n_additional >= n_input.')
    print('\n\n\n')
    if (n_additional >= n_input): print('Error should be SMALL since n_additional >= n_input!!')
    else: print('Error should be LARGE since n_additional < n_input!!')
    print('Full Jacobian J^T*J eye-J^T*J/J^T*J[0,0] == 0, Error:', np.abs(full_JTJ_normalized-np.eye(n_input)[np.newaxis,:,:]).max())
    print(np.abs(full_JTJ_normalized-np.eye(n_input)[np.newaxis,:,:]).max(2).max(1))


#####################################################################################
#############################  Invertible Euclidean Flows ###########################
#####################################################################################

# batch_size = 5
# n_input = 6
# for transform_to_check in [\
#                            transforms.PiecewisePlanarScalingFlow,
#                            # # transforms.LinearIARFlow, \
#                            transforms.NonLinearIARFlow, \
#                            transforms.RealNVPFlow, \
#                           ]:
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('\n\n\n')
#     print('            '+str(transform_to_check)+'               ')
#     print('\n\n\n')    
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

#     n_parameter = transform_to_check.required_num_parameters(n_input)
#     parameters = None
#     if n_parameter > 0: parameters = 10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)

#     z0 = tf.random_uniform(shape=(batch_size, n_input), dtype=tf.float32) # required for some transforms#
#     transform = transform_to_check(input_dim=n_input, parameters=parameters)
#     z, _ = transform.transform(z0, None)
#     z0_inv, _ = transform.inverse_transform(z, None)

#     init = tf.initialize_all_variables()
#     sess = tf.InteractiveSession()  
#     sess.run(init)
#     z0_np, z_np, z0_inv_np = sess.run([z0, z, z0_inv])
    
#     print('\n\n')
#     print('Max absolute difference between z0 and z0_inv: ', np.abs((z0_np-z0_inv_np)).max())
#     print(np.max(np.abs((z0_np-z0_inv_np)), axis=0))

#     print('\n\n')
#     print('Examples:')
#     print('\n\n')
#     print('z0:')
#     print(z0_np[0,:])
#     print('z:')
#     print(z_np[0,:])
#     print('z0_inv:')
#     print(z0_inv_np[0,:])

#####################################################################################
###########################  Serial Flows and Helpers ###############################
#####################################################################################



















































































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








# class SpecificOrderDimensionFlow():
#     """
#     Specific Order Dimension Flow class.
#     Args:
#       parameters: parameters of transformation all appended.
#       input_dim : input dimensionality of the transformation. 
#     Raises:
#       ValueError: 
#     """
#     def __init__(self, input_dim, order=None, parameters=None, name='specific_order_dimension_transform'):   
#         print('Unusable in a safe fashion: This creates as random ordering which changes from random seed to random seed '+ 
#               'and is not saved in any checkpoint, since the order is not stored in a tf.Variable. Therefore, if the seed '+
#               'does not match or if there is a random call order change, then the loaded model from the checkpoint will be wrong.')
#         quit()
#         self._input_dim = input_dim 
#         if order is None: # a specific but random order
#             # self._order = [*range(self._input_dim)]
#             # shuffle(self._order) # works for subset of O(n) but not SO(n) 
#             print('SpecificOrderDimensionFlow, creating random order: ')
#             start = time.time()
#             n_swaps = 10*self._input_dim
#             self._order = [*range(self._input_dim)]     
#             for t in range(n_swaps):
#                 index_1 = np.random.randint(len(self._order))
#                 index_2 = index_1
#                 while index_2 == index_1: index_2 = np.random.randint(len(self._order))
#                 temp = self._order[index_1]
#                 self._order[index_1] = self._order[index_2]
#                 self._order[index_2] = temp
#             assert (n_swaps % 2 == 0) # SO(n)
#             print('Time: {:.3f}\n'.format((time.time() - start)))
#         else: self._order = order
#         self._inverse_order = [-1]*self._input_dim
#         for i in range(self._input_dim): self._inverse_order[self._order[i]] = i
        
#         assert (len(self._order) == self._input_dim)
#         assert (len(self._inverse_order) == self._input_dim)
#         assert (parameters is None)
#         assert (self._input_dim > 1)

#     @property
#     def input_dim(self):
#         return self._input_dim

#     @property
#     def output_dim(self):
#         return self._input_dim

#     @staticmethod
#     def required_num_parameters(input_dim):  
#         return 0

#     def get_batched_rot_matrix(self):
#         batched_rot_matrix_np = np.zeros((1, self._input_dim, self._input_dim))
#         for i in range(len(self._order)): batched_rot_matrix_np[0, i, self._order[i]] = 1
#         return tf.constant(batched_rot_matrix_np, tf.float32)

#     def transform(self, z0, log_pdf_z0):
#         verify_size(z0, log_pdf_z0)

#         z = helper.tf_differentiable_specific_shuffle_with_axis(z0, self._order, axis=1)
#         log_pdf_z = log_pdf_z0
#         return z, log_pdf_z

#     def inverse_transform(self, z, log_pdf_z):
#         verify_size(z, log_pdf_z)

#         z0 = helper.tf_differentiable_specific_shuffle_with_axis(z, self._inverse_order, axis=1)
#         log_pdf_z0 = log_pdf_z
#         return z0, log_pdf_z0

# class CustomSpecificOrderDimensionFlow():
#     """
#     Custom Specific Order Dimension Flow class.
#     Args:
#       parameters: parameters of transformation all appended.
#       input_dim : input dimensionality of the transformation. 
#     Raises:
#       ValueError: 
#     """
#     def __init__(self, input_dim, order=None, parameters=None, name='custom_specific_order_dimension_transform'):   
#         print('Unusable in a safe fashion: This creates as random ordering which changes from random seed to random seed '+ 
#               'and is not saved in any checkpoint, since the order is not stored in a tf.Variable. Therefore, if the seed '+
#               'does not match or if there is a random call order change, then the loaded model from the checkpoint will be wrong.')
#         quit()
#         self._input_dim = input_dim 
#         self._sodf_1 = SpecificOrderDimensionFlow(input_dim=int(self._input_dim/2.))
#         self._sodf_2 = SpecificOrderDimensionFlow(input_dim=int(self._input_dim/2.))
        
#         assert (self._input_dim % 2 == 0)
#         assert (parameters is None)
#         assert (self._input_dim > 1)

#     @property
#     def input_dim(self):
#         return self._input_dim

#     @property
#     def output_dim(self):
#         return self._input_dim

#     @staticmethod
#     def required_num_parameters(input_dim):  
#         return 0

#     def transform(self, z0, log_pdf_z0):
#         verify_size(z0, log_pdf_z0)

#         z0_1 = z0[:, :int(self._input_dim/2.)] 
#         z0_2 = z0[:, int(self._input_dim/2.):]
#         # z_1, _ = self._sodf_1.transform(z0_1, tf.zeros(shape=[tf.shape(z0_1)[0], 1]))
#         z_1 = z0_1
#         z_2, _ = self._sodf_2.transform(z0_2, tf.zeros(shape=[tf.shape(z0_2)[0], 1]))
#         z = tf.concat([z_1, z_2], axis=1)
#         log_pdf_z = log_pdf_z0
#         return z, log_pdf_z

#     def inverse_transform(self, z, log_pdf_z):
#         verify_size(z, log_pdf_z)

#         z_1 = z[:, :int(self._input_dim/2.)] 
#         z_2 = z[:, int(self._input_dim/2.):]
#         # z0_1, _ = self._sodf_1.inverse_transform(z_1, tf.zeros(shape=[tf.shape(z_1)[0], 1]))
#         z0_1 = z_1
#         z0_2, _ = self._sodf_2.inverse_transform(z_2, tf.zeros(shape=[tf.shape(z_2)[0], 1]))
#         z0 = tf.concat([z0_1, z0_2], axis=1)
#         log_pdf_z0 = log_pdf_z
#         return z0, log_pdf_z0























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


                # print('Please change me (transforms.py) back to avoid problems with checkpoints. Current setting is chosen based on computational speed. Constants embedded to the graph are faster')
                # self._constant_rot_mats_list.append(tf.constant(helper.random_rot_mat(self._input_dim, mode='SO(n)'), dtype=tf.float32))
                # print('Please change me (transforms.py) back to avoid problems with checkpoints. Current setting is chosen based on computational speed. Constants embedded to the graph are faster')
                # return tf.constant(helper.random_rot_mat(self._input_dim, mode='SO(n)'), dtype=tf.float32)[np.newaxis, :, :]






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

#     def __init__(self, input_dim, parameters, margin_mode='NoGradient', scale_mode='Scale', name='connected_piecewise_orthogonal_map'):   
#         self._parameter_scale = 1.
#         self._parameters = parameters
#         self._parameters = self._parameter_scale*self._parameters
#         self._input_dim = input_dim
#         self._margin_mode = margin_mode
#         self._scale_mode = scale_mode
#         self._max_bounded_scale = 5
#         self._min_bounded_scale = 1/self._max_bounded_scale

#         assert (self._margin_mode == 'NoGradient' or self._margin_mode == 'ST')
#         assert (self._scale_mode == 'Scale' or self._scale_mode == 'BoundedScale')
#         assert (self._max_bounded_scale > 1 and self._min_bounded_scale >= 0 and self._max_bounded_scale > self._min_bounded_scale)

#         self._parameters.get_shape().assert_is_compatible_with([None, ConnectedPiecewiseOrthogonalMap.required_num_parameters(self._input_dim)])
        
#         param_index = 0
#         self._pos_rotation_param, param_index = helper.slice_parameters(self._parameters, param_index, ConnectedPiecewiseOrthogonalMap.rotation_flow_class.required_num_parameters(self._input_dim))
#         self._neg_rotation_param, param_index = helper.slice_parameters(self._parameters, param_index, ConnectedPiecewiseOrthogonalMap.rotation_flow_class.required_num_parameters(self._input_dim))
#         self._pos_pre_scale, param_index = helper.slice_parameters(self._parameters, param_index, 1)
#         self._neg_pre_scale, param_index = helper.slice_parameters(self._parameters, param_index, 1)
#         self._hyper_pre_bias, param_index = helper.slice_parameters(self._parameters, param_index, 1)
#         self._hyper_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim) 
#         self._output_shift_vec, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim) 

#         if self._scale_mode == 'Scale':
#             self._pos_scale = tf.clip_by_value(tf.nn.softplus(self._pos_pre_scale)/np.log(1+np.exp(0)), 1e-7, np.inf)  
#             self._neg_scale = tf.clip_by_value(tf.nn.softplus(self._neg_pre_scale)/np.log(1+np.exp(0)), 1e-7, np.inf)  
#         elif self._scale_mode == 'BoundedScale': 
#             gap = (self._max_bounded_scale-self._min_bounded_scale)
#             self._pos_scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(self._pos_pre_scale+scipy.special.logit(1/gap))*gap, 1e-7, np.inf)  
#             self._neg_scale = tf.clip_by_value(self._min_bounded_scale+tf.nn.sigmoid(self._neg_pre_scale+scipy.special.logit(1/gap))*gap, 1e-7, np.inf)  
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
#         log_scales = tf.log(scales)
#         return z, log_scales

