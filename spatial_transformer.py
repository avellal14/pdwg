# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
# import components
import pdb
import numpy as np
import distributions

def transformer(input_im, pixel_transformation_clousure, out_size, n_location_samples=None, out_comparison_im=None, name='SpatialTransformerSampled', **kwargs):
    """Spatial Transformer Layer
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    """

    def _gather_from_ims(im, yx_list, n_location_samples):
        im_flat = tf.reshape(im, [-1, tf.shape(im)[3]])
        base = tf.reshape(tf.tile((tf.range(tf.shape(im)[0])*(tf.shape(im)[2]*tf.shape(im)[1]))[:,np.newaxis], [1, n_location_samples]), [-1])
        Ilist = []
        for i in range(len(yx_list)):
            y, x = yx_list[i]
            flat_ind = base+y*tf.shape(im)[2]+x
            Ilist.append(tf.gather(im_flat, flat_ind))
        return Ilist

    def _is_approx_zero(x, scale=1e-3):
        # x >= -scale and x <= scale ---> 1 else ----> 0
        return tf.cast(tf.math.less(x, scale), tf.float32)*tf.cast(tf.math.greater(x, -scale), tf.float32)

    # def _interpolate(im, y, x, n_location_samples, use_mean_background=False, train_background=[1, 1, 1], vis_background=[0, 0, 0]):
    # def _interpolate(im, y, x, n_location_samples, use_mean_background=False, train_background=[0, 0, 0.], vis_background=[0, 0, 0]): this works well for single images
    def _interpolate(im, y, x, n_location_samples, use_mean_background=False, train_background=[0, 0, 0.], vis_background=[0, 0, 0]): 

        # do sampling
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1

        y0 = tf.clip_by_value(y0, tf.zeros([], dtype='int32'), tf.shape(im)[1]-1)
        y1 = tf.clip_by_value(y1, tf.zeros([], dtype='int32'), tf.shape(im)[1]-1)
        x0 = tf.clip_by_value(x0, tf.zeros([], dtype='int32'), tf.shape(im)[2]-1)
        x1 = tf.clip_by_value(x1, tf.zeros([], dtype='int32'), tf.shape(im)[2]-1)
        Ia, Ib, Ic, Id = _gather_from_ims(im, [(y0, x0), (y1, x0), (y0, x1), (y1, x1)], n_location_samples)

        # and finally calculate interpolated values
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        
        wd = ((x-x0_f)*(y-y0_f))[:, np.newaxis]
        wc = ((x-x0_f)*(y1_f-y))[:, np.newaxis]
        wb = ((x1_f-x)*(y-y0_f))[:, np.newaxis]
        wa = ((x1_f-x)*(y1_f-y))[:, np.newaxis]

        invalid_map = _is_approx_zero(wa+wb+wc+wd)
        raw_out = (wa*Ia)+(wb*Ib)+(wc*Ic)+(wd*Id)
        
        if use_mean_background:
            mean_rgb_per_im = tf.reduce_mean(im, axis=[1,2])
            train_background = tf.reduce_mean(mean_rgb_per_im, axis=0)
            vis_background = train_background
        else:
            train_background = tf.constant(train_background, tf.float32)
            vis_background = tf.constant(vis_background, tf.float32)
        train_output = raw_out+invalid_map*train_background[np.newaxis, :]
        vis_output = raw_out+invalid_map*vis_background[np.newaxis, :]
        return train_output, vis_output, invalid_map

    def _meshgrid(height, width):
        y_t = tf.tile(tf.linspace(-1.0, 1.0, height)[:, np.newaxis], [1, width])
        x_t = tf.tile(tf.linspace(-1.0, 1.0, width)[np.newaxis, :], [height, 1])
        grid = tf.concat([tf.reshape(y_t, [-1])[:, np.newaxis], tf.reshape(x_t, [-1])[:, np.newaxis]], axis=1)
        return grid

    # FULL OUTPUT GRID
    if n_location_samples is None:
        out_yx_unit = _meshgrid(out_size[0], out_size[1]) # [out_size[0]*out_size[1], 2]
        out_y_unit = out_yx_unit[:, 0, np.newaxis]
        out_x_unit = out_yx_unit[:, 1, np.newaxis]
        out_y = ((out_y_unit+1.)/2.)*(tf.cast(out_size[0], 'float32')-1.) 
        out_x = ((out_x_unit+1.)/2.)*(tf.cast(out_size[1], 'float32')-1.) 

    # SAMPLED OUTPUT GRID
    else:
        out_dist_y = distributions.UniformDiscreteDistribution(interval=[n_location_samples, 0, tf.cast(out_size[0], 'float32')-1.])
        out_dist_x = distributions.UniformDiscreteDistribution(interval=[n_location_samples, 0, tf.cast(out_size[1], 'float32')-1.])
        out_y = out_dist_y.sample()
        out_x = out_dist_x.sample()
        out_y_unit = (out_y/(tf.cast(out_size[0], 'float32')-1.))*2.-1.
        out_x_unit = (out_x/(tf.cast(out_size[1], 'float32')-1.))*2.-1.
        out_yx_unit = tf.concat([out_y_unit, out_x_unit], axis=1) #[n_location_samples, 2]

    out_yx_unit_tiled = tf.tile(out_yx_unit[np.newaxis, :, :], [tf.shape(input_im)[0], 1, 1])
    expanded_input_yx_unit = pixel_transformation_clousure(out_yx_unit_tiled) #[input_im.shape[0], n_location_samples, 2]
    expanded_input_y_unit = expanded_input_yx_unit[:, :, 0]
    expanded_input_x_unit = expanded_input_yx_unit[:, :, 1]

    expanded_input_y_sample = (expanded_input_y_unit+1.)*(tf.cast(tf.shape(input_im)[1], 'float32')-1.)/2.
    expanded_input_x_sample = (expanded_input_x_unit+1.)*(tf.cast(tf.shape(input_im)[2], 'float32')-1.)/2.

    expanded_input_y_sample_flat = tf.reshape(expanded_input_y_sample, [-1])
    expanded_input_x_sample_flat = tf.reshape(expanded_input_x_sample, [-1])

    out_comparison_im_gathered = None

    # FULL OUTPUT GRID
    if n_location_samples is None: 
        input_transformed_flat, vis_input_transformed_flat, invalid_map = _interpolate(input_im, expanded_input_y_sample_flat, expanded_input_x_sample_flat, out_size[0]*out_size[1])
        output = tf.reshape(input_transformed_flat, [tf.shape(input_im)[0], out_size[0], out_size[1], tf.shape(input_im)[3]])
        vis_output = tf.reshape(vis_input_transformed_flat, [tf.shape(input_im)[0], out_size[0], out_size[1], tf.shape(input_im)[3]])
        if out_comparison_im is not None:
            expanded_out_y_flat_int =  tf.cast(tf.floor(tf.reshape(tf.tile(out_y[np.newaxis,:,0], [tf.shape(out_comparison_im)[0], 1]), [-1])), 'int32')
            expanded_out_x_flat_int =  tf.cast(tf.floor(tf.reshape(tf.tile(out_x[np.newaxis,:,0], [tf.shape(out_comparison_im)[0], 1]), [-1])), 'int32')
            out_comparison_im_gathered_flat = _gather_from_ims(out_comparison_im, [(expanded_out_y_flat_int, expanded_out_x_flat_int)], out_size[0]*out_size[1])[0] 
            out_comparison_im_gathered = tf.reshape(out_comparison_im_gathered_flat, [tf.shape(out_comparison_im)[0], out_size[0], out_size[1], 3])
        return output, vis_output, out_comparison_im_gathered, invalid_map, None

    # SAMPLED OUTPUT GRID
    else:
        input_transformed_flat_sampled, vis_input_transformed_flat_sampled, invalid_map = _interpolate(input_im, expanded_input_y_sample_flat, expanded_input_x_sample_flat, n_location_samples)
        output = tf.reshape(input_transformed_flat_sampled, [tf.shape(input_im)[0], -1, 3])
        vis_output = tf.reshape(vis_input_transformed_flat_sampled, [tf.shape(input_im)[0], -1, 3])
        if out_comparison_im is not None:
            expanded_out_y_flat_int =  tf.cast(tf.floor(tf.reshape(tf.tile(out_y[np.newaxis,:,0], [tf.shape(out_comparison_im)[0], 1]), [-1])), 'int32')
            expanded_out_x_flat_int =  tf.cast(tf.floor(tf.reshape(tf.tile(out_x[np.newaxis,:,0], [tf.shape(out_comparison_im)[0], 1]), [-1])), 'int32')
            out_comparison_im_gathered_flat = _gather_from_ims(out_comparison_im, [(expanded_out_y_flat_int, expanded_out_x_flat_int)], n_location_samples)[0] 
            out_comparison_im_gathered = tf.reshape(out_comparison_im_gathered_flat, [tf.shape(out_comparison_im)[0], -1, 3])
            out_yx_int = tf.cast(tf.concat([out_y, out_x], axis=1), 'int32')
            location_mask = tf.cast(tf.scatter_nd(out_yx_int, tf.ones((n_location_samples,), tf.float32), out_size), 'bool')
        return output, vis_output, out_comparison_im_gathered, invalid_map, location_mask



        















        # out_yx_int = tf.cast(tf.concat([out_y, out_x], axis=1), 'int32')
        # location_mask = tf.cast(tf.scatter_nd(out_yx_int, tf.ones((n_location_samples,), tf.float32), out_size), 'bool')










































# def transformer_sampled(input_im, pixel_transformation_clousure, out_size, n_location_samples, name='SpatialTransformerSampled', **kwargs):
    
#     def _gather_from_ims(im, yx_list, n_location_samples):
#         im_flat = tf.reshape(im, [-1, tf.shape(im)[3]])
#         base = tf.reshape(tf.tile((tf.range(tf.shape(im)[0])*(tf.shape(im)[2]*tf.shape(im)[1]))[:,np.newaxis], [1, n_location_samples]), [-1])
#         Ilist = []
#         for i in range(len(yx_list)):
#             y, x = yx_list[i]
#             flat_ind = base+y*tf.shape(im)[2]+x
#             Ilist.append(tf.gather(im_flat, flat_ind))
#         return Ilist

#     def _interpolate(im, y, x, n_location_samples):

#         # do sampling
#         y0 = tf.cast(tf.floor(y), 'int32')
#         y1 = y0 + 1
#         x0 = tf.cast(tf.floor(x), 'int32')
#         x1 = x0 + 1

#         y0 = tf.clip_by_value(y0, tf.zeros([], dtype='int32'), tf.shape(im)[1]-1)
#         y1 = tf.clip_by_value(y1, tf.zeros([], dtype='int32'), tf.shape(im)[1]-1)
#         x0 = tf.clip_by_value(x0, tf.zeros([], dtype='int32'), tf.shape(im)[2]-1)
#         x1 = tf.clip_by_value(x1, tf.zeros([], dtype='int32'), tf.shape(im)[2]-1)

#         Ia, Ib, Ic, Id = _gather_from_ims(im, [(y0, x0), (y1, x0), (y0, x1), (y1, x1)], n_location_samples)

#         # and finally calculate interpolated values
#         y0_f = tf.cast(y0, 'float32')
#         y1_f = tf.cast(y1, 'float32')
#         x0_f = tf.cast(x0, 'float32')
#         x1_f = tf.cast(x1, 'float32')
        
#         wa = ((x1_f-x)*(y1_f-y))[:, np.newaxis]
#         wb = ((x1_f-x)*(y-y0_f))[:, np.newaxis]
#         wc = ((x-x0_f)*(y1_f-y))[:, np.newaxis]
#         wd = ((x-x0_f)*(y-y0_f))[:, np.newaxis]

#         output = (wa*Ia)+(wb*Ib)+(wc*Ic)+(wd*Id)
#         return output

#     dist_y = distributions.UniformDiscreteDistribution(interval=[n_location_samples, 0, tf.cast(out_size[0], 'float32')-1.])
#     dist_x = distributions.UniformDiscreteDistribution(interval=[n_location_samples, 0, tf.cast(out_size[1], 'float32')-1.])
#     y_sample = dist_y.sample()
#     x_sample = dist_x.sample()
#     y_sample_unit = (y_sample/(tf.cast(out_size[0], 'float32')-1.))*2.-1.
#     x_sample_unit = (x_sample/(tf.cast(out_size[1], 'float32')-1.))*2.-1.

#     yx_sample = tf.concat([y_sample, x_sample], axis=1)
#     yx_sample_int = tf.cast(yx_sample, 'int32')
#     yx_sample_unit = tf.concat([y_sample_unit, x_sample_unit], axis=1)
#     transformed_expanded_yx_sample_unit = pixel_transformation_clousure(yx_sample_unit)
   
#     transformed_expanded_y_sample_unit = transformed_expanded_yx_sample_unit[:, 0, :]
#     transformed_expanded_x_sample_unit = transformed_expanded_yx_sample_unit[:, 1, :]
    
#     transformed_expanded_y_sample = (transformed_expanded_y_sample_unit+1.)*(tf.cast(tf.shape(input_im)[1], 'float32')-1.)/2.
#     transformed_expanded_x_sample = (transformed_expanded_x_sample_unit+1.)*(tf.cast(tf.shape(input_im)[2], 'float32')-1.)/2.

#     transformed_expanded_y_sample_flat = tf.reshape(transformed_expanded_y_sample, [-1])
#     transformed_expanded_x_sample_flat = tf.reshape(transformed_expanded_x_sample, [-1])

#     input_transformed_sampled = _interpolate(input_im, transformed_expanded_y_sample_flat, transformed_expanded_x_sample_flat, n_location_samples)
#     input_transformed_sampled_reshaped = tf.reshape(input_transformed_sampled, [tf.shape(input_im)[0], -1, 3])

#     location_mask = tf.cast(tf.scatter_nd(yx_sample_int, tf.ones((n_location_samples,), tf.float32), out_size), 'bool')

#     return input_transformed_sampled_reshaped, location_mask, yx_sample_int































# def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
#     """Batch Spatial Transformer Layer

#     Parameters
#     ----------

#     U : float
#         tensor of inputs [num_batch,height,width,num_channels]
#     thetas : float
#         a set of transformations for each input [num_batch,num_transforms,6]
#     out_size : int
#         the size of the output [out_height,out_width]

#     Returns: float
#         Tensor of size [num_batch*num_transforms,out_height,out_width,num_channels]
#     """
#     with tf.variable_scope(name):
#         num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
#         indices = [[i]*num_transforms for i in xrange(num_batch)]
#         input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
#         return transformer(input_repeated, thetas, out_size)# Copyright 2016 The TensorFlow Authors. All Rights Reserved.








        # base = tf.reshape(tf.tile((tf.range(tf.shape(im)[0])*(tf.shape(im)[2]*tf.shape(im)[1]))[:,np.newaxis], [1, n_location_samples]), [-1])
        # base_y0 = base + y0*tf.shape(im)[2]
        # base_y1 = base + y1*tf.shape(im)[2]

        # # use indices to lookup pixels in the flat image and restore
        # # channels dim
        # Ia = tf.gather(im_flat, base_y0 + x0)
        # Ib = tf.gather(im_flat, base_y1 + x0)
        # Ic = tf.gather(im_flat, base_y0 + x1)
        # Id = tf.gather(im_flat, base_y1 + x1)









