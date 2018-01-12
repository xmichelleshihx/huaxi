# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 19, 2016

author: jakeret
Modified on Sep 20, 2017
author: Cheng
Modified on Oct. 04, 2017
Modified on Oct. 09, 2017
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import pdb
import tensorflow as tf
from math import floor

def conv_bn_relu(x, W, keep_prob, padding = 'SAME', strides = [1,1,1,1,1], is_train = True):
#    pdb.set_trace()
    if padding == 'SAME':
        conv_3d = tf.nn.conv3d(x, W, strides=strides, padding = 'SAME')
    elif padding == 'SYMMETRIC': # to deal with boundary effect!
        k_shape = W.get_shape().as_list()
        pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [  floor(k_shape[2] / 2 ) , floor(k_shape[2] / 2)], [0, 0 ]] )
        pd_offset = tf.cast(pd_offset, tf.int32)
        x = tf.pad(x, pd_offset, 'SYMMETRIC' )
        conv_3d = tf.nn.conv3d(x, W, strides=strides, padding = 'VALID')
    conv_3d = tf.nn.dropout(conv_3d, keep_prob)
    bn = batch_norm(conv_3d, is_training = is_train)
    return tf.nn.relu(bn)

def conv_bn_relu2d(x, W, keep_prob, padding = 'SAME', strides = [1,1,1,1], is_train = True):
    if padding == 'SAME':
        conv_2d = tf.nn.conv2d(x, W, strides=strides, padding = 'SAME')
    elif padding == 'SYMMETRIC': # to deal with boundary effect!
        k_shape = W.get_shape().as_list()
        pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [0, 0 ]] )
        pd_offset = tf.cast(pd_offset, tf.int32)
        x = tf.pad(x, pd_offset, 'SYMMETRIC' )
        conv_2d = tf.nn.conv2d(x, W, strides=strides, padding = 'VALID')
    conv_2d = tf.nn.dropout(conv_2d, keep_prob)
    bn = batch_norm(conv_2d, is_training = is_train)
    return tf.nn.relu(bn)

def deconv_bn_relu(x, W, deconv_step = [1,2,2,2,1], is_train = True):
    innode_shape = x.get_shape().as_list()

    output_shape = [1, innode_shape[1]* deconv_step[1], innode_shape[2]* deconv_step[2], innode_shape[3]* deconv_step[3], innode_shape[4] ]
    deconv = tf.nn.conv3d_transpose(x, W, tf.stack(output_shape), strides=deconv_step, padding='SAME')
    #deconv = deconv.set_shape(output_shape)
    print("uncomment the deconv function when running in server")
    h_bn = batch_norm(deconv, is_training = is_train, decay = 0.90)
    b_relu = tf.nn.relu(h_bn)
    return b_relu

def deconv_bn_relu2d(x, W, batch_size = 1, deconv_step = [1,2,2,1], is_train = True):
    #pdb.set_trace()
    innode_shape = x.get_shape().as_list()
    batch_size_ts = tf.shape(x)[0]
    output_shape = [batch_size_ts, innode_shape[1]* deconv_step[1], innode_shape[2]* deconv_step[2], innode_shape[3] ]
    deconv = tf.nn.conv2d_transpose(x, W, tf.convert_to_tensor(output_shape), strides=deconv_step, padding='SAME')
    deconv = tf.reshape(deconv, [-1, innode_shape[1]* deconv_step[1], innode_shape[2]* deconv_step[2], innode_shape[3] ])
    print("uncomment the deconv function when running in server")
    h_bn = batch_norm(deconv, is_training = is_train)
    b_relu = tf.nn.relu(h_bn)
    return b_relu

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def weight_variable_deconv(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W,keep_prob_, padding = 'SAME'):
    if padding == 'SAME':
        conv_2d = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
    elif padding == 'SYMMETRIC':
        k_shape = W.get_shape().as_list()
        pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [0, 0 ]] )
        pd_offset = tf.cast(pd_offset, tf.int32)
        x = tf.pad(x, pd_offset, 'SYMMETRIC' )
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'VALID', name = "conv_" + name)
    return tf.nn.dropout(conv_2d, keep_prob_)

def conv3d(x, W,keep_prob_, padding = 'SAME', name = "default", strides = [1,1,1,1,1]):
    if padding == 'SAME':
        conv_3d = tf.nn.conv3d(x, W, strides=strides, padding = 'SAME')
    elif padding == 'SYMMETRIC': # to deal with boundary effect!
        k_shape = W.get_shape().as_list()
        pd_offset = tf.constant( [[0, 0], [  floor(k_shape[0] / 2 ) , floor(k_shape[0] / 2 )], [  floor(k_shape[1] / 2 ) , floor(k_shape[1] / 2)], [  floor(k_shape[2] / 2 ) , floor(k_shape[2] / 2)], [0, 0 ]] )
        pd_offset = tf.cast(pd_offset, tf.int32)
        x = tf.pad(x, pd_offset, 'SYMMETRIC' )
        conv_3d = tf.nn.conv3d(x, W, strides=strides, padding = 'VALID', name = "conv_" + name)
    return tf.nn.dropout(conv_3d, keep_prob_)

def batch_norm(x, is_training = True):
    return tf.contrib.layers.batch_norm(x, is_training = is_training, decay = 0.90, scale = True, center = True,\
                    updates_collections = None, variables_collections = ["internal_batchnorm_variables"], trainable = True)
'''
def deconv2d(x, W,stride):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')
'''
# deconv kernel size: dim1 dim2 dim3 out_channel, in_channel
#def deconv3d(x, W,stride):
#    x_shape = tf.shape(x)
#    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]*2, x_shape[4]])
#    return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding='SAME')

def deconv3d(x, W, padding = 'SAME', strides = [1, 2, 2, 2, 1]):
    """ features are used to ensure the shape of tensor since in tensorflow it is rather undefined"""
    x_shape = tf.shape(x)

    output_shape = tf.stack([x_shape[0], x_shape[1]*strides[1], x_shape[2]*strides[2], x_shape[3]*strides[3], x_shape[4]])
    #output_shape = [-1, x_shape[1]*2, x_shape[2]*2, x_shape[3]*2, x_shape[4]]
    return tf.nn.conv3d_transpose(x, W, output_shape, strides=strides, padding='SAME')

def deconv2d(x, W, padding = 'SAME', batch_size = 1, strides = [1, 2, 2, 1]):
    """ features are used to ensure the shape of tensor since in tensorflow it is rather undefined"""
    x_shape = tf.shape(x)

    output_shape = tf.stack([x_shape[0], x_shape[1]*strides[1], x_shape[2]*strides[2], x_shape[3]])
    #output_shape = [-1, x_shape[1]*2, x_shape[2]*2, x_shape[3]*2, x_shape[4]]

    return tf.nn.conv2d_transpose(x, W, output_shape, strides=strides, padding='SAME')

def deconv_conv3d(x, W ,out_channel, strides = [1,2,2,2,1], padding = 'SAME', name = "default"):
    x_shape = tf.shape(x)

    output_shape = tf.stack([x_shape[0], x_shape[1]*strides[1], x_shape[2]*strides[2], x_shape[3]*strides[3], out_channel])
    #output_shape = [-1, x_shape[1]*2, x_shape[2]*2, x_shape[3]*2, x_shape[4]]
    return tf.nn.conv3d_transpose(x, W, output_shape, strides=strides, padding='SAME', name = "deconv_conv_" + name)

def deconv_conv2d(x, W ,out_channel, strides = [1,2,2,1], padding = 'SAME'):
    x_shape = tf.shape(x)

    output_shape = tf.stack([x_shape[0], x_shape[1]*strides[1], x_shape[2]*strides[2],  out_channel])
    #output_shape = [-1, x_shape[1]*2, x_shape[2]*2, x_shape[3]*2, x_shape[4]]
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=strides, padding='SAME')


def max_pool2d(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def max_pool3d(x,n, name = "default"):
    return tf.nn.max_pool3d(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='SAME', name = "max_pool_" + name)

def crop_and_concat(x1,x2, name = "default"):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3, name = "concat_" + name)

def crop_and_concat3d(x1,x2, name = "default"):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2,  (x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 4, name = "crop_concat_" + name)

def simple_concat3d(x1,x2, name = "default"):
    """ concatenation without offset check"""
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    try:
        tf.equal(x1_shape[0:-2], x2_shape[0: -2])
    except:
        print("x1_shape: %s"%str(x1.get_shape().as_list()))
        print("x2_shape: %s"%str(x2.get_shape().as_list()))
        raise ValueError("Cannot concatenate tensors with different shape, igonoring feature map depth")
    return tf.concat([x1, x2], 4, name = "concat3d_" + name)

def simple_concat2d(x1,x2):
    """ concatenation without offset check"""
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    try:
        tf.equal(x1_shape[0:-2], x2_shape[0: -2])
    except:
        print("x1_shape: %s"%str(x1.get_shape().as_list()))
        print("x2_shape: %s"%str(x2.get_shape().as_list()))
        raise ValueError("Cannot concatenate tensors with different shape, igonoring feature map depth")
    return tf.concat([x1, x2], 3)

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1,  tf.shape(output_map)[3]]))
    return tf.clip_by_value( tf.div(exponential_map,tensor_sum_exp), -1.0 * 1e15, 1.0* 1e15, name = "pixel_softmax_2d")

def pixel_wise_softmax_3(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 4, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, 1, tf.shape(output_map)[4]]))
    return tf.clip_by_value( tf.div(exponential_map,tensor_sum_exp), -1.0 * 1e15, 1.0* 1e15, name = "pixel_softmax_3d")

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#     return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))
