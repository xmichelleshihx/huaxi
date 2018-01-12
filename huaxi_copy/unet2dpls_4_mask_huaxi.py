"""
3D Unet for multi_organ segmentation project
Running version on Sep. 27
Add loss record
Modified on Oct 03. implemented pixel loss mask for hard negative minning
Memory would be a big issue!
Assume that augmentation code will contain a mask in it
Will that make training extremely slow?

Masks are downsampled to reduce calculation
Multi_GPU is another consideration. 4 GPUs might be used, i.e. place the mask in GPU 3
Please be prepared for Multi_GPU debug

Modified to 2d version with input 256 256 3
Modified on Oct 09 for larger perception field
Modified on Oct 16 for validating the assumption of corner performance
Modified on Oct 31 for PUM dataset
"""
# TODO:
# Mask node
#
import os
import time
import shutil
import numpy as np
from collections import OrderedDict
import __future__
#from collections import DefaultDict
import logging
import pdb
import multiprocessing
import matplotlib
from glob import glob
#import filelock
matplotlib.use('Qt4Agg')
import tensorflow as tf
#import pymedimage.visualize as viz
from tf_unet import util
from tf_unet.util import get_batch_patches
from tf_unet.layers import (weight_variable, batch_norm, bias_variable, weight_variable_deconv,simple_concat2d,deconv_bn_relu2d,conv_bn_relu2d,deconv_conv2d,
                            conv2d, deconv2d, max_pool2d, max_pool3d, crop_and_concat3d, crop_and_concat, simple_concat3d, pixel_wise_softmax_2,
                            pixel_wise_softmax_3, cross_entropy)
'''
#import preproc_pipeline.npz2tf as tfio
import pymedimage.visualize as viz'''
from tensorflow.python import debug as tf_debug
from math import floor
#from tf_unet.train import contour_map
'''
contour_map = {
    "Background": 0,
    "Bowel": 1,
    "Duodenum": 2,
    "L-kidney": 3,
    "R-kidney": 4,
    "Spinal_Cord": 5,
    "Liver": 6,
    "Stomach":7
}

contour_map = { 'CTV1': 1,
               'Bladder': 2,
               'Bome_Marrow': 3,
               'Femoral_Head_L': 4,
               'Femoral_Head_R': 5,
               'Rectum': 6,
               'Small_Intestine': 7
               }

'''

contour_map = { 'background': 0,
               'target': 1
               }

verbose = True
view = True
logging.basicConfig(filename = "curr_log", level=logging.DEBUG, format='%(asctime)s %(message)s')
if verbose == True:
    logging.getLogger().addHandler(logging.StreamHandler())
raw_size = [512, 512, 3] # original raw input size
#mask_size = [128, 128]
#mask_scale = 3
volume_size = [512, 512, 3] # volume size after processing
label_size = [512, 512, 1]
decomp_feature = {
            'dsize_dim0': tf.FixedLenFeature([], tf.int64),
            'dsize_dim1': tf.FixedLenFeature([], tf.int64),
            'dsize_dim2': tf.FixedLenFeature([], tf.int64),
            'lsize_dim0': tf.FixedLenFeature([], tf.int64),
            'lsize_dim1': tf.FixedLenFeature([], tf.int64),
            'lsize_dim2': tf.FixedLenFeature([], tf.int64),
            'data_vol': tf.FixedLenFeature([], tf.string),
            'label_vol': tf.FixedLenFeature([], tf.string)}

#mask_feature = {
#            'dsize_dim0': tf.FixedLenFeature([], tf.int64),
#            'dsize_dim1': tf.FixedLenFeature([], tf.int64),
#            'dsize_dim2': tf.FixedLenFeature([], tf.int64),
#            'lsize_dim0': tf.FixedLenFeature([], tf.int64),
#            'lsize_dim1': tf.FixedLenFeature([], tf.int64),
#            'lsize_dim2': tf.FixedLenFeature([], tf.int64),
#            'mask_scale': tf.FixedLenFeature([], tf.int64),
#            'data_vol': tf.FixedLenFeature([], tf.string),
#            'mask_vol': tf.FixedLenFeature([], tf.string),
#            'label_vol': tf.FixedLenFeature([], tf.string)}



# feature for the mask
# To keep it in consistance with the 3d version, the data exchange formats are
# kept as the same, while the dim2 is fixed to 3

def create_conv_net(x, keep_prob, batch_size = 10, channels = 3, n_class = 8, layers=4, features_root=16, filter_size=3, pool_size=2, summaries=True, image_summeris = False, is_train = True):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param batch_size: batch_size
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    # Placeholder for the input image
    nx = volume_size[0]
    ny = volume_size[1]
    x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
    in_node = x_image
    features = features_root
    # storint network object
    stddev = 0.1
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    bottom_h_convs = OrderedDict()
    batch_size = batch_size
    up_h_convs = OrderedDict()
    weights = []
    convs = []
    output = []
    aux_deconvs = []
    aux_probs = []
    in_size = 128
    size = in_size

    # down layers
    #with tf.device("/gpu:1"): this is the server version.
    with tf.device("/gpu:0"):
        with tf.name_scope("block1"):
            # 256 256 3 -> 128 128 16
            #pdb.set_trace()
            in_node = x_image
            w1_1 = weight_variable( [filter_size, filter_size, channels, features], stddev  )
            block1_1 = conv_bn_relu2d(in_node, w1_1, keep_prob, is_train = is_train )
            w1_2 = weight_variable( [ filter_size, filter_size, features, features], stddev  )
            block1_2 = conv_bn_relu2d(block1_1, w1_2, keep_prob, is_train = is_train )
            convs.append( (block1_1, block1_2))
            max_pool_block1 = max_pool2d(  block1_2, 2  )
            weights.append(w1_1)
            weights.append(w1_2)
            print("block1 shape %s"%(str(max_pool_block1.get_shape().as_list())))

        with tf.name_scope("block2"):
            # 128 128 16 -> 64 64 32

            features *= 2
            in_node = max_pool_block1
            w2_1 = weight_variable( [ filter_size, filter_size, features // 2, features], stddev  )
            block2_1 = conv_bn_relu2d(in_node, w2_1, keep_prob,is_train = is_train  )
            w2_2 = weight_variable( [ filter_size, filter_size, features, features], stddev  )
            block2_2 = conv_bn_relu2d(block2_1, w2_2, keep_prob ,is_train = is_train )
            convs.append( (block2_1, block2_2) )
            max_pool_block2 = max_pool2d(  block2_2, 2  )
            weights.append(w2_1)
            weights.append(w2_2)
            print("block2 shape %s"%(str(max_pool_block2.get_shape().as_list())))

        with tf.name_scope("block3"):
            # 64 64 32 -> 16 16 64

            # now 128 -> 16
            features *= 2
            in_node = max_pool_block2
            w3_1 = weight_variable( [ 5, 5, features // 2, features], stddev  )
            block3_1 = conv_bn_relu2d(in_node, w3_1, keep_prob ,is_train = is_train )
            w3_2 = weight_variable( [5, 5, features, features], stddev  )
            block3_2 = conv_bn_relu2d(block3_1, w3_2, keep_prob, strides = [1, 2, 2,  1] ,is_train = is_train )
            convs.append((block3_1, block3_2))
            max_pool_block3 = max_pool2d( block3_2, 4  )
            weights.append(w3_1)
            weights.append(w3_2)
            print("block3 shape %s"%(str(max_pool_block3.get_shape().as_list())))

        with tf.name_scope("block4"):
            # 16 16 4 64 -> 4 4 2 128
            # now 16 -> 2
            features *= 2
            in_node = max_pool_block3
            w4_1 = weight_variable( [5, 5, features // 2, features], stddev  )
            block4_1 = conv_bn_relu2d(in_node, w4_1, keep_prob ,is_train = is_train )
            w4_2 = weight_variable( [ 5, 5, features, features], stddev  )
            block4_2 = conv_bn_relu2d(block4_1, w4_2, keep_prob, strides = [1, 2, 2, 1] ,is_train = is_train )
            convs.append( (block4_1, block4_2) )
            max_pool_block4 = max_pool2d(  block4_2, 4 )
            weights.append(w4_1)
            weights.append(w4_2)
            print("block4 shape %s"%(str(max_pool_block4.get_shape().as_list())))

        with tf.name_scope("bottom"):
            # 4 4 2 128 -> 4 4 2 256
            # features = 64
            in_node = max_pool_block4
            wb_1 = weight_variable([ 5, 5, features, features], stddev)
            blockb1 = conv_bn_relu2d(max_pool_block4, wb_1, keep_prob, padding = 'SYMMETRIC',is_train = is_train )
            wb_2 = weight_variable( [ 5, 5, features, features * 2], stddev  )
            blockb2 = conv_bn_relu2d(blockb1, wb_2, keep_prob, padding = 'SYMMETRIC',is_train = is_train )
            weights.append(wb_1)
            weights.append(wb_2)
            print("bottum shape: %s"%str(blockb2.get_shape().as_list()))

        with tf.name_scope("up_block1"):
            # features = 128
            # 4 4 2 256 -> 16 16 4 256 -> 16 16 4 128 + 64 -> 16 16 4 128
            #pdb.set_trace()
            in_node = blockb2
            w_dc_1 = weight_variable_deconv( [ pool_size, pool_size, features *2, features * 2]  )
            up_block1_deconv = deconv_bn_relu2d(in_node, w_dc_1, batch_size, deconv_step = [1,8,8,1],is_train = is_train )
            concat_in_node = simple_concat2d( up_block1_deconv, convs[-1][0]  )
            w_up_1_1 = weight_variable( [5, 5, features *3, features], stddev  )
            up_block1_1 = conv_bn_relu2d( concat_in_node, w_up_1_1, keep_prob ,is_train = is_train )
            w_up_1_2 = weight_variable( [ 5, 5, features, features], stddev  )
            up_block1_2 = conv_bn_relu2d( up_block1_1, w_up_1_2, keep_prob ,is_train = is_train )
            weights.append(w_dc_1)
            weights.append(w_up_1_1)
            weights.append(w_up_1_2)
            print("up_1 shape %s"%str(up_block1_2.get_shape().as_list()))

        with tf.name_scope("up_block2"):
            features /= 2
            features = int(features)
            # features = 64
            # 16 16 4 128 -> 64 64 8 64
            in_node = up_block1_2
            w_dc_2 = weight_variable_deconv( [ pool_size, pool_size, features *2, features * 2]  )
            up_block2_deconv = deconv_bn_relu2d(in_node, w_dc_2, batch_size, deconv_step = [1,8,8,1],is_train = is_train )
            concat_in_node = simple_concat2d( up_block2_deconv, convs[-2][0]  )
            w_up_2_1 = weight_variable( [5, 5, features *3, features], stddev  )
            up_block2_1 = conv_bn_relu2d( concat_in_node, w_up_2_1, keep_prob ,is_train = is_train )
            w_up_2_2 = weight_variable( [5, 5, features, features], stddev  )
            up_block2_2 = conv_bn_relu2d( up_block2_1, w_up_2_2, keep_prob ,is_train = is_train )
            weights.append(w_dc_2)
            weights.append(w_up_2_1)
            weights.append(w_up_2_2)
            print("up_2 shape %s"%str(up_block2_2.get_shape().as_list()))

        with tf.name_scope("up_block3"):
            features /= 2
            features = int(features)
            # features = 32
            # 64 64 8 64 -> 128 128 16 32
            in_node = up_block2_2
            w_dc_3 = weight_variable_deconv( [ pool_size, pool_size, features *2, features * 2]  )
            up_block3_deconv = deconv_bn_relu2d(in_node, w_dc_3, batch_size, deconv_step = [1,2,2,1],is_train = is_train )
            concat_in_node = simple_concat2d( up_block3_deconv, convs[-3][0]  )
            w_up_3_1 = weight_variable( [filter_size, filter_size, features *3, features], stddev  )
            up_block3_1 = conv_bn_relu2d( concat_in_node, w_up_3_1, keep_prob ,is_train = is_train )
            w_up_3_2 = weight_variable( [filter_size, filter_size, features, features], stddev  )
            up_block3_2 = conv_bn_relu2d( up_block3_1, w_up_3_2, keep_prob ,is_train = is_train )
            weights.append(w_dc_3)
            weights.append(w_up_3_1)
            weights.append(w_up_3_2)
            print("up_3 shape %s"%str(up_block3_2.get_shape().as_list()))

        with tf.name_scope("up_block4"):
            features /= 2
            features = int(features)
            # features = 16
            # 128 128 16 32 -> 256 256 32 16
            in_node = up_block3_2
            w_dc_4 = weight_variable_deconv( [ pool_size, pool_size, features *2, features * 2]  )
            up_block3_deconv = deconv_bn_relu2d(in_node, w_dc_4, batch_size, deconv_step = [1,2,2,1],is_train = is_train )
            concat_in_node = simple_concat2d( up_block3_deconv, convs[-4][0]  )
            w_up_4_1 = weight_variable( [ filter_size, filter_size, features *3, features], stddev  )
            up_block4_1 = conv_bn_relu2d( concat_in_node, w_up_4_1, keep_prob ,is_train = is_train )
            w_up_4_2 = weight_variable( [ filter_size, filter_size, features, features], stddev  )
            up_block4_2 = conv_bn_relu2d( up_block4_1, w_up_4_2, keep_prob ,is_train = is_train )
            weights.append(w_dc_4)
            weights.append(w_up_4_1)
            weights.append(w_up_4_2)
            print("up_4 shape %s"%str(up_block4_2.get_shape().as_list()))

    with tf.device("/gpu:0"):
        with tf.name_scope("aux_pth1"):
            features = features_root
            features = int(features)
            conv_up_1_2 = convs[-1][1]
            # 8
            w12 = weight_variable([ filter_size, filter_size, features * 8, features], stddev)
            wd_12_1 = weight_variable_deconv([pool_size, pool_size, features, features], stddev)
            wd_12_2 = weight_variable_deconv([pool_size, pool_size, features, features], stddev)
            wd_12_3 = weight_variable_deconv([pool_size, pool_size,  n_class, features], stddev)

            aux1_conv = conv2d(conv_up_1_2, w12, keep_prob)
            aux1_deconv_1 = deconv2d(aux1_conv, wd_12_1, strides = [1, 8, 8, 1] )
            innode_shape = aux1_conv.get_shape().as_list()
            out_shape = [ None, innode_shape[1]*8, innode_shape[2]*8, innode_shape[3]]
            aux1_deconv_1.set_shape(out_shape)
            # 64

            aux1_deconv_2 = deconv2d(aux1_deconv_1, wd_12_2, strides = [1, 4, 4,  1] )
            innode_shape = aux1_deconv_1.get_shape().as_list()
            out_shape = [ None, innode_shape[1]*4, innode_shape[2]*4, innode_shape[3]]
            aux1_deconv_2.set_shape(out_shape)
            # 256
            aux1_prob = deconv_conv2d(aux1_deconv_2, wd_12_3, n_class, strides = [1,2,2,1])

            innode_shape = aux1_deconv_2.get_shape().as_list()
            out_shape = [ None, innode_shape[1]*2, innode_shape[2]*2, n_class]
            aux1_prob.set_shape(out_shape)

            print("aux1 deconv shape %s"%str(out_shape))

            aux_deconvs.append(aux1_conv)
            aux_deconvs.append(aux1_deconv_1)
            aux_deconvs.append(aux1_deconv_2)

        with tf.name_scope("aux_pth2"):

            conv_up_2_2 = convs[-2][1]
            w22 = weight_variable([ filter_size, filter_size, features * 4, features], stddev)
            wd_22_1 = weight_variable_deconv([ pool_size, pool_size, features, features], stddev)
            wd_22_2 = weight_variable_deconv([ pool_size, pool_size,  n_class, features], stddev)
            # 64

            aux2_conv = conv2d(conv_up_2_2, w22, keep_prob)
            aux2_deconv_1 = deconv2d(aux2_conv, wd_22_1, n_class, strides = [1,4,4,1] )

            innode_shape = aux2_conv.get_shape().as_list()
            out_shape = [ None, innode_shape[1]*4, innode_shape[2]*4,  innode_shape[3]]
            aux2_deconv_1.set_shape(out_shape)
            aux2_prob = deconv_conv2d(aux2_deconv_1, wd_22_2, n_class, strides = [1,2,2,1])
        # pdb.set_trace()
            innode_shape = aux2_deconv_1.get_shape().as_list()
            out_shape = [ None, innode_shape[1]*2, innode_shape[2]*2,  n_class]
            aux2_prob.set_shape(out_shape)

            aux_deconvs.append(aux2_conv)
            aux_deconvs.append(aux2_deconv_1)

        with tf.name_scope("aux_pth3"):
            conv_up_3_2 = convs[-3][1]
            w32 = weight_variable([filter_size, filter_size, features * 2, features], stddev)
            wd_32_1 = weight_variable_deconv([ pool_size, pool_size, n_class, features], stddev)

            aux3_conv = conv2d(conv_up_3_2, w32, keep_prob)
            aux3_prob = deconv_conv2d(aux3_conv, wd_32_1, n_class, strides = [1,2,2,1])

            innode_shape = aux3_conv.get_shape().as_list()
            out_shape = [None, innode_shape[1]*2, innode_shape[2]*2,  n_class]
            aux3_prob.set_shape(out_shape)
            aux_deconvs.append(aux3_conv)

        aux_probs.append(aux1_prob)
        aux_probs.append(aux2_prob)
        aux_probs.append(aux3_prob)

        weights.append(w12)
        weights.append(wd_12_1)
        weights.append(wd_12_2)
        weights.append(wd_12_3)
        weights.append(w22)
        weights.append(wd_22_1)
        weights.append(wd_22_2)
        weights.append(w32)
        weights.append(wd_32_1)
    # Output Map
    with tf.device("/gpu:0"):
        with tf.name_scope("output"):
            in_node = up_block4_2
            weight = weight_variable([1, 1, features_root, n_class], stddev)
            conv = conv2d(in_node, weight, tf.constant(1.0))
    #    output_map = tf.nn.relu(conv + bias)
            output_map = conv
            
    if verbose:
        logging.debug("shape of output: %s"%(str(output_map.get_shape().as_list())))

#            tf.summary.image('summary_conv_%02d_02'%i, get_volume_summary(c2))
#
#        for k in pools.keys():
#            tf.summary.image('summary_pool_%02d'%k, get_volume_summary(pools[k]))
#
#        for k in deconv.keys():
#            tf.summary.image('summary_deconv_concat_%02d'%k, get_volume_summary(deconv[k]))
#
#    if summaries is True:
#        for k in dw_h_convs.keys():
#            tf.summary.histogram("dw_convolution_%02d"%k + '/activations', dw_h_convs[k])
#
#        for k in up_h_convs.keys():
#            tf.summary.histogram("up_convolution_%s"%k + '/output', up_h_convs[k])
#
#        for k in bottom_h_convs.keys():
#            tf.summary.histogram("bottom_convolution_%s"%k + '/activations', bottom_h_convs[k])


    variables = []
    for w in weights:
        variables.append(w)

    return aux1_prob, aux2_prob, aux3_prob ,output_map, variables, int(in_size - size)


class Unet(object):
    """
    A unet implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, channels=1, n_class=8, batch_size = 5, cost="cross_entropy", test_flag = False, cost_kwargs={}, **kwargs):
        tf.reset_default_graph()

        self.n_class = n_class
        self.batch_size = batch_size # coz the deconv 2d function sucks
        self.summaries = kwargs.get("summaries", True)

        self.x = tf.placeholder("float", shape=[None, volume_size[0], volume_size[1], channels])
        self.y = tf.placeholder("float", shape=[None, label_size[0], label_size[1], self.n_class])
        #self.mask_flag = mask_flag
        '''
        if mask_flag is True:
            self.mask_raw = tf.placeholder("float", shape=[None, label_size[0] // mask_scale, label_size[1] // mask_scale, 1]) # this is for the convenience of using tensorflow bilinear interpolation
            self.mask = tf.image.resize_bilinear(self.mask_raw, [label_size[0], label_size[1]])'''
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        self.is_train = tf.placeholder_with_default(True, shape = None, name = "is_train_flag")
        self.aux1_prob, self.aux2_prob, self.aux3_prob, logits, self.variables, self.offset = create_conv_net(self.x, self.keep_prob, self.batch_size, channels, n_class, is_train = self.is_train, **kwargs)
        self.predicter = pixel_wise_softmax_2(logits)
        self.compact_pred = tf.argmax(self.predicter, 3)
        self.compact_y = tf.argmax(self.y, 3)
#        self.correct_pred = tf.equal(self.compact_pred, self.compact_y)

        self.cost, self.regularizer_loss = self._get_cost(logits, cost, cost_kwargs)
        '''
        if mask_flag is True:
            self.upd_mask = self._update_mask_node()'''

        self.confusion_matrix = tf.confusion_matrix( tf.reshape(self.compact_y,[-1]), tf.reshape(self.compact_pred, [-1]), num_classes = self.n_class )
        ###########auxs nodes for assumption validation####################
        self.manual_pred = tf.placeholder("float", shape=[None, None, None])
        self.manual_gth = tf.placeholder("float", shape=[None, None, None])
        self.aux_confusion_matrix = tf.confusion_matrix( tf.reshape(self.manual_gth,[-1]), tf.reshape(self.manual_pred, [-1]), num_classes = self.n_class )
#        self.gradients_node = tf.gradients(self.cost, self.variables)

#        self.cross_entropy = tf.reduce_mean(cross_entropy(tf.reshape(self.y, [-1, n_class]),
#                                                          tf.reshape(pixel_wise_softmax_3(logits), [-1, n_class])))

        #print("test shape %s"%(str(tf.argmax(self.y, axis = 4))))
#        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """
 #       flat_logits = tf.reshape(logits, [-1, self.n_class])
 #       flat_labels = tf.reshape(self.y, [-1, self.n_class])
 #       weighted cross_entropy loss

        loss = 0
        dice_flag = cost_kwargs.pop("dice_flag", True)
        miu_dice = cost_kwargs.pop("miu_dice", None)
        miu_cross = cost_kwargs.pop("miu_cross", None)
        cross_flag = cost_kwargs.pop("cross_flag", False)
        aux_flag = cost_kwargs.pop("aux_flag", True)
        miu_aux1 = cost_kwargs.pop("miu_aux1", None)
        miu_aux2 = cost_kwargs.pop("miu_aux2", None)
        miu_aux3 = cost_kwargs.pop("miu_aux3", None)
        '''
        if self.mask_flag is True:
            mask_flag = cost_kwargs.pop("mask_flag", False)'''
        lbd_fp = cost_kwargs.pop("lbd_fp", 1.0)
        lbd_p = cost_kwargs.pop("lbd_p", 1.0)

        if cross_flag is True:
            '''
            if self.mask_flag is True:
                self.weighted_loss = miu_cross * self._softmax_weighted_loss_with_fpmask(logits, self.y, self.n_class,mask = self.mask, lbd_fp = lbd_fp , lbd_p = lbd_p )
            else:
                self.weighted_loss = miu_cross * self._softmax_weighted_loss_with_fpmask(logits, self.y, self.n_class, lbd_fp = lbd_fp , lbd_p = lbd_p )
            '''
            self.weighted_loss = miu_cross * self._softmax_weighted_loss_with_fpmask(logits, self.y, self.n_class, lbd_fp = lbd_fp , lbd_p = lbd_p )
            loss += self.weighted_loss
            if aux_flag is True:
                self.aux1_weighted = self._softmax_weighted_loss_with_fpmask(self.aux1_prob, self.y, self.n_class , lbd_fp = lbd_fp , lbd_p = lbd_p )
                self.aux2_weighted = self._softmax_weighted_loss_with_fpmask(self.aux2_prob, self.y, self.n_class , lbd_fp = lbd_fp , lbd_p = lbd_p )
                self.aux3_weighted = self._softmax_weighted_loss_with_fpmask(self.aux3_prob, self.y, self.n_class , lbd_fp = lbd_fp , lbd_p = lbd_p )
                loss += miu_aux1 * miu_cross * self.aux1_weighted
                loss += miu_aux2 * miu_cross * self.aux2_weighted
                loss += miu_aux3 * miu_cross * self.aux3_weighted

        if dice_flag is True:
            ''''
            if self.mask_flag is True:
                raw_dice = self._dice_loss_fun(logits, self.y, self.n_class, mask = self.mask, lbd_fp = lbd_fp)
            else:
                raw_dice = self._dice_loss_fun(logits, self.y, self.n_class)'''
            raw_dice = self._dice_loss_fun(logits, self.y, self.n_class)
            self.dice_loss = miu_dice * raw_dice
            self.dice_eval = self._dice_eval( self.compact_pred, self.y, self.n_class  )

            loss += self.dice_loss
            if aux_flag is True:
                self.aux_1_dice = self._dice_loss_fun(self.aux1_prob, self.y, self.n_class, lbd_fp = lbd_fp )
                self.aux_2_dice = self._dice_loss_fun(self.aux2_prob, self.y, self.n_class , lbd_fp = lbd_fp)
                self.aux_3_dice = self._dice_loss_fun(self.aux3_prob, self.y, self.n_class , lbd_fp = lbd_fp)
                loss += miu_dice * miu_aux1 * self.aux_1_dice
                loss += miu_dice * miu_aux2 * self.aux_2_dice
                loss += miu_dice * miu_aux3 * self.aux_3_dice

        reg_coeff = cost_kwargs.pop("regularizer", 1.0e-4)
        regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])

        return loss, reg_coeff * regularizers

    def _softmax_weighted_loss_with_fpmask(self, logits, labels, num_cls = 8, mask = None, lbd_p = 1., lbd_fp = 1.):
        """
        Manually suppress false positive samples
        Loss = weighted * -target*log(softmax(logits))
        :param logits: probability score
        :param lbd_p, lbd_fp: lambda (weighting for positve and negative samples)
        :param labels: ground_truth
        :return: softmax-weifhted loss
        : That might run very very slow
        Note: this only works for binary classification!!!!!!!!!!!!!!!
        """
        gt = labels
        pred = logits
        softmaxpred = tf.nn.softmax(pred)
        loss = 0
        raw_loss = 0
        for i in range(num_cls):
            gti = gt[:,:,:,i]
            predi = softmaxpred[:,:,:,i]
            weighted = 1-(tf.reduce_sum(gti) / tf.reduce_sum(gt))
            if i == 0:
                if mask is None:
                    raw_loss += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1)) * lbd_fp
                else:
                    raw_loss += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))  * mask[...,0] * lbd_fp
            else:

#                fp_mask = tf.cast(tf.greater( predi, 0.5  ), dtype = tf.float32) * tf.cast( tf.less( gti, 0.5  ), dtype = tf.float32)
#                fp_mask *= lbd_fp
#                fp_mask += 1
#
                if mask is None:
                    raw_loss += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
                else:
                    raw_loss += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1)) * mask[..., 0]

        loss = tf.reduce_mean(raw_loss)
        return loss

    def _dice_loss_fun(self, pred, labels, num_cls = 8, mask = None, lbd_fp = None):
        """
        For the masked version, my understanding is to give more attention to masked pixels and care about if they are correctly labeled. For example, consider and extreme where mask-> inf,
        In this case only pixels under the mask will be taken into consideration
        Params:
            lbd_fp: if not None, a mask used for supressing false positive will be added
        Note: this only works for binary classification!!!!!!!!!!!!!!!
        """
        dice = 0
        mean_act_dice = 0
        pred = tf.nn.softmax(pred)
        for i in range(num_cls):
            if mask is None:
                inse = tf.reduce_sum(pred[ :, :, :, i]*labels[:, :, :, i]  )
                l = tf.reduce_sum(pred[:, :, :, i]*pred[:, :, :, i] ) + 0.0000001
                r = tf.reduce_sum(labels[:, :, :, i] * labels[:, :, :, i] )

            else:
                inse = tf.reduce_sum(pred[:, :, :, i]*labels[:, :, :, i] * mask[..., 0] )
                l = tf.reduce_sum(pred[:, :, :, i]*pred[:, :, :, i] * mask[..., 0] ) + 0.0000001
                r = tf.reduce_sum(labels[:, :, :, i] * labels[:, :, :, i] * mask[..., 0] )

            dice = dice + 2.0 * inse/(l+r)

        return -1.0 * dice  / (num_cls)

    def _dice_eval(self, compact_pred, labels, num_cls = 8):
        """
        calculate standard dice for evaluation
        """
        dice = 0
        pred = tf.one_hot( compact_pred, depth = num_cls, axis = -1  )
        for i in range(num_cls):
            inse = tf.reduce_sum(pred[ :, :, :, i]*labels[:, :, :, i]  )
            l = tf.reduce_sum(pred[:, :, :, i]*pred[:, :, :, i] ) + 0.0000001
            r = tf.reduce_sum(labels[:, :, :, i] * labels[:, :, :, i] )
            dice = dice + 2.0 * inse/(l+r)

        return 1.0 * dice  / (num_cls)


    def save(self, sess, model_path, global_step = 0):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path, global_step = global_step)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        :param new_LR: change this if we want to use the new learning rate
        """

        saver = tf.train.Saver(tf.contrib.framework.get_variables() + tf.get_collection_ref("internal_batchnorm_variables") )
        logging.info("Model restored from file: %s" % model_path)
        try:
            saver.restore(sess, model_path)
            logging.info("Model restored from file: %s" % model_path)
        except:
            variables = tf.global_variables()
            reader = tf.pywrap_tensorflow.NewCheckpointReader(model_path)
            var_keep_dic = reader.get_variable_to_shape_map()
            variables_to_restore = []
            for v in variables:
                if v.name.split(':')[0] in var_keep_dic:
                    variables_to_restore.append(v)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, model_path)

            logging.info("Model restored from file: %s with relaxation" % model_path)
            logging.info("Restored variables: ")
            for vname in var_keep_dic.keys():
                logging.info(str(vname))

class Trainer(object):
    """
    Trains a unet instance

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    aparam opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """

    prediction_path = "prediction"
    verification_batch_size = 10

    # mask_dict is obsotele
    def __init__(self, net, train_dir, label_dir, val_dir, test_dir, mask_folder = None, num_cls = 8, batch_size=1, optimizer="momentum", \
                 opt_kwargs={}, num_epochs = 100, checkpoint_space = 500, lr_update_flag = False, test_list = None):
        self.net = net
        self.batch_size = batch_size
        self.num_cls = num_cls
        self.checkpoint_space = checkpoint_space
        self.opt_kwargs = opt_kwargs
        self.optimizer = optimizer
        self.train_dir  = train_dir
        self.label_dir = label_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.patch_width = 512
        self.patch_height = 512
        self.patch_depth = 3
#        self.train_list = train_list
#        self.val_list =val_list
#        self.test_list = test_list
#        self.train_queue = tf.train.string_input_producer(train_list, num_epochs = None, shuffle = True)
#        self.val_queue = tf.train.string_input_producer(val_list, num_epochs = None, shuffle = True)
        
        self.dice = tf.Variable( -1 * np.ones( self.num_cls))
        self.jaccard = tf.Variable( -1 * np.ones( self.num_cls))
        self.loss_dict = {}
        self._write_pool = multiprocessing.Pool(processes = 4)
        self.lr_update_flag = lr_update_flag
        '''
        if self.net.mask_flag is True:
            self.mask_manager = mask_manager(mask_folder, [batch_size, label_size[0] // mask_scale, label_size[1] // mask_scale, 1])'''

    def _label_decomp(self, label_vol):
        """decompose label for softmax classifier """
        _batch_shape = list(label_vol.shape)
        _vol = np.zeros(_batch_shape)
        _vol[label_vol == 0] = 1
        _vol = _vol[..., np.newaxis]
        for i in range(self.num_cls):
            if i == 0:
                continue
            _n_slice = np.zeros(label_vol.shape)
            _n_slice[label_vol == i] = 1
            _vol = np.concatenate( (_vol, _n_slice[..., np.newaxis]), axis = 3 )
        return np.float32(_vol)

    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
#            learning_rate = self.opt_kwargs.pop("learning_rate", 2.0)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=training_iters,
                                                        decay_rate=decay_rate,
                                                        staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost +\
                                                                    self.net.regularizer_loss,
                                                                    global_step=global_step)
        elif self.optimizer == "adam":
            #pdb.set_trace()
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.0002)
            self.LR_refresh = learning_rate
            # this is ugly but it is the only thing I can do without changing
            # the structure of this code
            self.learning_rate_node = tf.Variable(learning_rate)
            # added for using batch_normalization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                                                **self.opt_kwargs).minimize(self.net.cost + \
                                                                self.net.regularizer_loss,
                                                                global_step=global_step)

        return optimizer

    def _initialize(self, training_iters, output_path,  restore):
        self.global_step = tf.Variable(0)
#        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]))
        scalar_summaries = []
        scalar_summaries.append(tf.summary.scalar('loss', self.net.cost))
        scalar_summaries.append(tf.summary.scalar('regularizer_loss', self.net.regularizer_loss))
        scalar_summaries.append(tf.summary.scalar('weighted_loss', self.net.weighted_loss))
        scalar_summaries.append(tf.summary.scalar('dice_loss', self.net.dice_loss))
        scalar_summaries.append(tf.summary.scalar('dice_eval', self.net.dice_eval))
        # also, save a compact output prediction
        train_images = []
        val_images = []
        train_images.append(tf.summary.image('summary_pred', tf.expand_dims(tf.cast(self.net.compact_pred, tf.float32), 3 )) )
        train_images.append(tf.summary.image('image', tf.expand_dims(tf.cast(self.net.x[:,:,:,1], tf.float32), 3 )) )
        val_images.append(tf.summary.image('val_pred', tf.expand_dims(tf.cast(self.net.compact_pred, tf.float32), 3)))
        val_images.append(tf.summary.image('image', tf.expand_dims(tf.cast(self.net.x[:,:,:,1], tf.float32), 3)))
        train_images.append(tf.summary.image('GND', tf.expand_dims(tf.cast(self.net.compact_y, tf.float32), 3)))
        val_images.append(tf.summary.image('validation_GND', tf.expand_dims(tf.cast(self.net.compact_y, tf.float32), 3)))
        
        '''
        if self.net.mask_flag is True:
            train_images.append(tf.summary.image('mask', tf.cast(self.net.mask, tf.float32)))
        '''
        #end
        self.optimizer = self._get_optimizer(training_iters, self.global_step)
        scalar_summaries.append(tf.summary.scalar('learning_rate', self.learning_rate_node))
        
        self.scalar_summary_op = tf.summary.merge(scalar_summaries)
        self.train_image_summary_op = tf.summary.merge(train_images)
        self.val_image_summary_op = tf.summary.merge(val_images)
        init_glb = tf.global_variables_initializer()
        init_loc = tf.variables_initializer(tf.local_variables())
        
        prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)
        
        if not restore:
            logging.info("Removing '{:}'".format(prediction_path))
            shutil.rmtree(prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(prediction_path):
            logging.info("Allocating '{:}'".format(prediction_path))
            os.makedirs(prediction_path)
        
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
        '''
        try:
            os.makedirs(self.mask_manager.mask_folder)
        except:
            pass
            '''
        return init_glb, init_loc

    def _tf_multicls_jaccard(self):
        _oplist = []
        for ii in range(self.num_cls):
            _gth_vol = tf.equal(self.net.compact_y, ii)
            _pred_vol = tf.equal(self.net.compact_pred, ii)
            _intersec = tf.reduce_sum( tf.cast( tf.logical_and(_gth_vol, _pred_vol), tf.float64) )
            _union = tf.reduce_sum( tf.cast( tf.logical_or(_gth_vol, _pred_vol), tf.float64 ) ) + 10e-7
            _oplist.append( _intersec / _union)
        self.jaccard = tf.stack(_oplist)
        self.actual_mean_jaccard = tf.reduce_mean( tf.slice(self.jaccard, [1], [self.num_cls -1])  )
        self.mean_jaccard = tf.reduce_mean(self.jaccard)

    def _tf_multicls_dice(self):
        _oplist = []
        for ii in range(self.num_cls):
            _gth_vol = tf.equal(self.net.compact_y, ii)
            _pred_vol = tf.equal(self.net.compact_pred, ii)
            _intersec = tf.reduce_sum( tf.cast( tf.logical_and(_gth_vol, _pred_vol), tf.float64) )
            _all = tf.reduce_sum( tf.cast(_pred_vol, tf.float64)) + tf.reduce_sum( tf.cast( _gth_vol, tf.float64  )  ) + 10e-7
            _oplist.append( _intersec / _all)
        self.dice = tf.stack(_oplist)
        self.actual_mean_dice = tf.reduce_mean( tf.slice(self.dice, [1], [self.num_cls -1])  )
        self.mean_dice = tf.reduce_mean(self.dice)

    def _test_ppl(self):
        """ test input pipeline!!!"""
        with tf.Session() as sess:

            init_glb = tf.global_variables_initializer()
            init_loc = tf.local_variables_initializer()
            sess.run([init_glb, init_loc])
            coord = tf.train.Coordinator()
            print("queue runner finished!")

            feed_all = self.next_batch(self.train_queue)
            print("start running session!")
            threads = tf.train.start_queue_runners(sess = sess, coord = coord, start = True)
            while True:
                batch = feed_all.eval()
                batch_x = batch[0,:,:,:,0]
                batch_y = batch[0,:,:,:,1]
#                viz.double_viewer(batch_x.T, batch_y.T)
#
#            while True:
#                batch_x = feed_x.eval()
#                batch_y = feed_y.eval()
#                batch_x = batch_x[0,:,:,:]
#                batch_y = batch_y[0,:,:,:]
#                viz.double_viewer(batch_x.T, batch_y.T)
            print("finished running session!")
            coord.request_stop()
            coord.join(threads)

    def train(self, output_path, restored_path = None, training_iters=100, epochs=100, dropout=0.75, display_step=20, restore=False):

        """
        Lauches the training process

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param restored_path: path where checkpoints are read from
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param show_gradient: Flag if explicitly calculate gradients
        """
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path
        init_glb, init_loc = self._initialize(training_iters, output_path, restore)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        with tf.Session(config=config) as sess:
#            pdb.set_trace()
            sess.run([ init_glb, init_loc] )
#            print(trainables)
            coord = tf.train.Coordinator()
            if restore:
                if restored_path is None:
                    raise Exception("No restore path is provided")
#                ckpt = tf.train.get_checkpoint_state(output_path)
                try:
                    ckpt = tf.train.get_checkpoint_state(restored_path)
                    if ckpt and ckpt.model_checkpoint_path:
                        self.net.restore(sess, ckpt.model_checkpoint_path)
                except:
                    print("Unable to restore, start from beginning")


            if self.lr_update_flag is True:
                _new_LR = self.opt_kwargs.pop("learning_rate",0.001)
                sess.run( tf.assign(self.learning_rate_node, self.LR_refresh)  )
                logging.info("New learning rate %s has been loaded"%str(_new_LR))

            train_summary_writer = tf.summary.FileWriter(output_path + "/train_log", graph=sess.graph)
            val_summary_writer = tf.summary.FileWriter(output_path + "/val_log", graph=sess.graph)
#            feed_all, feed_fid = self.next_batch(self.train_queue)
#            feed_all, feed_fid = self.next_batch(self.train_queue)
#            feed_val, feed_val_fid = self.next_batch(self.val_queue)
#            feed_all_ts = self.next_batch(self.val_queue)
            # load all volume files
            #load data
            train_data_dir_list = glob('{}/*.nii'.format(self.train_dir))
            train_label_dir_list = glob('{}/*.nii'.format(self.label_dir))
            train_data_dir_list.sort()
            train_label_dir_list.sort()
            #img_clec, label_clec = load_data_pairs(pair_list, self.resize_r, self.rename_map)
            
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            for epoch in range(epochs):
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    logging.info("Running step %s epoch %s ..."%(str(step), str(epoch)))
                    #pdb.set_trace()
                    start = time.time()
                    #batch, fid = sess.run([feed_all, feed_fid])
                    #batch label
                    rand_idx = np.arange(len(train_data_dir_list))
                    np.random.shuffle(rand_idx)
                    batch,label = get_batch_patches(train_data_dir_list[rand_idx[0]],train_label_dir_list[rand_idx[0]],self.batch_size,self.patch_width,self.patch_height,self.patch_depth)
                    # end of modification
                    batch_x = batch[:,:,:,0:3]
                    raw_y = label[:,:,:,1]#meidium slice as groundtruth
                        #pdb.set_trace()
                    batch_y = self._label_decomp(raw_y)
                    #fids = [ _single.decode('utf-8').split(":")[0] for _single in fid ]
#                    if self.net.mask_flag is True:
#                        raw_mask = self.mask_manager.mask_lookup(fids)
                    if verbose:
                        logging.info("Data for step %s epoch %s has been read"%(str(step), str(epoch)))
                    #pdb.set_trace()
                    '''
                    if self.net.mask_flag is True:
                        print("Warning, I manually disabled mask update, change weighting in update operation")
                        print("Warning, I modified dice loss")
#                        _, loss, lr, new_mask = sess.run((self.optimizer, self.net.cost, self.learning_rate_node, self.net.upd_mask),
#                                                      feed_dict={self.net.x: batch_x,
#                                                                 self.net.y: batch_y,
#                                                                 self.net.mask_raw: raw_mask,
#                                                                 self.net.keep_prob: dropout})
#    
                        _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate_node, self.net.upd_mask),
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: batch_y,
                                                                 self.net.keep_prob: dropout})

                    else:'''
                    _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate_node),
                                                  feed_dict={self.net.x: batch_x,
                                                             self.net.y: batch_y,
                                                             self.net.keep_prob: dropout})

                    logging.info("Training step %s epoch %s has been finished!"%(str(step), str(epoch)))
                    # write the updated mask
                    #pdb.set_trace()
                    logging.info("Time elapsed %s seconds"%(str(time.time() - start)))
                    #logging.info("Mean mask value %s"%str(np.mean(new_mask)))
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, train_summary_writer, step, batch_x, batch_y, raw_y)

                    if step % (display_step * 1) == 0:
                        #val_batch val_label
                        rand_idx = np.arange(len(train_data_dir_list))
                        np.random.shuffle(rand_idx)
                        val_batch,val_label = get_batch_patches(train_data_dir_list[rand_idx[0]],train_label_dir_list[rand_idx[0]],self.batch_size,self.patch_width,self.patch_height,self.patch_depth)
                        # end of modification
                        val_batch_x = val_batch[:,:,:,0:3]
                        val_raw_y = val_label[:,:,:,1]#meidium slice as groundtruth
                            #pdb.set_trace()
                        val_batch_y = self._label_decomp(val_raw_y)
                        
                        val_x = val_batch_x[:,:,:,0:3]
                        val_y = val_batch_y[:,:,:,1]
                        val_x = val_x
                        val_y = self._label_decomp(val_y)
                        '''detail_flag = False
                        if step % (display_step * 4) == 0:
                            detail_flag = True
                        self.val_stats(sess, val_summary_writer, step, val_x, val_y, detail = detail_flag)'''

                    if step % (self.checkpoint_space) == 0:
                        if step == 0:
                            pass
                        else:
                            save_path = self.net.save(sess, save_path, global_step = self.global_step.eval())
                            # then reduce the learning rate
                            logging.info("Now restore the model to avoid memory fragment by re-allocation")

                            last_ckpt = tf.train.get_checkpoint_state(output_path)
                            if last_ckpt and last_ckpt.model_checkpoint_path:
                                self.net.restore(sess, last_ckpt.model_checkpoint_path)
                            logging.info("Model has been restored for re-allocation")
                            _pre_lr = sess.run(self.learning_rate_node)
                            sess.run( tf.assign(self.learning_rate_node, _pre_lr * 0.9 )  )

                logging.info("Global step %s"%str(self.global_step.eval()))
            logging.info("Optimization Finished!")
            coord.request_stop()
            coord.join(threads)
            return save_path

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y, compact_y = None, mask = None):
        # Calculate batch loss and accuracy
        #pdb.set_trace()
        """
        compact_y is the compact label of y. Since it has already been calculated, for sake of time consumption, its
        reasonable to directly feed that in.
        """
        #pdb.set_trace()
        if mask is None:
            summary_str, summary_img, loss= sess.run([\
                                                self.scalar_summary_op,
                                                self.train_image_summary_op,
                                                self.net.cost],
                                                feed_dict={self.net.x: batch_x,
                                                self.net.y: batch_y,
                                                self.net.is_train: False,
                                                self.net.keep_prob: 1.})
        else:
            summary_str, summary_img, loss= sess.run([\
                                                self.scalar_summary_op,
                                                self.train_image_summary_op,
                                                self.net.cost],
                                                feed_dict={self.net.x: batch_x,
                                                self.net.y: batch_y,
                                                self.net.is_train: False,
                                                self.net.mask_raw: mask[..., np.newaxis],
                                                self.net.keep_prob: 1.})


        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(summary_img, step)
        summary_writer.flush()

###########################UNDER CONSTRUCTION####################
    def _indicator_eval(self, cm, verbose = True):
        """Decompose confusion matrix and get statistics
        """
        my_dice = _dice(cm)
        my_jaccard = _jaccard(cm)
        print(cm)
        #pdb.set_trace()
        for organ, ind in contour_map.items():
            print( "organ: %s"%organ  )
            print( "dice: %s"%( my_dice[int(ind)]  )  )
            print( "jaccard: %s"%( my_jaccard[int(ind)]  )  )
        return my_dice, my_jaccard

    def test_eval(self, sess, output_path, ctr_bias = 64, save = True):
        cm = np.zeros([self.num_cls, self.num_cls])
        _num_sample = len(self.test_list)
        pred_dict = {}
        #pdb.set_trace()
        for idx_file,fid in enumerate(self.test_list):
            if save is True:
                out_bname = "prediction" + os.path.basename(fid)
                output_folder = os.path.join(output_path, "npzresults")
                output_folder = os.path.join(output_folder, out_bname)
                try:
                    os.makedirs(output_folder)
                except:
                    logging.info("npz results folder exist")

            if not os.path.isfile(fid):
                raise Exception("cannot find sample %s"%str(fid))
            _npz_dict = np.load(fid)
            raw_x = _npz_dict['arr_0']
            raw_y = _npz_dict['arr_1']
#            pdb.set_trace()
#            pdb.set_trace()
            if save is True:
                # save the result
                out_x = raw_x.copy()
                tmp_y = np.zeros(raw_y.shape)

            frame_list = [kk for kk in range(raw_x.shape[2]) if kk != 0]
            logging.info("Warning! I have filpped the image!")
            del frame_list[-1]
            # manually choose batches
            np.random.shuffle(frame_list)
            for ii in range( floor( raw_x.shape[2] // self.net.batch_size  )  ):
                vol_x = np.zeros( [self.net.batch_size, raw_size[0], raw_size[1], raw_size[2]]  )
                slice_y = np.zeros( [self.net.batch_size, label_size[0], label_size[1]]  )
                for idx, jj in enumerate(frame_list[ ii * self.net.batch_size : (ii + 1) * self.net.batch_size  ]):

                    vol_x[idx, ...] = np.flip(raw_x[ctr_bias: -ctr_bias, ctr_bias: -ctr_bias , jj -1: jj+2  ].copy(), axis = 1)
                    slice_y[idx,...] = np.flip(raw_y[ctr_bias: -ctr_bias, ctr_bias: -ctr_bias, jj ].copy(), axis = 1)
#
                    vol_x[idx, ...] = np.flip( vol_x[idx, ...], axis = 0  )
                    slice_y[idx,...] = np.flip( slice_y[idx, ...], axis = 0  )
#

                vol_y = self._label_decomp(slice_y)
                pred, curr_conf_mat= sess.run([self.net.compact_pred, self.net.confusion_matrix], feed_dict =\
                                              {self.net.x: vol_x, self.net.y: vol_y, self.net.keep_prob: 1.0, self.net.is_train: False})

                if save is True:
                    for idx, jj in enumerate(frame_list[ ii * self.net.batch_size : (ii + 1) * self.net.batch_size  ]):
                        tmp_y[ctr_bias: -ctr_bias, ctr_bias: -ctr_bias, jj ] = np.flip(pred[idx, ... ].copy(), axis = 0)
                        tmp_y[..., jj] = np.flip(tmp_y[..., jj].copy(), axis = 1)
                logging.info(" part %s of %s of sample %s has been processed  "%(str(ii),\
                                        str( floor( raw_x.shape[2] // self.net.batch_size)  ), str(idx_file)   ) )

                cm += curr_conf_mat
#            viz.triple_viewer( out_x.T, tmp_y.T, raw_y.T  )

            self._save_npz_rediction(out_x, tmp_y, output_folder, out_bname)

            logging.info("%s of %s sample has been processed!"%(str(idx_file), str(_num_sample)))

        my_dice = _dice(cm)
        my_jaccard = _jaccard(cm)
        print(cm)
        #pdb.set_trace()
        for organ, ind in contour_map.items():
            print( "organ: %s"%organ  )
            print( "dice: %s"%( my_dice[int(ind)]  )  )
            print( "jaccard: %s"%( my_jaccard[int(ind)]  )  )

#        my_dice, my_jaccard = self._indicator_eval(cm)
        eval_fid = os.path.join(output_path, "result.txt")
        with open(eval_fid,'w') as fopen:
            for organ, ind in contour_map.items():
                fopen.write( "organ: %s \n"%organ  )
                fopen.write( "======================================= \n"  )
                fopen.write( "dice: %s \n"%( my_dice[int(ind)]  )  )
                fopen.write( "jaccard: %s \n"%( my_jaccard[int(ind)]  )  )
                fopen.write( "Confusion matrix: "  )
            fopen.close()

    def _save_npz_rediction(self, vol_x, comp_pred, out_folder, out_bname):
        """
        save prediction to npz file
        """
        decomp_pred = self._label_decomp(comp_pred)
        np.savez(os.path.join(out_folder, out_bname))
        for ii in range(1, decomp_pred.shape[-1]):
            _lb_name = _inverse_lookup(contour_map, ii) + "_" +out_bname
            np.savez( os.path.join(out_folder, _lb_name), decomp_pred[..., ii] )

        logging.info(out_folder + "has been saved!")

    def test(self, output_path, restored_path):
        """
        Lauches the test process

        :param output_path: path where to store checkpoints
        :param restored_path: path where checkpoints are read from
        """
        save_path = os.path.join(output_path, "model.cpkt")
        init_glb, init_loc = self._initialize(1, output_path, True)

        with tf.Session() as sess:
#           $pdb.set_trace()
            sess.run([ init_glb, init_loc] )
            ckpt = tf.train.get_checkpoint_state(restored_path)
            self.net.restore(sess, ckpt.model_checkpoint_path)
            logging.info("model has been loaded!")
            self.test_eval(sess, output_path)
            logging.info("testing finished")

def get_volume_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, 0, idx), (1, -1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    img_d = tf.shape(img)[3]
    V = tf.reshape(V, tf.stack((img_w, img_h, img_d, 1)))
    V = tf.transpose(V, (3, 0, 1, 2))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, img_d, 1)))
    return V

def get_output_summary(vol, idx=0, slice_position = 0.5):
    """
    Make a profile of training prediction
    """
    map_size = vol.get_shape().as_list()
    slice_idx = int(map_size[3] * 0.5)

    V = tf.slice(vol, (0, 0, 0, slice_idx), (1, -1, -1, 1))
    V = tf.reshape(V, [map_size[1], map_size[2], 1 ] )
#    pdb.set_trace()
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, map_size[1], map_size[2], 1)))
    return V

def _inverse_lookup(my_dict, _value):
    """ invsersed dictionary lookup, return the first key given its value """
    #pdb.set_trace()
    for key, dic_value in my_dict.items():
        if dic_value == _value:
            return key
    return None

def _jaccard(conf_matrix):
    num_cls = conf_matrix.shape[0]
    jac = np.zeros(num_cls)
    for ii in range(num_cls):
        pp = np.sum(conf_matrix[:,ii])
        gp = np.sum(conf_matrix[ii,:])
        hit = conf_matrix[ii,ii]
        jac[ii] = hit * 1.0 / (pp + gp - hit)
    return jac

def _dice(conf_matrix):
    num_cls = conf_matrix.shape[0]
    dic = np.zeros(num_cls)
    for ii in range(num_cls):
        pp = np.sum(conf_matrix[:,ii])
        gp = np.sum(conf_matrix[ii,:])
        hit = conf_matrix[ii,ii]
        dic[ii] = 2.0 * hit / (pp + gp)
    return dic

    '''
    def _softmax_weighted_loss(self, logits, labels, num_cls = 8, mask = None):
        """
        Loss = weighted * -target*log(softmax(logits))
        :param logits: probability score
        :param labels: ground_truth
        :return: softmax-weifhted loss
        : That might run very very slow
        """
        gt = labels
        pred = logits
        softmaxpred = tf.nn.softmax(pred)
        loss = 0
        raw_loss = 0
        for i in range(num_cls):
            gti = gt[:,:,:,i]
            predi = softmaxpred[:,:,:,i]
            weighted = 1-(tf.reduce_sum(gti) / tf.reduce_sum(gt))
            if i == 0:
                if mask is None:
                    raw_loss += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
                else:
                    raw_loss += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1)) * mask
            else:
                if mask is None:
                    raw_loss += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
                else:
                    raw_loss += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1)) * mask

        loss = tf.reduce_mean(raw_loss)
#            if mask is not None:
#                mask  = tf.clip(raw_loss / loss, [0.5, 10])
        # this leaves the mask update under discussion
        return loss
    '''
'''

def get_volume_summary(vol, idx=0, slice_position = 0.5):
    """
    Make an image summary for volume with index idx
    reloaded version of previous one
    However, it is of no egg use.
    Args:
        slice_position: percentage of slice index in depth direction
        if the volume is 20*40*20*3, slice_position = 0.5, feature map of slice 10 will be shown
    """
    map_size = vol.get_shape().as_list()
    slice_idx = int(map_size[3] * 0.5)

    V = tf.slice(vol, (0, 0, 0, slice_idx, idx), (1, -1, -1, 1, 1))
    V = tf.reshape(V, [map_size[1], map_size[2], 1 ] )
#    pdb.set_trace()
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, map_size[1], map_size[2], 1)))
    return V
'''
'''
def val_stats(self, sess, summary_writer, step, batch_x, batch_y, detail = False):

    """
    test status
    compact_y is the compact label of y. Since it has already been calculated, for sake of time consumption, its
    reasonable to directly feed that in.
    Mask is not applicable for test. Therefore it would be all ones
    """
    fake_mask = np.ones( [self.net.batch_size, label_size[0] // mask_scale, label_size[1] // mask_scale, 1]  )
    if detail is not True:
        summary_str, summary_img, loss= sess.run([\
                                        self.scalar_summary_op,
                                        self.val_image_summary_op,
                                        self.net.cost],
                                        feed_dict={self.net.x: batch_x,
                                        self.net.mask_raw: fake_mask,
                                        self.net.y: batch_y,
                                        self.net.is_train: False,
                                        self.net.keep_prob: 1.})

    else:
        pred, curr_conf_mat, summary_str, summary_img, loss= sess.run([\
                                        self.net.compact_pred,
                                        self.net.confusion_matrix,
                                        self.scalar_summary_op,
                                        self.val_image_summary_op,
                                        self.net.cost],
                                        feed_dict={self.net.x: batch_x,
                                        self.net.mask_raw: fake_mask,
                                        self.net.y: batch_y,
                                        self.net.is_train: False,
                                        self.net.keep_prob: 1.})


        self._indicator_eval(curr_conf_mat)
    summary_writer.add_summary(summary_str, step)
    summary_writer.add_summary(summary_img, step)
    summary_writer.flush()
    '''
        
'''
with tf.device("/gpu:0"):
    for layer in range(0, layers):
        features = 2**layer*features_root
    #        stddev = np.sqrt(2 / (filter_size**2 * features))
        stddev = 0.01
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, features, features], stddev)

        w2 = weight_variable([filter_size, filter_size, features, features*2], stddev)

        conv1 = conv2d(in_node, w1, keep_prob)
        bn1 = batch_norm(conv1)
        tmp_h_conv = tf.nn.relu(bn1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        bn2 = batch_norm(conv2)
        dw_h_convs[layer] = tf.nn.relu(bn2)

        weights.append(w1)
        weights.append(w2)
        convs.append(conv1)
        convs.append(conv2)
        if verbose:
            logging.debug("shape of layer %s conv1: %s"%(str(layer), str(conv1.get_shape().as_list())))
            logging.debug("shape of layer %s conv2: %s"%(str(layer), str(conv2.get_shape().as_list())))

        if layer < layers:
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2

    # bottom layer
    features = 2**(layer+1)*features_root
    #    stddev = np.sqrt(2 / (filter_size**2 * features))

    wb1 = weight_variable([filter_size, filter_size, features, features], stddev)
    convb1 = conv2d(in_node, wb1, keep_prob, padding = 'SYMMETRIC')
    bnb1 = batch_norm(convb1)
    tmp_b_conv = tf.nn.relu(bnb1)

    wb2 = weight_variable([filter_size, filter_size, features, features * 2], stddev)
    convb2 = conv2d(tmp_b_conv, wb2, keep_prob, padding = 'SYMMETRIC')
    bnb2 = batch_norm(convb2)
    bottom_h_convs[0] = tf.nn.relu(bnb2)
    in_node = bottom_h_convs[0]
    weights.append(wb1)
    weights.append(wb2)
    convs.append(convb1)
    convs.append(convb2)

    if verbose:
        logging.debug("bottom layer size %s"%str(in_node.get_shape().as_list()))

    # uplayers 0, 1, 2
    for layer in range(0, layers):
        features = 2**(layers - layer)*features_root # 256 for first deconv
    # stddev = np.sqrt(2 / (filter_size**2 * features))

        wd = weight_variable_deconv([pool_size, pool_size, features * 2, features * 2], stddev)
        innode_shape = in_node.get_shape().as_list()
        logging.info("innode shape %s"%str(innode_shape))
        h_deconv = deconv2d(in_node, wd, pool_size)
        out_shape = [None, innode_shape[1]*2, innode_shape[2]*2, innode_shape[3]]
        h_deconv.set_shape(out_shape)
        logging.info("After deconv shape %s"%str(out_shape))
        h_bn = batch_norm(h_deconv)
        h_relu = tf.nn.relu(h_bn)

        if verbose:
            print("Shape of deconv before concat")
            print(h_relu.get_shape().as_list())
        h_deconv_concat = crp_and_concat(dw_h_convs[layers - layer - 1], h_relu)
        logging.info("As to what is concatenated to up path, it is still questionable. Xing Yang implementation varies with unet for this point")
        deconv[layer] = h_deconv_concat

        weights.append(wd)

        w1 = weight_variable([ filter_size, filter_size, features + features * 2, features], stddev)
        w2 = weight_variable([ filter_size, filter_size, features, features], stddev)

        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        bnd1 = batch_norm(conv1)
        h_conv = tf.nn.relu(bnd1)
        conv2 = conv2d(h_conv, w2, keep_prob)
        bnd2 = batch_norm(conv2)
        in_node = tf.nn.relu(bnd2)
        up_h_convs[layer] = in_node
        if verbose:
            print("Size of the end of upsample %s: %s"%(str(layer),str(in_node.get_shape().as_list())))
        weights.append(w1)
        weights.append(w2)
        convs.append((conv1, conv2))

    ## adding skips
#    with tf.device("/gpu:1"): server version
with tf.device("/gpu:0"):
    _, conv_up_1_2 = convs[-4]
    w12 = weight_variable([filter_size, filter_size, filter_size, features * 8, features], stddev)
    wd_12_1 = weight_variable_deconv([ pool_size, pool_size, features, features], stddev)
    wd_12_2 = weight_variable_deconv([ pool_size, pool_size, features, features], stddev)
    wd_12_3 = weight_variable_deconv([ pool_size, pool_size,  n_class, features], stddev)

    aux1_conv = conv2d(conv_up_1_2, w12, keep_prob)
    aux1_deconv_1 = deconv2d(aux1_conv, wd_12_1, pool_size )
    innode_shape = aux1_conv.get_shape().as_list()
    out_shape = [ None, innode_shape[1]*2, innode_shape[2]*2, innode_shape[3]]
    aux1_deconv_1.set_shape(out_shape)

    aux1_deconv_2 = deconv2d(aux1_deconv_1, wd_12_2, pool_size )
    innode_shape = aux1_deconv_1.get_shape().as_list()
    out_shape = [ None, innode_shape[1]*2, innode_shape[2]*2, innode_shape[3] ]
    aux1_deconv_2.set_shape(out_shape)

    aux1_prob = deconv_conv2d(aux1_deconv_2, wd_12_3, pool_size, n_class)

    innode_shape = aux1_deconv_2.get_shape().as_list()
    out_shape = [ None, innode_shape[1]*2, innode_shape[2]*2,  n_class]
    aux1_prob.set_shape(out_shape)

    aux_deconvs.append(aux1_conv)
    aux_deconvs.append(aux1_deconv_1)
    aux_deconvs.append(aux1_deconv_2)

    _, conv_up_2_2 = convs[-3]
    w22 = weight_variable([filter_size, filter_size, features * 4, features], stddev)
    wd_22_1 = weight_variable_deconv([ pool_size, pool_size, features, features], stddev)
    wd_22_2 = weight_variable_deconv([ pool_size, pool_size,  n_class, features], stddev)

    aux2_conv = conv3d(conv_up_2_2, w22, keep_prob)
    aux2_deconv_1 = deconv2d(aux2_conv, wd_22_1, pool_size )

    innode_shape = aux2_conv.get_shape().as_list()
    out_shape = [ None, innode_shape[1]*2, innode_shape[2]*2, innode_shape[3]]
    aux2_deconv_1.set_shape(out_shape)
    aux2_prob = deconv_conv2d(aux2_deconv_1, wd_22_2, pool_size, n_class)
    #pdb.set_trace()

    innode_shape = aux2_deconv_1.get_shape().as_list()
    out_shape = [ None, innode_shape[1]*2, innode_shape[2]*2, n_class]
    aux2_prob.set_shape(out_shape)

    aux_deconvs.append(aux2_conv)
    aux_deconvs.append(aux2_deconv_1)

    _, conv_up_3_2 = convs[-2]
    w32 = weight_variable([filter_size, filter_size, features * 2, features], stddev)
    wd_32_1 = weight_variable_deconv([ pool_size, pool_size, n_class, features], stddev)
    #pdb.set_trace()
    aux3_conv = conv2d(conv_up_3_2, w32, keep_prob)
    aux3_prob = deconv_conv2d(aux3_conv, wd_32_1, pool_size, n_class)

    innode_shape = aux3_conv.get_shape().as_list()
    out_shape = [None, innode_shape[1]*2, innode_shape[2]*2, n_class]
    aux3_prob.set_shape(out_shape)
    aux_deconvs.append(aux3_conv)

    aux_probs.append(aux1_prob)
    aux_probs.append(aux2_prob)
    aux_probs.append(aux3_prob)

    weights.append(w12)
    weights.append(wd_12_1)
    weights.append(wd_12_2)
    weights.append(wd_12_3)
    weights.append(w22)
    weights.append(wd_22_1)
    weights.append(wd_22_2)
    weights.append(w32)
    weights.append(wd_32_1)

# Output Map
#    with tf.device("/gpu:1"): # server version
    with tf.device("/gpu:0"):
        weight = weight_variable([1, 1, features_root * 2, n_class], stddev)
        conv = conv2d(in_node, weight, tf.constant(1.0))
    #    output_map = tf.nn.relu(conv + bias)
        output_map = conv
        up_h_convs["final_output"] = output_map
'''
'''    
def next_batch(self, input_queue, capacity = 120, num_threads = 4, min_after_dequeue = 30, label_type = 'float'):
    """ move original input pipeline here"""
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(input_queue)
    parser = tf.parse_single_example(serialized_example, features = decomp_feature)
    dsize_dim0 = tf.cast(parser['dsize_dim0'], tf.int32)
    dsize_dim1 = tf.cast(parser['dsize_dim1'], tf.int32)
    dsize_dim2 = tf.cast(parser['dsize_dim2'], tf.int32)
    lsize_dim0 = tf.cast(parser['lsize_dim0'], tf.int32)
    lsize_dim1 = tf.cast(parser['lsize_dim1'], tf.int32)
    lsize_dim2 = tf.cast(parser['dsize_dim2'], tf.int32)
    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    label_vol = tf.decode_raw(parser['label_vol'], tf.float32)

    data_vol = tf.reshape(data_vol, raw_size)
    label_vol = tf.reshape(label_vol, raw_size)
    data_vol = tf.slice(data_vol, [0,0,0],volume_size)
    label_vol = tf.slice(label_vol, [0,0,1], label_size)

    data_feed, label_feed, fid_feed = tf.train.shuffle_batch([data_vol, label_vol, fid], batch_size =self.batch_size , capacity = capacity, \
                                                        num_threads = num_threads, min_after_dequeue = min_after_dequeue)

    pair_feed = tf.concat([data_feed, label_feed], axis = 3)

    return pair_feed, fid_feed
#        return all_feed
    

def next_batch_with_mask(self, input_queue, capacity = 120, num_threads = 4, min_after_dequeue = 30, label_type = 'float'):
    """ move original input pipeline here """
    #pdb.set_trace()
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(input_queue)
    parser = tf.parse_single_example(serialized_example, features = mask_feature)
    dsize_dim0 = tf.cast(parser['dsize_dim0'], tf.int32)
    dsize_dim1 = tf.cast(parser['dsize_dim1'], tf.int32)
    dsize_dim2 = tf.cast(parser['dsize_dim2'], tf.int32)
    lsize_dim0 = tf.cast(parser['lsize_dim0'], tf.int32)
    lsize_dim1 = tf.cast(parser['lsize_dim1'], tf.int32)
    lsize_dim2 = tf.cast(parser['dsize_dim2'], tf.int32)
    mask_scale = tf.cast(parser['mask_scale'], tf.int32)
    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    label_vol = tf.decode_raw(parser['label_vol'], tf.float32)
    mask_vol = tf.decode_raw(parser['mask_vol'], tf.float32)

    data_vol = tf.reshape(data_vol, raw_size)
    label_vol = tf.reshape(label_vol, raw_size)
    #pdb.set_trace()
    mask_vol = tf.reshape(mask_vol, [1, label_size[0] // mask_scale, label_size[1] // mask_scale, 1])
    mask_vol = tf.image.resize_bilinear(mask_vol, [label_size[0], label_size[1]])
    mask_vol = tf.reshape(mask_vol, [label_size[0], label_size[1], 1])
    data_vol = tf.slice(data_vol, [0,0,0],volume_size)
    label_vol = tf.slice(label_vol, [0,0,1], label_size)

    data_feed, label_feed, mask_feed, fid_feed = tf.train.shuffle_batch([data_vol, label_vol, mask_vol, fid], batch_size =self.batch_size , capacity = capacity, \
                                                        num_threads = num_threads, min_after_dequeue = min_after_dequeue)

    pair_feed = tf.concat([data_feed, label_feed, mask_feed], axis = 3)

    return pair_feed, fid_feed
'''
'''
def _update_mask_node(self, past = 1.0):
    # update mask
#        with tf.device("/gpu:3"): # server version An additional gpu seems to be required.
    #pdb.set_trace()
    with tf.device("/gpu:0"):
        mask = (1.0 - past) * (  tf.add( tf.cast(tf.not_equal(self.compact_pred, self.compact_y), tf.float32), 1.5)) + past * self.mask[...,0]
        mask = tf.expand_dims(mask, -1)
        mask = tf.image.resize_bilinear(mask, size = [label_size[0] // mask_scale, label_size[1]// mask_scale])
        mask = tf.reshape(mask, [-1, label_size[0] // mask_scale, label_size[1] // mask_scale])
    print("please do check this ")
    return mask'''
    
'''
    # TODO: move interpolation to tensorflow
class mask_manager(object):
    """
    in charge of mask io
    """
    def __init__(self, mask_folder, ref_size):
        """ ref size is in [batch_size, w, h, 1]"""
        self._mask_dict = {}
        self._ref_size = ref_size
        self.mask_folder = mask_folder

    def mask_lookup(self,fid_list):
        """ query masks """
        out_masks = np.zeros(self._ref_size)
        for idx,fid in enumerate(fid_list):

            if fid not in self._mask_dict.keys():
                #pdb.set_trace()
                out_fid = os.path.basename(fid).split(".tfrecords")[0]
                out_fid = os.path.join(self.mask_folder, "mask_" + out_fid + ".npz")

                self._mask_dict[fid] = out_fid
                _vol = np.ones([ self._ref_size[1], self._ref_size[2] ], dtype = np.float32)
                np.savez(out_fid, _vol)
            else:
                out_fid = self._mask_dict[fid]
                _vol = np.load(out_fid)['arr_0']
            out_masks[idx,:,:,0] = _vol
        return out_masks

    def mask_writer(self, mask_list, fid_list):
        """ update mask
        mask_list: [batch_size, w, h, 1]

        """
        #pdb.set_trace()
        for idx,fid in enumerate(fid_list):
            out_fid = self._mask_dict[fid]
            np.savez(out_fid, mask_list[idx, :, :])
            '''