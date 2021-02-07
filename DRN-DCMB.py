#!/usr/bin/env python
# coding: utf-8

#####published work#####
# Liu J, Kocak M, Supanich M, Deng J. Motion artifacts reduction in brain MRI by means of a deep residual network with densely connected multi-resolution blocks (DRN-DCMB). Magn Reson Imaging. 2020;71. doi:10.1016/j.mri.2020.05.002

#####Author: Junchi Liu
#####email: jliu118@hawk.iit.edu



#######################################################
from __future__ import print_function
import keras
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Dropout, merge, Input
from keras.layers.normalization import BatchNormalization
from keras.layers import AveragePooling2D, Input, Flatten, Concatenate, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
import math
import numpy as np
import os
import pickle
import time
from scipy import misc

###############
def conv_block(x, n_f, strides_x, strides_y):
    x2 = Conv2D(filters=n_f, kernel_size=(3,3), strides=(strides_x, strides_y), kernel_initializer='he_normal', padding='same')(x)
    x2 = BatchNormalization(axis=-1, epsilon=1e-3)(x2)
    x2 = LeakyReLU(alpha=0.3)(x2)
    return x2
    
#####multi resolution block####
def m_block(x):
    x2 = conv_block(x, n_f, 1, 1)
    x2 = conv_block(x2, n_f, 1, 1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x2)
    
    x3 = conv_block(pool2, n_f, 1, 1)
    x3 = conv_block(x3, n_f, 1, 1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x3)
    
    x4 = conv_block(pool3, n_f, 1, 1)
    x4 = conv_block(x4, n_f, 1, 1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(x4)

    x5 = conv_block(pool4, n_f, 1, 1)
    x5 = conv_block(x5, n_f, 1, 1)
    pool5 = MaxPooling2D(pool_size=(2, 2))(x5)
    
    x6 = conv_block(pool5, n_f, 1, 1)
    x6 = conv_block(x6, n_f, 1, 1)
    
    up7 = UpSampling2D(size = (2,2))(x6)
    up7 = conv_block(up7, n_f, 1, 1)
    merge7 = Concatenate(axis=-1)([x5,up7])
    x7 = conv_block(merge7, n_f, 1, 1)
    x7 = conv_block(x7, n_f, 1, 1)
    
    up8 = UpSampling2D(size = (2,2))(x7)
    up8 = conv_block(up8, n_f, 1, 1)
    merge8 = Concatenate(axis=-1)([x4,up8])
    x8 = conv_block(merge8, n_f, 1, 1)
    x8 = conv_block(x8, n_f, 1, 1)

    up9 = UpSampling2D(size = (2,2))(x8)
    up9 = conv_block(up9, n_f, 1, 1)
    merge9 = Concatenate(axis=-1)([x3,up9])
    x9 = conv_block(merge9, n_f, 1, 1)
    x9 = conv_block(x9, n_f, 1, 1)
    
    up10 = UpSampling2D(size = (2,2))(x9)
    up10 = conv_block(up10, n_f, 1, 1)
    merge10 = Concatenate(axis=-1)([x2,up10])
    x10 = conv_block(merge10, n_f, 1, 1)
    x10 = conv_block(x10, n_f, 1, 1)
    return x10

#####dense block#########
def dense_block(x,n_layer):
    list_feat = [x]
    for i in range(n_layer):
        x = m_block(x)
        list_feat.append(x)
        x = Concatenate(axis=-1)(list_feat)
    return x
    
    
################################main################################
inpt = Input(shape=(None,None,1)) ####### the input size if flexible
######dense net######
x_b = dense_block(inpt, 3) ##### e.g. 3 blocks, number of blocks can be customized
#########
x_b = conv_block(x_b, 1, 1, 1)
x_out = keras.layers.Subtract()([inpt, x_b])
########
autoencoder = Model(inputs=inpt, outputs=x_out)
