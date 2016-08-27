"""Predict based on cnn"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow.contrib import learn

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD',
           'AUDUSD', 'EURJPY', 'EURGBP']
FILE_PREX = '../data/fx/app/data'
optimizers = ['SGD']
# optimizers = ['GradientDescent', 'Adadelta',
#               'Momentum', 'Adam', 'Ftrl', 'RMSProp']


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def conv_model(X, y):
    X = tf.reshape(X, [-1, 24, 24, 1])
    # first conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('Conv_Layer1'):
        h_conv1 = learn.ops.conv2d(X, n_filters=32, filter_shape=[5, 5],
                                   bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    # second conv layer will compute 64 features for each 5x5 patch
    with tf.variable_scope('Conv_Layer2'):
        h_conv2 = learn.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                                   bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 64])
    # densely connected layer with 1024 neurons
    with tf.variable_scope('FC_Layer'):
        h_fc1 = learn.ops.dnn(
            h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
    with tf.variable_scope('LR_Layer'):
        o_linear = learn.models.linear_regression(h_fc1, y)
    return o_linear


if __name__ == '__main__':
    for fx in FX_LIST:
        for optimizer in optimizers:
            if optimizer == 'Momentum':
                re = learn.TensorFlowEstimator(
                    model_fn=conv_model,
                    n_classes=0,
                    batch_size=80, steps=20000,
                    optimizer=tf.train.MomentumOptimizer(
                        learning_rate=0.005, momentum=0.5))
            else:
                re = learn.TensorFlowEstimator(
                    model_fn=conv_model,
                    n_classes=0,
                    batch_size=80, steps=20000,
                    optimizer=optimizer,
                    learning_rate=0.005)
            path_f_final = ['%s/%s_FINAL_M.npy' % (FILE_PREX, fx),
                            '%s/%s_FINAL_S.pkl' % (FILE_PREX, fx)]
            data = np.load(path_f_final[0])
            data_s = pd.read_pickle(path_f_final[1])
            range_price = data_s['max_price'] - data_s['min_price']
            data = np.array([(data[i] - data_s['min_price'][i]) /
                             range_price[i] for i in range(data.shape[0])])
            start = time.time()
            logdir = '../../data/fx/app/tensorboard_models/%s' % fx
            re.fit(data, (data_s['change']),
                   logdir=logdir)
            end = time.time()
            print('%s over with %s' % (fx, optimizer))