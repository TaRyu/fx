"""Predict based on cnn"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import metrics

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD',
           'AUDUSD', 'EURJPY', 'EURGBP']
FX_LIST = ['EURUSD']
FILE_PREX = '../../../data/fx'
NUM_PIX = 12 * 12
time_format = '%Y%m%d%H%M'
optimizers = ['Adagrad']
# optimizers = ['GradientDescent', 'Adadelta',
#               'Momentum', 'Adam', 'Ftrl', 'RMSProp', 'SGD', 'Adagrad']


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def conv_model(X, y):
    X = tf.reshape(X, [-1, 12, 12, 1])
    # first conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('Conv1'):
        h_conv1 = learn.ops.conv2d(X, n_filters=16, filter_shape=[3, 3],
                                   bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    # second conv layer will compute 64 features for each 5x5 patch
    with tf.variable_scope('Conv2'):
        h_conv2 = learn.ops.conv2d(h_pool1, n_filters=32, filter_shape=[5, 5],
                                   bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 3 * 32])
    # densely connected layer with 1024 neurons
    with tf.variable_scope('F-C'):
        h_fc1 = learn.ops.dnn(
            h_pool2_flat, [512], activation=tf.nn.relu, dropout=0.5)
    with tf.variable_scope('LR'):
        o_linear = learn.models.linear_regression(h_fc1, y)
    return o_linear


num_test = 354
result_tmp = np.empty(0)

if __name__ == '__main__':
    df = pd.DataFrame()
    for fx in FX_LIST:
        for optimizer in optimizers:
            start = time.strftime(time_format, time.localtime())
            print('%s start at %s.' % (fx, start))
            model = learn.TensorFlowEstimator(
                model_fn=conv_model,
                n_classes=0,
                batch_size=100, steps=20000,
                optimizer=optimizer,
                learning_rate=0.001)
            fs_t_path = ['%s/NFs/%s_2H.npy' % (FILE_PREX, fx),
                         '%s/T/%s_2H.pkl' % (FILE_PREX, fx)]
            logdir = '%s/tensorboard_models/modelR/%s/%s' % (
                FILE_PREX,
                optimizer,
                fx)
            fs = np.load(fs_t_path[0])
            t = pd.read_pickle(fs_t_path[1])
            model.fit(fs[:-num_test],
                      t['change'][:-num_test],
                      logdir=logdir)
            model.save('%s/saves/modelR/%s/%s' % (FILE_PREX, optimizer, fx))
            prediction = model.predict(fs[-num_test:])
            score = metrics.explained_variance_score(
                prediction, t['change'][-num_test:])
            result_tmp = np.append(result_tmp, score)
            print(result_tmp)
            end = time.strftime(time_format, time.localtime())
            print('%s end at %s.' % (fx, end))
    result = pd.DataFrame(result_tmp.reshape(-1, len(optimizers)),
                          index=FX_LIST, columns=optimizers)
    print(result)
    result.to_pickle('%s/modelC_result_2H.pkl' % FILE_PREX)
