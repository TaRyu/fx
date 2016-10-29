"""Predict based on cnn"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import metrics, cross_validation

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD',
           'AUDUSD', 'EURJPY', 'EURGBP']
FX_LIST = ['EURUSD']
FILE_PREX = '../../../data/fx'
NUM_PIX = 24 * 24
time_format = '%Y%m%d%H%M'
optimizers = ['Adagrad']
# optimizers = ['GradientDescent', 'Adadelta',
#               'Momentum', 'Adam', 'Ftrl', 'RMSProp', 'SGD', 'Adagrad']


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def conv_model(X, y):
    X = tf.reshape(X, [-1, 24, 24, 1])
    # first conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('Conv1'):
        h_conv1 = learn.ops.conv2d(X, n_filters=32, filter_shape=[5, 5],
                                   bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    # second conv layer will compute 64 features for each 5x5 patch
    with tf.variable_scope('Conv2'):
        h_conv2 = learn.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                                   bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 64])
    # densely connected layer with 1024 neurons
    with tf.variable_scope('F-C'):
        h_fc1 = learn.ops.dnn(
            h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
    with tf.variable_scope('LR'):
        o_linear = learn.models.linear_regression(h_fc1, y)
    return o_linear


time_format = '%Y%m%d%H%M'
result_tmp1 = np.empty(0)
result_tmp2 = np.empty(0)
num_test = 354

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
            fs_t_path = ['%s/NFs/%s.npy' % (FILE_PREX, fx),
                         '%s/T/%s.pkl' % (FILE_PREX, fx)]
            logdir = '%s/tensorboard_models/exm_R_24/%s/%s' % (
                FILE_PREX,
                optimizer,
                fx)
            fs = np.load(fs_t_path[0])
            t = pd.read_pickle(fs_t_path[1])
            model.fit(fs[:-num_test],
                      t['change'][:-num_test],
                      logdir=logdir)
            model.save('%s/saves/exm_R_24/%s/%s' % (FILE_PREX, optimizer, fx))
            prediction1 = model.predict(fs[-num_test:])
            prediction2 = (prediction1 / 100 + 1) * \
                t['target_open'][-num_test:]
            score1 = metrics.explained_variance_score(
                prediction1, t['change'][-num_test:])
            score2 = metrics.mean_absolute_error(
                prediction2, t['real_target'][-num_test:])
            result_tmp1 = np.append(result_tmp1, score1)
            print(result_tmp1)
            result_tmp2 = np.append(result_tmp2, score2)
            print(result_tmp2)
            end = time.strftime(time_format, time.localtime())
            print('%s end at %s.' % (fx, end))
    result1 = pd.DataFrame(result_tmp1.reshape(-1, len(optimizers)),
                           index=FX_LIST, columns=optimizers)
    print(result1)
    result2 = pd.DataFrame(result_tmp2.reshape(-1, len(optimizers)),
                           index=FX_LIST, columns=optimizers)
    print(result2)
    result1.to_pickle('%s/exm_R_24_evs.pkl' % FILE_PREX)
    result2.to_pickle('%s/exm_R_24_mae.pkl' % FILE_PREX)
