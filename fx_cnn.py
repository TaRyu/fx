"""Predict based on cnn"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics

import tensorflow as tf
from tensorflow.contrib import learn

PATH_FILE_FINAL = ['EURUSD_FINAL_M.npy', 'EURUSD_FINAL_S.pkl']
NUM_PIX = 24 * 24

data = np.load(PATH_FILE_FINAL[0])[:4]
data_s = pd.read_pickle(PATH_FILE_FINAL[1])[:4]
labels = data_s['buy_or_sell']
data_s = preprocessing.normalize(data_s.reshape(
    len(data_s), NUM_PIX)).reshape(data_s.shape)


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def conv_model(X, y):
    X = tf.reshape(X, [-1, 24, 24, 1])
    # first conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = learn.ops.conv2d(X, n_filters=32, filter_shape=[5, 5],
                                   bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    # second conv layer will compute 64 features for each 5x5 patch
    with tf.variable_scope('conv_layer2'):
        h_conv2 = learn.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                                   bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # densely connected layer with 1024 neurons
    h_fc1 = learn.ops.dnn(
        h_pool2_flat, [1024], activation=tf.nn.relu, keep_prob=0.5)
    return learn.models.logistic_regression(h_fc1, y)

# Training and predicting
classifier = learn.TensorFlowEstimator(
    model_fn=conv_model, n_classes=2, batch_size=100, steps=20000,
    learning_rate=0.001)
classifier.fit(data, labels)
score = metrics.accuracy_score(labels, classifier.predict(data))
print('Accuracy: {0:f}'.format(score))
