"""Predict based on cnn"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import time

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessHour
import tensorflow as tf
from tensorflow.contrib import learn

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY']
NUM_PIX = 24 * 24
CUSTOM_BH = CustomBusinessHour(calendar=USFederalHolidayCalendar(),
                               start='00:00', end="23:59")
RANGE_TIME = pd.DatetimeIndex(start='200101030000',
                              end='201606302300', freq=CUSTOM_BH)


def m2h(file_in):
    data = pd.read_csv(file_in, dtype='str')
    data['DateTime'] = pd.to_datetime(
        data['<DTYYYYMMDD>'].map(str) + data['<TIME>'].map(str),
        format='%Y%m%d%H%M%S')
    data = data.set_index('DateTime')
    data = pd.Series(data['<CLOSE>']).map(float)
    data = data.resample('H', how='ohlc').fillna(method='pad')
    data = data.reindex(RANGE_TIME)
    return data


def one2two(data, fx_out):
    data = data['close']
    data = data.reshape(-1, 24)
    data = np.array([data[i:i + 24] for i in range(data.shape[0] - 24 + 1)])
    data_s = {
        'open_price': np.array([data[i][0][0]
                                for i in range(data.shape[0] - 1)]),
        'close_price': np.array([data[i][int(NUM_PIX / 24) - 1][23]
                                 for i in range(data.shape[0] - 1)]),
        'max_price': np.array([data[i].max()
                               for i in range(data.shape[0] - 1)]),
        'min_price': np.array([data[i].min()
                               for i in range(data.shape[0] - 1)]),
        'buy_or_sell': np.array(
            [data[i + 1][int(NUM_PIX / 24) - 1][23] > data[i + 1][0][0]
             for i in range(data.shape[0] - 1)])}
    data_s = pd.DataFrame(data_s)
    data = data.reshape(len(data), NUM_PIX)
    np.save(fx_out, data)
    return data[:len(data) - 1], data_s


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
        h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 64])
    # densely connected layer with 1024 neurons
    h_fc1 = learn.ops.dnn(
        h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
    return learn.models.logistic_regression(h_fc1, y)

# Training and predicting
classifier = learn.TensorFlowEstimator(
    model_fn=conv_model, n_classes=2, batch_size=100, steps=20000,
    learning_rate=0.001)
time_format = '%Y%m%d%H%M'
result = np.array(0)

if __name__ == '__main__':
    for fx in FX_LIST:
        data = m2h('../data/fx/%s.txt' % fx)
        data, data_s = one2two(data, '../data/fx/latest/%s.npy' % fx)
        range_price = data_s['max_price'] - data_s['min_price']
        data = data = np.array([(data[i] - data_s['min_price'][i]) /
                                range_price[i] for i in range(data.shape[0])])
        start = time.time()
        classifier.fit(data, data_s['buy_or_sell'],
                       logdir='../data/fx/latest/%s/' % fx)
        end = time.time()
        print('Fit cost %fs' % (end - start))
