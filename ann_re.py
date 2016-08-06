"""Predict based on cnn"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn import metrics
import time

from tensorflow.contrib import learn

FX_LIST = ['USDJPY', 'EURUSD', 'GBPUSD',
           'AUDUSD', 'EURJPY', 'EURGBP']
SCALE = ['DAY', 'MON']
FILE_PREX = '../data/fx'

time_format = '%Y%m%d%H%M'
result_tmp1 = np.empty(0)
result_tmp2 = np.empty(0)
num_test = 354


def my_model(x, y):
    layers = learn.ops.dnn(x, [10], dropout=0.5)
    return learn.models.linear_regression(layers, y)


def main(fx, scale):
    logdir = '../data/fx/ann/tensorboard_models/%s%s%s' % (
        scale,
        fx,
        time.strftime(time_format, time.localtime()))
    # Load dataset
    path_f_final = ['%s/%s_%s_f.npy' % (FILE_PREX, fx, scale),
                    '%s/%s_%s_t.pkl.npy' % (FILE_PREX, fx, scale)]
    path_f_in = '%s/%s_H.pkl' % (FILE_PREX, fx)
    pd_data = pd.read_pickle(path_f_in)['close']
    fx_max = max(pd_data)
    fx_min = min(pd_data)
    data = np.load(path_f_final[0])
    data_s = np.load(path_f_final[1])
    data_train = data[:data.shape[0] - num_test]
    data_test = data[data.shape[0] - num_test:]
    data_s_train = data_s[:data.shape[0] - num_test]
    data_s_test = data_s[data.shape[0] - num_test:]

    regressor = learn.TensorFlowEstimator(
        model_fn=my_model,
        n_classes=0, optimizer='SGD',
        batch_size=len(data_train), steps=20000,
        learning_rate=0.2)

    # Fit
    regressor.fit(data_train, data_s_train, logdir=logdir)

    # Predict and score
    prediction = regressor.predict(data_test)
    data = {'close_price': [i * (fx_max - fx_min) + fx_min for i in data_s_test],
            'predict': [i * (fx_max - fx_min) + fx_min for i in prediction]}
    frame = pd.DataFrame(data)
    frame.to_pickle('%s/%sprediction.pkl' % (logdir, fx))
    score1 = metrics.explained_variance_score(
        data_s_test, prediction)
    score2 = metrics.mean_absolute_error(
        data_s_test, prediction)
    print(score1, score2)
    return score1, score2


if __name__ == '__main__':
    for fx in FX_LIST:
        for scale in SCALE:
            score1, score2 = main(fx, scale)
            result_tmp1 = np.append(result_tmp1, score1)
            result_tmp2 = np.append(result_tmp2, score2)
    result1 = pd.DataFrame(result_tmp1.reshape(-1, len(SCALE)),
                           index=FX_LIST, columns=SCALE)
    print(result1)
    result2 = pd.DataFrame(result_tmp2.reshape(-1, len(SCALE)),
                           index=FX_LIST, columns=SCALE)
    print(result2)
    result1.to_pickle('../data/fx/ann/result1.pkl')
    result1.to_pickle('../data/fx/ann/result2.pkl')
