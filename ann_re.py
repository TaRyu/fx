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
SCALE = ['MON', 'DAY']
FILE_PREX = '../data/fx'

time_format = '%Y%m%d%H%M'
result_tmp1 = np.empty(0)
result_tmp2 = np.empty(0)
num_test = 354


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

    feature_columns = learn.infer_real_valued_columns_from_input(data_train)
    regressor = learn.DNNRegressor(
        feature_columns=feature_columns, hidden_units=[10])

    # Fit
    regressor.fit(data_train, data_s_train, steps=5000,
                  batch_size=1, logdir=logdir)

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
            score1, score2 = main(fx, SCALE)
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
