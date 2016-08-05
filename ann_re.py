"""Predict based on cnn"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn import metrics
import time

import tensorflow as tf
from tensorflow.contrib import learn

FX_LIST = ['USDJPY', 'EURUSD', 'GBPUSD',
           'AUDUSD', 'EURJPY', 'EURGBP']
FILE_PREX = '../data/fx'
optimizers = ['Momentum']
# optimizers = ['GradientDescent', 'Adadelta',
#               'Momentum', 'Adam', 'Ftrl', 'RMSProp']


def main(unused_argv):
    # Load dataset

    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    feature_columns = learn.infer_real_valued_columns_from_input(x_train)
    regressor = learn.DNNRegressor(
        feature_columns=feature_columns, hidden_units=[10, 10])

    # Fit
    regressor.fit(x_train, y_train, steps=5000, batch_size=1)

    # Predict and score
    y_predicted = regressor.predict(scaler.transform(x_test))
    score = metrics.mean_squared_error(y_predicted, y_test)

    print('MSE: {0:f}'.format(score))


time_format = '%Y%m%d%H%M'
result_tmp1 = np.empty(0)
result_tmp2 = np.empty(0)
num_test = 354

if __name__ == '__main__':
    for fx in FX_LIST:
        for optimizer in optimizers:
            if optimizer == 'Momentum':
                re = learn.TensorFlowEstimator(
                    model_fn=conv_model,
                    n_classes=0,
                    batch_size=70, steps=20000,
                    optimizer=tf.train.MomentumOptimizer(
                        learning_rate=0.001, momentum=0.5))
            else:
                re = learn.TensorFlowEstimator(
                    model_fn=conv_model,
                    n_classes=0,
                    batch_size=70, steps=20000,
                    optimizer=optimizer,
                    learning_rate=0.001)
            path_f_final = ['%s/%s_FINAL_M.npy' % (FILE_PREX, fx),
                            '%s/%s_FINAL_S.pkl' % (FILE_PREX, fx)]
            data = np.load(path_f_final[0])
            data_s = pd.read_pickle(path_f_final[1])
            range_price = data_s['max_price'] - data_s['min_price']
            data = np.array([(data[i] - data_s['min_price'][i]) /
                             range_price[i] for i in range(data.shape[0])])
            data_train = data[:data.shape[0] - num_test]
            data_test = data[data.shape[0] - num_test:]
            data_s_train = data_s[:data.shape[0] - num_test]
            data_s_test = data_s[data.shape[0] - num_test:]
            start = time.time()
            logdir = '../data/fx/ann/tensorboard_models/%s%s%s' % (
                optimizer,
                fx,
                time.strftime(time_format, time.localtime()))
            re.fit(data_train, (data_s_train['change']),
                   logdir=logdir)
            end = time.time()
            data_s_test['predict'] = re.predict(data_test)
            data_s_test.to_pickle('%s/%sprediction.pkl' % (logdir, fx))
            score1 = metrics.explained_variance_score(
                (data_s_test['change']), re.predict(data_test))
            score2 = metrics.mean_absolute_error(
                (data_s_test['change']), re.predict(data_test))
            result_tmp1 = np.append(result_tmp1, score1)
            print(result_tmp1)
            result_tmp2 = np.append(result_tmp2, score2)
            print(result_tmp2)
    result1 = pd.DataFrame(result_tmp1.reshape(-1, len(optimizers)),
                           index=FX_LIST, columns=optimizers)
    print(result1)
    result2 = pd.DataFrame(result_tmp2.reshape(-1, len(optimizers)),
                           index=FX_LIST, columns=optimizers)
    print(result2)
    result1.to_pickle('../data/fx/ann/result1.pkl')
    result1.to_pickle('../data/fx/ann/result2.pkl')
