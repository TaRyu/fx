"""Compere with other classifiers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

FX_LIST = ['EURGBP', 'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY']
FILE_PREX = '../data/fx'
names = ["k-NN", "SVM"]
res = [KNeighborsRegressor(), SVR()]
time_format = '%Y%m%d%H%M'
result_tmp1 = np.empty(0)
result_tmp2 = np.empty(0)
num_test = 354


if __name__ == '__main__':
    for fx in FX_LIST:
        path_f_final = ['%s/%s_DAY_f.npy' % (FILE_PREX, fx),
                        '%s/%s_DAY_t.pkl.npy' % (FILE_PREX, fx)]
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
        for i in range(len(names)):
            prdir = '../data/fx/prediction/1/%s' % names[i]
            re = res[i]
            re.fit(data_train, data_s_train)
            prediction = re.predict(data_test)
            data = {'close_price': [i * (fx_max - fx_min) + fx_min for i in data_s_test],
                    'predict': [i * (fx_max - fx_min) + fx_min for i in prediction]}
            frame = pd.DataFrame(data)
            frame.to_pickle('%s/%sprediction.pkl' % (prdir, fx))
            score1 = metrics.explained_variance_score(
                data_s_test, prediction)
            score2 = metrics.mean_absolute_error(
                data_s_test, prediction)
            result_tmp1 = np.append(result_tmp1, score1)
            result_tmp2 = np.append(result_tmp2, score2)
    result1 = pd.DataFrame(result_tmp1.reshape(-1, len(names)),
                           index=FX_LIST, columns=names)
    result2 = pd.DataFrame(result_tmp2.reshape(-1, len(names)),
                           index=FX_LIST, columns=names)
    result1.to_pickle('../data/fx/result_compere_re_day1.pkl')
    result2.to_pickle('../data/fx/result_compere_re_day2.pkl')
