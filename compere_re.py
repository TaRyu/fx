"""Compere with other classifiers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import time

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY']
FILE_PREX = '../data/fx'
names = ["Linear_SVM", "RBF_SVM", 'KNN', "Random_Forest"]
res = [SVR(kernel='linear'), SVR(),
       KNeighborsRegressor(), RandomForestRegressor()]
time_format = '%Y%m%d%H%M'
result_tmp = np.empty(0)


def columns_define():
    columns = []
    for name in names:
        columns.append('%s(S)' % name)
        columns.append('%s(T)' % name)
    return columns

if __name__ == '__main__':
    for fx in FX_LIST:
        path_f_final = ['%s/%s_FINAL_M.npy' % (FILE_PREX, fx),
                        '%s/%s_FINAL_S.pkl' % (FILE_PREX, fx)]
        data = np.load(path_f_final[0])
        data_s = pd.read_pickle(path_f_final[1])
        range_price = data_s['max_price'] - data_s['min_price']
        data = np.array([(data[i] - data_s['min_price'][i]) /
                         range_price[i] for i in range(data.shape[0])])
        data_train = data[:data.shape[0] - 354]
        data_test = data[data.shape[0] - 354:]
        data_s_train = data_s[:data.shape[0] - 354]
        data_s_test = data_s[data.shape[0] - 354:]
        for i in range(len(names)):
            re = res[i]
            pass
            start = time.time()
            re.fit(data_train, data_s_train['change'])
            end = time.time()
            time_cost = end - start
            score = metrics.explained_variance_score(
                data_s_test['change'], re.predict(data_test))
            result_tmp = np.append(result_tmp, [score, time_cost])
    result = pd.DataFrame(result_tmp.reshape(-1, 2 * len(names)),
                          index=FX_LIST, columns=columns_define())
    result.to_pickle('../data/fx/result_compere_re.pkl')
