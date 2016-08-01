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
names = ["SGD", 'Ridge', "SVR", 'KNN', "Random_Forest"]
res = [SGDRegressor(), Ridge(), SVR(),
       KNeighborsRegressor(), RandomForestRegressor()]
time_format = '%Y%m%d%H%M'
result_tmp = np.empty(0)
num_test = 354


if __name__ == '__main__':
    for fx in FX_LIST:
        path_f_final = ['%s/%s_MON_f.npy' % (FILE_PREX, fx),
                        '%s/%s_MON_t.pkl.npy' % (FILE_PREX, fx)]
        data = np.load(path_f_final[0])
        data_s = np.load(path_f_final[1])
        data_train = data[:data.shape[0] - num_test]
        data_test = data[data.shape[0] - num_test:]
        data_s_train = data_s[:data.shape[0] - num_test]
        data_s_test = data_s[data.shape[0] - num_test:]
        for i in range(len(names)):
            re = res[i]
            re.fit(data_train, data_s_train)
            prediction = re.predict(data_test)
            score = metrics.explained_variance_score(
                data_s_test, prediction)
            result_tmp = np.append(result_tmp, score)
    result = pd.DataFrame(result_tmp.reshape(-1, len(names)),
                          index=FX_LIST, columns=names)
    result.to_pickle('../data/fx/result_compere_re_month.pkl')
