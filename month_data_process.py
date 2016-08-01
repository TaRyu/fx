"""Data process"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

FX_LIST = ['EURGBP', 'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY']
FX_LIST = ['EURGBP']
FILE_PREX = '../data/fx'
PATH_FILE_IN = '../data/fx/EURUSD.txt'
PATH_FILE_FINAL = ['EURUSD_MON_f.npy', 'EURUSD_MON_t.npy']


def process(file_in=PATH_FILE_IN, file_out=PATH_FILE_FINAL):
    # data = pd.read_csv(file_in, dtype='str')
    # data['DateTime'] = pd.to_datetime(
    #     data['<DTYYYYMMDD>'].map(str) + data['<TIME>'].map(str),
    #     format='%Y%m%d%H%M%S')
    # data = data.set_index('DateTime')
    # data = pd.Series(data['<CLOSE>']).map(float)
    # data = data.resample('M').fillna(method='pad')
    # data = preprocessing.minmax_scale(data)
    # data_t = data[6:]
    # data_f = data.reshape(-1, 6)
    # data_f = np.array([data[i:i + 6] for i in range(data.shape[0] - 6 + 1)])
    # np.save(file_out[0], data_f[:len(data_f) - 1])
    # np.save(file_out[1], data_t)
    data = preprocessing.minmax_scale(pd.read_pickle(
        file_in)['close'])
    data = data.reshape(-1, 24)
    data_m = np.array([[data[i + x * 24][0] for x in range(6)]
                       for i in range(len(data) - 6 * 24 + 1)])
    data_m = data_m.reshape(-1, 6)
    data_s = np.array([data[i + 6 * 24][0]
                       for i in range(len(data) - 6 * 24)])
    np.save(file_out[0], data_m[:len(data_m) - 1])
    np.save(file_out[1], data_s)

if __name__ == '__main__':
    for fx in FX_LIST:
        path_f_in = '%s/%s_H.pkl' % (FILE_PREX, fx)
        path_f_final = ['%s/%s_MON_f.npy' % (FILE_PREX, fx),
                        '%s/%s_MON_t.pkl' % (FILE_PREX, fx)]
        process(path_f_in, path_f_final)
