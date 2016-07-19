"""Data process"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY']
FILE_PREX = '../data/fx'
PATH_FILE_IN = '../data/fx/EURUSD.txt'
PATH_FILE_FINAL = ['EURUSD_MON_f.npy', 'EURUSD_MON_t.npy']


def m2m(file_in=PATH_FILE_IN, file_out=PATH_FILE_FINAL):
    data = pd.read_csv(file_in, dtype='str')
    data['DateTime'] = pd.to_datetime(
        data['<DTYYYYMMDD>'].map(str) + data['<TIME>'].map(str),
        format='%Y%m%d%H%M%S')
    data = data.set_index('DateTime')
    data = pd.Series(data['<CLOSE>']).map(float)
    data = data.resample('M').fillna(method='pad')
    data = preprocessing.minmax_scale(data)
    data_t = data[6:]
    data_f = data.reshape(-1, 6)
    data_f = np.array([data[i:i + 6] for i in range(data.shape[0] - 6 + 1)])
    np.save(file_out[0], data_f[:len(data_f) - 1])
    np.save(file_out[1], data_t)


if __name__ == '__main__':
    for fx in FX_LIST:
        path_f_in = '%s/%s.txt' % (FILE_PREX, fx)
        path_f_final = ['%s/%s_MON_f.npy' % (FILE_PREX, fx),
                        '%s/%s_MON_t.pkl' % (FILE_PREX, fx)]
        m2m(path_f_in, path_f_final)
