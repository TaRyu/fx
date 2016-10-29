"""此程序提取了特征值和目标值。时间尺度为24个交易日。"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD',
           'AUDUSD', 'EURJPY', 'EURGBP']
FILE_PREX = '../../../data/fx'
NUM_PIX = 12 * 12
SCALE = 3


def get_fs_t(file_in, file_out):
    data = pd.read_pickle(file_in)['close']
    data = data.reshape(-1, 12)
    data = np.float32([data[i:i + 12]
                       for i in range(data.shape[0] - 12 + 1)])
    data = data.reshape(-1, NUM_PIX)
    data_t = {
        'open_price': np.float32([data[i][0]
                                  for i in range(data.shape[0] - SCALE)]),
        'close_price': np.float32([data[i][-1]
                                   for i in range(data.shape[0] - SCALE)]),
        'max_price': np.float32([data[i].max()
                                 for i in range(data.shape[0] - SCALE)]),
        'min_price': np.float32([data[i].min()
                                 for i in range(data.shape[0] - SCALE)]),
        'mean_price': np.float32([data[i].mean()
                                  for i in range(data.shape[0] - SCALE)]),
        'median_price': np.float32([np.median(data[i])
                                    for i in range(data.shape[0] - SCALE)]),
        'buy_or_sell': np.int_(
            [int(data[i + SCALE][-1] > data[i + SCALE][0])
             for i in range(data.shape[0] - SCALE)]),
        'change': np.float32(
            [(data[i + SCALE][-1] - data[i + SCALE][0]) /
             data[i + SCALE][0] * 100
             for i in range(data.shape[0] - SCALE)])}
    data_t = pd.DataFrame(data_t)
    np.save(file_out[0], data[:len(data) - SCALE])
    data_t.to_pickle(file_out[1])


if __name__ == '__main__':
    for fx in FX_LIST:
        file_in = '%s/H/%s_2H.pkl' % (FILE_PREX, fx)
        file_out = ['%s/Fs/%s_2H.npy' %
                    (FILE_PREX, fx), '%s/T/%s_2H.pkl' % (FILE_PREX, fx)]
        get_fs_t(file_in, file_out)
