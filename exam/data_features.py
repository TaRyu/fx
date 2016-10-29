"""此程序提取了特征值和目标值。时间尺度为24个交易日。"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD']
FILE_PREX = '../../../data/fx'
NUM_PIX = 24 * 24
SCALE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def get_fs_t(file_in, file_out, i):
    data = pd.read_pickle(file_in)['close']
    data = data.reshape(-1, 24)
    data = np.float32([data[i:i + 24]
                       for i in range(data.shape[0] - 24 + 1)])
    data = data.reshape(-1, NUM_PIX)
    data_t = {
        'open_price': np.float32([data[i][0]
                                  for i in range(data.shape[0] - i)]),
        'close_price': np.float32([data[i][-1]
                                   for i in range(data.shape[0] - i)]),
        'max_price': np.float32([data[i].max()
                                 for i in range(data.shape[0] - i)]),
        'min_price': np.float32([data[i].min()
                                 for i in range(data.shape[0] - i)]),
        'mean_price': np.float32([data[i].mean()
                                  for i in range(data.shape[0] - i)]),
        'median_price': np.float32([np.median(data[i])
                                    for i in range(data.shape[0] - i)]),
        'buy_or_sell': np.int_(
            [int(data[i + i][-1] > data[i + i][0])
             for i in range(data.shape[0] - i)]),
        'change': np.float32(
            [(data[i + i][-1] - data[i + i][0]) /
             data[i + i][0] * 100
             for i in range(data.shape[0] - i)]),
        'target_open': np.float32([data[i + i][0]
                                   for i in range(data.shape[0] - i)]),
        'real_target': np.float32([data[i + i][-1]
                                   for i in range(data.shape[0] - i)])
    }
    data_t = pd.DataFrame(data_t)
    np.save(file_out[0], data[:len(data) - i])
    data_t.to_pickle(file_out[1])


def get_24(i):
    for fx in FX_LIST:
        file_in = '%s/H/%s.pkl' % (FILE_PREX, fx)
        file_out = ['%s/Fs/%s_%i.npy' %
                    (FILE_PREX, fx, i),
                    '%s/T/%s_%i.pkl' %
                    (FILE_PREX, fx, i)]
        get_fs_t(file_in, file_out, i)


def get_fs_t_5(file_in, file_out, i):
    data = pd.read_pickle(file_in)['close']
    data = data.reshape(-1, 24)
    data = np.float32([[data[i + x][-1] for
                        x in range(5 * i) if x % i == 0]
                       for i in range(len(data) - 5 * i + 1)])
    data = data.reshape(-1, 5)
    data_t = {
        'change': np.float32(
            [(data[i + i][-1] - data[i + i][0]) /
             data[i + i][0] * 100
             for i in range(data.shape[0] - i)]),
        'target_open': np.float32([data[i + i][0]
                                   for i in range(data.shape[0] - i)]),
        'real_target': np.float32([data[i + i][-1]
                                   for i in range(data.shape[0] - i)])
    }
    data_t = pd.DataFrame(data_t)
    np.save(file_out[0], data[:len(data) - i])
    data_t.to_pickle(file_out[1])


def get_5(i):
    for fx in FX_LIST:
        file_in = '%s/H/%s.pkl' % (FILE_PREX, fx)
        file_out = ['%s/Fs/%s_5_%i.npy' %
                    (FILE_PREX, fx, i),
                    '%s/T/%s_5_%i.pkl' %
                    (FILE_PREX, fx, i)]
        get_fs_t(file_in, file_out, i)


if __name__ == '__main__':
    for i in SCALE:
        get_24(i)
        get_5(i)
