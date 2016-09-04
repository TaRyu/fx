"""Data process"""

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessHour
import pandas as pd
import numpy as np

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD',
           'AUDUSD', 'EURJPY', 'EURGBP']
FILE_PREX = '../data/fx'
PATH_FILE_IN = '../data/fx/EURUSD.csv'
PATH_FILE_OUT = 'EURUSD_H.pkl'
PATH_FILE_FINAL = ['EURUSD_FINAL_M.npy', 'EURUSD_FINAL_S.pkl']
NUM_PIX = 24 * 24
CUSTOM_BH = CustomBusinessHour(calendar=USFederalHolidayCalendar(),
                               start='00:00', end="23:59")
RANGE_TIME = pd.DatetimeIndex(start='200101030000',
                              end='201605312300', freq=CUSTOM_BH)


def one2two(file_in=PATH_FILE_OUT, file_out=PATH_FILE_FINAL):
    data = pd.read_pickle(file_in)['close']
    data = np.array([data[i:i + 576] for i in range(data.shape[0] - 576 + 1)])
    data = data.reshape(-1, 576)
    data_s = {
        'open_price': np.array([data[i][0]
                                for i in range(data.shape[0] - 576)]),
        'close_price': np.array([data[i][575]
                                 for i in range(data.shape[0] - 576)]),
        'max_price': np.array([data[i].max()
                               for i in range(data.shape[0] - 576)]),
        'min_price': np.array([data[i].min()
                               for i in range(data.shape[0] - 576)]),
        'mean_price': np.array([data[i].mean()
                                for i in range(data.shape[0] - 576)]),
        'median_price': np.array([np.median(data[i])
                                  for i in range(data.shape[0] - 576)]),
        'buy_or_sell': np.array(
            [int(data[i + 576][575] > data[i + 576][0])
             for i in range(data.shape[0] - 576)]),
        'change': np.array(
            [(data[i + 576][575] - data[i + 576][0]) /
             data[i + 576][575] * 100
             for i in range(data.shape[0] - 576)])}
    data_s = pd.DataFrame(data_s)
    bins = [-100, -5, -4, -3, -2, -1.5, -1, -
            0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 100]
    bins = [0.01 * x for x in bins]
    labels = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8]
    data_s['change_D_16'] = pd.cut(data_s['change'], bins, labels=labels)
    bins = [-100, -5, -2, 0, 2, 5, 100]
    bins = [0.01 * x for x in bins]
    labels = [-3, -2, -1, 1, 2, 3]
    data_s['change_D'] = pd.cut(data_s['change'], bins, labels=labels)
    np.save(file_out[0], data[:len(data) - 576])
    data_s.to_pickle(file_out[1])

if __name__ == '__main__':
    for fx in FX_LIST:
        path_f_out = '%s/%s_H.pkl' % (FILE_PREX, fx)
        path_f_final = ['%s/%s_FINAL_M_new100.npy' % (FILE_PREX, fx),
                        '%s/%s_FINAL_S_new100.pkl' % (FILE_PREX, fx)]
        one2two(path_f_out, path_f_final)
