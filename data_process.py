"""Data process"""

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessHour
import pandas as pd
import numpy as np

FX_LIST = ['EURGBP', 'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY']
FX_LIST = ['EURGBP']
FILE_PREX = '../data/fx'
PATH_FILE_IN = '../data/fx/EURUSD.csv'
PATH_FILE_OUT = 'EURUSD_H.pkl'
PATH_FILE_FINAL = ['EURUSD_FINAL_M.npy', 'EURUSD_FINAL_S.pkl']
NUM_PIX = 24 * 24
CUSTOM_BH = CustomBusinessHour(calendar=USFederalHolidayCalendar(),
                               start='00:00', end="23:59")
RANGE_TIME = pd.DatetimeIndex(start='200101030000',
                              end='201605312300', freq=CUSTOM_BH)


def m2h(file_in=PATH_FILE_IN, file_out=PATH_FILE_OUT):
    data = pd.read_csv(file_in, dtype='str')
    data['DateTime'] = pd.to_datetime(
        data['<DTYYYYMMDD>'].map(str) + data['<TIME>'].map(str),
        format='%Y%m%d%H%M%S')
    data = data.set_index('DateTime')
    data = pd.Series(data['<CLOSE>']).map(float)
    data = data.resample('H', how='ohlc').fillna(method='pad')
    data = data.reindex(RANGE_TIME)
    data.to_pickle(file_out)


def one2two(file_in=PATH_FILE_OUT, file_out=PATH_FILE_FINAL):
    data = pd.read_pickle(file_in)['close']
    data = data.reshape(-1, 24)
    data = np.array([data[i:i + 24] for i in range(data.shape[0] - 24 + 1)])
    data_s = {
        'open_price': np.array([data[i][0][0]
                                for i in range(data.shape[0] - 1)]),
        'close_price': np.array([data[i][int(NUM_PIX / 24) - 1][23]
                                 for i in range(data.shape[0] - 1)]),
        'max_price': np.array([data[i].max()
                               for i in range(data.shape[0] - 1)]),
        'min_price': np.array([data[i].min()
                               for i in range(data.shape[0] - 1)]),
        'mean_price': np.array([data[i].mean()
                                for i in range(data.shape[0] - 1)]),
        'median_price': np.array([np.median(data[i])
                                  for i in range(data.shape[0] - 1)]),
        'buy_or_sell': np.array(
            [int(data[i + 1][int(NUM_PIX / 24) - 1][23] > data[i + 1][0][0])
             for i in range(data.shape[0] - 1)]),
        'change': np.array(
            [(data[i + 1][int(NUM_PIX / 24) - 1][23] - data[i + 1][0][0]) /
             data[i + 1][int(NUM_PIX / 24) - 1][23] * 100
             for i in range(data.shape[0] - 1)])}
    data_s = pd.DataFrame(data_s)
    bins = [-100, -5, -4, -3, -2, -1.5, -1, -
            0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 100]
    labels = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8]
    data_s['change_D_16'] = pd.cut(data_s['change'], bins, labels=labels)
    bins = [-100, -5, -2, 0, 2, 5, 100]
    labels = [-3, -2, -1, 1, 2, 3]
    data_s['change_D'] = pd.cut(data_s['change'], bins, labels=labels)
    data = data.reshape(len(data), NUM_PIX)
    np.save(file_out[0], data[:len(data) - 1])
    data_s.to_pickle(file_out[1])


if __name__ == '__main__':
    for fx in FX_LIST:
        path_f_in = '%s/%s.txt' % (FILE_PREX, fx)
        path_f_out = '%s/%s_H.pkl' % (FILE_PREX, fx)
        path_f_final = ['%s/%s_FINAL_M.npy' % (FILE_PREX, fx),
                        '%s/%s_FINAL_S.pkl' % (FILE_PREX, fx)]
        m2h(path_f_in, path_f_out)
        one2two(path_f_out, path_f_final)
