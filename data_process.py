"""Data process"""

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessHour
import pandas as pd
import numpy as np

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
    len_data = len(data)
    data = data.reshape(len_data / 24, 24)
    data = np.array([data[i:i + 24] for i in range(len_data / 24 - 24 + 1)])
    data_s = {
        'open_price': np.array([data[i][0][0]
                                for i in range(data.shape[0] - 1)]),
        'close_price': np.array([data[i][NUM_PIX / 24 - 1][23]
                                 for i in range(data.shape[0] - 1)]),
        'max_price': np.array([data[i].max()
                               for i in range(data.shape[0] - 1)]),
        'min_price': np.array([data[i].min()
                               for i in range(data.shape[0] - 1)]),
        'buy_or_sell': np.array(
            [data[i + 1][NUM_PIX / 24 - 1][23] > data[i + 1][0][0]
             for i in range(data.shape[0] - 1)])}
    data_s = pd.DataFrame(data_s)
    data = data.reshape(len(data), NUM_PIX)
    np.save(file_out[0], data[:len(data) - 1])
    data_s.to_pickle(file_out[1])

if __name__ == '__main__':
    one2two()
