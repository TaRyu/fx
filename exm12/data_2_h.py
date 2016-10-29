"""此程序为将分钟数据转换为小时数据。只考虑啦交易日，缺省值自动填充为前一个值"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessHour
import pandas as pd

FX_LIST = ['USDJPY', 'EURUSD', 'GBPUSD',
           'AUDUSD', 'EURJPY', 'EURGBP']
FILE_PREX = '../../../data/fx'
CUSTOM_BH = CustomBusinessHour(calendar=USFederalHolidayCalendar(),
                               start='00:00', end="23:59")
RANGE_TIME = pd.DatetimeIndex(start='200101030000',
                              end='201605312300', freq=2 * CUSTOM_BH)


def m2h(file_in, file_out):
    data = pd.read_csv(file_in, dtype='str')
    data['DateTime'] = pd.to_datetime(
        data['<DTYYYYMMDD>'].map(str) + data['<TIME>'].map(str),
        format='%Y%m%d%H%M%S')
    data = data.set_index('DateTime')
    data = pd.Series(data['<CLOSE>']).map(float)
    data = data.resample('H', how='ohlc').fillna(method='pad')
    data = data.reindex(RANGE_TIME)
    data.to_pickle(file_out)
    return data


if __name__ == '__main__':
    for fx in FX_LIST:
        file_in = '%s/%s.txt' % (FILE_PREX, fx)
        file_out = '%s/H/%s_2H.pkl' % (FILE_PREX, fx)
        test = m2h(file_in, file_out)
    print(test)
