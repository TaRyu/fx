"""Data process"""

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessHour
import pandas as pd

PATH_FILE_IN = '../data/fx/EURUSD.csv'
PATH_FILE_OUT = 'EURUSD_H.pkl'
CUSTOM_BH = CustomBusinessHour(calendar=USFederalHolidayCalendar(),
                               start='00:00', end="23:59")
RANGE_TIME = pd.DatetimeIndex(start='200101030000',
                              end='201605312300', freq=CUSTOM_BH)


def m2h():
    data = pd.read_csv(PATH_FILE_IN, dtype='str')
    data['DateTime'] = pd.to_datetime(
        data['<DTYYYYMMDD>'].map(str) + data['<TIME>'].map(str),
        format='%Y%m%d%H%M%S')
    data = data.set_index('DateTime')
    data = pd.Series(data['<CLOSE>']).map(float)
    data = data.resample('H', how='ohlc').fillna(method='pad')
    data = data.reindex(RANGE_TIME)
    data.to_pickle(PATH_FILE_OUT)

if __name__ == '__main__':
    m2h()
