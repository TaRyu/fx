
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessHour
import pandas as pd
import numpy as np

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY']
FILE_PREX = '../data/fx/prediction'
CUSTOM_BH = CustomBusinessHour(calendar=USFederalHolidayCalendar(),
                               start='00:00', end="01:00")
RANGE_TIME = pd.DatetimeIndex(start='20150106',
                              end='20160531', freq=CUSTOM_BH)


def show_pre(input_file='%s/EURUSDprediction.pkl' % FILE_PREX):
    pre = pd.read_pickle(input_file).reset_index()
    t = pre['close_price'][1:].values
    p = np.array([pre['open_price'][i + 1] / (1 - pre['predict'][i] / 100)
                  for i in range(len(t))])
    data = {'Real rates': t, 'Forecast of rates': p}
    df = pd.DataFrame(data, index=RANGE_TIME)
    return df, t, p

if __name__ == '__main__':
    df, t, p = show_pre()
