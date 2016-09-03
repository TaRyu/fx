import pandas as pd
import numpy as np

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY', 'EURGBP']
FILE_PREX = '../../data/fx/app'


if __name__ == '__main__':
    pre = pd.read_pickle('%s/prediction.pkl' % FILE_PREX)
    df = pd.DataFrame()
    for fx in FX_LIST:
        df['%s' % fx] = np.array([pre['%s_open' % fx][i + 1] / (1 - pre['%s' % fx][i] / 100)
                                  for i in range(len(pre) - 1)])
    df.to_pickle('../../data/fx/app/real_pre.pkl')
