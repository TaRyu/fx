import pandas as pd


FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD',
           'AUDUSD', 'EURJPY', 'EURGBP']
FILE_PREX = '../../data/fx/app/data'


def get_data(fx):
    data = pd.read_pickle('%s/%s_H.pkl' % (FILE_PREX, fx))[-720:]
    data.to_csv('../../data/fx/app/%s.csv' % fx)

if __name__ == '__main__':
    for fx in FX_LIST:
        get_data(fx)
