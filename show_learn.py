import pandas as pd

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY']
FILE_PREX = '../data/fx/learn'


def read_learn():
    eurusd = pd.read_csv('%s/EURUSD.csv' %
                         FILE_PREX, index_col='Step')['Value'] / 100
    usdjpy = pd.read_csv('%s/USDJPY.csv' %
                         FILE_PREX, index_col='Step')['Value'] / 100
    gbpusd = pd.read_csv('%s/GBPUSD.csv' %
                         FILE_PREX, index_col='Step')['Value'] / 100
    audusd = pd.read_csv('%s/AUDUSD.csv' %
                         FILE_PREX, index_col='Step')['Value'] / 100
    eurjpy = pd.read_csv('%s/EURJPY.csv' %
                         FILE_PREX, index_col='Step')['Value'] / 100
    data = {'EURUSD': eurusd, 'USDJPY': usdjpy,
            'GBPUSD': gbpusd, 'AUDUSD': audusd, 'EURJPY': eurjpy}
    data = pd.DataFrame(data)
    data.to_pickle('%s/final.pkl' % FILE_PREX)


if __name__ == '__main__':
    read_learn()
