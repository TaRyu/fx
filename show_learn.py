import pandas as pd

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY']
FILE_PREX = '../data/fx/learn'


def read_learn(input_file='%s/EURUSD.csv' % FILE_PREX):
    data = pd.read_csv(input_file, index_col='Step')['Value']
    return data


if __name__ == '__main__':
    learn = read_learn()
