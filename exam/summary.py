import pandas as pd

FILE_PREX = '../../../data/fx'
FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD']
SCALE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def join_df():
    for i in SCALE:
        cnn_mse = pd.read_pickle('%s/exam_mse_%i.pkl' % (FILE_PREX, i))
        cnn_mape = pd.read_pickle('%s/exam_mape_%i.pkl' % (FILE_PREX, i))
        cnn_evs = pd.read_pickle('%s/exam_evs_%i.pkl' % (FILE_PREX, i))
        other_mse = pd.read_pickle('%s/exam_mse_5_%i.pkl' % (FILE_PREX, i))
        other_mape = pd.read_pickle('%s/exam_mse_5_%i.pkl' % (FILE_PREX, i))
        other_evs = pd.read_pickle('%s/exam_mse_5_%i.pkl' % (FILE_PREX, i))
        pass



