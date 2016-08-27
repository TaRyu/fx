import pandas as pd
import numpy as np
from tensorflow.contrib import learn

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD',
           'AUDUSD', 'EURJPY', 'EURGBP']
FILE_PREX = '../../data/fx/app'
NUM_PIX = 24 * 24


def data_process(fx='EURUSD'):
    data = pd.read_csv('%s/%s.csv' % (FILE_PREX, fx))
    data = data['close'].reshape(-1, 24)
    data = np.array([data[i:i + 24] for i in range(data.shape[0] - 24 + 1)])
    data = data.reshape(len(data), NUM_PIX)
    np.save('%s/%s.npy' % (FILE_PREX, fx))
    return data


def load_predict(fx='EURUSD'):
    data = data_process(fx)
    data = np.array([(data[i] - data[i].min()) / (data[i].max() -
                                                  data[i].min()) for i in range(data.shape[0])])
    re = learn.TensorFlowEstimator.restore(
        '%s/tensorboard_models/%s' % (FILE_PREX, fx))
    prediction = re.predict(data)
    return prediction

if __name__ == '__main__':
    df = pd.DataFrame()
    for fx in FX_LIST:
        df['%s' % fx] = load_predict(fx)
    df.to_pickle('%s/prediction.pkl')
