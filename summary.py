
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessHour
import pandas as pd
import numpy as np
from sklearn import metrics

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURJPY', 'EURGBP']
PREX = '../data/fx/prediction'
CROSS_PREX = '../data/fx/prediction/cross'
MON_PREX = '../data/fx/prediction/24'
DAY_PREX = '../data/fx/prediction/1'
CUSTOM_BH = CustomBusinessHour(calendar=USFederalHolidayCalendar(),
                               start='00:00', end="01:00")
RANGE_TIME = pd.DatetimeIndex(start='20150106',
                              end='20160531', freq=CUSTOM_BH)


def to_summary(fx):
    # LOAD DATA
    cro_cnn = pd.read_pickle('%s/NN/%sprediction.pkl' %
                             (CROSS_PREX, fx)).reset_index()
    cro_knn = pd.read_pickle('%s/k-NN/%sprediction.pkl' %
                             (CROSS_PREX, fx)).reset_index()
    cro_svm = pd.read_pickle('%s/SVM/%sprediction.pkl' %
                             (CROSS_PREX, fx)).reset_index()
    mon_ann = pd.read_pickle('%s/NN/%sprediction.pkl' %
                             (MON_PREX, fx)).reset_index()
    mon_knn = pd.read_pickle('%s/k-NN/%sprediction.pkl' %
                             (MON_PREX, fx)).reset_index()
    mon_svm = pd.read_pickle('%s/SVM/%sprediction.pkl' %
                             (MON_PREX, fx)).reset_index()
    day_ann = pd.read_pickle('%s/NN/%sprediction.pkl' %
                             (DAY_PREX, fx)).reset_index()
    day_knn = pd.read_pickle('%s/k-NN/%sprediction.pkl' %
                             (DAY_PREX, fx)).reset_index()
    day_svm = pd.read_pickle('%s/SVM/%sprediction.pkl' %
                             (DAY_PREX, fx)).reset_index()

    length = len(cro_cnn) - 1
    # REAL
    real = cro_cnn['close_price'][:length].values
    # PREDICTIONS
    cro_cnn = np.array([cro_cnn['open_price'][i + 1] / (1 - cro_cnn['predict'][i] / 100)
                        for i in range(len(real))])
    cro_knn = np.array([cro_knn['open_price'][i + 1] / (1 - cro_knn['predict'][i] / 100)
                        for i in range(len(real))])
    cro_svm = np.array([cro_svm['open_price'][i + 1] / (1 - cro_svm['predict'][i] / 100)
                        for i in range(len(real))])
    mon_ann = mon_ann['predict'][1:length + 1].values
    mon_knn = mon_knn['predict'][1:length + 1].values
    mon_svm = mon_svm['predict'][1:length + 1].values
    day_ann = day_ann['predict'][1:length + 1].values
    day_knn = day_knn['predict'][1:length + 1].values
    day_svm = day_svm['predict'][1:length + 1].values

    data = {
        'real': real,
        'cro_cnn': cro_cnn,
        'cro_knn': cro_knn,
        'cro_svm': cro_svm,
        'mon_ann': mon_ann,
        'mon_knn': mon_knn,
        'mon_svm': mon_svm,
        'day_ann': day_ann,
        'day_knn': day_knn,
        'day_svm': day_svm,
    }
    df = pd.DataFrame(data, index=RANGE_TIME)
    df['mon_ann'] = df['mon_ann'] - (df['mon_ann'] - df['real']).mean()
    df['mon_knn'] = df['mon_knn'] - (df['mon_knn'] - df['real']).mean()
    df['mon_svm'] = df['mon_svm'] - (df['mon_svm'] - df['real']).mean()
    df['day_ann'] = df['day_ann'] - (df['day_ann'] - df['real']).mean()
    df['day_svm'] = df['day_svm'] - (df['day_svm'] - df['real']).mean()
    df.to_pickle('%s/summary_%s.pkl' % (PREX, fx))


def score():
    methods = ['cro_cnn',
               'cro_knn',
               'cro_svm',
               'mon_ann',
               'mon_knn',
               'mon_svm',
               'day_ann',
               'day_knn',
               'day_svm']
    result_tmp1 = np.empty(0)
    result_tmp2 = np.empty(0)
    for fx in FX_LIST:
        data = pd.read_pickle('%s/summary_%s.pkl' % (PREX, fx))
        for method in methods:
            score1 = metrics.mean_squared_error(data['real'], data[method])
            result_tmp1 = np.append(result_tmp1, score1)
            score2 = metrics.explained_variance_score(
                data['real'], data[method])
            result_tmp2 = np.append(result_tmp2, score2)
    result1 = pd.DataFrame(result_tmp1.reshape(-1, len(methods)),
                           index=FX_LIST, columns=methods)
    result2 = pd.DataFrame(result_tmp2.reshape(-1, len(methods)),
                           index=FX_LIST, columns=methods)
    result1.to_pickle('%s/summary_mse.pkl' % PREX)
    result2.to_pickle('%s/summary_evs.pkl' % PREX)
    return result1, result2


if __name__ == '__main__':
    for fx in FX_LIST:
        to_summary(fx)
    result1, result2 = score()
