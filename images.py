import pandas as pd


PREX = '../data/fx/prediction'


def cross_prediction(fx):
    data = pd.read_pickle('%s/summary_%s.pkl' % (PREX, fx))
    data = {
        'Real': data['real'],
        'CNN on cross time scale': data['cro_cnn']
    }
    columns = ['Real', 'CNN on cross time scale']
    data = pd.DataFrame(data, columns=columns)
    return data


def me_prediction(fx='EURUSD'):
    data = pd.read_pickle('%s/summary_%s.pkl' % (PREX, fx))
    data = {
        'Real': data['real'],
        'CNN on cross time scale': data['cro_cnn'],
        'k-NN on cross time scale': data['cro_knn'],
        'SVM on cross time scale': data['cro_svm']
    }
    columns = ['Real', 'CNN on cross time scale',
               'k-NN on cross time scale', 'SVM on cross time scale']
    data = pd.DataFrame(data, columns=columns)
    return data


def mon_prediction(fx='EURUSD'):
    data = pd.read_pickle('%s/summary_%s.pkl' % (PREX, fx))
    data = {
        'Real': data['real'],
        'CNN on cross time scale': data['cro_cnn'],
        'ANN on 24-days scale': data['mon_ann'],
        'k-NN on 24-days scale': data['mon_knn'],
        'SVM on 24-days scale': data['mon_svm']
    }
    columns = ['Real', 'CNN on cross time scale', 'ANN on 24-days scale',
               'k-NN on 24-days scale', 'SVM on 24-days scale']
    data = pd.DataFrame(data, columns=columns)
    return data


def day_prediction(fx='EURUSD'):
    data = pd.read_pickle('%s/summary_%s.pkl' % (PREX, fx))
    data = {
        'Real': data['real'],
        'CNN on cross time scale': data['cro_cnn'],
        'ANN on 1-day scale': data['day_ann'],
        'k-NN on 1-day scale': data['day_knn'],
        'SVM on 1-day scale': data['day_svm']
    }
    columns = ['Real', 'CNN on cross time scale', 'ANN on 1-day scale',
               'k-NN on 1-day scale', 'SVM on 1-day scale']
    data = pd.DataFrame(data, columns=columns)
    return data

if __name__ == '__main__':
    eu = cross_prediction('EURUSD')
    uj = cross_prediction('USDJPY')
    gu = cross_prediction('GBPUSD')
    au = cross_prediction('AUDUSD')
    ej = cross_prediction('EURJPY')
    eg = cross_prediction('EURGBP')
    cross = me_prediction()
    mon = mon_prediction()
    day = day_prediction()
