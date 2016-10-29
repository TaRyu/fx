"""Predict based on cnn"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import metrics, svm

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD']
FX_LIST = ['EURUSD']  # !
FILE_PREX = '../../../data/fx'
NUM_PIX = 24 * 24
optimizers = ['Adagrad']
models = ['CNN', 'ANN-10', 'ANN-15', 'ANN-20', 'SVM']
models = ['CNN', 'SVM']  # !
time_format = '%Y%m%d%H%M'
SCALE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# SCALE = [2, 3, 4, 5]  # !
SCALE = [1]
# optimizers = ['GradientDescent', 'Adadelta',
#               'Momentum', 'Adam', 'Ftrl', 'RMSProp', 'SGD', 'Adagrad']


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def conv_model(X, y):
    X = tf.reshape(X, [-1, 24, 24, 1])
    # first conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('Conv1'):
        h_conv1 = learn.ops.conv2d(X, n_filters=32, filter_shape=[5, 5],
                                   bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    # second conv layer will compute 64 features for each 5x5 patch
    with tf.variable_scope('Conv2'):
        h_conv2 = learn.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                                   bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 64])
    # densely connected layer with 1024 neurons
    with tf.variable_scope('F-C'):
        h_fc1 = learn.ops.dnn(
            h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
    with tf.variable_scope('LR'):
        o_linear = learn.models.linear_regression(h_fc1, y)
    return o_linear


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def model10(x, y):
    layers = learn.ops.dnn(x, [10], dropout=0.5)
    return learn.models.linear_regression(layers, y)


def model15(x, y):
    layers = learn.ops.dnn(x, [15], dropout=0.5)
    return learn.models.linear_regression(layers, y)


def model20(x, y):
    layers = learn.ops.dnn(x, [20], dropout=0.5)
    return learn.models.linear_regression(layers, y)


def train_cnn():
    steps = 1
    for i in SCALE:
        result_tmp0 = np.empty(0)
        result_tmp1 = np.empty(0)
        result_tmp2 = np.empty(0)
        # df = pd.DataFrame()
        for fx in FX_LIST:
            result_tmp3 = np.empty(0)
            # fs_t_path = ['%s/NFs/%s_%i.npy' % (FILE_PREX, fx, i),
            #              '%s/T/%s_%i.pkl' % (FILE_PREX, fx, i)]
            # fs = np.load(fs_t_path[0])
            # t = pd.read_pickle(fs_t_path[1])
            f_train = np.load('%s/NFs/%s_train_%i.npy' % (FILE_PREX, fx, i))
            f_test = np.load('%s/NFs/%s_test_%i.npy' % (FILE_PREX, fx, i))
            t_train = pd.read_pickle('%s/T/%s_train_%i.pkl' %
                                     (FILE_PREX, fx, i))
            t_test = pd.read_pickle('%s/T/%s_test_%i.pkl' % (FILE_PREX, fx, i))
            for optimizer in optimizers:
                start = time.strftime(time_format, time.localtime())
                print('%s start at %s.' % (fx, start))
                model = learn.TensorFlowEstimator(
                    model_fn=conv_model,
                    n_classes=0,
                    batch_size=80, steps=steps,
                    optimizer=optimizer,
                    learning_rate=0.001)
                logdir = '%s/tensorboard_models/exam/%s/%s' % (
                    FILE_PREX,
                    optimizer,
                    fx)
                # model.fit(fs[:-num_test],
                #           t['change'][:-num_test],
                #           logdir=logdir)
                model.fit(f_train,
                          t_train['change'].values,
                          logdir=logdir)
                model.save('%s/saves/exam/%s/%s' % (FILE_PREX, optimizer, fx))
                # prediction1 = model.predict(fs[-num_test:])
                # prediction2 = (prediction1 / 100 + 1) * \
                #     t['target_open'][-num_test:]
                # score0 = mean_absolute_percentage_error(
                #     t['real_target'][-num_test:].values, prediction2)
                # score1 = metrics.explained_variance_score(
                #     t['change'][-num_test:].values, prediction1)
                # score2 = metrics.mean_squared_error(
                #     t['real_target'][-num_test:].values, prediction2)
                prediction1 = model.predict(f_test)
                prediction2 = (prediction1 / 100 + 1) * \
                    t_test['target_open'].values
                score0 = mean_absolute_percentage_error(
                    t_test['real_target'].values, prediction2)
                score1 = metrics.explained_variance_score(
                    t_test['change'].values, prediction1)
                score2 = metrics.mean_squared_error(
                    t_test['real_target'].values, prediction2)
                result_tmp0 = np.append(result_tmp0, score0)
                print(result_tmp0)
                result_tmp1 = np.append(result_tmp1, score1)
                print(result_tmp1)
                result_tmp2 = np.append(result_tmp2, score2)
                print(result_tmp2)
                result_tmp3 = np.append(result_tmp3, prediction2)
                end = time.strftime(time_format, time.localtime())
                print('%s end at %s.' % (fx, end))
            result_tmp3 = pd.DataFrame(
                result_tmp3.reshape(-1, len(optimizers)), columns=optimizers)
            result_tmp3['real'] = t_test['real_target'].values
            result_tmp3.to_pickle('%s/pre_result/%s_%i,pkl' %
                                  (FILE_PREX, fx, i))
        result0 = pd.DataFrame(result_tmp0.reshape(-1, len(optimizers)),
                               index=FX_LIST, columns=optimizers)
        print(result0)
        result1 = pd.DataFrame(result_tmp1.reshape(-1, len(optimizers)),
                               index=FX_LIST, columns=optimizers)
        print(result1)
        result2 = pd.DataFrame(result_tmp2.reshape(-1, len(optimizers)),
                               index=FX_LIST, columns=optimizers)
        print(result2)
        result0.to_pickle('%s/exam_mape_%i.pkl' % (FILE_PREX, i))
        result1.to_pickle('%s/exam_evs_%i.pkl' % (FILE_PREX, i))
        result2.to_pickle('%s/exam_mse_%i.pkl' % (FILE_PREX, i))


def train_5():
    steps = 1
    for i in SCALE:
        result_tmp0 = np.empty(0)
        result_tmp1 = np.empty(0)
        result_tmp2 = np.empty(0)
        # df = pd.DataFrame()
        for fx in FX_LIST:
            result_tmp3 = np.empty(0)
            # fs_t_path = ['%s/NFs/%s_%i.npy' % (FILE_PREX, fx, i),
            #              '%s/T/%s_%i.pkl' % (FILE_PREX, fx, i)]
            # fs = np.load(fs_t_path[0])
            # t = pd.read_pickle(fs_t_path[1])
            f_train = np.load('%s/NFs/%s_train_5_%i.npy' % (FILE_PREX, fx, i))
            f_test = np.load('%s/NFs/%s_test_5_%i.npy' % (FILE_PREX, fx, i))
            t_train = pd.read_pickle('%s/T/%s_train_5_%i.pkl' %
                                     (FILE_PREX, fx, i))
            t_test = pd.read_pickle(
                '%s/T/%s_test_5_%i.pkl' % (FILE_PREX, fx, i))
            for name in models:
                start = time.strftime(time_format, time.localtime())
                print('%s start at %s.' % (fx, start))
                logdir = '%s/tensorboard_models/exam/%s/%s' % (
                    FILE_PREX,
                    name,
                    fx)
                if name == 'ANN-10':
                    model = learn.TensorFlowEstimator(
                        model_fn=model10,
                        n_classes=0, optimizer='Adagrad',
                        batch_size=80, steps=steps,
                        learning_rate=0.001)
                    model.fit(f_train,
                              t_train['change'].values,
                              logdir=logdir)
                    model.save('%s/saves/exam/%s/%s' % (FILE_PREX, name, fx))
                elif name == 'ANN-15':
                    model = learn.TensorFlowEstimator(
                        model_fn=model15,
                        n_classes=0, optimizer='Adagrad',
                        batch_size=80, steps=steps,
                        learning_rate=0.001)
                    model.fit(f_train,
                              t_train['change'].values,
                              logdir=logdir)
                    model.save('%s/saves/exam/%s/%s' % (FILE_PREX, name, fx))
                elif name == 'ANN-20':
                    model = learn.TensorFlowEstimator(
                        model_fn=model20,
                        n_classes=0, optimizer='Adagrad',
                        batch_size=80, steps=steps,
                        learning_rate=0.001)
                    model.fit(f_train,
                              t_train['change'].values)
                    model.save('%s/saves/exam/%s/%s' % (FILE_PREX, name, fx))
                else:
                    model = svm.SVR()
                    model.fit(f_train,
                              t_train['change'].values)
                # model.fit(fs[:-num_test],
                #           t['change'][:-num_test],
                #           logdir=logdir)
                # prediction1 = model.predict(fs[-num_test:])
                # prediction2 = (prediction1 / 100 + 1) * \
                #     t['target_open'][-num_test:]
                # score0 = mean_absolute_percentage_error(
                #     t['real_target'][-num_test:].values, prediction2)
                # score1 = metrics.explained_variance_score(
                #     t['change'][-num_test:].values, prediction1)
                # score2 = metrics.mean_squared_error(
                #     t['real_target'][-num_test:].values, prediction2)
                prediction1 = model.predict(f_test)
                prediction2 = (prediction1 / 100 + 1) * \
                    t_test['target_open'].values
                score0 = mean_absolute_percentage_error(
                    t_test['real_target'].values, prediction2)
                score1 = metrics.explained_variance_score(
                    t_test['change'].values, prediction1)
                score2 = metrics.mean_squared_error(
                    t_test['real_target'].values, prediction2)
                result_tmp0 = np.append(result_tmp0, score0)
                print(result_tmp0)
                result_tmp1 = np.append(result_tmp1, score1)
                print(result_tmp1)
                result_tmp2 = np.append(result_tmp2, score2)
                print(result_tmp2)
                result_tmp3 = np.append(result_tmp3, prediction2)
                end = time.strftime(time_format, time.localtime())
                print('%s end at %s.' % (fx, end))
            result_tmp3 = pd.DataFrame(
                result_tmp3.reshape(-1, len(models)), columns=models)
            result_tmp3['real'] = t_test['real_target'].values
            result_tmp3.to_pickle('%s/pre_result/%s_5_%i,pkl' %
                                  (FILE_PREX, fx, i))
        result0 = pd.DataFrame(result_tmp0.reshape(-1, len(models)),
                               index=FX_LIST, columns=models)
        print(result0)
        result1 = pd.DataFrame(result_tmp1.reshape(-1, len(models)),
                               index=FX_LIST, columns=models)
        print(result1)
        result2 = pd.DataFrame(result_tmp2.reshape(-1, len(models)),
                               index=FX_LIST, columns=models)
        print(result2)
        result0.to_pickle('%s/exam_mape_5_%i.pkl' % (FILE_PREX, i))
        result1.to_pickle('%s/exam_evs_5_%i.pkl' % (FILE_PREX, i))
        result2.to_pickle('%s/exam_mse_5_%i.pkl' % (FILE_PREX, i))


def train_all_models():
    columns = np.empty(0)
    steps = 20000   # !
    batch_size = 30
    learning_rate = 0.001
    result_tmp0 = np.empty(0)
    for i in SCALE:
        for fx in FX_LIST:
            result_tmp1 = pd.DataFrame()
            result_tmp2 = pd.DataFrame()
            for name in models:
                if i == SCALE[-1]:
                    columns = np.append(columns, '%s%s_MSE' % (fx, name))
                    columns = np.append(columns, '%s%s_MAPE' % (fx, name))
                    # columns = np.append(columns, '%s%s_EVS' % (fx, name))
                    columns = np.append(columns, '%s%s_R2' % (fx, name))
                    columns = np.append(columns, '%s%s_R2_R' % (fx, name))
                    # columns = np.append(columns, '%s%s_MAPE_C' % (fx, name))
                start = time.strftime(time_format, time.localtime())
                print('%s with %s for h=%i start at %s.' %
                      (fx, name, i, start))
                logdir = '%s/tensorboard_models/exam/%s/%s_%i' % (
                    FILE_PREX,
                    name,
                    fx,
                    i)
                if name == 'CNN':
                    f_train = np.load('%s/NFs/%s_train_%i.npy' %
                                      (FILE_PREX, fx, i))
                    f_test = np.load('%s/NFs/%s_test_%i.npy' %
                                     (FILE_PREX, fx, i))
                    f_plot = np.load('%s/NFs/%s_plot_%i.npy' %
                                     (FILE_PREX, fx, i))
                    t_train = pd.read_pickle('%s/T/%s_train_%i.pkl' %
                                             (FILE_PREX, fx, i))
                    t_test = pd.read_pickle(
                        '%s/T/%s_test_%i.pkl' % (FILE_PREX, fx, i))
                    t_plot = pd.read_pickle(
                        '%s/T/%s_plot_%i.pkl' % (FILE_PREX, fx, i))
                    model = learn.TensorFlowEstimator(
                        model_fn=conv_model,
                        n_classes=0,
                        batch_size=batch_size, steps=steps,
                        optimizer='Adagrad',
                        learning_rate=learning_rate)
                    model.fit(f_train,
                              t_train['change'].values,
                              logdir=logdir)
                    model.save('%s/saves/exam/%s/%s_%i' %
                               (FILE_PREX, name, fx, i))
                else:
                    f_train = np.load('%s/NFs/%s_train_5_%i.npy' %
                                      (FILE_PREX, fx, i))
                    f_test = np.load('%s/NFs/%s_test_5_%i.npy' %
                                     (FILE_PREX, fx, i))
                    f_plot = np.load('%s/NFs/%s_plot_5_%i.npy' %
                                     (FILE_PREX, fx, i))
                    t_train = pd.read_pickle('%s/T/%s_train_5_%i.pkl' %
                                             (FILE_PREX, fx, i))
                    t_test = pd.read_pickle(
                        '%s/T/%s_test_5_%i.pkl' % (FILE_PREX, fx, i))
                    t_plot = pd.read_pickle(
                        '%s/T/%s_plot_5_%i.pkl' % (FILE_PREX, fx, i))
                    if name == 'ANN-10':
                        model = learn.TensorFlowEstimator(
                            model_fn=model10,
                            n_classes=0, optimizer='Adagrad',
                            batch_size=batch_size, steps=steps,
                            learning_rate=learning_rate)
                        model.fit(f_train,
                                  t_train['change'].values,
                                  logdir=logdir)
                        model.save('%s/saves/exam/%s/%s_%i' %
                                   (FILE_PREX, name, fx, i))
                    elif name == 'ANN-15':
                        model = learn.TensorFlowEstimator(
                            model_fn=model15,
                            n_classes=0, optimizer='Adagrad',
                            batch_size=batch_size, steps=steps,
                            learning_rate=learning_rate)
                        model.fit(f_train,
                                  t_train['change'].values,
                                  logdir=logdir)
                        model.save('%s/saves/exam/%s/%s_%i' %
                                   (FILE_PREX, name, fx, i))
                    elif name == 'ANN-20':
                        model = learn.TensorFlowEstimator(
                            model_fn=model20,
                            n_classes=0, optimizer='Adagrad',
                            batch_size=batch_size, steps=steps,
                            learning_rate=learning_rate)
                        model.fit(f_train,
                                  t_train['change'].values,
                                  logdir=logdir)
                        model.save('%s/saves/exam/%s/%s_%i' %
                                   (FILE_PREX, name, fx, i))
                    else:
                        model = svm.SVR()
                        model.fit(f_train,
                                  t_train['change'].values)
                prediction1 = model.predict(f_test)
                prediction2 = (prediction1 / 100 + 1) * \
                    t_test['target_open'].values
                prediction3 = model.predict(f_plot)
                prediction4 = (prediction3 / 100 + 1) * \
                    t_plot['target_open'].values
                score0 = metrics.mean_squared_error(
                    t_test['real_target'].values, prediction2)
                score1 = mean_absolute_percentage_error(
                    t_test['real_target'].values, prediction2)
                # score2 = metrics.explained_variance_score(
                #     t_test['real_target'].values, prediction2)
                score2 = metrics.r2_score(
                    t_test['change'].values, prediction1)
                score3 = metrics.r2_score(
                    t_test['real_target'].values, prediction2)
                result_tmp0 = np.append(result_tmp0, score0)
                result_tmp0 = np.append(result_tmp0, score1)
                result_tmp0 = np.append(result_tmp0, score2)
                result_tmp0 = np.append(result_tmp0, score3)
                result_tmp1['%s' % name] = prediction2
                result_tmp2['%s' % name] = prediction4
                end = time.strftime(time_format, time.localtime())
                print('%s with %s for h=%i end at %s.\
                    \nMSE: %f\nMAPE: %f\nR2: %f\nR2_R: %f' %
                      (fx, name, i, end, score0, score1, score2, score3))
            # result_tmp1 = pd.DataFrame(
            #     result_tmp1.reshape(len(models), -1), columns=models)
            result_tmp1['real'] = t_test['real_target'].values
            result_tmp1.to_pickle('%s/pre_result/%s_all_%i.pkl' %
                                  (FILE_PREX, fx, i))
            # result_tmp2 = pd.DataFrame(
            #     result_tmp2.reshape(-1, len(models)), columns=models)
            result_tmp2['real'] = t_plot['real_target'].values
            result_tmp2.to_pickle('%s/pre_result/%s_plot_%i.pkl' %
                                  (FILE_PREX, fx, i))
    result0 = pd.DataFrame(result_tmp0.reshape(-1, len(columns)),
                           index=SCALE, columns=columns)
    result0.to_pickle('%s/exam_all.pkl' % FILE_PREX)


if __name__ == '__main__':
    train_all_models()
