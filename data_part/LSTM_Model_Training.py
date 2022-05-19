# 导入需要的库
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from keras.integration_test.preprocessing_test_utils import BATCH_SIZE
from numpy.lib.format import BUFFER_SIZE


# 创建时间窗，时间窗是包含固定数值集合，此函数返回时间窗以供训练
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history`1_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


# 多变量时间窗口
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)  # step表示间隔采样步长，6表示每个小时只使用一个采样值（原数据集每10分钟采集一次）
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


# 创建步长
def create_time_steps(length):
    return list(range(-length, 0))


# 绘图函数
def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])  # 横轴刻度
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    return plt


# 训练数据基线
def baseline(history):
    return np.mean(history)


# LSTM单步单变量预测（针对于温度变量因子）
def LSTM1_predict(data):
    # 使用单变量单步进行预测，采用数据集中的一个特征（温度）训练模型，并使用训练模型进行预测
    uni_data = data['T']
    uni_data.index = data['Date_Time']
    # 测试数据集是否正确
    # uni_data.head()

    # 观察温度的整体走势
    uni_data.plot(subplots=True)
    plt.title('Distribution of ShenZhen Temperature')
    plt.xticks(rotation=20)
    plt.show()

    # 数据集标准化和归一化
    uni_data = uni_data.values
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()
    uni_data = (uni_data - uni_train_mean) / uni_train_std

    # 调参点
    univariate_past_history = 20
    univariate_future_target = 0
    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)
    # 验证集
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)

    print('Single window of past history')
    print(x_train_uni[0])
    print('\n Target temperature to predict')
    print(y_train_uni[0])

    show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Training goal of LSTM model')
    plt.show()
    show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
              'Training goal of LSTM Model with Baseline')
    plt.show()

    # 建模调参
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),  # input_shape=(20,1) 不包含批处理维度
        tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='adam', loss='mae')
    EVALUATION_INTERVAL = 200
    EPOCHS = 10
    # 训练模型
    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, validation_steps=50)
    print(val_univariate.take(3))
    for x, y in val_univariate.take(3):
        plot = show_plot([x[0].numpy(), y[0].numpy(),
                          simple_lstm_model.predict(x)[0]], 0, 'Predict Process Model')
        plot.show()


# 绘制损失曲线
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


# LSTM 多变量单步预测，选取会影响温度的三个变量因子来训练模型，这里选取温度，压强和相对湿度
def LSTM2_predict(data):
    # 选取表格中的因子
    feature_considered = ['T', 'Po', 'U']
    feature = data[feature_considered]
    feature.index = data['Date_Time']
    # 检查数据集
    # feature.head()
    feature.plot(subplots=True)
    plt.title('Distribution of variables')
    plt.xticks(rotation=20)
    # 观察数据和因子的走向趋势
    plt.show()
    # 数据标准化
    multi_data = feature.values
    multi_data_mean = multi_data[:TRAIN_SPLIT].mean(axis=0)
    multi_data_std = multi_data[:TRAIN_SPLIT].std(axis=0)
    multi_data = (multi_data - multi_data_mean) / multi_data_std
    past_history = 720
    future_target = 72
    STEP = 6

    x_train_single, y_train_single = multivariate_data(multi_data, multi_data[:, 1], 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True)
    x_val_single, y_val_single = multivariate_data(multi_data, multi_data[:, 1],
                                                   TRAIN_SPLIT, None, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
    # 划分数据集
    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    # 注意这里的BUFFER_SIZE和BATCH_SIZE
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
    # 建模
    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32,
                                               input_shape=x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))

    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    EVALUATION_INTERVAL = 200
    EPOCHS = 10
    single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_single,
                                                validation_steps=50)

    # 训练
    plot_train_history(single_step_history,
                       'Training loss and validation loss')
    for x, y in val_data_single.take(3):
        plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                          single_step_model.predict(x)[0]], 12,
                         'Predict Process Model')
        plot.show()


# 设置绘图画布格式
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['axes.grid'] = False

# 读取数据
data_set = pd.read_csv('./data/ShenZhen_Weather_data(full).csv')
# 输出看一下数据格式是否正确
# data_set.head()

# 数据分片，使用数据的前……条数据作为预测数据，后面的数据作为测试集,调参点,注意这个地方
TRAIN_SPLIT = 1750

# 设置种子以确保可重复性。
tf.random.set_seed(13)

# 单变量预测
# LSTM1_predict(data_set)

# 多变量预测
LSTM2_predict(data_set)
