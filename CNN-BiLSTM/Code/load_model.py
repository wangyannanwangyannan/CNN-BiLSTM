import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 对数据进行归一化处理
# from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Conv2D, MaxPooling2D, Reshape
from keras.models import Sequential, load_model
from keras import optimizers
# from keras.models import load_model
from sklearn.metrics import r2_score, mean_squared_error
from keras import metrics
from keras import backend, regularizers
import time
import os

"""
数据集说明：
        1.范围1978-2019年共42条数据
        2.Target：MF
        3.Factors：People Urbanization GDP OIL GAS COAL 一次能源消费量
        4.无用数据：Country 和 Date

"""

# 两个归一化对象target_data  factors_data，定义在函数外面，便于反归一化
target_scaler = MinMaxScaler()  # 最大最小归一化
factors_scaler = MinMaxScaler()


def Read_MinMaxScaler(file_read_path):
    """
    读取数据集并进行归一化处理
    :param file_read_path: 文件读取路径
    :return: 归一化后的数据以及years
    """
    # 读取数据------------------------------------------------------------
    read_data = pd.read_csv(file_read_path)
    print(read_data.head())
    years = list(read_data['DATE'])
    columns_list = list(read_data.columns.values)
    # 去除无用数据country date
    if "Country" in columns_list:
        read_data = read_data.drop(['Country', 'DATE'], axis=1)
        print("有country列")
    else:
        read_data = read_data.drop(['DATE'], axis=1)
        print("无country列")
    # 分别提取target 和 factors,用年份来确定范围，19年及以前的数据用于训练及测试，之后的用于预测
    # train_and_test_length = 2019 - first_year + 1
    # print("可用数据集长度 = ", train_and_test_length)
    # 提取目标值和影响因子
    target_data = read_data.iloc[:, 0]
    # 处理成np.array 类型便于后面归一化计算
    target_data = np.array(target_data).reshape(-1, 1)
    # 影响因子获取所有行，但只有前train_and_test_length行用于训练及测试
    factors_data = read_data.iloc[:, 1:]
    factor_num = factors_data.shape[1]  # 影响因子个数
    # 处理成np.array 类型便于后面归一化计算
    factors_data = np.array(factors_data).reshape(-1, factor_num)
    # 归一化------------------------------------------------------------
    target_data_scaled = target_scaler.fit_transform(target_data)
    factors_data_scaled = factors_scaler.fit_transform(factors_data)
    print("归一化后目标值数据形状：", target_data_scaled.shape)
    print("归一化后影响因子数据形状：", factors_data_scaled.shape)
    # 返回
    return target_data_scaled, factors_data_scaled, years


"""
先处理好，再划分数据集
现在用过去n年的因子数据来预测后一年的MF数据
如n=2则表示用1990,1991两年的因子的数据来预测1992年的MF数据
2023-12-26-10:20修改，90 91 预测 91
"""


def createTarget(dataset, time_step):
    """
    获取target
    , time_window
    :param dataset:训练集或者测试集
    :param time_step:表示用过去几年的数据进行预测
    # :param time_window:表示想要预测未来几年的TARGET
    :return:y_data
    """
    y_data = []
    # len(dataset) - time_step
    for i in range(len(dataset) - time_step + 1):
        # time_step + i:time_step + i + 1, 0
        y = dataset[time_step + i - 1:time_step + i, 0]
        y_data.append(y)
    return np.array(y_data)


def createFactor(dataset, time_step):
    """
    获取factor
    :param dataset:训练集或者测试集
    :param time_step:表示用过去几年的数据进行预测
    :param time_window:表示想要预测未来几年的TARGET
    :return:x_data
    """
    x_data = []
    # len(dataset) - time_step + 1   --- 90 91 -> 91
    for i in range(len(dataset) - time_step + 1):
        x = dataset[i:time_step + i, :]
        x_data.append(x)
    return np.array(x_data)


# 划分数据集--训练集和测试集
def split_dataset(dataset, train_val):
    """
    数据集划分为训练集和测试集
    :param dataset: 总数据集（归一化后）
    :param train_val:训练集比例，一般0.8 或 0.7
    :return: 划分好的数据集
    """
    train_size = int(dataset.shape[0] * train_val)
    dataset_train = dataset[:train_size]  # 取前面所有行所有列
    dataset_test = dataset[train_size:]  # 取剩余的所有数据
    return dataset_train, dataset_test


def get_filenames(dir_path):
    """
    获取文件名称
    :param dir_path: 文件夹路径
    :return: 文件名列表
    """
    file_names = os.listdir(dir_path)
    return file_names


# 文件读取路径
# 文件夹
file_save_path = "../DataSet/Result/load_model/"

# read
file_path = "../DataSet/Data/world_copy"  # 如需切换其他文件夹，只需更改此处
# 获取文件名
file_names = get_filenames(file_path)
file_number = 0
file_name = file_names[file_number]
# 组合文件路径
file_read_path = file_path + "/" + file_name

# 获取归一化后的数据集
target_data_scaled, factors_data_scaled, years = Read_MinMaxScaler(file_read_path)

# 定义一些参数----2023-12-26 10:30修改90 91 -> 91,时间戳在此以后得均模型结构与之前的不同
INPUT_SIZE = int(factors_data_scaled.shape[1])  # 特征数
print("input_size = ", INPUT_SIZE)
TIME_STEP = 7  # 时间戳
train_size = 0.8

# =====================获取经过time_step处理后的数据集=====================
# 直接用于模型的数据
target = createTarget(target_data_scaled, TIME_STEP)
print("按照time_step处理后的target数据集形状：", target.shape)  # (3265, 1)
factors = createFactor(factors_data_scaled, TIME_STEP)
print("按照time_step处理后的factors数据集形状：", factors.shape)  # (3265, 3, 7)
# 获取训练集和测试集
x_train, x_test = split_dataset(factors, train_size)
y_train, y_test = split_dataset(target, train_size)
print("y_train.shape = ", y_train.shape)
print("y_test.shape = ", y_test.shape)
# x_train = np.expand_dims(x_train, -1)  # Conv2D需要四维
# x_test = np.expand_dims(x_test, -1)  # Conv2D需要四维
print("x_train.shape = ", x_train.shape)
print("x_test.shape = ", x_test.shape)


# 定义RMSE函数
def rmse(y_true, y_pred):
    # RMSE的取值范围是0到正无穷大,数值越小表示模型的预测误差越小,模型的预测能力越强
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# 加载网络模型
model_path = "../DataSet/Result/world/model/77 country-2024-01-09-20-49-43(0.639).h5"
my_model = load_model(model_path, custom_objects={'rmse': rmse})

# 8.summary(),使用 model.summary() 来查看你的神经网络的架构和参数量等信息。
my_model.summary()

# 10.输出R2指标、测试等
# 训练集---------------------------------------------------------
# 此时的y_pre值是归一化的值
y_pred_train = my_model.predict(x_train)
# 对训练集target的预测值反归一化
y_pred_train = target_scaler.inverse_transform(y_pred_train)
# 对y_train进行反归一化
y_train = target_scaler.inverse_transform(y_train)
print("训练集R2 = ", r2_score(y_train, y_pred_train))

# 测试集---------------------------------------------------
y_test_pre_scaled = my_model.predict(x_test)
# 计算反归一化前的RMSE
RMSE1 = np.sqrt(mean_squared_error(y_test, y_test_pre_scaled))
print("反归一化前RMSE = ", RMSE1)
# 反归一化前R2
r2_no_inverse = r2_score(y_test, y_test_pre_scaled)
print("反归一化前-测试集R2 = ", r2_no_inverse)

# 对 y_test_pre_scaled进行反归一化
y_test_pre = target_scaler.inverse_transform(y_test_pre_scaled)
# 对 y_test进行反归一化
y_test = target_scaler.inverse_transform(y_test)
RMSE = np.sqrt(np.mean(np.square(y_test - y_test_pre)))
print("数据反归一化后RMSE = ", RMSE)
# 计算R2
r2 = r2_score(y_test, y_test_pre)
print("反归一化后-测试集R2 = ", r2)

# 计算NRMSE--归一化的RMSE = RMSE / (y_max - y_min)
y_max = max(y_test)
y_min = min(y_test)
NRMSE = RMSE / (y_max - y_min)
print("NRMSE = ", NRMSE)

r2 = round(r2, 3)
NRMSE = NRMSE[0]
NRMSE = round(NRMSE, 3)

df = pd.DataFrame()
df["r2"] = [r2]
df["nrmse"] = [NRMSE]

loc = time.strftime("%Y-%m-%d-%H-%M-%S")
save_root = file_save_path + "bilstm/" + loc + "(" + str(r2) + ")" + ".csv"
print(df)
df.to_csv(save_root, index=False)
print("文件{}保存成功".format(save_root))
# 真实-预测值:所有
# factors = np.expand_dims(factors, -1)
# print("此时factors形状为：", factors.shape)

target_pre = target_scaler.inverse_transform(my_model.predict(factors))
target_true = target_scaler.inverse_transform(target)
# 设置保留三位小数
target_pre = np.around(target_pre, 3)
years = years[TIME_STEP - 1:]

# 绘制真实-预测曲线-测试集
plt.figure(figsize=(6, 4), dpi=100)
plt.grid(True)  # 添加网格
plt.plot(y_test, color='r')
plt.plot(y_test_pre, color='b')
plt.title("Test: pre and true")
plt.ylabel("MF")
plt.xlabel("Date")
plt.legend(['y_true', 'y_pre'], loc='upper left')
# plt.show()
