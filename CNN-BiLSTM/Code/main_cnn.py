import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 对数据进行归一化处理
# from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Conv2D, MaxPooling2D, Reshape
from keras.models import Sequential
from keras import optimizers
# from keras.models import load_model
from sklearn.metrics import r2_score, mean_squared_error
from keras import metrics
from keras import backend, regularizers
import time
import os

"""


"""

# 两个归一化对象target_data  factors_data，定义在函数外面，便于反归一化
target_scaler = MinMaxScaler()  # 最大最小归一化
factors_scaler = MinMaxScaler()


# target_scaler = StandardScaler()   # 标准归一化
# factors_scaler = StandardScaler()


def Read_MinMaxScaler(file_read_path):
    """
    读取数据集并进行归一化处理
    :param file_read_path: 文件读取路径
    :return: 归一化后的数据以及years
    """
    # 读取数据------------------------------------------------------------
    read_data = pd.read_csv(file_read_path)
    print(read_data.head())
    # # 先获取数据集起始年份用于确定数据集范围
    # first_year = read_data['DATE'][0]
    # print("first_year：", first_year)
    # 绘图横坐标
    years = list(read_data['DATE'])
    columns_list = list(read_data.columns.values)
    # 去除无用数据country date
    if "Country" in columns_list:
        read_data = read_data.drop(['Country', 'DATE'], axis=1)
        print("有country列")
    else:
        read_data = read_data.drop(['DATE'], axis=1)
        print("无country列")
    print("DataSet Length = ", len(read_data))
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
file_save_path = "../DataSet/Result/world_cnn _9322/"

# read
file_path = "../DataSet/Data/world_9322"  # 如需切换其他文件夹，只需更改此处
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
TIME_STEP = 3  # 时间戳
LSTM_UNITS = 32  # LSTM隐藏层数量
lr = 1e-3  # Adam优化器学习率
Epochs = 2000  # 训练次数
batch_size = 32  # 批处理尺寸
dropout1 = 0.00
dropout2 = 0.00
train_size = 0.8
decay = 1e-4
l2_C = 1e-6  # 正则化系数，防止过拟合
n1_filter = 32  # 卷积层过滤器个数
n2_filter = 16  # 卷积层过滤器个数
kernel_size = (4, 4)  # 卷积核尺寸
pool_size = (2, 4)  # 池化核尺寸

# # 将这些参数保存到输出的CSV文件中， 在下面
# parameters_key = pd.Series(
#     ["INPUT_SIZE", "TIME_STEP", "LSTM_UNITS", "lr", "Epochs", "batch_size", "dropout1", "dropout2", "train_size",
#      "decay", "l2_C", "n_filter"])
# parameters_value = pd.Series([INPUT_SIZE, TIME_STEP,
#                               LSTM_UNITS, lr, Epochs, batch_size,
#                               dropout1, dropout2, train_size, decay, l2_C, n_filter])

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
x_train = np.expand_dims(x_train, -1)  # Conv2D需要四维
x_test = np.expand_dims(x_test, -1)  # Conv2D需要四维
print("x_train.shape = ", x_train.shape)
print("x_test.shape = ", x_test.shape)

# ------------------------------------6.搭建模型--------------------------------------
my_model = Sequential()

# -----------------------------------CNN-----------------------------------
# 卷积层1
my_model.add(
    Conv2D(
        filters=n1_filter, kernel_size=kernel_size,
        activation="relu", padding="same", input_shape=(TIME_STEP, INPUT_SIZE, 1)
    )
)

# 最大池化层1
my_model.add(
    MaxPooling2D(pool_size=pool_size, padding="same")
)

# 卷积层2
my_model.add(
    Conv2D(
        filters=n2_filter, kernel_size=kernel_size,
        activation="relu", padding="same"
    )
)

# 最大池化层2
my_model.add(
    MaxPooling2D(pool_size=pool_size, padding="same")
)

# Reshape层
my_model.add(
    # 两层pool
    Reshape(
        (
            math.ceil(math.ceil(TIME_STEP / pool_size[0]) / pool_size[0]),
            math.ceil(math.ceil(INPUT_SIZE / pool_size[1]) / pool_size[1]) * n2_filter
        )
        # 一层pool
        # Reshape(
        #     (
        #         math.ceil(TIME_STEP / pool_size[0]),
        #        math.ceil(INPUT_SIZE / pool_size[1]) * n2_filter
        #     )
    )
)

# -----------------------------------BiLSTM----------------------------------
# 第一层：BiLSTM,l2正则，relu激活函数
my_model.add(
    Bidirectional(
        LSTM(units=LSTM_UNITS,
             return_sequences=True,
             kernel_regularizer=regularizers.l2(l2_C),  # 在权重参数w添加L2正则化
             activation='relu'),
        # input_shape=(TIME_STEP, INPUT_SIZE)
    )
)

# 第二层dropout
my_model.add(Dropout(dropout1))

# 第三层BiLSTM，l2正则，relu激活函数
my_model.add(Bidirectional(LSTM(units=LSTM_UNITS,
                                kernel_regularizer=regularizers.l2(l2_C),
                                activation='relu')))
# 第四层dropout
my_model.add(Dropout(dropout2))

# 第五层，全连接层，此层输出维度units
my_model.add(Dense(units=1, activation='sigmoid'))


# --------------------------------------------------------------------------


# 定义RMSE函数
def rmse(y_true, y_pred):
    # RMSE的取值范围是0到正无穷大,数值越小表示模型的预测误差越小,模型的预测能力越强
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# 8.summary(),使用 model.summary() 来查看你的神经网络的架构和参数量等信息。
my_model.summary()

# 7.模型 compile，指定 loss function，optimizer，metrics
my_model.compile(
    # lr = lr*(1/(1+decay*epoch))
    optimizer=optimizers.Adam(lr=lr, decay=decay),
    loss='MSE',
    metrics=[rmse]
)
# 9.train
history = my_model.fit(
    x_train, y_train,
    epochs=Epochs,
    batch_size=batch_size,
    verbose=2,
    validation_split=0.1
)

# 定义模型/文件保存路径
# 文件夹
model_save_path = file_save_path + "model/"
csv_save_path = file_save_path + "csv/"
# 获取时间戳 年-月-日(时:分:秒)
loca = time.strftime('%Y-%m-%d-%H-%M-%S')
save_name = file_name.split(".")[0] + "-" + loca

# 加载网络模型
# my_model = load_model("../DataSet/Result/model_scaler2_2022/model/" + file_name + ".h5",
#                       custom_objects={'rmse': rmse})

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

# 将这些参数保存到输出的CSV文件中
parameters_key = pd.Series(
    ["INPUT_SIZE",
     "TIME_STEP",
     "LSTM_UNITS",
     "lr", "Epochs",
     "batch_size",
     "dropout1",
     "dropout2",
     "train_size",
     "decay",
     "l2_C",
     "n1_filter",
     "n2_filter",
     "NRMSE",
     "kernel_size",
     "pool_size"]
)

parameters_value = pd.Series(
    [INPUT_SIZE,
     TIME_STEP,
     LSTM_UNITS,
     lr, Epochs,
     batch_size,
     dropout1,
     dropout2,
     train_size,
     decay,
     l2_C,
     n1_filter,
     n2_filter,
     NRMSE,
     kernel_size,
     pool_size]
)

# 设置保存文件
r2 = round(r2, 3)  # 保留三位小数
model_save_name = save_name + "(" + str(r2) + ")" + ".h5"
csv_save_name = save_name + "(" + str(r2) + ")" + ".csv"
# 设置保存root
model_save_root = model_save_path + model_save_name
csv_save_root = csv_save_path + csv_save_name
# 保存模型
my_model.save(model_save_root)
print("模型：{} 已保存成功！".format(model_save_name))

# 真实-预测值:所有
factors = np.expand_dims(factors, -1)
print("此时factors形状为：", factors.shape)

target_pre = target_scaler.inverse_transform(my_model.predict(factors))
target_true = target_scaler.inverse_transform(target)
# 设置保留三位小数
target_pre = np.around(target_pre, 3)
years = years[TIME_STEP - 1:]

# 保存预测值和真实值
new_df = pd.DataFrame(target_true, columns=['true'])
new_df['PRE'] = target_pre
# 保存自定义参数
new_df["parameters"] = parameters_key
new_df["parameters_value"] = parameters_value

new_df.to_csv(csv_save_root)
print("文件：{} 已保存成功！".format(csv_save_name))

# 11.绘制loss和rmse曲线图
plt.figure(figsize=(6, 4), dpi=100)
plt.grid(True)  # 添加网格
plt.plot(history.history['loss'], color='r')
plt.plot(history.history['rmse'], color='b')
plt.title("model loss and rmse")
plt.xlabel("epochs")
plt.ylabel("loss & rmse")
plt.legend(['loss', 'rsme'], loc='upper left')
plt.show()
# 绘制真实-预测曲线-测试集
plt.figure(figsize=(6, 4), dpi=100)
plt.grid(True)  # 添加网格
plt.plot(y_test, color='r')
plt.plot(y_test_pre, color='b')
plt.title("Test: pre and true")
plt.ylabel("MF")
plt.xlabel("Date")
plt.legend(['y_true', 'y_pre'], loc='upper left')
plt.show()
