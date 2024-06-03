import os
import time

import numpy as np
import pandas as pd


def get_filenames(dir_path):
    """
    读取dir_path文件夹下的所有文件名
    :param dir_path: 文件夹路径
    :return: 文件名列表
    """
    filename_list = os.listdir(dir_path)
    print("{}下共有{}个文件".format(dir_path, len(filename_list)))
    return filename_list


def get_nrmse(dir_path, filename_list, file_save_path):
    df = pd.DataFrame()
    df_temp = pd.read_csv(dir_path + "/" + filename_list[0])
    for i in range(len(filename_list) - 1):
        file_name = filename_list[i + 1]
        root = dir_path + "/" + file_name
        df = pd.read_csv(root)
        df = pd.concat([df_temp, df], axis=0,join="outer")
        df_temp = df
    print(df)
    df.to_csv(file_save_path + "/" + "r2_nrmse.csv")
    print("文件保存成功！")


read_dir_path = "../../DataSet/Result/load_model/bilstm"
filename_list = get_filenames(read_dir_path)
print(filename_list)
file_save_path = "../../DataSet/Result/load_model/bilstm"
get_nrmse(read_dir_path, filename_list, file_save_path)
