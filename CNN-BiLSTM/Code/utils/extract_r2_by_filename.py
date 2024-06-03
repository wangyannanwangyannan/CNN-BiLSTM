"""
用于从文件名中提取R2指标，并将其存入统一的excel文件中
"""
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


def extract_r2(filename_list, threshold):
    """
    从文件名中提取R2指标
    :param filename_list: 文件名列表
    :param threshold: r2阈值，即提取大于该值的r2
    :return: r2_list
    """
    r2_list = []
    for filename in filename_list:
        r2 = eval(filename.split("(")[1].split(")")[0])
        if r2 > threshold:
            r2_list.append(r2)
    print("符合条件r2 > {} 的数据共有：{}条".format(threshold, len(r2_list)))
    r2_list.sort(reverse=True)  # 从大到小排序
    return r2_list


def save_to_csv(lis: list, save_dir):
    """
    将数组数据保存为csv文件
    :param lis: 要保存的字典数据
    :param save_dir: 保存路径文件夹
    :return: 无
    """
    df = pd.DataFrame()
    df["R2"] = lis
    save_name = str(threshold) + "-r2.csv"
    save_root = save_dir + "/" + save_name
    df.to_csv(save_root)
    print("文件{}保存成功！".format(save_name))


if __name__ == '__main__':
    loca = time.strftime('%Y-%m-%d-%H-%M-%S')
    print("~~~~~~~~~~~~~~~~~~~~~~~提示信息:{}~~~~~~~~~~~~~~~~~~~~~~~".format(loca))
    # 1.获取指定文件夹下的文件名列表
    # file_root = "../../DataSet/Result/world/csv"
    file_root = "../../DataSet/Result/world_cnn/csv"
    filename_list = get_filenames(file_root)
    # 2.通过1中的结果来获取lis数据
    threshold = 0.6
    lis = extract_r2(filename_list, threshold)
    # 保存数据
    # file_save_root = "../../DataSet/Result/world/R2"
    file_save_root = "../../DataSet/Result/world_cnn/R2"
    save_to_csv(lis, file_save_root)
