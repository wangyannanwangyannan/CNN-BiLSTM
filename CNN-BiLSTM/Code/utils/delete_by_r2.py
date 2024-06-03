import os
import pandas as pd


# 通过判断文件名中的R2值，删除文件

def get_filenames_path(dir_path):
    """
    获取文件夹下的所有文件的文件名的路径
    :param dir_path: 文件夹
    :return: 文件的路径列表
    """
    file_paths = []
    file_names = os.listdir(dir_path)
    print("此文件夹：{} 下共有{}个文件".format(dir_path, len(file_names)))
    for file_name in file_names:
        file_path = dir_path + "/" + file_name
        file_paths.append(file_path)
    return file_paths

def get_filenum(dir_path):
    file_names = os.listdir(dir_path)
    print("此文件夹：{} 下共有{}个文件".format(dir_path, len(file_names)))

def delete_by_r2(file_paths: list):
    """
    删除文件
    :param file_paths: 文件路径
    :return:
    """
    number = 0
    for file_path in file_paths:
        # 提取r2值
        r2 = eval(file_path.split("(")[1].split(")")[0])
        if r2 <= 0.0:
            number += 1
            os.remove(file_path)
        else:
            continue
    print("所有R2 <= 0 的文件全部删除！")
    print("共删除{}个文件".format(number))

if __name__ == '__main__':
    read_dir_path = "../../DataSet/Result/G20/csv_copy"
    file_paths = get_filenames_path(read_dir_path)
    delete_by_r2(file_paths)
    get_filenum(read_dir_path)

