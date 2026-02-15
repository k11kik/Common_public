"""
to be deleted
"""

import spacepy.pycdf as pycdf
import pandas as pd
import numpy as np
from common import display

display.error('This file is to be deleted')


def get_data(cdf_file_path):
    """
    read cdf data
    :param cdf_file_path:
    :return:
    """
    cdf_data = pycdf.CDF(cdf_file_path)
    return cdf_data


def variable_list(cdf_file_path):
    """
    get variables
    :param cdf_file_path:
    :return:
    """
    cdf_data = get_data(cdf_file_path)
    var_names = list(cdf_data.keys())
    return var_names


def info(cdf_file_path):
    """
    information of the cdf file
    :param cdf_file_path:
    :return:
    """
    print(f"cdf file path: {cdf_file_path}")
    cdf_data = get_data(cdf_file_path)
    print(cdf_data)
    var_names = variable_list(cdf_file_path)
    # shape of each data
    shape_list = [np.array(cdf_data[i][...]).shape for i in var_names]

    dict_var_names = {
        "variables": var_names,
        "shapes": shape_list,
    }
    df_var_names = pd.DataFrame(dict_var_names)
    print(df_var_names)


def get(cdf_file_path, index):
    cdf_data = get_data(cdf_file_path)
    var_names = variable_list(cdf_file_path)
    # どの変数を取り出すか確認
    print(f"variable to get: {var_names[index]}")
    cdf_data_to_return = np.array(cdf_data[index][...])

    return cdf_data_to_return


def merge_two(file1_path, file2_path, output_path=None, *args):
    cdf1 = get_data(file1_path)
    cdf2 = get_data(file2_path)
    merged_cdf = pycdf.CDF(output_path, create=True)
    if "all" in args:
        for var_name in cdf1.keys():
            merged_cdf[var_name] = np.array(cdf1[var_name][...])
        for var_name in cdf2.keys():
            if var_name not in merged_cdf:
                merged_cdf[var_name] = np.array(cdf2[var_name][...])
            else:
                merged_cdf[var_name] = np.append(merged_cdf[var_name], cdf2[var_name][...])
    else:
        for var_name in args:
            merged_cdf[var_name] = np.array(cdf1[var_name][...])
        for var_name in args:
            if var_name not in merged_cdf:
                merged_cdf[var_name] = np.array(cdf2[var_name][...])
            else:
                merged_cdf[var_name] = np.concatenate((merged_cdf[var_name], cdf2[var_name][...]))
    cdf1.close()
    cdf2.close()
    merged_cdf.close()


def merge():
    return
