"""
24.oct.2024
cdf data

Contents
    * cdf file data を読み込み, 任意の data を取り出す.

Details of functions
    * data: cdf_file_path の cdf data を読み込む
    * variable_list: cdf file の変数リストを作成
    * info: 変数名と shape を index 付きで表示
    * get: 任意の index の data を得る
"""
import spacepy.pycdf as pycdf
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.fft import fft, ifft, fftfreq
from cdflib import CDF
import time
import logging

from common.base import display, path


def get(cdf_file_path, display_cdf=False):
    """
    read cdf data
    :param cdf_file_path:
    :return:
    """
    cdf_data = pycdf.CDF(cdf_file_path)
    if display_cdf:
        print(f'{cdf_file_path=}')
        print(cdf_data)
        print('---------------------------')
    return cdf_data


def variable_list(cdf_file_path):
    """
    get variables
    :param cdf_file_path:
    :return:
    """
    cdf_data = get(cdf_file_path)
    var_names = list(cdf_data.keys())
    return var_names


def info(cdf_file_path):
    """
    information of the cdf file
    :param cdf_file_path:
    :return:
    """
    print("----- cdf info -----")
    print(f"cdf file path: {cdf_file_path}")
    cdf_data = get(cdf_file_path)
    # print(cdf_data)
    var_names = variable_list(cdf_file_path)
    # shape of each data
    shape_list = [np.array(cdf_data[i][:]).shape for i in var_names]

    dict_var_names = {}
    for i in range(len(var_names)):
        dict_var_names[var_names[i]] = shape_list[i]
    # dict_var_names = {
    #     "variables": var_names,
    #     "shapes": shape_list,
    # }
    display.print_dict(dict_var_names)

    return dict_var_names


def get_data(cdf_file_path, var):
    """

    :param cdf_file_path: str
    :param var: str or int (index)
    :return:
    """

    cdf = get(cdf_file_path)
    cdf_data_to_return = cdf[var][...]

    return cdf_data_to_return


def get_vardata(cdf, var):
    return cdf[var][...]


def merge_two(file1_path, file2_path, output_path=None, *args):
    cdf1 = get(file1_path)
    cdf2 = get(file2_path)
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


def get_cdf_to_read(paths: list, info: bool = True):# -> getdata > get_files_to_read
    """

    :param paths: list of str
    :return: str list of cdf files to read (sorted)
    """
    # CDFファイル一覧を取得
    paths = sorted(paths)

    # extract existing cdf files
    cdf_to_read = []
    for i, filepath in enumerate(paths):
        if os.path.exists(filepath) and filepath.endswith(".cdf"):
            cdf_to_read.append(filepath)

    if len(cdf_to_read) == 0 and info:
        # logging.error('No cdf file to read\n'
        #               f'given: {paths}')
        return None

    else:
        return cdf_to_read


def read_and_combine_cdf_files(
        paths: list,
        vars: list,
        display_progress_bar: bool=False,
        vars_not_combine: list | None = None,
        info: bool = True
):
    """
    Read and combine data from multiple CDF files.

    :param paths: List of str
    :param vars: List of str
    :return: dict
    """
    # Validate input paths
    cdf_files = get_cdf_to_read(paths, info=info)
    if not cdf_files:
        if info:
            display.warning(f'No cdf file to read\ngiven: {paths}')
        return None

    # initialize dict to return
    combined_data = {}
    for var in vars:
        combined_data[var] = []

    num_cdf_files = len(cdf_files)
    # print(f"number of cdf files to read: {len(cdf_files)}")

    # Read each file and combine data
    dt_start_loop = datetime.now()
    for i, file_path in enumerate(cdf_files):
        # progress bar
        if display_progress_bar:
            display.progress_bar(i, num_cdf_files, dt_start_loop)
        
        vars_available = variable_list(file_path)

        # print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Reading file: {file_path}")
        with pycdf.CDF(file_path) as cdf:
        # with CDF(file_path) as cdf: # cdflibだとepochがnumpy.datetime64になり, 取扱が大変.
            # for vars_not_combine,
            if i == 0 and vars_not_combine is not None:
                if not isinstance(vars_not_combine, list):
                    raise ValueError("vars_not_combine must be list.")
                for var in vars_not_combine:
                    if var in vars_available:
                        combined_data[var].append(cdf[var][:])
                        combined_data[var] = np.concatenate(combined_data[var])
                        vars.remove(var)
                    else:
                        display.error('cdfdata/read_and_combine', f'var {var} is not available: {vars_available=}')

            # 変数ごとに結合処理
            for var in vars:
                if var in vars_available:
                    # combined_data[var].append(cdf.varget(var))
                    combined_data[var].append(cdf[var][:])
                else:
                    display.error('cdfdata/read_and_combine', f'var {var} is not available: {vars_available=}')

    for var in vars:
        combined_data[var] = np.concatenate(combined_data[var])

    return combined_data


def dict_to_cdf(
        dict_data: dict,
        cdf
):
    for key, value in dict_data.items():
        cdf[key] = value
    return cdf


def dict_to_cdffile(
        dict_data: dict,
        savecdf
):
    path.make_directory(savecdf)
    if os.path.exists(savecdf):
        os.remove(savecdf)
    with pycdf.CDF(savecdf, '') as cdf:
        for key, value in dict_data.items():
            try:
                if not isinstance(value, (list, np.ndarray)):
                    value = [value]
                cdf[key] = value
            except Exception as e:
                display.error(f'Error in processing {key=}: {e}')
    display.info(f'Saved: {savecdf}')
    return


def cdffile_to_dict(
        cdf_filepath
):
    vars = variable_list(cdf_filepath)
    dict_data = {}
    for var in vars:
        try:
            dict_data[var] = get_data(cdf_filepath, var)
        except Exception as e:
            display.error(f'Error in processing {var}: {e}')
    return dict_data


def check_cdf_variables(cdf_filepath, var_names):
    """
    CDFファイル内に指定された変数がすべて存在するかチェックする。
    
    Parameters
    ----------
    cdf_filepath : str
        チェック対象のCDFファイルのパス。
    var_names : list of str
        存在を確認したい変数名のリスト。
        
    Returns
    -------
    dict
        'all_exist': bool (全て存在すればTrue)
        'missing': list (見つからなかった変数名のリスト)
        'existing': list (存在した変数名のリスト)
    """
    # そもそもファイルが存在するかチェック
    if not os.path.exists(cdf_filepath):
        print(f"Error: File not found: {cdf_filepath}")
        return None
    
    if not isinstance(var_names, list):
        var_names = list(var_names)

    dict_data = cdffile_to_dict(cdf_filepath)

    # CDF内の変数名リストを取得（dictのキー）
    available_vars = list(dict_data.keys())
    
    existing = []
    missing = []
    
    for var in var_names:
        if var in available_vars:
            existing.append(var)
        else:
            missing.append(var)
            
    all_exist = len(missing) == 0
    
    if not all_exist:
        display.warning(f"Warning: Missing variables in {cdf_filepath}: {missing}")

    return {
        'all_exist': all_exist,
        'missing': missing,
        'existing': existing
    }

