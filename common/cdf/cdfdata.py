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
from common import util


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
    util.print_dict(dict_var_names)

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


def merge():
    return


def correction(data, threshold=1e+30, quiet=True):
    count_over_threshold = np.sum(np.abs(data) >= threshold)
    if count_over_threshold == 0:
        return data
    else:
        data = np.where(np.abs(data) >= threshold, np.nan, data)
        count_over_threshold_after = np.sum(np.abs(data) >= threshold)
        if not quiet:
            print(f"# invalid data detected (threshold: {threshold})")
            print("before", count_over_threshold)
            print("after", count_over_threshold_after)

        return data


def replace_into_nan(data, invalid=-1e+31, quiet=True):
    count = np.sum(data == invalid)
    if count == 0:
        return data
    else:
        data = np.where(data == invalid, np.nan, data)
        count_after = np.sum(np.abs(data) == invalid)
        if not quiet:
            print("# invalid values detected")
            print("before", count)
            print("after", count)

        return data


def digital_filter(data, dt, cutoff_freq_low=None, cutoff_freq_high=None):
    """
    特定の周波数成分を除去するフィルタ関数。

    :param data: フィルタをかけるデータ（1次元配列）
    :param dt: サンプリング間隔 [s]
    :param cutoff_freq_low: 除去する周波数帯域の下限（Hz）
    :param cutoff_freq_high: 除去する周波数帯域の上限（Hz）
    :return: フィルタ後のデータ（1次元配列）
    """
    # フーリエ変換
    freq_domain = fft(data)
    freqs = fftfreq(len(data), d=dt)

    if cutoff_freq_low > cutoff_freq_high:
        raise ValueError("cutoff_freq_low must be smaller than cutoff_freq_high.")

    # フィルタリング（指定された周波数成分を0にする）
    if cutoff_freq_low is None and cutoff_freq_high is None:
        raise ValueError("cutoff_freq_low or cutoff_freq_high should be given.")

    if cutoff_freq_low is not None and cutoff_freq_high is not None:
        freq_domain[(np.abs(freqs) >= cutoff_freq_low) & (np.abs(freqs) <= cutoff_freq_high)] = 0

    if cutoff_freq_low is not None and cutoff_freq_high is None:
        freq_domain[(np.abs(freqs) >= cutoff_freq_low)] = 0

    if cutoff_freq_low is None and cutoff_freq_high is not None:
        freq_domain[(np.abs(freqs) <= cutoff_freq_high)] = 0

    # 逆フーリエ変換
    filtered_data = ifft(freq_domain).real

    return filtered_data


def moving_average_principle(
        time: np.ndarray,
        data: np.ndarray,
        window_size: int,
        mode: str = "valid"
):
    """
    時間方向の移動平均を取る関数。

    :param data: 移動平均を取るデータ（1次元配列）
    :param window_size: 移動平均のウィンドウサイズ
    :return: time (n,), data (n,)
    """
    if window_size > len(data):
        print("[!] window size is larger than data length. -> window size = 0.1 * len(data)")
        window_size = int(.1 * len(data))

    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    len_data = len(data)
    # data_averaged = [np.mean(data[i:i+window_size]) for i in range(len_data - window_size)]
    # time = time[:len_data - window_size]

    if mode == "valid":
        data_averaged = []
        for i in range(len(data) - window_size):
            arr_to_ave = data[i:i+window_size]
            data_averaged.append(np.mean(arr_to_ave))
        data_averaged = np.array(data_averaged)
        time_averaged = time[:-window_size]
        return time_averaged, data_averaged

    elif mode == "same":
        data_averaged = np.zeros_like(data)
        # 移動平均のインデックス範囲を設定して制限
        for i in range(len_data):
            idx_range = np.clip(np.arange(i - window_size // 2, i + window_size // 2 + 1), 0, len_data - 1)
            data_averaged[i] = np.mean(data[idx_range])
        return time, data_averaged

    # if valid:
    #     idx_valid = slice(window_size // 2, -window_size // 2 + 1)
    #     time = time[idx_valid]
    #     data_averaged = data_averaged[idx_valid]



def moving_average(
        time, data,
        window_size: int,
        mode: str = "same"
):
    """
    Apply a moving average to data along a specified axis.

    :param time: Time array (1D array)
    :param data: Input data array (1D or 2D array)
    :param window_size: Window size for the moving average
    :param mode: Mode for handling boundaries ('same', 'valid', 'full'). Default: 'same'
    :return: Tuple (time, averaged_data)
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    if window_size > len(data):
        print("[!] window size is larger than data length. -> window size = 0.1 * len(data)")
        window_size = int(.1 * len(data))

    if mode not in ["same", "valid", "full"]:
        raise ValueError(f"Invalid mode: {mode}. Use 'same', 'valid', or 'full'.")

    # Create a uniform kernel for averaging
    kernel = np.ones(window_size) / window_size

    if mode == "same":
        data_averaged = np.convolve(data, kernel, mode="same")
        return time, data_averaged

    elif mode == "valid":
        data_averaged = np.convolve(data, kernel, mode="valid")
        valid_idx = slice(window_size // 2, -window_size // 2 + 1 or None)
        return time[valid_idx], data_averaged

    elif mode == "full":
        data_averaged = np.convolve(data, kernel, mode="full")
        return time, data_averaged  # Full mode may not align with original time array

    raise RuntimeError("Unexpected error during moving average computation.")

    # # Ensure data is a NumPy array
    # data = np.asarray(data)
    # if data.ndim > 2:
    #     raise ValueError("Only 1D or 2D arrays are supported.")
    #
    #
    # # Handle 1D data
    # if data.ndim == 1:
    #     averaged_data = np.convolve(data, kernel, mode=mode)
    #     if mode == "same":
    #         return time, averaged_data
    #     elif mode == "valid":
    #         valid_idx = slice(window_size // 2, -window_size // 2 + 1 or None)
    #         return time[valid_idx], averaged_data
    #     elif mode == "full":
    #         return time, averaged_data  # Full mode may not align with original time array
    #
    # # Handle 2D data
    # elif data.ndim == 2:
    #     if axis == 0:
    #         averaged_data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode=mode), axis=0, arr=data)
    #     elif axis == 1:
    #         averaged_data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode=mode), axis=1, arr=data)
    #     else:
    #         raise ValueError(f"Invalid axis: {axis}. Must be 0 or 1.")
    #
    #     if mode == "same":
    #         return time, averaged_data
    #     elif mode == "valid":
    #         if axis == 0:
    #             valid_idx = slice(window_size // 2, -window_size // 2 + 1 or None)
    #             return time[valid_idx], averaged_data[valid_idx, :]
    #         elif axis == 1:
    #             return time, averaged_data[:, window_size // 2: -window_size // 2 + 1 or None]
    #     elif mode == "full":
    #         return time, averaged_data
    #
    # raise RuntimeError("Unexpected error during moving average computation.")

def multi(folder_path, var_epoch, var, display=False, idx_slice=None, dataframe=False, filter_string=None):
    """
    make DataFrame with epoch data & some data of all cdf file in the folder
    :param folder_path: str
    :param var_epoch: str, variable of epoch
    :param var: str, variable that you will get
    :param display: if True, print DataFrame. Default: False
    :param idx_slice: cdf files to get. e.g.) [0, 5]
    :param dataframe: if True, return DataFrame. otherwise, return dictionary data
    :return: dict data or DataFrame
    """
    # CDFファイル一覧を取得
    cdf_files = [f for f in os.listdir(folder_path) if f.endswith('.cdf')]
    cdf_files = sorted(cdf_files)

    if filter_string is not None:
        if isinstance(filter_string, (list, np.ndarray)):
            if not isinstance(filter_string[0], str):
                raise ValueError("filter string must be str.")
            cdf_files = [f for f in cdf_files if any(substr in f for substr in filter_string)]
        else:
            if not isinstance(filter_string, str):
                raise ValueError("filter string must be str.")
            cdf_files = [f for f in cdf_files if filter_string in f]

    if idx_slice is not None:
        cdf_files = cdf_files[idx_slice[0]:idx_slice[1]]

    print(f"number of cdf files to read: {len(cdf_files)}")

    # 日付データを保存するリスト
    timestamps = np.array([])

    # 各CDFファイルのデータを読み込んで結合
    for i, cdf_file in enumerate(cdf_files):  # CDFファイルをソートして時間順に読み込む
        file_path = os.path.join(folder_path, cdf_file)
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {file_path}")

        if i == 0:
            with pycdf.CDF(file_path) as cdf_data:
                timestamps = np.array(cdf_data[var_epoch])
                all_data = np.array(cdf_data[var][:])

        else:
            # CDFファイルを開く
            with pycdf.CDF(file_path) as cdf_data:
                # 時間データの変数を取得（仮に 'Epoch' が時間の変数だとする）
                time_data = cdf_data[var_epoch][:]
                # 観測データの変数を取得（仮に 'data_var' が観測データだとする）
                data = cdf_data[var][:]

                # データをリストに追加
                timestamps = np.append(timestamps, time_data)

                if data.ndim == 1:
                    all_data = np.append(all_data, data)
                if data.ndim == 2:
                    all_data = np.vstack((all_data, data))

    if dataframe:
        dict_data = {"time": timestamps}
        if all_data.ndim == 1:
            dict_data["data"] = all_data
        elif all_data.ndim == 2:
            dict_data = {"time": timestamps}
            dim1 = np.shape(all_data)[1]  # 行列の列の次元 (n,m)のm
            for i in range(dim1):
                dict_data[f"data_{i+1}"] = all_data[:, i]
        else:
            raise ValueError("all_data.ndim > 2.")

        df = pd.DataFrame(dict_data)

        if display:
            print(df)

        return df

    else:
        dict_data = {"time": timestamps, "data": all_data}
        return dict_data


def multi_files(paths: list, var_epoch, var, display=False, dataframe=False):
    # CDFファイル一覧を取得
    paths = sorted(paths)

    # 存在しないfile pathを削除
    path_not_exist = []
    for i, filepath in enumerate(paths):
        if not os.path.exists(filepath):
            path_not_exist.append(filepath)
            print(f"the file does not exist: {filepath}")

    [paths.remove(i) for i in path_not_exist]
    cdf_files = paths

    if len(cdf_files) == 0:
        print("No cdf file to read.")
        return None

    else:
        print(f"number of cdf files to read: {len(cdf_files)}")

        # 日付データを保存するリスト
        timestamps = np.array([])

        # 各CDFファイルのデータを読み込んで結合
        for i, cdf_file in enumerate(cdf_files):  # CDFファイルをソートして時間順に読み込む
            file_path = cdf_file
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {file_path}")

            if i == 0:
                with pycdf.CDF(file_path) as cdf_data:
                    timestamps = np.array(cdf_data[var_epoch])
                    all_data = np.array(cdf_data[var][:])

            else:
                # CDFファイルを開く
                with pycdf.CDF(file_path) as cdf_data:
                    # 時間データの変数を取得（仮に 'Epoch' が時間の変数だとする）
                    time_data = cdf_data[var_epoch][:]
                    # 観測データの変数を取得（仮に 'data_var' が観測データだとする）
                    data = cdf_data[var][:]

                    # データをリストに追加
                    timestamps = np.append(timestamps, time_data)

                    if data.ndim == 1:
                        all_data = np.append(all_data, data)
                    if data.ndim == 2:
                        all_data = np.vstack((all_data, data))

        if dataframe:
            dict_data = {"time": timestamps}
            if all_data.ndim == 1:
                dict_data["data"] = all_data
            elif all_data.ndim == 2:
                dict_data = {"time": timestamps}
                dim1 = np.shape(all_data)[1]  # 行列の列の次元 (n,m)のm
                for i in range(dim1):
                    dict_data[f"data_{i + 1}"] = all_data[:, i]
            else:
                raise ValueError("all_data.ndim > 2.")

            df = pd.DataFrame(dict_data)

            if display:
                print(df)

            return df

        else:
            dict_data = {"time": timestamps, "data": all_data}
            return dict_data


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


def save_data_as_cdf(
        dict_data: dict,
        cdf_filename: str,
        units=None,
        description=None,
        cdf_epoch: list=None
):
    path.remove_file_if_exists(cdf_filename)
    path.make_directory(cdf_filename)

    if cdf_epoch:
        for var in cdf_epoch:
            # datetime64 → UNIXタイム（秒単位）に変換
            dict_data[var] = (dict_data[var] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

    with pycdf.CDF(cdf_filename, "") as cdf:
        for key, value in dict_data.items():
            cdf[key] = value
            # 属性 (オプションが指定されている場合のみ設定)
            if units and key in units:
                cdf[key].attrs["units"] = units[key]
            if description and key in description:
                cdf[key].attrs["description"] = description[key]

    display.current_time_comment(comment=f"CDF file successfully created: {cdf_filename}")
    return


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

