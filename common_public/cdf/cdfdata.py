"""
cdf/cdfdata.py

Contents
    * cdf file data を読み込み, 任意の data を取り出す.
"""
import spacepy.pycdf as pycdf
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.fft import fft, ifft, fftfreq

from common import display, path


class CDFVar(np.ndarray):
    """
    既存の `dict_data['varname']` が直接 NumPy ndarray として振る舞いつつ、
    `.unit` や `.desc` も保持するための、np.ndarrayを継承した特殊クラス。
    """
    def __new__(cls, input_array, unit=None, desc=None):
        # 入力データをndarrayとしてビューキャスト
        obj = np.asarray(input_array).view(cls)
        # 新しい属性を割り当て
        obj.unit = unit
        obj.desc = desc
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.unit = getattr(obj, 'unit', None)
        self.desc = getattr(obj, 'desc', None)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == 'data':
                return np.asarray(self)
            elif item == 'unit':
                return self.unit
            elif item == 'desc':
                return self.desc
        # それ以外（NumPyスライシングや boolean インデックスなど）は通常の ndarray の処理に安全に委譲
        return super().__getitem__(item)

    def get(self, item, default=None):
        """辞書エミュレーション用の get メソッド"""
        if item == 'data': return np.asarray(self)
        if item == 'unit': return self.unit
        if item == 'desc': return self.desc
        return default
    
    def __repr__(self):
        """コンソールで評価された際、CDFVar(...) 表記を隠して array(...) として出力します。"""
        return repr(np.asarray(self))

    def __str__(self):
        """print() で出力された際、CDFVar(...) 表記を完全に隠して素の [...] として出力します。"""
        return str(np.asarray(self))



class CDFDict(dict):
    """
    旧仕様の辞書型（varname -> NumPy配列）を透過的に再現しつつ、
    新仕様の構造化アクセスも可能にするカスタム辞書。
    """
    def __getitem__(self, key):
        val = super().__getitem__(key)
        # もし値がすでに CDFVar にラップされているならそのまま返す
        return val


def info(cdf_file_path, level='INFO'):
    target_level = level.upper()
    if display.LOG_LEVELS.get(target_level, 0) < display.CURRENT_LOG_LEVEL_THRESHOLD:
        return

    display.info(f'CDF filepath: {cdf_file_path}')
    dict_var_names = {}
    # 1回のオープンですべての情報を取得する
    with pycdf.CDF(cdf_file_path) as cdf_data:
        var_names = list(cdf_data.keys())
        for var in var_names:
            v_obj = cdf_data[var]
            description_i = f'{v_obj.shape}'
            
            if 'UNITS' in v_obj.attrs:
                description_i += f' [{v_obj.attrs["UNITS"]}]'
            if 'CATDESC' in v_obj.attrs:
                description_i += f' {v_obj.attrs["CATDESC"]}'
            
            dict_var_names[var] = description_i
        
    display.print_dict(dict_var_names, display_prefix=False)
    print('-' * 50)
    return dict_var_names


def dict_to_cdf(
        dict_data: dict,
        cdf_obj
):
    """
    開いているCDFオブジェクトに構造化データを書き込みます。
    """
    for key, value_item in dict_data.items():
        if isinstance(value_item, CDFVar):
            cdf_obj[key] = np.asarray(value_item)
            if value_item.unit:
                cdf_obj[key].attrs['UNITS'] = value_item.unit
            if value_item.desc:
                cdf_obj[key].attrs['CATDESC'] = value_item.desc
        elif isinstance(value_item, dict) and 'data' in value_item:
            cdf_obj[key] = value_item['data']
            if value_item.get('unit'):
                cdf_obj[key].attrs['UNITS'] = value_item['unit']
            if value_item.get('desc'):
                cdf_obj[key].attrs['CATDESC'] = value_item['desc']
        else:
            cdf_obj[key] = value_item
    return cdf_obj


def dict_to_cdffile(
        dict_data: dict,
        savecdf,
        dict_description=None,
        dict_units=None,
        global_attrs=None,
):
    """
    params
    --------
    dict_data: 'data', 'unit', 'desc'

    構造化辞書・旧仕様平面辞書をシームレスに判別し、新しいCDFファイルを生成します。
    
    ※ 既存コードの dict_description, dict_units の個別引き渡しにも
       完全に対応（上書きマージ）することで互換性を最大化しています。
    """
    path.make_directory(savecdf)
    if os.path.exists(savecdf):
        os.remove(savecdf)
        
    with pycdf.CDF(savecdf, '') as cdf_file:
        # グローバル属性の設定
        if global_attrs is not None:
            if isinstance(global_attrs, str):
                cdf_file.attrs['TITLE'] = global_attrs
                cdf_file.attrs['TEXT'] = global_attrs
            elif isinstance(global_attrs, dict):
                for attr_key, attr_val in global_attrs.items():
                    cdf_file.attrs[attr_key] = attr_val

        # 各変数データの書き込み
        for key, value_item in dict_data.items():
            try:
                # データの型による属性の振り分け
                if isinstance(value_item, CDFVar):
                    data_val = np.asarray(value_item)
                    unit_val = value_item.unit
                    desc_val = value_item.desc
                elif isinstance(value_item, dict) and 'data' in value_item:
                    data_val = value_item['data']
                    unit_val = value_item.get('unit')
                    desc_val = value_item.get('desc')
                else:
                    # 完全な平面データ（旧仕様互換）
                    data_val = value_item
                    unit_val = None
                    desc_val = None

                # 外部定義引数の優先・フォールバック上書き
                if dict_description and key in dict_description:
                    desc_val = dict_description[key]
                if dict_units and key in dict_units:
                    unit_val = dict_units[key]

                if not isinstance(data_val, (list, np.ndarray)):
                    data_val = [data_val]
                
                cdf_file[key] = data_val
                
                # 属性の書き込み
                if desc_val is not None:
                    cdf_file[key].attrs["CATDESC"] = str(desc_val)
                    cdf_file[key].attrs["description"] = str(desc_val)
                if unit_val is not None:
                    cdf_file[key].attrs["UNITS"] = str(unit_val)
                    cdf_file[key].attrs["units"] = str(unit_val)

            except Exception as e:
                display.error(f'Error in processing {key=}: {e}')
                
    display.info(f'Saved: {savecdf}')
    return


def cdffile_to_dict(cdf_filepath):
    """
    CDFファイルを読み込み、互換形式を兼ね備えた CDFDict を返します。

    これによって、既存の以下のコード：
    >>> dict_data = cdffile_to_dict(filepath)
    >>> b_array = dict_data['mag']  # 直接NumPy配列のように扱える
    
    および、新機能の：
    >>> b_unit = dict_data['mag'].unit  # 単位もシームレスに取り出せる
    >>> b_array_explicit = dict_data['mag']['data']  # 入れ子アクセスも可能
    
    がすべて同じコードの変更なしで同時に動作します。
    """
    dict_data = CDFDict()
    if not os.path.exists(cdf_filepath):
        display.warning(f'Not found: {cdf_filepath}')
        return None
        
    with pycdf.CDF(cdf_filepath) as cdf_file:
        var_names = list(cdf_file.keys())
        for var in var_names:
            try:
                v_obj = cdf_file[var]
                data_val = v_obj[...]
                
                # 単位および説明の抽出
                unit_val = None
                if 'UNITS' in v_obj.attrs:
                    unit_val = str(v_obj.attrs['UNITS'])
                elif 'units' in v_obj.attrs:
                    unit_val = str(v_obj.attrs['units'])
                    
                desc_val = None
                if 'CATDESC' in v_obj.attrs:
                    desc_val = str(v_obj.attrs['CATDESC'])
                elif 'description' in v_obj.attrs:
                    desc_val = str(v_obj.attrs['description'])
                
                # 特殊な多機能クラスでラップして格納
                dict_data[var] = CDFVar(data_val, unit=unit_val, desc=desc_val)
            except Exception as e:
                display.error(f'Error in processing {var}: {e}')
    return dict_data


# -------------------------------------------
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
    with pycdf.CDF(cdf_file_path) as cdf_data:
        return list(cdf_data.keys())




def _info(cdf_file_path):# 20260506
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

    dict_var_names = {}
    for i, var in enumerate(var_names):
        description_i = ''
        var_shape = cdf_data[var].shape
        # var_shape = np.array(cdf_data[var][:]).shape
        description_i += f'{var_shape}'
        var_attrs = cdf_data[var].attrs
        
        if 'UNITS' in var_attrs:
            units = var_attrs['UNITS']
            description_i += f' [{units}]'
        if 'CATDESC' in var_attrs:
            catdesc = var_attrs['CATDESC']
            description_i += f' {catdesc}'
        dict_var_names[var] = description_i
        
    # # shape of each data
    # shape_list = [np.array(cdf_data[i][:]).shape for i in var_names]

    # dict_var_names = {}
    # for i in range(len(var_names)):
    #     dict_var_names[var_names[i]] = shape_list[i]
    # dict_var_names = {
    #     "variables": var_names,
    #     "shapes": shape_list,
    # }
    display.print_dict(dict_var_names, prefix='CDF variables', only_prefix=True)

    return dict_var_names


def get_data(cdf_file_path, var):
    """
    データを読み込んだら即座に閉じる
    """
    with pycdf.CDF(cdf_file_path) as cdf:
        # [...] または [...] を使って全データをメモリにコピーして返す
        return cdf[var][...]


def _get_data(cdf_file_path, var):# 20260506
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
                        display.error(f'var {var} is not available: {vars_available=}')

            # 変数ごとに結合処理
            for var in vars:
                if var in vars_available:
                    # combined_data[var].append(cdf.varget(var))
                    combined_data[var].append(cdf[var][:])
                else:
                    display.error(f'var {var} is not available: {vars_available=}')

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


# def dict_to_cdf(# 20260613
#         dict_data: dict,
#         cdf
# ):
#     for key, value in dict_data.items():
#         cdf[key] = value
#     return cdf


# def dict_to_cdffile(# 20260613
#         dict_data: dict,
#         savecdf,
#         dict_description = None,
#         dict_units = None,
#         global_attrs=None,
# ):
#     """
#     dict -> cdf file

#     Parameters
#     ----------
#     dict_data : dict
#         書き出すデータの辞書。
#     savecdf : str
#         保存先のCDFファイルパス。
#     dict_description : dict, optional
#         変数ごとの説明（CATDESC）の辞書。
#     dict_units : dict, optional
#         変数ごとの単位（UNITS）の辞書。
#     global_attrs : dict or str, optional
#         ファイル全体の属性（グローバル属性）。
#         文字列が渡された場合は、自動的に標準的な {'TITLE': global_attrs} 構造に変換されます。
#     """
#     path.make_directory(savecdf)
#     if os.path.exists(savecdf):
#         os.remove(savecdf)
#     with pycdf.CDF(savecdf, '') as cdf:
#         if global_attrs is not None:
#             if isinstance(global_attrs, str):
#                 # 単一の文字列が渡された場合、標準的なタイトルとして割り当てる
#                 cdf.attrs['TITLE'] = global_attrs
#                 cdf.attrs['TEXT'] = global_attrs
#             elif isinstance(global_attrs, dict):
#                 # 辞書が渡された場合は、すべてのキー・値をグローバル属性としてそのまま追加
#                 for attr_key, attr_val in global_attrs.items():
#                     cdf.attrs[attr_key] = attr_val

#         for key, value in dict_data.items():
#             try:
#                 if not isinstance(value, (list, np.ndarray)):
#                     value = [value]
#                 cdf[key] = value
#                 # description
#                 if dict_description and key in dict_description:
#                     cdf[key].attrs["CATDESC"] = dict_description[key] # category description
#                 # unit
#                 if dict_units and key in dict_units:
#                     cdf[key].attrs["UNITS"] = dict_units[key] # unit

#             except Exception as e:
#                 display.error(f'Error in processing {key=}: {e}')
#     display.info(f'Saved: {savecdf}')
#     return


# def cdffile_to_dict(cdf_filepath):# 20260613
#     """
#     cdf file -> dict

#     Not existing cdf_filepath => return None
#     """
#     dict_data = {}
#     if not os.path.exists(cdf_filepath):
#         display.warning(f'Not found: {cdf_filepath}')
#         return
#     with pycdf.CDF(cdf_filepath) as cdf:
#         var_names = list(cdf.keys())
#         for var in var_names:
#             try:
#                 # 既に開いている cdf オブジェクトからデータを取得
#                 dict_data[var] = cdf[var][...]
#             except Exception as e:
#                 display.error(f'Error in processing {var}: {e}')
#     return dict_data


def _cdffile_to_dict(# 20260506
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

