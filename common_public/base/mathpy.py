"""
mathpy.py

* 基本的な数学的処理
"""

import numpy as np
import math
from . import display


def moving_average(
        data: np.ndarray,
        window_size: int,
):
    """
    Apply a centered moving average on 1D data, ignoring NaNs.
    """
    while window_size > len(data):
        print("[!] window size is larger than data length. -> window size = 0.1 * len(data)")
        window_size = int(.1 * len(data))

    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    half_window = window_size // 2
    averaged_data = []
    indices = []

    for i in range(len(data)):
        # Define window range
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]

        if np.all(np.isnan(window)):
            averaged_data.append(np.nan)
        else:
            averaged_data.append(np.nanmean(window))
            indices.append(i)

    averaged_data = np.array(averaged_data)
    return averaged_data


def moving_average_vec(
        data: np.ndarray,  # (n, *)
        window_size: int,
):
    if data.ndim == 1:
        return moving_average(data, window_size)
    else:
        averaged_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            averaged_data[:, i] = moving_average(data[:, i], window_size)
        return averaged_data


def moving_average_by_time(times, data, window_sec):
    """
    Computes a moving average based on a time window, ignoring NaNs.
    Supports 1D arrays (n,) and ND arrays (n, m, ...).
    
    :param times: 1D array of timestamps (n,)
    :param data: Data array where the first dimension is time (n, ...)
    :param window_sec: Window width in seconds
    :return: Averaged data with the same shape as input
    """
    n = len(times)
    result = np.full_like(data, np.nan, dtype=float)
    half_window = window_sec / 2.0
    
    # 1. NaN を 0 に置き換えたデータを作成
    data_fixed = np.where(np.isnan(data), 0, data)
    
    # 2. 有効なデータかどうかのマスクを作成 (NaN=0, Valid=1)
    mask = np.where(np.isnan(data), 0, 1)

    # 3. データの累積和
    cumsum_shape = (n + 1,) + data.shape[1:]
    cumsum = np.zeros(cumsum_shape, dtype=float)
    cumsum[1:] = np.cumsum(data_fixed, axis=0)
    
    # 4. 有効データ数の累積和 (count の代わり)
    cum_mask = np.zeros(cumsum_shape, dtype=float)
    cum_mask[1:] = np.cumsum(mask, axis=0)

    # 各時刻のウィンドウ範囲のインデックス
    left_indices = np.searchsorted(times, times - half_window, side='left')
    right_indices = np.searchsorted(times, times + half_window, side='right')

    for i in range(n):
        l, r = left_indices[i], right_indices[i]
        
        # ウィンドウ内の有効なデータ数を取得
        valid_count = cum_mask[r] - cum_mask[l]
        
        # 全ての要素が 0 より大きいかチェック (多次元対応)
        # 1つでも有効なデータがあれば平均を計算
        if np.any(valid_count > 0):
            # 有効なデータのみの合計 / 有効なデータ数
            # 0除算を防ぐため、valid_count が 0 の場所は NaN にする
            with np.errstate(divide='ignore', invalid='ignore'):
                avg = (cumsum[r] - cumsum[l]) / valid_count
                # valid_count が 0 の要素がある場合、その要素は NaN になる
                result[i] = avg
        else:
            # ウィンドウ内に有効なデータが一つもない場合
            result[i] = np.nan
            
    return result

def _moving_average_by_time(times, data, window_sec):
    """
    Computes a moving average based on a time window.
    Supports 1D arrays (n,) and ND arrays (n, m, ...).
    
    :param times: 1D array of timestamps (n,)
    :param data: Data array where the first dimension is time (n, ...)
    :param window_sec: Window width in seconds
    :return: Averaged data with the same shape as input
    """
    n = len(times)
    # 入力データの形状を保持し、結果の格納先を初期化
    result = np.zeros_like(data, dtype=float)
    half_window = window_sec / 2.0
    
    # 高速化のため累積和を使用
    # data が (n,) の場合は (n+1,)、(n, 3) の場合は (n+1, 3) となるように shape を設定
    cumsum_shape = (n + 1,) + data.shape[1:]
    cumsum = np.zeros(cumsum_shape, dtype=float)
    
    # 累積和の計算 (axis=0 は常に時間軸)
    cumsum[1:] = np.cumsum(data, axis=0)

    # 各時刻 t において [t - window/2, t + window/2] の範囲のインデックスを特定
    left_indices = np.searchsorted(times, times - half_window, side='left')
    right_indices = np.searchsorted(times, times + half_window, side='right')

    # count を計算 (n,) の配列
    counts = right_indices - left_indices
    
    # 形状に合わせて割り算を行うための準備
    # data が多次元の場合、counts の次元を合わせる (例: (n, 1) や (n, 1, 1))
    if data.ndim > 1:
        reshaped_counts = counts.reshape((n,) + (1,) * (data.ndim - 1))
    else:
        reshaped_counts = counts

    # インデックスをループで処理（ここがボトルネックになる場合はスライシングで一括計算も可能）
    for i in range(n):
        l, r = left_indices[i], right_indices[i]
        c = counts[i]
        if c > 0:
            # 累積和から範囲内の平均を算出
            # cumsum[r] と cumsum[l] は data[i] と同じ形状
            result[i] = (cumsum[r] - cumsum[l]) / c
        else:
            result[i] = data[i]
            
    return result


def fast_moving_average_vec(data: np.ndarray, window_size: int, mode='valid'):
    # 窓サイズのチェック (元のロジックを維持)
    while window_size > len(data):
        print("[!] window size is larger than data length. -> window size = 0.1 * len(data)")
        window_size = int(.1 * len(data))

    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    # 多次元対応：(n, m, ...) の形式でも最初の軸に対して処理を行う
    # 1Dの場合は (n, 1) 的に扱えるよう調整
    input_shape = data.shape
    if data.ndim == 1:
        data_2d = data[:, np.newaxis]
    else:
        # (n, m) の形に平坦化して一括処理
        data_2d = data.reshape(data.shape[0], -1)

    n = data_2d.shape[0]
    half_window = window_size // 2

    # NaNを0に置換したデータと、有効なデータのマスク(1 or 0)を作成
    mask = (~np.isnan(data_2d)).astype(float)
    data_zeroed = np.nan_to_num(data_2d, nan=0.0)

    # 累積和を計算 (先頭に0を追加して差分計算を容易にする)
    cumsum_data = np.cumsum(np.vstack([np.zeros((1, data_2d.shape[1])), data_zeroed]), axis=0)
    cumsum_mask = np.cumsum(np.vstack([np.zeros((1, data_2d.shape[1])), mask]), axis=0)

    # 各要素 i に対して [start:end] の範囲を計算
    # start = max(0, i - half_window)
    # end = min(n, i + half_window + 1)
    i = np.arange(n)
    starts = np.maximum(0, i - half_window)
    ends = np.minimum(n, i + half_window + 1)

    # 区間和の計算 (ベクトル化)
    window_sums = cumsum_data[ends] - cumsum_data[starts]
    window_counts = cumsum_mask[ends] - cumsum_mask[starts]

    # 平均の計算 (countが0の場合はNaNにする)
    # 実行速度向上のため、0除算の警告を無視し、後でNaN補正
    with np.errstate(divide='ignore', invalid='ignore'):
        result = window_sums / window_counts
        result[window_counts == 0] = np.nan

    # 元の形状に復元
    return result.reshape(input_shape)


def solve_equation(
        xdata,
        equation,
        precision = 1e-3,
        get_detail=False,# True -> return dict
    ):
    """
    Solve: (equation) = 0
        * equation: 1d-array
    
    Params
    -------
    * get_detail=False
        True -> return dict: 'all', 'neg_to_pos', 'pos_to_neg'
    """
    if len(xdata) != len(equation):
        display.error('ioncomp/solve_eq', 'length of xdata and equation must be same')
        return None

    roots_at_x = []
    roots_neg_to_pos = [] # 負 -> 正
    roots_pos_to_neg = [] # 正 -> 負
    
    
    for i in range(len(equation) - 1):
        y1, y2 = equation[i], equation[i+1]
        x1, x2 = xdata[i], xdata[i+1]
        
        root_val = None
        
        # 1. 符号反転のチェック
        if np.sign(y1) * np.sign(y2) < 0:
            # 精度(precision)に収まる近い方のxを選択
            if np.abs(y1) <= np.abs(y2) and np.abs(y1) < precision:
                root_val = x1
            elif np.abs(y1) > np.abs(y2) and np.abs(y2) < precision:
                root_val = x2
            
            # 反転方向の分類
            if root_val is not None:
                if y1 < 0 and y2 > 0:
                    roots_neg_to_pos.append(root_val)
                elif y1 > 0 and y2 < 0:
                    roots_pos_to_neg.append(root_val)
                    
        # 2. ちょうど0の場合のチェック
        elif y1 == 0:
            root_val = x1
            # ちょうど0の場合は反転方向が定まらないため詳細リストには含めない
            # (必要であれば前後の符号を見て分類するロジックを追加可能)

        # 全体の解リストに追加
        if root_val is not None:
            roots_at_x.append(root_val)
    
    # 重複除去やソートが必要な場合はここで行う（np.uniqueなど）
    roots_at_x = np.array(roots_at_x)
    roots_neg_to_pos = np.array(roots_neg_to_pos)
    roots_pos_to_neg = np.array(roots_pos_to_neg)

    if get_detail:
        return {
            'all': roots_at_x,
            'neg_to_pos': roots_neg_to_pos,
            'pos_to_neg': roots_pos_to_neg
        }
    else:
        return roots_at_x


# def solve_equation(# 2026.01.07
#         xdata,
#         equation,
#         precision = 1e-3,
#         get_detail=False,# True -> return dict
#     ):
#     """
#     Solve: (equation) = 0
#         * equation: 1d-array
#     """
#     if len(xdata) != len(equation):
#         display.error('ioncomp/solve_eq', 'length of xdata and equation must be same')
#         return None

#     roots_at_x = []
#     roots_neg_to_pos = [] # 負 -> 正
#     roots_pos_to_neg = [] # 正 -> 負
    
    
#     for i in range(len(equation) - 1):
#         if np.sign(equation[i]) * np.sign(equation[i+1]) < 0:  # sign inversion
#             if np.abs(equation[i]) <= np.abs(equation[i+1]) and np.abs(equation[i]) < precision:
#                 roots_at_x.append(xdata[i])
#             elif np.abs(equation[i]) > np.abs(equation[i+1]) and np.abs(equation[i+1]) < precision:
#                 roots_at_x.append(xdata[i+1])
#         elif equation[i] == 0:
#             roots_at_x.append(xdata[i])
    
#     roots_at_x = np.array(roots_at_x)

#     return roots_at_x


def interp_vec(
        x,
        xp,
        fp_vec
):
    if fp_vec.ndim == 1:
        return np.interp(x, xp, fp_vec)
    elif fp_vec.ndim == 2:
        result = np.zeros((len(x), fp_vec.shape[1]))
        for i in range(fp_vec.shape[1]):
            result[:, i] = np.interp(x, xp, fp_vec[:, i])
        return result
    else:
        display.error('ndim > 3 is not supported')
        return None
    

def convert_nan_to_zero(
        array_data
):
    indices_isnan = np.isnan(array_data)
    array_data_returun = np.copy(array_data)
    array_data_returun[indices_isnan] = 0
    return array_data_returun


def rounding(x):
    return math.floor(x + 0.5)


def resample_data_1d(
    original_times_unix: np.ndarray,
    original_data: np.ndarray,
    # original_sampling_rate: float = 20.0, # Hz
    target_sampling_rate: float = 10.0,   # Hz
):
    start_time_unix = original_times_unix[0]
    end_time_unix = original_times_unix[-1]

    target_dt = 1.0 / target_sampling_rate

    resampled_times_unix = np.arange(start_time_unix, end_time_unix + target_dt / 2.0, target_dt)

    if len(resampled_times_unix) == 0:
        print("Warning: No new timestamps generated. Check original time range and target sampling rate.")
        return np.array([]), np.array([])
    
    resampled_data = np.interp(resampled_times_unix, original_times_unix, original_data)
    return resampled_times_unix, resampled_data


def resample_data(
    original_times_unix: np.ndarray,
    original_data: np.ndarray, # (N, 3) 形状の磁場データ (Bx, By, Bz)
    # original_sampling_rate: float = 20.0, # Hz
    target_sampling_rate: float = 10.0,   # Hz
    force_upsampling=False,
    epsilon_sampling_rate = 1, # permitted differece between original sampling rate and target one [Hz]
    get_support_data=False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    磁場データを指定されたサンプリングレートにリサンプリングします。
    線形補間を使用します。

    Parameters
    ----------
    original_times_unix : np.ndarray
        元の磁場データのUnixタイムスタンプ (1次元配列)。
    original_mag_data : np.ndarray
        元の磁場データ (N, 3) 形状。
    original_sampling_rate : float
        元のデータのサンプリングレート (Hz)。
    target_sampling_rate : float
        目標とするサンプリングレート (Hz)。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        リサンプリングされたUnixタイムスタンプ (1次元配列) と
        リサンプリングされた磁場データ (N_new, 3) 形状。
    
    get_support_data=True -> dict
        * 'target_sampling_rate'
    """

    if original_times_unix.ndim != 1:
        raise ValueError("original_times_unix must be a 1D numpy array.")
    # if original_data.ndim != 2 or original_data.shape[1] != 3:
    #     raise ValueError("original_mag_data must be a 2D numpy array with 3 columns (Bx, By, Bz).")
    if original_times_unix.shape[0] != original_data.shape[0]:
        raise ValueError("Lengths of original_times_unix and original_mag_data do not match.")
    if target_sampling_rate <= 0:
        raise ValueError("Target sampling rate must be greater than 0.")

    dt_original = np.mean(np.diff(original_times_unix))
    # original_sampling_rate = int(1 / dt_original)
    original_sampling_rate = rounding(1 / dt_original)
    display.info(f'Resampling: {original_sampling_rate} ({1/dt_original:.2f}) Hz -> {target_sampling_rate} Hz')
    if target_sampling_rate - original_sampling_rate > epsilon_sampling_rate:
        display.warning("Upsampling detected.")
        if not force_upsampling:
            target_sampling_rate = original_sampling_rate
            display.info(f'target sampling rate was changed into original sampling rate: {original_sampling_rate} ({1/dt_original:.2f}) Hz -> {target_sampling_rate} Hz')
    
    if original_data.ndim == 1:
        return resample_data_1d(
            original_times_unix,
            original_data,
            target_sampling_rate=target_sampling_rate
        )
    
    else:
        # 1. リサンプリング後の時間軸を生成
        # ----------------------------------
        # 元データの開始時刻と終了時刻を取得
        start_time_unix = original_times_unix[0]
        end_time_unix = original_times_unix[-1]

        # リサンプリング後の時間間隔 (秒)
        target_dt = 1.0 / target_sampling_rate

        # 新しいUnixタイムスタンプ配列を生成
        # np.arange は終点を含まない可能性があるので、少し余裕を持たせる
        resampled_times_unix = np.arange(start_time_unix, end_time_unix + target_dt / 2.0, target_dt)

        # データ期間が短いなどでresampled_times_unixが空になる場合があるためチェック
        if len(resampled_times_unix) == 0:
            print("Warning: No new timestamps generated. Check original time range and target sampling rate.")
            return np.array([]), np.array([])
            
        # 2. 磁場データを新しい時間軸に補間
        # ------------------------------
        resampled_mag_data = np.zeros((len(resampled_times_unix), 3), dtype=original_data.dtype)

        # 各成分 (Bx, By, Bz) ごとに補間を行う
        for i in range(original_data.shape[1]): # 0, 1, 2
            # `kind='linear'` は線形補間
            # `fill_value='extrapolate'` は、新しいタイムスタンプが元のデータの範囲外にある場合に、
            # 端の傾きで外挿する。これによりNaNが生成されるのを防ぐが、外挿は注意が必要。
            # 範囲外をNaNにしたい場合は `fill_value=np.nan` を使用する。
            # interpolator = interp1d(original_times_unix, original_mag_data[:, i], kind='linear', fill_value='extrapolate')
            # resampled_mag_data[:, i] = interpolator(resampled_times_unix)
            resampled_mag_data[:, i] = np.interp(resampled_times_unix, original_times_unix, original_data[:, i])

        dict_support_data = {
            'target_sampling_rate': target_sampling_rate
        }
        if get_support_data:
            return resampled_times_unix, resampled_mag_data, dict_support_data
        else:
            return resampled_times_unix, resampled_mag_data



# def old_moving_average(
#         time: np.ndarray,
#         data: np.ndarray,
#         window_size: int,
#         extract_indices: bool = False
# ):
#     """
#     Apply a centered moving average on 1D data.
#     """
#     while window_size > len(data):
#         print("[!] window size is larger than data length. -> window size = 0.1 * len(data)")
#         window_size = int(.1 * len(data))

#     if window_size < 1:
#         raise ValueError("window_size must be at least 1")

#     # Preallocate the result array
#     averaged_data = np.zeros(len(data) - window_size + 1)

#     # Compute the sum over the first window
#     window_sum = np.sum(data[:window_size])

#     # First value of the moving average
#     averaged_data[0] = window_sum / window_size

#     # Slide the window and update the sum
#     for i in range(1, len(data) - window_size + 1):
#         # Subtract the element leaving the window and add the new element
#         window_sum = window_sum - data[i - 1] + data[i + window_size - 1]
#         averaged_data[i] = window_sum / window_size

#     # Align the time array to match the averaged data
#     if window_size % 2 == 0:
#         start_idx = window_size // 2
#         end_idx = -(window_size // 2) + 1
#         indices = np.arange(start_idx, len(data) - window_size // 2 + 1)  # Indices of the data points used
#     else:
#         start_idx = window_size // 2
#         end_idx = -(window_size // 2)
#         indices = np.arange(start_idx, len(data) - window_size // 2)  # Indices of the data points used

#     aligned_time = time[start_idx:end_idx]

#     if extract_indices:
#         return aligned_time, averaged_data, indices
#     else:
#         return aligned_time, averaged_data


# def old_moving_average_v2(
#         # time: np.ndarray,
#         data: np.ndarray,
#         window_size: int,
#         # extract_indices: bool = False
# ):
#     """
#     Apply a centered moving average on 1D data, ignoring NaNs.
#     """
#     data = np.asarray(data)
#     # time = np.asarray(time)

#     while window_size > len(data):
#         print("[!] window size is larger than data length. -> window size = 0.1 * len(data)")
#         window_size = int(.1 * len(data))

#     if window_size < 1:
#         raise ValueError("window_size must be at least 1")

#     half_window = window_size // 2
#     averaged_data = []
#     indices = []

#     for i in range(len(data)):
#         # Define window range
#         start = max(0, i - half_window)
#         end = min(len(data), i + half_window + 1)
#         window = data[start:end]

#         if np.all(np.isnan(window)):
#             averaged_data.append(np.nan)
#         else:
#             averaged_data.append(np.nanmean(window))
#             indices.append(i)

#     averaged_data = np.array(averaged_data)
#     # aligned_time = time  # サイズは変えない前提（NaN埋めで長さ一致）

#     # if extract_indices:
#     #     return aligned_time, averaged_data, np.array(indices)
#     # else:
#     return averaged_data
    

