import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime, timedelta, timezone
from common import display, mathpy

display.error('Old version -> Use mathpy.resample_data')

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
    original_sampling_rate = mathpy.rounding(1 / dt_original)
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
