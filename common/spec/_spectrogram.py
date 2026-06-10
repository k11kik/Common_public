import numpy as np
import pandas as pd
from scipy import fft
from common import display


def _fft(times, data, window="hanning", positive_freq=False, info=False):
    """
    :return dict; freq, spec, specabs, powspec, psd
    """
    dt = np.mean(np.diff(times))
    time_total = times[-1] - times[0]
    nyquist_freq = .5 / dt  # Nyquist frequency
    dict_info = {
        'times': len(times),
        'data': len(data),
        'dt': f'{dt:.02} s',
        'total time': f'{time_total:.02} s',
        'Nyquist freq': f'{nyquist_freq}',
        'windwow': window,
    }

    len_data = len(data)

    # window
    if window == "hanning":
        w = np.hanning(len_data)  # ハニング窓
    else:  # no window
        w = np.ones_like(data)


    fft_data = fft.fft(data * w) / len_data  # [x]

    if positive_freq:
        freq = np.linspace(0, int(nyquist_freq), len_data//2)
        fft_data = fft_data[:len_data // 2]  # 正の周波数部分を抽出
        fft_data *= 2  # 負の周波数のエネルギー分
    else:
        fft_data = fft.fftshift(fft_data)
        freq = fft.fftfreq(len_data, dt)
        freq = fft.fftshift(freq)

    # 補正
    acf = 1 / (np.sum(w) / len_data)
    fft_data *= acf  # 補正 [x]

    if info:
        display.print_dict(dict_info, title='FFT information')


    fft_amp = np.abs(fft_data)  # amplitude
    power_spectrum = fft_amp ** 2  # [x^2]
    psd = power_spectrum * time_total  # [x^2/Hz]

    dict_return = {
        'freq': freq,
        'spec': fft_data,
        'specabs': fft_amp,
        'powspec': power_spectrum,
        'psd': psd
    }
    return dict_return


def _stft(
        times,
        data,
        window="hanning",
        window_size=256, rate_overlap=.1,
        info=False,
        positive_freq=True,
        nyquist_freq=None
):
    """
    :return: dict; times, freqs, spectrogram (len_times, len_freqs), spectrogram_psd
    """


    if rate_overlap < 0 or rate_overlap >= 1:
        raise ValueError("rate_overlap must be in the range of (0, 1)")

    if window_size >= len(data):
        window_size = int(len(data) / 10)
        print("window_size must be smaller than data length")
        # return None

    n_overlap = int(window_size * rate_overlap)
    step = window_size - n_overlap

    dt = np.mean(np.diff(times))

    if nyquist_freq is None:
        nyquist_freq = .5 / dt
    delta_t = window_size * dt  # time resolution

    if positive_freq:
        if nyquist_freq > 1:
            freq = np.linspace(0, int(nyquist_freq), window_size//2)
        else:
            freq = np.linspace(0, nyquist_freq, window_size//2)
    else:
        freq = fft.fftfreq(window_size, dt)
        freq = fft.fftshift(freq)

    freq_res = freq[1] - freq[0]

    spectrogram = []
    spectrogram_psd = []
    time_midpoints = []

    # window ごとに処理
    for i in range(0, len(times) - window_size, step):
        times_for_fft = times[i:i+window_size]
        data_for_fft = data[i:i+window_size]
        dict_fft = _fft(times_for_fft, data_for_fft, positive_freq=positive_freq, window=window)
        spectrogram.append(dict_fft['spec'])
        spectrogram_psd.append(dict_fft['psd'])

        # セグメントの中央の時間を記録
        time_midpoint = times[i + window_size // 2]
        time_midpoints.append(time_midpoint)

    time_midpoints = np.array(time_midpoints)
    spectrogram = np.array(spectrogram)
    spectrogram_psd = np.array(spectrogram_psd)

    if info:
        dict_info = {
            'times': len(times),
            'data': len(data),
            'window': window,
            'window size': window_size,
            'rate overlap': rate_overlap,
            'positive freq': positive_freq,
            'dt': dt,
            'frequency res': freq_res,
            'time res': delta_t,
            'freqs': len(freq),
            'time_midpoints': len(time_midpoints),
            'spectrogram': spectrogram.shape,
            'spectrogram_psd': spectrogram_psd.shape
        }
        display.print_dict(dict_info, title='STFT information')

    dict_to_return = {
        'freqs': freq,
        'times': time_midpoints,
        'spectrogram': spectrogram,
        'spectrogram_psd': spectrogram_psd,
    }

    return dict_to_return


def stft_vec(
        times,
        data_vec,
        window_size=256, rate_overlap=.1,
        nyquist_freq=None
):
    """
    STFT 3D
    :param epoch: np.ndarray (n,)
    :param waveform_x: np.ndarray (n,)
    :param waveform_y: np.ndarray (n,)
    :param waveform_z: np.ndarray (n,)
    :param datatype: Default: spec
    :param window_size: window size of FFT
    :param rate_overlap: overlapping rate
    :return: freqs, times, spec_x, spec_y, spec_z, indices
    """
    data_x = data_vec[:, 0]
    data_y = data_vec[:, 1]
    data_z = data_vec[:, 2]
    dict_stft_x = _stft(times, data_x, window_size=window_size, rate_overlap=rate_overlap, nyquist_freq=nyquist_freq)
    dict_stft_y = _stft(times, data_y, window_size=window_size, rate_overlap=rate_overlap, nyquist_freq=nyquist_freq)
    dict_stft_z = _stft(times, data_z, window_size=window_size, rate_overlap=rate_overlap, nyquist_freq=nyquist_freq)

    dict_to_return = {
        'times': dict_stft_x['times'],
        'freqs': dict_stft_x['freqs'],
        'spectrogram_x': dict_stft_x['spectrogram'],
        'spectrogram_y': dict_stft_y['spectrogram'],
        'spectrogram_z': dict_stft_z['spectrogram'],
        'spectrogram_psd_x': dict_stft_x['spectrogram_psd'],
        'spectrogram_psd_y': dict_stft_y['spectrogram_psd'],
        'spectrogram_psd_z': dict_stft_z['spectrogram_psd'],
    }
    return dict_to_return


def stft_vec_variable(
    times,
    data_vec,
    window_second=2.0,
    rate_overlap=0.1,
    window="hanning",
    positive_freq=True,
    n_freq_bins=None
):
    """
    サンプリング周波数が変動するデータに対し、窓の「秒数」を一定に保ち、
    FFTポイント数を2の乗数に調整して処理するSTFT実装。
    
    Returns:
        dict: 複素数データとPSDデータを含む辞書
    """
    n_data = len(times)
    
    # 処理結果の一時保管用
    raw_results = {
        'times': [],
        'freqs': [],
        'fft_x': [], 'fft_y': [], 'fft_z': [],
        'psd_x': [], 'psd_y': [], 'psd_z': []
    }
    
    curr_idx = 0
    # actual_window_sizes = [] # 実際のデータポイント数
    # fft_points = []          # 2の乗数に調整されたFFTポイント数

    window_changes = {
        'indices': [],       # raw_results['times'] の何番目で切り替わったか
        'n_ffts': []         # 2の乗数に調整されたFFTポイント数
    }
    last_n_fft = None

    while True:
        t_start = times[curr_idx]
        t_end_target = t_start + window_second
        
        if t_end_target > times[-1]:
            break
            
        end_idx = np.searchsorted(times, t_end_target, side='left')
        win_size_local = end_idx - curr_idx
        
        # 窓内のデータが少なすぎる場合はスキップ
        if win_size_local < 4:
            curr_idx += 1
            if curr_idx >= n_data: break
            continue

        # 2の乗数への調整 (win_size_local 以上の最小の2の乗数)
        n_fft = 2**int(np.ceil(np.log2(win_size_local)))
        
        # actual_window_sizes.append(win_size_local)
        # fft_points.append(n_fft)

        t_seg = times[curr_idx : end_idx]
        d_seg = data_vec[curr_idx : end_idx, :]
        dt_local = np.mean(np.diff(t_seg))
        time_total_local = t_seg[-1] - t_seg[0]

        # 窓関数の作成
        if window == "hanning":
            w = np.hanning(win_size_local)
        else:
            w = np.ones(win_size_local)
        
        # 振幅補正係数 (ゼロパディングを考慮せず、元の窓関数に対して計算)
        acf = 1.0 / (np.sum(w) / win_size_local)
        nyquist_local = 0.5 / dt_local

        if positive_freq:
            num_bins = n_fft // 2
            f_local = np.linspace(0, nyquist_local, num_bins)
        else:
            num_bins = n_fft
            f_local = fft.fftshift(fft.fftfreq(n_fft, dt_local))

        # データの蓄積用バッファ
        step_ffts = []
        step_psds = []

        for axis in range(3):
            # FFT実行 (n=n_fft で自動的にゼロパディングされる)
            # 正規化は元の窓サイズ win_size_local で行う
            fft_raw = fft.fft(d_seg[:, axis] * w, n=n_fft) / win_size_local
            
            if positive_freq:
                fft_val = fft_raw[:num_bins] * 2.0
            else:
                fft_val = fft.fftshift(fft_raw)
            
            fft_val *= acf
            
            # 複素数データとして保持
            step_ffts.append(fft_val)
            
            # PSD算出
            psd = (np.abs(fft_val)**2) * time_total_local
            step_psds.append(psd)

        # データの格納
        raw_results['times'].append(t_start + (time_total_local / 2.0))
        raw_results['freqs'].append(f_local)
        raw_results['fft_x'].append(step_ffts[0])
        raw_results['fft_y'].append(step_ffts[1])
        raw_results['fft_z'].append(step_ffts[2])
        raw_results['psd_x'].append(step_psds[0])
        raw_results['psd_y'].append(step_psds[1])
        raw_results['psd_z'].append(step_psds[2])

        if n_fft != last_n_fft:
            idx_spectrogram = len(raw_results['times']) - 1
            window_changes['indices'].append(idx_spectrogram)
            window_changes['n_ffts'].append(n_fft)
            last_n_fft = n_fft

        # 次のステップ計算
        step_second = window_second * (1.0 - rate_overlap)
        next_t_start = t_start + step_second
        curr_idx = np.searchsorted(times, next_t_start, side='left')
        if curr_idx >= n_data - 1: break

    if not raw_results['times']:
        return None

    # --- 共通周波数軸への補間処理 ---
    max_f = max([f[-1] for f in raw_results['freqs']])
    if n_freq_bins is None:
        delta_f_physical = 1.0 / window_second
        # 最大周波数までを物理的分解能で割ってビン数を決定
        n_freq_bins = int(np.ceil(max_f / delta_f_physical))
    
        # # 補間後のビン数も2の乗数の半分程度をデフォルトにする
        # n_freq_bins = max([len(f) for f in raw_results['freqs']])
    
    common_freqs = np.linspace(0, max_f, n_freq_bins)
    
    def interp_real(f_src, s_src):
        return np.interp(common_freqs, f_src, s_src, left=0.0, right=0.0)

    def interp_complex(f_src, s_src):
        real_part = np.interp(common_freqs, f_src, s_src.real, left=0.0, right=0.0)
        imag_part = np.interp(common_freqs, f_src, s_src.imag, left=0.0, right=0.0)
        return real_part + 1j * imag_part

    final_fft_x = np.array([interp_complex(f, s) for f, s in zip(raw_results['freqs'], raw_results['fft_x'])])
    final_fft_y = np.array([interp_complex(f, s) for f, s in zip(raw_results['freqs'], raw_results['fft_y'])])
    final_fft_z = np.array([interp_complex(f, s) for f, s in zip(raw_results['freqs'], raw_results['fft_z'])])
    
    final_psd_x = np.array([interp_real(f, s) for f, s in zip(raw_results['freqs'], raw_results['psd_x'])])
    final_psd_y = np.array([interp_real(f, s) for f, s in zip(raw_results['freqs'], raw_results['psd_y'])])
    final_psd_z = np.array([interp_real(f, s) for f, s in zip(raw_results['freqs'], raw_results['psd_z'])])

    return {
        'times': np.array(raw_results['times'], dtype=np.float64),
        'freqs': common_freqs, 
        'spectrogram_x': final_fft_x,
        'spectrogram_y': final_fft_y,
        'spectrogram_z': final_fft_z,
        'spectrogram_psd_x': final_psd_x,
        'spectrogram_psd_y': final_psd_y,
        'spectrogram_psd_z': final_psd_z,
        'window': window_changes,
    }


def _stft_vec_variable(
    times,
    data_vec,
    window_second=2.0,
    rate_overlap=0.1,
    window="hanning",
    positive_freq=True,
    n_freq_bins=None
):
    """
    サンプリング周波数が変動するデータに対し、窓の「秒数」を一定に保ち、
    Numpyのみを使用して共通の周波数軸へ補間・整形するSTFT実装。
    
    Returns:
        dict: 複素数データ(spectrogram_x)とPSDデータ(spectrogram_psd_x)を含む辞書
    """
    n_data = len(times)
    
    # 処理結果の一時保管用
    raw_results = {
        'times': [],
        'freqs': [],
        'fft_x': [], 'fft_y': [], 'fft_z': [],
        'psd_x': [], 'psd_y': [], 'psd_z': []
    }
    
    curr_idx = 0
    window_sizes = []
    while True:
        t_start = times[curr_idx]
        t_end_target = t_start + window_second
        
        if t_end_target > times[-1]:
            break
            
        end_idx = np.searchsorted(times, t_end_target, side='left')
        win_size_local = end_idx - curr_idx

        window_sizes.append(win_size_local)
        
        # 窓内のデータが少なすぎる場合はスキップ
        if win_size_local < 4:
            curr_idx += 1
            if curr_idx >= n_data: break
            continue

        t_seg = times[curr_idx : end_idx]
        d_seg = data_vec[curr_idx : end_idx, :]
        dt_local = np.mean(np.diff(t_seg))
        time_total_local = t_seg[-1] - t_seg[0]

        # 窓関数の作成
        if window == "hanning":
            w = np.hanning(win_size_local)
        else:
            w = np.ones(win_size_local)
        
        acf = 1.0 / (np.sum(w) / win_size_local)
        nyquist_local = 0.5 / dt_local

        if positive_freq:
            num_bins = win_size_local // 2
            f_local = np.linspace(0, nyquist_local, num_bins)
        else:
            num_bins = win_size_local
            f_local = fft.fftshift(fft.fftfreq(win_size_local, dt_local))

        # データの蓄積用バッファ
        step_ffts = []
        step_psds = []

        for axis in range(3):
            # FFT実行 (正規化)
            fft_raw = fft.fft(d_seg[:, axis] * w) / win_size_local
            
            # 正の周波数のみ抽出、またはシフト
            # fft_val は振幅補正(acf)と正の周波数の2倍補正を含む
            if positive_freq:
                fft_val = fft_raw[:num_bins] * 2.0
            else:
                fft_val = fft.fftshift(fft_raw)
            
            fft_val *= acf
            
            # 複素数データとして保持
            step_ffts.append(fft_val)
            
            # PSD算出 (振幅の2乗 * 時間幅)
            psd = (np.abs(fft_val)**2) * time_total_local
            step_psds.append(psd)

        # データの格納
        raw_results['times'].append(t_start + (time_total_local / 2.0))
        raw_results['freqs'].append(f_local)
        raw_results['fft_x'].append(step_ffts[0])
        raw_results['fft_y'].append(step_ffts[1])
        raw_results['fft_z'].append(step_ffts[2])
        raw_results['psd_x'].append(step_psds[0])
        raw_results['psd_y'].append(step_psds[1])
        raw_results['psd_z'].append(step_psds[2])

        # 次のステップ計算
        step_second = window_second * (1.0 - rate_overlap)
        next_t_start = t_start + step_second
        curr_idx = np.searchsorted(times, next_t_start, side='left')
        if curr_idx >= n_data - 1: break

    if not raw_results['times']:
        return None

    # --- 共通周波数軸への補間処理 (Numpyのみ) ---
    max_f = max([f[-1] for f in raw_results['freqs']])
    if n_freq_bins is None:
        n_freq_bins = max([len(f) for f in raw_results['freqs']])
    
    common_freqs = np.linspace(0, max_f, n_freq_bins)
    
    def interp_real(f_src, s_src):
        """実数データの補間"""
        return np.interp(common_freqs, f_src, s_src, left=0.0, right=0.0)

    def interp_complex(f_src, s_src):
        """複素数データの補間 (実部と虚部を分けて補間)"""
        real_part = np.interp(common_freqs, f_src, s_src.real, left=0.0, right=0.0)
        imag_part = np.interp(common_freqs, f_src, s_src.imag, left=0.0, right=0.0)
        return real_part + 1j * imag_part

    # 全時間ステップを行列化
    # 複素数スペクトログラム
    final_fft_x = np.array([interp_complex(f, s) for f, s in zip(raw_results['freqs'], raw_results['fft_x'])])
    final_fft_y = np.array([interp_complex(f, s) for f, s in zip(raw_results['freqs'], raw_results['fft_y'])])
    final_fft_z = np.array([interp_complex(f, s) for f, s in zip(raw_results['freqs'], raw_results['fft_z'])])
    
    # PSDスペクトログラム
    final_psd_x = np.array([interp_real(f, s) for f, s in zip(raw_results['freqs'], raw_results['psd_x'])])
    final_psd_y = np.array([interp_real(f, s) for f, s in zip(raw_results['freqs'], raw_results['psd_y'])])
    final_psd_z = np.array([interp_real(f, s) for f, s in zip(raw_results['freqs'], raw_results['psd_z'])])

    display.debug(f'{window_sizes=}')
    return {
        'times': np.array(raw_results['times'], dtype=np.float64),
        'freqs': common_freqs, 
        'spectrogram_x': final_fft_x,     # Complex 2D array
        'spectrogram_y': final_fft_y,     # Complex 2D array
        'spectrogram_z': final_fft_z,     # Complex 2D array
        'spectrogram_psd_x': final_psd_x, # Real 2D array (PSD)
        'spectrogram_psd_y': final_psd_y,
        'spectrogram_psd_z': final_psd_z,
        'window_size': window_sizes
    }


def _stft_vec_variable(
    times,
    data_vec,
    window_second=2.0,
    rate_overlap=0.1,
    window="hanning",
    positive_freq=True,
    n_freq_bins=None
):
    """
    サンプリング周波数が変動するデータに対し、窓の「秒数」を一定に保ち、
    Numpyのみを使用して共通の周波数軸へ補間・整形するSTFT実装。
    """
    n_data = len(times)
    
    # 処理結果の一時保管用
    raw_results = {
        'times': [],
        'freqs': [],
        'psd_x': [],
        'psd_y': [],
        'psd_z': []
    }
    
    curr_idx = 0
    while True:
        t_start = times[curr_idx]
        t_end_target = t_start + window_second
        
        if t_end_target > times[-1]:
            break
            
        end_idx = np.searchsorted(times, t_end_target, side='left')
        win_size_local = end_idx - curr_idx
        
        # 窓内のデータが少なすぎる場合はスキップ
        if win_size_local < 4:
            curr_idx += 1
            if curr_idx >= n_data: break
            continue

        t_seg = times[curr_idx : end_idx]
        d_seg = data_vec[curr_idx : end_idx, :]
        dt_local = np.mean(np.diff(t_seg))
        time_total_local = t_seg[-1] - t_seg[0]

        # 窓関数の作成
        if window == "hanning":
            w = np.hanning(win_size_local)
        else:
            w = np.ones(win_size_local)
        
        acf = 1.0 / (np.sum(w) / win_size_local)
        nyquist_local = 0.5 / dt_local

        if positive_freq:
            num_bins = win_size_local // 2
            f_local = np.linspace(0, nyquist_local, num_bins)
        else:
            num_bins = win_size_local
            f_local = fft.fftshift(fft.fftfreq(win_size_local, dt_local))

        # 各軸のPSD計算
        current_psds = []
        for axis in range(3):
            # FFT実行
            fft_raw = fft.fft(d_seg[:, axis] * w) / win_size_local
            # 正の周波数のみ抽出、またはシフト
            fft_val = fft_raw[:num_bins] * 2.0 if positive_freq else fft.fftshift(fft_raw)
            fft_val *= acf
            # PSD算出
            psd = (np.abs(fft_val)**2) * time_total_local
            current_psds.append(psd)

        # データの蓄積
        raw_results['times'].append(t_start + (time_total_local / 2.0))
        raw_results['freqs'].append(f_local)
        raw_results['psd_x'].append(current_psds[0])
        raw_results['psd_y'].append(current_psds[1])
        raw_results['psd_z'].append(current_psds[2])

        # 次のステップ計算
        step_second = window_second * (1.0 - rate_overlap)
        next_t_start = t_start + step_second
        curr_idx = np.searchsorted(times, next_t_start, side='left')
        if curr_idx >= n_data - 1: break

    if not raw_results['times']:
        return None

    # --- 共通周波数軸への補間処理 (Numpyのみ) ---
    
    # 全セグメントの最大周波数と最大ビン数を特定
    max_f = max([f[-1] for f in raw_results['freqs']])
    if n_freq_bins is None:
        n_freq_bins = max([len(f) for f in raw_results['freqs']])
    
    # 共通の周波数軸を作成
    common_freqs = np.linspace(0, max_f, n_freq_bins)
    
    # 補間用関数: numpy.interp を使用
    def interpolate_spectrum(f_src, s_src):
        # np.interp(補間先x, 元のx, 元のy)
        # 範囲外は left/right 引数で 0.0 を指定
        return np.interp(common_freqs, f_src, s_src, left=0.0, right=0.0)

    # 全時間ステップを行列化
    final_psd_x = np.array([interpolate_spectrum(f, s) for f, s in zip(raw_results['freqs'], raw_results['psd_x'])])
    final_psd_y = np.array([interpolate_spectrum(f, s) for f, s in zip(raw_results['freqs'], raw_results['psd_y'])])
    final_psd_z = np.array([interpolate_spectrum(f, s) for f, s in zip(raw_results['freqs'], raw_results['psd_z'])])

    return {
        'times': np.array(raw_results['times'], dtype=np.float64),
        'freqs': common_freqs, 
        'spectrogram_x': final_psd_x, # 2D array (times, n_freq_bins)
        'spectrogram_y': final_psd_y,
        'spectrogram_z': final_psd_z,
        'spectrogram_psd_x': final_psd_x, # 互換性のための重複
        'spectrogram_psd_y': final_psd_y,
        'spectrogram_psd_z': final_psd_z
    }


def _stft_vec_variable(
    times,
    data_vec,
    window_second=2.0,
    rate_overlap=0.1,
    window="hanning",
    positive_freq=True
):
    """
    サンプリング周波数が変動する3軸ベクトルデータ用のSTFT実装。
    窓長を秒単位で指定し、平均サンプリングレートから最適な窓サイズ(n点)を自動計算します。
    
    Args:
        times (1D array): 各データ点のタイムスタンプ (秒)
        data_vec (2D array): (n_data, 3) の形状を持つベクトルデータ [X, Y, Z]
        window_second (float): 窓長 (秒)
        rate_overlap (float): 重なり率 (0.0 以上 1.0 未満)
        window (str): 窓関数の種類 ('hanning' または 'rect')
        positive_freq (bool): Trueの場合、正の周波数のみを返す
        
    Returns:
        dict: 以下の要素を含む辞書
            - 'times': 各セグメントの中央時刻 (1D array)
            - 'freqs': 各セグメントに対応する周波数軸 (2D array)
            - 'spectrogram_x/y/z': 複素スペクトログラム (2D array)
            - 'spectrogram_psd_x/y/z': パワースペクトル密度 (2D array)
    """
    if rate_overlap < 0 or rate_overlap >= 1:
        raise ValueError("rate_overlap は [0, 1) の範囲である必要があります。")

    n_data = len(times)
    
    # 1. 全体の平均サンプリング間隔(dt)を計算して window_size を決定
    # 変動がある前提だが、FFTの点数は一定にする必要があるため平均値を使用
    avg_dt = (times[-1] - times[0]) / (n_data - 1)
    window_size = int(window_second / avg_dt)
    
    # 窓サイズが小さすぎたり大きすぎたりする場合のチェック
    if window_size < 4:
        window_size = 4
    if window_size > n_data:
        window_size = n_data

    n_overlap = int(window_size * rate_overlap)
    step = window_size - n_overlap

    # 結果格納用のリスト
    time_midpoints = []
    freqs_2d = [] 
    
    spec_x, spec_y, spec_z = [], [], []
    psd_x, psd_y, psd_z = [], [], []

    # 窓関数の設定
    if window == "hanning":
        w = np.hanning(window_size)
    else:
        w = np.ones(window_size)
    
    # 窓関数による振幅減衰の補正係数 (ACF)
    acf = 1.0 / (np.sum(w) / window_size)

    # スライディングウィンドウ処理
    for i in range(0, n_data - window_size, step):
        t_seg = times[i : i + window_size]
        d_seg = data_vec[i : i + window_size, :]

        # このセグメント内での局所的なサンプリング間隔を計算
        dt_local = np.mean(np.diff(t_seg))
        if dt_local <= 0:
            continue
            
        nyquist_local = 0.5 / dt_local
        time_total_local = t_seg[-1] - t_seg[0]

        # 周波数軸の生成 (セグメントごとのdt_localを反映)
        if positive_freq:
            f_local = np.linspace(0, nyquist_local, window_size // 2)
            num_bins = window_size // 2
        else:
            f_local = fft.fftshift(fft.fftfreq(window_size, dt_local))
            num_bins = window_size

        current_specs = []
        current_psds = []
        
        for axis in range(3):
            # FFT実行
            fft_raw = fft.fft(d_seg[:, axis] * w) / window_size
            
            if positive_freq:
                fft_val = fft_raw[:num_bins] * 2.0
            else:
                fft_val = fft.fftshift(fft_raw)
            
            # 振幅補正
            fft_val *= acf
            
            # PSDの計算
            amp = np.abs(fft_val)
            psd = (amp**2) * time_total_local
            
            current_specs.append(fft_val)
            current_psds.append(psd)

        # データの蓄積
        time_midpoints.append(times[i + window_size // 2])
        freqs_2d.append(f_local)
        
        spec_x.append(current_specs[0]); spec_y.append(current_specs[1]); spec_z.append(current_specs[2])
        psd_x.append(current_psds[0]); psd_y.append(current_psds[1]); psd_z.append(current_psds[2])

    return {
        'times': np.array(time_midpoints),
        'freqs': np.array(freqs_2d),
        'spectrogram_x': np.array(spec_x),
        'spectrogram_y': np.array(spec_y),
        'spectrogram_z': np.array(spec_z),
        'spectrogram_psd_x': np.array(psd_x),
        'spectrogram_psd_y': np.array(psd_y),
        'spectrogram_psd_z': np.array(psd_z),
        'calculated_window_size': window_size  # デバッグ用に計算された窓サイズも返す
    }


def _stft_vec_variable(
        times,
        data_vec,
        window_second=1.0,
        rate_overlap=0.1,
        window="hanning",
        num_freq_bins=None,
        force_pow2=True
):
    """
    安定版：可変サンプリングSTFT
    - 非単調/重複時刻を排除
    - dt安全チェック強化
    - force_pow2はゼロパディングで対応（時間を壊さない）
    """

    if rate_overlap < 0 or rate_overlap >= 1:
        raise ValueError("rate_overlap must be in [0, 1)")

    times = np.asarray(times)
    data_vec = np.asarray(data_vec)

    # --- 0. 前処理：時刻の健全化 ---
    mask_valid = np.isfinite(times)
    times = times[mask_valid]
    data_vec = data_vec[mask_valid]

    # 単調増加に強制（重複排除）
    uniq_idx = np.unique(times, return_index=True)[1]
    times = times[uniq_idx]
    data_vec = data_vec[uniq_idx]

    if len(times) < 10:
        return None

    total_len = len(times)
    start_time = times[0]
    end_time = times[-1]

    current_time = start_time
    step_second = window_second * (1.0 - rate_overlap)

    all_segments = []
    max_nyquist = 0.0

    # --- 1. 実時間ベーススライド ---
    while current_time + window_second <= end_time:

        idx_start = np.searchsorted(times, current_time)
        idx_end = np.searchsorted(times, current_time + window_second)

        if idx_end - idx_start < 8:
            current_time += step_second
            continue

        t_seg = times[idx_start:idx_end]
        d_seg = data_vec[idx_start:idx_end]

        # --- 2. dtチェック ---
        dt_arr = np.diff(t_seg)

        if len(dt_arr) == 0:
            current_time += step_second
            continue

        if np.any(dt_arr <= 0) or np.any(~np.isfinite(dt_arr)):
            current_time += step_second
            continue

        dt_seg = np.mean(dt_arr)
        if dt_seg <= 0:
            current_time += step_second
            continue

        ny_seg = 0.5 / dt_seg
        max_nyquist = max(max_nyquist, ny_seg)

        # --- 3. FFTサイズ調整（安全版：ゼロパディング） ---
        N = len(t_seg)

        if force_pow2:
            target_N = int(2 ** np.ceil(np.log2(N)))
            pad_len = target_N - N

            if pad_len > 0:
                t_pad = t_seg[-1] + dt_seg * np.arange(1, pad_len + 1)
                d_pad = np.zeros((pad_len, 3))

                t_seg = np.concatenate([t_seg, t_pad])
                d_seg = np.vstack([d_seg, d_pad])

        # --- 4. FFT ---
        seg_results = []
        valid_segment = True

        for j in range(3):
            try:
                res = _fft(t_seg, d_seg[:, j], window=window, positive_freq=True)

                # NaNチェック
                if (res is None or
                    np.any(~np.isfinite(res['spec'])) or
                    np.any(~np.isfinite(res['psd']))):
                    valid_segment = False
                    break

                seg_results.append(res)

            except Exception:
                valid_segment = False
                break

        if not valid_segment:
            current_time += step_second
            continue

        mid_time = current_time + window_second / 2.0

        all_segments.append({
            'time': mid_time,
            'results': seg_results,
            'ws': len(t_seg)
        })

        display.debug(f'window size={len(t_seg)}')

        current_time += step_second

    if not all_segments:
        return None

    # --- 5. 共通周波数グリッド ---
    if num_freq_bins is None:
        max_ws = max(seg['ws'] for seg in all_segments)
        num_freq_bins = max_ws // 2

    common_freqs = np.linspace(0, max_nyquist, num_freq_bins)

    n_times = len(all_segments)

    output = {
        'times': np.zeros(n_times),
        'freqs': common_freqs
    }

    for axis in ['x', 'y', 'z']:
        output[f'spectrogram_{axis}'] = np.zeros((n_times, num_freq_bins), dtype=complex)
        output[f'spectrogram_psd_{axis}'] = np.zeros((n_times, num_freq_bins))

    # --- 6. 補間（NaN耐性あり） ---
    for i, seg in enumerate(all_segments):
        output['times'][i] = seg['time']

        for j, axis in enumerate(['x', 'y', 'z']):
            r = seg['results'][j]

            freq = r['freq']
            spec = r['spec']
            psd = r['psd']

            # NaN除去
            valid = np.isfinite(freq) & np.isfinite(spec.real) & np.isfinite(psd)

            if np.sum(valid) < 2:
                continue

            freq = freq[valid]
            spec = spec[valid]
            psd = psd[valid]

            # 単調保証
            sort_idx = np.argsort(freq)
            freq = freq[sort_idx]
            spec = spec[sort_idx]
            psd = psd[sort_idx]

            re = np.interp(common_freqs, freq, spec.real, left=0, right=0)
            im = np.interp(common_freqs, freq, spec.imag, left=0, right=0)

            output[f'spectrogram_{axis}'][i, :] = re + 1j * im
            output[f'spectrogram_psd_{axis}'][i, :] = np.interp(common_freqs, freq, psd, left=0, right=0)

    return output

def _stft_vec_variable(
        times,
        data_vec,
        window_second=1.0,
        rate_overlap=0.1,
        window="hanning",
        num_freq_bins=None,
        force_pow2=True
):
    """
    可変サンプリングレートに対応したベクトルSTFT。
    【重要】ウィンドウの進め方を「インデックスベース」から「実時間ベース」に変更し、時刻の逆転を完全に防止。
    """
    if rate_overlap < 0 or rate_overlap >= 1:
        raise ValueError("rate_overlap must be in [0, 1)")

    total_len = len(times)
    if total_len < 2:
        return None

    all_segments = []
    max_nyquist = 0.0
    
    start_time = times[0]
    end_time = times[-1]
    
    current_time = start_time
    step_second = window_second * (1.0 - rate_overlap)

    # 1. 実時間ベースでスライディング（時刻の逆転は絶対に起きない）
    while current_time + window_second <= end_time:
        
        # 物理的な「時間範囲」でデータを抽出
        idx_start = np.searchsorted(times, current_time)
        idx_end = np.searchsorted(times, current_time + window_second)
        
        N = idx_end - idx_start
        
        # データ点が少なすぎる場合はスキップして時間を進める
        if N < 8:
            current_time += step_second
            continue
            
        # 2. force_pow2: FFTサイズを2のべき乗に揃えるために後方のデータを追加取得
        if force_pow2:
            target_N = int(2 ** np.ceil(np.log2(N)))
            idx_end = min(idx_start + target_N, total_len)
            N = idx_end - idx_start
            if N < 8:
                current_time += step_second
                continue

        t_seg = times[idx_start:idx_end]
        d_seg = data_vec[idx_start:idx_end]
        
        # 3. セグメントのdtの安全確保 (NaN防止)
        dt_seg = (t_seg[-1] - t_seg[0]) / (N - 1)
        if dt_seg <= 1e-12:
            current_time += step_second
            continue
            
        ny_seg = 0.5 / dt_seg
        max_nyquist = max(max_nyquist, ny_seg)
        
        # 4. 中央時刻の設定（逆転しない等間隔な時間軸が保証される）
        mid_time = current_time + window_second / 2.0
        
        # 各軸のFFTを実行
        seg_results = []
        for j in range(3):
            res_dict = _fft(t_seg, d_seg[:, j], window=window, positive_freq=True)
            seg_results.append(res_dict)
            
        all_segments.append({
            'time': mid_time, 
            'results': seg_results, 
            'ws': N
        })
        
        # 確実に時間を定数ステップで進める
        current_time += step_second

    if not all_segments:
        return None

    # 5. 共通グリッドの作成と補間
    if num_freq_bins is None:
        max_ws = max(seg['ws'] for seg in all_segments)
        num_freq_bins = max_ws // 2 + 1

    common_freqs = np.linspace(0, max_nyquist, num_freq_bins)
    n_times = len(all_segments)
    
    res_keys = ['spectrogram_x', 'spectrogram_y', 'spectrogram_z', 
                'spectrogram_psd_x', 'spectrogram_psd_y', 'spectrogram_psd_z']
    
    output = {k: np.zeros((n_times, num_freq_bins)) for k in res_keys}
    for k in ['spectrogram_x', 'spectrogram_y', 'spectrogram_z']:
        output[k] = output[k].astype(complex)
        
    output['times'] = np.zeros(n_times)
    output['freqs'] = common_freqs

    for i, seg in enumerate(all_segments):
        output['times'][i] = seg['time']
        for j, axis in enumerate(['x', 'y', 'z']):
            r = seg['results'][j]
            # 実部と虚部を個別に補間
            re = np.interp(common_freqs, r['freq'], r['spec'].real, left=0, right=0)
            im = np.interp(common_freqs, r['freq'], r['spec'].imag, left=0, right=0)
            output[f'spectrogram_{axis}'][i, :] = re + 1j * im
            
            output[f'spectrogram_psd_{axis}'][i, :] = np.interp(common_freqs, r['freq'], r['psd'], left=0, right=0)

    return output
