import numpy as np
from common import pytplot, display
from ._spectrogram import stft_vec, stft_vec_variable
from ._svd import polarization_from_spec


def spectrogram_vec(
        var_mag_vec: str,
        window_second=200,
        window_size: int = None,
        rate_overlap: float = .1,
        nyquist_freq=None,
        get_support=True
):
    dict_support = {
        'time_res [s]': window_second * (1 - rate_overlap),
        'freq_res [Hz]': 1 / window_second
    }

    # get data
    dat_mag_vec = pytplot.get_data(var_mag_vec)
    if dat_mag_vec is None:
        display.error('spec/spectrogram_vec', 'No data')
        return
    times, mag_vec = dat_mag_vec.times, dat_mag_vec.y

    # spectrogram
    if nyquist_freq is None:
        if window_size is not None:
            display.warning('window_size is ignored. If you would like to set window_size, nyquist_freq must be given')
        dict_spectrogram = stft_vec_variable(times, mag_vec, window_second=window_second, rate_overlap=rate_overlap)
        if get_support:
            dict_support['window'] = dict_spectrogram['window']
    else:
        if window_size is None:
            display.warning('window_size must be given -> Default: 256')
            window_size = 256
        dict_spectrogram = stft_vec(times, mag_vec, window_size=window_size, rate_overlap=rate_overlap, nyquist_freq=nyquist_freq)

    pytplot.store_data(f'{var_mag_vec}_x_dpwrspc', {'x': dict_spectrogram['times'], 'y': dict_spectrogram['spectrogram_x'], 'v': dict_spectrogram['freqs']})
    pytplot.store_data(f'{var_mag_vec}_y_dpwrspc', {'x': dict_spectrogram['times'], 'y': dict_spectrogram['spectrogram_y'], 'v': dict_spectrogram['freqs']})
    pytplot.store_data(f'{var_mag_vec}_z_dpwrspc', {'x': dict_spectrogram['times'], 'y': dict_spectrogram['spectrogram_z'], 'v': dict_spectrogram['freqs']})
    pytplot.store_data(f'{var_mag_vec}_x_dpwrspc_psd', {'x': dict_spectrogram['times'], 'y': dict_spectrogram['spectrogram_psd_x'], 'v': dict_spectrogram['freqs']})
    pytplot.store_data(f'{var_mag_vec}_y_dpwrspc_psd', {'x': dict_spectrogram['times'], 'y': dict_spectrogram['spectrogram_psd_y'], 'v': dict_spectrogram['freqs']})
    pytplot.store_data(f'{var_mag_vec}_z_dpwrspc_psd', {'x': dict_spectrogram['times'], 'y': dict_spectrogram['spectrogram_psd_z'], 'v': dict_spectrogram['freqs']})

    # psd abs
    psd_abs = np.sqrt(dict_spectrogram['spectrogram_psd_x'] ** 2 + dict_spectrogram['spectrogram_psd_y'] ** 2 + dict_spectrogram['spectrogram_psd_z'] ** 2)

    pytplot.store_data(f'{var_mag_vec}_psd_abs', {'x': dict_spectrogram['times'], 'y': psd_abs, 'v': dict_spectrogram['freqs']})

    return dict_support


def _normalize_by_fcp(
        var_spec: str,
        var_fcp: str,
        new_name: str = None
):
    dat_spec = pytplot.get_data(var_spec)
    dat_fcp = pytplot.get_data(var_fcp)
    if dat_spec is None or dat_fcp is None:
        display.error('spec/normalize_by_fcp', f'No data: {var_spec} and/or {var_fcp}')
        return None

    times, freqs, specs = dat_spec.times, dat_spec.v, dat_spec.y
    times_fcp, fcp = dat_fcp.times, dat_fcp.y

    
    if np.any(np.isnan(fcp)):
        times_fcp = times_fcp[~np.isnan(fcp)]
        fcp = fcp[~np.isnan(fcp)]

    if len(times) != len(times_fcp):
        fcp = np.interp(times, times_fcp, fcp)
    
    # 正規化周波数軸を定義（全時間ステップに共通な軸）
    f_norm = np.linspace(np.min(freqs / fcp.max()), np.max(freqs / fcp.min()), len(freqs))

    # 空の補間後スペクトル
    spec_interp = np.full((len(times), len(freqs)), np.nan)

    for i in range(len(times)):
        f_scaled = freqs / fcp[i]  # 時刻iでの f/fcp
        valid = (f_norm >= np.min(f_scaled)) & (f_norm <= np.max(f_scaled))
        spec_interp[i, valid] = np.interp(f_norm[valid], f_scaled, specs[i, :])

    # pytplot形式で保存
    if new_name is None:
        new_name = f'{var_spec}_norm'
    pytplot.store_data(new_name, {'x': times, 'y': spec_interp, 'v': f_norm})
    return


def normalize_by_fcp(
    var_spec: str,
    var_fcp: str,
    new_name: str = None
):
    """
    周波数軸を (n_times, n_freqs) の2次元配列として保持することで、
    解像度の低下を防ぎ、かつメモリ効率（補間による巨大化の回避）を両立させる。
    """
    dat_spec = pytplot.get_data(var_spec)
    dat_fcp = pytplot.get_data(var_fcp)
    
    if dat_spec is None or dat_fcp is None:
        print(f'Error: No data for {var_spec} and/or {var_fcp}')
        return None

    times = dat_spec.times
    freqs_orig = dat_spec.v # (n_freqs,)
    specs = dat_spec.y      # (n_times, n_freqs)
    
    fcp_raw = dat_fcp.y
    t_fcp = dat_fcp.times

    # fcpのNaN除去と補間
    mask = ~np.isnan(fcp_raw)
    fcp_interp = np.interp(times, t_fcp[mask], fcp_raw[mask])

    # --- 2次元周波数軸の生成 ---
    # 各時刻 i における正規化周波数 rho[i, j] = f[j] / fcp[i]
    # np.outer を使うと (n_times, n_freqs) が一気に作れる
    # ただし、specs と同じ shape になるように調整
    v_2d = freqs_orig[np.newaxis, :] / fcp_interp[:, np.newaxis]

    if new_name is None:
        new_name = f'{var_spec}_norm'
    
    # store_data に v として 2D 配列を渡す
    pytplot.store_data(new_name, {
        'x': times, 
        'y': specs,  # データそのものは補間せずそのまま
        'v': v_2d    # ここを 2D にする
    })
    
    return new_name


def polarization_from_spectrogram(
        var_spec_x: str,
        var_spec_y: str,
        var_spec_z: str,
        mov_ave_time=1,
        mov_ave_freq=3,
        quiet = False,
        varname_out_polari='polarization',
        varname_out_wna='wna',
        varname_out_planarity='planarity',
):
    dict_support = {
        'mov_ave_time': mov_ave_time,
        'mov_ave_freq': mov_ave_freq,
    }
    # get data
    dat_var_spec_x = pytplot.get_data(var_spec_x)
    dat_var_spec_y = pytplot.get_data(var_spec_y)
    dat_var_spec_z = pytplot.get_data(var_spec_z)

    # polarization analysis
    dict_polari = polarization_from_spec(
        dat_var_spec_x.times,
        dat_var_spec_x.v,
        dat_var_spec_x.y,
        dat_var_spec_y.y,
        dat_var_spec_z.y,
        mov_ave_time=mov_ave_time,
        mov_ave_freq=mov_ave_freq,
        quiet=quiet
    )
    pytplot.store_data(varname_out_polari, {'x': dict_polari['times'], 'y': dict_polari['polarization'], 'v': dict_polari['freqs']})
    pytplot.store_data(varname_out_wna, {'x': dict_polari['times'], 'y': dict_polari['wna'], 'v': dict_polari['freqs']})
    pytplot.store_data(varname_out_planarity, {'x': dict_polari['times'], 'y': dict_polari['planarity'], 'v': dict_polari['freqs']})
    return dict_support
