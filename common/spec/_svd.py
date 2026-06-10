import numpy as np
from common import display


def spectral_matrix_from_spec(times, freqs, spectrogram_x, spectrogram_y, spectrogram_z):
    """

    :param freqs:
    :param times:
    :param spectrogram_x: (n_times, n_freqs)
    :param spectrogram_y:
    :param spectrogram_z:
    :return:
    """
    spectral_matrices = np.zeros((len(times), len(freqs), 3, 3), dtype=complex)

    spectral_matrices[:, :, 0, 0] = spectrogram_x * np.conjugate(spectrogram_x)  # Pxx
    spectral_matrices[:, :, 0, 1] = spectrogram_x * np.conjugate(spectrogram_y)  # Pxy
    spectral_matrices[:, :, 0, 2] = spectrogram_x * np.conjugate(spectrogram_z)  # Pxz
    spectral_matrices[:, :, 1, 0] = spectrogram_y * np.conjugate(spectrogram_x)  # Pyx
    spectral_matrices[:, :, 1, 1] = spectrogram_y * np.conjugate(spectrogram_y)  # Pyy
    spectral_matrices[:, :, 1, 2] = spectrogram_y * np.conjugate(spectrogram_z)  # Pyz
    spectral_matrices[:, :, 2, 0] = spectrogram_z * np.conjugate(spectrogram_x)  # Pzx
    spectral_matrices[:, :, 2, 1] = spectrogram_z * np.conjugate(spectrogram_y)  # Pzy
    spectral_matrices[:, :, 2, 2] = spectrogram_z * np.conjugate(spectrogram_z)  # Pzz

    return spectral_matrices


def moving_average_time(matrix, mov_ave=3):
    """

    :param matrix: (num_times, num_freqs, n, m)
    :param mov_ave:
    :return:
    """
    num_times, num_freqs, _, _ = matrix.shape
    averaged_matrix = np.zeros_like(matrix)

    for t in range(num_times):
        # 移動平均のインデックス範囲を設定して制限
        idx_range = np.clip(np.arange(t - mov_ave // 2, t + mov_ave // 2 + 1), 0, num_times - 1)

        # インデックス範囲に対して平均を計算して代入
        averaged_matrix[t, :, :, :] = np.mean(matrix[idx_range, :, :, :], axis=0)

    return averaged_matrix


def moving_average_freq(matrix, mov_ave=3):
    """

    :param matrix: (num_times, num_freqs, n, m)
    :param mov_ave:
    :return:
    """
    num_times, num_freqs, _, _ = matrix.shape
    averaged_matrix = np.zeros_like(matrix)

    for f in range(num_freqs):
        # 移動平均のインデックス範囲を設定して制限
        idx_range = np.clip(np.arange(f - mov_ave // 2, f + mov_ave // 2 + 1), 0, num_freqs - 1)

        # インデックス範囲に対して平均を計算して代入
        averaged_matrix[:, f, :, :] = np.mean(matrix[:, idx_range, :, :], axis=1)

    return averaged_matrix


def svd_mag(spectral_matrices, get_real_matrix=False):
    """
    spectral matrix から SVD法を用いて特異値を求める.
    :param spectral_matrices: 4D array of shape (num_times, num_freqs, 3, 3)
    :param get_real_matrix: if True, return real_matrix
    :return: U-matrix, singular values (3つ), V-matrix
    """
    num_times, num_freqs, _, _ = spectral_matrices.shape
    real_matrix = np.zeros((num_times, num_freqs, 6, 3))

    # Construct the Real Matrix
    real_matrix[:, :, :3, :] = np.real(spectral_matrices)  # 対角成分と対称部分を処理

    real_matrix[:, :, 4, 0] = np.imag(spectral_matrices[:, :, 0, 1])  # (4, 0) -> Im(S12)
    real_matrix[:, :, 5, 0] = np.imag(spectral_matrices[:, :, 0, 2])  # (5, 0) -> Im(S13)
    real_matrix[:, :, 5, 1] = np.imag(spectral_matrices[:, :, 1, 2])  # (5, 1) -> Im(S23)
    real_matrix[:, :, 3, 1] = -real_matrix[:, :, 4, 0]  # 対称部分の符号反転
    real_matrix[:, :, 3, 2] = -real_matrix[:, :, 5, 0]
    real_matrix[:, :, 4, 2] = -real_matrix[:, :, 5, 1]

    # SVDの結果を保存する配列 (U, S, V)
    u_matrices = np.zeros((num_times, num_freqs, 6, 3))
    s_values = np.zeros((num_times, num_freqs, 3))  # SVDの特異値は3つのみ
    v_matrices = np.zeros((num_times, num_freqs, 3, 3))
    k_vec = np.zeros((num_times, num_freqs, 3))

    # NaNを0に置換
    real_matrix = np.nan_to_num(real_matrix, nan=0.0)

    # 各num_freq, num_timesごとにSVDを適用
    for t in range(num_times):
        for f in range(num_freqs):
            matrix_to_svd = real_matrix[t, f, :, :]  # (6, 3) の行列を取り出す -> array shape: (6, 3)

            # SVDの実行
            u, s, vh = np.linalg.svd(matrix_to_svd, full_matrices=False)

            # 結果を保存
            u_matrices[t, f, :, :] = u  # U行列
            s_values[t, f, :] = s  # 特異値 S
            v_matrices[t, f, :, :] = vh  # V行列

            # k_vecに最小特異値に対応するV行列ベクトルを保存
            min_idx = np.argmin(s)  # 最小特異値のインデックス
            k_vec[t, f, :] = vh[min_idx, :]

    dict_svd_result = {
        "u_matrices": u_matrices,
        "s_values": s_values,
        "v_matrices": v_matrices,
        "k_vec": k_vec
    }

    if get_real_matrix:
        dict_svd_result["real_matrix"] = real_matrix

    return dict_svd_result



def polarization_from_specmtx(
        spectral_matrices,
        quiet=False
):
    """

    :param spectral_matrices: (n_times, n_freqs, n ,m)
    :return:
    """
    num_times, num_freqs, _, _ = spectral_matrices.shape

    if not quiet:
        display.current_time_comment('#', 'svd')
    dict_svd_result = svd_mag(spectral_matrices, get_real_matrix=True)
    u_mtx, s_values, v_mtx, k_vec, real_mtx = dict_svd_result["u_matrices"], dict_svd_result["s_values"], dict_svd_result["v_matrices"], dict_svd_result["k_vec"], dict_svd_result["real_matrix"],

    if not quiet:
        display.current_time_comment('#', 'polarization')
    # 波面法線角度（Wave Normal Angle）の計算
    k_xy = np.sqrt(k_vec[:, :, 0] ** 2 + k_vec[:, :, 1] ** 2)  # xy平面成分の大きさ
    k_z = k_vec[:, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        # wna = np.where(k_z == 0.0, 90.0, np.abs(np.degrees(np.arctan(k_xy / k_z))))
        wna = np.where(k_z == 0.0, 90.0, np.abs(np.arctan(k_xy / k_z)) * 180 / np.pi)

    # 偏波（Polarization）と平面性（Planarity）の計算
    s0_nonzero = s_values[:, :, 0] != 0  # s_valuesの3番目の成分が0でないかチェック
    with np.errstate(divide="ignore", invalid="ignore"):
        polari = np.where(s0_nonzero, s_values[:, :, 1] / s_values[:, :, 0], np.nan)
        planarity = np.where(s0_nonzero, 1.0 - np.sqrt(s_values[:, :, 2] / s_values[:, :, 0]), np.nan)

    # 条件に基づいてpolariの符号を反転
    # polari = np.where(real_mtx[:, :, 4, 0] > 0, -polari, polari) -> vrai?
    polari = np.where(real_mtx[:, :, 4, 0] < 0, -polari, polari)

    return {
        "polari": polari,
        "wna": wna,
        "planarity": planarity
    }


def polarization_from_spec(
        times, freqs, spec_x, spec_y, spec_z,
        mov_ave_time=1,
        mov_ave_freq=3,
        quiet=False
):
    """

    :param times:
    :param freqs:
    :param spec_x: (n_times, n_freqs)
    :param spec_y:
    :param spec_z:
    :param mov_ave_freq:
    :return: dict; times, freqs, polarization, wna, planarity
    """
    spec_mtx = spectral_matrix_from_spec(times, freqs, spec_x, spec_y, spec_z)

    # moving average
    spec_mtx = moving_average_time(spec_mtx, mov_ave=mov_ave_time)
    spec_mtx = moving_average_freq(spec_mtx, mov_ave=mov_ave_freq)
    wave_polarization = polarization_from_specmtx(spec_mtx, quiet=quiet)
    polari, wna, planarity = wave_polarization["polari"], wave_polarization["wna"], wave_polarization["planarity"]

    dict_to_return = {
        'times': times,
        'freqs': freqs,
        'polarization': polari,
        'wna': wna,
        'planarity': planarity
    }
    return dict_to_return


