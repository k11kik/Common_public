import numpy as np
from common import display, mathpy


def make_rotation_matrix(
        mag_data,  # (n, 3)
        orb_data,  # (n, 3)
):
    # check
    if mag_data.ndim != 2 or mag_data.shape[1] != 3:
        display.warning(f'Invalid data shape: {mag_data.shape=}')
        return None
    if orb_data.ndim != 2 or orb_data.shape[1] != 3:
        display.warning(f'Invalid data shape: {orb_data.shape=}')
        return None
    if len(mag_data) != len(orb_data):
        display.warning(f'mag_data and orb_data must be the same length: {len(mag_data)=}, {len(orb_data)=}')
        return None
    
    # main
    b_norm = np.linalg.norm(mag_data, axis=1, keepdims=True) + 1e-10
    zhat = mag_data / b_norm

    yhat = np.cross(zhat, orb_data)
    yhat /= np.linalg.norm(yhat, axis=1, keepdims=True) + 1e-10
    xhat = np.cross(yhat, zhat)

    # 回転行列の作成
    rot_mtx = np.stack([xhat, yhat, zhat], axis=1)  # shape: (num_times, 3, 3)
    # rot_mtx = np.transpose(rot_mtx, axes=(0, 2, 1))

    return rot_mtx


def convert_to_mfa_const(
        times_mag,
        mag_data,
        times_orb,
        orb_data,
        window_size_mfa=256,
):
    """
    Convert into MFA coordinate
    :param epoch:
    :param mag_x:
    :param mag_y:
    :param mag_z:
    :param window_size_mfa:
    :param moving_mode:
    :param kwargs: "type_xaxis", "return_indices", "return_rot_mtx", "df_rmlatmlt"
    :return: dictionary: "epoch", "Bx", "By", "Bz", ("indices", "rotation matrix")
    """
    # mean B-field
    # mag_ave_x = uniform_filter1d(mag_data[:, 0], size=window_size_mfa, mode='nearest')
    # mag_ave_y = uniform_filter1d(mag_data[:, 1], size=window_size_mfa, mode='nearest')
    # mag_ave_z = uniform_filter1d(mag_data[:, 2], size=window_size_mfa, mode='nearest')
    # mag_ave = np.stack([mag_ave_x, mag_ave_y, mag_ave_z], axis=1)
    
    mag_ave = mathpy.moving_average_vec(mag_data, window_size_mfa)

    orb_interp = mathpy.interp_vec(times_mag, times_orb, orb_data)
    rot_mtx = make_rotation_matrix(mag_ave, orb_interp)

    mag_mfa = np.zeros_like(mag_ave)
    for t in range(len(times_mag)):
        mag_mfa[t, :] = rot_mtx[t, :, :] @ mag_data[t, :]

    return {
        'mag_mfa': mag_mfa,
        'mag_ave': mag_ave,
        'rot_mtx': rot_mtx
    }


def convert_to_mfa_dynamic(
        times_mag,
        mag_data,
        times_orb,
        orb_data,
        window_mfa_sec=30.0,
):
    """
    Convert into MFA coordinate with a time-based window
    :param times_mag: Time array for magnetic field data
    :param mag_data: Magnetic field data (N, 3)
    :param times_orb: Time array for orbit data
    :param orb_data: Orbit data (M, 3)
    :param window_mfa_sec: Window width in seconds for moving average
    :return: dictionary: "mag_mfa", "mag_ave", "rot_mtx"
    """
    


    # 1. 平均磁場 (Background Field) の算出 (window_size_mfaの代わりに動的計算を使用)
    mag_ave = mathpy.moving_average_by_time(times_mag, mag_data, window_mfa_sec)

    # 2. 軌道データを磁場の時刻に補間
    orb_interp = mathpy.interp_vec(times_mag, times_orb, orb_data)

    # 3. 回転行列の作成
    rot_mtx = make_rotation_matrix(mag_ave, orb_interp)

    # 4. 座標変換の実行 (現状のループ処理を踏襲)
    mag_mfa = np.zeros_like(mag_ave)
    for t in range(len(times_mag)):
        mag_mfa[t, :] = rot_mtx[t, :, :] @ mag_data[t, :]

    return {
        'mag_mfa': mag_mfa,
        'mag_ave': mag_ave,
        'rot_mtx': rot_mtx
    }


def convert_to_mfa(
        times_mag,
        mag_data,
        times_orb,
        orb_data,
        mode='dynamic',
        window_mfa_sec=None,
        window_size_mfa=None,
):
    """
    Params
    -------
    * mode: 
        * 'const': window_size_mfa must be given
        * 'dynamic': window_mfa_sec must be given
    """
    if mode == 'const':
        if window_size_mfa is None:
            display.warning('window_size_mfa must be given -> Default: 256')
            window_size_mfa = 256
        return convert_to_mfa_const(
            times_mag,
            mag_data,
            times_orb,
            orb_data,
            window_size_mfa=window_size_mfa
        )
    
    elif mode == 'dynamic':
        if window_mfa_sec is None:
            display.warning('window_mfa_sec must be given -> Default: 30')
            window_mfa_sec = 30
        return convert_to_mfa_dynamic(
            times_mag,
            mag_data,
            times_orb,
            orb_data,
            window_mfa_sec=window_mfa_sec
        )
    
    else:
        raise ValueError(f'Invalid mode: {mode}')


def convert_to_mfa_fluct_const(
        times_mag,
        mag_data,
        times_mag_ambient,
        mag_ambient_ave,
        times_orb,
        orb_data,
        window_size_mfa=256,
):
    """
    For fluctuating magnetic field
    :param epoch:
    :param mag_x:
    :param mag_y:
    :param mag_z:
    :param window_size_mfa:
    :param moving_mode:
    :param kwargs: "type_xaxis", "return_indices", "return_rot_mtx", "df_rmlatmlt"
    :return: dictionary: "epoch", "Bx", "By", "Bz", ("indices", "rotation matrix")
    """
    # check
    if len(mag_data) != len(mag_ambient_ave):
        mag_ambient_interp = mathpy.interp_vec(times_mag, times_mag_ambient, mag_ambient_ave)
    else:
        mag_ambient_interp = mag_ambient_ave
    
    if len(mag_data) != len(orb_data):
        orb_interp = mathpy.interp_vec(times_mag, times_orb, orb_data)
    else:
        orb_interp = orb_data
    
    # averaged ambient mag
    # -------------------
    # mag_ambient_interp = mathpy.moving_average_vec(mag_ambient_interp, window_size_mfa)
    mag_ambient_interp = mathpy.fast_moving_average_vec(mag_ambient_interp, window_size_mfa)
    # -------------------

    rot_mtx = make_rotation_matrix(mag_ambient_interp, orb_interp)

    mag_mfa = np.zeros_like(mag_data)
    for t in range(len(times_mag)):
        mag_mfa[t, :] = rot_mtx[t, :, :] @ mag_data[t, :]

    return {
        'mag_mfa': mag_mfa,
        'mag_ave': mag_ambient_interp,
        'rot_mtx': rot_mtx
    }


def convert_to_mfa_fluct_dynamic(
        times_mag,
        mag_data,
        times_mag_ambient,
        mag_ambient_ave,
        times_orb,
        orb_data,
        window_mfa_sec=30.0,
):
    """New time-based window version for fluctuations."""
    # Interpolation
    if len(mag_data) != len(mag_ambient_ave):
        mag_ambient_interp = mathpy.interp_vec(times_mag, times_mag_ambient, mag_ambient_ave)
    else:
        mag_ambient_interp = mag_ambient_ave
    
    if len(mag_data) != len(orb_data):
        orb_interp = mathpy.interp_vec(times_mag, times_orb, orb_data)
    else:
        orb_interp = orb_data
    
    # Averaging (using the new time-based logic)
    # 磁場の時刻(times_mag)に合わせた窓で平均化
    mag_ambient_interp = mathpy.moving_average_by_time(times_mag, mag_ambient_interp, window_mfa_sec)

    # Rotation
    rot_mtx = make_rotation_matrix(mag_ambient_interp, orb_interp)
    mag_mfa = np.zeros_like(mag_data)
    for t in range(len(times_mag)):
        mag_mfa[t, :] = rot_mtx[t, :, :] @ mag_data[t, :]

    return {'mag_mfa': mag_mfa, 'mag_ave': mag_ambient_interp, 'rot_mtx': rot_mtx}


def convert_to_mfa_fluct(
        times_mag,
        mag_data,
        times_mag_ambient,
        mag_ambient_ave,
        times_orb,
        orb_data,
        mode='const',
        window_mfa_sec=None,
        window_size_mfa=None,
):
    if mode == 'const':
        if window_size_mfa is None:
            display.warning('window_size_mfa must be given -> Default: 256')
            window_size_mfa = 256
        return convert_to_mfa_fluct_const(
            times_mag, mag_data, times_mag_ambient, mag_ambient_ave,
            times_orb, orb_data, window_size_mfa=window_size_mfa
        )
    elif mode == 'dynamic':
        if window_mfa_sec is None:
            display.warning('window_mfa_sec must be given -> Default: 30')
            window_mfa_sec = 30.0
        return convert_to_mfa_fluct_dynamic(
            times_mag, mag_data, times_mag_ambient, mag_ambient_ave,
            times_orb, orb_data, window_mfa_sec=window_mfa_sec
        )
    else:
        raise ValueError(f'Invalid mode: {mode}')



# def old_convert_to_mfa_Miyashita(times_mag, mag_data, times_orb, orb_data, window_size=256):
#     """
#     MSO座標系から背景磁場座標系(MFA)に変換
    
#     Parameters:
#     window_size (int): 背景磁場計算用の移動平均ウィンドウサイズ（データポイント数）
#     """
#     orb_interp = mathpy.interp_vec(times_mag, times_orb, orb_data)
    
#     # 背景磁場の計算（移動平均）
#     B_bg_x = uniform_filter1d(mag_data[:, 0], size=window_size, mode='nearest')
#     B_bg_y = uniform_filter1d(mag_data[:, 1], size=window_size, mode='nearest')
#     B_bg_z = uniform_filter1d(mag_data[:, 2], size=window_size, mode='nearest')
    
#     # 背景磁場の大きさ
#     B_bg_magnitude = np.sqrt(B_bg_x**2 + B_bg_y**2 + B_bg_z**2)
    
#     # e1 (背景磁場方向の単位ベクトル) - parallel component
#     e1_x = B_bg_x / B_bg_magnitude
#     e1_y = B_bg_y / B_bg_magnitude
#     e1_z = B_bg_z / B_bg_magnitude
    
#     # 速度ベクトル（位置の時間微分で近似）
#     dt_median = np.median(np.diff(times_mag))
#     # dt = dt.fillna(dt_median)
    
#     # Use median time interval as scalar spacing for gradient calculation
#     # np.gradient expects either scalar spacing or coordinate arrays, not spacing arrays
#     v_x = np.gradient(orb_interp[:, 0], dt_median)
#     v_y = np.gradient(orb_interp[:, 1], dt_median)
#     v_z = np.gradient(orb_interp[:, 2], dt_median)
    
#     # e2 = (B × V) / |B × V| - perpendicular component 1
#     cross_x = B_bg_y * v_z - B_bg_z * v_y
#     cross_y = B_bg_z * v_x - B_bg_x * v_z
#     cross_z = B_bg_x * v_y - B_bg_y * v_x
    
#     cross_magnitude = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
    
#     # ゼロ除算を避ける
#     cross_magnitude = np.where(cross_magnitude == 0, 1e-10, cross_magnitude)
    
#     e2_x = cross_x / cross_magnitude
#     e2_y = cross_y / cross_magnitude
#     e2_z = cross_z / cross_magnitude
    
#     # e3 = e1 × e2 - perpendicular component 2
#     e3_x = e1_y * e2_z - e1_z * e2_y
#     e3_y = e1_z * e2_x - e1_x * e2_z
#     e3_z = e1_x * e2_y - e1_y * e2_x
    
#     # MFA座標系への変換
#     dict_mfa = {}
#     # B_parallel (e1方向)
#     dict_mfa['B_PAR'] = (mag_data[:, 0] * e1_x + 
#                             mag_data[:, 1] * e1_y + 
#                             mag_data[:, 2] * e1_z)
    
#     # B_perp1 (e2方向)
#     dict_mfa['B_PERP1'] = (mag_data[:, 0] * e2_x + 
#                             mag_data[:, 1] * e2_y + 
#                             mag_data[:, 2] * e2_z)
    
#     # B_perp2 (e3方向)
#     dict_mfa['B_PERP2'] = (mag_data[:, 0] * e3_x + 
#                             mag_data[:, 1] * e3_y + 
#                             mag_data[:, 2] * e3_z)
    
#     dict_mfa['mag_mfa'] = np.stack([dict_mfa['B_PAR'], dict_mfa['B_PERP1'], dict_mfa['B_PERP2']], axis=1)
    
#     # 背景磁場成分も保存
#     dict_mfa['B_BG_X'] = B_bg_x
#     dict_mfa['B_BG_Y'] = B_bg_y
#     dict_mfa['B_BG_Z'] = B_bg_z
#     dict_mfa['B_BG_MAG'] = B_bg_magnitude
    
#     print(f"✓ Converted to MFA coordinate system (window_size={window_size})")

#     return dict_mfa
