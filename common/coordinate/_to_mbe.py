import numpy as np
from common import mathpy, display


# def make_rotation_matrix_mbe(mag_ave_data):
#     """
#     MSO座標系からMBE (Mercury Magnetic Electric) 座標系への回転行列を作成します。
    
#     MBEの定義 (修正版):
#     - y_mbe (yhat): 背景磁場方向 (B)
#     - x_mbe (xhat): MSO-X を磁場に垂直な平面に投影し、正規化した方向 (太陽方向)
#     - z_mbe (zhat): x_mbe × y_mbe (右手法)
    
#     この定義により、磁場が太陽方向(X)と完全に平行でない限り、
#     x_mbe は必ず MSO-X と正の相関（内積 > 0）を持ちます。
#     """
    
#     if mag_ave_data.ndim != 2 or mag_ave_data.shape[1] != 3:
#         return None

#     # 1. y-hat: 磁場方向 (Background Magnetic Field)
#     b_norm = np.linalg.norm(mag_ave_data, axis=1, keepdims=True)
#     b_norm[b_norm == 0] = 1e-10
#     yhat = mag_ave_data / b_norm # (n, 3)

#     # 2. x-hat: MSO-X (1,0,0) を yhat に垂直な平面に投影する
#     # Formula: x_proj = e_x - (e_x . yhat) * yhat
#     e_x = np.array([1, 0, 0])
    
#     # e_x と yhat の内積 (n, 1)
#     dot_x_y = np.sum(e_x * yhat, axis=1, keepdims=True)
    
#     # 投影ベクトル
#     xhat = e_x - dot_x_y * yhat
    
#     x_norm = np.linalg.norm(xhat, axis=1, keepdims=True)
    
#     # 特異点処理: 磁場が完全に太陽方向 (±X) を向いている場合
#     # この場合、e_x を投影すると長さが 0 になるため、代わりに e_y を投影する
#     mask_singularity = (x_norm.flatten() < 1e-6)
#     if np.any(mask_singularity):
#         e_temp = np.array([0, 1, 0])
#         dot_temp_y = np.sum(e_temp * yhat[mask_singularity], axis=1, keepdims=True)
#         xhat[mask_singularity] = e_temp - dot_temp_y * yhat[mask_singularity]
#         x_norm[mask_singularity] = np.linalg.norm(xhat[mask_singularity], axis=1, keepdims=True)

#     xhat /= (x_norm + 1e-10)

#     # 3. z-hat: x と y の外積 (右手系)
#     zhat = np.cross(xhat, yhat)

#     # 4. 回転行列の構成
#     # V_mbe = R * V_mso
#     # 行列の各行に新基底ベクトルの MSO 成分を並べる
#     # R = [xhat_mso]
#     #     [yhat_mso]
#     #     [zhat_mso]
#     rot_mtx = np.stack([xhat, yhat, zhat], axis=1)

#     return rot_mtx

def make_rotation_matrix_mbe(mag_ave_data):
    """
    MSO座標系からMBE (Mercury Magnetic Electric) 座標系への回転行列を作成します。
    
    MBEの定義:
    - y_mbe (yhat): 背景磁場方向 (mag_ave_data)
    - z_mbe (zhat): x_mbe × y_mbe (右手法)
    - x_mbe (xhat): 概ね太陽方向 (MSO-X方向を投影)
    
    手順:
    1. y = B / |B|
    2. z = Unit(X_mso × y)  <-- これにより z は X-Y 平面に垂直になる
    3. x = y × z            <-- 右手系を維持しつつ X_mso に最も近い方向
    """
    
    # 1. 入力チェック
    if mag_ave_data.ndim != 2 or mag_ave_data.shape[1] != 3:
        return None

    # 2. y-hat: 磁場方向 (Background Magnetic Field)
    b_norm = np.linalg.norm(mag_ave_data, axis=1, keepdims=True)
    b_norm[b_norm == 0] = 1e-10
    yhat = mag_ave_data / b_norm

    # 3. z-hat: MSO-X と 磁場の外積
    # 太陽方向 e_x = [1, 0, 0]
    e_x = np.array([1, 0, 0])
    zhat = np.cross(e_x, yhat) # (n, 3)
    
    z_norm = np.linalg.norm(zhat, axis=1, keepdims=True)
    
    # 特異点処理: 磁場が太陽方向に平行な場合、MSO-Yを仮のXとして扱う
    mask_singularity = (z_norm.flatten() < 1e-6)
    if np.any(mask_singularity):
        e_temp = np.array([0, 1, 0])
        zhat[mask_singularity] = np.cross(e_temp, yhat[mask_singularity])
        z_norm[mask_singularity] = np.linalg.norm(zhat[mask_singularity], axis=1, keepdims=True)

    zhat /= (z_norm + 1e-10)

    # 4. x-hat: y と z の外積 (右手系: z = x * y => x = y * z)
    # これにより xhat は yhat と zhat に直交し、概ね e_x 方向を向く
    xhat = np.cross(yhat, zhat)

    # 5. 回転行列の構成
    # V_mbe = R * V_mso
    # 行列の各行に新基底ベクトルの MSO 成分を並べる
    rot_mtx = np.stack([xhat, yhat, zhat], axis=1)

    return rot_mtx


def convert_to_mbe(
        transtimes,
        transvalues,
        times_mag,
        mag_data,
        window_sec,
):
    # averaged mag data
    mag_ave = mathpy.moving_average_by_time(times_mag, mag_data, window_sec)

    # transvaluesデータを磁場の時刻に補間
    transvalues_interp = mathpy.interp_vec(times_mag, transtimes, transvalues)

    # 3. 回転行列の作成
    rot_mtx = make_rotation_matrix_mbe(mag_ave)

    # 4. 座標変換の実行 (現状のループ処理を踏襲)
    values_mbe = np.zeros_like(mag_ave)
    for t in range(len(times_mag)):
        values_mbe[t, :] = rot_mtx[t, :, :] @ transvalues_interp[t, :]

    return {
        'values_mbe': values_mbe,
        'mag_ave': mag_ave,
        'rot_mtx': rot_mtx
    }
