import numpy as np

def rmlatmlt_to_polar(radius, mlat, mlt):  # R MLAT MLT -> r, theta, phi
    theta = np.pi / 2 - mlat * np.pi / 180
    phi = mlt * np.pi / 12
    return radius, theta, phi

def polar_to_rmlatmlt(radius, theta, phi): # r, theta, phi -> R MLAT MLT
    mlat = (np.pi / 2 - theta) * 180 / np.pi
    mlt = np.mod(phi * 12 / np.pi + 12, 24)
    return radius, mlat, mlt

def polar_to_xyz(radius, theta, phi):  # polar -> xyz
    comp_x = radius * np.sin(theta) * np.cos(phi)
    comp_y = radius * np.sin(theta) * np.sin(phi)
    comp_z = radius * np.cos(theta)
    return comp_x, comp_y, comp_z

def xyz_to_polar(comp_x: np.ndarray, comp_y: np.ndarray, comp_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    直交座標 (x, y, z) を球面座標 (radius, theta, phi) に変換します。

    Parameters
    ----------
    comp_x : np.ndarray
        x成分の一次元配列またはスカラ。
    comp_y : np.ndarray
        y成分の一次元配列またはスカラ。
    comp_z : np.ndarray
        z成分の一次元配列またはスカラ。

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        球面座標のタプル: (radius, theta, phi)。
        - radius (半径): 原点からの距離。
        - theta (天頂角): z軸からの角度 [0, pi]。
        - phi (方位角): x軸からの角度 [-pi, pi]。
    """
    # 1. 半径 (r) の計算
    radius = np.sqrt(comp_x**2 + comp_y**2 + comp_z**2)
    
    # 2. 天頂角 (theta) の計算
    # r=0の点をnp.arccos(z/r)で計算すると警告が出るので、
    # ゼロ除算を避けるためにmasked_arrayを使うか、np.whereで分岐する
    # ここではnp.whereを使用
    theta = np.where(
        radius != 0,
        np.arccos(comp_z / radius),
        0.0 # radiusが0の場合はthetaも0と定義
    )
    
    # 3. 方位角 (phi) の計算
    # np.arctan2(y, x) を使用することで、xとyの符号を考慮して正しい角度を計算
    # np.arctan2 は phi が [-pi, pi] の範囲になる
    phi = np.arctan2(comp_y, comp_x)
    
    return radius, theta, phi

