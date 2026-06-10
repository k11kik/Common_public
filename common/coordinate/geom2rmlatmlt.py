import numpy as np
from common import time

# --- 北双極子極の定義 (地理座標, IGRF 2020 近似) ---
# Pole location: 80.65 N, 72.62 W (西経は負)
PLat = 80.65  # 極の緯度 (度)
PLon = -72.62 # 極の経度 (度)
PLat_r = np.deg2rad(PLat)
PLon_r = np.deg2rad(PLon)

# 極ベクトル P (地理直交座標系)
Px = np.cos(PLat_r) * np.cos(PLon_r)
Py = np.cos(PLat_r) * np.sin(PLon_r)
Pz = np.sin(PLat_r)

# 地球の平均半径 (km)
RE = 6371.2 


def _unix_to_ut_hour(t):
    """単一のUnixタイムスタンプ (秒) をUTCでの小数時間 (0.0-24.0) に変換するヘルパー関数。"""
    if np.isscalar(t):
        t = [t]
    
    ut_hours = []
    for val in t:
        # Unix Time (秒) からUTC datetimeオブジェクトを作成
        dt_utc = time.convert(val, frm='unix', into='datetime')
        # dt_utc = dt.datetime.utcfromtimestamp(val)
        # UTCでの小数時間 (Hour) を計算
        ut_h = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
        ut_hours.append(ut_h)
        
    return np.asarray(ut_hours)


def geom2rmlatmlt(unix_time, alt_val, lat1, lon1, to='rmlatmlt'):
    """
    地理座標 (GLAT, GLON, ALT) <-> 磁気座標 (R, MLAT, MLT) 間の双方向座標変換関数 (双極子近似)。

    Parameters
    ----------
    unix_time : float or array-like
        UTCでのUnixタイムスタンプ (秒)。
    alt_val : float or array-like
        'rmlatmlt'への変換時: 地理高度 ALT (km)
        'geom'への変換時: 磁気半径 R (Re, 地球中心からの距離)
    lat1 : array-like
        'rmlatmlt'への変換時: 地理緯度 GLAT (度, -90 to 90)
        'geom'への変換時: 磁気緯度 MLAT (度, -90 to 90)
    lon1 : array-like
        'rmlatmlt'への変換時: 地理経度 GLON (度, -180 to 180)
        'geom'への変換時: 磁気地方時 MLT (時, 0.0 to 24.0)
    to : str
        変換方向を指定 ('rmlatmlt' または 'geom')。

    Returns
    -------
    alt2, lat2, lon2 : tuple of array-like
        'rmlatmlt'への変換時: (R [Re], MLAT [deg], MLT [h])
        'geom'への変換時: (ALT_approx [km], GLAT_approx [deg], GLON_approx [deg])
    """
    ut_hour = _unix_to_ut_hour(unix_time)
    
    # 配列化
    alt_val = np.asarray(alt_val)
    lat1 = np.asarray(lat1)
    lon1 = np.asarray(lon1)
    
    if to.lower() == 'rmlatmlt':
        # -----------------------------------------------------
        # 変換方向: GLAT, GLON, ALT -> R, MLAT, MLT (順変換)
        # -----------------------------------------------------
        glat = np.deg2rad(lat1)
        glon = np.deg2rad(lon1)
        alt_km = alt_val 
        
        # R [Re] を計算: R = (RE + ALT [km]) / RE
        R = (RE + alt_km) / RE 
        
        # 1. Magnetic Latitude (MLAT) - 双極子近似
        # ユーザー点 R の GEO 直交座標系での単位ベクトル R_hat
        Rx = np.cos(glat) * np.cos(glon)
        Ry = np.cos(glat) * np.sin(glon)
        Rz = np.sin(glat)

        # cos(psi) = R_hat . P (Pは磁気極ベクトル)
        cos_psi = Rx * Px + Ry * Py + Rz * Pz

        # psi (磁気余緯度)
        psi = np.arccos(np.clip(cos_psi, -1.0, 1.0))
        mlat = np.rad2deg(np.pi / 2.0 - psi)

        # 2. Magnetic Local Time (MLT)
        
        # 太陽の地理経度 (Lon_sun) を UT から計算 (赤道上の太陽の経度)
        lon_sun = (180.0 - (ut_hour % 24.0) * 15.0) % 360.0
        lon_sun_r = np.deg2rad(lon_sun)

        # 太陽ベクトル S (地理直交座標系、赤道上)
        Sx = np.cos(lon_sun_r)
        Sy = np.sin(lon_sun_r)
        Sz = 0.0

        # S の磁気赤道平面への射影 S_proj = S - (S . P) * P
        S_dot_P = Sx * Px + Sy * Py + Sz * Pz
        S_proj_x = Sx - S_dot_P * Px
        S_proj_y = Sy - S_dot_P * Py
        S_proj_z = Sz - S_dot_P * Pz
        
        # R_hat の磁気赤道平面への射影 R_proj = R_hat - (R_hat . P) * P
        R_dot_P = Rx * Px + Ry * Py + Rz * Pz
        R_proj_x = Rx - R_dot_P * Px
        R_proj_y = Ry - R_dot_P * Py
        R_proj_z = Rz - R_dot_P * Pz
        
        # Y = (P x S_proj) . R_proj の計算 (スカラー三重積)
        # Pを回転軸とするS_projからR_projへの回転角度を計算
        Y_component = (Px * (S_proj_y * R_proj_z - S_proj_z * R_proj_y) +
                       Py * (S_proj_z * R_proj_x - S_proj_x * R_proj_z) +
                       Pz * (S_proj_x * R_proj_y - S_proj_y * R_proj_x))

        X_component = (S_proj_x * R_proj_x + S_proj_y * R_proj_y + S_proj_z * R_proj_z)

        # 磁気経度差 (MLON') を atan2 で計算 (ラジアン)
        mlon_prime = np.arctan2(Y_component, X_component)

        # MLT を計算 (12時が正午)
        mlt = (12.0 + np.rad2deg(mlon_prime) / 15.0) % 24.0
        
        # R [Re], MLAT [deg], MLT [h] を返す
        return R, mlat, mlt

    elif to.lower() == 'geom':
        # -----------------------------------------------------
        # 変換方向: R, MLAT, MLT -> ALT, GLAT, GLON (逆変換, 回転行列ベースに修正)
        # -----------------------------------------------------
        mlat = np.deg2rad(lat1)
        mlt = lon1
        R_in = alt_val # R [Re]
        
        # 1. 高度の近似 ALT_approx [km]
        alt_approx = R_in * RE - RE 
        
        # 2. MLT から磁気経度差 MLON (MLT=12を基準) を計算
        mlon_deg = (mlt - 12.0) * 15.0
        mlon = np.deg2rad(mlon_deg) # ラジアン

        # 3. 磁気座標系の基底ベクトル (X', Y', Z') を GEO-XYZ 座標系で表現
        
        # Z'軸 (磁気極 P)
        Z_prime_x, Z_prime_y, Z_prime_z = Px, Py, Pz
        
        # 磁気正午方向の単位ベクトル X' (GEO-XYZ)
        lon_sun = (180.0 - (ut_hour % 24.0) * 15.0) % 360.0
        lon_sun_r = np.deg2rad(lon_sun)
        
        Sx, Sy, Sz = np.cos(lon_sun_r), np.sin(lon_sun_r), 0.0
        
        # 太陽ベクトル S の磁気赤道平面への射影 S_proj = S - (S . P) * P
        S_dot_P = Sx * Px + Sy * Py + Sz * Pz
        S_proj_x = Sx - S_dot_P * Px
        S_proj_y = Sy - S_dot_P * Py
        S_proj_z = Sz - S_dot_P * Pz
        
        # X'軸 (S_proj を規格化)
        norm_S_proj = np.sqrt(S_proj_x**2 + S_proj_y**2 + S_proj_z**2)
        # norm_S_proj が0になるケースを回避 (極と太陽が重なることはほぼない)
        X_prime_x = S_proj_x / norm_S_proj
        X_prime_y = S_proj_y / norm_S_proj
        X_prime_z = S_proj_z / norm_S_proj
        
        # Y'軸 (P x X')
        Y_prime_x = Py * X_prime_z - Pz * X_prime_y
        Y_prime_y = Pz * X_prime_x - Px * X_prime_z
        Y_prime_z = Px * X_prime_y - Py * X_prime_x
        
        # 4. MLAT/MLON から MAG-XYZ' 座標系での直交座標 R' を計算し、GEO-XYZ 成分として取得
        # R_hat = cos(MLAT) * cos(MLON) * X' + cos(MLAT) * sin(MLON) * Y' + sin(MLAT) * Z'
        
        R_hat_x = (np.cos(mlat) * np.cos(mlon) * X_prime_x + 
                   np.cos(mlat) * np.sin(mlon) * Y_prime_x + 
                   np.sin(mlat) * Z_prime_x)

        R_hat_y = (np.cos(mlat) * np.cos(mlon) * X_prime_y + 
                   np.cos(mlat) * np.sin(mlon) * Y_prime_y + 
                   np.sin(mlat) * Z_prime_y)

        R_hat_z = (np.cos(mlat) * np.cos(mlon) * X_prime_z + 
                   np.cos(mlat) * np.sin(mlon) * Y_prime_z + 
                   np.sin(mlat) * Z_prime_z)
        
        # GEO-XYZ 座標 R = R_in * R_hat
        X = R_in * R_hat_x
        Y = R_in * R_hat_y
        Z = R_in * R_hat_z
        
        # 5. GEO-XYZ から GLAT/GLON に逆変換 (半径 R_in [Re] を使用)
        glat_approx = np.rad2deg(np.arcsin(Z / R_in))
        glon_approx = np.rad2deg(np.arctan2(Y, X))
        
        # GLON を -180 から 180 の範囲に正規化
        glon_approx = np.mod(glon_approx + 180.0, 360.0) - 180.0
        
        # ALT_approx [km], GLAT_approx [deg], GLON_approx [deg] を返す
        return alt_approx, glat_approx, glon_approx
        
    else:
        raise ValueError("The 'to' argument must be 'rmlatmlt' or 'geom'")
    