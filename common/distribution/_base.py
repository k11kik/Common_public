import numpy as np
import os
from spacepy import pycdf
from datetime import datetime
from common import pytplot, display, path, cdf


def rmlatmlt_meshgrid_rmlat_half(
        varname, # rmlatmlt
        outcdf=False,
        save_cdf=None,
        datatype='orbit',# 'orbit' -> return dwell time
        delta_t_sec=None,
        varname_data=None,
        info=True,
        r_bins=None,
        mlt_bins=None,
        mlat_bins=None,
):
    """
    Return
    -----
    dict: 
    * 'mesh_theta_rmlt'
    * 'mesh_r_rmlt'
    * 'rmlt_grid'
    * 'mesh_theta_rmlat'
    * 'mesh_r_rmlat'
    * 'rmlat_grid'
    """
    if outcdf and save_cdf is None:
        display.warning('save_cdf must be defined -> return None')
        return None
    
    # check datatype
    valid_datatype = ['count', 'orbit', 'average']
    if not datatype in valid_datatype:
        display.warning(f'Invalid datatype: {datatype}. Available: {valid_datatype}')
        return
    
    if datatype == 'average':
        if varname_data is None:
            display.warning("datatype = 'average' => varname_data must be defined")
            return
    
    # グリッド設定
    if r_bins is None:
        r_bins = np.arange(1, 7.5 + 0.1, 0.5)  # 1～7まで0.5刻み
    if mlt_bins is None:
        mlt_bins = np.arange(0, 24 + 1, 1)  # 0～24まで1刻み
    if mlat_bins is None:
        mlat_bins = np.arange(-90, 90 + 5, 5)  # -90～90まで5刻み

    # カウント用配列
    rmlt_grid_count = np.zeros((len(r_bins) - 1, len(mlt_bins) - 1))  # (r, mlt)の2D

    rmlat_grid_count = np.zeros((len(r_bins) - 1, len(mlat_bins) - 1))  # (r, mlat)の2D

    dat_rmlatmlt = pytplot.get_data(varname)
    if dat_rmlatmlt is None:
        display.info('dat_rmlatmlt is None')
        return None
    times_rmlatmlt, rmlatmlt = dat_rmlatmlt.times, dat_rmlatmlt.y

    loop_start_time = datetime.now()
    for i in range(len(times_rmlatmlt)):
        if info:
            display.progress_bar(i, len(times_rmlatmlt), loop_start_time)
        r, mlat, mlt = rmlatmlt[i]

        if mlt > 24:
            mlt = mlt % 24

        # それぞれbinに落とし込む
        r_idx = np.digitize(r, r_bins) - 1
        mlt_idx = np.digitize(mlt, mlt_bins) - 1
        mlat_idx = np.digitize(mlat, mlat_bins) - 1

        if (0 <= r_idx < len(r_bins) - 1) and (0 <= mlt_idx < len(mlt_bins) - 1):
            rmlt_grid_count[r_idx, mlt_idx] += 1
        if (0 <= r_idx < len(r_bins) - 1) and (0 <= mlat_idx < len(mlat_bins) - 1):
            rmlat_grid_count[r_idx, mlat_idx] += 1

    if datatype == 'count':
        rmlt_grid = rmlt_grid_count
        rmlat_grid = rmlat_grid_count

    elif datatype == 'orbit':
        if delta_t_sec is None:
            delta_t_sec = (times_rmlatmlt[1] - times_rmlatmlt[0])

        # count number -> second
        rmlt_grid = rmlt_grid_count * delta_t_sec
        rmlat_grid = rmlat_grid_count * delta_t_sec
    
    elif datatype == 'average':
        # data
        dat = pytplot.get_data(varname_data)
        data = dat.y
        if data.ndim != 1:
            display.warning('data must be 1darray')
            return
        if len(data) != len(times_rmlatmlt):
            display.warning('The length of data and rmlatmlt must be same')
            return
        if np.sum(np.isnan(data)) != 0:
            display.warning('data include nan')

        rmlt_grid_data = np.zeros((len(r_bins) - 1, len(mlt_bins) - 1))  # (r, mlt)の2D
        rmlat_grid_data = np.zeros((len(r_bins) - 1, len(mlat_bins) - 1))  # (r, mlat)の2D

        dat_rmlatmlt = pytplot.get_data(varname)
        if dat_rmlatmlt is None:
            display.info('dat_rmlatmlt is None')
            return None
        times_rmlatmlt, rmlatmlt = dat_rmlatmlt.times, dat_rmlatmlt.y

        loop_start_time = datetime.now()
        for i in range(len(times_rmlatmlt)):
            if info:
                display.progress_bar(i, len(times_rmlatmlt), loop_start_time)
            r, mlat, mlt = rmlatmlt[i]

            if mlt > 24:
                mlt = mlt % 24

            # それぞれbinに落とし込む
            r_idx = np.digitize(r, r_bins) - 1
            mlt_idx = np.digitize(mlt, mlt_bins) - 1
            mlat_idx = np.digitize(mlat, mlat_bins) - 1

            if (0 <= r_idx < len(r_bins) - 1) and (0 <= mlt_idx < len(mlt_bins) - 1):
                rmlt_grid_data[r_idx, mlt_idx] += data[i]
            if (0 <= r_idx < len(r_bins) - 1) and (0 <= mlat_idx < len(mlat_bins) - 1):
                rmlat_grid_data[r_idx, mlat_idx] += data[i]
        
        # average
        rmlt_grid = np.where(rmlt_grid_count == 0, 0, rmlt_grid_data / rmlt_grid_count)
        rmlat_grid = np.where(rmlat_grid_count == 0, 0, rmlat_grid_data / rmlat_grid_count)


    # (r, mlt)
    theta_mlt = (mlt_bins / 24) * 2 * np.pi
    mesh_theta_rmlt, mesh_r_rmlt = np.meshgrid(theta_mlt, r_bins)

    # (r, mlat)
    theta_mlat = np.deg2rad(mlat_bins)  # -90度～90度 → -π/2～π/2ラジアン
    mesh_theta_rmlat, mesh_r_rmlat = np.meshgrid(theta_mlat, r_bins)

    # output cdf file
    if outcdf:
        path.make_directory(save_cdf)
        if os.path.exists(save_cdf):
            os.remove(save_cdf)
        with pycdf.CDF(save_cdf, '') as cdf:
            cdf['r_bins'] = r_bins
            cdf['mlt_bins'] = mlt_bins
            cdf['mlat_bins'] = mlat_bins
            cdf['mesh_theta_rmlt'] = mesh_theta_rmlt
            cdf['mesh_r_rmlt'] = mesh_r_rmlt
            cdf['rmlt_grid'] = rmlt_grid
            cdf['mesh_theta_rmlat'] = mesh_theta_rmlat
            cdf['mesh_r_rmlat'] = mesh_r_rmlat
            cdf['rmlat_grid'] = rmlat_grid
        
        display.current_time_comment(comment=f'Saved cdf: {save_cdf}')

    return {
        'mesh_theta_rmlt': mesh_theta_rmlt,
        'mesh_r_rmlt': mesh_r_rmlt,
        'rmlt_grid': rmlt_grid,
        'rmlt_grid_count': rmlt_grid_count,
        'mesh_theta_rmlat': mesh_theta_rmlat,
        'mesh_r_rmlat': mesh_r_rmlat,
        'rmlat_grid': rmlat_grid,
        'rmlat_grid_count': rmlat_grid_count,
    }


def calculate_time_intervals(times):
    """
    各データポイントが占める時間幅(sec)を計算する。
    中央差分的な考え方で、隣接ポイントとの中点間の時間を割り当てる。
    """
    if len(times) < 2:
        return np.array([0.0])
    
    dt = np.zeros_like(times)
    diff = np.diff(times)
    # 最初と最後は隣との差分、中間は前後との差分の平均
    dt[0] = diff[0]
    dt[-1] = diff[-1]
    dt[1:-1] = (diff[:-1] + diff[1:]) / 2.0
    return dt


def exclude_nan(varname):
    if varname not in pytplot.tplot_names(quiet=True):
        display.warning(f"Not found: {varname}")
        return
    
    # exclude NaN
    dat = pytplot.get_data(varname)
    data = dat.y
    is_nan = np.isnan(data)
    if np.sum(is_nan) == 0:
        pass
    else:
        display.info(f'{np.sum(is_nan)} NaN values detected -> converted into zero')
        data = np.where(is_nan, 0, data)
        pytplot.store_data(varname, {'x': dat.times, 'y': data}, replace=True)
    
    return


def rmlatmlt_meshgrid_rmlat_whole(
        varname, # 1d-array
        outcdf=False,
        save_cdf=None,
        datatype='orbit',
        delta_t_sec=None, # 固定値を使いたい場合のみ使用
        varname_data=None, 
        info=True,
        r_bins=None,
        mlt_bins=None,
        mlat_bins=None,
):
    # exclude nan
    exclude_nan(varname)

    if varname_data is not None:
        exclude_nan(varname_data)

    # --- 初期設定 ---
    if r_bins is None: r_bins = np.arange(1, 7.5 + 0.1, 0.5)
    if mlt_bins is None: mlt_bins = np.arange(0, 24 + 1, 1)
    if mlat_bins is None: mlat_bins = np.arange(-90, 90 + 5, 5)
    n_mlat = len(mlat_bins) - 1

    dat_rmlatmlt = pytplot.get_data(varname)
    if dat_rmlatmlt is None: return None
    times_rmlatmlt, rmlatmlt = dat_rmlatmlt.times, dat_rmlatmlt.y

    # --- 時間解像度の変動への対応 ---
    if datatype == 'orbit':
        if varname_data is not None: # その地点で加算すべき時間幅
            # 外部から計算済みの dt * flag を受け取る
            dat_weights = pytplot.get_data(varname_data)
            if dat_weights is None:
                display.warning('dat_weights is None')
                return
            dts = dat_weights.y
            if len(dts) != len(times_rmlatmlt):
                display.warning(f'The length of {varname} and {varname_data} must be same')
                return
        elif delta_t_sec is not None:
            dts = np.ones_like(times_rmlatmlt) * delta_t_sec
        else:
            dts = calculate_time_intervals(times_rmlatmlt)
    else:
        dts = np.ones_like(times_rmlatmlt)

    # グリッド初期化
    rmlt_grid = np.zeros((len(r_bins) - 1, len(mlt_bins) - 1))
    rmlat_grid = np.zeros((len(r_bins) - 1, 2 * n_mlat))
    # 平均計算用のカウント（常に1を足す）
    rmlt_grid_count = np.zeros_like(rmlt_grid)
    rmlat_grid_count = np.zeros_like(rmlat_grid)

    # average用のデータ
    data_val = None
    if datatype == 'average':
        dat = pytplot.get_data(varname_data)
        data_val = np.where(np.isnan(dat.y), 0, dat.y)

    loop_start_time = datetime.now()
    for i in range(len(times_rmlatmlt)):
        if info: display.progress_bar(i, len(times_rmlatmlt), loop_start_time)
        
        r, mlat, mlt = rmlatmlt[i]
        if mlt > 24: mlt = mlt % 24

        r_idx = np.digitize(r, r_bins) - 1
        mlt_idx = np.digitize(mlt, mlt_bins) - 1
        mlat_idx_base = np.digitize(mlat, mlat_bins) - 1

        # 有効なインデックス範囲内かチェック
        if not (0 <= r_idx < len(r_bins) - 1): continue

        # R-MLT 集計
        if 0 <= mlt_idx < len(mlt_bins) - 1:
            if datatype == 'orbit':
                rmlt_grid[r_idx, mlt_idx] += dts[i]
            elif datatype == 'average':
                rmlt_grid[r_idx, mlt_idx] += data_val[i]
            else: # count
                rmlt_grid[r_idx, mlt_idx] += 1
            rmlt_grid_count[r_idx, mlt_idx] += 1

        # R-MLAT 集計
        if 0 <= mlat_idx_base < n_mlat:
            if 6 <= mlt < 18:
                target_mlat_idx = mlat_idx_base
            else:
                target_mlat_idx = (n_mlat - 1 - mlat_idx_base) + n_mlat
            
            if datatype == 'orbit':
                rmlat_grid[r_idx, target_mlat_idx] += dts[i]
            elif datatype == 'average':
                rmlat_grid[r_idx, target_mlat_idx] += data_val[i]
            else: # count
                rmlat_grid[r_idx, target_mlat_idx] += 1
            rmlat_grid_count[r_idx, target_mlat_idx] += 1

    # --- 平均処理 ---
    if datatype == 'average':
        rmlt_grid = np.divide(rmlt_grid, rmlt_grid_count, out=np.zeros_like(rmlt_grid), where=rmlt_grid_count != 0)
        rmlat_grid = np.divide(rmlat_grid, rmlat_grid_count, out=np.zeros_like(rmlat_grid), where=rmlat_grid_count != 0)

    # --- メッシュ作成 (省略せず既存ロジックを維持) ---
    theta_mlt = (mlt_bins / 24) * 2 * np.pi
    mesh_theta_rmlt, mesh_r_rmlt = np.meshgrid(theta_mlt, r_bins)
    dayside_theta = np.deg2rad(180 - mlat_bins) 
    nightside_theta = np.deg2rad(mlat_bins[::-1]) 
    full_theta_mlat = np.concatenate([dayside_theta[:-1], nightside_theta])
    mesh_theta_rmlat, mesh_r_rmlat = np.meshgrid(full_theta_mlat, r_bins)

    dict_return = {
        'mesh_theta_rmlt': mesh_theta_rmlt,
        'mesh_r_rmlt': mesh_r_rmlt,
        'rmlt_grid': rmlt_grid,
        'rmlt_grid_count': rmlt_grid_count,
        'mesh_theta_rmlat': mesh_theta_rmlat,
        'mesh_r_rmlat': mesh_r_rmlat,
        'rmlat_grid': rmlat_grid,
        'rmlat_grid_count': rmlat_grid_count,
    }
    
    if outcdf: cdf.dict_to_cdffile(dict_return, save_cdf)
    return dict_return


def old_rmlatmlt_meshgrid_rmlat_whole(
        varname, # rmlatmlt
        outcdf=False,
        save_cdf=None,
        datatype='orbit',
        delta_t_sec=None,
        varname_data=None, # 1D array
        info=True,
        r_bins=None,
        mlt_bins=None,
        mlat_bins=None,
):
    """
    (R, MLAT)分布をMLTに基づいて円形(360度)に展開するように修正
    """
    if outcdf and save_cdf is None:
        display.warning('save_cdf must be defined -> return None')
        return None
    
    valid_datatype = ['count', 'orbit', 'average']
    if not datatype in valid_datatype:
        display.warning(f'Invalid datatype: {datatype}. Available: {valid_datatype}')
        return
    
    if datatype == 'average' and varname_data is None:
        display.warning("datatype = 'average' => varname_data must be defined")
        return
    
    # --- グリッド設定 ---
    if r_bins is None:
        r_bins = np.arange(1, 7.5 + 0.1, 0.5)
    if mlt_bins is None:
        mlt_bins = np.arange(0, 24 + 1, 1)
    
    # MLATを円形にするための新しいビン定義
    # 0~360度(または-pi~pi)の範囲でデータを集計する必要があるため、
    # 内部的に「疑似方位角(phi)」を導入します。
    # 0-180度を昼側(MLT:6-18)、180-360度を夜側(MLT:18-6)に割り当てるイメージです。
    if mlat_bins is None:
        mlat_bins = np.arange(-90, 90 + 5, 5)
    
    # 出力用のグリッド形状 (R x MLATのインデックス)
    # R-MLAT表示を円にするために、実際には「どのMLTにいるか」で集計先を分ける
    # 昼側(Dayside)と夜側(Nightside)でMLATのビンを分けたグリッドを作成
    # index: 0 ~ len(mlat_bins)-2 は昼側、len(mlat_bins)-1 ~ 2*(len(mlat_bins)-2) は夜側
    n_mlat = len(mlat_bins) - 1
    rmlat_grid_count = np.zeros((len(r_bins) - 1, 2 * n_mlat)) 
    rmlt_grid_count = np.zeros((len(r_bins) - 1, len(mlt_bins) - 1))

    dat_rmlatmlt = pytplot.get_data(varname)
    if dat_rmlatmlt is None: return None
    times_rmlatmlt, rmlatmlt = dat_rmlatmlt.times, dat_rmlatmlt.y

    # average用のデータ準備
    data_val = None
    if datatype == 'average':
        dat = pytplot.get_data(varname_data)
        times_data = dat.times
        if len(times_data) != len(times_rmlatmlt):
            display.warning(f'The length of {varname} and {varname_data} must be same')
            return
        
        data_val = dat.y
        data_val = np.where(np.isnan(data_val), 0, data_val)
        rmlat_grid_data = np.zeros_like(rmlat_grid_count)
        rmlt_grid_data = np.zeros_like(rmlt_grid_count)

    loop_start_time = datetime.now()
    for i in range(len(times_rmlatmlt)):
        if info: display.progress_bar(i, len(times_rmlatmlt), loop_start_time)
        r, mlat, mlt = rmlatmlt[i]
        if mlt > 24: mlt = mlt % 24

        r_idx = np.digitize(r, r_bins) - 1
        mlt_idx = np.digitize(mlt, mlt_bins) - 1
        mlat_idx_base = np.digitize(mlat, mlat_bins) - 1

        # R-MLT 集計
        if (0 <= r_idx < len(r_bins) - 1) and (0 <= mlt_idx < len(mlt_bins) - 1):
            rmlt_grid_count[r_idx, mlt_idx] += 1
            if datatype == 'average': rmlt_grid_data[r_idx, mlt_idx] += data_val[i]

        # R-MLAT 集計 (MLTを用いて昼夜を識別)
        # 6 <= mlt < 18 を「右側(昼)」、それ以外を「左側(夜)」とする
        if (0 <= r_idx < len(r_bins) - 1) and (0 <= mlat_idx_base < n_mlat):
            if 6 <= mlt < 18:
                # 昼側: そのままのインデックス (右半分)
                target_mlat_idx = mlat_idx_base
            else:
                # 夜側: 反転させてオフセットを加える (左半分)
                # MLAT +90(北極)付近を共通にするなら、インデックスの並び順に注意
                # ここでは単純に 夜側の区画(n_mlat 以降)に格納
                # 夜側のMLATインデックスを逆順にすると円として繋がりやすい
                target_mlat_idx = (n_mlat - 1 - mlat_idx_base) + n_mlat
            
            rmlat_grid_count[r_idx, target_mlat_idx] += 1
            if datatype == 'average': rmlat_grid_data[r_idx, target_mlat_idx] += data_val[i]

    # --- メッシュ作成 ---
    # R-MLT メッシュ
    theta_mlt = (mlt_bins / 24) * 2 * np.pi
    mesh_theta_rmlt, mesh_r_rmlt = np.meshgrid(theta_mlt, r_bins)

    # R-MLAT メッシュ (円形化)
    dayside_theta = np.deg2rad(180 - mlat_bins) 
    nightside_theta = np.deg2rad(mlat_bins[::-1]) 
    
    # 注意: target_mlat_idx の格納順序と theta の順序を合わせる必要があります
    # 昼側データ(index 0~n_mlat-1)は dayside_theta の各区間に対応
    full_theta_mlat = np.concatenate([dayside_theta[:-1], nightside_theta])
    mesh_theta_rmlat, mesh_r_rmlat = np.meshgrid(full_theta_mlat, r_bins)

    # --- データ形成 ---
    if datatype == 'count':
        rmlt_grid = rmlt_grid_count
        rmlat_grid = rmlat_grid_count
    elif datatype == 'orbit':
        if delta_t_sec is None: delta_t_sec = times_rmlatmlt[1] - times_rmlatmlt[0]
        rmlt_grid = rmlt_grid_count * delta_t_sec
        rmlat_grid = rmlat_grid_count * delta_t_sec
    elif datatype == 'average':
        rmlt_grid = np.divide(
            rmlt_grid_data, 
            rmlt_grid_count, 
            out=np.full_like(rmlt_grid_data, 0), 
            where=rmlt_grid_count != 0
        )
        
        rmlat_grid = np.divide(
            rmlat_grid_data, 
            rmlat_grid_count, 
            out=np.full_like(rmlat_grid_data, 0), 
            where=rmlat_grid_count != 0
        )
        # rmlt_grid = np.where(rmlt_grid_count == 0, np.nan, rmlt_grid_data / rmlt_grid_count)
        # rmlat_grid = np.where(rmlat_grid_count == 0, np.nan, rmlat_grid_data / rmlat_grid_count)

    dict_return = {
        'mesh_theta_rmlt': mesh_theta_rmlt,
        'mesh_r_rmlt': mesh_r_rmlt,
        'rmlt_grid': rmlt_grid,
        'rmlt_grid_count': rmlt_grid_count,
        'mesh_theta_rmlat': mesh_theta_rmlat,
        'mesh_r_rmlat': mesh_r_rmlat,
        'rmlat_grid': rmlat_grid,
        'rmlat_grid_count': rmlat_grid_count,
    }
    # output cdf file
    if outcdf:
        cdf.dict_to_cdffile(dict_return, save_cdf)
        # path.make_directory(save_cdf)
        # if os.path.exists(save_cdf):
        #     os.remove(save_cdf)
        # with pycdf.CDF(save_cdf, '') as cdf:
        #     cdf['r_bins'] = r_bins
        #     cdf['mlt_bins'] = mlt_bins
        #     cdf['mlat_bins'] = mlat_bins
        #     cdf['mesh_theta_rmlt'] = mesh_theta_rmlt
        #     cdf['mesh_r_rmlt'] = mesh_r_rmlt
        #     cdf['rmlt_grid'] = rmlt_grid
        #     cdf['mesh_theta_rmlat'] = mesh_theta_rmlat
        #     cdf['mesh_r_rmlat'] = mesh_r_rmlat
        #     cdf['rmlat_grid'] = rmlat_grid
        # 
        # display.current_time_comment(comment=f'Saved cdf: {save_cdf}')

    return dict_return


def rmlatmlt_meshgrid(
        varname, # rmlatmlt
        outcdf=False,
        save_cdf=None,
        datatype='orbit',# 'orbit' -> return dwell time
        delta_t_sec=None,
        varname_data=None,
        info=False,
        r_bins=None,
        mlt_bins=None,
        mlat_bins=None,
        rmlat_whole=True
):
    """
    * varname: rmlatmlt
    * datatype:
        * 'count'
        * 'orbit'
            * varname_data -> data by time point to add
            * delta_t_sec -> constant value
        * 'average' -> varname_data is to be set
    
    Return
    -----
    Not including NaN value

    dict: 
    * 'mesh_theta_rmlt'
    * 'mesh_r_rmlt'
    * 'rmlt_grid'
    * 'rmlt_grid_count'
    * 'mesh_theta_rmlat'
    * 'mesh_r_rmlat'
    * 'rmlat_grid'
    * 'rmlat_grid_count'
    """
    if rmlat_whole:
        return rmlatmlt_meshgrid_rmlat_whole(
            varname=varname,
            outcdf=outcdf,
            save_cdf=save_cdf,
            datatype=datatype,
            delta_t_sec=delta_t_sec,
            varname_data=varname_data,
            info=info,
            r_bins=r_bins,
            mlt_bins=mlt_bins,
            mlat_bins=mlat_bins
        )
    
    else:
        return rmlatmlt_meshgrid_rmlat_half(
            varname=varname,
            outcdf=outcdf,
            save_cdf=save_cdf,
            datatype=datatype,
            delta_t_sec=delta_t_sec,
            varname_data=varname_data,
            info=info,
            r_bins=r_bins,
            mlt_bins=mlt_bins,
            mlat_bins=mlat_bins
        )


def update_dict(
        dict_data,
        dict_data_i
):
    dict_return = dict_data.copy()
    if dict_return:
        sum_rmlt_grid_data = dict_data['rmlt_grid'] * dict_data['rmlt_grid_count']
        sum_rmlat_grid_data = dict_data['rmlat_grid'] * dict_data['rmlat_grid_count']

        sum_rmlt_grid_i = dict_data_i['rmlt_grid'] * dict_data_i['rmlt_grid_count']
        sum_rmlat_grid_i = dict_data_i['rmlat_grid'] * dict_data_i['rmlat_grid_count']

        sum_rmlt_grid = sum_rmlt_grid_data + sum_rmlt_grid_i
        sum_rmlat_grid = sum_rmlat_grid_data + sum_rmlat_grid_i

        sum_rmlt_grid_count = dict_data['rmlt_grid_count'] + dict_data_i['rmlt_grid_count']
        sum_rmlat_grid_count = dict_data['rmlat_grid_count'] + dict_data_i['rmlat_grid_count']

        rmlt_grid = np.zeros_like(sum_rmlt_grid)
        nonzero_rmlt_count = sum_rmlt_grid_count != 0
        rmlt_grid[nonzero_rmlt_count] = sum_rmlt_grid[nonzero_rmlt_count] / sum_rmlt_grid_count[nonzero_rmlt_count]
        dict_return['rmlt_grid'] = rmlt_grid

        rmlat_grid = np.zeros_like(sum_rmlat_grid)
        nonzero_rmlat_count = sum_rmlat_grid_count != 0
        rmlat_grid[nonzero_rmlat_count] = sum_rmlat_grid[nonzero_rmlat_count] / sum_rmlat_grid_count[nonzero_rmlat_count]
        dict_return['rmlat_grid'] = rmlat_grid

        dict_return['rmlt_grid_count'] = sum_rmlt_grid_count
        dict_return['rmlat_grid_count'] = sum_rmlat_grid_count
    
    else:
        dict_return = dict_data_i

    return dict_return


def update_dict_sum(dict_data, dict_data_i):
    """
    滞在時間(orbit)やカウント(count)など、単純加算すべきデータ用のマージ関数。
    """
    if not dict_data:
        return dict_data_i.copy()
    
    dict_return = dict_data.copy()
    
    # 単純加算すべきフィールドのリスト
    target_fields = ['rmlt_grid', 'rmlt_grid_count', 'rmlat_grid', 'rmlat_grid_count']
    
    for field in target_fields:
        if field in dict_data and field in dict_data_i:
            dict_return[field] = dict_data[field] + dict_data_i[field]
            
    return dict_return