import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from matplotlib.colors import LogNorm
from common import path, display


def plot_rmlatmlt(
        mesh_theta_rmlt,
        mesh_r_rmlt,
        rmlt_grid,
        mesh_theta_rmlat,
        mesh_r_rmlat,
        rmlat_grid,
        savefig=None,
        suptitle=None,
        zlabel_rmlt=None,
        zlabel_rmlat=None,
        pos_label_rmlt=0.4,
        pos_label_rmlat=0.6,
        zrange=None,
        zlog=False,
        colormap='viridis',
        mask_zero=True,# True -> zero value is masked
        rmlat_whole=True,
        offset_label_mlat_zero = 0.2 
):
    def get_safe_zrange(data, current_zrange, is_log):
        if current_zrange is not None:
            return current_zrange
        
        # 有限な値（nan, inf 以外）のみを抽出
        finite_data = data[np.isfinite(data)]
        
        if finite_data.size == 0:
            display.warning("All data is non-finite or masked. Using default zrange.")
            return [0.1, 1.0] if is_log else [0.0, 1.0]
        
        vmin = np.nanmin(finite_data)
        vmax = np.nanmax(finite_data)
        
        if is_log:
            # 対数スケールの場合、vmin は正の値である必要がある
            if vmin <= 0:
                display.warning('Negative value is invalid inr log scale')
                positive_data = finite_data[finite_data > 0]
                vmin = np.nanmin(positive_data) if positive_data.size > 0 else 1e-3
            # vmin と vmax が同じ場合の回避
            if vmin == vmax:
                display.warning('vmin and vmax is same')
                vmin = vmin / 10.0
                vmax = vmax * 10.0
        else:
            if vmin == vmax:
                display.warning('vmin and vmax is same')
                vmin = vmin - 1.0
                vmax = vmax + 1.0
                
        return [vmin, vmax]
    
    # label
    r_lim_rmlt = np.amax(mesh_r_rmlt) if mesh_r_rmlt.size > 0 else 10.0
    label_r_rmlt = r_lim_rmlt + pos_label_rmlt  # 最大半径よりわずかに外側 (バッファとして0.4を使用)

    # R-MLATプロットの最大半径を決定し、ラベル位置を設定
    r_lim_rmlat = np.amax(mesh_r_rmlat) if mesh_r_rmlat.size > 0 else 10.0
    label_r_rmlat = r_lim_rmlat + pos_label_rmlat  # 最大半径よりわずかに外側 (バッファとして0.4を使用)

    # plot
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 8))
    if suptitle is not None:
        fig.suptitle(suptitle)

    # マスク処理：duration=0 をマスクして灰色に
    if mask_zero:
        masked_rmlt = ma.masked_where(rmlt_grid == 0, rmlt_grid)
        masked_rmlat = ma.masked_where(rmlat_grid == 0, rmlat_grid)
    else:
        masked_rmlt = rmlt_grid
        masked_rmlat = rmlat_grid

    # カラーマップ：ゼロのところは灰色に
    cmap = plt.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgray')  # マスク部分を灰色に

    # (r, mlt)
    if zlabel_rmlt is None:
        zlabel_rmlt = '(R, MLT)'
    axes[0].set_rlim(0, r_lim_rmlt)

    safe_zrange_rmlt = get_safe_zrange(masked_rmlt, zrange, zlog)
    # if zrange is None:
    #     zrange = [np.nanmin(masked_rmlt), np.nanmax(masked_rmlt)]
    
    if zlog:
        pcm0 = axes[0].pcolormesh(mesh_theta_rmlt, mesh_r_rmlt, masked_rmlt, shading='auto', cmap=cmap, norm=LogNorm(vmin=safe_zrange_rmlt[0], vmax=safe_zrange_rmlt[1]))
    else:
        pcm0 = axes[0].pcolormesh(mesh_theta_rmlt, mesh_r_rmlt, masked_rmlt, shading='auto', cmap=cmap, vmin=safe_zrange_rmlt[0], vmax=safe_zrange_rmlt[1])
    fig.colorbar(
        pcm0, ax=axes[0], label=zlabel_rmlt,
        orientation='horizontal', pad=0.15, fraction=0.046,
    )

    # supress automatical tick labels
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    # add MLT labels
    axes[0].text(90 * np.pi / 180, label_r_rmlt, '06 MLT', fontsize=10, ha='center', va='center', color='black')
    axes[0].text(180 * np.pi / 180, label_r_rmlt, '12 MLT', fontsize=10, ha='center', va='center', color='black', rotation=90)
    axes[0].text(270 * np.pi / 180, label_r_rmlt, '18 MLT', fontsize=10, ha='center', va='center', color='black', rotation=180)
    axes[0].text(360 * np.pi / 180, label_r_rmlt, '24 MLT', fontsize=10, ha='center', va='center', color='black', rotation=-90)

    # axes[0].text(90 * np.pi / 180, 8.4, '06 MLT', fontsize=10, ha='center', va='center', color='black')
    # axes[0].text(180 * np.pi / 180, 8.4, '12 MLT', fontsize=10, ha='center', va='center', color='black', rotation=90)
    # axes[0].text(270 * np.pi / 180, 8.4, '18 MLT', fontsize=10, ha='center', va='center', color='black', rotation=180)
    # axes[0].text(360 * np.pi / 180, 8.4, '24 MLT', fontsize=10, ha='center', va='center', color='black', rotation=-90)

    # oplot earth
    theta2 = np.linspace(-np.pi / 2, np.pi / 2, 100)  # 0-180 deg
    r = theta2 * 0 + 1  # radius = 1
    axes[0].fill(theta2, r, color='black', alpha=1.0)  # fill the semicircle

    # (r, mlat)
    if zlabel_rmlat is None:
        zlabel_rmlat = '(R, MLAT)'
    axes[1].set_rlim(0, r_lim_rmlat)

    if zrange is None:
        zrange = [np.nanmin(masked_rmlat), np.nanmax(masked_rmlat)]
    if zlog:
        pcm1 = axes[1].pcolormesh(mesh_theta_rmlat, mesh_r_rmlat, masked_rmlat, shading='auto', cmap=cmap, norm=LogNorm(vmin=safe_zrange_rmlt[0], vmax=safe_zrange_rmlt[1]))
    else:
        pcm1 = axes[1].pcolormesh(mesh_theta_rmlat, mesh_r_rmlat, masked_rmlat, shading='auto', cmap=cmap, vmin=safe_zrange_rmlt[0], vmax=safe_zrange_rmlt[1])

    fig.colorbar(
        pcm1, ax=axes[1], label=zlabel_rmlat,
        orientation='horizontal', pad=0.15, fraction=0.046
    )

    if rmlat_whole:
        # 軸ラベルを消去してカスタムラベルを配置
        axes[1].set_xticklabels([])
        axes[1].set_yticklabels([])
        
        # 角度 180 (左) が昼側の赤道、0 (右) が夜側の赤道
        # 角度 90 (上) が北極、270 (下) が南極
        axes[1].text(0, label_r_rmlat + offset_label_mlat_zero, '0 MLAT', fontsize=9, ha='left', va='center')
        
        # axes[1].text(0, label_r_rmlat, '0', fontsize=9, ha='center', va='center')
        axes[1].text(45 * np.pi / 180, label_r_rmlat, '45', fontsize=9, ha='center', va='center')
        axes[1].text(90 * np.pi / 180, label_r_rmlat, '90', fontsize=9, ha='center', va='center')
        axes[1].text(270 * np.pi / 180, label_r_rmlat, '-90', fontsize=9, ha='center', va='center')
        axes[1].text(315 * np.pi / 180, label_r_rmlat, '-45', fontsize=9, ha='center', va='center')
        
        # 地球を黒く塗る (全円)
        axes[1].fill(theta2, r, color='black', alpha=1.0)  # fill the semicircle
        # t_earth = np.linspace(0, 2 * np.pi, 100)
        # axes[1].fill(t_earth, np.ones_like(t_earth), color='black')
    else:
        # [-90°, 90°]だけに制限
        axes[1].set_thetamin(-90)
        axes[1].set_thetamax(90)

        axes[1].text(0, label_r_rmlat, 'MLAT', fontsize=10, ha='center', va='center', color='black')
    # axes[1].text(0, 9.5, 'MLAT', fontsize=10, ha='center', va='center', color='black')

    # 地球を黒く塗る
    # theta2 = np.linspace(-np.pi / 2, np.pi / 2, 100)
    # r_earth = np.ones_like(theta2) * 1  # radius=1
    # ax.fill(theta2, r_earth, color='black', alpha=1.0)

    path.savefig(savefig)

    return


def plot_rmlatmlt_dict(
        dict_distribution,
        varname_mesh_theta_rmlt='mesh_theta_rmlt',
        varname_mesh_r_rmlt='mesh_r_rmlt',
        varname_rmlt_grid='rmlt_grid',
        varname_mesh_theta_rmlat='mesh_theta_rmlat',
        varname_mesh_r_rmlat='mesh_r_rmlat',
        varname_rmlat_grid='rmlat_grid',
        savefig=None,
        suptitle=None,
        zlabel_rmlt=None,
        zlabel_rmlat=None,
        pos_label_rmlt=0.4,
        pos_label_rmlat=0.6,
        zrange=None,
        zlog=False,
        colormap='viridis',
        mask_zero=True,# True -> zero value is masked
        rmlat_whole=True,
        offset_label_mlat_zero = 0.2 
):
    return plot_rmlatmlt(
        dict_distribution[varname_mesh_theta_rmlt],
        dict_distribution[varname_mesh_r_rmlt],
        dict_distribution[varname_rmlt_grid],
        dict_distribution[varname_mesh_theta_rmlat],
        dict_distribution[varname_mesh_r_rmlat],
        dict_distribution[varname_rmlat_grid],
        savefig=savefig,
        suptitle=suptitle,
        zlabel_rmlt=zlabel_rmlt,
        zlabel_rmlat=zlabel_rmlat,
        pos_label_rmlt=pos_label_rmlt,
        pos_label_rmlat=pos_label_rmlat,
        zrange=zrange,
        zlog=zlog,
        colormap=colormap,
        mask_zero=mask_zero,
        rmlat_whole=rmlat_whole,
        offset_label_mlat_zero=offset_label_mlat_zero
    )