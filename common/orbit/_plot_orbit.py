import numpy as np
import matplotlib.pyplot as plt
from common import path
from ._base import rmlatmlt_to_polar


def plot_orbit(
        orb_data, # (n, 3)
        savefig: str | None = None,
        suptitle: str | None = None,
    ):
    """
    Time color-coded orbit plot
    
    Parameters:
    df (pandas.DataFrame): Data frame
    save_plot (bool): Whether to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    if suptitle is not None:
        fig.suptitle(suptitle)
    
    # Get data ranges
    orb_x = orb_data[:, 0]
    orb_y = orb_data[:, 1]
    orb_z = orb_data[:, 2]
    x_min, x_max = np.min(orb_x), np.max(orb_x)
    y_min, y_max = np.min(orb_y), np.max(orb_y)
    z_min, z_max = np.min(orb_z), np.max(orb_z)
    
    # Extend axis ranges to show Mercury
    # margin = max(planet_radius * 0.5, 1000)
    x_min_plot = min(x_min, -1.2)
    x_max_plot = max(x_max, 1.2)
    y_min_plot = min(y_min, -1.2)
    y_max_plot = max(y_max, 1.2)
    z_min_plot = min(z_min, -1.2)
    z_max_plot = max(z_max, 1.2)
    
    # Normalize time for color mapping
    # sum_times = times[-1] - times[0]
    time_normalized = np.linspace(0, 1, len(orb_data))
    
    # X-Y plane
    # Draw Mercury as filled black circle (draw first)
    theta = np.linspace(0, 2*np.pi, 100)
    x_mercury = np.cos(theta)
    y_mercury = np.sin(theta)
    z_mercury = np.sin(theta)

    ax1.fill(x_mercury, y_mercury, 'black', alpha=0.8, label='Mercury')
    ax1.plot(x_mercury, y_mercury, 'k-', linewidth=1)
    
    # X-Y plane orbit (draw on top of Mercury)
    scatter1 = ax1.scatter(orb_x, orb_y, c=time_normalized, 
                           cmap='viridis', s=1, alpha=0.7)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_min_plot, x_max_plot)
    ax1.set_ylim(y_min_plot, y_max_plot)
    ax1.axis('equal')
    ax1.set_title('X-Y Plane')
    
    # Add colorbar
    # cbar1 = plt.colorbar(scatter1, ax=ax1)
    # cbar1.set_label('Time (Normalized)')

    # Y-Z plane orbit (draw on top of Mercury)
    ax2.fill(x_mercury, y_mercury, 'black', alpha=0.8, label='Mercury')
    ax2.plot(x_mercury, y_mercury, 'k-', linewidth=1)

    scatter2 = ax2.scatter(orb_y, orb_z, c=time_normalized, 
                           cmap='viridis', s=1, alpha=0.7)
    
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(y_min_plot, y_max_plot)
    ax2.set_ylim(z_min_plot, z_max_plot)
    ax2.axis('equal')
    ax2.set_title('X-Y Plane')
    
    
    # Z-X plane
    # Draw Mercury as filled black circle (draw first)
    ax3.fill(x_mercury, z_mercury, 'black', alpha=0.8, label='Mercury')
    ax3.plot(x_mercury, z_mercury, 'k-', linewidth=1)
    
    # X-Z plane orbit (draw on top of Mercury)
    scatter3 = ax3.scatter(orb_x, orb_z, c=time_normalized, 
                           cmap='viridis', s=1, alpha=0.7)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(x_min_plot, x_max_plot)
    ax3.set_ylim(z_min_plot, z_max_plot)
    ax3.axis('equal')
    ax3.set_title('X-Z Plane')
    
    # Add colorbar
    # cbar2 = plt.colorbar(scatter2, ax=ax2)
    # cbar2.set_label('Time (Normalized)')
    
    # plt.tight_layout()

    # 2つのサブプロットの間に少しスペースを空ける
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.85, wspace=0.3)
    
    # カラーバー用の新しい軸をfigに追加
    # [left, bottom, width, height]
    # bottomを0.1にすることで、サブプロットと重ならないように下に配置
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    
    # カラーバーを横向きに作成
    # scatter1をカラーマッピングの元として使用
    cbar = fig.colorbar(scatter1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Time (Normalized)', fontsize=12)
    
    path.savefig(savefig)


def plot_orbit_polar(
        orb_data,  # (n, 3)
        savefig: str | None = None,
        suptitle: str | None = None
):
    """
    地球の軌道を極座標プロットする。
    """
    # 軌道データを極座標に変換
    # epoch = df_rmlatmlt.index  # datetime array

    radius = orb_data[:, 0]
    theta = orb_data[:, 1]
    phi = orb_data[:, 2]

    time_index = np.linspace(0, 1, len(radius))

    # 描画設定
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 6))
    if suptitle is not None:
        fig.suptitle(suptitle)

    # 地球を描画 (円: r = 1)
    for ax in axes:
        ax.set_ylim(0, np.max(radius) + 0.5)
        ax.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100), color='black', linewidth=1)

        # planet
        shadow_theta = np.linspace(0, 2 * np.pi, 100)
        ax.fill_between(shadow_theta, 0, 1, color='black')

    # 上面図 (北からの視点)
    sc0 = axes[0].scatter(phi, radius * np.sin(theta), c=time_index, cmap="viridis", s=1)
    axes[0].set_title("Polar View (North)")

    # 磁気赤道面図
    sc1 = axes[1].scatter(np.arctan2(np.cos(theta), np.sin(theta) * np.cos(phi)), radius * np.sqrt((np.sin(theta) * np.cos(phi)) ** 2 + np.cos(theta) ** 2), c=time_index, cmap="viridis", s=1)
    axes[1].set_title("Magnetic Equator View")

    # カラーバーをグラフ全体に追加
    cbar_ax = fig.add_axes((0.2, 0.1, 0.6, 0.02))  # [left, bottom, width, height]
    cbar = fig.colorbar(sc0, cax=cbar_ax, orientation="horizontal")
    cbar.set_label('Time (Normalized)', fontsize=12)
    # cbar.set_ticks([0, .25, .5, .75, 1])
    # cbar.ax.set_xticklabels(["00", "30", "00", "30", "00"])

    # 保存
    path.savefig(savefig)


def plot_orbit_rmlatmlt(
        orb_data,  # (n, 3)
        savefig: str | None = None,
        suptitle: str | None = None
):
    r = orb_data[:, 0]
    mlat = orb_data[:, 1]
    mlt = orb_data[:, 2]

    radius, theta, phi = rmlatmlt_to_polar(r, mlat, mlt)

    time_index = np.linspace(0, 1, len(radius))

    # 描画設定
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 6))
    if suptitle is not None:
        fig.suptitle(suptitle)

    # 地球を描画 (円: r = 1)
    for ax in axes:
        ax.set_ylim(0, np.max(radius) + 0.5)
        ax.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100), color='black', linewidth=1)

        # 地球の影部分 (270°〜90°)
        shadow_theta = np.linspace(- np.pi / 2, np.pi / 2, 100)
        ax.fill_between(shadow_theta, 0, 1, color='black')

    # (R, MLT)
    # phi += np.pi # (r, phi)から180 deg回転して(R, MLT)とする. 
    sc0 = axes[0].scatter(phi, radius * np.sin(theta), c=time_index, cmap="viridis", s=1)
    # sc0 = axes[0].scatter(phi, radius, c=time_index, cmap="viridis", s=1)
    axes[0].set_title("(R, MLT)")
    axes[0].set_xticks([0, np.pi / 2, np.pi, 3 / 2 * np.pi])
    axes[0].set_xticklabels(['00', '06', '12', '18'])
    # axes[0].set_xticklabels([])
    # axes[0].set_yticklabels([])
    # add MLT labels
    # axes[0].text(0, 8.4, '00 MLT', fontsize=10, ha='center', va='center', color='black', rotation=-90)
    # axes[0].text(90 * np.pi / 180, 8.4, '06 MLT', fontsize=10, ha='center', va='center', color='black')
    # axes[0].text(180 * np.pi / 180, 8.4, '12 MLT', fontsize=10, ha='center', va='center', color='black', rotation=90)
    # axes[0].text(270 * np.pi / 180, 8.4, '18 MLT', fontsize=10, ha='center', va='center', color='black', rotation=180)


    # (R, MLAT)
    sc1 = axes[1].scatter(np.arctan2(np.cos(theta), np.sin(theta) * np.cos(phi)), radius * np.sqrt((np.sin(theta) * np.cos(phi)) ** 2 + np.cos(theta) ** 2), c=time_index, cmap="viridis", s=1)
    theta_mlat = np.zeros_like(theta)
    theta_mlat = np.where(((mlt >= 18) | (mlt < 6)), np.pi / 2 - theta, theta_mlat) # nightside
    theta_mlat = np.where(((mlt >= 6) & (mlt < 18)), np.pi / 2 + theta, theta_mlat) # dayside
    # sc1 = axes[1].scatter(theta_mlat, radius, c=time_index, cmap="viridis", s=1)
    axes[1].set_title("(R, MLAT)")

    # カラーバーをグラフ全体に追加
    cbar_ax = fig.add_axes((0.2, 0.1, 0.6, 0.02))  # [left, bottom, width, height]
    cbar = fig.colorbar(sc0, cax=cbar_ax, orientation="horizontal")
    cbar.set_label('Time (Normalized)', fontsize=12)
    # cbar.set_ticks([0, .25, .5, .75, 1])
    # cbar.ax.set_xticklabels(["00", "30", "00", "30", "00"])

    # 保存
    path.savefig(savefig)


def plot_orbit_rmlatmlt_itself(
        orb_data,  # (n, 3) [R, MLAT(deg), MLT(hour)]
        savefig: str | None = None,
        suptitle: str | None = None,
        r_min: float = 1.0  # オフセットの基準（惑星半径）
):
    r = orb_data[:, 0]
    mlat = orb_data[:, 1]
    mlt = orb_data[:, 2]

    radius, theta, phi = rmlatmlt_to_polar(r, mlat, mlt)

    time_index = np.linspace(0, 1, len(radius))

    # 描画設定
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 6))
    if suptitle is not None:
        fig.suptitle(suptitle)

    # 地球を描画 (円: r = 1)
    for ax in axes:
        ax.set_ylim(0, np.max(radius) + 0.5)
        ax.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100), color='black', linewidth=1)

        # 地球の影部分 (270°〜90°)
        shadow_theta = np.linspace(- np.pi / 2, np.pi / 2, 100)
        ax.fill_between(shadow_theta, 0, 1, color='black')

    # (R, MLT)
    # phi += np.pi # (r, phi)から180 deg回転して(R, MLT)とする. 
    # sc0 = axes[0].scatter(phi, radius * np.sin(theta), c=time_index, cmap="viridis", s=1)
    sc0 = axes[0].scatter(phi, radius, c=time_index, cmap="viridis", s=1)
    axes[0].set_title("(R, MLT)")
    axes[0].set_xticks([0, np.pi / 2, np.pi, 3 / 2 * np.pi])
    axes[0].set_xticklabels(['00', '06', '12', '18'])
    # axes[0].set_xticklabels([])
    # axes[0].set_yticklabels([])
    # add MLT labels
    # axes[0].text(0, 8.4, '00 MLT', fontsize=10, ha='center', va='center', color='black', rotation=-90)
    # axes[0].text(90 * np.pi / 180, 8.4, '06 MLT', fontsize=10, ha='center', va='center', color='black')
    # axes[0].text(180 * np.pi / 180, 8.4, '12 MLT', fontsize=10, ha='center', va='center', color='black', rotation=90)
    # axes[0].text(270 * np.pi / 180, 8.4, '18 MLT', fontsize=10, ha='center', va='center', color='black', rotation=180)


    # (R, MLAT)
    # sc1 = axes[1].scatter(np.arctan2(np.cos(theta), np.sin(theta) * np.cos(phi)), radius * np.sqrt((np.sin(theta) * np.cos(phi)) ** 2 + np.cos(theta) ** 2), c=time_index, cmap="viridis", s=1)
    theta_mlat = np.zeros_like(theta)
    theta_mlat = np.where(((mlt >= 18) | (mlt < 6)), np.pi / 2 - theta, theta_mlat) # nightside
    theta_mlat = np.where(((mlt >= 6) & (mlt < 18)), np.pi / 2 + theta, theta_mlat) # dayside
    sc1 = axes[1].scatter(theta_mlat, radius, c=time_index, cmap="viridis", s=1)
    axes[1].set_title("(R, MLAT)")

    # カラーバーをグラフ全体に追加
    cbar_ax = fig.add_axes((0.2, 0.1, 0.6, 0.02))  # [left, bottom, width, height]
    cbar = fig.colorbar(sc0, cax=cbar_ax, orientation="horizontal")
    cbar.set_label('Time (Normalized)', fontsize=12)
    # cbar.set_ticks([0, .25, .5, .75, 1])
    # cbar.ax.set_xticklabels(["00", "30", "00", "30", "00"])

    # 保存
    path.savefig(savefig)


def old_plot_orbit_rmlatmlt(
        orb_data,  # (n, 3), R, MLAT, MLT
        savefig: str | None = None,
        suptitle: str | None = None
):
    """
    R-MLAT-MLT座標の軌道をプロットし、太陽の陰の部分を黒塗りする。
    R-MLTとR-MLATの2つのプロットを表示する。
    """
    
    # 軌道データを取得
    radius = orb_data[:, 0]
    mlat = orb_data[:, 1]
    mlt = orb_data[:, 2]

    time_index = np.linspace(0, 1, len(radius))

    # 描画設定
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if suptitle is not None:
        fig.suptitle(suptitle)

    # ---------- R-MLT プロット (極座標) ----------
    ax0 = plt.subplot(1, 2, 1, projection='polar')
    ax0.set_ylim(0, np.max(radius) + 0.5)
    
    # 太陽の陰の部分（MLT 6h〜18h）を黒塗り
    # MLTをラジアンに変換 (MLT * 15度)
    shadow_start_rad = np.radians(6 * 15)
    shadow_end_rad = np.radians(18 * 15)
    shadow_angles = np.linspace(shadow_start_rad, shadow_end_rad, 100)
    ax0.fill_between(shadow_angles, 0, 1, color='black', alpha=0.8)

    # 惑星の円周を描画
    ax0.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100), color='black', linewidth=1)

    # 軌道プロット
    sc0 = ax0.scatter(np.radians(mlt * 15), radius, c=time_index, cmap="viridis", s=10)
    
    ax0.set_title("R-MLT Plane")
    ax0.set_xlabel("MLT (Magnetic Local Time)")
    
    # MLTの目盛りを設定
    ax0.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax0.set_xticklabels(['12h', '13h', '14h', '15h', '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h'])
    
    # ax0.set_xticklabels([
    #     '0h', '1h', '2h', '3h', '4h', '5h', '6h',
    #     '7h', '8h', '9h', '10h', '11h', '12h', '13h',
    #     '14h', '15h', '16h', '17h', '18h', '19h', '20h',
    #     '21h', '22h', '23h'
    # ])
    
    # ---------- R-MLAT プロット ----------
    ax1 = plt.subplot(1, 2, 2)
    ax1.set_xlim(np.min(mlat) - 5, np.max(mlat) + 5)
    ax1.set_ylim(0, np.max(radius) + 0.5)

    # 惑星の円周（横から見た図）を描画
    theta = np.linspace(0, 2*np.pi, 100)
    x_mercury_lat = np.sin(theta) * 90/np.pi
    y_mercury_lat = np.cos(theta)
    ax1.fill_betweenx(y_mercury_lat, -90, 90, color='gray', alpha=0.8)
    ax1.plot(x_mercury_lat, y_mercury_lat, 'k-', linewidth=1)
    
    sc1 = ax1.scatter(mlat, radius, c=time_index, cmap="viridis", s=10)
    
    ax1.set_title("R-MLAT Plane")
    ax1.set_xlabel("MLAT (Magnetic Latitude)")
    ax1.set_ylabel("R (Radius)")
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # カラーバーをグラフ全体に追加
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.85, wspace=0.3)
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    cbar = fig.colorbar(sc0, cax=cbar_ax, orientation="horizontal")
    cbar.set_label('Time (Normalized)', fontsize=12)
    
    path.savefig(savefig)
