"""
plot

Notes
    * in plt.subplots()
    * metadata
        mdat = pytplot.get_data('xxx', metadata=True)

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
import logging
import pytplot
from datetime import datetime, timezone
from matplotlib import gridspec

from common import util
from common.base import path, display


def add_to_dict(
        dict_data: dict,
        dict_ref: dict,
        name_ref: str,
        default_value=None,
        new_name: str | None = None
):
    if new_name is None:
        new_name = name_ref
    if name_ref in dict_ref.keys():
        dict_data[new_name] = dict_ref[name_ref]
    else:
        dict_data[new_name] = default_value
    return


def categorize_type():
    return


def _normal(
        ax,
        dat,
        mdat,
        display_options: bool = False,
        xlim: list | None = None
):
    dict_options = {}

    # xlim
    if xlim is None:
        ids, ide = 0, len(dat.times) - 1
    else:
        ids = np.abs(dat.times - pytplot.time_double(xlim[0])).argmin()
        ide = np.abs(dat.times - pytplot.time_double(xlim[1])).argmin()
        # ax.set_xlim(pytplot.time_double(xlim[0]), pytplot.time_double(xlim[1]))

    # extras
    dict_extras = mdat['plot_options']['extras']
    add_to_dict(dict_options, dict_extras, 'line_color', ['black'])
    add_to_dict(dict_options, dict_extras, 'alpha', 1)

    # line option
    dict_line_opt = mdat['plot_options']['line_opt']
    add_to_dict(dict_options, dict_line_opt, 'line_style_name', ['solid'])
    add_to_dict(dict_options, dict_line_opt, 'line_width', [1])
    add_to_dict(dict_options, dict_line_opt, 'marker_size', None)
    add_to_dict(dict_options, dict_line_opt, 'marker', None)


    # y-axis option
    dict_yaxis_opt = mdat['plot_options']['yaxis_opt']
    add_to_dict(dict_options, dict_yaxis_opt, 'y_axis_type', 'linear')
    add_to_dict(dict_options, dict_yaxis_opt, 'y_range')
    add_to_dict(dict_options, dict_yaxis_opt, 'axis_label', new_name='yaxis_label')
    if dict_options['y_axis_type'] == 'linear':
        pass
    elif dict_options['y_axis_type'] == 'log':
        ax.set_yscale('log')
    yrange = dict_options['y_range']
    if yrange is not None and not np.isnan(dat.y).all():
        ax.set_ylim([yrange[0], yrange[1]])
    if dict_options['yaxis_label'] is not None:
        ax.set_ylabel(dict_options['yaxis_label'])

    # check options
    if display_options:
        util.print_dict(dict_options, 'options')

    # plot
    if dat.y.ndim == 1:
        ax.plot(
            dat.times[ids:ide], dat.y[ids:ide],
            color=dict_options['line_color'][0],
            alpha=dict_options['alpha'],
            linestyle=dict_options['line_style_name'][0],
            linewidth=dict_options['line_width'][0],
            markersize=dict_options['marker_size'],
            marker=dict_options['marker']
        )

    else:
        n_dim = dat.y.shape[1]
        list_color = ['blue', 'green', 'red', 'aqua', 'orange', 'violet']
        color = dict_options['line_color']
        if len(color) != n_dim:
            color = []
            for i in range(n_dim):
                idx_color = i % len(list_color)
                color.append(list_color[idx_color])
        line_style_name = dict_options['line_style_name']
        if len(line_style_name) != n_dim:
            line_style_name = [line_style_name[0]] * n_dim
        line_width = dict_options['line_width']
        if len(line_width) != n_dim:
            line_width = None

        for i in range(dat.y.shape[1]):
            ax.plot(
                dat.times[ids:ide], dat.y[ids:ide, i],
                color=color[i],
                alpha=dict_options['alpha'],
                linestyle=line_style_name[i],
                linewidth=line_width,
                markersize=dict_options['marker_size'],
                marker=dict_options['marker']
            )

    if 'legend_names' in mdat['plot_options']['yaxis_opt']:
        ax.legend(mdat['plot_options']['yaxis_opt']['legend_names'], loc='upper right')

    return


def normal(
    ax,
    dat,
    mdat,
    display_options: bool = False,
    xlim: list | None = None
):
    """
    1次元データをプロットし、凡例ラベルを設定します。
    スペクトログラムではない通常のラインプロット用です。

    Args:
        ax (matplotlib.axes.Axes): プロットを描画するAxesオブジェクト。
        dat (pytplot.data_type): プロットするデータ（pytplot.get_dataの戻り値）。
        mdat (dict): データのメタデータ（pytplot.get_data(metadata=True)の戻り値）。
        display_options (bool): オプションを表示するかどうか。
        xlim (list): X軸の表示範囲 [開始時刻, 終了時刻]。

    Returns:
        tuple: プロットされたラインのリストと凡例ラベルのリスト。
               何もプロットされなければ ([], []) を返します。
    """
    dict_options = {}
    lines_plotted = []
    labels_plotted = []

    # xlim (時間範囲) の設定
    ids, ide = 0, len(dat.times) - 1
    if xlim is not None:
        ids = np.abs(dat.times - pytplot.time_double(xlim[0])).argmin()
        ide = np.abs(dat.times - pytplot.time_double(xlim[1])).argmin()
        # ax.set_xlim(pytplot.time_double(xlim[0]), pytplot.time_double(xlim[1])) # overplot関数でまとめて設定

    # extras オプションの適用
    dict_extras = mdat['plot_options']['extras']
    add_to_dict(dict_options, dict_extras, 'line_color', ['black'])
    add_to_dict(dict_options, dict_extras, 'alpha', 1)

    # line option の適用
    dict_line_opt = mdat['plot_options']['line_opt']
    add_to_dict(dict_options, dict_line_opt, 'line_style_name', ['solid'])
    add_to_dict(dict_options, dict_line_opt, 'line_width', [1])
    add_to_dict(dict_options, dict_line_opt, 'marker_size', None)
    add_to_dict(dict_options, dict_line_opt, 'marker', None)

    # y-axis option の適用
    dict_yaxis_opt = mdat['plot_options']['yaxis_opt']
    add_to_dict(dict_options, dict_yaxis_opt, 'y_axis_type', 'linear')
    add_to_dict(dict_options, dict_yaxis_opt, 'y_range')
    add_to_dict(dict_options, dict_yaxis_opt, 'axis_label', new_name='yaxis_label')
    
    if dict_options['y_axis_type'] == 'log':
        ax.set_yscale('log')
    
    yrange = dict_options['y_range']
    if yrange is not None and not np.isnan(dat.y).all():
        ax.set_ylim([yrange[0], yrange[1]])
    
    if dict_options['yaxis_label'] is not None:
        ax.set_ylabel(dict_options['yaxis_label'])

    if display_options:
        util.print_dict(dict_options, 'options')

    # プロットの実行と凡例ラベルの決定
    # 凡例ラベルは、mdat['plot_options']['yaxis_opt']['legend_names']
    # または変数名 itself (dat.var_name) を使用
    
    # 単一の1次元データの場合
    if dat.y.ndim == 1:
        # 凡例ラベルの取得: yaxis_optにlegend_namesがあればそれを使う、なければ変数名
        label = mdat['plot_options']['yaxis_opt'].get('legend_names', [mdat.get('var_name', None)])[0]
        
        line, = ax.plot(
            dat.times[ids:ide], dat.y[ids:ide],
            color=dict_options['line_color'][0],
            alpha=dict_options['alpha'],
            linestyle=dict_options['line_style_name'][0],
            linewidth=dict_options['line_width'][0],
            markersize=dict_options['marker_size'],
            marker=dict_options['marker'],
            label=label # ここで凡例ラベルを設定
        )
        lines_plotted.append(line)
        labels_plotted.append(label)

    # 複数次元データの場合 (例: dat.y.shape = (時間, 成分数))
    else:
        n_dim = dat.y.shape[1]
        list_color = ['blue', 'green', 'red', 'aqua', 'orange', 'violet']
        colors = dict_options['line_color']
        if len(colors) != n_dim: # 指定された色が次元数と合わない場合
            colors = [list_color[i % len(list_color)] for i in range(n_dim)]
            
        line_styles = dict_options['line_style_name']
        if len(line_styles) != n_dim:
            line_styles = [line_styles[0]] * n_dim # 最初のスタイルを繰り返す

        line_widths = dict_options['line_width']
        if line_widths is None or len(line_widths) != n_dim:
             line_widths = [1] * n_dim # デフォルトの幅

        # 凡例ラベルの取得 (複数次元の場合、legend_namesはリストであることが期待される)
        base_label = mdat.get('var_name', 'data')
        legend_names = mdat['plot_options']['yaxis_opt'].get('legend_names')
        
        for i in range(dat.y.shape[1]):
            # 各成分の凡例ラベルを決定
            if legend_names and len(legend_names) > i:
                label = legend_names[i]
            else:
                label = f"{base_label}_{i+1}" # デフォルトのラベル
            
            line, = ax.plot(
                dat.times[ids:ide], dat.y[ids:ide, i],
                color=colors[i],
                alpha=dict_options['alpha'],
                linestyle=line_styles[i],
                linewidth=line_widths[i] if isinstance(line_widths, list) else line_widths, # Noneの場合を考慮
                markersize=dict_options['marker_size'],
                marker=dict_options['marker'],
                label=label # ここで凡例ラベルを設定
            )
            lines_plotted.append(line)
            labels_plotted.append(label)
    
    return lines_plotted, labels_plotted


def get_time_edges(times):
    dt = np.diff(times) / 2
    t_edges = np.zeros(len(times) + 1)
    t_edges[1:-1] = times[:-1] + dt
    t_edges[0] = times[0] - dt[0]
    t_edges[-1] = times[-1] + dt[-1]
    return t_edges


def spectrogram(
        fig,
        ax,
        dat,
        mdat,
        ax_cbar,
        display_options: bool = False,
        xlim: list | None = None
):
    dict_options = {}

    # xlim
    if xlim is None:
        ids, ide = 0, len(dat.times) - 1
    else:
        ids = np.abs(dat.times - pytplot.time_double(xlim[0])).argmin()
        ide = np.abs(dat.times - pytplot.time_double(xlim[1])).argmin()
        # ax.set_xlim(pytplot.time_double(xlim[0]), pytplot.time_double(xlim[1]))

    # extras
    dict_extras = mdat['plot_options']['extras']
    add_to_dict(dict_options, dict_extras, 'colormap', 'jet')
    add_to_dict(dict_options, dict_extras, 'alpha', 1)

    # y-axis option
    dict_yaxis_opt = mdat['plot_options']['yaxis_opt']
    add_to_dict(dict_options, dict_yaxis_opt, 'y_axis_type', 'linear')
    add_to_dict(dict_options, dict_yaxis_opt, 'y_range')
    add_to_dict(dict_options, dict_yaxis_opt, 'axis_label', new_name='yaxis_label')
    if dict_options['y_axis_type'] == 'linear':
        pass
    elif dict_options['y_axis_type'] == 'log':
        ax.set_yscale('log')
    yrange = dict_options['y_range']
    if yrange is not None:
        ax.set_ylim([yrange[0], yrange[1]])
    if dict_options['yaxis_label'] is not None:
        ax.set_ylabel(dict_options['yaxis_label'])


    # z-axis option
    dict_zaxis_opt = mdat['plot_options']['zaxis_opt']
    add_to_dict(dict_options, dict_zaxis_opt, 'z_range', [np.min(dat.y), np.max(dat.y)])
    add_to_dict(dict_options, dict_zaxis_opt, 'axis_label', new_name='zaxis_label')
    zrange = dict_options['z_range']
    zlabel = dict_options['zaxis_label']

    # check options
    if display_options:
        util.print_dict(dict_options, 'options')

    # plot
    # time_edges = get_time_edges(dat.times[ids:ide+1])  # ide+1で右端までカバー

    cmap = dict_options['colormap']
    if mdat['plot_options']['zaxis_opt']['z_axis_type'] == 'linear':
        pcm = ax.pcolormesh(dat.times[ids:ide], dat.v, dat.y[ids:ide, :].T, cmap=cmap, alpha=dict_options['alpha'])
    elif mdat['plot_options']['zaxis_opt']['z_axis_type'] == 'log':
        pcm = ax.pcolormesh(dat.times[ids:ide], dat.v, dat.y[ids:ide, :].T, norm=LogNorm(vmin=zrange[0], vmax=zrange[1]), cmap=cmap, alpha=dict_options['alpha'])
    else:
        logging.error('z_axis_type must be linear or log.')
        pcm = None

    fig.colorbar(pcm, cax=ax_cbar, label=zlabel)


def _overplot(
        fig,
        ax,
        vars_overplot: list,
        ax_cbar,
        xlim: list | None = None
):
    vars_plotted = []
    is_spec = []
    for var_plot in vars_overplot:
        dat = pytplot.get_data(var_plot)
        mdat = pytplot.get_data(var_plot, metadata=True)

        if dat is None:
            logging.error(f'{var_plot} is None')
            is_spec.append(-1)
            continue


        vars_plotted.append(var_plot)
        dict_mdat_extras = mdat['plot_options']['extras']
        if 'spec' in dict_mdat_extras.keys() and dict_mdat_extras['spec'] == 1:
            is_spec.append(1)
        else:
            is_spec.append(0)

    vars_overplot_spec = []  # spec vars
    vars_overplot_nonspec = []  # non-spec vars (to overplot)
    for i, var in enumerate(vars_overplot):
        if not is_spec:
            return
        if is_spec[i] == 1:
            vars_overplot_spec.append(var)
        elif is_spec[i] == -1:
            continue
        else:
            vars_overplot_nonspec.append(var)

    if not vars_overplot_spec:  # no spec data
        list_yrange = []
        for i, var in enumerate(vars_overplot_nonspec):
            dat_var = pytplot.get_data(var)
            mdat_var = pytplot.get_data(var, metadata=True)
            normal(ax, dat_var, mdat_var, xlim=xlim)
        
            # yrange
            if 'y_range' in mdat_var['plot_options']['yaxis_opt'].keys():
                yrange = mdat_var['plot_options']['yaxis_opt']['y_range']
            else:
                if dat_var is None:
                    yrange = [-1, 1]
                else:
                    yrange = [np.min(dat_var.y), np.max(dat_var.y)]
            list_yrange.append(yrange)
        list_yrange = np.array(list_yrange)
        if list_yrange.ndim == 1:
            yrange = list_yrange
        elif list_yrange.ndim == 2:
            yrange = [np.min(list_yrange[:, 0]), np.max(list_yrange[:, 1])]
        else:
            logging.error(f'Unsupported ndim: {list_yrange.ndim=}')
        ax_cbar.axis('off')
        return

    else:
        if len(vars_overplot_spec) >= 2:
            logging.error('vars_overplot_spec must be 1 or zero')
        var_overplot_spec = vars_overplot_spec[0]
        dat_overplot_spec = pytplot.get_data(var_overplot_spec)
        mdat_overplot_spec = pytplot.get_data(var_overplot_spec, metadata=True)

        # spectrogram
        spectrogram(fig, ax, dat_overplot_spec, mdat_overplot_spec, ax_cbar, xlim=xlim)

        # over plot
        for i, var in enumerate(vars_overplot_nonspec):
            dat_var = pytplot.get_data(var)
            mdat_var = pytplot.get_data(var, metadata=True)
            normal(ax, dat_var, mdat_var, xlim=xlim)

        # yrange
        if 'y_range' in mdat_overplot_spec['plot_options']['yaxis_opt'].keys():
            yrange = mdat_overplot_spec['plot_options']['yaxis_opt']['y_range']
        else:
            yrange = [dat_overplot_spec.v[0], dat_overplot_spec[-1]]

    ax.set_ylim(yrange)


    return vars_plotted


def overplot(
    fig,
    ax,
    vars_overplot: list,
    ax_cbar,
    xlim: list | None = None
):
    """
    複数の変数を一つのAxesに重ねてプロットします。
    凡例を正しく表示するために、normal関数からラベルを収集して最後にまとめて表示します。

    Args:
        fig (matplotlib.figure.Figure): Figureオブジェクト。
        ax (matplotlib.axes.Axes): メインのAxesオブジェクト。
        vars_overplot (list): プロットする変数名のリスト。
        ax_cbar (matplotlib.axes.Axes): カラーバー用のAxesオブジェクト（スペクトログラム用）。
        xlim (list): X軸の表示範囲 [開始時刻, 終了時刻]。

    Returns:
        list: プロットされた変数名のリスト。
    """
    vars_plotted = []
    is_spec = []
    
    # 凡例表示のために、プロットされたオブジェクトとラベルを収集するリスト
    all_lines = []
    all_labels = []

    for var_plot in vars_overplot:
        dat = pytplot.get_data(var_plot)
        mdat = pytplot.get_data(var_plot, metadata=True)

        if dat is None:
            logging.error(f'{var_plot} is None')
            is_spec.append(-1)
            continue

        vars_plotted.append(var_plot)
        dict_mdat_extras = mdat['plot_options']['extras']
        if 'spec' in dict_mdat_extras.keys() and dict_mdat_extras['spec'] == 1:
            is_spec.append(1)
        else:
            is_spec.append(0)

    vars_overplot_spec = []  # スペクトログラム変数
    vars_overplot_nonspec = []  # 1次元プロット変数 (重ね書き用)
    for i, var in enumerate(vars_overplot):
        # is_spec リストのチェック方法を修正
        # if not is_spec: return の行は不要、または条件を見直すべきです。
        # is_specが空の場合の処理は、この後の if not vars_overplot_spec: で対処されます。
        
        if is_spec[i] == 1:
            vars_overplot_spec.append(var)
        elif is_spec[i] == -1: # データがNoneだった場合
            continue
        else: # is_spec[i] == 0 (非スペクトログラム)
            vars_overplot_nonspec.append(var)

    if not vars_overplot_spec:  # スペクトログラムデータがない場合 (1次元プロットのみ)
        list_yrange = []
        for var in vars_overplot_nonspec:
            dat_var = pytplot.get_data(var)
            mdat_var = pytplot.get_data(var, metadata=True)
            
            # normal 関数からプロットされたラインとラベルを受け取る
            lines, labels = normal(ax, dat_var, mdat_var, xlim=xlim)
            all_lines.extend(lines)
            all_labels.extend(labels)
            
            # yrange の収集 (これは既存ロジック)
            if 'y_range' in mdat_var['plot_options']['yaxis_opt'].keys():
                yrange = mdat_var['plot_options']['yaxis_opt']['y_range']
            else:
                if dat_var is None:
                    yrange = [-1, 1]
                else:
                    yrange = [np.min(dat_var.y), np.max(dat_var.y)]
            list_yrange.append(yrange)
        
        list_yrange = np.array(list_yrange)
        if list_yrange.ndim == 1:
            yrange_final = list_yrange
        elif list_yrange.ndim == 2:
            yrange_final = [np.min(list_yrange[:, 0]), np.max(list_yrange[:, 1])]
        else:
            logging.error(f'Unsupported ndim: {list_yrange.ndim=}')
            yrange_final = [-1, 1] # Fallback
            
        ax_cbar.axis('off') # スペクトログラムがないのでカラーバーはオフ
        ax.set_ylim(yrange_final) # 収集したyrangeをまとめて設定

    else: # スペクトログラムデータがある場合
        if len(vars_overplot_spec) >= 2:
            logging.error('vars_overplot_spec must be 1 or zero')
            return # 複数のスペクトログラムは処理しない
        
        var_overplot_spec = vars_overplot_spec[0]
        dat_overplot_spec = pytplot.get_data(var_overplot_spec)
        mdat_overplot_spec = pytplot.get_data(var_overplot_spec, metadata=True)

        # スペクトログラムの描画
        spectrogram(fig, ax, dat_overplot_spec, mdat_overplot_spec, ax_cbar, xlim=xlim)

        # スペクトログラムに重ねて1次元データをプロット
        for var in vars_overplot_nonspec:
            dat_var = pytplot.get_data(var)
            mdat_var = pytplot.get_data(var, metadata=True)
            
            # normal 関数からプロットされたラインとラベルを受け取る
            lines, labels = normal(ax, dat_var, mdat_var, xlim=xlim)
            all_lines.extend(lines)
            all_labels.extend(labels)

        # yrange (スペクトログラムのy軸範囲を使用)
        if 'y_range' in mdat_overplot_spec['plot_options']['yaxis_opt'].keys():
            yrange_final = mdat_overplot_spec['plot_options']['yaxis_opt']['y_range']
        else:
            # dat_overplot_spec.v は周波数/エネルギー軸などのデータ
            yrange_final = [dat_overplot_spec.v[0], dat_overplot_spec.v[-1]]

        ax.set_ylim(yrange_final)

    # X軸範囲の最終設定 (全てのプロットに対して共通)
    if xlim is not None:
        ax.set_xlim(pytplot.time_double(xlim[0]), pytplot.time_double(xlim[1]))
    
    # 凡例の表示: normal関数で設定されたラベルを持つラインを対象にする
    # 実際には `ax.legend()` は `ax.plot()` から自動的に `label` 引数を収集します。
    # しかし、明示的にハンドルとラベルを渡すことで、より制御がしやすくなります。
    # ここでは単純に `ax.legend()` を呼び出し、plot呼び出し時のlabel引数に依存します。
    if all_labels: # プロットされたラインがあれば凡例を表示
        # duplicate=False を追加して、もし同じラベルが複数回現れても1つだけ表示
        # handler_map はより複雑な凡例カスタマイズ用ですが、ここでは不要
        ax.legend(loc='upper right')

    return vars_plotted



def get_xticks_datetime_from_unix(
        ax,
        times,
        delta: int = 15,
        timeunit: str = 'minutes',
        xlim: list | None = None
):
    unit_to_seconds = {
        'hours': 3600,
        'minutes': 60,
        'seconds': 1
    }

    if timeunit not in unit_to_seconds:
        raise ValueError(f"timeunit must be one of {list(unit_to_seconds.keys())}, got '{timeunit}'")

    interval_sec = delta * unit_to_seconds[timeunit]

    if xlim is not None:
        t0 = datetime.strptime(xlim[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        t1 = datetime.strptime(xlim[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        start = int(np.ceil(t0 / interval_sec) * interval_sec)
        end = int(np.floor(t1 / interval_sec) * interval_sec)

    else:
        # Round start and end to the nearest interval
        start = int(np.floor(times[0] / interval_sec) * interval_sec)
        end = int(np.ceil(times[-1] / interval_sec) * interval_sec)

    # Set ticks
    xticks = np.arange(start, end + 1, interval_sec)
    ax.set_xticks(xticks)

    # Format for display
    if timeunit == 'hours':
        fmt = '%H:%M'
    elif timeunit == 'minutes':
        fmt = '%H:%M'
    else:  # seconds
        fmt = '%H:%M:%S'

    xtick_labels = [datetime.fromtimestamp(t, tz=timezone.utc).strftime(fmt) for t in xticks]
    # 最後に日付を追加（x軸の右下に）
    final_time = datetime.fromtimestamp(times[-1], tz=timezone.utc)
    date_str = final_time.strftime('%Y-%m-%d')

    last_xtick_label = f'{xtick_labels[-1]}\n{date_str}'
    xtick_labels[-1] = last_xtick_label
    return xtick_labels


def get_xticks_datetime_with_orbit(
        ax,
        times,
        dat_orbit,
        delta: int = 15,
        timeunit: str = 'minutes',
        deltay_ticks = 0.2,
        xlim: list | None = None
):
    unit_to_seconds = {
        'hours': 3600,
        'minutes': 60,
        'seconds': 1
    }

    if timeunit not in unit_to_seconds:
        raise ValueError(f"timeunit must be one of {list(unit_to_seconds.keys())}, got '{timeunit}'")

    interval_sec = delta * unit_to_seconds[timeunit]

    if xlim is not None:
        t0 = datetime.strptime(xlim[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        t1 = datetime.strptime(xlim[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        start = int(np.ceil(t0 / interval_sec) * interval_sec)
        end = int(np.floor(t1 / interval_sec) * interval_sec)

    else:
        # Round start and end to the nearest interval
        start = int(np.floor(times[0] / interval_sec) * interval_sec)
        end = int(np.ceil(times[-1] / interval_sec) * interval_sec)

    # Set ticks
    xticks = np.arange(start, end + 1, interval_sec)
    ax.set_xticks(xticks)

    # Format for display
    if timeunit == 'hours':
        fmt = '%H:%M'
    elif timeunit == 'minutes':
        fmt = '%H:%M'
    else:  # seconds
        fmt = '%H:%M:%S'

    # 衛星軌道データから、各xtickに対応するインデックス取得
    times_orb = dat_orbit.times
    rmlatmlt = dat_orbit.y  # shape = (N, 3)

    xtick_labels = []
    for xtick in xticks:
        label = ''
        idx = np.argmin(np.abs(times_orb - xtick))  # 最も近い時刻のインデックス
        for i in range(rmlatmlt.shape[1]):
            data_orbit = rmlatmlt[idx, i]
            str_orbit = f"{data_orbit:.2f}"
            label += f"{str_orbit}\n"
        time_str = datetime.fromtimestamp(xtick, tz=timezone.utc).strftime(fmt)
        label += f"{time_str}"
        xtick_labels.append(label)
        
        # r, mlat, mlt = rmlatmlt[idx]

        # # フォーマット
        # r_str = f"{r:.2f}"
        # mlat_str = f"{mlat:.2f}"
        # mlt_str = f"{mlt:.2f}"
        # time_str = datetime.fromtimestamp(xtick, tz=timezone.utc).strftime(fmt)

        # label = f"{r_str}\n{mlat_str}\n{mlt_str}\n{time_str}"
        # xtick_labels.append(label)

    # 最後だけ日付も足す
    date_str = datetime.fromtimestamp(times[-1], tz=timezone.utc).strftime('%Y-%m-%d')
    xtick_labels[-1] += f"\n{date_str}"

    return xtick_labels


# def label_orbit(
#         fig,
#         ax,
#         dat,
#         dat_orbit,
#         delta: int = 15,
#         timeunit: str = 'minutes',
#         deltay_ticks = 0.2,
#         label_pos_x=-1,
#         label_pos_y=-1,
#         list_label: list = None,
#         label_offset_x: float = 0.1,
#         label_offset_y: float = 0.005,
#         left_margin: float = 0.2,
#         ax_label=None
# ):
#     # xticks: Unix時間
#     unit_to_seconds = {'hours': 3600, 'minutes': 60, 'seconds': 1}
#     interval_sec = delta * unit_to_seconds[timeunit]
#     start = int(np.floor(dat.times[0] / interval_sec) * interval_sec)
#     end = int(np.ceil(dat.times[-1] / interval_sec) * interval_sec)
#     xticks = np.arange(start, end + 1, interval_sec)
#     ax.set_xticks(xticks)
#
#     # 衛星軌道データから、各xtickに対応するインデックス取得
#     times_orb = dat_orbit.times
#     rmlatmlt = dat_orbit.y  # shape = (N, 3)
#
#     labels = []
#     for xtick in xticks:
#         idx = np.argmin(np.abs(times_orb - xtick))  # 最も近い時刻のインデックス
#         r, mlat, mlt = rmlatmlt[idx]
#
#         # フォーマット
#         r_str = f"{r:.2f}"
#         mlat_str = f"{mlat:.2f}"
#         mlt_str = f"{mlt:.2f}"
#         time_str = datetime.fromtimestamp(xtick, tz=timezone.utc).strftime('%H:%M')
#         # time_str = datetime.utcfromtimestamp(xtick).strftime('%H:%M')
#
#         label = f"{r_str}\n{mlat_str}\n{mlt_str}\n{time_str}"
#         labels.append(label)
#
#     # 最後だけ日付も足す
#     date_str = datetime.fromtimestamp(dat.times[-1], tz=timezone.utc).strftime('%Y-%m-%d')
#     labels[-1] += f"\n{date_str}"
#
#     # label
#     if list_label is None:
#         list_label = ["R [Re]", "MLAT [deg]", "MLT [hr]", "TIME [HH:MM]"]
#
#     for i, label in enumerate(list_label):
#         y = 1.0 - label_offset_y - i * deltay_ticks
#         ax_label.text(1.0, y, label, ha='right', va='center', transform=ax_label.transAxes, fontsize=9)
#
#     # if ax_label is not None:
#     #     ax_label.axis('off')
#     #     for i, label in enumerate(list_label):
#     #         y = 1.0 - label_offset_y - i * deltay_ticks  # 上から順に
#     #         ax_label.text(
#     #             1.0, y, label,
#     #             ha='right', va='center',
#     #             transform=ax_label.transAxes,
#     #             fontsize=9
#     #         )
#
#
#
#
#     # # plotより左に、ラベル用Axesを作る
#     # label_ax_width = 0.1  # 適宜調整
#     # main_ax_pos = ax.get_position()
#     # label_ax_left = main_ax_pos.x0 - label_ax_width - 0.01  # ちょい余白入れて
#     #
#     # ax_label = fig.add_axes([
#     #     label_ax_left,
#     #     main_ax_pos.y0,
#     #     .1,
#     #     .1
#     # ])
#     # # ax_label = fig.add_axes([
#     # #     label_ax_left,
#     # #     main_ax_pos.y0 - 1,
#     # #     label_ax_width,
#     # #     main_ax_pos.height
#     # # ])
#     # # ax_label.axis('off')  # 軸は非表示にする
#     #
#     # if list_label is None:
#     #     list_label = ["R [Re]", "MLAT [deg]", "MLT [hr]", "TIME [HH:MM]"]
#     # for i, label in enumerate(list_label):
#     #     y = 1.0 - label_offset_y - i * deltay_ticks  # 上から下に
#     #     ax_label.text(
#     #         1.0, y, label,
#     #         ha='right', va='center',
#     #         transform=ax_label.transAxes,
#     #         fontsize=9
#     #     )
#
#
#
#     # ==== ラベル左に行ラベルを追加 ====
#     # fig = ax.figure
#     # ax_pos = ax.get_position()  # Bbox object with (x0, y0, x1, y1)
#     # x = ax_pos.x0 - label_offset_x  # 少し左へオフセット
#     # # x = ax_pos.x0 - left_margin - label_offset_x
#     # display.debug('label_orbit', f'{left_margin=}')  # left_margin=0.5 => x=0.4, 0=>-0.1
#     # display.debug('label_orbit', f'{ax_pos.x0=}')  # (left_margin, ax.x0) = (0.2, 0.125)
#     # display.debug('label_orbit', f'{x=}')  # left_margin=0.5 => x=0.4, 0=>-0.1
#     # if list_label is None:
#     #     list_label = ["R [Re]", "MLAT [deg]", "MLT [hr]", "TIME [HH:MM]"]
#     # for i, label in enumerate(list_label):
#     #     y = ax_pos.y0 - label_offset_y - i * deltay_ticks
#     #     fig.text(x, y, label, ha='right', va='center', fontsize=9)
#
#
#     # # データ座標じゃなくて「axes座標（0〜1）」を使うのがポイント
#     # ax_xmin, ax_xmax = ax.get_xlim()
#     # ax_ymin, ax_ymax = ax.get_ylim()
#     # x_pos = xticks[0] - (xticks[1] - xticks[0]) * 0.5  # xtickの外側に少し左へ
#     #
#     # # y位置の調整（軸下に配置）
#     # y = ax.get_position().y0  # axesのy座標（下端）
#     # fig = ax.get_figure()
#
#     # # 各ラベルのy位置（データ座標じゃなくaxes座標）
#     # delta_y_ticks = deltay_ticks
#
#     # for i, label in enumerate(list_label):
#     #     ax.annotate(
#     #         label,
#     #         xy=(label_pos_x, label_pos_y - i * deltay_ticks),  # (x, y) in axis coords
#     #         xycoords='axes fraction',
#     #         ha='right',
#     #         va='center',
#     #         fontsize=9
#     #     )
#
#
#     # for i, label in enumerate(list_label):
#     #     fig.text(
#     #         ax.get_position().x0 - label_pos_x,
#     #         ax.get_position().y0 - label_pos_y - (deltay_ticks * i),  # 行ごとにずらす
#     #         label,
#     #         ha='right',
#     #         va='center',
#     #         fontsize=9
#     #     )
#
#
#     return labels


# def tplot_without_orbit(
#         vars_plot: list,
#         save_png: str = None,
#         figsize: tuple = (12, 6),
#         suptitle: str = None,
#         delta_xticks = 15,
#         timeunit_xticks: str = 'minutes',
#         hspace: float = 0.1,
#         tight_layout=True,
#         xlim: list = None
# ):
#     if not isinstance(vars_plot, list):
#         logging.info('vars_plot was converted to list')
#
#
#     vars_plotted = []
#
#     fig, axes = plt.subplots(
#         len(vars_plot), 2,
#         figsize=figsize,
#         tight_layout=tight_layout,
#         sharex='col',
#         gridspec_kw={
#             'width_ratios': [50, 1],
#             'height_ratios': [1] * len(vars_plot)
#         },
#     )
#     fig.subplots_adjust(hspace=hspace)
#
#     for i in range(len(vars_plot)):
#         if len(vars_plot) == 1:
#             ax = axes[0]
#             ax_cbar = axes[1]
#         else:
#             ax = axes[i, 0]
#             ax_cbar = axes[i, 1]
#
#         if isinstance(vars_plot[i], str):
#             var_plot = vars_plot[i]
#             dat = pytplot.get_data(var_plot)
#             if dat is None:
#                 logging.error(f'{var_plot} is None')
#                 continue
#             vars_plotted.append(var_plot)
#             mdat = pytplot.get_data(var_plot, metadata=True)
#
#             # dict_mdat_extras = mdat['plot_options']['extras']
#             extras = mdat.get('plot_options', {}).get('extras', {})
#             if extras.get('spec') == 1: #'spec' in dict_mdat_extras.keys() and dict_mdat_extras['spec'] == 1:
#                 # spectrogram
#                 spectrogram(fig, ax, dat, mdat, ax_cbar, xlim=xlim)
#             else:
#                 # normal
#                 normal(ax, dat, mdat, xlim=xlim)
#                 ax_cbar.axis('off')
#
#         # over plot
#         elif isinstance(vars_plot[i], list):
#             vars_overplot = vars_plot[i]
#             var_ovplot = overplot(fig, ax, vars_overplot, ax_cbar, xlim=xlim)
#             if var_ovplot is not None:
#                 vars_plotted.append(var_ovplot)
#
#     ax_xticks = axes[0] if len(vars_plot) == 1 else axes[-1][0]
#
#     display.debug('tplot_wo_orb', f'{vars_plotted=}')
#
#     if isinstance(vars_plotted[-1], str):
#         dat_last = pytplot.get_data(vars_plotted[-1])
#     else:
#         dat_last = pytplot.get_data(vars_plotted[-1][0])
#     # dat_last = pytplot.get_data(vars_plot[-1] if isinstance(vars_plot[-1], str) else vars_plot[-1][0])
#
#     # ax_xticks.set_xticklabels(xtick_labels)
#     # if len(vars_plot) == 1:
#     #     ax_xticks = axes[0][0]
#     # else:
#     #     ax_xticks = axes[len(vars_plot)-1][0]
#     #
#     # if isinstance(vars_plot[-1], str):
#     #     dat_last = pytplot.get_data(vars_plot[-1])
#     # else:
#     #     dat_last = pytplot.get_data(vars_plot[-1][0])
#     # times = dat_last.times
#
#     # x-ticks: unix -> datetime
#     xtick_labels = get_xticks_datetime_from_unix(ax_xticks, dat_last.times, delta=delta_xticks,
#                                                  timeunit=timeunit_xticks)
#
#     ax_xticks.set_xticklabels(xtick_labels)
#
#     if suptitle is not None:
#         fig.suptitle(suptitle)
#
#     path.savefig(save_png)
#     return


def apply_options(
        dict_options: dict,
        ax,
        ax_cbar=None
):
    if ax is None:
        display.error('plot/apply_options', f'ax is None')
        return
    
    # legend
    legend_loc = dict_options.get('legend_loc', 'upper right')
    legend = dict_options.get('legend', True)
    if legend:
        if 'legend_names' in dict_options.keys():
            legend_names = dict_options.get('legend_names')
            if not isinstance(legend_names, list):
                legend_names = list(legend_names)
            ax.legend(legend_names, loc=legend_loc)
        else:
            ax.legend(loc=legend_loc)
    else:
        ax.legend().set_visible(False)
            

    for key, value in dict_options.items():
        # title
        if key == 'ytitle':
            ax.set_ylabel(value)
            continue

        if key == 'ztitle' and ax_cbar is not None:
            ax_cbar.set_ylabel(value)
            continue

        # if key == 'legend_names':
        #     if not isinstance(value, list):
        #         value = list(value)
        #     ax.legend(value, loc='upper right')
        #     continue

        # if key == 'legend':
        #     if value:
        #         ax.legend(loc='upper right')
        #     else:
        #         ax.legend().set_visible(False)
        
    return

def add_ax_options(
        axes,
        axes_cbar,
        ax_options: dict
):
    for ax_i, info_options in ax_options.items():
        if int(ax_i) > len(axes):
            return
        ax = axes[int(ax_i) - 1]
        ax_cbar = axes_cbar[int(ax_i) - 1]
        apply_options(info_options, ax, ax_cbar=ax_cbar)
    return


def tplot_without_orbit_gs(
        vars_plot: list,
        save_png: str | None = None,
        figsize: tuple = (12, 6),
        suptitle: str | None = None,
        delta_xticks = 15,
        timeunit_xticks: str = 'minutes',
        hspace: float = 0.05,
        wspace: float = 0.02,
        top: float = 0.90,
        xlim: list | None = None,
        ax_options = None
):
    if not isinstance(vars_plot, list):
        logging.info('vars_plot was converted to list')

    nrow = len(vars_plot)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(top=top)
    gs = gridspec.GridSpec(
        nrow+1, 2,
        width_ratios=[50, 1],
        height_ratios=[1] + [2] * nrow,
        hspace=hspace,
        wspace=wspace,
    )
    # variables plotted
    vars_plotted = []

    # suptitle
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis('off')

    axes = []
    axes_cbar = []
    ax_master = None
    for i in range(1, nrow+1):
        if ax_master is None:
            ax = fig.add_subplot(gs[i, 0])
            ax_master = ax
        else:
            ax = fig.add_subplot(gs[i, 0], sharex=ax_master)
        ax_cbar = fig.add_subplot(gs[i, 1])
        axes.append(ax)
        axes_cbar.append(ax_cbar)

    for i in range(len(vars_plot)):
        ax = axes[i]
        ax_cbar = axes_cbar[i]

        if isinstance(vars_plot[i], str):
            var_plot = vars_plot[i]
            dat = pytplot.get_data(var_plot)
            mdat = pytplot.get_data(var_plot, metadata=True)

            if dat is None:
                logging.error(f'{var_plot} is None')
                continue

            vars_plotted.append(var_plot)

            # dict_mdat_extras = mdat['plot_options']['extras']
            extras = mdat.get('plot_options', {}).get('extras', {})
            if extras.get('spec') == 1: #'spec' in dict_mdat_extras.keys() and dict_mdat_extras['spec'] == 1:
                # spectrogram
                spectrogram(fig, ax, dat, mdat, ax_cbar, xlim=xlim)
            else:
                # normal
                normal(ax, dat, mdat, xlim=xlim)
                ax_cbar.axis('off')

        # over plot
        elif isinstance(vars_plot[i], list):
            vars_overplot = vars_plot[i]
            vars_ovplot = overplot(fig, ax, vars_overplot, ax_cbar, xlim=xlim)
            if vars_ovplot is not None:
                vars_plotted.append(vars_ovplot)

    # for ax in axes[:-1]:
    #     ax.set_xticklabels([])
    #     ax.set_xlabel('')

    # ax_xticks = axes[0] if len(vars_plot) == 1 else axes[-1]
    ax_xticks = axes[-1]

    if isinstance(vars_plotted[-1], str):
        dat_last = pytplot.get_data(vars_plotted[-1])
    else:
        dat_last = pytplot.get_data(vars_plotted[-1][0])

    # x-ticks: unix -> datetime
    xtick_labels = get_xticks_datetime_from_unix(ax_xticks, dat_last.times, delta=delta_xticks,
                                                 timeunit=timeunit_xticks, xlim=xlim)

    ax_xticks.set_xticklabels(xtick_labels)

    # not display x-ticks except for the bottom ax
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].tick_params(labelbottom=True)

    # set x-lim
    # if xlim is not None:
    #     tlim_unix = [
    #         datetime.strptime(xlim[0], '%Y-%m-%d %H:%M:%S').timestamp(),
    #         datetime.strptime(xlim[1], '%Y-%m-%d %H:%M:%S').timestamp()
    #     ]
    #     for ax in axes:
    #         ax.set_xlim(tlim_unix)

    if suptitle is not None:
        fig.suptitle(suptitle)
    
    if ax_options is not None:
        add_ax_options(axes, axes_cbar, ax_options)


    path.savefig(save_png)
    return


def tplot_with_orbit_gs(
        vars_plot: list,
        var_orbit: str | None = None,
        save_png: str | None = None,
        figsize: tuple = (12, 6),
        suptitle: str | None = None,
        delta_xticks = 15,
        timeunit_xticks: str = 'minutes',
        hspace: float = 0.05,
        wspace: float = 0.02,
        top: float = 0.90,
        list_label: list | None = None,
        delta_y_orb: float = 0.21,
        offset_x: float = 0.3,
        offset_y: float = 0.05,
        display_label_orb: bool = False,
        print_savepath: bool = True,
        xlim: list | None = None,
        height_raios: list | None = None,
        ax_options: dict | None = None
):
    if not isinstance(vars_plot, list):
        logging.info('vars_plot was converted to list')

    if xlim is not None:
        tlim_unix = [
                datetime.strptime(xlim[0], '%Y-%m-%d %H:%M:%S').timestamp(),
                datetime.strptime(xlim[1], '%Y-%m-%d %H:%M:%S').timestamp()
            ]


    dict_default_pos_orb = {
        'num_vars': [1, 4],
        'delta_y_orb': [.21, .35],
        'offset_x': [0, 0],
        'offset_y': [.05, .1]
    }

    vars_plotted = []

    nrow = len(vars_plot)
    height_raios_default = [1] + [2] * nrow + [1]
    if height_raios is None:
        height_raios = height_raios_default
    else:
        if isinstance(height_raios, list):
            if len(height_raios) == 3:
                height_raios = [height_raios[0]] + [height_raios[1]] * nrow + [height_raios[2]]
            else:
                logging.error(f'length of height_raios must be 3: {len(height_raios)=}\n'
                              f'-> set by default')
                height_raios = height_raios_default
        else:
            logging.error(f'Unsupported type of height_ratios: {type(height_raios)=}\n'
                          f'-> set by default')
            height_raios = height_raios_default

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(top=top)
    gs = gridspec.GridSpec(
        nrow+2, 3,
        width_ratios=[5, 50, 1],
        height_ratios=height_raios,
        hspace=hspace,
        wspace=wspace,
    )

    # suptitle
    ax_title = fig.add_subplot(gs[0, 1])
    ax_title.axis('off')

    # main plot
    axes = []
    axes_cbar = []
    ax_master = None
    for i in range(1, nrow+1):
        if ax_master is None:
            ax = fig.add_subplot(gs[i, 1])
            ax_master = ax
        else:
            ax = fig.add_subplot(gs[i, 1], sharex=ax_master)
        ax_cbar = fig.add_subplot(gs[i, 2])
        axes.append(ax)
        axes_cbar.append(ax_cbar)

    for i in range(len(vars_plot)):
        ax = axes[i]
        ax_cbar = axes_cbar[i]

        if isinstance(vars_plot[i], str):
            var_plot = vars_plot[i]
            dat = pytplot.get_data(var_plot)
            mdat = pytplot.get_data(var_plot, metadata=True)

            if mdat is None:
                logging.error(f'{var_plot} is None')
                continue

            vars_plotted.append(var_plot)

            # dict_mdat_extras = mdat['plot_options']['extras']
            extras = mdat.get('plot_options', {}).get('extras', {})
            if extras.get('spec') == 1: #'spec' in dict_mdat_extras.keys() and dict_mdat_extras['spec'] == 1:
                # spectrogram
                spectrogram(fig, ax, dat, mdat, ax_cbar, xlim=xlim)
            else:
                # normal
                normal(ax, dat, mdat, xlim=xlim)
                ax_cbar.axis('off')

        # over plot
        elif isinstance(vars_plot[i], list):
            vars_overplot = vars_plot[i]
            vars_ovplot = overplot(fig, ax, vars_overplot, ax_cbar, xlim=xlim)
            if vars_ovplot is not None:
                vars_plotted.append(vars_ovplot)

        # if xlim is not None:
        #     ax.set_xlim(tlim_unix)

    if xlim is not None:
        axes[0].set_xlim(pytplot.time_double(xlim[0]), pytplot.time_double(xlim[1]))

    if not vars_plotted:
        pass

    else:
        if len(vars_plotted) == 1:
            ax_xticks = axes[-1]
        else:
            ax_xticks = axes[-2]

        if isinstance(vars_plotted[-1], str):
            dat_last = pytplot.get_data(vars_plotted[-1])
        else:
            dat_last = pytplot.get_data(vars_plotted[-1][0])

        # # set x-lim
        # if xlim is not None:
        #     tlim_unix = [
        #         datetime.strptime(xlim[0], '%Y-%m-%d %H:%M:%S').timestamp(),
        #         datetime.strptime(xlim[1], '%Y-%m-%d %H:%M:%S').timestamp()
        #     ]
        # else:
        #     tlim_unix = [np.min(dat_last.times), np.max(dat_last.times)]
        #
        # for ax in axes:
        #     ax.set_xlim(tlim_unix)

        # orbit
        dat_orbit = pytplot.get_data(var_orbit)
        # x-ticks: unix -> datetime
        xtick_labels = get_xticks_datetime_with_orbit(
            ax_xticks, dat_last.times, dat_orbit,
            delta=delta_xticks,
            timeunit=timeunit_xticks,
            xlim=xlim
        )
        ax_xticks.set_xticklabels(xtick_labels)

    # label: rmlatmlt
    ax_label = fig.add_subplot(gs[-1, 0])
    if display_label_orb:
        ax_label.set_facecolor('lightyellow')
    else:
        ax_label.axis('off')
    if list_label is None:
        list_label = ["R [Re]", "MLAT [deg]", "MLT [hr]", "TIME [HH:MM]"]
    for i, label in enumerate(list_label):
        x = 1 - offset_x
        y = 1 - offset_y - i * delta_y_orb
        # ax_label.text(x, y, label, ha='right', va='center', transform=ax_label.transAxes, fontsize=9)
        ax_label.text(
            x, y, label,
            ha='right', va='top',
            transform=ax_label.transAxes, # 軸座標系 [0, 1]
            fontsize=9,
        )

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    axes[-1].tick_params(labelbottom=True)


    if suptitle is not None:
        fig.suptitle(suptitle)

    if ax_options is not None:
        add_ax_options(axes, axes_cbar, ax_options)

    path.savefig(save_png, print_savepath=print_savepath)
    return


def tplot(
        vars_plot: list,
        var_orbit: str | None = None,
        save_png: str | None = None,
        figsize: tuple = (12, 6),
        suptitle: str | None = None,
        delta_xticks = 15,
        timeunit_xticks: str = 'minutes',
        hspace: float = 0.05,
        wspace: float = 0.02,
        top: float = 0.90,
        list_label_orbit: list | None = None,
        delta_y_orb: float = 1,
        offset_x: float = 1,
        offset_y: float = 1,
        display_label_orb: bool = False,
        tight_layout: bool = False,
        print_savepath: bool = True,
        xlim: list | None = None,
        height_ratios: list | None = None,
        ax_options: dict | None = None
):
    """

    :param vars_plot:
    :param var_orbit:
    :param save_png:
    :param figsize:
    :param suptitle:
    :param delta_xticks:
    :param timeunit_xticks:
    :param hspace:
    :param wspace:
    :param top:
    :param list_label:
    :param delta_y_orb:
    :param offset_x:
    :param offset_y:
    :param display_label_orb:
    :param tight_layout:
    :param print_savepath:
    :param xlim:
    :return:

    offset memo
        num |  figsize  | height_ratios | offset value

        1   | (12, 6)   | [1, 5, 1]     | offset_x=1, offset_y=1, delta_y_orb=1

        2   | (12, 12)  | [1, 5, 1]     | offset_x=1, offset_y=1, delta_y_orb=0.8

        3   | (12, 10)  | [1, 5, 1]     | offset_x=1, offset_y=1, delta_y_orb=1.5
        3   | (10, 12)  | [1, 5, 1]     | offset_x=1, offset_y=1, delta_y_orb=1.2
        3   | (12, 12)  | [1, 5, 1]     | offset_x=1, offset_y=1, delta_y_orb=1.2

        4   | (12, 12)  | [1, 5, 1]     | offset_x=1, offset_y=1, delta_y_orb=1.6

        6   | (12, 16)  | [1, 50, 1]    | offset_x=4, offset_y=1, delta_y_orb=17

        8   | (12, 10)  | [1, 5, 1]     | offset_x=1, offset_y=1.8, delta_y_orb=3.6
    """
    # Default setting
    if height_ratios is None:
        height_ratios = [1, 5, 1]

    offset_x = .1 * offset_x
    if offset_y == 1:
        offset_y = .02 * offset_y
    else:
        offset_y = .02 * offset_y * 10  # 10倍して動く感度を上げる
    delta_y_orb = .25 * delta_y_orb

    if var_orbit is None:
        tplot_without_orbit_gs(
            vars_plot,
            save_png=save_png,
            figsize=figsize,
            suptitle=suptitle,
            delta_xticks=delta_xticks,
            timeunit_xticks=timeunit_xticks,
            hspace=hspace,
            wspace=wspace,
            top=top,
            xlim=xlim,
            ax_options=ax_options
        )

    else:
        tplot_with_orbit_gs(
            vars_plot,
            var_orbit,
            save_png=save_png,
            figsize=figsize,
            suptitle=suptitle,
            delta_xticks=delta_xticks,
            timeunit_xticks=timeunit_xticks,
            hspace=hspace,
            wspace=wspace,
            top=top,
            list_label=list_label_orbit,
            delta_y_orb=delta_y_orb,
            offset_x=offset_x,
            offset_y=offset_y,
            display_label_orb=display_label_orb,
            print_savepath=print_savepath,
            xlim=xlim,
            height_raios=height_ratios,
            ax_options=ax_options
        )

    return


# def tplot(
#         vars_plot: list,
#         save_png: str = None,
#         figsize: tuple = (12, 6),
#         suptitle: str = None,
#         delta_xticks = 15,
#         timeunit_xticks: str = 'minutes',
#         var_label: str = None,
#         deltay_ticks = 0.016,
#         label_pos_x=0.04,
#         label_pos_y=0.025,
#         list_label: list = None,
#         left_margin: float = 0.5,
#         label_offset_x: float = 0.1,
#         label_offset_y: float = 0.005,
# ):
#     if not isinstance(vars_plot, list):
#         logging.info('vars_plot was converted to list')
#
#     # ==== custom GridSpec with extra row for labels ====
#     nrow = len(vars_plot)
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(
#         nrow, 3,
#         # height_ratios=[1] * nrow + [0.2],
#         width_ratios=[5, 40, 1]
#     )
#
#     axes = []
#     label_axes = []
#     for i in range(nrow):
#         ax_label = fig.add_subplot(gs[i, 0])
#         ax = fig.add_subplot(gs[i, 1])
#         ax_cbar = fig.add_subplot(gs[i, 2])
#         axes.append((ax, ax_cbar))
#
#         ax_label.axis('off')  # ラベル用は軸非表示
#         label_axes.append(ax_label)
#         axes.append((ax, ax_cbar))
#
#     # fig, axes = plt.subplots(
#     #     len(vars_plot), 2,
#     #     figsize=figsize,
#     #     # tight_layout=True,
#     #     sharex='col',
#     #     gridspec_kw={'width_ratios': [40, 1]}
#     # )
#     # # fig.subplots_adjust(left=left_margin)
#
#     for i in range(len(vars_plot)):
#         # if len(vars_plot) == 1:
#         #     ax = axes[0]
#         #     ax_cbar = axes[1]
#         # else:
#         #     ax = axes[i, 0]
#         #     ax_cbar = axes[i, 1]
#
#         ax, ax_cbar = axes[i]
#         ax_label = label_axes[i]
#
#         if isinstance(vars_plot[i], str):
#             var_plot = vars_plot[i]
#             dat = pytplot.get_data(var_plot)
#             mdat = pytplot.get_data(var_plot, metadata=True)
#
#             dict_mdat_extras = mdat['plot_options']['extras']
#             if 'spec' in dict_mdat_extras.keys() and dict_mdat_extras['spec'] == 1:
#                 # spectrogram
#                 spectrogram(fig, ax, dat, mdat, ax_cbar)
#             else:
#                 # normal
#                 normal(ax, dat, mdat)
#                 ax_cbar.axis('off')
#
#         # over plot
#         elif isinstance(vars_plot[i], list):
#             vars_overplot = vars_plot[i]
#             overplot(fig, ax, vars_overplot, ax_cbar)
#
#     # x軸ラベル設定（最後の行のplot用Ax）
#     ax_xticks = axes[-1][0]
#     # if len(vars_plot) == 1:
#     #     ax_xticks = axes[0][0]
#     # else:
#     #     ax_xticks = axes[len(vars_plot)-1][0]
#
#     dat_last = pytplot.get_data(vars_plot[-1])
#     if var_label is None:
#         # x-ticks: unix -> datetime
#         xtick_labels = get_xticks_datetime_from_unix(ax_xticks, dat_last, delta=delta_xticks, timeunit=timeunit_xticks)
#
#     else:  # orbit data
#         dat_rmlatmlt = pytplot.get_data(var_label)
#         xtick_labels = label_orbit(
#             fig,
#             ax_xticks, dat_last, dat_rmlatmlt,
#             delta=delta_xticks, timeunit=timeunit_xticks,
#             deltay_ticks=deltay_ticks,
#             # label_pos_x=label_pos_x,
#             # label_pos_y=label_pos_y,
#             list_label=list_label,
#             label_offset_x=label_offset_x,
#             label_offset_y=label_offset_y,
#             left_margin=left_margin,
#             ax_label=label_axes[-1]
#         )
#
#     ax_xticks.set_xticklabels(xtick_labels)
#
#     if suptitle is not None:
#         fig.suptitle(suptitle)
#
#     fig.tight_layout()
#
#     path.savefig(save_png)
#     return


def tplot_with_orbit_labels(
        vars_plot: list,
        var_label: str,
        save_png: str | None = None,
        figsize: tuple = (12, 6),
        suptitle: str | None = None,
        delta_xticks: int = 5,
        timeunit_xticks: str = 'minutes',
        deltay_ticks: float = .4,
        list_label: list | None = None,
        label_offset_y: float = 0.05,
):
    dat_orbit = pytplot.get_data(var_label)
    unit_to_seconds = {'hours': 3600, 'minutes': 60, 'seconds': 1}

    if not isinstance(vars_plot, list):
        vars_plot = [vars_plot]

    nrow = len(vars_plot)
    fig = plt.figure(figsize=figsize)

    # グリッド：n行＋ラベル用1行, 3列
    gs = gridspec.GridSpec(
        nrow + 1, 3,
        width_ratios=[1, 10, 0.5],
        height_ratios=[1] * nrow + [0.1],
        hspace=0.2,
        wspace=0.05,
    )

    axes_plot = []
    axes_cbar = []
    axes_rowlabel = []

    for i in range(nrow):
        ax_rowlabel = fig.add_subplot(gs[i, 0])
        ax = fig.add_subplot(gs[i, 1])
        ax_cbar = fig.add_subplot(gs[i, 2])

        ax_rowlabel.axis("off")
        axes_plot.append(ax)
        axes_cbar.append(ax_cbar)
        axes_rowlabel.append(ax_rowlabel)

    # === 最下段：軌道情報ラベル描画用 ===
    ax_label = fig.add_subplot(gs[nrow, 1])
    # ax_label.axis("off")

    ax_last_plot = axes_plot[-1]

    # === xticks (時刻) を共有化するための対象Axes ===
    ax_xticks = axes_plot[-1]
    dat = vars_plot[-1] if isinstance(vars_plot[-1], dict) else pytplot.get_data(vars_plot[-1])

    # === xtick計算 ===
    interval_sec = delta_xticks * unit_to_seconds[timeunit_xticks]
    start = int(np.floor(dat.times[0] / interval_sec) * interval_sec)
    end = int(np.ceil(dat.times[-1] / interval_sec) * interval_sec)
    xticks = np.arange(start, end + 1, interval_sec)
    ax_xticks.set_xticks(xticks)

    # === 軌道データからxtickごとの[R, MLAT, MLT]を取得 ===
    times_orb = dat_orbit.times
    rmlatmlt = dat_orbit.y  # shape = (N, 3)

    labels = []
    for xtick in xticks:
        idx = np.argmin(np.abs(times_orb - xtick))
        r, mlat, mlt = rmlatmlt[idx]
        time_str = datetime.fromtimestamp(xtick, tz=timezone.utc).strftime('%H:%M')
        labels.append(f"{r:.1f}\n{mlat:.1f}\n{mlt:.1f}\n{time_str}")

    date_str = datetime.fromtimestamp(dat.times[-1], tz=timezone.utc).strftime('%Y-%m-%d')
    labels[-1] += f"\n{date_str}"
    ax_xticks.set_xticklabels(labels)

    # === 左端の行ラベル ===
    if list_label is None:
        list_label = ["R [Re]", "MLAT [deg]", "MLT [hr]", "TIME [HH:MM]"]
    ax_last_plot_pos = ax_last_plot.get_position()
    # ax_label = axes_rowlabel[nrow-1]
    pos_ax_label = ax_label.get_position()
    for i, label in enumerate(list_label):
        # y = ax_last_plot_pos.y0 - label_offset_y - i * deltay_ticks
        # y = 1.0 - label_offset_y - i * deltay_ticks
        x = -0.01
        y = 1.8 - i * deltay_ticks
        ax_label.text(x, y, label, ha='right', va='center', transform=ax_label.transAxes, fontsize=9)

    # === 各plotを描画 ===
    for i, var in enumerate(vars_plot):
        ax = axes_plot[i]
        ax_cbar = axes_cbar[i]

        if isinstance(var, str):
            dat = pytplot.get_data(var)
            mdat = pytplot.get_data(var, metadata=True)
            extras = mdat.get('plot_options', {}).get('extras', {})

            if extras.get('spec', 0) == 1:
                spectrogram(fig, ax, dat, mdat, ax_cbar)
            else:
                normal(ax, dat, mdat)
                ax_cbar.axis("off")
        elif isinstance(var, list):
            overplot(fig, ax, var, ax_cbar)

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()

    path.savefig(save_png)

    return



# ======================================================================================================================

def two_dimension(
        data_x,
        data_y,
        **kwargs
):
    if len(data_x) != len(data_y):
        raise ValueError("len(data_x) must be equal to len(data_y).")

    params = {
        "figsize": (8, 5),
        "legend": None,
        "suptitle": None,
        "xlabel": None,
        "ylabel": None,
        "savefig": None,
        "print_savepath": False,
        "xlim": None,
        "ylim": None,
        "marker": None
    }
    params = {**params, **kwargs}

    if params["xlim"] is not None and isinstance(params["xlim"][0], str):
        params["xlim"][0] = pd.Timestamp(params["xlim"][0])
        params["xlim"][1] = pd.Timestamp(params["xlim"][1])

    fig, ax = plt.subplots(figsize=params["figsize"])
    ax.plot(data_x, data_y, marker=params["marker"])
    ax.set_xlabel(params["xlabel"])
    ax.set_ylabel(params["ylabel"])
    ax.set_xlim(params["xlim"])
    ax.set_ylim(params["ylim"])
    if params["legend"] is not None:
        fig.legend(params["legend"])
    fig.suptitle(params["suptitle"])
    path.savefig(params["savefig"], print_savepath=params["print_savepath"])
    return


def matrix_nx3x3(data_x, matrix, **kwargs):
    if matrix.shape[1] != 3 or matrix.shape[2] != 3:
        raise ValueError("matrix shape must be (n, 3, 3)")

    params = {
        "figsize": (8, 5),
        "legend": None,
        "suptitle": None,
        "xlabel": None,
        "ylabel": None,
        "savefig": None,
        "print_savepath": False,
        "sharex": "all",
        "sharey": "all",
        "xlim": None,
        "ylim": [-1, 1],
        "marker": None
    }
    params = {**params, **kwargs}
    fig, axes = plt.subplots(3, 3, sharex=params["sharex"], sharey=params["sharey"], tight_layout=True, figsize=params["figsize"])
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            ax.plot(data_x, matrix[:, j, i], marker=params["marker"])
            ax.set_title(f"({j}, {i})")
            if i == 2:
                ax.set_xlabel(params["xlabel"])
            if j == 0:
                ax.set_ylabel(params["ylabel"])
            ax.set_xlim(params["xlim"])
            ax.set_ylim(params["ylim"])
    if params["legend"] is not None:
        fig.legend(params["legend"])
    fig.suptitle(params["suptitle"])
    path.savefig(params["savefig"], print_savepath=params["print_savepath"])
    return


def check_rot_mtx_orthogonal(data_x, matrix, **kwargs):
    if matrix.shape[1] != 3 or matrix.shape[2] != 3:
        raise ValueError("matrix shape must be (n, 3, 3)")

    params = {
        "figsize": (8, 5),
        "legend": None,
        "suptitle": None,
        "xlabel": None,
        "ylabel": None,
        "savefig": None,
        "print_savepath": False,
        "sharex": "all",
        "sharey": "all",
        "xlim": None,
        "ylim": None,
        "marker": None
    }
    params = {**params, **kwargs}

    fig, axes = plt.subplots(3, 3, sharex=params["sharex"], sharey=params["sharey"], tight_layout=True,
                             figsize=params["figsize"])
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            ax.plot(data_x, [np.dot(matrix[t, :, i], matrix[t, :, j]) for t in range(matrix.shape[0])], marker=params["marker"])
            ax.set_title(f"$e_{i+1} dot e_{j+1}$")
            if i == 2:
                ax.set_xlabel(params["xlabel"])
            if j == 0:
                ax.set_ylabel(params["ylabel"])
            ax.set_xlim(params["xlim"])
            ax.set_ylim(params["ylim"])
    if params["legend"] is not None:
        fig.legend(params["legend"])
    fig.suptitle(params["suptitle"])
    path.savefig(params["savefig"], print_savepath=params["print_savepath"])

    return


def heatmap(
        fig, ax, data_x, data_y, data_z,
        lognorm=True,
        clim=None,
        cmap="jet",
        clabel=None,
):
    """

    :param fig:
    :param ax:
    :param data_x: (n,)
    :param data_y: (m,)
    :param data_z: (m, n)
    :return:
    """
    if clim is None:
        clim = [5e-2, 5e5]

    # pcolormesh
    if lognorm:
        pcm = ax.pcolormesh(
            data_x, data_y, data_z,
            norm=LogNorm(vmin=clim[0], vmax=clim[1]), cmap=cmap
        )
    else:
        pcm = ax.pcolormesh(
            data_x, data_y, data_z, vmin=clim[0], vmax=clim[1], cmap=cmap
        )

    # colorbar
    fig.colorbar(pcm, ax=ax, label=clabel)

    return


def xlim_strtime(ax, xlim=None):
    if xlim is not None and isinstance(xlim, list) and isinstance(xlim[0], str):
        xlim[0] = pd.Timestamp(xlim[0])
        xlim[1] = pd.Timestamp(xlim[1])
    ax.set_xlim(xlim)
    return

