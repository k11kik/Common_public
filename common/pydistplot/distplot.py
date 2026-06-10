import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from common import display, path
from ._core_pydistplot import get_data, get_safe_zrange
from common.pytplot.plot import add_panel_label

def _init_options(opt):
    if opt is None:
        return
    if not opt['datatype'] is None:
        if opt['datatype'] == 'rmlt':
            opt['projection'] = 'polar'
            opt['xtick_values'] = [90 * np.pi / 180, 180 * np.pi / 180, 270 * np.pi / 180, 360 * np.pi / 180]
            opt['xtick_labels'] = ['6h', '12h', '18h', '24h']
        if opt['datatype'] == 'rmlat':
            opt['projection'] = 'polar'
            opt['xtick_values'] = [0 * np.pi / 180, 45 * np.pi / 180, 90 * np.pi / 180, 135 * np.pi / 180, 180 * np.pi / 180, 225 * np.pi / 180, 270 * np.pi / 180, 315 * np.pi / 180]
            opt['xtick_labels'] = ['0', '45', '90', '45', '0', '-45', '-90', '-45']
    
    if 'xtick_pad' not in opt:
        opt['xtick_pad'] = None
            
    return

def plot(
        vars_plot, 
        shape=None,
        figsize=None, 
        savefig=None, 
        suptitle=None,
        panel_label=True,
    ):
    """
    保存された分布データをプロットします。
    """
    if isinstance(vars_plot, str):
        vars_plot = [vars_plot]

    for var_name in vars_plot:
        opt = get_data(var_name, get_options=True)
        _init_options(opt)
    
    num_plots = len(vars_plot)
    if shape is None:
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))
    else:
        rows, cols = shape

    # 各変数の projection を確認
    proj = get_data(vars_plot[0], get_options=True).get('projection')
    
    fig, axes = plt.subplots(rows, cols, 
                             figsize=figsize or (8*cols, 6*rows), 
                             subplot_kw={'projection': proj} if proj else None,
                             squeeze=False)
    
    if suptitle:
        fig.suptitle(suptitle)
        
    axes_flat = axes.flatten()

    label_idx = 0

    for i, var_name in enumerate(vars_plot):
        if i >= len(axes_flat): break
        ax = axes_flat[i]

        if var_name == '':
            ax.set_axis_off()
            continue

        dat = get_data(var_name)
        opt = get_data(var_name, get_options=True)
        if dat is None: continue

        # マスク処理
        plot_z = dat.z
        if opt['mask_zero']:
            plot_z = ma.masked_where(plot_z == 0, plot_z)
        
        # カラーマップ設定
        cmap = plt.get_cmap(opt['colormap']).copy()
        cmap.set_bad(color='lightgray')

        # レンジ計算
        z_min, z_max = get_safe_zrange(plot_z, opt['zlim'], opt['zlog'])
        norm = LogNorm(vmin=z_min, vmax=z_max) if opt['zlog'] else None
        
        # プロット
        im = ax.pcolormesh(dat.x, dat.y, plot_z, 
                           cmap=cmap, 
                           norm=norm, 
                           vmin=None if norm else z_min, 
                           vmax=None if norm else z_max,
                           shading=opt['shading'])
        
        # panel label
        if panel_label:
            add_panel_label(ax, label_idx, x_loc=0, y_loc=1)
            label_idx += 1

        if opt['xtick_values'] is not None:
            ax.set_xticks(opt['xtick_values'])
            if opt['xtick_labels'] is not None:
                ax.set_xticklabels(opt['xtick_labels'])
        
        if opt['ytick_values'] is not None:
            ax.set_yticks(opt['ytick_values'])
            if opt['ytick_labels'] is not None:
                ax.set_yticklabels(opt['ytick_labels'])
        
        if opt['xtick_labelsize'] is not None:
            ax.tick_params(axis='y', labelsize=opt['xtick_labelsize'])
        
        if opt['ytick_labelsize'] is not None:
            ax.tick_params(axis='y', labelsize=opt['ytick_labelsize'])
        
        if opt['xtick_color'] is not None:
            ax.tick_params(axis='x', colors=opt['xtick_color'])
            # Polarプロットの円周部分のグリッド色なども変更したい場合は grid(color=...) が必要
        
        if opt['ytick_color'] is not None:
            ax.tick_params(axis='y', colors=opt['ytick_color'])


        if opt['title'] is not None:
            ax.set_title(opt['title'])

        if not proj:
            if opt['xlabel']: ax.set_xlabel(opt['xlabel'])
            if opt['ylabel']: ax.set_ylabel(opt['ylabel'])
            if opt['xlog']: ax.set_xscale('log')
            if opt['ylog']: ax.set_yscale('log')
            
        if proj == 'polar':
            pad_val = opt.get('xtick_pad')
            if pad_val is None:
                fs = plt.rcParams.get('font.size', 12)
                pad_val = max(4, int(fs * 0.8))
            
            ax.tick_params(axis='x', pad=pad_val)
            if opt['at_earth']:
                theta_earth = np.linspace(-np.pi/2, np.pi/2, 100)
                ax.fill(theta_earth, np.ones_like(theta_earth), color='black', alpha=1.0)

        x_limits = opt.get('xlim') if opt.get('xlim') is not None else opt.get('xrange')
        y_limits = opt.get('ylim') if opt.get('ylim') is not None else opt.get('yrange')
        
        if x_limits is not None: ax.set_xlim(x_limits)
        if y_limits is not None: ax.set_ylim(y_limits)


        if opt['zlabel'] is None:
            zlabel = var_name
        else:
            zlabel = opt['zlabel']
        fig.colorbar(im, ax=ax, label=zlabel, pad=0.1, fraction=0.046)

    plt.tight_layout()
    if savefig:
        path.savefig(savefig)
    else:
        plt.show()