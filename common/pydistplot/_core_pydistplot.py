import numpy as np
from common import display

_dist_data_storage = {}

class DistData:
    """
    分布データを保持するクラス。
    dat.x, dat.y, dat.z でメッシュおよびデータにアクセス可能。
    """
    def __init__(self, x, y, z, metadata=None):
        self._x = x # mesh_x (e.g., mesh_theta)
        self._y = y # mesh_y (e.g., mesh_r)
        self._z = z # grid_data
        self._metadata = metadata if metadata is not None else {}

    @property
    def x(self): return self._x

    @property
    def y(self): return self._y

    @property
    def z(self): return self._z
    
    @property
    def metadata(self): return self._metadata

def init_dist_options():
    return {
        'var': '',
        'title': None,
        'xlabel': None,
        'ylabel': None,
        'zlabel': None,
        'xlog': False,
        'ylog': False,
        'zlog': False,
        'xlim': None,
        'ylim': None,
        'zlim': None,
        'xrange': None,
        'yrange': None,
        'zrange': None,
        'colormap': 'viridis',
        'shading': 'auto',
        'projection': None,  # 'polar' など
        'mask_zero': True,   # 0をマスクするかどうか
        'at_earth': True,    # 地球を描画するか（極座標用）
        'xtick_values': None, # [0, np.pi/2, ...] (polarの場合はradian)
        'xtick_labels': None, # ['0h', '6h', ...]
        'ytick_values': None,
        'ytick_labels': None,
        'xtick_labelsize': None,
        'ytick_labelsize': None,
        'xtick_color': None,  # x軸ラベル/目盛りの色
        'ytick_color': None,   # y軸ラベル/目盛りの色
        'datatype': None, # 'rmlt', 'rmlat'
        'xtick_pad': None,
    }

def store_data(name, data, **kwargs):
    """
    data: {'x': mesh_x, 'y': mesh_y, 'z': grid_data}
        * polar -> 'x': theta, 'y': r
    """
    if not isinstance(data, dict) or not all(k in data for k in ('x', 'y', 'z')):
        display.error(f"Error: Data for '{name}' must contain 'x', 'y', and 'z'.")
        return None

    variable_info = {
        'data': {
            'x': np.array(data['x']),
            'y': np.array(data['y']),
            'z': np.array(data['z'])
        },
        'metadata': kwargs,
        'options': init_dist_options()
    }
    variable_info['options']['var'] = name
    _dist_data_storage[name] = variable_info
    return name

def get_data(name, get_options=False):
    """
    変数を取得します。
    get_options=True の場合はオプション辞書を返します。
    """
    if name not in _dist_data_storage:
        display.warning(f"Variable '{name}' not found.")
        return None
    
    entry = _dist_data_storage[name]
    if get_options:
        return entry['options']
    
    return DistData(
        entry['data']['x'], 
        entry['data']['y'], 
        entry['data']['z'], 
        entry['metadata']
    )

def options(name, **kwargs):
    """
    Title
    -----
    * 'title': None

    Label
    -----
    * 'xlabel': None,
    * 'ylabel': None,
    * 'zlabel': None,

    Axis
    -----
    * 'xlog': False,
    * 'ylog': False,
    * 'zlog': False,
    * 'xlim': None,
    * 'ylim': None,
    * 'zlim': None,

    Z-axis
    -----
    * 'colormap': 'viridis',
    * 'shading': 'auto',
    * 'projection': None,  # 'polar' など
    * 'mask_zero': True,   # 0をマスクするかどうか
    * 'at_earth': True,    # 地球を描画するか（極座標用）

    X-axis
    -----
    * 'xtick_values': None, # [0, np.pi/2, ...] (polarの場合はradian)
    * 'xtick_labels': None, # ['0h', '6h', ...]
    * 'xtick_color': None,  # x軸ラベル/目盛りの色
    * 'xtick_pad': int(rcParams['font.size'] * 0.8) # (polar) margin between figure and xticklabels

    Y-axis
    -----
    * 'ytick_values': None,
    * 'ytick_labels': None,
    * 'ytick_color': None,   # y軸ラベル/目盛りの色

    Datatype
    -----
    * 'datatype': None
        * 'rmlt' 
            opt['projection'] = 'polar'
            opt['xtick_values'] = [90 * np.pi / 180, 180 * np.pi / 180, 270 * np.pi / 180, 360 * np.pi / 180]
            opt['xtick_labels'] = ['6h', '12h', '18h', '24h']
        * 'rmlat'
            opt['projection'] = 'polar'
            opt['xtick_values'] = [0 * np.pi / 180, 45 * np.pi / 180, 90 * np.pi / 180, 135 * np.pi / 180, 180 * np.pi / 180, 225 * np.pi / 180, 270 * np.pi / 180, 315 * np.pi / 180]
            opt['xtick_labels'] = ['0', '45', '90', '45', '0', '-45', '-90', '-45']
    """
    if name not in _dist_data_storage:
        display.warning(f"Variable '{name}' not found.")
        return

    opt = _dist_data_storage[name]['options']
    
    # 相互エイリアスの同期処理
    if 'xlim' in kwargs:
        kwargs['xrange'] = kwargs['xlim']
    elif 'xrange' in kwargs:
        kwargs['xlim'] = kwargs['xrange']
        
    if 'ylim' in kwargs:
        kwargs['yrange'] = kwargs['ylim']
    elif 'yrange' in kwargs:
        kwargs['ylim'] = kwargs['yrange']
        
    if 'zlim' in kwargs:
        kwargs['zrange'] = kwargs['zlim']
    elif 'zrange' in kwargs:
        kwargs['zlim'] = kwargs['zrange']

    opt.update(kwargs)

    # if name in _dist_data_storage:
    #     _dist_data_storage[name]['options'].update(kwargs)
    # else:
    #     display.warning(f"Variable '{name}' not found.")

def dist_names(quiet=False):
    """
    現在ストアされている分布図変数名の一覧を表示します。
    """
    var_names = list(_dist_data_storage.keys())
    if quiet:
        return var_names
    
    title = '=' * 5 + ' dist_names ' + '=' * 5
    print(title)
    for i, name in enumerate(var_names):
        dat = get_data(name)
        if dat is None:
            print(f"{i} : {name} (Error: No data)")
            continue
        
        # データの形状を表示 (zの形状と、x/yメッシュの形状)
        z_shape = dat.z.shape
        x_shape = dat.x.shape
        y_shape = dat.y.shape
        print(f"{i} : {name}  x:{x_shape}, y:{y_shape}, z:{z_shape}")
        
    print('=' * len(title))
    return var_names

def get_safe_zrange(data, current_zrange, is_log):
    if current_zrange is not None:
        return current_zrange
    
    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        return [0.1, 1.0] if is_log else [0.0, 1.0]
    
    vmin = np.nanmin(finite_data)
    vmax = np.nanmax(finite_data)
    
    if is_log:
        if vmin <= 0:
            positive_data = finite_data[finite_data > 0]
            vmin = np.nanmin(positive_data) if positive_data.size > 0 else 1e-3
    
    if vmin == vmax:
        vmin, vmax = (vmin/10, vmax*10) if is_log else (vmin-1, vmax+1)
            
    return [vmin, vmax]
