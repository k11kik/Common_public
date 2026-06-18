import numpy as np
from datetime import datetime
import copy
from .time_double import time_double
from common import display, mathpy

# データストア (グローバルな辞書として定義)
# key: tplot変数名 (str)
# value: データとそのメタデータを含む辞書
_tplot_data_storage = {}


class TplotData:
    """
    pytplot変数データをドット記法でアクセス可能にするためのクラス。
    """
    def __init__(self, times, y, v=None, metadata=None):
        self._times = times
        self._y = y
        self._v = v # yが多次元の場合のV座標など (今回はシンプルに省略可)
        self._metadata = metadata if metadata is not None else {}

    @property
    def times(self):
        """時間データにアクセスするためのプロパティ"""
        return self._times

    @property
    def y(self):
        """Y軸データにアクセスするためのプロパティ"""
        return self._y
    
    @property
    def v(self):
        """V軸データにアクセスするためのプロパティ (yが2D以上の場合)"""
        return self._v

    # 必要に応じて、他のメタデータ（単位、ラベルなど）もプロパティとして追加できます
    @property
    def units(self):
        return self._metadata.get('units')


def init_options():
    opt = {
        # var name
        'var': '',
        # color
        'color': ['black'],
        'alpha': 1,
        # line
        'linestyle': ['solid'],
        'linewidth': [1],
        'marker_size': None,
        'marker': None,
        # spec
        'spec': 0,
        # x axis
        'xlabel': None,
        # y axis
        'ylog': 0,
        'yrange': None,
        'ylabel': None,
        # z axis
        'zrange': None,
        'zlabel': None,
        'zlog': 0,
        # legend names
        'legend_names': None,
        'legend': 0,
        'legend_loc': 'upper right',
        # colormap
        'colormap': 'viridis',
        # grid
        'grid': False
    }
    return opt


def store_data(name, data=None, suffix="", replace=False, verify_get_data=True, **kwargs):
    """
    tplot変数にデータを保存します。

    Args:
        name (str): tplot変数名。
        data (list or numpy.ndarray or dict): 時間データと値データ。
            例: [[time1, val1], [time2, val2]] または {'x': [...], 'y': [...]}
        suffix (str): 変数名に追加するサフィックス。
        replace (bool): 既存の変数を上書きするかどうか。
        verify_get_data (bool): データを読み取れることを確認するかどうか。
        **kwargs: メタデータオプション（'time', 'y', 'v', 'data', 'dtype', 'units', 'labels', 'display_type', etc.）
    """
    # スレッドセーフにする場合
    # with _tplot_data_storage_lock:

    full_name = name + suffix

    if full_name in _tplot_data_storage and not replace:
        display.warning(f"Warning: Variable '{full_name}' already exists. Use replace=True to overwrite.")
        return

    variable_info = {
        'data': {},
        'metadata': {},
        'options': init_options()
    }
    
    if data is not None:
        if isinstance(data, dict):
            if 'x' in data and 'y' in data:
                variable_info['options']['var'] = name
                # ここを修正: 時間データを必ず float 型（Unixタイムスタンプ）に変換して保存
                # time_double は単一値の場合float、リスト/ndarrayの場合リストを返す
                # numpy配列として保存するため、最終的にnp.array()でラップする
                
                if len(data['x']) != len(data['y']):
                    display.warning(f'{name}: x and y must be the same length: x {data['x'].shape}, y {data['y'].shape}')
                    return

                # 入力がリストの場合も考慮し、time_doubleが返すリストをnp.arrayに変換
                converted_x = time_double(data['x'])
                if not isinstance(converted_x, np.ndarray):
                    converted_x = np.array(converted_x)
                
                variable_info['data']['x'] = converted_x
                variable_info['data']['y'] = np.array(data['y'])

                if 'v' in data:
                    variable_info['data']['v'] = np.array(data['v'])
                    variable_info['options']['spec'] = 1 # vがあればspecフラグを立てる
            else:
                print("Error: Dict data must contain 'x' and 'y' keys.")
                return
        # elif isinstance(data, (list, np.ndarray)):
        #     # [[time, value], ...] 形式の場合
        #     data_np = np.array(data)
        #     if data_np.ndim == 2 and data_np.shape[1] >= 2:
        #         variable_info['data']['x'] = data_np[:, 0] # 時間
        #         variable_info['data']['y'] = data_np[:, 1:] # 値 (複数列も対応)
        #     else:
        #         print("Error: List/ndarray data must be 2D with at least 2 columns (time, value).")
        #         return
        else:
            print("Error: 'data' must be a dict.")
            return

    # kwargsからメタデータを設定 (pytplotのoptions()と重複する部分もあるが、store_dataで初期設定できる)
    # pytplotはkwargsをそのままmetadataに突っ込むことが多い
    variable_info['metadata'].update(kwargs)

    _tplot_data_storage[full_name] = variable_info

    if verify_get_data:
        try:
            # 試しにget_dataで取得してみて、データが壊れていないか確認
            test_data = get_data(full_name, xarray=False)
            if test_data is None:
                print(f"Warning: get_data failed for '{full_name}' after store_data.")
        except Exception as e:
            print(f"Error during verify_get_data for '{full_name}': {e}")


    # pytplotにはないが、作成された変数名を返すなど
    return full_name


def get_data(name, xarray=False, cdf_epoch=False, metadata=False, get_options=False):
    """
    tplot変数からデータを取得します。

    Args:
        name (str): tplot変数名。
        xarray (bool): データをxarray形式で返すかどうか。
        cdf_epoch (bool): 時間データをCDF Epoch形式で返すかどうか。

    Returns:
        TplotData オブジェクトまたは xarray.DataArray オブジェクト、
        またはデータがない場合は None。
    """
    if name not in _tplot_data_storage:
        display.warning(f"Variable '{name}' not found.")
        return None

    variable_info = _tplot_data_storage[name]
    data = variable_info['data']
    _retrieved_options = variable_info['options']
    _retrieved_metadata = variable_info['metadata']

    if get_options:
        return _retrieved_options

    if metadata:
        return _retrieved_metadata
    

    if 'x' not in data or 'y' not in data:
        print(f"Error: Incomplete data for variable '{name}'.")
        return None

    times = data['x']
    values = data['y']
    v_coords = data.get('v') # v座標もあれば取得 (オプション)

    if xarray:
        # xarrayをインストールしている場合のみ
        try:
            import xarray as xr
            coords = {'time': times}
            dims = ['time']
            # yが多次元の場合の次元名と座標
            if values.ndim > 1:
                # pytplotでは、yが2Dの場合、v座標も管理します
                if v_coords is not None and len(v_coords) == values.shape[1]:
                    dims.append('v_dim')
                    coords['v_dim'] = v_coords
                else:
                    dims.append('component') # デフォルトの次元名
                    coords['component'] = np.arange(values.shape[1])

            xarray_data = xr.DataArray(values, coords=coords, dims=dims, name=name)
            xarray_data.attrs.update(metadata) # メタデータをattrsに格納
            return xarray_data
        except ImportError:
            print("Warning: xarray not installed. Returning TplotData object.")
            # xarrayがない場合は TplotData オブジェクトを返す
            return TplotData(times, values, v=v_coords, metadata=metadata)
    
    # xarray=False の場合、TplotData オブジェクトを返す
    return TplotData(times, values, v=v_coords, metadata=metadata)


def options(name, **kwargs):
    """
    Options
    ------
    # var name
    'var': '',
    # color
    'color': ['black'],
    'alpha': 1,
    # line
    'linestyle': ['solid'],
    'linewidth': [1],
    'marker_size': None,
    'marker': None,
    # spec
    'spec': 0,
    # x axis
    'xlabel': None,
    # y axis
    'ylog': 0,
    'yrange': None,
    'ylabel': None,
    # z axis
    'zrange': None,
    'zlabel': None,
    'zlog': 0,
    # legend names
    'legend_names': None,
    'legend': 0,
    'legend_loc': 'upper right',
    # colormap
    'colormap': 'viridis',
    # grid
    'grid': False
    """
    # スレッドセーフにする場合
    # with _tplot_data_storage_lock:

    if name not in _tplot_data_storage:
        display.warning(f"Error: Variable '{name}' not found.")
        return None

    variable_info = _tplot_data_storage[name]
    
    # オプションを設定
    # if option_name is not None and option_value is not None:
    #     variable_info['options'][option_name] = option_value
        # pytplotは'metadata'も'options'も透過的に扱えることが多い
        # variable_info['metadata'][option_name] = option_value # 必要に応じて
    
    # kwargsで複数のオプションを設定
    if kwargs:
        # check variables
        for key in kwargs.keys():
            if not key in variable_info['options'].keys():
                display.error(f"'{name}' | Invalid option key: {key}")
        variable_info['options'].update(kwargs)
        # variable_info['metadata'].update(kwargs)

    # オプションを取得
    # if option_name is not None and option_value is None and not kwargs:
    #     return variable_info['options'].get(option_name)
    # elif option_name is None and option_value is None and not kwargs:
    #     return variable_info['options'] # 全てのオプションを返す
    
    return None # 設定が完了した場合はNoneを返す


def del_data(name=None, silent=False):
    """
    tplot変数または全ての変数を削除します。

    Args:
        name (str, optional): 削除するtplot変数名。Noneの場合は全ての変数を削除。
        silent (bool): 削除時のメッセージを表示しないかどうか。
    """
    # スレッドセーフにする場合
    # with _tplot_data_storage_lock:

    if name is None:
        # 全ての変数を削除
        display.info('Deleting all tplot variables.')
        _tplot_data_storage.clear()
    elif name in _tplot_data_storage:
        del _tplot_data_storage[name]
        display.info(f"Deleted: '{name}'.")
    else:
        display.warning(f"Variable '{name}' not found. No variable deleted.")


def tplot_names(quiet=False):
    """
    現在ストアされているtplot変数名のリストと、各変数のデータ形状を表示します。
    3次元データの場合は (時間, 周波数) の形式で表示します。
    """
    var_names = list(_tplot_data_storage.keys())
    if quiet:
        return var_names
    else:
        title = '=' * 10 + ' tplot_names ' + '=' * 10
        print(title)
        for i, var_name in enumerate(var_names):
            dat_var = get_data(var_name)
            if dat_var is None:
                print(f"{i} {var_name} (No data available)")
                continue

            y_shape = dat_var.y.shape
            print(f'{i} {var_name} {y_shape}')

        print('=' * len(title))
        return var_names


def rename(
        varname,
        varname_renamed,
        info=True
):
    """
    tplot変数の名前を変更します。元の変数名は削除され、データは新しい変数名に移動します。

    Args:
        varname (str): 変更元のtplot変数名。
        varname_renamed (str): 新しいtplot変数名。

    Returns:
        str: 新しいtplot変数名、または失敗した場合はNone。
    """
    if varname not in _tplot_data_storage:
        print(f"Error: Source variable '{varname}' not found.")
        return None

    if varname == varname_renamed:
        print(f"Warning: Source and target variable names are the same: '{varname}'. No operation performed.")
        return varname_renamed

    # 既存のデータとオプションを取得
    # NOTE: 辞書は参照渡しされるため、variable_infoの変更は_tplot_data_storage[varname_renamed]にも反映されます
    variable_info = _tplot_data_storage[varname]

    # 'var' オプションを新しい名前に更新
    # これはプロット時にラベルなどに使われる可能性があるため重要です
    variable_info['options']['var'] = varname_renamed
    
    # 既存の変数を削除する前に、新しい名前にコピー（または移動）
    # varname_renamedが既に存在する場合、上書きされます
    _tplot_data_storage[varname_renamed] = variable_info

    # 元の変数を削除
    del _tplot_data_storage[varname]

    display.info(f"Renamed: '{varname}' to '{varname_renamed}'")
    return varname_renamed


def copy_data(
        varname,
        varname_copied=None,
):
    if varname_copied is None:
        varname_copied = f'{varname}_copy'

    if varname not in _tplot_data_storage:
        display.warning(f"Source variable '{varname}' not found.")
        return None
    
    if varname == varname_copied:
        display.warning(f"Source and target variable names are the same: '{varname}'. No operation performed.")
        return varname_copied
    
    # 元のデータを取得 (参照ではなく中身を完全にコピー)
    original_info = _tplot_data_storage[varname]
    copied_info = copy.deepcopy(original_info)
    
    # 'var' オプションを新しい名前に更新
    copied_info['options']['var'] = varname_copied
    
    # ストレージに格納
    _tplot_data_storage[varname_copied] = copied_info
    
    display.info(f"Copied: '{varname}' to '{varname_copied}'")
    return varname_copied


def split(
        varname,
        list_suffix=None
):
    """
    split data
    * list_suffix is None -> 'varname'_{i}
    """
    if varname not in _tplot_data_storage:
        display.warning(f"varname not found: '{varname}'")
        return
    
    dat = get_data(varname)
    data = dat.y

    dim = data.ndim

    if dim == 1:
        display.warning(f"'{varname}' is 1d-array and cannnot be splitted")
        return
    
    else:
        num_components = data.shape[1]
        if list_suffix is not None:
            if len(list_suffix) != num_components:
                display.warning(f'The length of num_components ({num_components},) and list_suffix ({len(list_suffix)},) do not match -> list_suffix is None')
                list_suffix = None
        for i in range(num_components):
            if list_suffix is None:
                varname_i = f'{varname}_{i}'
            else:
                varname_i = f'{varname}_{list_suffix[i]}'
            data_i = data[:, i]
            store_data(varname_i, {'x': dat.times, 'y': data_i})
    return


def exist_vars(vars_required):
    """
    指定されたすべての変数が存在するか確認します。
    共通化により、警告を抑制する silent モードを搭載。

    Args:
        vars_required (str or list): 確認したいtplot変数名、またはそのリスト。
        silent (bool): False の場合、存在しない変数名を自動的に警告ログ出力します。

    Returns:
        bool: すべて存在すれば True、一つでも欠けていれば False。
    """
    if not isinstance(vars_required, list):
        vars_required = [vars_required]
        
    present = set(vars_required).issubset(_tplot_data_storage.keys())
    
    if not present:
        missing_vars = [v for v in vars_required if v not in _tplot_data_storage]
        display.warning(f"Required variables not found in storage: {missing_vars}")
        
    return present



def interp(parent_var: str, data_var: str, newname: str | None = None) -> str | None:
    """
    指定した2つのtplot変数において、parent_var の時間軸 (times) に合わせて 
    data_var のデータ (y) を線形補間（リサンプリング）します。
    
    時間帯に全く重複（重なり）がない場合は警告を出し、安全に処理を中断します。

    Parameters
    ----------
    parent_var : str
        基準（補間先）となる時間軸を提供するtplot変数名 (例: 'mag')。
    data_var : str
        実際に補間（再サンプリング）されるデータを保持しているtplot変数名 (例: 'pos_rmlatmlt')。
    new_name : str, optional
        補間後の新規tplot変数名。
        Noneの場合は自動的に '{data_var}_interpolated' となります。

    Returns
    -------
    str or None
        新規に生成・登録されたtplot変数名。全く重なりがなく失敗した場合は None。
    """
    # 1. 存在確認
    if not exist_vars([parent_var, data_var]):
        return None

    # 2. データのロード
    dat_parent = get_data(parent_var)
    dat_data = get_data(data_var)
    
    times_parent = dat_parent.times
    times_data = dat_data.times
    values_data = dat_data.y

    if len(times_parent) == 0 or len(times_data) == 0:
        display.warning("One of the input variables has an empty time array.")
        return None

    # 3. 【重要】時間軸の完全重複チェック (Overlap Check)
    # parentとdataの時間軸にお互いに重なる時間帯が全くない場合は補間不可
    p_min, p_max = times_parent[0], times_parent[-1]
    d_min, d_max = times_data[0], times_data[-1]

    # 全く重複がない（親がデータより前にある、または親がデータより後ろにある）場合
    if p_max < d_min or p_min > d_max:
        display.warning(
            f"Time ranges do not overlap at all.\n"
            f"  Parent: [{datetime.fromtimestamp(p_min)} to {datetime.fromtimestamp(p_max)}]\n"
            f"  Data  : [{datetime.fromtimestamp(d_min)} to {datetime.fromtimestamp(d_max)}]\n"
            f"  -> Interpolation cancelled."
        )
        return None

    # 4. 補間の実行
    interpolated_y = mathpy.interp_vec(times_parent, times_data, values_data)

    if interpolated_y is None:
        display.warning("Interpolation failed during vector calculation.")
        return None

    # 5. 新しい変数名の決定
    if newname is None:
        newname = f"{data_var}_interp"

    # 6. 新しい変数を登録
    # 元の data_var のメタデータ (units等) を取得してそのまま引き渡す
    metadata_orig = get_data(data_var, metadata=True)
    
    # 補間されたデータをストア
    if data_var == newname:
        replace = True
    else:
        replace = False
    store_data(newname, {'x': times_parent, 'y': interpolated_y}, replace=replace, **metadata_orig)

    # 7. プロットオプションの完全継承 (Deep Copy)
    options_orig = get_data(data_var, get_options=True)
    new_options = copy.deepcopy(options_orig)
    
    # オプション内の表示名称のみ新名称に変更
    new_options['var'] = newname
    
    # # 補間後のラベル名を少し調整（凡例やY軸ラベルに"(interpolated)"などを適宜付与）
    # if new_options.get('ylabel') is not None:
    #     # すでに (interpolated) が付いていない場合のみ追記
    #     if "(interpolated)" not in new_options['ylabel']:
    #         new_options['ylabel'] = f"{new_options['ylabel']} (interpolated)"
            
    _tplot_data_storage[newname]['options'].update(new_options)

    display.info(f"Interpolated: {newname} | '{data_var}' onto the timeline of '{parent_var}'")
    return newname


# def old_tplot_names():
#     """
#     現在ストアされているtplot変数名のリストを返します。
#     """
#     # スレッドセーフにする場合
#     # with _tplot_data_storage_lock:

#     var_names = list(_tplot_data_storage.keys())
#     title = '=' * 5 + 'tplot_names' + '=' * 5
#     print(title)
#     for i, var_name in enumerate(var_names):
#         dat_var = get_data(var_name)
#         print(f"{i} {var_name} {dat_var.y.shape}")
#     print('=' * len(title))
#     return var_names
