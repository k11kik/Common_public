"""
[erg_analysis] util

* useful function
"""

# import pytplot
import numpy as np
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
import logging
import argparse
from . import mathpy
import os
import requests
import base64
import re
from common import pytplot, display


def options(
        name: str,
        dict_option: dict
):
    display.warning('util/options', 'out of date: Use pytplot.options')
    """ Set a large variety of options for individual plots.

        Parameters
        ----------
            name : str or list[str]
                Names of tplot variables to be updated (wildcards accepted).
            option : str, optional
                The name of the option. See the options section below.
            value : str, int, float, list, optional
                The value of the option. See the options section below.
            opt_dict : dict, optional
                This can be a dictionary of option-value pairs. 'option' and 'value'
                will not be needed if this dictionary item is supplied.
            quiet: bool, optional
                If True, do not complain about unrecognized options.

        Options
        -------

        Many of the options are passed directly to matplotlib calls.  For more extensive documentation about how to use these
        obtions, see the matplotlib documentation: https://matplotlib.org/stable/users/index.html

        Note that many X-axis options are controlled at the level of the entire plot, rather than per-variable (since plots with multiple panels will
        share many X axis properties).  See the tplot_options() routine for available per-plot options,


            ======================  ===========  ===========================================================================================================================
            Panel Options           Value type   Notes
            ======================  ===========  ===========================================================================================================================
            title                    str         The title of the plot.
            panel_size              flt          Number between (0,1], representing the percent size of the plot.
            alpha                   flt          Number between [0,1], gives the transparency of the plot lines.
            line_width              flt          Sets plot line width.
            legend_names            list         A list of strings that will be used to identify the lines.
            line_style              str          scatter (to make scatter plots), or solid_line, dot, dash, dash_dot, dash_dot_dot_dot, long_dash.
            border                  bool         Turns on or off the top/right axes that would create a box around the plot.
            var_label_ticks         int          Sets the number of ticks if this variable is displayed as an alternative x axis.
            char_size               int          Defines character size for plot labels, etc.
            right_axis              bool         If true,display a second Y axis on the right side of the plot.
            second_axis_size        numeric      The size of the second axis to display
            data_gap                numeric      If there is a gap in the data larger than this number in seconds, then insert
            visible                 bool         If False, do not display lines for this variable.
            (cont)                  (cont)       NaNs. This is similar to using the degap procedure on the variable, but is
            (cont)                  (cont)       applied at plot-time, and does not persist in the variable data.
            ======================  ===========  ===========================================================================================================================

            ======================  ===========  ===========================================================================================================================
            Legend Options          Value type   Notes
            ======================  ===========  ===========================================================================================================================
            legend_names            list         A list of strings that will be used to identify the legends.
            legend_size             numeric      The font size of the legend names
            legend_shadow           bool         Turns on or off drop shadows on the legend box
            legend_title            str          The title to display on the legend
            legend_titlesize        numeric      The font size of the legend title
            legend_color            [str]        The color of the legend names
            legend_edgecolor        str          The edge color of markers displayed in the legend
            legend_facecolor        str          The face color of markers displayed in the legend
            legend_markerfirst      boolean      Put the marker to the left of the line in the legend
            legend_markerscale      numeric      The scale size of markers displayed in the legend
            legend_markersize       numeric      The size of the markers displayed in the legend
            ======================  ===========  ===========================================================================================================================

            ======================  ===========  ===========================================================================================================================
            X Axis Options          Value type   Notes
            ======================  ===========  ===========================================================================================================================
            xtitle                  str          The title of the x axis.
            xsubtitle               str          The title of the x axis subtitle.
            xtitle_color            str          The color of the x axis title.
            xtick_length            numeric      The length of the x tick marks
            xtick_width             numeric      The width of the x tick marks
            xtick_color             str          The color of the x tick marks
            xtick_labelcolor        str          The color of the x tick marks
            xtick_direction         numeric      The direction of the x tick marks
            ======================  ===========  ===========================================================================================================================


            ======================  ===========  ===========================================================================================================================
            Y Axis Options          Value type   Notes
            ======================  ===========  ===========================================================================================================================
            y_range                 flt/list     Two numbers that give the y axis range of the plot. If a third argument is present, set linear or log scaling accordingly.
            ylog                    bool         True sets the y axis to log scale, False reverts.
            ytitle                  str          Title shown on the y axis. Use backslash for new lines.
            ysubtitle               str          Subtitle shown on the y axis.
            ytitle_color            str          The color of the y axis title.
            ytick_length            numeric      The length of the Y tick marks
            ytick_width             numeric      The width of the Y tick marks
            ytick_color             str          The color of the Y tick marks
            ytick_labelcolor        str          The color of the Y tick marks
            ytick_direction         numeric      The direction of the Y tick marks
            y_major_ticks           [numeric]    A list of values that will be used to set the major ticks on the Y axis.
            y_minor_tick_interval   numeric      The interval between minor ticks on the Y axis.
            ======================  ===========  ===========================================================================================================================

            ======================  ===========  ===========================================================================================================================
            Error Bar Options       Value type   Notes
            ======================  ===========  ===========================================================================================================================
            errorevery              numeric      Interval at which to show error bars
            capsize                 numeric      The size of error bar caps
            ecolor                  str          The color of the error bar lines
            elinewidth              numeric      The width of the error bar lines
            ======================  ===========  ===========================================================================================================================

            ======================  ===========  ===========================================================================================================================
            Marker/Symbol  Options  Value type   Notes
            ======================  ===========  ===========================================================================================================================
            marker_size             numeric      The size of the markers displayed in the plot
            markevery               numeric      Interval at which to show markers
            symbols                 bool         If True, display as a scatter plot (no lines)
            ======================  ===========  ===========================================================================================================================


            ======================  ===========  ===========================================================================================================================
            Z  / Specplot Options   Value type   Notes
            ======================  ===========  ===========================================================================================================================
            spec                    bool         Display this variable as a spectrogram.
            colormap                str/list     Color map to use for specplots https://matplotlib.org/examples/color/colormaps_reference.html.
            colormap_width          numeric      The width of the specplot color bar
            z_range                 flt/list     Two numbers that give the z axis range of the plot. If a third argument is present, set linear or log scaling accordingly.
            zlog                    int          True sets the z axis to log scale, False reverts (spectrograms only).
            ztitle                  str          Title shown on the z axis. Spec plots only. Use backslash for new lines.
            zsubtitle               str          Subtitle shown on the z axis. Spec plots only.
            ztitle_color            str          The color of the z axis title.
            x_interp                bool         If true, perform smoothing of spectrograms in the X direction
            x_interp_points         numeric      Number of interpolation points to use in the X direction
            y_interp                bool         If true, perform smoothing of spectrograms in the Y direction
            y_interp_points         numeric      Number of interpolation points to use in the Y direction
            xrange_slice            flt/list     Two numbers that give the x axis range of spectrogram slicing plots.
            yrange_slice            flt/list     Two numbers that give the y axis range of spectrogram slicing plots.
            xlog_slice              bool         Sets x axis on slice plot to log scale if True.
            ylog_slice              bool         Sets y axis on slice plot to log scale if True.
            spec_dim_to_plot        int/str      If variable has more than two dimensions, this sets which dimension the "v"
            (cont)                  (cont)       variable will display on the y axis in spectrogram plots.
            (cont)                  (cont)       All other dimensions are summed into this one, unless "spec_slices_to_use"
            (cont)                  (cont)       is also set for this variable.
            spec_slices_to_use      str          Must be a dictionary of coordinate:values. If a variable has more than two
            (cont)                  (cont)       dimensions, spectrogram plots will plot values at that particular slice of
            (cont)                  (cont)       that dimension. See examples for how it works.
            ======================  ===========  ===========================================================================================================================

            Many options have synonyms or variant spellings that are commonly used.  The first column gives the name that is used
            throughout the plotting code.  The second column gives the synonyms that are accepted.

            ======================  ======================================================================================================================================
            Canonical name          Accepted synonyms
            ======================  ======================================================================================================================================
            title                   name
            line_color              color, colors, col, cols, line_colors
            legend_names            labels, legend_name, legend_label, legend_labels
            legend_location         labels_location, legends_location, label_location, labels_location
            legend_size             labels_size, label_size
            legend_shadow           labels_shadow. label_shadow
            legend_title            label_title, labels_title
            legend_titlesize        lable_titlesize, labels_titlesize
            legend_color            legends_color, label_color, labels_color
            legend_edgecolor        label_edgecolor, labels_edgecolor
            legend_facecolor        label_facecolor, labels_facecolor
            legend_markerfirst      label_markerfirst, labels_markerfirst
            legend_markerscale      label_markerscale, labels_markerscale
            legend_markersize       label_markersize, labels_markersize
            legend_frameon          label_frameon, labels_frameon
            legend_ncols            label_ncols, labels_ncols
            line_style_name         line_style, linestyle
            visible                 nodata
            char_size               charsize
            marker                  markers
            marker_size             markersize
            markevery               markerevery mark_every marker_every
            symbol                  symbols
            line_width              thick
            y_range                 yrange
            z_range                 zrange
            data_gap                datagap
            spec_dim_to_plot        spec_plot_dim
            ======================  ======================================================================================================================================



        Returns
        -------
            None

        Examples
        --------
            >>> # Change the y range of Variable1
            >>> import pyspedas
            >>> x_data = [1,2,3,4,5]
            >>> y_data = [1,2,3,4,5]
            >>> pyspedas.store_data("Variable1", data={'x':x_data, 'y':y_data})
            >>> pyspedas.options('Variable1', 'yrange', [2,4])

            >>> # Change Variable1 to use a log scale
            >>> pyspedas.options('Variable1', 'ylog', 1)
            >>> pyspedas.tplot('Variable1')

            >>> # Multi-dimensional variable
            >>> y_data = np.random.rand(5, 4, 3)
            >>> v1_data = [0, 1, 3, 4]
            >>> v2_data = [1, 2, 3]
            >>> pyspedas.store_data("Variable2", data={'x': x_data, 'y': y_data, 'v1': v1_data, 'v2': v2_data})
            >>> # Set the spectrogram plots to show dimension 'v2' at slice 'v1' = 0
            >>> pyspedas.options('Variable2', 'spec', 1)
            >>> pyspedas.options("Variable2", "spec_dim_to_plot", 'v2')
            >>> pyspedas.options("Variable2", "spec_slices_to_use", {'v1': 0})
            >>> pyspedas.tplot('Variable2')

        """
    if not name in pytplot.tplot_names(quiet=True):
        logging.error(f'variable name does not exist: {name}')
        return

    for key, value in dict_option.items():
        pytplot.options(name, key, value)
    return


# def clip(
#         var_name: str,
#         epoch_clip: list
# ):
#     dat_var = pytplot.get_data(var_name)
#     times, dat = dat_var.times, dat_var.y

#     ids = np.abs(times - pytplot.time_double(epoch_clip[0])).argmin()
#     ide = np.abs(times - pytplot.time_double(epoch_clip[1])).argmin()

#     if dat.ndim == 1:
#         dat_clipped = dat[ids:ide]
#     elif dat.ndim == 2:
#         dat_clipped = dat[ids:ide, :]
#     else:
#         raise ValueError('Unsupported dimension of data.')

#     pytplot.store_data(f'{var_name}_clip', {'x': times[ids:ide], 'y': dat_clipped})
#     return


def clip(
        var_name: str,
        epoch_clip: list,
        new_name: str | None = None,
        exclude_nan: bool = True
):
    """
    # -> pytplot/timeclip
    clip the data by epoch_clip, and exclude nan from the data
    :param var_name:
    :param epoch_clip:
    :param new_name:
    :param exclude_nan:
    :return:
    """
    display.warning('util/clip', 'Out of date. Use pytplot.timeclip')
    dat_var = pytplot.get_data(var_name)
    if dat_var is None:
        logging.error('No data to clip')
        return
    times, dat = dat_var.times, dat_var.y

    ids = np.abs(times - pytplot.time_double(epoch_clip[0])).argmin()
    ide = np.abs(times - pytplot.time_double(epoch_clip[1])).argmin()

    times_clipped = times[ids:ide]

    if new_name == var_name:
        replace = True
    else:
        replace = False

    if not hasattr(dat, 'v'):
        dat_clipped = dat[ids:ide]
        if new_name is None:
            new_name = f'{var_name}_clip'
        if exclude_nan:
            if dat_clipped.ndim == 1:
                idx_not_nan = ~np.isnan(dat_clipped)
                times_exnan = times_clipped[idx_not_nan]
                dat_exnan = dat_clipped[idx_not_nan]
            else:
                idx_not_nan = ~np.isnan(dat_clipped)[:, 0]
                times_exnan = times_clipped[idx_not_nan]
                dat_exnan = dat_clipped[idx_not_nan, :]
            if len(times_exnan) != 0:
                pytplot.store_data(new_name, {'x': times_exnan, 'y': dat_exnan}, replace=replace)
        else:
            if len(times[ids:ide]) != 0:
                pytplot.store_data(new_name, {'x': times[ids:ide], 'y': dat_clipped}, replace=replace)

    else:
        dat_clipped = dat[ids:ide, :]
        if new_name is None:
            new_name = f'{var_name}_clip'
        if exclude_nan:
            idx_not_nan = ~np.isnan(dat_clipped)[:, 0]
            times_exnan = times_clipped[idx_not_nan]
            dat_exnan = dat_clipped[idx_not_nan, :]
            if len(times_exnan) != 0:
                pytplot.store_data(new_name, {'x': times_exnan, 'y': dat_exnan, 'v': dat_var.v}, replace=replace)
        else:
            if len(times[ids:ide]) != 0:
                pytplot.store_data(new_name, {'x': times[ids:ide], 'y': dat_clipped, 'v': dat_var.v}, replace=replace)

    return


def print_dict(
        dict_to_print: dict,
        title: str='info'
):
    str_start = f'========== {title} =========='
    str_end = '=' * len(str_start)
    print(str_start)
    for i, (key, value) in enumerate(dict_to_print.items()):
        print(f'{i} {key}: {value}')
    print(str_end)
    return


# def unix_to_datetime_array(unix_array, to_utc=True):
#     """
#     Convert a 1D array of Unix timestamps to datetime objects.

#     Parameters:
#     - unix_array (array-like): Unix timestamps (float or int)
#     - to_utc (bool): If True, convert as UTC. If False, convert as local time.

#     Returns:
#     - list of datetime.datetime objects
#     """
#     xtick_labels = [datetime.fromtimestamp(t, tz=timezone.utc).strftime(fmt) for t in xticks]
#     unix_array = np.array(unix_array)
#     if to_utc:
#         return [datetime.utcfromtimestamp(ts) for ts in unix_array]
#     else:
#         return [datetime.fromtimestamp(ts) for ts in unix_array]


def interpolate_vector(
        target_times,
        times,
        vector,
):
    vector_interp = []
    for i in range(vector.shape[1]):
        arr_to_interp = vector[:, i]
        arr_interp = np.interp(target_times, times, arr_to_interp)
        vector_interp.append(arr_interp)
    vector_interp = np.stack(vector_interp, axis=1)
    return vector_interp


def moving_average(
        var_name: str,
        delta_sec=None,
        res_sec=None,
        new_name: str | None = None
):
    dat_var = pytplot.get_data(var_name)
    times, dat = dat_var.times, dat_var.y
    if delta_sec is None:
        delta_sec = np.mean(np.diff(times))
        logging.error(f'[util/moving_average] delta_sec is None => set auto: {delta_sec}')
    if res_sec is None:
        window_size = int(.1 * len(times))
        logging.error(f'[util/moving_average] res_sec is None => set auto: {window_size}')
    else:
        window_size = int(res_sec / delta_sec)
    dat_ave = mathpy.moving_average_v2(dat, window_size)

    if new_name is None:
        new_name = var_name + '_ave'
    pytplot.store_data(new_name, {'x': times, 'y': dat_ave})
    return



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_float_list(s):
    try:
        return [float(x) for x in s.split(',')]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Could not parse list from string: {s}")


def unix2datetime(transvalue, into: str = 'datetime'):
    """
    Unix時間とdatetimeの相互変換関数。

    :param transvalue: 変換対象。float (Unix timestamp) または datetime。
                       リストやNumPy配列でも可。
    :param into: 'datetime'（デフォルト）でUnix→datetime、
                 'unix' でdatetime→Unix。
    :return: 変換結果（同じ構造を保持：単体 or リスト）
    """
    
    def to_datetime(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    def to_unix(dt):
        return dt.replace(tzinfo=timezone.utc).timestamp()
    
    if into == 'datetime':
        if isinstance(transvalue, (list, tuple, np.ndarray)):
            return [to_datetime(t) for t in transvalue]
        else:
            return to_datetime(transvalue)
    elif into == 'unix':
        if isinstance(transvalue, (list, tuple, np.ndarray)):
            return [to_unix(t) for t in transvalue]
        else:
            return to_unix(transvalue)
    
    elif into == 'strdatetime':
        if isinstance(transvalue, (list, tuple, np.ndarray)):
            return [to_datetime(t).strftime('%Y-%m-%d %H:%M:%S') for t in transvalue]
        else:
            return to_datetime(transvalue).strftime('%Y-%m-%d %H:%M:%S')
    else:
        raise ValueError("引数 'into' は 'datetime' または 'unix' を指定してください。")

    # # 単一値かリスト・配列かで処理を分岐
    # if isinstance(transvalue, (list, tuple, np.ndarray)):
    #     if into == 'datetime':
    #         return [to_datetime(t) for t in transvalue]
    #     elif into == 'unix':
    #         return [to_unix(t) for t in transvalue]
    # else:
    #     if into == 'datetime':
    #         return to_datetime(transvalue)
    #     elif into == 'unix':
    #         return to_unix(transvalue)

    # raise ValueError("引数 'into' は 'datetime' または 'unix' を指定してください。")


def make_time_list(
        trange: list,
        delta_value=1,
        timeunit: str = 'hours'
):
    """

    :param trange: ['YY-mm-dd HH:MM:SS', 'YY-mm-dd HH:MM:SS']
    :param delta_value:
    :param timeunit: 'years', 'days', 'hours', 'minutes', 'seconds'
    :return:
    """
    valid_units = ['years', 'months', 'days', 'hours', 'minutes', 'seconds']
    if timeunit not in valid_units:
        display.error('util/make_time_list', f'Invalid timeunit: {timeunit}. Valid units are {valid_units}')
        return None
    start_str, end_str = trange
    dt_start = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
    dt_end = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')

    # 時間のdeltaを作る
    # delta_args = {timeunit: delta_value}
    # delta = timedelta(**delta_args)

    time_list = []
    current = dt_start

    if timeunit in ['years', 'months']:
        while current < dt_end:
            # 次の時刻をrelativedeltaで計算
            delta_args = {timeunit: delta_value}
            next_time = current + relativedelta(**delta_args)
            
            # 最後の区間がend_timeを超えないように調整
            end_of_interval = min(next_time, dt_end)
            
            time_list.append([
                current.strftime('%Y-%m-%d %H:%M:%S'),
                end_of_interval.strftime('%Y-%m-%d %H:%M:%S')
            ])
            current = end_of_interval
    
    else:
        delta_args = {timeunit: delta_value}
        delta = timedelta(**delta_args)
        while current < dt_end:
            next_time = current + delta
            time_list.append([
                current.strftime('%Y-%m-%d %H:%M:%S'),
                next_time.strftime('%Y-%m-%d %H:%M:%S')
            ])
            current = next_time

    # while current < dt_end:
    #     next_time = min(current + delta, dt_end)
    #     time_list.append([
    #         current.strftime('%Y-%m-%d %H:%M:%S'),
    #         next_time.strftime('%Y-%m-%d %H:%M:%S')
    #     ])
    #     current = next_time

    return time_list



# def download_github_file(
#     github_url: str,
#     local_base_dir: str = ".",
#     github_token: str | None = None
# ):
#     """
#     指定したGitHubファイルURLまたはディレクトリURL（rawでなくてもOK）からファイル/ディレクトリをダウンロードし、ローカルに上書き保存する。
#     プライベートリポジトリの場合はPersonal Access Tokenが必要。

#     Parameters
#     ----------
#     github_url : str
#         GitHub上のファイルまたはディレクトリURL（例: https://github.com/owner/repo/blob/branch/path/to/file.py または https://github.com/owner/repo/tree/branch/path/to/dir）
#     local_base_dir : str
#         ローカルで保存するベースディレクトリ（デフォルト: カレントディレクトリ）
#     github_token : str
#         GitHub Personal Access Token（プライベートリポジトリの場合必須）

#     Returns
#     -------
#     保存したローカルファイルのパスまたはファイルリスト
#     """
#     # ファイルURL
#     m_file = re.match(r"https://github.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", github_url)
#     # ディレクトリURL
#     m_dir = re.match(r"https://github.com/([^/]+)/([^/]+)/tree/([^/]+)/(.*)", github_url)
#     # ルートディレクトリ（tree/branchのみ）
#     m_dir_root = re.match(r"https://github.com/([^/]+)/([^/]+)/tree/([^/]+)$", github_url)

#     if m_file:
#         owner, repo, branch, file_path = m_file.groups()
#         return _download_github_file_api(owner, repo, branch, file_path, local_base_dir, github_token)
#     elif m_dir:
#         owner, repo, branch, dir_path = m_dir.groups()
#         return download_github_dir(owner, repo, branch, dir_path, local_base_dir, github_token)
#     elif m_dir_root:
#         owner, repo, branch = m_dir_root.groups()
#         return download_github_dir(owner, repo, branch, '', local_base_dir, github_token)
#     else:
#         raise ValueError("URL形式が不正です: " + github_url)


# def _download_github_file_api(owner, repo, branch, file_path, local_base_dir, github_token):
#     api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
#     headers = {}
#     if github_token:
#         headers["Authorization"] = f"token {github_token}"
#     r = requests.get(api_url, headers=headers)
#     if r.status_code != 200:
#         raise Exception(f"GitHub API error: {r.status_code} {r.text}")
#     content = r.json()
#     file_content = base64.b64decode(content["content"])
#     local_path = os.path.join(local_base_dir, file_path)
#     os.makedirs(os.path.dirname(local_path), exist_ok=True)
#     with open(local_path, "wb") as f:
#         f.write(file_content)
#     print(f"Downloaded: {file_path} -> {local_path}")
#     return local_path


# def _download_github_dir(owner, repo, branch, dir_path, local_base_dir, github_token):
#     """
#     GitHubリポジトリのディレクトリを再帰的にダウンロード
#     """
#     api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{dir_path}?ref={branch}"
#     headers = {}
#     if github_token:
#         headers["Authorization"] = f"token {github_token}"
#     r = requests.get(api_url, headers=headers)
#     if r.status_code != 200:
#         print(f"[DEBUG] API URL: {api_url}")
#         print(f"[DEBUG] Headers: {headers}")
#         print(f"[DEBUG] Response: {r.text}")
#         raise Exception(f"GitHub API error: {r.status_code} {r.text}")
#     items = r.json()
#     downloaded_files = []
#     for item in items:
#         if item['type'] == 'file':
#             file_path = item['path']
#             downloaded_files.append(_download_github_file_api(owner, repo, branch, file_path, local_base_dir, github_token))
#         elif item['type'] == 'dir':
#             # 再帰的にダウンロード
#             downloaded_files.extend(download_github_dir(owner, repo, branch, item['path'], local_base_dir, github_token))
#     return downloaded_files


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt

#     times = np.linspace(0, 1, 10)
#     vector = np.stack([
#         np.sin(times),
#         np.cos(times),
#         np.sin(2 * times)
#     ], axis=1)
#     print(f'{vector.shape=}')
#     target_time = np.linspace(0, 1, 10000)
#     vector_interp = interpolate_vector(target_time, times, vector)
#     print(f'{vector_interp.shape=}')

#     fig, axes = plt.subplots(2, 1)
#     axes[0].plot(times, vector, '.--')
#     axes[1].plot(target_time, vector_interp, '.--')
#     plt.show()
