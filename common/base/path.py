"""
12.nov.2024
path

Contents
    * file path 関連
"""

import os
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from glob import glob
import shutil
from .import display


def ensure_directory_exists(pathname):
    """
    指定されたパスが存在しなければ、ディレクトリを作成する関数。

    :param pathname: str, 作成するディレクトリのパス
    """
    if pathname == '':
        print('No directory')
        return
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print(f"Directory created: {pathname}")
    else:
        print(f"Directory already exists: {pathname}")


def create_directory(dir_name):# -> make_directory
    """
    # -> make_directory
    """
    display.warning('path/create_directory', 'Out of date: Use make directory')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory created: {dir_name}")
    return


# def make_directory(filename):
#     dir_name = os.path.dirname(filename)
#     if dir_name and not os.path.exists(dir_name):
#         os.makedirs(dir_name, exist_ok=True)
#         display.current_time_comment(comment=f'Directory created: {dir_name}')


def make_directory(pathname: str):
    """
    引数が「ディレクトリパス」でも「ファイルパス」でも、適切なディレクトリを作成する関数。

    Parameters
    ----------
    path : str
        作成したいディレクトリのパス、または保存したいファイルのパス。
        - 拡張子あり (例: 'data/result.csv') -> ファイルとみなし、親ディレクトリ 'data/' を作成
        - 拡張子なし (例: 'data/result')     -> ディレクトリとみなし、'data/result/' を作成
    """
    if not pathname:
        return

    # 拡張子の有無を取得
    _, ext = os.path.splitext(pathname)

    if ext:
        # 拡張子がある場合 -> ファイルパスとみなす
        # 例: "dir/sub/file.txt" -> "dir/sub"
        dir_name = os.path.dirname(pathname)
    else:
        # 拡張子がない場合 -> ディレクトリパスとみなす
        # 例: "dir/sub" -> "dir/sub"
        dir_name = pathname

    # ディレクトリ名が空（カレントディレクトリ）でない、かつ存在しない場合に作成
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        display.current_time_comment(comment=f'Directory created: {dir_name}')

    return


def remove_file_if_exists(filename):
    """
    指定したファイルが存在すれば削除する。

    :param filename: 削除対象のファイルパス
    """
    if os.path.isfile(filename):
        os.remove(filename)
        # print(f"Deleted: {filename}")
    return


def savefig(save_filename, print_savepath=True, dpi=None, fig=None, bbox_inches='tight', pad_inches=0.05):
    """
    画像を指定したパスと名前で保存する関数。
    :param save_filename: str, 保存するファイル名（フルパスを含む）
    :param ensure_dir: bool
    """
    if save_filename is None:
        plt.show()
        return
    
    if dpi is None:
        dpi = 'figure'

    # ディレクトリを取得
    directory = os.path.dirname(save_filename)

    # ディレクトリが存在するか確認し、なければ作成
    # if ensure_dir:
    #     ensure_directory_exists(directory)
    if directory == '':
        pass
    else:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")

    # 現在のプロットを保存
    if fig is None:
        plt.savefig(save_filename, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    else:
        fig.savefig(save_filename, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)

    display.info(f'Saved image: {save_filename}')

    if fig is None:
        plt.close()
    else:
        plt.close(fig)

    return


def old_savefig01(fig, save_filename, print_savepath=True):
    """
    画像を指定したパスと名前で保存する関数。
    :param save_filename: str, 保存するファイル名（フルパスを含む）
    :param ensure_dir: bool
    """
    if save_filename is None:
        plt.show()
        return

    # ディレクトリを取得
    directory = os.path.dirname(save_filename)

    # ディレクトリが存在するか確認し、なければ作成
    # if ensure_dir:
    #     ensure_directory_exists(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")

    # 現在のプロットを保存
    fig.savefig(save_filename)
    if print_savepath:
        print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} saved image: {save_filename}")
    return

def savecsv(
        dataframe: pd.DataFrame,
        save_filename: str,
        sep=",",
        print_savepath: bool=True,
        index=True
):
    if not save_filename.endswith(".csv"):
        raise ValueError("save_filename must end with .csv")

    # ディレクトリを取得
    directory = os.path.dirname(save_filename)

    # ディレクトリが存在するか確認し、なければ作成
    # if ensure_dir:
    #     ensure_directory_exists(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")

    # save dataframe as a csv file
    dataframe.to_csv(save_filename, sep=sep, index=index)
    display.info(f'Saved csv: {save_filename}')
    # if print_savepath:
    #     print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} saved csv: {save_filename}")
    return


def copy_data_by_datetime(
        datetimes,
        dir_from,
        dir_to,
        characters,
        extension='.png'
):
    if not isinstance(datetimes, list):
        datetimes = list(datetimes)
    if not isinstance(characters, list):
        characters = list(characters)
    
    # path
    if not os.path.exists(dir_from):
        raise ValueError(f'No dir_from: {dir_from}')

    for i, dt_i in enumerate(datetimes):
        year = dt_i.year
        month = dt_i.month
        day = dt_i.day
        hour = dt_i.hour
        minute = dt_i.minute
        base_dir_ql = os.path.join(
            dir_from,
            f'{year:04}',
            f'{month:02}'
        )

        minute_1digit = minute % 10
        minute_ql = int(minute - minute_1digit)

        filename_candidate = f'{base_dir_ql}/*{year:04}{month:02}{day:02}{hour:02}{minute_ql:02}*{extension}'
        display.debug(f'{filename_candidate=}')
        list_file_to_be_copied = glob(filename_candidate)

        for f in list_file_to_be_copied:
            for char in characters:
                if not char in f:
                    list_file_to_be_copied.remove(f)

        if len(list_file_to_be_copied) != 1:
            display.warning(f'The length of file_to_be_copied is not 1: {list_file_to_be_copied=}\n->skip')
            continue

        file_to_be_copied = list_file_to_be_copied[0]

        # directory for the copy
        dir_for_copy = os.path.join(
            dir_to,
            f'{year:04}',
            f'{month:02}'
        )
        make_directory(dir_for_copy)

        # copy
        filename_copied = shutil.copy(file_to_be_copied, dir_for_copy)
        display.current_time_comment(comment=f'Copied: {filename_copied}')

    
    return


def copy_file(
        filepath_to_copy,
        dir_for_copy,
        filename_copied=None,
):
    if not os.path.exists(filepath_to_copy):
        display.warning(f'No existing filepath: {filepath_to_copy}')
        return
    
    if filename_copied is not None:
        dest_path = os.path.join(dir_for_copy, filename_copied)
    else:
        dest_path = dir_for_copy
  
    make_directory(dir_for_copy)

    # copy
    filepath_copied = shutil.copy(filepath_to_copy, dest_path)
    display.info(comment=f'Copied: {filepath_copied}')

    return filepath_copied


def glob_one(
        filepath_search
):
    list_filepath_candidate = glob(filepath_search)
    if len(list_filepath_candidate) == 0:
        display.warning(f'Not found: {filepath_search}')
        return
    
    elif len(list_filepath_candidate) == 1:
        return list_filepath_candidate[0]
    
    else:
        display.info(f'Multiple files found -> Return 1st one')
        display.print_list(list_filepath_candidate, prefix='list_filepath_candidate')
        return list_filepath_candidate[0]
    

def get_filename(
        filepath,
        without_ext=True
):
    if without_ext:
        filename = os.path.splitext(os.path.basename(filepath))[0]
    else:
        filename = os.path.basename(filepath)
    return filename

