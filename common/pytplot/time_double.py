"""
Transform strings to datetime decimals.
"""
from dateutil import parser
from datetime import datetime, timezone
import numpy as np
from collections.abc import Iterable
import pandas as pd


def time_float_one(s_time=None):
    # この関数は変更なし。単一の datetime.datetime, Timestamp, datetime64 を処理できる
    if s_time is None:
        s_time = str(datetime.now())

    if isinstance(s_time, (int, float, np.integer, np.float64)):
        return float(s_time)
    
    # Pandas Timestamp オブジェクトの処理
    if isinstance(s_time, pd.Timestamp):
        return s_time.timestamp()

    # NumPy datetime64 オブジェクトの処理
    if isinstance(s_time, np.datetime64):
        return s_time.astype('int64') / 1e9 # ナノ秒を秒に変換

    # Python標準の datetime.datetime オブジェクトの処理を追加
    if isinstance(s_time, datetime):
        return s_time.replace(tzinfo=timezone.utc).timestamp()

    # 文字列の処理
    try:
        in_datetime = parser.isoparse(s_time)
    except ValueError:
        in_datetime = parser.parse(s_time)

    float_time = in_datetime.replace(tzinfo=timezone.utc).timestamp()

    return float_time


def time_float(str_time=None):
    """
    Transform a list of datetimes from string, pandas.Timestamp, or numpy.datetime64 to decimal.
    """
    if str_time is None:
        return time_float_one()

    # NumPy配列の処理を最初に強化
    if isinstance(str_time, np.ndarray):
        if str_time.dtype.kind == 'M' or pd.api.types.is_datetime64_any_dtype(str_time):
            # datetime64 または Pandas DatetimeIndex の場合
            # tolist()でdatetime.datetimeオブジェクトのリストになるので、それを処理
            return [time_float_one(t) for t in str_time.tolist()]
        elif str_time.dtype.kind in ('i', 'u', 'f'): # 整数または浮動小数点数のNumPy配列
            # この場合は直接floatに変換してリストに
            return str_time.astype(float).tolist()
        elif str_time.dtype == object: # object型配列の場合
            # 配列の最初の要素をチェックして、datetime.datetime オブジェクトかどうかを推測
            # これはすべての要素が同じ型であることを前提としている
            if len(str_time) > 0 and isinstance(str_time[0], datetime):
                # datetime.datetime オブジェクトの配列の場合、time_float_oneで各要素を変換
                return [time_float_one(t) for t in str_time]
            else:
                # それ以外のobject型の場合は、そのままastype(float)は危険
                # ユーザーが意図しない形式のデータを渡している可能性
                raise TypeError(f"Cannot convert object array containing type {type(str_time[0])} to float times.")
        else:
            # 他のNumPy配列の型の場合の処理（必要であれば追加）
            raise TypeError(f"Unsupported numpy array dtype: {str_time.dtype}")

    # 単一の文字列の場合
    if isinstance(str_time, str):
        return time_float_one(str_time)
    
    # Iterable (リスト、タプルなど) の場合
    time_list = []
    if isinstance(str_time, Iterable):
        for t in str_time:
            time_list.append(time_float_one(t))
        return time_list
    else:
        # 単一の非文字列/非数値/非datetimeオブジェクトが来た場合
        # time_float_one で処理できるのでそのまま渡す
        return time_float_one(str_time)

def time_double(str_time=None):
    return time_float(str_time)

