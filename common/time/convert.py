from datetime import datetime, timezone
import numpy as np
from common import display


def _convert_unix(transvalue, into='datetime', fmt='%Y-%m-%d %H:%M:%S'):
    # check
    valid_into = ['datetime', 'str']

    if not into in valid_into:
        display.error(f'Invalid into: {into}. Use {valid_into}')
        return None
    
    if not isinstance(transvalue, (int, float, list, tuple, np.ndarray)):
        display.error(f'Invalid `transvalue` type: {type(transvalue)}. Must be a number or a sequence of numbers.')
        return None
    
    def _to_datetime_single(t):
        return datetime.fromtimestamp(t, tz=timezone.utc)

    def _to_str_single(t):
        return _to_datetime_single(t).strftime(fmt)
    

    if isinstance(transvalue, (list, tuple, np.ndarray)):
        if into == 'datetime':
            return [_to_datetime_single(t) for t in transvalue]
        elif into == 'str':
            return [_to_str_single(t) for t in transvalue]
    else:
        if into == 'datetime':
            return _to_datetime_single(transvalue)
        elif into == 'str':
            return _to_str_single(transvalue)


def _convert_datetime(transvalue, into='unix', fmt='%Y-%m-%d %H:%M:%S'):
    # check
    valid_into = ['unix', 'str']

    if into not in valid_into:
        display.error(f'Invalid `into` value: {into}. Please use one of {valid_into}.')
        return None

    def _is_valid_type(val):
        return isinstance(val, datetime)

    if not (_is_valid_type(transvalue) or isinstance(transvalue, (list, tuple, np.ndarray))):
        display.error(f'Invalid `transvalue` type: {type(transvalue)}. Must be a datetime object or a sequence of datetime objects.')
        return None
    
    # helper func
    def _to_unix_single(t_dt):
        if t_dt.tzinfo is None:
            t_dt = t_dt.replace(tzinfo=timezone.utc)
        return t_dt.timestamp()

    def _to_str_single(t_dt):
        return t_dt.strftime(fmt)

    if isinstance(transvalue, (list, tuple)):
        if into == 'unix':
            return [_to_unix_single(t) for t in transvalue]
        elif into == 'str':
            return [_to_str_single(t) for t in transvalue]
    elif isinstance(transvalue, np.ndarray):
        if into == 'unix':
            return np.array([_to_unix_single(t) for t in transvalue])
        elif into == 'str':
            return np.array([_to_str_single(t) for t in transvalue])
    else:
        if into == 'unix':
            return _to_unix_single(transvalue)
        elif into == 'str':
            return _to_str_single(transvalue)


def _convert_str(transvalue, into='datetime', fmt='%Y-%m-%d %H:%M:%S', out_fmt=None):
    valid_into = ['datetime', 'unix', 'str']

    if into not in valid_into:
        display.error(f'Invalid `into` value: {into}. Please use one of {valid_into}.')
        return None

    # out_fmtが指定されていない場合、入力フォーマットfmtを使用
    if into == 'str' and out_fmt is None:
        display.warning("`out_fmt` is not specified for `into='str'`. Using `fmt` as the output format.")
        out_fmt = fmt
    elif into == 'str' and out_fmt is not None:
        pass # out_fmtが指定されている場合、そのまま使用
    else:
        out_fmt = None # 他のinto='str'でない場合はout_fmtは使用しない

    # 文字列または文字列のシーケンスであることを確認
    def _is_valid_type(val):
        return isinstance(val, str)

    if not (_is_valid_type(transvalue) or isinstance(transvalue, (list, tuple, np.ndarray))):
        display.error(f'Invalid `transvalue` type: {type(transvalue)}. Must be a string or a sequence of strings.')
        return None

    # 変換ヘルパー関数
    def _to_datetime_single(t_str):
        # 1. ユーザー指定のフォーマットで試行
        try:
            return datetime.strptime(t_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            # 2. 失敗した場合、日付のみのフォーマットを試行
            try:
                # fmtがデフォルトと異なる場合も考慮し、一般的な日付形式をフォールバックとして用意
                fallback_fmts = ['%Y-%m-%d', '%Y/%m/%d']
                for f in fallback_fmts:
                    try:
                        return datetime.strptime(t_str, f).replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                # 全て失敗した場合はエラーをレイズ
                raise ValueError(f"Time data '{t_str}' does not match format '{fmt}' or fallback date formats.")
            except ValueError as e:
                display.error(str(e))
                raise
    # def _to_datetime_single(t_str):
    #     return datetime.strptime(t_str, fmt).replace(tzinfo=timezone.utc)

    def _to_unix_single(t_str):
        return _to_datetime_single(t_str).timestamp()
    
    def _to_str_single(t_str):
        # datetimeオブジェクトに変換後、指定されたフォーマットで文字列に戻す
        return _to_datetime_single(t_str).strftime(out_fmt)


    # `transvalue`がシーケンスかどうかで処理を分岐
    if isinstance(transvalue, (list, tuple, np.ndarray)):
        if into == 'datetime':
            return [_to_datetime_single(t) for t in transvalue]
        elif into == 'unix':
            return [_to_unix_single(t) for t in transvalue]
        elif into == 'str':
            return [_to_str_single(t) for t in transvalue]
    else:  # 単一の値
        if into == 'datetime':
            return _to_datetime_single(transvalue)
        elif into == 'unix':
            return _to_unix_single(transvalue)
        elif into == 'str':
            return _to_str_single(transvalue)
    

def convert(
        transvalue,
        frm='unix',
        into='datetime',
        fmt='%Y-%m-%d %H:%M:%S',
        out_fmt=None
):
    """
    Params
    -----
    * frm: 'unix', 'datetime', 'str'
    * into: 'unix', 'datetime', 'str'
    """
    # check
    valid_frm = ['unix', 'datetime', 'str']
    valid_into = ['unix', 'datetime', 'str']
    if not frm in valid_frm:
        display.error(f'Invalid frm: {frm}. Use {valid_frm}')
        return None
    if not into in valid_into:
        display.error(f'Invalid into: {into}. Use {valid_into}')
        return None
    # --------------------------------------------------------------------------

    if frm == 'unix':
        return _convert_unix(transvalue, into=into, fmt=fmt)
    
    elif frm == 'datetime':
        return _convert_datetime(transvalue, into=into, fmt=fmt)
    
    elif frm == 'str':
        return _convert_str(transvalue, into=into, fmt=fmt, out_fmt=out_fmt)

    return

# def _to_datetime(ts):
#         return datetime.fromtimestamp(ts, tz=timezone.utc)


# def _to_unix(dt):
#         return dt.replace(tzinfo=timezone.utc).timestamp()


# def _to_datetime_single(t):
#         return datetime.fromtimestamp(t, tz=timezone.utc)


# def _to_str_single(t, fmt='%Y-%m-%d %H:%M:%S'):
#     return _to_datetime_single(t).strftime(fmt)


# def unix2datetime(transvalue, into: str = 'datetime'):
#     """
#     Unix時間とdatetimeの相互変換関数。

#     :param transvalue: 変換対象。float (Unix timestamp) または datetime。
#                        リストやNumPy配列でも可。
#     :param into: 'datetime'（デフォルト）でUnix→datetime、
#                  'unix' でdatetime→Unix。
#     :return: 変換結果（同じ構造を保持：単体 or リスト）
#     """
#     if into == 'datetime':
#         if isinstance(transvalue, (list, tuple, np.ndarray)):
#             return [_to_datetime(t) for t in transvalue]
#         else:
#             return _to_datetime(transvalue)
#     elif into == 'unix':
#         if isinstance(transvalue, (list, tuple, np.ndarray)):
#             return [_to_unix(t) for t in transvalue]
#         else:
#             return _to_unix(transvalue)
    
#     elif into == 'strdatetime':
#         if isinstance(transvalue, (list, tuple, np.ndarray)):
#             return [_to_datetime(t).strftime('%Y-%m-%d %H:%M:%S') for t in transvalue]
#         else:
#             return _to_datetime(transvalue).strftime('%Y-%m-%d %H:%M:%S')
#     else:
#         raise ValueError("引数 'into' は 'datetime' または 'unix' を指定してください。")






