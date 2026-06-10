import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from common import time, display


def make_time_list(
    trange: list,
    delta_value: int = 1,
    timeunit: str = 'hours',
    fmt='%Y-%m-%d %H:%M:%S',
    getdata: bool = False
):
    """
    指定された期間を分割したリストを生成する。
    getdata=True の場合、終了時刻を単位の区切り（月末、年末など）まで拡張する。

    :param trange: ['YYYY-mm-dd HH:MM:SS', 'YYYY-mm-dd HH:MM:SS']
    :param delta_value: 間隔
    :param timeunit: 'years', 'months', 'days', 'hours', 'minutes', 'seconds'
    :param getdata: Trueの場合、最終区間が中途半端でもその単位（月や日など）を丸ごと含めるように終了時間を拡張する
    :return: List of [start_str, end_str]
    """
    def parse_datetime(ts):
        formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']
        for fmt in formats:
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        raise ValueError(f"Time data '{ts}' does not match format '%Y-%m-%d %H:%M:%S' or '%Y-%m-%d'")


    valid_units = ['years', 'months', 'days', 'hours', 'minutes', 'seconds']
    if timeunit not in valid_units:
        # 外部のdisplay.errorに依存しないよう標準的な例外処理、または簡易的なprint
        print(f'Invalid timeunit: {timeunit}. Valid units are {valid_units}')
        return None

    dt_start = parse_datetime(trange[0])
    dt_end = parse_datetime(trange[1])

    # dt_start = datetime.strptime(trange[0], '%Y-%m-%d %H:%M:%S')
    # dt_end = datetime.strptime(trange[1], '%Y-%m-%d %H:%M:%S')

    # getdata=True の場合、終了時刻をその単位の「キリが良いところ」まで押し上げる
    if getdata:
        if timeunit == 'years':
            # 次の年の1月1日 00:00:00 の直前まで
            if not (dt_end.month == 1 and dt_end.day == 1 and dt_end.hour == 0):
                dt_end = (dt_end + relativedelta(years=1)).replace(month=1, day=1, hour=0, minute=0, second=0)
        elif timeunit == 'months':
            # 次の月の1日 00:00:00 の直前まで
            if not (dt_end.day == 1 and dt_end.hour == 0):
                dt_end = (dt_end + relativedelta(months=1)).replace(day=1, hour=0, minute=0, second=0)
        elif timeunit == 'days':
            # 次の日の 00:00:00 の直前まで
            if not (dt_end.hour == 0 and dt_end.minute == 0):
                dt_end = (dt_end + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        # hours, minutes 等も必要に応じて拡張可能（通常はdays以上で利用を想定）

    time_list = []
    current = dt_start

    while current < dt_end:
        # 次のステップを計算
        if timeunit in ['years', 'months']:
            delta_args = {timeunit: delta_value}
            next_step = current + relativedelta(**delta_args)
        else:
            delta_args = {timeunit: delta_value}
            next_step = current + timedelta(**delta_args)

        # getdata=False の場合は元の dt_end でクリップする
        # getdata=True の場合は拡張された dt_end でクリップする
        end_of_interval = min(next_step, dt_end)

        time_list.append([
            current.strftime(fmt),
            end_of_interval.strftime(fmt)
        ])
        
        current = next_step
        # 最後にクリップされた場合でも、ループを抜けるためにcurrentを更新
        if current >= dt_end:
            break

    return time_list


# def old_make_time_list(# 20260225
#         trange: list,
#         delta_value=1,
#         timeunit: str = 'hours',
# ):
#     """

#     :param trange: ['YY-mm-dd HH:MM:SS', 'YY-mm-dd HH:MM:SS']
#     :param delta_value:
#     :param timeunit: 'years', 'days', 'hours', 'minutes', 'seconds'
#     :return:
#     """
#     valid_units = ['years', 'months', 'days', 'hours', 'minutes', 'seconds']
#     if timeunit not in valid_units:
#         display.error(f'Invalid timeunit: {timeunit}. Valid units are {valid_units}')
#         return None
#     start_str, end_str = trange
#     dt_start = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
#     dt_end = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')

#     # 時間のdeltaを作る
#     # delta_args = {timeunit: delta_value}
#     # delta = timedelta(**delta_args)

#     time_list = []
#     current = dt_start

#     if timeunit in ['years', 'months']:
#         while current < dt_end:
#             # 次の時刻をrelativedeltaで計算
#             delta_args = {timeunit: delta_value}
#             next_time = current + relativedelta(**delta_args)
            
#             # 最後の区間がend_timeを超えないように調整
#             end_of_interval = min(next_time, dt_end)
            
#             time_list.append([
#                 current.strftime('%Y-%m-%d %H:%M:%S'),
#                 end_of_interval.strftime('%Y-%m-%d %H:%M:%S')
#             ])
#             current = end_of_interval
    
#     else:
#         delta_args = {timeunit: delta_value}
#         delta = timedelta(**delta_args)
#         while current < dt_end:
#             next_time = current + delta
#             time_list.append([
#                 current.strftime('%Y-%m-%d %H:%M:%S'),
#                 next_time.strftime('%Y-%m-%d %H:%M:%S')
#             ])
#             current = next_time

#     # while current < dt_end:
#     #     next_time = min(current + delta, dt_end)
#     #     time_list.append([
#     #         current.strftime('%Y-%m-%d %H:%M:%S'),
#     #         next_time.strftime('%Y-%m-%d %H:%M:%S')
#     #     ])
#     #     current = next_time

#     return time_list


def sort_trange(
        trange_list,
):
    trange_list_processing = np.array(trange_list.copy())
    unix_starts = time.convert(trange_list_processing[:, 0], frm='str', into='unix')
    sort_idx = np.argsort(unix_starts)
    return trange_list_processing[sort_idx].tolist()


def filter_trange(
    trange_list,
    min_datetime: str = None,
    max_datetime: str = None,
):
    trange_list_processing = trange_list.copy()
    # sort
    trange_list_processing = sort_trange(trange_list_processing)
    trange_list_processing = np.array(trange_list_processing)

    # min_datetime
    if min_datetime is None:
        idx_min_datetime = 0
    else:
        # str -> unix
        unix_starts = time.convert(trange_list_processing[:, 0], frm='str', into='unix')
        unix_starts = np.array(unix_starts)
        unix_min_datetime = time.convert(min_datetime, frm='str', into='unix')

        sign_starts_min = np.sign(unix_starts - unix_min_datetime)
        diff_sign_starts_min = np.diff(sign_starts_min)
        if np.sum(diff_sign_starts_min) == 0:
            display.info('min_datetime is out of trange')
            idx_min_datetime = 0
        else:
            idx_min_datetime = np.where(diff_sign_starts_min != 0)[0][0] + 1
    
    # max_datetime
    if max_datetime is None:
        idx_max_datetime = len(trange_list) - 1
    else:
        unix_ends = time.convert(trange_list_processing[:, 1], frm='str', into='unix')
        unix_ends = np.array(unix_ends)
        unix_max_datetime = time.convert(max_datetime, frm='str', into='unix')

        sign_ends_max = np.sign(unix_ends - unix_max_datetime)
        diff_sign_ends_max = np.diff(sign_ends_max)
        if np.sum(diff_sign_ends_max) == 0:
            display.info('max_datetime is out of trange')
            idx_max_datetime = len(trange_list) - 1
        else:
            idx_max_datetime = np.where(diff_sign_ends_max != 0)[0][0] + 1
    
    return trange_list[idx_min_datetime:idx_max_datetime]
