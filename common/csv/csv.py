import pandas as pd
import numpy as np
from common import display

def get_trange_list(
        csv_filepath,
        varname_start='start',
        varname_end='end',
):
    df = pd.read_csv(csv_filepath)
    starts = np.array(df[varname_start])
    ends = np.array(df[varname_end])
    trange_list = []
    for i in range(len(starts)):
        trange_list.append([starts[i], ends[i]])
    
    return trange_list


def get_trange_list_from_csvs(
        csv_file_list,
        varname_start='start',
        varname_end='end',
):
    trange_list = None
    for csv_file in csv_file_list:
        trange_list_i = get_trange_list(csv_file, varname_start=varname_start, varname_end=varname_end)
        if trange_list_i is None:
            continue
        else:
            if trange_list is None:
                trange_list = trange_list_i
            else:
                trange_list.extend(trange_list_i)
            
    return trange_list


def sort_by_time(
        csv_filepath,
        output_path=None,
        varname_start='start',
):
    if not csv_filepath.lower().endswith('.csv'):
        raise ValueError('csv_filepath must be .csv')
    
    if output_path is None:
        output_path = csv_filepath

    df = pd.read_csv(csv_filepath, parse_dates=[varname_start]) # parse_dates: 'start'をdatetimeとして読み込み
    df_sorted = df.sort_values(by=varname_start, ascending=True).reset_index(drop=True) # reset_index: 元のindexをreset
    df_sorted.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S')

    display.current_time_comment(comment=f"sorted: {output_path}")

    return df_sorted
