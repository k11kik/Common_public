import os
import sys
import re
from datetime import datetime, timedelta
import numpy as np
from common import time, display, cdf

def get_dst_from_html(html_filepath):
    """
    HTMLファイルを読み込み、時刻(Unix time)とDst指数の配列を返します。
    pyspedasのparse_dst_htmlロジックを参考に最適化しています。
    """
    if not os.path.exists(html_filepath):
        display.error(f"File not found: {html_filepath}")
        return None, None

    with open(html_filepath, 'r', encoding='euc-jp', errors='ignore') as f:
        html_text = f.read()

    # ファイル名から年月を取得 (path/to/202001/index.html 形式を想定)
    # またはHTML内部からパースも可能ですが、呼び出し元から推測するのが確実です
    dirname = os.path.basename(os.path.dirname(html_filepath))
    year_str = dirname[:4]
    month_str = dirname[4:6]

    times = []
    dst_values = []

    # テーブルの前後にある不要なHTMLタグをカット (pyspedas方式)
    start_idx = html_text.find("Hourly Equatorial Dst Values")
    if start_idx == -1:
        display.error("Data header not found in HTML.")
        return None, None
    
    html_data = html_text[start_idx:]
    end_marker = "<!-- vvvvv S yyyymm_part3.html vvvvv -->"
    end_idx = html_data.find(end_marker)
    if end_idx != -1:
        html_data = html_data[:end_idx]

    html_lines = html_data.split("\n")
    # ヘッダー部分（ユニットやUTの行）を飛ばしてデータ行から開始
    # 通常5行目付近からデータが始まります
    data_strs = html_lines[5:]

    for day_str in data_strs:
        # 数値のみを抽出。最初の要素が「日」、残りが「Dst値(1-24時)」
        hourly_data = re.findall(r'[-+]?\d+', day_str)
        if len(hourly_data) < 2:
            continue
            
        day_val = hourly_data[0]
        actual_values = hourly_data[1:]

        for idx, val in enumerate(actual_values):
            # 京都大学のデータ形式は1値4文字固定長。
            # 結合して抽出される場合(例: -15-20)があるためpyspedas同様のチェックを行う
            
            # 欠損値チェック
            if val[:4] == '9999':
                continue
                
            # 複数値が繋がって抽出された場合の処理 (4の倍数でチェック)
            # pyspedasのロジック：4文字に収まらない場合は分割して考える
            clean_val = None
            remainder = len(val) % 4
            
            if len(val) <= 4:
                clean_val = float(val)
            elif remainder > 0:
                clean_val = float(val[0:remainder])
            elif val[0:4] != '9999':
                clean_val = float(val[0:4])

            if clean_val is not None:
                # 時刻文字列の作成 (YYYY-MM-DD/HH:30)
                # pyspedas に倣い 30分を代表時刻とする
                time_str = f"{year_str}-{month_str}-{int(day_val):02} {idx:02}:30:00"
                unix_time = time.convert(time_str, frm='str', into='unix')
                
                times.append(unix_time)
                dst_values.append(clean_val)

    return np.array(times), np.array(dst_values, dtype=np.float32)


def make_dst_data(
        trange,
        basedir_html,
        basedir_savecdf,
):
    """
    指定期間のHTMLをスキャンし、月ごとのCDFファイルを作成します。
    """
    trange_list = time.make_time_list(trange, 1, 'months')
    
    for trange_i in trange_list:
        dt_start = time.convert(trange_i[0], frm='str', into='datetime')
        year = dt_start.year
        month = dt_start.month
        
        html_filepath = os.path.join(
            basedir_html,
            f'{year:04}{month:02}',
            'index.html'
        )
        
        if not os.path.exists(html_filepath):
            display.warning(f'Not found: {html_filepath}')
            continue

        times, dst_values = get_dst_from_html(html_filepath)
        
        if times is None or len(times) == 0:
            display.error(f'No data extracted from {html_filepath}')
            continue

        dict_data = {
            'times': times,
            'dst_index': dst_values
        }
        
        savecdf = os.path.join(
            basedir_savecdf,
            f'{year:04}',
            f'dst_index_{year:04}{month:02}.cdf'
        )
        
        # 保存先ディレクトリの作成
        os.makedirs(os.path.dirname(savecdf), exist_ok=True)
        
        cdf.dict_to_cdffile(dict_data, savecdf)
        
    return