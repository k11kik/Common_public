"""
log level:
* DEBUG: debug
* INFO: information
* WARNING: error in modules/functions
* ERROR: error: error in main process such as try/except sequence
"""

from datetime import datetime, timedelta
import inspect


# ログレベルの整数値定義 (標準loggingモジュールに倣い、大きいほど重要)
LOG_LEVELS = {
    'DEBUG': 10, # debug
    'INFO': 20, # information
    'WARNING': 30, # warning: error in modules/functions
    'ERROR': 40, # error: error in main process
    'CRITICAL': 50 # 現在は未使用だが将来的な拡張のために定義
}

# グローバルな表示しきい値（デフォルトはINFO以上を表示）
CURRENT_LOG_LEVEL_THRESHOLD = LOG_LEVELS['INFO']

def set_log_level(level_name: str):
    """
    表示するログの最低レベルを設定します。
    level_name: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    global CURRENT_LOG_LEVEL_THRESHOLD
    level_name = level_name.upper()
    
    if level_name in LOG_LEVELS:
        CURRENT_LOG_LEVEL_THRESHOLD = LOG_LEVELS[level_name]
        
        # 設定変更をINFOレベルで表示 (内部ロギング関数を使用)
        # Note: set_log_level自体はしきい値のチェックを受けないようにしています
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(color_letters(f'INFO {time_str} [set_log_level] ', color='white') + f"Log level set to {level_name}.")
    else:
        # 無効なレベルが設定された場合はエラー表示
        current_level_name = [k for k, v in LOG_LEVELS.items() if v == CURRENT_LOG_LEVEL_THRESHOLD][0]
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(color_letters(f'ERROR {time_str} [set_log_level] ', color='red') + f"Invalid log level name: {level_name}. Level remains at {current_level_name}.")
        

def get_caller_name():
    """
    この関数を呼び出した関数（親関数）の名前を取得します。
    
    inspect.currentframe() を使用し、スタックフレームを辿ります。
    - フレーム 0: get_caller_name()
    - フレーム 1: debug/error/warning()
    - フレーム 2: ユーザーが log 関数を呼んだ関数 (これが必要な情報)
    """
    try:
        # 現在のフレームを取得
        frame = inspect.currentframe()
        # 1つ上のフレーム（debug/error/warning）をスキップし、さらに1つ上のフレームを取得
        # Noneチェックとスタックの深さチェックを行います
        if frame is not None:
            # frame.f_back は1つ前のスタックフレーム（呼び出し元）
            # f_back.f_back はさらにその前のスタックフレーム（ログ関数を呼んだ場所）
            caller_frame = frame.f_back.f_back
            
            if caller_frame is not None:
                return caller_frame.f_code.co_name
    except Exception:
        # 何らかのエラーが発生した場合（スタックの深さ不足など）
        pass
    
    return 'global_scope' # 関数外で呼ばれた場合のデフォルト値


def current_time_comment(puces: str = None, comment: str = None):
    if comment is None:
        comment = ""

    if puces is None:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} " + comment)

    else:
        print(puces + f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} " + comment)
    return


# def old_progress_bar(iteration, progress_size):
#     """
#     used in for sentence
#     :param iteration:
#     :param progress_size:
#     :return:
#     """
#     pro_bar = ('=' * iteration) + (' ' * (progress_size - iteration))
#     print(f'\r[{pro_bar}] {iteration}/{progress_size}', end='')
#     return


def progress_bar(iteration, total, start_time, bar_length=40, color='white', update=True, level='INFO'):
    """
    Display a progress bar with percentage and time per iteration.

    Params
    ------
    * iteration: Current iteration (int)
    * total: Total number of iterations (int)
    * start_time: Start time just before the loop (float, obtained from datetime.now())
    * bar_length: Length of the progress bar (default: 40)

        
    Return
    --------
    dict
    * 'parcent'
    * 'time_per_iter'
    * 'eta'
    """
    def format_timespan(seconds):
        """秒数を日・時間・分・秒の読みやすい形式に変換する"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        
        td = timedelta(seconds=int(seconds))
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)

    target_level = level.upper()
    level_value = LOG_LEVELS.get(target_level, 20)
    if level_value < CURRENT_LOG_LEVEL_THRESHOLD:
        return None

    # start from 1
    prog_iteration = iteration + 1

    # Calculate percentage and elapsed time
    percent = (prog_iteration / float(total)) * 100
    elapsed_time = (datetime.now() - start_time).total_seconds()
    avg_time_per_iter = elapsed_time / iteration if iteration > 0 else 0

    # Build progress bar
    filled_length = int(bar_length * prog_iteration // total)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)

    # Remain time
    remain_iter = total - prog_iteration
    remain_time_sec = avg_time_per_iter * remain_iter
    remain_time = timedelta(seconds=remain_time_sec)

    # Print progress bar with percentage and elapsed time
    # print(f'\r[{bar}] {prog_iteration}/{total} ({percent:.2f}%) | Time/Iter: {avg_time_per_iter:.2f}s | ETA: {remain_time}', end='')
    if update:
        print(color_letters(f'\r[{bar}] {prog_iteration}/{total} ({percent:.2f}%) | Time/Iter: {avg_time_per_iter:.2f}s | ETA: {remain_time}', color=color), end='')
    else:
        print(color_letters(f'\r[{bar}] {prog_iteration}/{total} ({percent:.2f}%) | Time/Iter: {avg_time_per_iter:.2f}s | ETA: {remain_time}', color=color))
    
    # Add newline and Summary when progress completes
    if prog_iteration >= total:
        total_duration_str = format_timespan(elapsed_time)
        finish_msg = f"\n  ✔︎ Total Elapsed Time: {total_duration_str} ({avg_time_per_iter:.4f} s/iter)"
        print(finish_msg)
    
    # # Add newline when progress completes
    # if iteration == (total - 1):
    #     print()
    
    return {
        'percent': percent,
        'time_per_iter': avg_time_per_iter,
        'eta': remain_time
    }


def color_letters(letter: str, color: str='red'):
    """
    Parameters
    ----------
        letter : str
        color : str
            valid options: black, red, green, yellow, blue, magenta, cyan, white
    """
    dict_color_code = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'white': 37
    }
    if not color in dict_color_code.keys():
        print(f'Unsupported color name: {color}\n'
              '-> color is set to red')
        color = 'red'

    color_code = dict_color_code[color]
    return f'\033[{color_code}m{letter}\033[0m'


def debug(comment: str, arg_debug=None):
    if arg_debug is not None:
        print(color_letters('[display/debug] Out of date. arg_debug is to be deleted', color='red'))
    
    if LOG_LEVELS['DEBUG'] < CURRENT_LOG_LEVEL_THRESHOLD:
        return

    function_name = get_caller_name()
    print(color_letters(f'DEBUG {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{function_name}] ', color='green') + f"{comment}")
    return


def info(comment: str, arg_debug=None):
    if arg_debug is not None:
        print(color_letters('[display/debug] Out of date. arg_debug is to be deleted', color='red'))
    
    if LOG_LEVELS['INFO'] < CURRENT_LOG_LEVEL_THRESHOLD:
        return

    function_name = get_caller_name()
    print(color_letters(f'INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{function_name}] ', color='white') + f"{comment}")
    return


def warning(comment, arg_debug=None):
    if arg_debug is not None:
        print(color_letters('[display/warning] Out of date. arg_debug is to be deleted', color='red'))
    
    if LOG_LEVELS['WARNING'] < CURRENT_LOG_LEVEL_THRESHOLD:
        return
    
    function_name = get_caller_name()
    print(color_letters(f'WARNING {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{function_name}] ', color='yellow') + f"{comment}")


def error(comment: str, arg_debug=None):
    if arg_debug is not None:
        print(color_letters('[display/error] Out of date. arg_debug is to be deleted', color='red'))

    if LOG_LEVELS['ERROR'] < CURRENT_LOG_LEVEL_THRESHOLD:
        return
    
    function_name = get_caller_name()
    print(color_letters(f'ERROR {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{function_name}] ', color='red') + f"{comment}")
    return




# def debug(function_name, comment: str):
#     print(color_letters(f'DEBUG {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{function_name}] ', color='green') + f"{comment}")
#     # print(color_letters(f"--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Debug | {function_name}] {comment}", color='green'))
#     return


# def error(function_name, comment: str):
#     print(color_letters(f'ERROR {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{function_name}] ', color='red') + f"{comment}")
#     # print(color_letters(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Error! | {function_name}] {comment}', color='red'))
#     return


# def warning(function_name, comment):
#     print(color_letters(f'WARNING {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{function_name}] ', color='yellow') + f"{comment}")
#     # print(color_letters(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Warning | {func_name}] {comment}', color='yellow'))


def print_list(
        list_data,
        prefix='List',
        level='INFO',
):
    target_level = level.upper()
    if LOG_LEVELS.get(target_level, 0) < CURRENT_LOG_LEVEL_THRESHOLD:
        return

    if not isinstance(list_data, list):
        raise ValueError('list_data is not list type')
    
    color = 'green' if target_level == 'DEBUG' else 'white' if target_level == 'INFO' else 'yellow'
    
    function_name = get_caller_name()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_comment = f'----- {color_letters(f'{target_level} {time_str}', color=color)} [{function_name}] {prefix} -----'
    plain_text = f"{target_level} {time_str} [{function_name}] {prefix}"
    content_len = len(plain_text) + 12
    
    print(start_comment)
    for i, v in enumerate(list_data):
        print(f'{i} {v}')
    end_comment = '-' * content_len
    print(end_comment)
    return

def print_dict(
        dict_data,
        prefix='Dict',
        level='INFO',
        only_prefix=False,
):
    target_level = level.upper()
    if LOG_LEVELS.get(target_level, 0) < CURRENT_LOG_LEVEL_THRESHOLD:
        return

    if not isinstance(dict_data, dict):
        raise ValueError('dict_data is not dict type')
    
    color = 'green' if target_level == 'DEBUG' else 'white' if target_level == 'INFO' else 'yellow'
    
    function_name = get_caller_name()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if only_prefix:
        start_comment = f'----- {prefix} -----'
        plain_text = f'{prefix}'
    else:
        start_comment = f'----- {color_letters(f'{target_level} {time_str}', color=color)} [{function_name}] {prefix} -----'
        plain_text = f"{target_level} {time_str} [{function_name}] {prefix}"
    content_len = len(plain_text) + 12
    print(start_comment)
    for i, (key, value) in enumerate(dict_data.items()):
        print(f'{i} {key}: {value}')
    end_comment = '-' * content_len
    print(end_comment)
    return
